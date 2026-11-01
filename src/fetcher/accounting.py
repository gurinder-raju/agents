"""What became of everything the read tools found, source by source.

Each source gets a row of counts, one for its candidates (its posts, feed entries, or the page itself)
and one for any groups a local tool marked in them. A group only counts for a job with a link pattern:
it's the values that job is after. Every candidate and group ends up as exactly one of these:

- rendered: it's on the page.
- rejected: the model turned it down with a reason, in a left_out row or an item with an empty fill.
- truncated: it was cut before the model read it, so nobody decided anything about it.
- unaccounted: the model read it and said nothing, which the report flags.

So found = truncated + assessed + unaccounted, and assessed = rendered + rejected. submitted counts
what the model named in items, which is what was rendered plus any it said had no value.

Truncation is about the input, and it's never the model's to explain: the source's read already says
so. What the model does answer for is what it saw. A row naming a source covers the rest of its
candidates, but not its groups: a value has to be named, or have its post named, so that one sentence
can't wave through something nobody looked at.
"""

from collections import Counter, defaultdict
from typing import Any

from .report import canonical_link, contains_words, count_records


def source_key(name: str) -> str:
    """The same source however it's written: r/Python or python, a feed's link with or without www."""
    name = name.strip().removeprefix("links on ")
    if name.lower().startswith(("http://", "https://")):
        return canonical_link(name)
    return "r/" + name.lower().strip("/").removeprefix("r/")


def account_sources(
    requested: list[str],
    reads: list[dict[str, str]],
    sources: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    left_out: list[dict[str, Any]],
    values_matter: bool,
) -> list[dict[str, Any]]:
    """One row per source read or asked for. Each row of `left_out` has `scope` and `key`: see _place_left_out in tools.py."""
    asked = {source_key(name): name for name in requested}
    read_by_key: dict[str, dict[str, str]] = {}
    for read in reads:
        read_by_key.setdefault(source_key(read["source"]), read)
    reported: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        reported[canonical_link(record["links"][0]["url"])].append(record)
    return [
        _account_source(key, read_by_key.get(key), asked.get(key), sources, reported, left_out, values_matter)
        for key in dict.fromkeys([*read_by_key, *asked])
    ]


def _account_source(
    key: str,
    read: dict[str, str] | None,
    asked: str | None,
    sources: dict[str, dict[str, Any]],
    reported: dict[str, list[dict[str, Any]]],
    left_out: list[dict[str, Any]],
    values_matter: bool,
) -> dict[str, Any]:
    found = {link: facts for link, facts in sources.items() if source_key(facts.get("read_from") or "") == key}
    if not found and asked and key in sources:  # a page the prompt named is its own one candidate
        found = {key: sources[key]}
    source_rows = [row for row in left_out if row["scope"] == "source" and row["key"] == key]
    whole_source = next((row for row in source_rows if not row.get("value")), None)

    posts: dict[str, str] = {}
    groups: list[tuple[str, dict[str, Any], str]] = []
    unaccounted_groups = []
    submitted = 0
    for link, facts in found.items():
        post_rows = [row for row in left_out if row["scope"] == "post" and row["key"] == link]
        posts[link] = _post_verdict(facts, reported.get(link, []), post_rows, whole_source is not None)
        submitted += bool(reported.get(link)) or any(row.get("no_value") for row in post_rows)
        for group in facts.get("groups", ()) if values_matter else ():
            records = reported.get(link, [])
            reasons = post_rows + source_rows
            verdict = _group_verdict(group, records, reasons)
            groups.append((link, group, verdict))
            if verdict == "unaccounted":
                missing = [text for text, part in _group_parts(group, records, reasons) if part == "unaccounted"]
                unaccounted_groups.append(f"{_short_name(link, facts)}: {' | '.join(missing)}")

    name = read["source"] if read else asked
    reason = whole_source["reason"] if whole_source else None
    if reason is None and read and not found:  # nothing to decide about, and its read says why
        reason = read["line"]
    return {
        "source": name,
        "requested": asked is not None,
        "status": read["status"] if read else "not_read",
        "line": read["line"] if read else f"{name}: never read.",
        "reason": reason,
        "posts": _tally(list(posts.values()), submitted),
        "groups": _tally([verdict for _, _, verdict in groups]),
        "report": count_records([record for link in found for record in reported.get(link, [])]),
        "unaccounted": {
            "posts": [_short_name(link, found[link]) for link, verdict in posts.items() if verdict == "unaccounted"],
            "groups": unaccounted_groups,
        },
    }


def _post_verdict(facts: dict[str, Any], records: list[dict[str, Any]], post_rows: list[dict[str, Any]], whole_source: bool) -> str:
    if records:
        return "rendered"
    if post_rows:
        return "rejected"
    if not facts.get("shown", True):
        return "truncated"
    return "rejected" if whole_source else "unaccounted"


def _group_verdict(group: dict[str, Any], records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    """A group is accounted for only when every line has a disposition."""
    verdicts = {verdict for _, verdict in _group_parts(group, records, rows)}
    return next(verdict for verdict in ("unaccounted", "truncated", "rendered", "rejected") if verdict in verdicts)


def _group_parts(group: dict[str, Any], records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Use the supplied lines; literal slashes and prose are never split into guessed values."""
    parts = list(dict.fromkeys(line for line in group["lines"] if line.strip())) or [group["text"]]
    result = []
    for text in parts:
        if any(
            record["fill"] is None or (group in record.get("groups", ()) and contains_words(text, record["fill"]))
            for record in records
        ):
            verdict = "rendered"
        elif any(_explains(row, group | {"text": text}) for row in rows):
            verdict = "rejected"
        else:
            verdict = "unaccounted" if group["shown"] else "truncated"
        result.append((text, verdict))
    return result


def _explains(row: dict[str, Any], group: dict[str, Any]) -> bool:
    """A row with a value explains that value, and a post's row with none explains all of that post.

    An empty fill doesn't: it says the post has no value, which is no answer for a value that's there.
    """
    if row.get("no_value"):
        return False
    return contains_words(group["text"], row["value"]) if row.get("value") else row["scope"] == "post"


def _short_name(link: str, facts: dict[str, Any]) -> str:
    return facts.get("id") or link


def _tally(verdicts: list[str], submitted: int | None = None) -> dict[str, int]:
    count = Counter(verdicts)
    return {
        "found": len(verdicts),
        "truncated": count["truncated"],
        "assessed": count["rendered"] + count["rejected"],
        "submitted": count["rendered"] if submitted is None else submitted,
        "rejected": count["rejected"],
        "rendered": count["rendered"],
        "unaccounted": count["unaccounted"],
    }


def accounting_lines(ledger: list[dict[str, Any]]) -> list[str]:
    """Each source's counts in words, for the page."""
    lines = []
    for row in ledger:
        for noun, tally in (("candidates", row["posts"]), ("groups", row["groups"])):
            if tally["found"]:
                lines.append(
                    f"{row['source']}: {tally['found']} {noun} found, {tally['truncated']} cut off before the model read "
                    f"them, {tally['assessed']} assessed, {tally['submitted']} submitted, {tally['rejected']} rejected, "
                    f"{tally['rendered']} rendered, {tally['unaccounted']} unaccounted."
                )
    return lines
