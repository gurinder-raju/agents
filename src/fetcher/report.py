"""Render a link report: the prompt it answers, what counted as a match, and the matching links.

The page is a single HTML file with no scripts and nothing loaded from elsewhere, so it opens
offline and can be shared as is. Everything in it comes from the model or from strangers' posts,
so every value is escaped and only http(s) links are kept.

What the model sends is checked here first, whole, against one of two flat item contracts. The same
contract builds the schema the model is shown, so what it's told and what's enforced can't drift apart.

Each match then travels as one record per value, from the tool call to the page and the saved data: which
post it's from, its value, what was said for it, and its links. Nothing is folded into anything else, so
what reaches the page is what was sent. Only the page groups records, under their post.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from html import escape
from typing import Any
from urllib.parse import urlsplit

# The shape of what the harness records about a report. It stays out of the tool's arguments: the
# model says what it found, and the harness keeps its own records.
REPORT_SCHEMA_VERSION = 2

ARGUMENTS = ("title", "criteria", "items", "sort", "left_out")
SORTS = ("as_given", "comments")
_URL_FIELDS = ("link", "extra_link")
_VALUE_FIELDS = ("fill", "value")


@dataclass(frozen=True)
class ItemContract:
    """The fields of one item in write_report's `items`, each a string and none nested.

    Small models such as gemma4:e2b often close a list nested inside an item with the wrong bracket,
    and Ollama then can't parse the call and drops it.
    """

    fields: dict[str, str]  # each field's name and what the model puts in it
    required: tuple[str, ...]
    one_of: tuple[str, ...] = ()  # an item gives exactly one of these
    note: str = ""


_DESCRIPTION = "One sentence on why it matches."
_QUOTE = "The exact words from the post or its comments that match, copied word for word."

# For a job whose prompt has one link pattern, such as https://example.com/search?q=<name>. The model
# names a source and the value it found there, and the code builds both links.
SOURCE_ITEM = ItemContract(
    fields={
        "post": "The post's id from read_subreddit, such as Python-3. For a feed entry or a page, its link "
        "exactly as your tools returned it.",
        "fill": "The value you found in this post for the placeholder in the user's link pattern, as it's written "
        "there. The report builds the link from it. If this post has none, use an empty string: the item is left out.",
        "description": _DESCRIPTION,
        "quote": _QUOTE,
    },
    required=("post", "fill", "description", "quote"),
    note="Every item has all four. A post with several values gets one item per value, each with its own quote.",
)

# For any other job: a post by id, or a feed entry or page by its link, and perhaps a second link.
LINK_ITEM = ItemContract(
    fields={
        "post": "For a Reddit post, its id from read_subreddit, such as Python-3. The report builds the post's link "
        "from it.",
        "link": "For a feed entry or a page, its link exactly as your tools returned it. One URL.",
        "description": _DESCRIPTION,
        "quote": _QUOTE,
        "extra_link": "Optional: a second link your tools returned, such as the page a link post points to. One URL. "
        "Never make one up.",
        "extra_label": 'Optional: one or two words for the second link, such as "download".',
    },
    required=("description",),
    one_of=("post", "link"),
    note="Every item has post or link, not both.",
)

# One row for whatever the model read and isn't reporting, so that leaving something out is a decision
# with a reason, not a silence. See accounting.py.
LEFT_OUT = ItemContract(
    fields={
        "source": "What you're leaving out: a source you read, such as r/Python, for the rest of its posts, or "
        "one post's id or link.",
        "value": "Only for one value you saw in that post and aren't reporting, as it's written there. Leave it out "
        "for a whole post or source.",
        "reason": "One sentence on why it isn't in items.",
    },
    required=("source", "reason"),
)


def _row_schema(contract: ItemContract) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {name: {"type": "string", "description": text} for name, text in contract.fields.items()},
        "required": list(contract.required),
        "additionalProperties": False,
    }


def report_schema(contract: ItemContract) -> dict[str, Any]:
    """The input schema write_report advertises for items that follow `contract`.

    Gemma's tool template leaves out any item field called "description", so the prose names the fields too.
    """
    fields = ", ".join(contract.fields)
    return {
        "json": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": 'Short title for the report, for example "r/Python posts that mention uv".',
                },
                "criteria": {
                    "type": "string",
                    "description": "In one or two sentences, what made a post or page count as a match.",
                },
                "items": {
                    "type": "array",
                    "description": f"One flat object of strings per match, with only these fields: {fields}. "
                    f"{contract.note} description is one sentence on why it matches.",
                    "items": _row_schema(contract),
                },
                "sort": {
                    "type": "string",
                    "enum": list(SORTS),
                    "description": '"comments" to list the most commented posts first, or "as_given" to keep your order.',
                },
                "left_out": {
                    "type": "array",
                    "description": "Why whatever you read isn't in items, as flat objects of strings with only these "
                    f"fields: {', '.join(LEFT_OUT.fields)}. One row naming a source covers the rest of its posts. A value "
                    "you saw and aren't reporting needs a row of its own, with its post as source.",
                    "items": _row_schema(LEFT_OUT),
                },
            },
            "required": ["title", "criteria", "items"],
            "additionalProperties": False,
        }
    }


class ReportError(ValueError):
    """write_report saved nothing. `stage` says which step stopped it, and `errors` lists what went wrong."""

    def __init__(self, stage: str, errors: list[str]) -> None:
        super().__init__("\n- ".join([f"write_report saved nothing. {_STAGE_ADVICE[stage]}", *errors]))
        self.stage = stage
        self.errors = errors


_STAGE_ADVICE = {
    "validation": "Fix all of these and call it again:",
    "render": "It could not be rendered, which sending it again won't change. Tell the user:",
    "write": "The file could not be written, which you can't fix. Tell the user:",
}


def parse_report(
    title: Any, criteria: Any, items: Any, sort: Any, contract: ItemContract, unknown: list[str] = (), left_out: Any = ()
) -> tuple[list[dict[str, str]], int, list[dict[str, str]]]:
    """Check everything write_report was given, before anything acts on it.

    `unknown` names any arguments the tool doesn't have. Raises ReportError with every problem at once,
    each with its field's path and the type expected, so one retry can fix them all. Returns the items
    as trimmed strings, without empty optional fields, how many had to be adapted from an older shape,
    and the left_out rows, trimmed the same way.
    """
    errors = [f"{name}: unknown argument, expected only {', '.join(ARGUMENTS)}" for name in unknown]
    for name, value in (("title", title), ("criteria", criteria)):
        if not isinstance(value, str):
            errors.append(f"{name}: expected a string, got {_kind(value)}")
        elif not value.strip():
            errors.append(f"{name}: expected a non-empty string, got an empty one")
    if not isinstance(sort, str) or sort not in SORTS:
        errors.append(f"sort: expected one of {', '.join(SORTS)}, got {_shown(sort)}")
    if not isinstance(items, list):
        errors.append(f"items: expected a list of objects, got {_kind(items)}")
        items = []
    adapted = [adapt_legacy_item(item, contract) for item in items]
    for index, (item, _) in enumerate(adapted):
        errors += _item_errors(f"items[{index}]", item, contract)
    if not isinstance(left_out, list | tuple):
        errors.append(f"left_out: expected a list of objects, got {_kind(left_out)}")
        left_out = []
    for index, row in enumerate(left_out):
        errors += _item_errors(f"left_out[{index}]", row, LEFT_OUT)
    if errors:
        raise ReportError("validation", errors)
    flat = [_trimmed(item) for item, _ in adapted]
    return flat, sum(changed for _, changed in adapted), [_trimmed(row) for row in left_out]


def _trimmed(row: dict[str, str]) -> dict[str, str]:
    return {name: value.strip() for name, value in row.items() if value.strip() or name == "fill"}


def _item_errors(path: str, item: Any, contract: ItemContract) -> list[str]:
    listing = ", ".join(contract.fields)
    if not isinstance(item, dict):
        return [f"{path}: expected an object with {listing}, got {_kind(item)}"]
    errors = [f"{path}.{name}: unknown field, expected only {listing}" for name in item if name not in contract.fields]
    for name in contract.fields:
        if name not in item:
            if name in contract.required:
                errors.append(f"{path}.{name}: missing, expected a string")
            continue
        value = item[name]
        if not isinstance(value, str):
            errors.append(f"{path}.{name}: expected a string, got {_kind(value)}")
        elif value.strip():
            if name in _URL_FIELDS and (problem := url_problem(value.strip())):
                errors.append(f"{path}.{name}: {problem}")
            if name in _VALUE_FIELDS and _LINE_BREAK.search(value.strip()):
                errors.append(f"{path}.{name}: expected one value on one line, got {_shown(value)}")
        elif name in contract.required and name != "fill":  # an empty fill says this post has no value
            errors.append(f"{path}.{name}: expected a non-empty string, got an empty one")
    given = [name for name in contract.one_of if isinstance(item.get(name), str) and item[name].strip()]
    if contract.one_of and len(given) != 1:
        errors.append(f"{path}: expected exactly one of {' or '.join(contract.one_of)}, got {len(given)}")
    return errors


def _kind(value: Any) -> str:
    """How a value looks in JSON, for an error the model can act on."""
    names = {str: "a string", bool: "true or false", int: "a number", float: "a number", list: "a list", dict: "an object"}
    return "null" if value is None else names.get(type(value), type(value).__name__)


def _shown(value: Any) -> str:
    return repr(value if len(value) <= 80 else value[:80] + "...") if isinstance(value, str) else _kind(value)


_NOT_IN_A_URL = re.compile(r"[\s\x00-\x1f\x7f]")
_SECOND_URL = re.compile(r"[,;|]\s*https?://", re.I)
_LINE_BREAK = re.compile(r"[\r\n]")


def url_problem(value: str) -> str | None:
    """Why `value` isn't one http(s) URL, checked here and now with no network; None if it is one.

    A link field holds a raw URL and is kept as given, so a URL's own punctuation survives: the ")" in
    https://en.wikipedia.org/wiki/Mercury_(planet) is part of it. Two URLs in one field are refused rather
    than guessed at, but a URL inside another, as in https://web.archive.org/web/2026/https://example.com/, is fine.
    """
    if _NOT_IN_A_URL.search(value) or _SECOND_URL.search(value):
        return f"expected one URL with no spaces, got {_shown(value)}"
    try:
        parts = urlsplit(value)
        _ = parts.port  # raises if it isn't a number from 0 to 65535
    except ValueError:
        parts = None
    if parts is None or parts.scheme not in ("http", "https") or not parts.hostname:
        return f"expected an http:// or https:// URL, got {_shown(value)}"
    return None


def adapt_legacy_item(item: Any, contract: ItemContract) -> tuple[Any, bool]:
    """Turn an item in a shape write_report used to take into `contract`. Returns it and whether it changed.

    Items used to carry their own `title`, which the report now always takes from the source, a `url`, a
    `links` value of URLs or {label, url} objects, or `link` beside `fill`. Small models still send these
    now and then, and these items mean what they say, so turning them down would only cost a retry. The
    result is validated like any other item, and anything else, such as a third link, is left for that to refuse.
    """
    if not isinstance(item, dict):
        return item, False
    adapted = {name: value for name, value in item.items() if name != "title"}
    if "url" in adapted and "link" not in adapted:
        adapted["link"] = adapted.pop("url")
    if "links" in adapted and "link" not in adapted:
        links = _legacy_links(adapted["links"])
        if 1 <= len(links) <= 2:
            del adapted["links"]
            adapted["link"] = _legacy_url(links[0])
            if len(links) == 2:
                adapted["extra_link"] = _legacy_url(links[1])
                if isinstance(links[1], dict) and "label" in links[1]:
                    adapted["extra_label"] = links[1]["label"]
    if "link" in adapted and "link" not in contract.fields and "post" not in adapted:
        adapted["post"] = adapted.pop("link")
    return adapted, adapted != item


def _legacy_links(value: Any) -> list[Any]:
    if isinstance(value, str):
        return urls_in_text(value) or [value]
    return [value] if isinstance(value, dict) else value if isinstance(value, list) else []


def _legacy_url(link: Any) -> Any:
    return link.get("url") if isinstance(link, dict) else link


def urls_in_text(text: str) -> list[str]:
    """The URLs written in free text, such as a legacy `links` string. A link field is never read this way."""
    return [trim_sentence_end(url) for url in re.findall(r"https?://[^\s<>\"']+", text)]


def trim_sentence_end(url: str) -> str:
    """Drop the sentence's punctuation from a URL found in prose, but keep a ")" the URL itself opened."""
    url = url.rstrip(".,;:")
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1].rstrip(".,;:")
    return url


def normalize_items(items: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Give each checked row, its source's link now resolved, the record that carries it to the page.

    `post` is the source as the model named it, and `fill` is its value, or None if the job has none.
    `evidence` is what was said for it: {description, quote}. Each link is {label, url, check}: links[0]
    is the post, the rest belong to the value, and `check` says what's known of the page over HTTP: nothing yet.
    """
    records = []
    for item in items:
        fill = item.get("fill")
        links = [{"label": "post", "url": item["link"], "check": "unverified"}]
        if extra_link := item.get("extra_link"):
            label = item.get("extra_label") or fill or urlsplit(extra_link).netloc
            links.append({"label": label, "url": extra_link, "check": "unverified"})
        records.append({
            "post": item.get("post") or item["link"],
            "title": item["link"],
            "fill": fill,
            "evidence": [{"description": item["description"], "quote": item.get("quote", "").strip('"\u201c\u201d')}],
            "links": links,
        })
    return records


def combine_values(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """One record per post and value, keeping everything that was said for it.

    The same post with another value stays a record of its own. The same post and value adds its
    quote and description to the first, and only an exact repeat of both is dropped. Returns the
    records and how many exact repeats there were.
    """
    combined: dict[tuple[Any, ...], dict[str, Any]] = {}
    repeats = 0
    for record in records:
        page, *own_links = (canonical_link(link["url"]) for link in record["links"])
        first = combined.setdefault((page, record["fill"], *own_links), record)
        if first is record:
            continue
        for evidence in record["evidence"]:
            if evidence in first["evidence"]:
                repeats += 1
            else:
                first["evidence"].append(evidence)
    return list(combined.values()), repeats


def match_sources(records: list[dict[str, Any]], sources: dict[str, dict[str, Any]]) -> list[str]:
    """Fill in facts from the posts and pages the tools returned, rather than trusting the model's copy.

    A record that links to a known source takes its title, short id and comment count, and the groups
    in it that hold its value. Its quotes are checked against the source's text. Returns the quotes
    that weren't found, and marks those quotes.
    """
    not_found = []
    for record in records:
        keys = (canonical_link(link["url"]) for link in record["links"])
        source = next((sources[key] for key in keys if key in sources), None)
        if source is None:
            continue
        if source.get("title"):
            record["title"] = source["title"]
        if source.get("id"):
            record["post"] = source["id"]
        if source.get("comment_count") is not None:
            record["comment_count"] = source["comment_count"]
        if record["fill"]:
            record["groups"] = [group for group in source.get("groups", ()) if contains_words(group["text"], record["fill"])]
        for evidence in record["evidence"]:
            if evidence["quote"] and not contains_words(source["text"], evidence["quote"]):
                evidence["quote_not_found"] = True
                not_found.append(evidence["quote"])
    return not_found


def group_by_post(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Each post's records together, posts in the order they first come up."""
    posts: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        posts.setdefault(canonical_link(record["links"][0]["url"]), []).append(record)
    return list(posts.values())


def count_records(records: list[dict[str, Any]]) -> dict[str, int]:
    """Three different counts: the posts in a report, the values found in them, and what was said for those."""
    return {
        "posts": len(group_by_post(records)),
        "values": sum(record["fill"] is not None for record in records),
        "evidence": sum(len(record["evidence"]) for record in records),
    }


def contains_words(text: str, phrase: str) -> bool:
    """Whether `phrase` appears in `text` as whole words, whatever the case, punctuation and spacing."""
    return f" {_words(phrase)} " in f" {_words(text)} "


def canonical_link(url: str) -> str:
    """The same page's link, written the same way: no scheme, www. or old., trailing slash or fragment."""
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.").removeprefix("old.")
    return host + parts.path.rstrip("/") + (f"?{parts.query}" if parts.query else "")


def _words(text: str) -> str:
    """Lowercase words only, so a quote matches whatever the punctuation and spacing around it."""
    return " ".join(re.findall(r"\w+", text.lower()))


def render_report(
    title: str,
    prompt: str,
    criteria: str,
    items: list[dict[str, Any]],
    created: datetime,
    coverage: list[str] = (),
    by_comments: bool = False,
    notices: list[str] = (),
    empty: str = "Nothing matched the criteria.",
    left_out: list[dict[str, Any]] = (),
) -> str:
    """Return the report as a complete HTML page. `items` must already be records.

    `coverage` has a line per source the tools read, such as how many posts didn't fit. `notices` says
    what makes this a partial report, at the top where it can't be missed. `empty` is what to say if
    there are no items. `left_out` has the rows that say why something isn't in the report.
    """
    if items:
        entries = "\n".join(_render_post(post) for post in group_by_post(items))
        body = f'<ol class="items">\n{entries}\n</ol>'
    else:
        body = f'<p class="empty">{escape(empty)}</p>'
    count = _summary(count_records(items)) + (", most commented first" if by_comments else "")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>{escape(title)}</title>
<style>{_STYLE}</style>
</head>
<body>
<main>
<header>
<h1>{escape(title)}</h1>
<p class="meta">{escape(created.strftime("%Y-%m-%d %H:%M %Z").strip())} · {count}</p>
</header>{_render_notices(notices)}
<section class="brief">
<h2>Prompt</h2>
<p class="prompt">{escape(prompt)}</p>
<h2>Criteria</h2>
<p>{escape(criteria)}</p>{_render_coverage(coverage)}
</section>
<section>
<h2>Links</h2>
{body}
</section>{_render_left_out(left_out)}
</main>
</body>
</html>
"""


def _summary(counts: dict[str, int]) -> str:
    posts, values = counts["posts"], counts["values"]
    if values:
        return f"{posts} post{'' if posts == 1 else 's'}, {values} value{'' if values == 1 else 's'}"
    return f"{posts} {'match' if posts == 1 else 'matches'}"


def _render_notices(notices: list[str]) -> str:
    if not notices:
        return ""
    lines = "\n".join(f"<li>{escape(line)}</li>" for line in notices)
    return f'\n<section class="partial">\n<h2>Partial report</h2>\n<ul class="coverage">\n{lines}\n</ul>\n</section>'


def _render_coverage(coverage: list[str]) -> str:
    if not coverage:
        return ""
    lines = "\n".join(f"<li>{escape(line)}</li>" for line in coverage)
    return f'\n<h2>Read</h2>\n<ul class="coverage">\n{lines}\n</ul>'


def _render_left_out(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = "\n".join(f"<li>{_render_left_out_row(row)}</li>" for row in rows)
    return f'\n<section>\n<h2>Left out</h2>\n<ul class="coverage">\n{lines}\n</ul>\n</section>'


def _render_left_out_row(row: dict[str, Any]) -> str:
    name = _anchor(row["link"], row.get("title") or row["source"]) if row.get("link") else escape(row["source"])
    value = f' <span class="value">{escape(row["value"])}</span>' if row.get("value") else ""
    return f"{name}{value}: {escape(row['reason'])}"


def _render_post(records: list[dict[str, Any]]) -> str:
    first = records[0]
    post = first["links"][0]
    count = first.get("comment_count")
    meta = f'\n<p class="meta">{count} comment{"" if count == 1 else "s"}</p>' if count is not None else ""
    values = "".join(f"\n{_render_record(record)}" for record in records)
    return f"""<li>
<h3>{_anchor(post["url"], first["title"])}</h3>{meta}{values}
<p class="links">{_render_link(post)}</p>
</li>"""


def _render_record(record: dict[str, Any]) -> str:
    """One value: the value itself, its own links, and everything said for it."""
    lines = []
    if record["fill"] is not None:
        flag = ' <span class="missing">(value not in the post)</span>' if record.get("value_not_found") else ""
        lines.append(f'<p class="value">{escape(record["fill"])}{flag}</p>')
    if own_links := record["links"][1:]:
        lines.append(f'<p class="links">{" · ".join(_render_link(link) for link in own_links)}</p>')
    lines += [_render_evidence(evidence) for evidence in record["evidence"]]
    body = "\n".join(line for line in lines if line)
    return f'<div class="match">\n{body}\n</div>'


def _render_evidence(evidence: dict[str, Any]) -> str:
    description = f'<p class="description">{escape(evidence["description"])}</p>' if evidence["description"] else ""
    if not evidence["quote"]:
        return description
    flag = ' <span class="missing">(not found in the post)</span>' if evidence.get("quote_not_found") else ""
    return f'{description}\n<blockquote>{escape(evidence["quote"])}{flag}</blockquote>'.strip()


def _render_link(link: dict[str, Any]) -> str:
    html = _anchor(link["url"], link["label"])
    if link.get("check") == "missing":
        html += ' <span class="missing">(not found)</span>'
    if link.get("closest"):
        html += f' · {_anchor(link["closest"], "closest page that exists")}'
    if link.get("check") == "unreachable":
        html += ' <span class="missing">(could not be reached)</span>'
    if link.get("unsourced"):
        html += ' <span class="missing">(no tool returned this link)</span>'
    return html


def _anchor(url: str, text: str) -> str:
    """A link, unless `url` isn't one: a feed can hand the tools any address it likes, such as javascript:."""
    return escape(text) if url_problem(url) else f'<a href="{escape(url)}" rel="noopener noreferrer">{escape(text)}</a>'


def report_filename(run_id: str, title: str) -> str:
    """For example 20261026-131500-a1b2c3-python-posts-about-uv.html, named after the run's trace."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60].strip("-") or "report"
    return f"{run_id}-{slug}.html"


_STYLE = """
:root { --bg: #fbfbf9; --fg: #1d1d1b; --muted: #6b6b66; --line: #e2e1dc; --panel: #f2f1ec; --link: #1f5fbf; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161615; --fg: #ecebe6; --muted: #9c9b95; --line: #2e2e2b; --panel: #1f1f1d; --link: #8ab4f8; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 760px; margin: 0 auto; padding: 40px 16px 64px; }
h1 { font-size: 1.75rem; line-height: 1.2; margin: 0 0 4px; }
h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 28px 0 8px; }
h3 { font-size: 1.05rem; margin: 0 0 4px; }
p { margin: 0 0 8px; }
a { color: var(--link); overflow-wrap: anywhere; }
.meta { color: var(--muted); font-size: 0.9rem; }
.brief { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 4px 16px 12px; margin-top: 24px; }
.brief h2:first-child { margin-top: 14px; }
.partial { border: 1px solid #b3261e; border-radius: 8px; padding: 4px 16px 12px; margin-top: 24px; }
.partial h2 { color: #b3261e; margin-top: 14px; }
.prompt { white-space: pre-wrap; }
.items { padding-left: 1.4em; margin: 0; }
.coverage { margin: 0; padding-left: 1.2em; }
.items li { padding: 14px 0; border-bottom: 1px solid var(--line); }
.items li:last-child { border-bottom: 0; }
.match { margin-top: 10px; }
.value { font-weight: 600; overflow-wrap: anywhere; }
.links { font-size: 0.9rem; color: var(--muted); }
.empty { color: var(--muted); }
blockquote { margin: 6px 0 8px; padding: 2px 0 2px 12px; border-left: 3px solid var(--line); color: var(--muted); }
.missing { color: #b3261e; }
"""
