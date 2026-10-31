"""Tools the agent can call.

Strands turns each @tool function into a tool spec for the model: the function name becomes the
tool name, the docstring becomes its description, and the typed arguments become its JSON schema.
The model never runs these. It asks for a call, the harness runs the function and sends back the result.
If a tool raises, Strands sends the error message back as the result instead, so the model can react.
"""

import fcntl
import ipaddress
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import quote, unquote, urljoin, urlsplit
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import httpx
from strands import ToolContext, tool

from .report import (
    ARGUMENTS,
    LINK_ITEM,
    SOURCE_ITEM,
    REPORT_SCHEMA_VERSION,
    ItemContract,
    ReportError,
    canonical_link,
    combine_values,
    contains_words,
    count_records,
    match_sources,
    normalize_items,
    parse_report,
    render_report,
    report_filename,
    report_schema,
    trim_sentence_end,
)

# About 3,000 tokens: enough for most articles, small enough for a local model's context window.
MAX_PAGE_CHARS = 12_000


@tool
def current_time(timezone: str = "UTC") -> str:
    """Get the current date and time.

    Args:
        timezone: IANA timezone name, for example "America/Los_Angeles". Defaults to UTC.
    """
    return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")


@tool
def fetch_url(url: str, links: str | None = None, agent: Any = None) -> str:
    """Fetch a web page and return its readable text, with the HTML stripped out.

    To find a link on the page instead, such as where to download something, pass `links`: words
    from the link's text or address. You get the matching links, one "text: url" per line.

    Args:
        url: Full http:// or https:// address of the page.
        links: Words that the links you want contain, for example "3.13". Omit to get the page text.
    """
    source = url if links is None else f"links on {url}"  # a search says nothing about reading the page's text
    with _noting_failure(agent, source):
        response = http_get(url)
    if links is not None:
        _remember_coverage(agent, source, "complete", f"{source}: searched.")
        return find_links(response.text, str(response.url), links)
    is_html = "html" in response.headers.get("content-type", "")
    text = html_to_text(response.text) if is_html else response.text
    _remember_sources(agent, {url: {"text": text}})
    if len(text) > MAX_PAGE_CHARS:
        _remember_coverage(agent, url, "incomplete", f"{url}: read the first {MAX_PAGE_CHARS} of {len(text)} characters.")
        text = text[:MAX_PAGE_CHARS] + f"\n\n[truncated: showing {MAX_PAGE_CHARS} of {len(text)} characters]"
    else:
        _remember_coverage(agent, url, "complete", f"{url}: read in full.")
    return text


@tool
def read_feed(url: str, since_hours: float | None = None, max_entries: int = 25, agent: Any = None) -> str:
    """Read an RSS or Atom feed and return its entries in feed order, as JSON.

    Each entry has id, title, author, published (UTC), link and text: the entry's body as plain
    text, cut to 800 characters. "omitted" counts matching entries left out to keep the result small.

    Args:
        url: Full http:// or https:// address of the feed.
        since_hours: Only keep entries published within this many hours. Omit to keep them all.
        max_entries: Keep at most this many entries.
    """
    with _noting_failure(agent, url):
        title, entries = parse_feed(http_get(url).text)
    entries = [{key: value for key, value in entry.items() if key != "html"} for entry in entries]
    if since_hours is not None:
        cutoff = datetime.now(UTC) - timedelta(hours=since_hours)
        entries = [e for e in entries if e["published"] is None or e["published"] >= cutoff]

    # Local transforms see each whole text. Only then is it cut to what the model reads.
    entries = [entry | {"title": _transform(agent, entry["title"] or ""), "text": _transform(agent, entry["text"])} for entry in entries]
    kept = [entry | {"text": _shorten(entry["text"], FEED_TEXT_CHARS)} for entry in entries[:max_entries]]
    while True:
        result = json.dumps(
            {"feed": title, "entries": kept, "omitted": len(entries) - len(kept)}, default=_iso, ensure_ascii=False
        )
        if len(result) <= MAX_PAGE_CHARS or not kept:
            break
        kept = kept[:-1]
    left_out = len(entries) - len(kept)
    found = {
        entry["link"]: {
            "title": entry["title"],
            "text": f"{entry['title']}\n{_shorten(entry['text'], FEED_TEXT_CHARS)}",
            "read_from": url,
            "shown": number < len(kept),
            "groups": _entry_groups(agent, entry, number < len(kept)),
        }
        for number, entry in enumerate(entries)
    }
    _remember_sources(agent, found)
    cut_groups = _groups_cut_off(found)
    line = f"{title or url}: {len(kept)} entries" + (f", and {left_out} more that were left out." if left_out else ".")
    if cut_groups:
        line += f" {cut_groups} groups a local tool found were cut off before the model read them."
    _remember_coverage(agent, url, "incomplete" if left_out or cut_groups else "complete", line)
    return result


# About 10,000 tokens: room for a day of posts and comments in the model's 32k-token context window.
MAX_SUBREDDIT_CHARS = 40_000
COMMENT_CHARS = 500
MIN_COMMENTS_PER_POST = 5
# Each page of the subreddit's comment feed holds 100 comments and costs one request, about a minute.
MAX_COMMENT_PAGES = 10
# The post listing holds the newest 100 posts, which covers a day for all but the busiest subreddits.
LISTING_POSTS = 100
_SUBREDDIT_NAME = re.compile(r"^[A-Za-z0-9_]{2,21}$")
_POST_ID = re.compile(r"/comments/([a-z0-9]+)/")
# Reddit ends each post's feed entry with "submitted by /u/name [link] [comments]".
_REDDIT_TRAILER = re.compile(r"\s*submitted by\s+/u/\S+\s*(\[link\])?\s*(\[comments\])?\s*$")


@tool
def read_subreddit(subreddit: str, since_hours: float = 24, sort: str = "new", agent: Any = None) -> str:
    """Read a subreddit's newest posts and their comments, as JSON.

    Each post has id (a short name for it, such as Python-3, to give write_report), title, author,
    published (UTC), link, url (the page a link post points to, if any), text (the post body),
    comment_count and comments: a flat list of {author, text}, oldest first.
    Without logging in, Reddit allows about one request a minute. This reads the post list, then the
    subreddit's comment feed 100 comments at a time, so a day of a busy subreddit takes a few minutes.
    If everything doesn't fit, "omitted" counts the posts and comments that were left out.

    Args:
        subreddit: Subreddit name, for example "Python".
        since_hours: Only include posts published within this many hours.
        sort: "new" for newest first, or "comments" for the most commented first.
    """
    name = subreddit.removeprefix("r/").strip("/")
    if not _SUBREDDIT_NAME.match(name):
        raise ValueError(f"not a valid subreddit name: {subreddit!r}")
    if sort not in ("new", "comments"):
        raise ValueError('sort must be "new" or "comments"')

    cutoff = datetime.now(UTC) - timedelta(hours=since_hours)
    with _noting_failure(agent, f"r/{name}"):
        _, listing = parse_feed(http_get(f"https://www.reddit.com/r/{name}/new/.rss?limit={LISTING_POSTS}").text)
        recent = [post for post in listing if post["published"] and post["published"] >= cutoff]
        # Every comment on a post from the window was written inside the window too, so reading the
        # subreddit's comment feed back to the cutoff finds all of them in a few requests, instead of
        # one request per post. comments_complete is false if the page limit stopped it early.
        comments, comments_complete = _read_comment_feed(name, cutoff) if recent else ([], True)
    # A full listing with nothing older than the cutoff in it may have stopped short of the cutoff.
    posts_complete = len(listing) < LISTING_POSTS or len(recent) < len(listing)
    by_post: dict[str, list[dict[str, str]]] = {}
    for comment in reversed(comments):  # the feed is newest first; keep each thread oldest first
        match = _POST_ID.search(comment["link"] or "")
        if match:
            by_post.setdefault(match.group(1), []).append({"author": comment["author"], "text": comment["text"]})

    posts = []
    for post in recent:
        thread = by_post.get(post["id"].removeprefix("t3_"), [])
        posts.append(
            {key: post[key] for key in ("title", "author", "published", "link")}
            | {
                "url": _linked_url(post["html"]),
                "text": _REDDIT_TRAILER.sub("", post["text"]),
                "comment_count": len(thread),
                "comments": thread,
            }
        )
    return present_posts(name, posts, since_hours, sort, comments_complete, agent, posts_complete)


def present_posts(
    name: str,
    posts: list[dict[str, Any]],
    since_hours: float,
    sort: str,
    comments_complete: bool,
    agent: Any,
    posts_complete: bool = True,
) -> str:
    """Turn a subreddit's posts into what the model reads: rewritten by any local text transforms,
    numbered, remembered for write_report, and cut to fit the model's context window."""
    # Local transforms see each whole text. Only then is it cut to what the model reads.
    for post in posts:
        post["title"] = _transform(agent, post["title"])
        post["text"] = _transform(agent, post["text"])
        post["comments"] = [comment | {"text": _transform(agent, comment["text"])} for comment in post["comments"]]
    if sort == "comments":
        posts.sort(key=lambda post: post["comment_count"], reverse=True)
    # A short id per post, such as Python-3, for write_report: small models often mangle the long
    # ids in Reddit links when they copy them.
    posts = [{"id": f"{name}-{number}"} | post for number, post in enumerate(posts, 1)]
    cut = [
        post | {
            "text": _shorten(post["text"], FEED_TEXT_CHARS),
            "comments": [comment | {"text": _shorten(comment["text"], COMMENT_CHARS)} for comment in post["comments"]],
        }
        for post in posts
    ]
    result = _fit_subreddit(name, cut, 0, comments_complete)
    read = json.loads(result)
    omitted = read["omitted"]
    seen = {post["id"]: post for post in read["posts"]}
    found = {
        post["link"]: {
            "id": post["id"],
            "title": post["title"],
            "comment_count": post["comment_count"],
            "text": "\n".join([short["title"], short["text"], *(c["text"] for c in short["comments"])]),
            "read_from": f"r/{name}",
            "shown": post["id"] in seen,
            "groups": _post_groups(agent, post, seen.get(post["id"])),
        }
        for post, short in zip(posts, cut)
    }
    _remember_sources(agent, found)
    _remember_post_ids(agent, {post["id"]: post["link"] for post in posts})
    cut_groups = _groups_cut_off(found)
    coverage = (
        f"r/{name}: {len(posts)} posts and {sum(post['comment_count'] for post in posts)} comments from the last "
        f"{since_hours:g} hours."
    )
    if omitted["posts"] or omitted["comments"]:
        coverage += f" {omitted['posts']} posts and {omitted['comments']} comments didn't fit in what the model could read."
    if not comments_complete:
        coverage += f" Comment counts may be low: the comment feed stopped after {MAX_COMMENT_PAGES} pages."
    if not posts_complete:
        coverage += f" The post list stopped at {LISTING_POSTS} posts, so older posts in that window weren't read."
    if cut_groups:
        coverage += f" {cut_groups} groups a local tool found were cut off before the model read them."
    read_in_full = posts_complete and comments_complete and not omitted["posts"] and not omitted["comments"] and not cut_groups
    _remember_coverage(agent, f"r/{name}", "complete" if read_in_full else "incomplete", coverage)
    return result


def _transform(agent: Any, text: str) -> str:
    """Apply the local text transforms (from jobs/tools/) in turn. One that fails leaves the text as it was."""
    for transform in getattr(agent, "text_transforms", None) or []:
        try:
            text = transform(text)
        except Exception as error:  # a private transform's bug shouldn't stop the run
            _progress(f"text transform {getattr(transform, '__module__', transform)} failed: {error}")
    return text


def _find_groups(agent: Any, text: str, field: str, seen_chars: int) -> list[dict[str, Any]]:
    """What the local tools marked in one text (from jobs/tools/), with where each is and if the model saw it.

    `seen_chars` is how much of the text the model got. One that fails finds nothing, like a failed transform.
    """
    groups = []
    for finder in getattr(agent, "group_finders", None) or []:
        try:
            groups += [
                {key: group[key] for key in ("text", "lines", "start", "end")} | {"field": field, "shown": group["end"] <= seen_chars}
                for group in finder(text)
            ]
        except Exception as error:  # a private tool's bug shouldn't stop the run
            _progress(f"group finder {getattr(finder, '__module__', finder)} failed: {error}")
    return groups


def _post_groups(agent: Any, post: dict[str, Any], seen: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The groups in a whole post. `seen` is the post as the model got it, if it did."""
    comments_seen = len(seen["comments"]) if seen else 0
    fields = [("title", post["title"], len(post["title"]) if seen else 0), ("text", post["text"], FEED_TEXT_CHARS if seen else 0)]
    fields += [
        (f"comments[{number}]", comment["text"], COMMENT_CHARS if number < comments_seen else 0)
        for number, comment in enumerate(post["comments"])
    ]
    return [group for field, text, seen_chars in fields for group in _find_groups(agent, text, field, seen_chars)]


def _entry_groups(agent: Any, entry: dict[str, Any], seen: bool) -> list[dict[str, Any]]:
    return [
        *_find_groups(agent, entry["title"], "title", len(entry["title"]) if seen else 0),
        *_find_groups(agent, entry["text"], "text", FEED_TEXT_CHARS if seen else 0),
    ]


def _groups_cut_off(found: dict[str, dict[str, Any]]) -> int:
    """How many groups the model didn't get in posts that it did. The ones in posts it never got go with those posts."""
    return sum(not group["shown"] for facts in found.values() if facts["shown"] for group in facts["groups"])


def _read_comment_feed(name: str, cutoff: datetime) -> tuple[list[dict[str, Any]], bool]:
    """Read the subreddit's newest comments, page by page, until they are older than the cutoff."""
    comments: list[dict[str, Any]] = []
    seen: set[str] = set()
    after = ""
    for page in range(1, MAX_COMMENT_PAGES + 1):
        _progress(f"read_subreddit: reading page {page} of r/{name}'s comments")
        _, entries = parse_feed(http_get(f"https://www.reddit.com/r/{name}/comments/.rss?limit=100{after}").text)
        entries = [c for c in entries if c["id"] not in seen]
        if not entries:  # the same page again: Reddit ignored `after`
            return comments, False
        seen.update(c["id"] for c in entries)
        comments += [c for c in entries if c["published"] and c["published"] >= cutoff]
        reached_cutoff = any(c["published"] and c["published"] < cutoff for c in entries)
        if reached_cutoff or len(entries) < 100 or not entries[-1]["id"]:
            return comments, True
        after = f"&after={entries[-1]['id']}"
    return comments, False


def _linked_url(html: str) -> str | None:
    """A Reddit link post's feed entry links to its target as "[link]"; a text post's [link] is the post itself."""
    match = re.search(r'<a href="([^"]+)">\[link\]</a>', html)
    url = unescape(match.group(1)) if match else None
    return None if url is None or "reddit.com/r/" in url else url


# Reports are output, so they go under outputs/, which git ignores. Relative to where you run the agent.
REPORT_DIR = Path("outputs/reports")


# Typed Any on purpose: Strands would stop at pydantic's first complaint. parse_report checks the same
# contract as the schema advertised here, and answers with every problem at once.
@tool(context=True, inputSchema=report_schema(LINK_ITEM))
def write_report(
    tool_context: ToolContext, title: Any = None, criteria: Any = None, items: Any = None, sort: Any = "as_given"
) -> str:
    """Save the matches as an HTML report and return the file's path.

    The report shows the user's prompt word for word, your criteria, and one entry per match. Call it
    once, at the end, with every match. If nothing matched, still call it, with an empty list: the
    report shows the user what was checked. It takes each post's title and comment count from the
    post itself, and checks your quotes and links.
    """
    agent = tool_context.agent
    try:
        record, reply = _save_report(title, criteria, items, sort, tool_context)
    except ReportError as error:
        _progress(f"write_report: {error.stage} failed")
        agent.state.set(REPORT_FAILURE, {"stage": error.stage, "errors": error.errors})
        raise
    agent.state.set(REPORT, record)  # only now, with the file in place
    agent.state.delete(REPORT_FAILURE)
    return reply


def _save_report(title: Any, criteria: Any, items: Any, sort: Any, tool_context: ToolContext) -> tuple[dict[str, Any], str]:
    """Validate, resolve, check, render and save, in that order. Returns the report's record and the reply to the model."""
    agent = tool_context.agent
    settings = tool_context.invocation_state
    prompt = _user_prompt(agent.messages)
    sources = agent.state.get(SOURCES) or {}
    templates = link_patterns(prompt)
    unknown = [name for name in tool_context.tool_use.get("input", {}) if name not in ARGUMENTS]
    rows, adapted = parse_report(title, criteria, items, sort, item_contract(prompt), unknown)

    kept, unknown_posts, without_value = _resolve_items(rows, templates, agent.state.get(POST_IDS) or {}, sources)
    records, repeats = combine_values(normalize_items(kept))
    unmatched_quotes = match_sources(records, sources)
    made_up_values = _check_filled_values(records, sources)
    unsourced = _find_unsourced_links(records, prompt, agent.messages)
    # The one step that uses the network. Everything else here works offline.
    verified = settings.get("verify_links", True)
    if verified:
        verify_links(records)
    if sort == "comments":
        records.sort(key=lambda record: record.get("comment_count", -1), reverse=True)

    problems = _problems(records, unknown_posts, made_up_values, unmatched_quotes, unsourced)
    left_out = f"Left out {len(without_value)} items with no value for the link pattern the user gave." if without_value else ""
    reads = list((agent.state.get(COVERAGE) or {}).values())
    gaps = [read["line"] for read in reads if read["status"] != "complete"]
    other_warnings = {
        "no_value": without_value,
        "no_sources_read": not reads,
        "source_incomplete": any(read["status"] == "incomplete" for read in reads),
        "source_failed": any(read["status"] == "failed" for read in reads),
        "duplicate_items": repeats,
        "legacy_item": adapted,
        "links_not_checked": not verified,
    }
    warnings = [*problems, *(code for code, found in other_warnings.items() if found)]
    notices = gaps + [f"{sentence[0].upper()}{sentence[1:]}." for sentence in problems.values()]
    notices += ([left_out] if left_out else []) + ([] if reads else ["No source was read."])

    created = datetime.now().astimezone()
    coverage = [read["line"] for read in reads] + ([] if verified else ["Links were not checked over the network."])
    try:
        html = render_report(
            title, prompt, criteria, records, created, coverage, by_comments=sort == "comments", notices=notices, empty=_nothing_found(reads)
        )
    except Exception as error:  # a bug in rendering, which the model can't fix by sending the items again
        raise ReportError("render", [repr(error)]) from error
    # The CLI passes the trace's run id, so a report and its trace share a name. Several agents can run
    # at once in one folder, so without one, a random suffix keeps their reports apart.
    run_id = settings.get("run_id") or created.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    path = REPORT_DIR / report_filename(run_id, title)
    # The same records the page was built from, so whatever reads this gets exactly what the page shows.
    record = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "path": str(path.resolve()),
        "status": "saved",
        "counts": count_records(records),
        "warnings": warnings,
        "skipped": {"unknown_post": len(unknown_posts), "no_value": len(without_value), "duplicate": repeats},
        "items": records,
    }
    try:
        # For whatever reads reports downstream: the same facts as the page, as data. The page goes last,
        # because it's the report: it can't be there without the data that was asked for.
        if settings.get("report_json"):
            data = path.with_suffix(".json")
            record["data_path"] = str(data.resolve())
            details = {"title": title, "prompt": prompt, "criteria": criteria, "created": created.isoformat(timespec="seconds")}
            _save_atomically(data, json.dumps(record | details | {"sources": reads}, ensure_ascii=False, indent=2))
        _save_atomically(path, html)
    except OSError as error:
        raise ReportError("write", [str(error)]) from error
    _progress(f"write_report: saved {path}")

    note = f" {left_out}" if left_out else ""
    if repeats:
        note += f" Dropped {repeats} exact repeats of an item."
    if adapted:
        note += f" Read {adapted} items in an older shape; next time use only the fields this tool lists."
    if gaps or not reads:
        note += f" It's marked partial because not everything was read: {' '.join(gaps) or 'no source was read.'} Say so in your reply."
    if problems:
        return record, (
            f"Saved {path.resolve()}, but {'; and '.join(problems.values())}. Fix those items or leave them out, then "
            f"call write_report again with the same title to replace the report.{note}"
        )
    return record, f"Saved {path.resolve()}.{note}" if note else str(path.resolve())


def _resolve_items(
    items: list[dict[str, str]], templates: list[str], post_ids: dict[str, str], sources: dict[str, Any]
) -> tuple[list[dict[str, str]], list[str], list[dict[str, str]]]:
    """Give each item its links. Returns the items kept, the post ids no read tool handed out, and the items with no value."""
    kept = []
    unknown_posts = []
    without_value = []
    for item in items:
        resolved = _resolve_source(item, post_ids, sources)
        if resolved is None:
            unknown_posts.append(item["post"])
        elif "fill" not in item:
            value = _template_value(resolved.get("extra_link", ""), templates)
            kept.append(resolved | ({"fill": value} if value else {}))
        elif item["fill"]:
            kept.append(_fill_pattern(resolved, templates[0]))
        else:  # the user asked for a link built from a value in each post, and this post has none
            without_value.append(resolved)
    return kept, unknown_posts, without_value


def _problems(
    records: list[dict[str, Any]], unknown_posts: list[str], made_up_values: list[str], unmatched_quotes: list[str], unsourced: list[str]
) -> dict[str, str]:
    """What's wrong with the report's records, as warning codes and the sentences that explain them."""
    links = list({link["url"]: link for record in records for link in record["links"]}.values())
    missing = [_with_closest(link) for link in links if link["check"] == "missing"]
    unreachable = [link["url"] for link in links if link["check"] == "unreachable"]
    checks = {
        "unknown_post": (unknown_posts, f"these post ids aren't from read_subreddit, so those items were left out: {', '.join(unknown_posts)}"),
        "value_not_found": (made_up_values, f"these values aren't in their post or its comments: {' | '.join(made_up_values)}"),
        "quote_not_found": (unmatched_quotes, f"these quotes aren't in their post or its comments: {' | '.join(unmatched_quotes)}"),
        "link_not_from_tools": (unsourced, f"no tool returned these links and they don't fill in a pattern the user gave: {', '.join(unsourced)}"),
        "link_missing": (missing, f"these links don't exist: {', '.join(missing)}"),
        "link_unreachable": (unreachable, f"these links' hosts didn't answer, so they may or may not work: {', '.join(unreachable)}"),
    }
    return {code: sentence for code, (found, sentence) in checks.items() if found}


def saved_report(state: Any, run_id: str) -> dict[str, Any] | None:
    """The report this run saved, going by the record write_report left, if its file is really there."""
    report = state.get(REPORT)
    return report if report and report["run_id"] == run_id and Path(report["path"]).is_file() else None


def _nothing_found(reads: list[dict[str, str]]) -> str:
    """What an empty report says, which only means nothing matched if the sources were all read."""
    if not reads or all(read["status"] == "failed" for read in reads):
        return "No source could be read, so nothing was checked."
    if all(read["status"] == "complete" for read in reads):
        return "Nothing matched the criteria."
    return "Nothing matched in what could be read."


def _save_atomically(path: Path, text: str) -> None:
    """Save text as UTF-8, so that `path` only ever holds a whole file, and a failure leaves what was there.

    The text goes to a temporary file beside `path`, then moves into place in one step. A retry replaces
    an earlier report the same way. Text a model sent can hold half of a surrogate pair, which UTF-8 can't
    encode; that character becomes a "?" rather than costing the whole report.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:6]}.tmp")
    try:
        with open(temporary, "xb") as file:  # unlike mkstemp, this gives the file the usual permissions
            file.write(text.encode("utf-8", "replace"))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


_URL = re.compile(r"https?://[^\s\"'<>\\]+")
# A URL in the prompt with a placeholder, such as https://example.com/search?q=<name>.
_URL_TEMPLATE = re.compile(r"https?://[^\s\"'<>{}\\]*(?:<[^<>]+>|\{[^{}]+\})[^\s\"'<>{}\\]*")
_PLACEHOLDER = re.compile(r"<[^<>]+>|\{[^{}]+\}")


_REDDIT_POST = re.compile(r"/r/(\w+)/comments/\w+/([^/?#]+)", re.I)


def link_patterns(prompt: str) -> list[str]:
    """The URLs in the prompt with a placeholder, such as https://example.com/search?q=<name>."""
    return [trim_sentence_end(pattern) for pattern in _URL_TEMPLATE.findall(prompt)]


def item_contract(prompt: str) -> ItemContract:
    """The contract for this prompt's items. With several patterns the model gives each extra_link itself."""
    return SOURCE_ITEM if len(link_patterns(prompt)) == 1 else LINK_ITEM


def report_tool(prompt: str) -> Any:
    """write_report as this job's model sees it: the schema of the one contract parse_report will enforce."""
    job_tool = copy(write_report)
    job_tool.tool_spec = write_report.tool_spec | {"inputSchema": report_schema(item_contract(prompt))}
    return job_tool


def _resolve_source(item: dict[str, str], post_ids: dict[str, str], sources: dict[str, Any]) -> dict[str, str] | None:
    """Give an item its source's link, and fix a Reddit link whose id the model mangled.

    Small models mangle long URLs, so the model can name a source by its short id instead. Returns None
    for a `post` that no read tool handed out. A `link` stays as it is, for the provenance check to judge.
    """
    named = item.get("post") or item["link"]
    by_id = next((link for post_id, link in post_ids.items() if post_id.lower() == named.lower()), None)
    if by_id:
        return item | {"link": by_id}
    if canonical_link(named) in sources:
        return item | {"link": named}
    wanted = _REDDIT_POST.search(named)
    if wanted:
        # Same subreddit and same title words in the link: the post the model meant, if only one matches.
        same = [known for known in post_ids.values()
                if (m := _REDDIT_POST.search(known)) and m.group(1).lower() == wanted.group(1).lower()
                and m.group(2).lower() == wanted.group(2).lower()]
        if len(same) == 1:
            return item | {"link": same[0]}
    return None if "post" in item else item


def _check_filled_values(records: list[dict[str, Any]], sources: dict[str, Any]) -> list[str]:
    """Mark values that aren't in their post.

    A small model asked to decode or look up a value may write a plausible one instead. Returns those values.
    """
    made_up = []
    for record in records:
        source = sources.get(canonical_link(record["links"][0]["url"]))
        if source is not None and record["fill"] and not contains_words(source["text"], record["fill"]):
            record["value_not_found"] = True
            made_up.append(record["fill"])
    return made_up


def _template_value(url: str, templates: list[str]) -> str | None:
    """The value a URL put in place of the placeholder of one of the user's patterns, if it fills one."""
    for template in templates:
        prefix, suffix = _PLACEHOLDER.split(template, maxsplit=1)
        fills = url.lower().startswith(prefix.lower()) and url.lower().endswith(suffix.lower())
        if fills and len(url) > len(prefix) + len(suffix):
            return unquote(url[len(prefix): len(url) - len(suffix)]).strip()
    return None


def _fill_pattern(item: dict[str, str], pattern: str) -> dict[str, str]:
    """Build an item's second link from the user's link pattern and the value the model found for it.

    The model only has to find the value; a small model often mangles a URL it writes itself, for
    example python<version> as python315.
    """
    return item | {"extra_link": _PLACEHOLDER.sub(lambda _: quote(item["fill"], safe=""), pattern, count=1)}


# The part of a URL before a placeholder such as <name> or {name}.
_URL_PATTERN = re.compile(r"(https?://[^\s\"'<>{}\\]+)[<{]")


def _find_unsourced_links(records: list[dict[str, Any]], prompt: str, messages: list[dict[str, Any]]) -> list[str]:
    """Mark links that no tool returned and that don't fill in a pattern from the prompt: they may be made up.

    A pattern is a URL in the prompt with a placeholder, such as https://example.com/search?q=<name>.
    """
    returned = {
        canonical_link(url)
        for message in messages
        for block in message["content"]
        for part in block.get("toolResult", {}).get("content", [])
        for url in _URL.findall(part.get("text", ""))
    }
    patterns = [match.group(1).lower() for match in _URL_PATTERN.finditer(prompt)]
    unsourced = []
    for link in (link for record in records for link in record["links"]):
        filled_in = any(link["url"].lower().startswith(pattern) for pattern in patterns)
        if canonical_link(link["url"]) not in returned and not filled_in:
            link["unsourced"] = True
            unsourced.append(link["url"])
    return list(dict.fromkeys(unsourced))


def verify_links(items: list[dict[str, Any]], client: httpx.Client | None = None) -> None:
    """Ask each link's host whether the page is there, and note what came back in the link's `check`.

    Links start out "unverified". "reachable" means the host answered, and not with a 404 or 410.
    "missing" means it did answer 404 or 410, and if a page nearby exists it goes in `closest`: the link
    keeps its own url either way. "unreachable" means no answer, which says nothing about the page.
    Reddit links come from read_subreddit and Reddit allows one request a minute, so they stay unverified.
    A post with several values has its link in each of their records, and it's only asked for once.
    """
    client = client or httpx.Client(
        follow_redirects=True,
        timeout=10,
        headers={"User-Agent": "fetcher/0.1"},
        event_hooks={"request": [refuse_non_public_host]},
    )
    checked: dict[str, dict[str, str]] = {}
    with client:
        for link in (link for item in items for link in item["links"]):
            host = (urlsplit(link["url"]).hostname or "").lower()
            if host == "reddit.com" or host.endswith(".reddit.com"):
                continue
            if link["url"] not in checked:
                found = {"check": _check_link(client, link["url"])}
                if found["check"] == "missing" and (closest := _closest_existing_page(client, link["url"])):
                    found["closest"] = closest
                checked[link["url"]] = found
            link.update(checked[link["url"]])


def _check_link(client: httpx.Client, url: str) -> str:
    try:
        with client.stream("GET", url) as response:
            return "missing" if response.status_code in (404, 410) else "reachable"
    except (httpx.HTTPError, httpx.InvalidURL, ValueError, OSError):
        return "unreachable"


def _with_closest(link: dict[str, Any]) -> str:
    return f"{link['url']} (the closest page that exists is {link['closest']})" if "closest" in link else link["url"]


def _closest_existing_page(client: httpx.Client, url: str) -> str | None:
    """The nearest page to `url` that exists: a less specific version of it (python3.12.4 becomes
    python3.12), or else a page above it in its path, but not the bare site."""
    parts = urlsplit(url)
    segments = [segment for segment in parts.path.split("/") if segment]
    trailing_slash = "/" if parts.path.endswith("/") else ""
    last = segments[-1] if segments else ""
    while (shorter := re.sub(r"(\d)\.\d+$", r"\1", last)) != last:
        last = shorter
        candidate = f"{parts.scheme}://{parts.netloc}/{'/'.join([*segments[:-1], last])}{trailing_slash}"
        if _check_link(client, candidate) == "reachable":
            return candidate
    for depth in range(len(segments) - 1, 0, -1):
        candidate = f"{parts.scheme}://{parts.netloc}/{'/'.join(segments[:depth])}/"
        if _check_link(client, candidate) == "reachable":
            return candidate
    return None


# Key in the agent's state for what the read tools returned, by link, so write_report can check the
# report against it and account for what was found. As well as the facts it checks against, each has
# `read_from`, the source it was found in, unless it's a page fetched on its own; `shown`, whether the
# model got it at all; and `groups`, whatever the local tools marked in it, each with its text, lines,
# `field`, `start`, `end` and `shown`. Strands passes the running agent to tools that have an `agent` parameter.
SOURCES = "sources"

# Strands runs a reply's tool calls at the same time, and each update below reads the state, changes it
# and writes it back. Without this, two reads ending together could lose one, and a failed read with it.
_STATE_LOCK = threading.Lock()


def _remember_sources(agent: Any, sources: dict[str | None, dict[str, Any]]) -> None:
    if agent is None:  # called directly, as in the tests
        return
    with _STATE_LOCK:
        known = agent.state.get(SOURCES) or {}
        for link, source in sources.items():
            if link:
                key = canonical_link(link)
                known[key] = _read_again(known[key], source) if key in known else source
        agent.state.set(SOURCES, known)


def _read_again(earlier: dict[str, Any], later: dict[str, Any]) -> dict[str, Any]:
    """What's known of a source read twice, such as a post whose page was then fetched: both count."""
    text = later["text"] if earlier["text"] in later["text"] else f"{earlier['text']}\n{later['text']}"
    return earlier | later | {"text": text, "shown": earlier.get("shown", True) or later.get("shown", True)}


# Key in the agent's state for how each read went, by source, so the report says what it covers and an
# empty report can't pass for sources with nothing in them. Each is {source, status, line}: status is
# "complete", "incomplete" if some of the source was left out, or "failed"; the line says so in words.
COVERAGE = "coverage"


# Keys in the agent's state for the report this run saved, and for why its last attempt saved nothing.
# See write_report; outcome.py turns them into the job's outcome.
REPORT = "report"
REPORT_FAILURE = "report_failure"


# Key in the agent's state for the short post ids read_subreddit hands out, mapped to each post's link.
POST_IDS = "post_ids"


def _remember_post_ids(agent: Any, ids: dict[str, str]) -> None:
    if agent is None:
        return
    with _STATE_LOCK:
        agent.state.set(POST_IDS, (agent.state.get(POST_IDS) or {}) | ids)


def _remember_coverage(agent: Any, source: str, status: str, line: str) -> None:
    if agent is None:
        return
    with _STATE_LOCK:
        reads = agent.state.get(COVERAGE) or {}
        earlier = reads.get(source.lower(), {}).get("status")
        if status == "failed" and earlier in ("complete", "incomplete"):
            return  # what the earlier read returned still stands
        agent.state.set(COVERAGE, reads | {source.lower(): {"source": source, "status": status, "line": line}})


@contextmanager
def _noting_failure(agent: Any, source: str) -> Iterator[None]:
    """Note a read that fails, so a report can't pass it off as a source with nothing in it."""
    try:
        yield
    except Exception as error:
        _remember_coverage(agent, source, "failed", f"{source}: could not be read ({error}).")
        raise


def _user_prompt(messages: list[dict[str, Any]]) -> str:
    """The text of the first user message: the prompt this run was started with."""
    for message in messages:
        if message["role"] == "user":
            texts = [block["text"] for block in message["content"] if "text" in block]
            if texts:
                return "\n".join(texts)
    return ""


def _fit_subreddit(name: str, posts: list[dict[str, Any]], posts_left_out: int, comments_complete: bool) -> str:
    """Cut comments from the end of every post's list, then drop the last posts, until the result fits."""
    comments_per_post = max((len(post["comments"]) for post in posts), default=0)
    while True:
        shown = [post | {"comments": post["comments"][:comments_per_post]} for post in posts]
        omitted_comments = sum(len(post["comments"]) for post in posts) - sum(len(p["comments"]) for p in shown)
        result = json.dumps(
            {
                "subreddit": name,
                "posts": shown,
                "omitted": {"posts": posts_left_out, "comments": omitted_comments},
                "comments_complete": comments_complete,
            },
            default=_iso,
            # Real characters, not \u escapes: fewer tokens, and the model copies quotes from this text.
            ensure_ascii=False,
        )
        if len(result) <= MAX_SUBREDDIT_CHARS or not posts:
            return result
        if comments_per_post > MIN_COMMENTS_PER_POST:
            comments_per_post -= 1
        else:
            posts = posts[:-1]
            posts_left_out += 1


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " [...]"


def _progress(message: str) -> None:
    """Long tool calls report progress on stderr, so the terminal shows what the agent is waiting on."""
    print(message, file=sys.stderr, flush=True)


def _iso(value: Any) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


# When each host lets us call again, from the rate-limit headers it sent back. It's kept in a small
# file per host, locked while a request is in flight, so agents running at once on this machine take
# turns instead of using up each other's budget. Reddit allows one request a minute without logging in.
RATE_LIMIT_DIR = Path(tempfile.gettempdir()) / "fetcher-rate-limits"
MAX_RATE_LIMIT_WAIT_S = 120
MAX_ATTEMPTS = 3


def http_get(url: str) -> httpx.Response:
    """GET a public URL. Waits out the host's rate limit, and retries after a 429."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")
    host = httpx.URL(url).host

    # The model picks the URL, and a fetched page can contain text that tries to steer it, so every
    # request (including each redirect) is checked to stay on the public internet.
    with httpx.Client(
        follow_redirects=True,
        timeout=20,
        headers={"User-Agent": "fetcher/0.1"},
        event_hooks={"request": [refuse_non_public_host]},
    ) as client, _host_turn(host) as turn:
        for _ in range(MAX_ATTEMPTS):
            _wait_for(host, turn)
            response = client.get(url)
            _remember_rate_limit(response, turn)
            if response.status_code != 429:
                break
    response.raise_for_status()
    return response


@contextmanager
def _host_turn(host: str) -> Iterator[TextIO]:
    """Lock the host's rate-limit file for one request. Threads and other processes wait their turn."""
    RATE_LIMIT_DIR.mkdir(exist_ok=True)
    with open(RATE_LIMIT_DIR / re.sub(r"[^A-Za-z0-9.-]", "_", host), "a+") as turn:
        fcntl.flock(turn, fcntl.LOCK_EX)  # released when the file closes
        yield turn


def _wait_for(host: str, turn: TextIO) -> None:
    turn.seek(0)
    delay = (_seconds(turn.read()) or 0) - time.time()
    if delay > MAX_RATE_LIMIT_WAIT_S:
        raise RuntimeError(f"{host} asked us to wait {round(delay)} s before the next request; try again later")
    if delay > 0:
        _progress(f"waiting {round(delay)} s for {host}'s rate limit")
        time.sleep(delay)


def _remember_rate_limit(response: httpx.Response, turn: TextIO) -> None:
    """Note when the host allows the next request: Reddit's x-ratelimit-* headers or standard Retry-After."""
    remaining = _seconds(response.headers.get("x-ratelimit-remaining"))
    if response.status_code != 429 and (remaining is None or remaining >= 1):
        return
    wait = _seconds(response.headers.get("x-ratelimit-reset")) or _seconds(response.headers.get("retry-after"))
    turn.truncate(0)
    turn.write(str(time.time() + (wait if wait is not None else 60) + 1))
    turn.flush()


def _seconds(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def refuse_non_public_host(request: httpx.Request) -> None:
    """Block requests to this machine and private networks, such as localhost:11434 where Ollama runs."""
    host = request.url.host
    for *_, sockaddr in socket.getaddrinfo(host, None):
        address = ipaddress.ip_address(sockaddr[0])
        if not address.is_global:
            raise ValueError(f"refusing to fetch {host}: it resolves to a non-public address ({address})")


_ATOM = "{http://www.w3.org/2005/Atom}"
_DC = "{http://purl.org/dc/elements/1.1/}"
FEED_TEXT_CHARS = 800


def parse_feed(xml_text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Parse an Atom or RSS 2.0 feed into (feed title, entries)."""
    # ElementTree doesn't fetch external entities, and the expat it uses limits entity expansion.
    root = ElementTree.fromstring(xml_text)
    if root.tag == f"{_ATOM}feed":
        return root.findtext(f"{_ATOM}title"), [_atom_entry(e) for e in root.iter(f"{_ATOM}entry")]
    channel = root.find("channel")
    if root.tag != "rss" or channel is None:
        raise ValueError("not an RSS or Atom feed")
    return channel.findtext("title"), [_rss_item(item) for item in channel.iter("item")]


def _atom_entry(entry: ElementTree.Element) -> dict[str, Any]:
    link = entry.find(f"{_ATOM}link")
    return {
        "id": entry.findtext(f"{_ATOM}id"),
        "title": entry.findtext(f"{_ATOM}title"),
        "author": entry.findtext(f"{_ATOM}author/{_ATOM}name"),
        "published": _parse_time(entry.findtext(f"{_ATOM}published") or entry.findtext(f"{_ATOM}updated")),
        "link": link.get("href") if link is not None else None,
        "text": _entry_text(entry.findtext(f"{_ATOM}content") or entry.findtext(f"{_ATOM}summary")),
        "html": entry.findtext(f"{_ATOM}content") or entry.findtext(f"{_ATOM}summary") or "",
    }


def _rss_item(item: ElementTree.Element) -> dict[str, Any]:
    return {
        "id": item.findtext("guid") or item.findtext("link"),
        "title": item.findtext("title"),
        "author": item.findtext("author") or item.findtext(f"{_DC}creator"),
        "published": _parse_time(item.findtext("pubDate")),
        "link": item.findtext("link"),
        "text": _entry_text(item.findtext("description")),
        "html": item.findtext("description") or "",
    }


def _parse_time(value: str | None) -> datetime | None:
    """Atom uses ISO 8601 (2026-10-26T16:52:54+00:00); RSS uses RFC 822 (Mon, 26 Oct 2026 16:52:54 GMT)."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(UTC)


def _entry_text(html: str | None) -> str:
    return html_to_text(html or "")


MAX_LINKS = 25


def find_links(html: str, page_url: str, words: str) -> str:
    """The page's links whose text or address contains every one of `words`, in page order."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    wanted = words.lower().split()
    found: dict[str, str] = {}
    for text, href in parser.links:
        url = urljoin(page_url, href)
        if urlsplit(url).scheme in ("http", "https") and all(w in f"{text} {url}".lower() for w in wanted):
            found.setdefault(url, text or url)
    if not found:
        return f"No links on {page_url} contain {words!r}."
    lines = [f"{text}: {url}" for url, text in list(found.items())[:MAX_LINKS]]
    more = f"\n[{len(found) - MAX_LINKS} more not shown]" if len(found) > MAX_LINKS else ""
    return "\n".join(lines) + more


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "template"}
    _BLOCK = {
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "footer", "h1", "h2",
        "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table",
        "td", "th", "title", "tr", "ul",
    }

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.links: list[tuple[str, str]] = []  # (text, href) of each <a href>, in page order
        self._skip_depth = 0
        self._link: tuple[str, list[str]] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")
        if tag == "a" and (href := dict(attrs).get("href")):
            self._link = (href, [])

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK:
            self.parts.append("\n")
        if tag == "a" and self._link:
            href, text = self._link
            self.links.append((" ".join("".join(text).split()), href))
            self._link = None

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)
            if self._link:
                self._link[1].append(data)


def html_to_text(html: str) -> str:
    """Keep the visible text of an HTML page, one block per line, and drop scripts and styles."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)
