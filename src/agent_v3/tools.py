"""Tools the agent can call.

Strands turns each @tool function into a tool spec for the model: the function name becomes the
tool name, the docstring becomes its description, and the typed arguments become its JSON schema.
The model never runs these. It asks for a call, the harness runs the function and sends back the result.
If a tool raises, Strands sends the error message back as the result instead, so the model can react.
"""

import fcntl
import ipaddress
import json
import re
import socket
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import quote, urljoin, urlsplit
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import httpx
from strands import ToolContext, tool

from .report import canonical_link, match_sources, normalize_items, render_report, report_filename

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
    response = http_get(url)
    if links is not None:
        return find_links(response.text, str(response.url), links)
    is_html = "html" in response.headers.get("content-type", "")
    text = html_to_text(response.text) if is_html else response.text
    _remember_sources(agent, {url: {"title": None, "text": text}})
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + f"\n\n[truncated: showing {MAX_PAGE_CHARS} of {len(text)} characters]"
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
    title, entries = parse_feed(http_get(url).text)
    entries = [{key: value for key, value in entry.items() if key != "html"} for entry in entries]
    if since_hours is not None:
        cutoff = datetime.now(UTC) - timedelta(hours=since_hours)
        entries = [e for e in entries if e["published"] is None or e["published"] >= cutoff]

    kept = entries[:max_entries]
    _remember_sources(agent, {e["link"]: {"title": e["title"], "text": f"{e['title']}\n{e['text']}"} for e in kept})
    while True:
        result = json.dumps(
            {"feed": title, "entries": kept, "omitted": len(entries) - len(kept)}, default=_iso, ensure_ascii=False
        )
        if len(result) <= MAX_PAGE_CHARS or not kept:
            return result
        kept = kept[:-1]


# About 10,000 tokens: room for a day of posts and comments in the model's 32k-token context window.
MAX_SUBREDDIT_CHARS = 40_000
COMMENT_CHARS = 500
MIN_COMMENTS_PER_POST = 5
# Each page of the subreddit's comment feed holds 100 comments and costs one request, about a minute.
MAX_COMMENT_PAGES = 10
_SUBREDDIT_NAME = re.compile(r"^[A-Za-z0-9_]{2,21}$")
_POST_ID = re.compile(r"/comments/([a-z0-9]+)/")
# Reddit ends each post's feed entry with "submitted by /u/name [link] [comments]".
_REDDIT_TRAILER = re.compile(r"\s*submitted by\s+/u/\S+\s*(\[link\])?\s*(\[comments\])?\s*$")


@tool
def read_subreddit(subreddit: str, since_hours: float = 24, sort: str = "new", agent: Any = None) -> str:
    """Read a subreddit's newest posts and their comments, as JSON.

    Each post has title, author, published (UTC), link, url (the page a link post points to, if any),
    text (the post body), comment_count and comments: a flat list of {author, text}, oldest first.
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

    # The listing holds the newest 100 posts, which covers a day for all but the busiest subreddits.
    _, listing = parse_feed(http_get(f"https://www.reddit.com/r/{name}/new/.rss?limit=100").text)
    cutoff = datetime.now(UTC) - timedelta(hours=since_hours)
    recent = [post for post in listing if post["published"] and post["published"] >= cutoff]

    # Every comment on a post from the window was written inside the window too, so reading the
    # subreddit's comment feed back to the cutoff finds all of them in a few requests, instead of
    # one request per post. comments_complete is false if the page limit stopped it early.
    comments, comments_complete = _read_comment_feed(name, cutoff) if recent else ([], True)
    by_post: dict[str, list[dict[str, str]]] = {}
    for comment in reversed(comments):  # the feed is newest first; keep each thread oldest first
        match = _POST_ID.search(comment["link"] or "")
        if match:
            by_post.setdefault(match.group(1), []).append(
                {"author": comment["author"], "text": _shorten(comment["text"], COMMENT_CHARS)}
            )

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
    if sort == "comments":
        posts.sort(key=lambda post: post["comment_count"], reverse=True)

    _remember_sources(agent, {
        post["link"]: {
            "title": post["title"],
            "comment_count": post["comment_count"],
            "text": "\n".join([post["title"], post["text"], *(c["text"] for c in post["comments"])]),
        }
        for post in posts
    })
    result = _fit_subreddit(name, posts, 0, comments_complete)
    omitted = json.loads(result)["omitted"]
    coverage = (
        f"r/{name}: {len(posts)} posts and {sum(post['comment_count'] for post in posts)} comments from the last "
        f"{since_hours:g} hours."
    )
    if omitted["posts"] or omitted["comments"]:
        coverage += f" {omitted['posts']} posts and {omitted['comments']} comments didn't fit in what the model could read."
    if not comments_complete:
        coverage += f" Comment counts may be low: the comment feed stopped after {MAX_COMMENT_PAGES} pages."
    _remember_coverage(agent, coverage)
    return result


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


# Each item is a flat object of strings. Small models such as gemma4:e2b often close a list nested
# inside an item with the wrong bracket, and Ollama then can't parse the call and drops it.
@tool(context=True)
def write_report(
    title: str, criteria: str, items: list[dict[str, str]], tool_context: ToolContext, sort: str = "as_given"
) -> str:
    """Save the matches as an HTML report and return the file's path.

    The report shows the user's prompt word for word, your criteria, and one entry per match. Call it
    once, at the end, with every match. If nothing matched, still call it, with an empty list: the
    report shows the user what was checked. It takes each post's title and comment count from the
    post itself, and checks your quotes and links.

    Args:
        title: Short title for the report, for example "r/Python posts that mention uv".
        criteria: In one or two sentences, what made a post or page count as a match.
        items: One object per match, with these fields:
            link: the post or page that matches, exactly as your tools returned it.
            description: one sentence on why it matches.
            quote: the exact words from the post or its comments that match, copied word for word.
            fill: if the user gave a link pattern with a placeholder, such as
                https://example.com/search?q=<name>, the value you found for the placeholder in this
                post, such as a name. The report builds the second link from it.
            extra_link: otherwise, an optional second link your tools returned, such as the page a
                link post points to. Never make one up.
            extra_label: one or two words for the second link, such as "download".
        sort: "comments" to list the most commented posts first, or "as_given" to keep your order.
    """
    if sort not in ("as_given", "comments"):
        raise ValueError('sort must be "as_given" or "comments"')
    agent = tool_context.agent
    prompt = _user_prompt(agent.messages)
    templates = [template.rstrip(".,;:)") for template in _URL_TEMPLATE.findall(prompt)]
    items = normalize_items([_fill_template(item, templates) for item in items])
    unmatched_quotes = match_sources(items, agent.state.get(SOURCES) or {})
    unverified = _find_unverified_links(items, prompt, agent.messages)
    missing = _find_missing_links(items)
    if sort == "comments":
        items.sort(key=lambda item: item.get("comment_count", -1), reverse=True)

    created = datetime.now().astimezone()
    coverage = agent.state.get(COVERAGE) or []
    html = render_report(title, prompt, criteria, items, created, coverage, by_comments=sort == "comments")
    # The CLI passes the trace's run id, so a report and its trace share a name. Several agents can run
    # at once in one folder, so without one, a random suffix keeps their reports apart.
    run_id = tool_context.invocation_state.get("run_id") or created.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    path = REPORT_DIR / report_filename(run_id, title)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    _progress(f"write_report: saved {path}")

    replaced = [f"{link['closest_to']} -> {link['url']}" for item in items for link in item["links"] if "closest_to" in link]
    note = f" These links don't exist, so it links the closest page that does: {', '.join(replaced)}." if replaced else ""
    problems = []
    if unmatched_quotes:
        problems.append(f"these quotes aren't in their post or its comments: {' | '.join(unmatched_quotes)}")
    if unverified:
        problems.append(f"no tool returned these links and they don't fill in a pattern the user gave: {', '.join(unverified)}")
    if missing:
        problems.append(f"these links don't exist: {', '.join(missing)}")
    if problems:
        return (
            f"Saved {path.resolve()}, but {'; and '.join(problems)}. Fix those items or leave them out, then "
            f"call write_report again with the same title to replace the report.{note}"
        )
    return f"Saved {path.resolve()}.{note}" if note else str(path.resolve())


_URL = re.compile(r"https?://[^\s\"'<>\\]+")
# A URL in the prompt with a placeholder, such as https://example.com/search?q=<name>.
_URL_TEMPLATE = re.compile(r"https?://[^\s\"'<>{}\\]*(?:<[^<>]+>|\{[^{}]+\})[^\s\"'<>{}\\]*")
_PLACEHOLDER = re.compile(r"<[^<>]+>|\{[^{}]+\}")


def _fill_template(item: Any, templates: list[str]) -> Any:
    """Build an item's second link from the user's link pattern and the value the model found for it.

    The model only has to find the value; a small model often mangles a URL it writes itself, for
    example python<version> as python315. With more than one pattern, the model gives extra_link instead.
    """
    if not isinstance(item, dict) or not str(item.get("fill") or "").strip() or len(templates) != 1:
        return item
    value = quote(str(item["fill"]).strip(), safe="")
    return item | {"extra_link": _PLACEHOLDER.sub(lambda _: value, templates[0], count=1)}
# The part of a URL before a placeholder such as <name> or {name}.
_URL_PATTERN = re.compile(r"(https?://[^\s\"'<>{}\\]+)[<{]")


def _find_unverified_links(items: list[dict[str, Any]], prompt: str, messages: list[dict[str, Any]]) -> list[str]:
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
    unverified = []
    for link in (link for item in items for link in item["links"]):
        filled_in = any(link["url"].lower().startswith(pattern) for pattern in patterns)
        if canonical_link(link["url"]) not in returned and not filled_in:
            link["unverified"] = True
            unverified.append(link["url"])
    return unverified


def _find_missing_links(items: list[dict[str, Any]], client: httpx.Client | None = None) -> list[str]:
    """Check links, so broken ones don't reach the report as if they worked.

    A link that answers 404 or 410 is replaced with the closest page above it that exists, such as
    https://example.com/files/ for https://example.com/files/2.1/, and keeps the original in
    `closest_to`. If there is none, or the host doesn't answer, it's marked missing. Reddit links
    come from read_subreddit and Reddit allows one request a minute, so they aren't checked.
    """
    missing = []
    client = client or httpx.Client(
        follow_redirects=True,
        timeout=10,
        headers={"User-Agent": "agent-v3/0.1"},
        event_hooks={"request": [refuse_non_public_host]},
    )
    with client:
        for link in (link for item in items for link in item["links"]):
            if httpx.URL(link["url"]).host.removeprefix("www.").endswith("reddit.com"):
                continue
            link["missing"] = not _page_exists(client, link["url"])
            if link["missing"] and (closest := _closest_existing_page(client, link["url"])):
                link.update(closest_to=link["url"], url=closest, missing=False)
            if link["missing"]:
                missing.append(link["url"])
    return missing


def _page_exists(client: httpx.Client, url: str) -> bool:
    try:
        with client.stream("GET", url) as response:
            return response.status_code not in (404, 410)
    except (httpx.HTTPError, ValueError, OSError):
        return False


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
        if _page_exists(client, candidate):
            return candidate
    for depth in range(len(segments) - 1, 0, -1):
        candidate = f"{parts.scheme}://{parts.netloc}/{'/'.join(segments[:depth])}/"
        if _page_exists(client, candidate):
            return candidate
    return None


# Key in the agent's state for what the read tools returned, by link, so write_report can check
# the report against it. Strands passes the running agent to tools that have an `agent` parameter.
SOURCES = "sources"


def _remember_sources(agent: Any, sources: dict[str | None, dict[str, Any]]) -> None:
    if agent is None:  # called directly, as in the tests
        return
    known = agent.state.get(SOURCES) or {}
    known.update({canonical_link(link): source for link, source in sources.items() if link})
    agent.state.set(SOURCES, known)


# Key in the agent's state for one line per source read, such as how many posts a subreddit had and
# how many didn't fit, so the report says what it covers.
COVERAGE = "coverage"


def _remember_coverage(agent: Any, line: str) -> None:
    if agent is None:
        return
    lines = agent.state.get(COVERAGE) or []
    agent.state.set(COVERAGE, [*lines, line] if line not in lines else lines)


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
RATE_LIMIT_DIR = Path(tempfile.gettempdir()) / "agent-v3-rate-limits"
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
        headers={"User-Agent": "agent-v3/0.1"},
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
    return _shorten(html_to_text(html or ""), FEED_TEXT_CHARS)


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
