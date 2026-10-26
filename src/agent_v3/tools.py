"""Tools the agent can call.

Strands turns each @tool function into a tool spec for the model: the function name becomes the
tool name, the docstring becomes its description, and the typed arguments become its JSON schema.
The model never runs these. It asks for a call, the harness runs the function and sends back the result.
If a tool raises, Strands sends the error message back as the result instead, so the model can react.
"""

import ipaddress
import json
import socket
import threading
import time
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import httpx
from strands import tool

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
def fetch_url(url: str) -> str:
    """Fetch a web page and return its readable text, with the HTML stripped out.

    Args:
        url: Full http:// or https:// address of the page.
    """
    response = http_get(url)
    is_html = "html" in response.headers.get("content-type", "")
    text = html_to_text(response.text) if is_html else response.text
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + f"\n\n[truncated: showing {MAX_PAGE_CHARS} of {len(text)} characters]"
    return text


@tool
def read_feed(url: str, since_hours: float | None = None, max_entries: int = 25) -> str:
    """Read an RSS or Atom feed and return its entries in feed order, as JSON.

    Each entry has id, title, author, published (UTC), link and text: the entry's body as plain
    text, cut to 800 characters. "omitted" counts matching entries left out to keep the result small.

    Args:
        url: Full http:// or https:// address of the feed.
        since_hours: Only keep entries published within this many hours. Omit to keep them all.
        max_entries: Keep at most this many entries.
    """
    title, entries = parse_feed(http_get(url).text)
    if since_hours is not None:
        cutoff = datetime.now(UTC) - timedelta(hours=since_hours)
        entries = [e for e in entries if e["published"] is None or e["published"] >= cutoff]

    kept = entries[:max_entries]
    while True:
        result = json.dumps({"feed": title, "entries": kept, "omitted": len(entries) - len(kept)}, default=_iso)
        if len(result) <= MAX_PAGE_CHARS or not kept:
            return result
        kept = kept[:-1]


def _iso(value: Any) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


# Earliest time each host lets us call again, from the rate-limit headers it sent back.
_next_request_at: dict[str, float] = {}
_rate_limit_lock = threading.Lock()
MAX_RATE_LIMIT_WAIT_S = 120


def http_get(url: str) -> httpx.Response:
    """GET a public URL. Waits out the host's rate limit, and retries once after a 429."""
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
    ) as client:
        # One request at a time, so tool calls the model makes in parallel share the host's budget.
        with _rate_limit_lock:
            for _ in range(2):
                _wait_for(host)
                response = client.get(url)
                _remember_rate_limit(host, response)
                if response.status_code != 429:
                    break
    response.raise_for_status()
    return response


def _wait_for(host: str) -> None:
    delay = _next_request_at.get(host, 0) - time.monotonic()
    if delay > MAX_RATE_LIMIT_WAIT_S:
        raise RuntimeError(f"{host} asked us to wait {round(delay)} s before the next request; try again later")
    if delay > 0:
        time.sleep(delay)


def _remember_rate_limit(host: str, response: httpx.Response) -> None:
    """Note when the host allows the next request: Reddit's x-ratelimit-* headers or standard Retry-After."""
    remaining = _seconds(response.headers.get("x-ratelimit-remaining"))
    if response.status_code != 429 and (remaining is None or remaining >= 1):
        return
    wait = _seconds(response.headers.get("x-ratelimit-reset")) or _seconds(response.headers.get("retry-after"))
    _next_request_at[host] = time.monotonic() + (wait if wait is not None else 60) + 1


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
    }


def _rss_item(item: ElementTree.Element) -> dict[str, Any]:
    return {
        "id": item.findtext("guid") or item.findtext("link"),
        "title": item.findtext("title"),
        "author": item.findtext("author") or item.findtext(f"{_DC}creator"),
        "published": _parse_time(item.findtext("pubDate")),
        "link": item.findtext("link"),
        "text": _entry_text(item.findtext("description")),
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
    text = html_to_text(html or "")
    return text if len(text) <= FEED_TEXT_CHARS else text[:FEED_TEXT_CHARS] + " [...]"


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
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Keep the visible text of an HTML page, one block per line, and drop scripts and styles."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)
