"""Render a link report: the prompt it answers, what counted as a match, and the matching links.

The page is a single HTML file with no scripts and nothing loaded from elsewhere, so it opens
offline and can be shared as is. Everything in it comes from the model or from strangers' posts,
so every value is escaped and only http(s) links are kept.
"""

import re
from datetime import datetime
from html import escape
from typing import Any
from urllib.parse import urlsplit


def normalize_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check the model's items and give each one title, description, quote and links: [{label, url}].

    An item gives its links as `link` and an optional `extra_link`. Small models don't always follow
    the schema exactly, so a `url` or a `links` list of URLs or {label, url} objects works too.
    """
    normalized = []
    for number, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError(f"item {number} must be an object with title, description and links")
        if item.get("link"):
            raw_links = [{"label": "post", "url": _urls_in(str(item["link"]))[0]}]
            if item.get("extra_link"):
                raw_links += [{"label": item.get("extra_label"), "url": url} for url in _urls_in(str(item["extra_link"]))]
        else:
            raw_links = item.get("links") or ([item["url"]] if item.get("url") else [])
        if isinstance(raw_links, (str, dict)):
            raw_links = [raw_links]
        links = []
        for link in raw_links:
            url = link if isinstance(link, str) else str(link.get("url", ""))
            label = None if isinstance(link, str) else link.get("label")
            if urlsplit(url).scheme not in ("http", "https") or not urlsplit(url).netloc:
                raise ValueError(f"item {number}: {url!r} is not an http:// or https:// link")
            links.append({"label": str(label or urlsplit(url).netloc), "url": url})
        if not links:
            raise ValueError(f"item {number} has no links")
        normalized.append({
            "title": str(item.get("title") or links[0]["url"]),
            "description": str(item.get("description") or ""),
            "quote": str(item.get("quote") or "").strip().strip('"\u201c\u201d'),
            "links": links,
        })
    return normalized


def _urls_in(value: str) -> list[str]:
    """The URLs in a link field. Small models sometimes put two in one ("https://a/ and https://b/")."""
    return [url.rstrip(".,)") for url in re.findall(r"https?://[^\s<>\"']+", value)] or [value.strip()]


def match_sources(items: list[dict[str, Any]], sources: dict[str, dict[str, Any]]) -> list[str]:
    """Fill in facts from the posts and pages the tools returned, rather than trusting the model's copy.

    An item that links to a known source takes its title and comment count, and its quote is checked
    against the source's text. Returns the quotes that weren't found, and marks those items.
    """
    not_found = []
    for item in items:
        keys = (canonical_link(link["url"]) for link in item["links"])
        source = next((sources[key] for key in keys if key in sources), None)
        if source is None:
            continue
        if source.get("title"):
            item["title"] = source["title"]
        if source.get("comment_count") is not None:
            item["comment_count"] = source["comment_count"]
        if item["quote"] and not contains_words(source["text"], item["quote"]):
            item["quote_not_found"] = True
            not_found.append(item["quote"])
    return not_found


def merge_duplicates(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """One entry per post: items whose first link is the same page become one, keeping every second link.

    Returns the merged items and how many were folded into others.
    """
    merged: list[dict[str, Any]] = []
    by_page: dict[str, dict[str, Any]] = {}
    for item in items:
        page = canonical_link(item["links"][0]["url"])
        first = by_page.get(page)
        if first is None:
            by_page[page] = item
            merged.append(item)
            continue
        seen = {canonical_link(link["url"]) for link in first["links"]}
        first["links"] += [link for link in item["links"][1:] if canonical_link(link["url"]) not in seen]
        first["quote"] = first["quote"] or item["quote"]
        first["description"] = first["description"] or item["description"]
    return merged, len(items) - len(merged)


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
) -> str:
    """Return the report as a complete HTML page. `items` must already be normalized.

    `coverage` has a line per source the tools read, such as how many posts didn't fit.
    """
    if items:
        entries = "\n".join(_render_item(item) for item in items)
        body = f'<ol class="items">\n{entries}\n</ol>'
    else:
        body = '<p class="empty">Nothing matched the criteria.</p>'
    count = f"{len(items)} {'match' if len(items) == 1 else 'matches'}" + (", most commented first" if by_comments else "")
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
</header>
<section class="brief">
<h2>Prompt</h2>
<p class="prompt">{escape(prompt)}</p>
<h2>Criteria</h2>
<p>{escape(criteria)}</p>{_render_coverage(coverage)}
</section>
<section>
<h2>Links</h2>
{body}
</section>
</main>
</body>
</html>
"""


def _render_coverage(coverage: list[str]) -> str:
    if not coverage:
        return ""
    lines = "\n".join(f"<li>{escape(line)}</li>" for line in coverage)
    return f'\n<h2>Read</h2>\n<ul class="coverage">\n{lines}\n</ul>'


def _render_item(item: dict[str, Any]) -> str:
    links = " · ".join(
        f'<a href="{escape(link["url"])}" rel="noopener noreferrer">{escape(link["label"])}</a>'
        + (' <span class="missing">(not found)</span>' if link.get("missing") else "")
        + (' <span class="missing">(unverified)</span>' if link.get("unverified") else "")
        + (f' <span class="note" title="{escape(link["closest_to"])}">(closest page that exists)</span>' if link.get("closest_to") else "")
        + (' <span class="missing">(value not in the post)</span>' if link.get("value_not_found") else "")
        for link in item["links"]
    )
    count = item.get("comment_count")
    meta = f'\n<p class="meta">{count} comment{"" if count == 1 else "s"}</p>' if count is not None else ""
    description = f'\n<p class="description">{escape(item["description"])}</p>' if item["description"] else ""
    quote = ""
    if item["quote"]:
        flag = ' <span class="missing">(not found in the post)</span>' if item.get("quote_not_found") else ""
        quote = f'\n<blockquote>{escape(item["quote"])}{flag}</blockquote>'
    return f"""<li>
<h3><a href="{escape(item["links"][0]["url"])}" rel="noopener noreferrer">{escape(item["title"])}</a></h3>{meta}{description}{quote}
<p class="links">{links}</p>
</li>"""


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
.prompt { white-space: pre-wrap; }
.items { padding-left: 1.4em; margin: 0; }
.coverage { margin: 0; padding-left: 1.2em; }
.items li { padding: 14px 0; border-bottom: 1px solid var(--line); }
.items li:last-child { border-bottom: 0; }
.links { font-size: 0.9rem; color: var(--muted); }
.empty { color: var(--muted); }
blockquote { margin: 6px 0 8px; padding: 2px 0 2px 12px; border-left: 3px solid var(--line); color: var(--muted); }
.missing { color: #b3261e; }
.note { color: var(--muted); font-size: 0.85rem; }
"""
