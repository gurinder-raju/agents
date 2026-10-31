"""Render a link report: the prompt it answers, what counted as a match, and the matching links.

The page is a single HTML file with no scripts and nothing loaded from elsewhere, so it opens
offline and can be shared as is. Everything in it comes from the model or from strangers' posts,
so every value is escaped and only http(s) links are kept.

What the model sends is checked here first, whole, against one of two flat item contracts. The same
contract builds the schema the model is shown, so what it's told and what's enforced can't drift apart.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from html import escape
from typing import Any
from urllib.parse import urlsplit

ARGUMENTS = ("title", "criteria", "items", "sort")
SORTS = ("as_given", "comments")
_URL_FIELDS = ("link", "extra_link")


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
    note="Every item has all four.",
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
                    "items": {
                        "type": "object",
                        "properties": {
                            name: {"type": "string", "description": text} for name, text in contract.fields.items()
                        },
                        "required": list(contract.required),
                        "additionalProperties": False,
                    },
                },
                "sort": {
                    "type": "string",
                    "enum": list(SORTS),
                    "description": '"comments" to list the most commented posts first, or "as_given" to keep your order.',
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
    title: Any, criteria: Any, items: Any, sort: Any, contract: ItemContract, unknown: list[str] = ()
) -> tuple[list[dict[str, str]], int]:
    """Check everything write_report was given, before anything acts on it.

    `unknown` names any arguments the tool doesn't have. Raises ReportError with every problem at once,
    each with its field's path and the type expected, so one retry can fix them all. Returns the items
    as trimmed strings, without empty optional fields, and how many had to be adapted from an older shape.
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
    if errors:
        raise ReportError("validation", errors)
    flat = [{name: value.strip() for name, value in item.items() if value.strip() or name == "fill"} for item, _ in adapted]
    return flat, sum(changed for _, changed in adapted)


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
    """Give each checked item, its source's link now resolved, a title, description, quote and links: [{label, url}]."""
    normalized = []
    for item in items:
        links = [{"label": "post", "url": item["link"]}]
        if extra_link := item.get("extra_link"):
            links.append({"label": item.get("extra_label") or urlsplit(extra_link).netloc, "url": extra_link})
        normalized.append({
            "title": item["link"],
            "description": item["description"],
            "quote": item.get("quote", "").strip('"\u201c\u201d'),
            "links": links,
        })
    return normalized


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
