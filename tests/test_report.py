import json
import os
import stat
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from strands.agent.state import AgentState

from fetcher import tools
from fetcher.report import (
    LINK_ITEM,
    SOURCE_ITEM,
    ReportError,
    canonical_link,
    combine_values,
    normalize_items,
    parse_report,
    report_schema,
    url_problem,
)
from fetcher.tools import item_contract, link_patterns, report_tool, write_report

POST = "https://www.reddit.com/r/Python/comments/1abc23/switched_to_uv/"
QUIET = "https://www.reddit.com/r/Python/comments/9zz/quiet_post/"
ARTICLE = "https://example.com/uv-in-production"

# What read_subreddit remembered about the posts it returned.
SOURCES = {
    canonical_link(POST): {"title": "Switched to uv", "comment_count": 14, "text": "Switched to uv\nWe moved to uv 0.9 last week."},
    canonical_link(QUIET): {"title": "A quiet post", "comment_count": 2, "text": "A quiet post\nPython 3.14 is out."},
}


# What read_subreddit recorded about how the read went.
READ = {"source": "r/Python", "status": "complete", "line": "r/Python: 2 posts and 16 comments from the last 24 hours."}
FAILED_READ = {"source": "r/rust", "status": "failed", "line": "r/rust: could not be read (no route to host)."}
SHORT_READ = {"source": "r/rust", "status": "incomplete", "line": "r/rust: 40 posts and 300 comments. 9 posts didn't fit."}


def tool_context(prompt="Find posts in r/Python about uv", run_id=None, tool_output=f"{POST} {QUIET} {ARTICLE}", tool_input=None, **settings):
    """The parts of Strands' ToolContext that write_report reads: the conversation, the agent's state, the
    run's settings, which keep these tests off the network, and the call as the model made it."""
    messages = [
        {"role": "user", "content": [{"text": prompt}]},
        {"role": "assistant", "content": [{"text": "Reading r/Python."}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": tool_output}]}}]},
    ]
    agent = SimpleNamespace(messages=messages, state=AgentState({tools.SOURCES: SOURCES, tools.COVERAGE: {"r/python": READ}}))
    invocation_state = {"verify_links": False} | ({"run_id": run_id} if run_id else {}) | settings
    return SimpleNamespace(agent=agent, invocation_state=invocation_state, tool_use={"input": tool_input or {}})


@pytest.fixture
def report_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "REPORT_DIR", tmp_path)
    return tmp_path


def test_writes_the_prompt_criteria_and_links(report_dir):
    path = write_report(
        title="r/Python posts about uv",
        criteria="Posts about moving to uv.",
        items=[{
            "link": POST,
            "description": "The author moved three services to uv.",
            "quote": "moved to uv 0.9",
            "extra_link": ARTICLE,
            "extra_label": "article",
        }],
        tool_context=tool_context(run_id="20261026-131500-a1b2c3"),
    )
    assert path == str(report_dir / "20261026-131500-a1b2c3-r-python-posts-about-uv.html")
    html = open(path).read()
    assert "Find posts in r/Python about uv" in html
    assert "Posts about moving to uv." in html
    assert '<a href="' + POST + '" rel="noopener noreferrer">Switched to uv</a>' in html  # the post's own title
    assert f'<a href="{POST}" rel="noopener noreferrer">post</a>' in html
    assert f'<a href="{ARTICLE}" rel="noopener noreferrer">article</a>' in html
    assert "<blockquote>moved to uv 0.9</blockquote>" in html
    assert "1 match" in html and "14 comments" in html


def test_an_empty_report_says_nothing_matched(report_dir):
    path = write_report(title="Nothing here", criteria="Posts about uv.", items=[], tool_context=tool_context())
    html = open(path).read()
    assert "Nothing matched the criteria." in html
    assert "Partial report" not in html


def test_reports_from_runs_without_a_run_id_get_different_files(report_dir):
    first = write_report(title="Same title", criteria="c", items=[], tool_context=tool_context())
    second = write_report(title="Same title", criteria="c", items=[], tool_context=tool_context())
    assert first != second


def test_takes_titles_and_counts_from_the_posts_and_sorts_by_comments(report_dir):
    items = [
        {"link": QUIET, "description": "Mentions 3.14."},
        {"link": POST.replace("https://www.", "https://"), "description": "Moved to uv."},
    ]
    html = open(write_report(title="t", criteria="c", items=items, sort="comments", tool_context=tool_context())).read()

    assert html.index("Switched to uv") < html.index("A quiet post")  # 14 comments before 2
    assert "most commented first" in html


def test_says_when_a_quote_is_not_in_the_post(report_dir):
    result = write_report(
        title="t", criteria="c", items=[{"link": POST, "description": "d", "quote": "Python 3.13 benchmarks"}], tool_context=tool_context()
    )
    assert "Python 3.13 benchmarks" in result and "call write_report again" in result
    [path] = report_dir.iterdir()
    assert "(not found in the post)" in path.read_text()


def test_flags_links_no_tool_returned_unless_they_fill_in_a_pattern_from_the_prompt(report_dir):
    made_up = "https://www.python.org/downloads/release/python-3130a1/"
    search = "https://example.org/search?q=river_stone"
    prompt = "Look things up on https://example.net/find/<id> and https://example.org/search?q=<name>"
    result = write_report(
        title="t",
        criteria="c",
        items=[{"link": POST, "description": "d", "extra_link": made_up}, {"link": QUIET, "description": "d", "extra_link": search}],
        tool_context=tool_context(prompt),
    )
    assert made_up in result and search not in result
    [path] = report_dir.iterdir()
    assert path.read_text().count("(no tool returned this link)") == 1


def test_says_when_a_link_does_not_exist(report_dir, monkeypatch):
    def mark_extra_link_missing(items):
        items[0]["links"][1]["check"] = "missing"

    monkeypatch.setattr(tools, "verify_links", mark_extra_link_missing)
    items = [{"link": POST, "description": "d", "extra_link": ARTICLE}]
    result = write_report(title="t", criteria="c", items=items, tool_context=tool_context(verify_links=True))
    assert f"these links don't exist: {ARTICLE}" in result
    [path] = report_dir.iterdir()
    assert "(not found)" in path.read_text()


def test_escapes_text_from_posts(report_dir):
    context = tool_context()
    context.agent.state.set(tools.SOURCES, {canonical_link(ARTICLE): {"title": "<img src=x onerror=alert(1)>", "text": "x"}})
    items = [{"link": ARTICLE, "description": "a & b"}]
    html = open(write_report(title="<script>alert(1)</script>", criteria="c", items=items, tool_context=context)).read()
    assert "<script>alert" not in html and "<img" not in html
    assert "a &amp; b" in html


def test_refuses_links_that_are_not_http(report_dir):
    with pytest.raises(ReportError, match=r"items\[0\]\.link: expected an http:// or https:// URL"):
        write_report(title="t", criteria="c", items=[{"link": "javascript:alert(1)", "description": "d"}], tool_context=tool_context())
    assert list(report_dir.iterdir()) == []


def test_every_problem_comes_back_at_once_with_its_path_and_expected_type():
    items = [
        {"post": "Python-1", "fill": None, "description": "d", "quote": "q"},
        {"post": "Python-2", "fill": {"name": "x"}, "description": "d", "quote": 3},
        {"post": "Python-3", "description": "", "quote": "q", "score": "9"},
        "Python-4",
    ]
    with pytest.raises(ReportError) as caught:
        parse_report(None, 7, items, "newest", SOURCE_ITEM, unknown=["version"])

    assert caught.value.stage == "validation"
    assert caught.value.errors == [
        "version: unknown argument, expected only title, criteria, items, sort, left_out",
        "title: expected a string, got null",
        "criteria: expected a string, got a number",
        "sort: expected one of as_given, comments, got 'newest'",
        "items[0].fill: expected a string, got null",
        "items[1].fill: expected a string, got an object",
        "items[1].quote: expected a string, got a number",
        "items[2].score: unknown field, expected only post, fill, description, quote",
        "items[2].fill: missing, expected a string",
        "items[2].description: expected a non-empty string, got an empty one",
        "items[3]: expected an object with post, fill, description, quote, got a string",
    ]


def test_a_tool_call_with_an_unknown_argument_saves_nothing(report_dir):
    context = tool_context(tool_input={"title": "t", "criteria": "c", "items": [], "schema_version": "2"})
    with pytest.raises(ReportError, match="schema_version: unknown argument"):
        write_report(title="t", criteria="c", items=[], tool_context=context)
    assert list(report_dir.iterdir()) == []


@pytest.mark.parametrize("value", [None, "", 3, {"url": POST}])
def test_a_link_item_needs_a_post_or_a_link_string(value):
    with pytest.raises(ReportError, match=r"items\[0\]"):
        parse_report("t", "c", [{"link": value, "description": "d"}], "as_given", LINK_ITEM)


def test_a_link_item_gives_its_post_or_its_link_not_both():
    with pytest.raises(ReportError, match=r"items\[0\]: expected exactly one of post or link, got 2"):
        parse_report("t", "c", [{"post": "Python-1", "link": POST, "description": "d"}], "as_given", LINK_ITEM)


def test_a_links_own_punctuation_survives():
    for url in (
        "https://en.wikipedia.org/wiki/Mercury_(planet)",
        "https://example.com/a,b;c?q=1&r=2#frag.",
        "https://web.archive.org/web/2026/https://example.com/",
        "https://example.com/go?to=https://example.org/",
    ):
        assert url_problem(url) is None
        [item] = normalize_items(parse_report("t", "c", [{"link": url, "description": "d"}], "as_given", LINK_ITEM)[0])
        assert item["links"][0]["url"] == url


@pytest.mark.parametrize("value", [
    "https://example.com/a/ and https://example.com/b/.",
    "https://example.com/a/,https://example.com/b/",
    "https://example.com/a/; https://example.com/b/",
    "https://example.com/two words",
    "example.com/no-scheme",
    "ftp://example.com/file",
    "https://",
    "https://example.com:port/",
    "http://[broken/",
])
def test_refuses_anything_but_one_http_url(value):
    assert url_problem(value)
    with pytest.raises(ReportError, match=r"items\[0\]\.extra_link"):
        parse_report("t", "c", [{"post": "Python-1", "description": "d", "extra_link": value}], "as_given", LINK_ITEM)


def test_adapts_the_older_item_shapes():
    items, adapted, _ = parse_report("t", "c", [
        {"title": "A title the model made up", "url": POST, "description": "d"},
        {"links": [POST, {"label": "docs", "url": ARTICLE}], "description": "d"},
        {"links": f"See {POST} and {ARTICLE}.", "description": "d"},
        {"post": "Python-1", "description": "d"},
    ], "as_given", LINK_ITEM)

    assert adapted == 3
    assert items[0] == {"link": POST, "description": "d"}
    assert items[1] == {"link": POST, "extra_link": ARTICLE, "extra_label": "docs", "description": "d"}
    assert items[2] == {"link": POST, "extra_link": ARTICLE, "description": "d"}


def test_a_link_beside_fill_is_the_source_in_a_pattern_job():
    item = {"link": POST, "fill": "uv", "description": "d", "quote": "q"}
    assert parse_report("t", "c", [item], "as_given", SOURCE_ITEM)[:2] == ([{"post": POST, "fill": "uv", "description": "d", "quote": "q"}], 1)


@pytest.mark.parametrize("item", [
    {"links": [POST, ARTICLE, "https://example.com/third"], "description": "d"},  # the contract has two link fields
    {"links": [3], "description": "d"},
    {"links": [{"label": "no url"}], "description": "d"},
    {"links": 3, "description": "d"},
])
def test_refuses_older_shapes_it_can_not_adapt(item):
    with pytest.raises(ReportError, match=r"items\[0\]"):
        parse_report("t", "c", [item], "as_given", LINK_ITEM)


def test_the_advertised_schema_is_the_contract_for_the_job():
    pattern = "Link each post to https://example.org/search?q=<name>."
    assert item_contract(pattern) is SOURCE_ITEM
    assert item_contract("Find posts about uv.") is LINK_ITEM
    assert item_contract("Link https://a.example/<x> or https://b.example/<y>.") is LINK_ITEM

    schema = report_tool(pattern).tool_spec["inputSchema"]["json"]
    assert schema["required"] == ["title", "criteria", "items"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["sort"]["enum"] == ["as_given", "comments"]
    item = schema["properties"]["items"]["items"]
    assert item["required"] == ["post", "fill", "description", "quote"] == list(item["properties"])
    assert item["additionalProperties"] is False
    assert all(field["type"] == "string" for field in item["properties"].values())  # flat: small models break nesting

    assert write_report.tool_spec["inputSchema"] == report_schema(LINK_ITEM)  # the job's copy changed only itself
    assert "fill" not in report_tool("Find posts about uv.").tool_spec["inputSchema"]["json"]["properties"]["items"]["items"]["properties"]


def test_schema_versions_and_run_metadata_are_not_arguments():
    for contract in (SOURCE_ITEM, LINK_ITEM):
        assert list(report_schema(contract)["json"]["properties"]) == ["title", "criteria", "items", "sort", "left_out"]


def test_a_pattern_at_the_end_of_a_sentence_keeps_its_own_punctuation():
    assert link_patterns("See https://example.org/<name>_(band), or https://example.net/?q={x}.") == [
        "https://example.org/<name>_(band)",
        "https://example.net/?q={x}",
    ]


def links_of(*urls):
    return [{"links": [{"label": "l", "url": url, "check": "unverified"} for url in urls]}]


def web(pages=(), down=()):
    """A client for a fake web: these paths answer 200, those hosts don't answer, and the rest is 404."""
    def fake_web(request):
        if request.url.host in down:
            raise httpx.ConnectError("no answer")
        return httpx.Response(200 if request.url.path in pages else 404)

    return httpx.Client(transport=httpx.MockTransport(fake_web))


def test_tries_a_less_specific_version_before_the_pages_above():
    [item] = links_of("https://example.com/latest/python3.12.4/")
    tools.verify_links([item], web(pages=("/latest/python3.12/", "/latest/")))
    [link] = item["links"]
    assert link["closest"] == "https://example.com/latest/python3.12/"


def test_a_missing_link_keeps_its_own_url_beside_the_closest_page_that_exists():
    url = "https://example.com/files/2.1/"
    [item] = links_of(url)
    tools.verify_links([item], web(pages=("/files/",)))
    [link] = item["links"]
    assert link["url"] == url and link["check"] == "missing"
    assert link["closest"] == "https://example.com/files/"

    html = tools.render_report("t", "p", "c", [{"title": "A", "fill": None, "evidence": [], **item}], datetime.now())
    assert f'<a href="{url}" rel="noopener noreferrer">l</a> <span class="missing">(not found)</span>' in html
    assert '<a href="https://example.com/files/" rel="noopener noreferrer">closest page that exists</a>' in html


def test_tells_reachable_missing_and_unreachable_links_apart():
    def fake_web(request):
        answers = {"/ok": 200, "/moved-away": 410, "/members": 403, "/broken": 503}
        if request.url.host == "down.example":
            raise httpx.ConnectError("no answer")
        return httpx.Response(answers.get(request.url.path, 404))

    urls = ["ok", "gone", "moved-away", "members", "broken"]
    items = links_of(*(f"https://example.com/{path}" for path in urls), "https://down.example/page", POST, "https://notreddit.com/ok")
    tools.verify_links(items, httpx.Client(transport=httpx.MockTransport(fake_web)))
    assert [link["check"] for link in items[0]["links"]] == [
        "reachable", "missing", "missing", "reachable", "reachable", "unreachable", "unverified", "reachable",
    ]


def test_a_host_that_does_not_answer_is_not_searched_for_a_closest_page():
    item = links_of("https://down.example/a/b/c")[0]
    requested = []

    def fake_web(request):
        requested.append(request.url.path)
        raise httpx.ConnectTimeout("no answer")

    tools.verify_links([item], httpx.Client(transport=httpx.MockTransport(fake_web)))
    assert requested == ["/a/b/c"] and "closest" not in item["links"][0]


def test_a_host_the_guard_refuses_counts_as_unreachable(monkeypatch):
    monkeypatch.setattr(tools.socket, "getaddrinfo", lambda *args: [(0, 0, 0, "", ("127.0.0.1", 0))])
    items = links_of("http://localhost:11434/api/tags")
    tools.verify_links(items)
    assert items[0]["links"][0]["check"] == "unreachable"


def test_says_when_a_links_host_did_not_answer(report_dir, monkeypatch):
    def mark_unreachable(items):
        items[0]["links"][0]["check"] = "unreachable"

    monkeypatch.setattr(tools, "verify_links", mark_unreachable)
    items = [{"link": ARTICLE, "description": "d"}]
    result = write_report(title="t", criteria="c", items=items, tool_context=tool_context(verify_links=True))
    assert f"these links' hosts didn't answer, so they may or may not work: {ARTICLE}" in result
    [path] = report_dir.iterdir()
    assert "(could not be reached)" in path.read_text()


def test_validation_and_rendering_work_with_no_network_at_all(report_dir, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("a report with link checks off must not touch the network")

    monkeypatch.setattr(tools.httpx, "Client", no_network)
    monkeypatch.setattr(tools.socket, "getaddrinfo", no_network)
    path = write_report(title="t", criteria="c", items=[{"link": POST, "description": "d"}], tool_context=tool_context())
    assert "Links were not checked over the network." in open(path).read()


def test_links_are_checked_unless_the_run_says_not_to(report_dir, monkeypatch):
    checked = []
    monkeypatch.setattr(tools, "verify_links", checked.append)
    context = tool_context()
    del context.invocation_state["verify_links"]
    write_report(title="t", criteria="c", items=[], tool_context=context)
    assert len(checked) == 1


def test_canonical_links_ignore_how_a_link_is_written():
    assert canonical_link("https://old.reddit.com/r/Python/comments/1abc/x/#c") == canonical_link(
        "http://www.Reddit.com/r/Python/comments/1abc/x"
    )


def test_builds_the_second_link_from_the_users_pattern_and_the_value_found(report_dir):
    prompt = "For each post, also link https://example.org/latest/python<version>/."
    result = write_report(
        title="t",
        criteria="c",
        items=[
            {"post": QUIET, "fill": "3.14", "description": "d", "quote": "Python 3.14 is out"},
            {"post": POST, "fill": "uv 0.9", "description": "d", "quote": "moved to uv 0.9"},
        ],
        tool_context=tool_context(prompt),
    )
    assert "but" not in result.lower()  # built from the user's pattern, with values that are in the posts
    html = open(result).read()
    assert '<a href="https://example.org/latest/python3.14/" rel="noopener noreferrer">3.14</a>' in html  # labelled with its value
    assert 'href="https://example.org/latest/pythonuv%200.9/"' in html


def test_fill_is_required_and_never_taken_from_the_quote(report_dir):
    prompt = "Link each post to https://example.org/search?q=<name>."
    item = {"post": QUIET, "description": "d", "quote": "Python"}
    with pytest.raises(ReportError, match=r"items\[0\]\.fill: missing, expected a string"):
        write_report(title="t", criteria="c", items=[item], tool_context=tool_context(prompt))
    assert list(report_dir.iterdir()) == []


def test_refuses_a_link_the_model_wrote_itself_in_a_pattern_job(report_dir):
    prompt = "Link each post to https://example.org/search?q=<name>."
    item = {"post": QUIET, "fill": "Python", "description": "d", "quote": "q", "extra_link": "https://example.org/search?q=Python"}
    with pytest.raises(ReportError, match=r"items\[0\]\.extra_link: unknown field"):
        write_report(title="t", criteria="c", items=[item], tool_context=tool_context(prompt))


def test_leaves_out_items_with_no_value_for_the_users_pattern(report_dir):
    prompt = "Link each post to https://example.org/search?q=<name>."
    items = [
        {"post": QUIET, "fill": "Python", "description": "d", "quote": "Python 3.14 is out"},
        {"post": POST, "fill": "", "description": "No name in this one.", "quote": "moved to uv 0.9"},
    ]
    result = write_report(title="t", criteria="c", items=items, tool_context=tool_context(prompt))
    assert "Left out 1 items with no value" in result
    [path] = report_dir.iterdir()
    html = path.read_text()
    assert "A quiet post" in html
    assert "1 post, 1 value" in html
    assert 'Switched to uv</a>: No name in this one.' in html  # under Left out, with what the model said about it


def test_flags_a_filled_value_that_is_not_in_the_post(report_dir):
    prompt = "Link each post to https://example.org/search?q=<name>."
    items = [
        {"post": QUIET, "fill": "Phantom", "description": "d", "quote": "Python 3.14 is out"},
        {"post": POST, "fill": "Ghost", "description": "d", "quote": "moved to uv 0.9"},
    ]
    result = write_report(title="t", criteria="c", items=items, tool_context=tool_context(prompt))

    assert "these values aren't in their post or its comments: Phantom | Ghost" in result
    [path] = report_dir.iterdir()
    assert path.read_text().count("(value not in the post)") == 2


def test_resolves_post_ids_and_fixes_a_mangled_reddit_link(report_dir):
    context = tool_context()
    context.agent.state.set(tools.POST_IDS, {"Python-1": POST, "Python-2": QUIET})
    mangled = QUIET.replace("/9zz/", "/9zzq/")  # the id garbled, the title words intact
    result = write_report(
        title="t",
        criteria="c",
        items=[{"post": "python-1", "description": "By id."}, {"link": mangled, "description": "By link."}, {"post": "Python-9", "description": "d"}],
        tool_context=context,
    )
    assert "these post ids aren't from read_subreddit, so those items were left out: Python-9" in result
    html = (report_dir / Path(result.split("Saved ")[1].split(",")[0]).name).read_text()
    assert f'<a href="{POST}" rel="noopener noreferrer">Switched to uv</a>' in html
    assert f'<a href="{QUIET}" rel="noopener noreferrer">A quiet post</a>' in html
    assert "(no tool returned this link)" not in html


def test_a_source_can_be_named_by_its_link_in_a_pattern_job(report_dir):
    prompt = "Link each post to https://example.org/search?q=<name>."
    items = [
        {"post": QUIET, "fill": "Python", "description": "d", "quote": "Python 3.14 is out"},
        {"post": "https://example.com/never-read", "fill": "Python", "description": "d", "quote": "q"},
    ]
    result = write_report(title="t", criteria="c", items=items, tool_context=tool_context(prompt))
    assert "so those items were left out: https://example.com/never-read" in result
    [path] = report_dir.iterdir()
    assert 'href="https://example.org/search?q=Python"' in path.read_text()


def test_a_post_with_several_values_is_one_entry_with_each_value_in_it(report_dir):
    prompt = "Link each post to https://example.org/search?q=<name>."
    items = [
        {"post": QUIET, "fill": "Python", "description": "d", "quote": "Python 3.14 is out"},
        {"post": QUIET, "fill": "quiet", "description": "d", "quote": "A quiet post"},
        {"post": QUIET, "fill": "Python", "description": "d", "quote": "Python 3.14 is out"},
    ]
    result = write_report(title="t", criteria="c", items=items, tool_context=tool_context(prompt))
    assert "Dropped 1 exact repeats" in result
    html = open(result.split("Saved ")[1].split(".html")[0] + ".html").read()
    assert html.count("<h3>") == 1
    assert html.count('href="https://example.org/search?q=') == 2  # q=Python and q=quiet, each once


PATTERN = "Link each post to https://example.org/search?q=<name>."
SEARCH = "https://example.org/search?q="
TWO_VALUES = [
    {"post": POST, "fill": "uv", "description": "Names the tool.", "quote": "moved to uv"},
    {"post": POST, "fill": "0.9", "description": "Names the version.", "quote": "uv 0.9 last week"},
]


def saved_report_of(items, prompt=PATTERN, **settings):
    context = tool_context(prompt, **settings)
    write_report(title="t", criteria="c", items=items, tool_context=context)
    report = context.agent.state.get(tools.REPORT)
    return report, Path(report["path"]).read_text()


def matches_in(html):
    return html.split('<div class="match">')[1:]


def test_every_value_keeps_its_own_quote_description_and_link(report_dir):
    report, html = saved_report_of(TWO_VALUES)

    tool, version = report["items"]
    assert (tool["post"], tool["fill"], tool["links"][1]["url"]) == (POST, "uv", SEARCH + "uv")
    assert tool["evidence"] == [{"description": "Names the tool.", "quote": "moved to uv"}]
    assert (version["fill"], version["links"][1]["url"]) == ("0.9", SEARCH + "0.9")
    assert version["evidence"] == [{"description": "Names the version.", "quote": "uv 0.9 last week"}]

    first, second = matches_in(html)
    assert '<p class="value">uv</p>' in first and "Names the tool." in first and "moved to uv" in first and SEARCH + "uv" in first
    assert '<p class="value">0.9</p>' in second and "Names the version." in second and "uv 0.9 last week" in second and SEARCH + "0.9" in second
    assert "Names the version." not in first and "Names the tool." not in second


def test_a_repeated_post_is_one_entry_and_keeps_each_value(report_dir):
    report, html = saved_report_of(TWO_VALUES)
    assert report["counts"] == {"posts": 1, "values": 2, "evidence": 2}
    assert html.count("<h3>") == 1 and len(matches_in(html)) == 2
    assert "1 post, 2 values" in html


def test_the_same_value_with_another_quote_keeps_both_quotes(report_dir):
    items = [
        {"post": POST, "fill": "uv", "description": "In the title.", "quote": "Switched to uv"},
        {"post": POST, "fill": "uv", "description": "In the body.", "quote": "moved to uv 0.9"},
    ]
    report, html = saved_report_of(items)

    [record] = report["items"]
    assert record["evidence"] == [
        {"description": "In the title.", "quote": "Switched to uv"},
        {"description": "In the body.", "quote": "moved to uv 0.9"},
    ]
    assert report["counts"] == {"posts": 1, "values": 1, "evidence": 2} and "duplicate_items" not in report["warnings"]
    [match] = matches_in(html)
    assert "<blockquote>Switched to uv</blockquote>" in match and "<blockquote>moved to uv 0.9</blockquote>" in match
    assert match.index("In the title.") < match.index("Switched to uv") < match.index("In the body.") < match.index("moved to uv 0.9")


def test_only_an_exact_repeat_is_dropped(report_dir):
    item = {"post": POST, "fill": "uv", "description": "d", "quote": "moved to uv"}
    report, _ = saved_report_of([item, item | {}, item | {"description": "Said another way."}])

    assert report["skipped"]["duplicate"] == 1 and "duplicate_items" in report["warnings"]
    assert report["counts"] == {"posts": 1, "values": 1, "evidence": 2}


def test_a_value_that_differs_only_in_case_is_another_value(report_dir):
    items = [{"post": POST, "fill": fill, "description": "d", "quote": "moved to uv"} for fill in ("uv", "UV")]
    report, html = saved_report_of(items)
    assert [record["fill"] for record in report["items"]] == ["uv", "UV"]
    assert f'href="{SEARCH}uv"' in html and f'href="{SEARCH}UV"' in html


def test_a_post_named_by_id_and_by_link_is_one_post(report_dir):
    context = tool_context(PATTERN)
    context.agent.state.set(tools.POST_IDS, {"Python-1": POST})
    items = [TWO_VALUES[0] | {"post": "Python-1"}, TWO_VALUES[1]]
    write_report(title="t", criteria="c", items=items, tool_context=context)
    report = context.agent.state.get(tools.REPORT)
    assert report["counts"]["posts"] == 1 and report["counts"]["values"] == 2


def test_plain_link_items_for_one_post_keep_each_second_link_and_quote(report_dir):
    items = [
        {"link": POST, "description": "Moved.", "quote": "moved to uv", "extra_link": ARTICLE, "extra_label": "article"},
        {"link": POST, "description": "Version.", "quote": "uv 0.9", "extra_link": QUIET, "extra_label": "follow-up"},
        {"link": POST, "description": "Plain.", "quote": "last week"},
    ]
    report, html = saved_report_of(items, prompt="Find posts in r/Python about uv")
    assert report["counts"] == {"posts": 1, "values": 0, "evidence": 3}
    first, second, third = matches_in(html)
    assert f'<a href="{ARTICLE}" rel="noopener noreferrer">article</a>' in first and "moved to uv" in first
    assert f'<a href="{QUIET}" rel="noopener noreferrer">follow-up</a>' in second and "uv 0.9" in second
    assert "last week" in third
    assert "3 matches" not in html and "1 match" in html


def test_a_second_link_filling_one_of_several_patterns_shows_its_value(report_dir):
    prompt = "Link https://a.example/<x> or https://b.example/<y>."
    report, html = saved_report_of([{"link": POST, "description": "d", "extra_link": "https://a.example/uv"}], prompt=prompt)
    assert report["items"][0]["fill"] == "uv"
    assert '<p class="value">uv</p>' in html and '<a href="https://a.example/uv" rel="noopener noreferrer">uv</a>' in html


def test_a_value_is_shown_as_it_was_given_slashes_and_all(report_dir):
    context = tool_context(PATTERN)
    context.agent.state.set(tools.SOURCES, {canonical_link(POST): {"title": "Bands", "text": "AC/DC and Hi / Yo played."}})
    items = [{"post": POST, "fill": "AC/DC", "description": "d", "quote": "AC/DC"}, {"post": POST, "fill": "Hi / Yo", "description": "d", "quote": "Hi / Yo"}]
    write_report(title="t", criteria="c", items=items, tool_context=context)
    report = context.agent.state.get(tools.REPORT)

    assert [record["fill"] for record in report["items"]] == ["AC/DC", "Hi / Yo"]
    html = Path(report["path"]).read_text()
    assert f'href="{SEARCH}AC%2FDC"' in html and f'href="{SEARCH}Hi%20%2F%20Yo"' in html
    assert "value_not_found" not in report["warnings"]


def test_two_values_in_one_fill_are_refused_when_they_are_on_two_lines():
    item = {"post": POST, "fill": "Hi\nYo", "description": "d", "quote": "q"}
    with pytest.raises(ReportError, match=r"items\[0\]\.fill: expected one value on one line, got 'Hi\\nYo'"):
        parse_report("t", "c", [item], "as_given", SOURCE_ITEM)


def test_everything_submitted_reaches_the_page_and_the_saved_data(report_dir):
    context = tool_context("Find in r/Python. " + PATTERN, run_id="20261026-131500-a1b2c3", report_json=True)
    items = [
        {"post": POST, "fill": "uv", "description": "Names the tool.", "quote": "Switched to uv"},
        {"post": POST, "fill": "uv", "description": "Used at work.", "quote": "moved to uv 0.9"},
        {"post": POST, "fill": "0.9", "description": "Names the version.", "quote": "uv 0.9 last week"},
        {"post": QUIET, "fill": "Python", "description": "Names the language.", "quote": "Python 3.14 is out"},
        {"post": QUIET, "fill": "3.14", "description": "Names its version.", "quote": "3.14 is out"},
    ]
    write_report(title="t", criteria="c", items=items, tool_context=context)
    report = context.agent.state.get(tools.REPORT)
    html = Path(report["path"]).read_text()
    data = json.loads(Path(report["data_path"]).read_text())

    assert data["counts"] == {"posts": 2, "values": 4, "evidence": 5}
    assert data["items"] == report["items"]
    pairs = {(record["fill"], evidence["quote"], evidence["description"]) for record in data["items"] for evidence in record["evidence"]}
    assert pairs == {(item["fill"], item["quote"], item["description"]) for item in items}
    links = {record["fill"]: record["links"][1]["url"] for record in data["items"]}
    assert links == {"uv": SEARCH + "uv", "0.9": SEARCH + "0.9", "Python": SEARCH + "Python", "3.14": SEARCH + "3.14"}
    for item in items:
        assert f'<p class="value">{item["fill"]}</p>' in html
        assert f'href="{SEARCH}{item["fill"]}"' in html
        assert f"<blockquote>{item['quote']}</blockquote>" in html
        assert item["description"] in html


def test_the_saved_data_is_the_records_the_page_was_built_from(report_dir, monkeypatch):
    rendered = []
    real_render = tools.render_report

    def spy(title, prompt, criteria, items, *args, **kwargs):
        rendered.append(deepcopy(items))
        return real_render(title, prompt, criteria, items, *args, **kwargs)

    monkeypatch.setattr(tools, "render_report", spy)
    report, _ = saved_report_of(TWO_VALUES, report_json=True)
    assert rendered == [report["items"]] == [json.loads(Path(report["data_path"]).read_text())["items"]]


@pytest.mark.parametrize("quote", ["First line\nSecond line", "First line\\nSecond line", "first line second line"])
def test_a_quote_over_a_line_break_matches_however_the_break_arrives(report_dir, quote):
    context = tool_context()
    context.agent.state.set(tools.SOURCES, {canonical_link(POST): {"title": "T", "text": "T\nFirst line\nSecond line"}})
    write_report(title="t", criteria="c", items=[{"link": POST, "description": "d", "quote": quote}], tool_context=context)
    assert "quote_not_found" not in context.agent.state.get(tools.REPORT)["warnings"]


@pytest.mark.parametrize("fill", ["Hi\nYo", "Hi\r\nYo", "Hi\\nYo"])
def test_a_value_with_a_line_break_in_it_is_refused_however_the_break_arrives(fill):
    with pytest.raises(ReportError, match=r"items\[0\]\.fill: expected one value on one line"):
        parse_report("t", "c", [{"post": POST, "fill": fill, "description": "d", "quote": "q"}], "as_given", SOURCE_ITEM)


def test_the_same_call_again_changes_nothing_and_says_to_stop(report_dir, monkeypatch):
    context = tool_context()
    items = [{"link": POST, "description": "d", "quote": "not in the post"}]
    first = write_report(title="t", criteria="c", items=items, tool_context=context)
    assert "call write_report again" in first

    def no_second_save(*args):
        raise AssertionError("an unchanged call has nothing to save")

    monkeypatch.setattr(tools, "_save_report", no_second_save)
    assert write_report(title="t", criteria="c", items=items, tool_context=context) == tools.SAME_REPORT


def test_a_changed_call_or_a_new_read_saves_again(report_dir):
    context = tool_context()
    items = [{"link": POST, "description": "d"}]
    write_report(title="t", criteria="c", items=items, tool_context=context)
    assert write_report(title="t", criteria="c", items=items + [{"link": QUIET, "description": "d"}], tool_context=context) != tools.SAME_REPORT

    assert write_report(title="t", criteria="c", items=items, tool_context=context) != tools.SAME_REPORT
    context.agent.state.set(tools.COVERAGE, {"r/python": READ, "r/rust": FAILED_READ})
    assert write_report(title="t", criteria="c", items=items, tool_context=context) != tools.SAME_REPORT
    assert "source_failed" in context.agent.state.get(tools.REPORT)["warnings"]


def test_changed_source_evidence_revalidates_the_same_call_even_when_coverage_is_unchanged(report_dir):
    context = tool_context()
    items = [{"link": POST, "description": "d", "quote": "new evidence"}]
    write_report(title="t", criteria="c", items=items, tool_context=context)
    assert "quote_not_found" in context.agent.state.get(tools.REPORT)["warnings"]
    sources = context.agent.state.get(tools.SOURCES)
    sources[canonical_link(POST)]["text"] += "\nnew evidence"
    context.agent.state.set(tools.SOURCES, sources)
    assert write_report(title="t", criteria="c", items=items, tool_context=context) != tools.SAME_REPORT
    assert "quote_not_found" not in context.agent.state.get(tools.REPORT)["warnings"]


def test_the_same_call_saves_again_if_the_file_has_gone(report_dir):
    context = tool_context(run_id="20261026-131500-a1b2c3")
    path = Path(write_report(title="t", criteria="c", items=[], tool_context=context))
    path.unlink()
    write_report(title="t", criteria="c", items=[], tool_context=context)
    assert path.is_file()


def test_going_back_to_the_saved_report_clears_a_failed_attempt(report_dir):
    context = tool_context()
    write_report(title="t", criteria="c", items=[], tool_context=context)
    with pytest.raises(ReportError):
        write_report(title="t", criteria="c", items=None, tool_context=context)
    assert context.agent.state.get(tools.REPORT_FAILURE) is not None
    assert write_report(title="t", criteria="c", items=[], tool_context=context) == tools.SAME_REPORT
    assert context.agent.state.get(tools.REPORT_FAILURE) is None


def test_a_failed_call_is_refused_every_time(report_dir):
    context = tool_context()
    for _ in range(2):
        with pytest.raises(ReportError):
            write_report(title="t", criteria="c", items=None, tool_context=context)


def test_each_quote_is_flagged_on_its_own(report_dir):
    items = [{"post": POST, "fill": "uv", "description": "d", "quote": "moved to uv"}, {"post": POST, "fill": "uv", "description": "d", "quote": "made this up"}]
    report, html = saved_report_of(items)
    [record] = report["items"]
    assert [evidence.get("quote_not_found", False) for evidence in record["evidence"]] == [False, True]
    assert html.count("(not found in the post)") == 1 and "<blockquote>moved to uv</blockquote>" in html


def test_a_links_check_is_asked_once_however_many_values_share_it(report_dir, monkeypatch):
    requested = []

    def fake_web(request):
        requested.append(str(request.url))
        return httpx.Response(200)

    items = [{"link": ARTICLE, "description": "One."}, {"link": ARTICLE, "description": "Two.", "extra_link": "https://example.com/b"}]
    records, _ = combine_values(normalize_items(parse_report("t", "c", items, "as_given", LINK_ITEM)[0]))
    tools.verify_links(records, httpx.Client(transport=httpx.MockTransport(fake_web)))
    assert requested == [ARTICLE, "https://example.com/b"]
    assert [link["check"] for record in records for link in record["links"]] == ["reachable"] * 3


NASTY = '"><script>alert(1)</script>'


def test_escapes_every_value_quote_label_and_reason(report_dir):
    context = tool_context(PATTERN)
    context.agent.state.set(tools.SOURCES, {canonical_link(POST): {"title": NASTY, "text": f"{NASTY} here"}})
    items = [{"post": POST, "fill": NASTY, "description": NASTY, "quote": NASTY}]
    left_out = [{"source": POST, "value": NASTY + " excluded", "reason": NASTY}, {"source": NASTY, "reason": NASTY}]
    write_report(title=NASTY, criteria=NASTY, items=items, left_out=left_out, tool_context=context)
    html = Path(context.agent.state.get(tools.REPORT)["path"]).read_text()

    assert "<script" not in html
    assert html.count("&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;") >= 8
    assert f'href="{SEARCH}%22%3E%3Cscript%3Ealert%281%29%3C%2Fscript%3E"' in html  # the link's value, encoded


def test_escapes_link_labels_and_attributes(report_dir):
    url = 'https://example.com/a"onmouseover="alert(1)'
    items = [{"link": POST, "description": "d", "extra_link": url, "extra_label": NASTY}]
    _, html = saved_report_of(items, prompt="Find posts in r/Python about uv")
    assert "onmouseover=\"" not in html and "<script" not in html
    assert 'href="https://example.com/a&quot;onmouseover=&quot;alert(1)"' in html


def test_a_link_from_a_feed_that_is_not_http_is_shown_as_plain_text(report_dir):
    context = tool_context(PATTERN)
    bad = "javascript:alert(1)"
    context.agent.state.set(tools.SOURCES, {canonical_link(bad): {"title": "Click me", "text": "Click me value"}})
    write_report(title="t", criteria="c", items=[{"post": bad, "fill": "value", "description": "d", "quote": "value"}], tool_context=context)
    html = Path(context.agent.state.get(tools.REPORT)["path"]).read_text()
    assert "<h3>Click me</h3>" in html and 'href="javascript' not in html


def test_with_several_patterns_the_model_gives_each_second_link(report_dir):
    prompt = "Link https://a.example/<x> or https://b.example/<y>."
    items = [{"link": POST, "description": "d", "fill": "v"}]
    with pytest.raises(ReportError, match=r"items\[0\]\.fill: unknown field"):
        write_report(title="t", criteria="c", items=items, tool_context=tool_context(prompt))

    items = [{"link": POST, "description": "d", "extra_link": "https://a.example/v"}]
    write_report(title="t", criteria="c", items=items, tool_context=tool_context(prompt))
    [path] = report_dir.iterdir()
    assert 'href="https://a.example/v"' in path.read_text()


def test_saves_utf8_in_a_temporary_file_beside_the_report_then_moves_it_into_place(report_dir, monkeypatch):
    replaced = []
    real_replace = os.replace

    def spy(source, destination):
        replaced.append((Path(source), Path(destination), Path(source).read_bytes()))
        real_replace(source, destination)

    monkeypatch.setattr(tools.os, "replace", spy)
    path = write_report(title="Café 🐍", criteria="naïve \ud83d", items=[], tool_context=tool_context(run_id="20261026-131500-a1b2c3"))

    [(temporary, destination, written)] = replaced
    assert temporary.parent == destination.parent == report_dir and temporary.name.endswith(".tmp")
    assert destination == Path(path) == report_dir / "20261026-131500-a1b2c3-caf.html"
    assert "Café 🐍".encode() in written and b"na\xc3\xafve ?" in written  # half a surrogate pair can't be encoded
    assert Path(path).read_bytes() == written
    assert [entry.name for entry in report_dir.iterdir()] == [destination.name]


def test_a_saved_report_gets_the_usual_permissions(report_dir):
    previous = os.umask(0o022)
    try:
        path = write_report(title="t", criteria="c", items=[], tool_context=tool_context())
    finally:
        os.umask(previous)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o644


def test_creates_the_report_folder(report_dir, monkeypatch):
    monkeypatch.setattr(tools, "REPORT_DIR", report_dir / "new" / "reports")
    path = write_report(title="t", criteria="c", items=[], tool_context=tool_context())
    assert Path(path).parent == report_dir / "new" / "reports"


def test_a_failed_write_keeps_the_earlier_report_and_leaves_no_temporary_file(report_dir, monkeypatch):
    path = write_report(title="t", criteria="first", items=[], tool_context=tool_context(run_id="20261026-131500-a1b2c3"))

    def out_of_space(source, destination):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(tools.os, "replace", out_of_space)
    with pytest.raises(ReportError, match="No space left") as caught:
        write_report(title="t", criteria="second", items=[], tool_context=tool_context(run_id="20261026-131500-a1b2c3"))

    assert caught.value.stage == "write"
    assert [entry.name for entry in report_dir.iterdir()] == [Path(path).name]
    assert "first" in Path(path).read_text()


def test_a_report_folder_that_can_not_be_made_is_a_write_failure(report_dir, monkeypatch):
    (report_dir / "taken").write_text("a file, where the folder should go")
    monkeypatch.setattr(tools, "REPORT_DIR", report_dir / "taken" / "reports")
    with pytest.raises(ReportError) as caught:
        write_report(title="t", criteria="c", items=[], tool_context=tool_context())
    assert caught.value.stage == "write"


def test_a_failed_render_is_told_apart_from_a_failed_write(report_dir, monkeypatch):
    def broken(*args, **kwargs):
        raise KeyError("title")

    monkeypatch.setattr(tools, "render_report", broken)
    with pytest.raises(ReportError, match="could not be rendered") as caught:
        write_report(title="t", criteria="c", items=[], tool_context=tool_context())
    assert caught.value.stage == "render"
    assert list(report_dir.iterdir()) == []


def test_says_what_the_tools_read(report_dir):
    path = write_report(title="t", criteria="c", items=[], tool_context=tool_context())
    assert f"<li>{READ['line']}</li>" in open(path).read()


def test_records_the_saved_report_once_the_file_is_in_place(report_dir):
    context = tool_context(run_id="20261026-131500-a1b2c3")
    context.agent.state.set(tools.POST_IDS, {"Python-1": POST})
    items = [{"post": "Python-1", "description": "d"}, {"post": "Python-9", "description": "d"}, {"post": "Python-1", "description": "d"}]
    write_report(title="t", criteria="c", items=items, tool_context=context)

    report = context.agent.state.get(tools.REPORT)
    assert {key: report[key] for key in ("schema_version", "run_id", "path", "status", "counts", "warnings", "skipped")} == {
        "schema_version": 2,
        "run_id": "20261026-131500-a1b2c3",
        "path": str(report_dir.resolve() / "20261026-131500-a1b2c3-t.html"),
        "status": "saved",
        "counts": {"posts": 1, "values": 0, "evidence": 1},
        "warnings": ["unknown_post", "duplicate_items", "links_not_checked"],
        "skipped": {"unknown_post": 1, "no_value": 0, "duplicate": 1},
    }


def test_counts_the_items_left_out_for_having_no_value(report_dir):
    context = tool_context("Link each post to https://example.org/search?q=<name>.")
    items = [{"post": POST, "fill": "", "description": "d", "quote": "q"}]
    write_report(title="t", criteria="c", items=items, tool_context=context)
    report = context.agent.state.get(tools.REPORT)
    assert report["skipped"] == {"unknown_post": 0, "no_value": 1, "duplicate": 0} and "no_value" in report["warnings"]


@pytest.mark.parametrize("stage, setup, expected", [
    ("validation", lambda monkeypatch: None, "items: expected a list of objects, got null"),
    ("render", lambda monkeypatch: monkeypatch.setattr(tools, "render_report", lambda *a, **k: 1 / 0), "ZeroDivisionError('division by zero')"),
    ("write", lambda monkeypatch: monkeypatch.setattr(tools.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full"))), "disk full"),
])
def test_a_failed_attempt_records_its_stage_and_never_a_saved_report(report_dir, monkeypatch, stage, setup, expected):
    context = tool_context()
    setup(monkeypatch)
    with pytest.raises(ReportError):
        write_report(title="t", criteria="c", items=None if stage == "validation" else [], tool_context=context)
    assert context.agent.state.get(tools.REPORT) is None
    assert context.agent.state.get(tools.REPORT_FAILURE) == {"stage": stage, "errors": [expected]}


def test_a_later_save_clears_the_failure_and_a_later_failure_keeps_the_report(report_dir):
    context = tool_context()
    with pytest.raises(ReportError):
        write_report(title="t", criteria="c", items=None, tool_context=context)
    write_report(title="t", criteria="c", items=[], tool_context=context)
    assert context.agent.state.get(tools.REPORT_FAILURE) is None

    saved = context.agent.state.get(tools.REPORT)
    with pytest.raises(ReportError):
        write_report(title="t", criteria="c", items=None, tool_context=context)
    assert context.agent.state.get(tools.REPORT) == saved


def test_an_empty_report_is_not_a_clean_one_if_a_source_failed(report_dir):
    context = tool_context()
    context.agent.state.set(tools.COVERAGE, {"r/python": READ, "r/rust": FAILED_READ})
    result = write_report(title="t", criteria="c", items=[], tool_context=context)

    assert "not everything was read: r/rust: could not be read (no route to host)." in result
    html = (report_dir / Path(result.split("Saved ")[1].split(".html")[0] + ".html").name).read_text()
    assert "Partial report" in html and "<li>r/rust: could not be read (no route to host).</li>" in html
    assert "Nothing matched in what could be read." in html and "Nothing matched the criteria." not in html
    assert "source_failed" in context.agent.state.get(tools.REPORT)["warnings"]


def test_an_empty_report_with_nothing_read_says_nothing_was_checked(report_dir):
    context = tool_context()
    context.agent.state.set(tools.COVERAGE, {"r/rust": FAILED_READ})
    write_report(title="t", criteria="c", items=[], tool_context=context)
    context.agent.state.delete(tools.COVERAGE)
    write_report(title="t", criteria="c", items=[], tool_context=context)

    pages = [path.read_text() for path in report_dir.iterdir()]
    assert len(pages) == 2 and all("No source could be read, so nothing was checked." in page for page in pages)
    assert "no_sources_read" in context.agent.state.get(tools.REPORT)["warnings"]


def test_a_source_cut_short_makes_a_report_partial_even_with_matches(report_dir):
    context = tool_context()
    context.agent.state.set(tools.COVERAGE, {"r/python": SHORT_READ})
    write_report(title="t", criteria="c", items=[{"link": POST, "description": "d"}], tool_context=context)
    [path] = report_dir.iterdir()
    assert "Partial report" in path.read_text()
    assert "source_incomplete" in context.agent.state.get(tools.REPORT)["warnings"]


def test_problems_with_the_items_appear_at_the_top_of_the_report(report_dir):
    context = tool_context()
    write_report(title="t", criteria="c", items=[{"link": POST, "description": "d", "quote": "not in the post"}], tool_context=context)
    [path] = report_dir.iterdir()
    assert "Partial report" in path.read_text()
    assert "<li>These quotes aren&#x27;t in their post or its comments: not in the post.</li>" in path.read_text()
