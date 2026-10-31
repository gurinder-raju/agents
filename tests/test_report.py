from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from strands.agent.state import AgentState

from fetcher import tools
from fetcher.report import LINK_ITEM, SOURCE_ITEM, ReportError, canonical_link, normalize_items, parse_report, report_schema, url_problem
from fetcher.tools import item_contract, link_patterns, report_tool, write_report

POST = "https://www.reddit.com/r/Python/comments/1abc23/switched_to_uv/"
QUIET = "https://www.reddit.com/r/Python/comments/9zz/quiet_post/"
ARTICLE = "https://example.com/uv-in-production"

# What read_subreddit remembered about the posts it returned.
SOURCES = {
    canonical_link(POST): {"title": "Switched to uv", "comment_count": 14, "text": "Switched to uv\nWe moved to uv 0.9 last week."},
    canonical_link(QUIET): {"title": "A quiet post", "comment_count": 2, "text": "A quiet post\nPython 3.14 is out."},
}


def tool_context(prompt="Find posts in r/Python about uv", run_id=None, tool_output=f"{POST} {QUIET} {ARTICLE}", tool_input=None):
    """The parts of Strands' ToolContext that write_report reads: the conversation, the agent's state, the
    run id and the call as the model made it."""
    messages = [
        {"role": "user", "content": [{"text": prompt}]},
        {"role": "assistant", "content": [{"text": "Reading r/Python."}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": tool_output}]}}]},
    ]
    agent = SimpleNamespace(messages=messages, state=AgentState({tools.SOURCES: SOURCES}))
    return SimpleNamespace(agent=agent, invocation_state={"run_id": run_id} if run_id else {}, tool_use={"input": tool_input or {}})


@pytest.fixture
def report_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(tools, "_find_missing_links", lambda items: [])  # no network in these tests
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
    assert "Nothing matched the criteria." in open(path).read()


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
    assert path.read_text().count("(unverified)") == 1


def test_says_when_a_link_does_not_exist(report_dir, monkeypatch):
    def mark_extra_link_missing(items):
        items[0]["links"][1]["missing"] = True
        return [ARTICLE]

    monkeypatch.setattr(tools, "_find_missing_links", mark_extra_link_missing)
    result = write_report(title="t", criteria="c", items=[{"link": POST, "description": "d", "extra_link": ARTICLE}], tool_context=tool_context())
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
        "version: unknown argument, expected only title, criteria, items, sort",
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
    items, adapted = parse_report("t", "c", [
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
    assert parse_report("t", "c", [item], "as_given", SOURCE_ITEM) == ([{"post": POST, "fill": "uv", "description": "d", "quote": "q"}], 1)


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
        names = str(report_schema(contract)["json"]["properties"].keys())
        assert names == "dict_keys(['title', 'criteria', 'items', 'sort'])"


def test_a_pattern_at_the_end_of_a_sentence_keeps_its_own_punctuation():
    assert link_patterns("See https://example.org/<name>_(band), or https://example.net/?q={x}.") == [
        "https://example.org/<name>_(band)",
        "https://example.net/?q={x}",
    ]


def test_tries_a_less_specific_version_before_the_pages_above():
    def fake_web(request):
        return httpx.Response(200 if request.url.path in ("/latest/python3.12/", "/latest/") else 404)

    items = normalize_items([{"link": "https://example.com/latest/python3.12.4/", "description": "d"}])
    tools._find_missing_links(items, httpx.Client(transport=httpx.MockTransport(fake_web)))
    assert items[0]["links"][0]["url"] == "https://example.com/latest/python3.12/"


def test_checks_links_and_falls_back_to_the_closest_page_that_exists():
    pages = {"/ok", "/files/"}

    def fake_web(request):
        return httpx.Response(200 if request.url.path in pages else 404)

    links = ["https://example.com/ok", "https://example.com/missing", "https://example.com/files/2.1/", POST]
    items = [{"links": [{"label": "l", "url": url} for url in links]}]
    missing = tools._find_missing_links(items, httpx.Client(transport=httpx.MockTransport(fake_web)))

    assert missing == ["https://example.com/missing"]  # nothing above it but the bare site
    ok, gone, replaced, reddit = items[0]["links"]
    assert ok["missing"] is False and gone["missing"] is True
    assert replaced["url"] == "https://example.com/files/" and replaced["closest_to"] == "https://example.com/files/2.1/"
    assert "missing" not in reddit  # Reddit links aren't checked


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
    assert '<a href="https://example.org/latest/python3.14/" rel="noopener noreferrer">example.org</a>' in html
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
    assert "A quiet post" in html and "Switched to uv" not in html
    assert "1 match" in html


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
    assert "(unverified)" not in html


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


def test_merges_items_that_repeat_a_post(report_dir):
    prompt = "Link each post to https://example.org/search?q=<name>."
    items = [
        {"post": QUIET, "fill": "Python", "description": "d", "quote": "Python 3.14 is out"},
        {"post": QUIET, "fill": "quiet", "description": "d", "quote": "A quiet post"},
        {"post": QUIET, "fill": "Python", "description": "d", "quote": "Python 3.14 is out"},
    ]
    result = write_report(title="t", criteria="c", items=items, tool_context=tool_context(prompt))
    assert "Merged 2 items" in result
    html = open(result.split("Saved ")[1].split(".html")[0] + ".html").read()
    assert html.count("<li>") == 1
    assert html.count('href="https://example.org/search?q=') == 2  # q=Python and q=quiet, each once


def test_with_several_patterns_the_model_gives_each_second_link(report_dir):
    prompt = "Link https://a.example/<x> or https://b.example/<y>."
    items = [{"link": POST, "description": "d", "fill": "v"}]
    with pytest.raises(ReportError, match=r"items\[0\]\.fill: unknown field"):
        write_report(title="t", criteria="c", items=items, tool_context=tool_context(prompt))

    items = [{"link": POST, "description": "d", "extra_link": "https://a.example/v"}]
    write_report(title="t", criteria="c", items=items, tool_context=tool_context(prompt))
    [path] = report_dir.iterdir()
    assert 'href="https://a.example/v"' in path.read_text()


def test_says_what_the_tools_read(report_dir):
    context = tool_context()
    context.agent.state.set(tools.COVERAGE, ["r/Python: 26 posts and 812 comments from the last 168 hours."])
    html = open(write_report(title="t", criteria="c", items=[], tool_context=context)).read()
    assert "<li>r/Python: 26 posts and 812 comments from the last 168 hours.</li>" in html
