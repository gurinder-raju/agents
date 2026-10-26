from types import SimpleNamespace

import httpx
import pytest
from strands.agent.state import AgentState

from fetcher import tools
from fetcher.report import canonical_link, normalize_items
from fetcher.tools import write_report

POST = "https://www.reddit.com/r/Python/comments/1abc23/switched_to_uv/"
QUIET = "https://www.reddit.com/r/Python/comments/9zz/quiet_post/"
ARTICLE = "https://example.com/uv-in-production"

# What read_subreddit remembered about the posts it returned.
SOURCES = {
    canonical_link(POST): {"title": "Switched to uv", "comment_count": 14, "text": "Switched to uv\nWe moved to uv 0.9 last week."},
    canonical_link(QUIET): {"title": "A quiet post", "comment_count": 2, "text": "A quiet post\nPython 3.14 is out."},
}


def tool_context(prompt="Find posts in r/Python about uv", run_id=None, tool_output=f"{POST} {QUIET} {ARTICLE}"):
    """The parts of Strands' ToolContext that write_report reads: the conversation, the agent's state and the run id."""
    messages = [
        {"role": "user", "content": [{"text": prompt}]},
        {"role": "assistant", "content": [{"text": "Reading r/Python."}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": tool_output}]}}]},
    ]
    agent = SimpleNamespace(messages=messages, state=AgentState({tools.SOURCES: SOURCES}))
    return SimpleNamespace(agent=agent, invocation_state={"run_id": run_id} if run_id else {})


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
        {"link": QUIET, "description": "Mentions 3.14.", "title": "A title the model made up"},
        {"link": POST.replace("https://www.", "https://"), "description": "Moved to uv."},
    ]
    html = open(write_report(title="t", criteria="c", items=items, sort="comments", tool_context=tool_context())).read()

    assert "A title the model made up" not in html
    assert html.index("Switched to uv") < html.index("A quiet post")  # 14 comments before 2
    assert "most commented first" in html


def test_says_when_a_quote_is_not_in_the_post(report_dir):
    result = write_report(
        title="t", criteria="c", items=[{"link": POST, "quote": "Python 3.13 benchmarks"}], tool_context=tool_context()
    )
    assert "Python 3.13 benchmarks" in result and "call write_report again" in result
    [path] = report_dir.iterdir()
    assert "(not found in the post)" in path.read_text()


def test_flags_links_no_tool_returned_unless_they_fill_in_a_pattern_from_the_prompt(report_dir):
    made_up = "https://www.python.org/downloads/release/python-3130a1/"
    search = "https://example.org/search?q=river_stone"
    prompt = "Look things up on https://www.python.org/downloads/ and https://example.org/search?q=<name>"
    result = write_report(
        title="t",
        criteria="c",
        items=[{"link": POST, "extra_link": made_up}, {"link": QUIET, "extra_link": search}],
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
    result = write_report(title="t", criteria="c", items=[{"link": POST, "extra_link": ARTICLE}], tool_context=tool_context())
    assert f"these links don't exist: {ARTICLE}" in result
    [path] = report_dir.iterdir()
    assert "(not found)" in path.read_text()


def test_escapes_text_from_posts(report_dir):
    items = [{"link": ARTICLE, "title": "<img src=x onerror=alert(1)>", "description": "a & b"}]
    html = open(write_report(title="<script>alert(1)</script>", criteria="c", items=items, tool_context=tool_context())).read()
    assert "<script>alert" not in html and "<img" not in html
    assert "a &amp; b" in html


def test_refuses_links_that_are_not_http(report_dir):
    with pytest.raises(ValueError, match="item 1"):
        write_report(title="t", criteria="c", items=[{"link": "javascript:alert(1)"}], tool_context=tool_context())


def test_splits_a_link_field_that_holds_two_urls():
    [item] = normalize_items([{"link": POST, "extra_link": "https://example.com/a/ and https://example.com/b/.", "extra_label": "download"}])
    assert [(link["label"], link["url"]) for link in item["links"]] == [
        ("post", POST),
        ("download", "https://example.com/a/"),
        ("download", "https://example.com/b/"),
    ]


def test_accepts_a_single_url_or_bare_link_strings():
    items = normalize_items([{"title": "A", "url": POST}, {"title": "B", "links": [POST, ARTICLE]}])
    assert items[0]["links"] == [{"label": "www.reddit.com", "url": POST}]
    assert [link["label"] for link in items[1]["links"]] == ["www.reddit.com", "example.com"]


def test_tries_a_less_specific_version_before_the_pages_above():
    def fake_web(request):
        return httpx.Response(200 if request.url.path in ("/latest/python3.12/", "/latest/") else 404)

    items = normalize_items([{"link": "https://example.com/latest/python3.12.4/"}])
    tools._find_missing_links(items, httpx.Client(transport=httpx.MockTransport(fake_web)))
    assert items[0]["links"][0]["url"] == "https://example.com/latest/python3.12/"


def test_checks_links_and_falls_back_to_the_closest_page_that_exists():
    pages = {"/ok", "/files/"}

    def fake_web(request):
        return httpx.Response(200 if request.url.path in pages else 404)

    links = ["https://example.com/ok", "https://example.com/missing", "https://example.com/files/2.1/", POST]
    items = normalize_items([{"title": "A", "links": links}])
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
        items=[{"link": POST, "fill": "3.15", "extra_label": "download"}, {"link": QUIET, "fill": "free threading"}],
        tool_context=tool_context(prompt),
    )
    assert "But" not in result and "but" not in result  # built from the user's pattern, so not unverified
    html = open(result).read()
    assert '<a href="https://example.org/latest/python3.15/" rel="noopener noreferrer">download</a>' in html
    assert 'href="https://example.org/latest/pythonfree%20threading/"' in html


def test_ignores_fill_when_the_prompt_has_no_single_pattern(report_dir):
    prompt = "Link https://a.example/<x> or https://b.example/<y>."
    html = open(write_report(title="t", criteria="c", items=[{"link": POST, "fill": "v"}], tool_context=tool_context(prompt))).read()
    assert 'href="https://a.example' not in html and 'href="https://b.example' not in html


def test_says_what_the_tools_read(report_dir):
    context = tool_context()
    context.agent.state.set(tools.COVERAGE, ["r/Python: 26 posts and 812 comments from the last 168 hours."])
    html = open(write_report(title="t", criteria="c", items=[], tool_context=context)).read()
    assert "<li>r/Python: 26 posts and 812 comments from the last 168 hours.</li>" in html
