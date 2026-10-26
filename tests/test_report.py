from types import SimpleNamespace

import pytest

from agent_v3 import tools
from agent_v3.report import normalize_items
from agent_v3.tools import write_report

POST = "https://www.reddit.com/r/Python/comments/1abc23/switched_to_uv/"
ARTICLE = "https://example.com/uv-in-production"


def tool_context(prompt, run_id=None):
    """The parts of Strands' ToolContext that write_report reads."""
    messages = [
        {"role": "user", "content": [{"text": prompt}]},
        {"role": "assistant", "content": [{"text": "Reading r/Python."}]},
    ]
    return SimpleNamespace(agent=SimpleNamespace(messages=messages), invocation_state={"run_id": run_id} if run_id else {})


@pytest.fixture
def report_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "REPORT_DIR", tmp_path)
    return tmp_path


def test_writes_the_prompt_criteria_and_links(report_dir):
    path = write_report(
        title="r/Python posts about uv",
        criteria="Posts where someone describes moving a project to uv.",
        items=[{
            "title": "Switched to uv",
            "description": "The author moved three services from pip to uv.",
            "links": [{"label": "post", "url": POST}, {"label": "article", "url": ARTICLE}],
        }],
        tool_context=tool_context("Find posts in r/Python about uv", run_id="20261026-131500-a1b2c3"),
    )
    assert path == str(report_dir / "20261026-131500-a1b2c3-r-python-posts-about-uv.html")
    html = open(path).read()
    assert "Find posts in r/Python about uv" in html
    assert "Posts where someone describes moving a project to uv." in html
    assert f'<a href="{POST}" rel="noopener noreferrer">post</a>' in html
    assert f'<a href="{ARTICLE}" rel="noopener noreferrer">article</a>' in html
    assert "1 match" in html


def test_an_empty_report_says_nothing_matched(report_dir):
    path = write_report(title="Nothing here", criteria="Posts about uv.", items=[], tool_context=tool_context("Find uv posts"))
    assert "Nothing matched the criteria." in open(path).read()


def test_reports_from_runs_without_a_run_id_get_different_files(report_dir):
    first = write_report(title="Same title", criteria="c", items=[], tool_context=tool_context("p"))
    second = write_report(title="Same title", criteria="c", items=[], tool_context=tool_context("p"))
    assert first != second


def test_escapes_text_from_posts(report_dir):
    path = write_report(
        title="<script>alert(1)</script>",
        criteria="c",
        items=[{"title": "<img src=x onerror=alert(1)>", "description": "a & b", "url": POST}],
        tool_context=tool_context("p"),
    )
    html = open(path).read()
    assert "<script>alert" not in html and "<img" not in html
    assert "a &amp; b" in html


def test_accepts_a_single_url_or_bare_link_strings():
    items = normalize_items([{"title": "A", "url": POST}, {"title": "B", "links": [POST, ARTICLE]}])
    assert items[0]["links"] == [{"label": "www.reddit.com", "url": POST}]
    assert [link["label"] for link in items[1]["links"]] == ["www.reddit.com", "example.com"]


def test_refuses_links_that_are_not_http():
    with pytest.raises(ValueError, match="item 1"):
        normalize_items([{"title": "A", "url": "javascript:alert(1)"}])


def test_refuses_items_without_links():
    with pytest.raises(ValueError, match="no links"):
        normalize_items([{"title": "A", "description": "no link"}])
