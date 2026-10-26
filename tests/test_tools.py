from datetime import datetime

import pytest

from hello_agent.tools import current_time, fetch_url, html_to_text


def test_current_time_returns_iso_timestamp_in_the_given_zone():
    stamp = current_time(timezone="Asia/Kolkata")
    assert datetime.fromisoformat(stamp).utcoffset().total_seconds() == 5.5 * 3600


def test_current_time_has_a_tool_spec_built_from_the_docstring():
    spec = current_time.tool_spec
    assert spec["name"] == "current_time"
    assert "timezone" in spec["inputSchema"]["json"]["properties"]


def test_html_to_text_keeps_visible_text_one_block_per_line():
    html = """<html><head><title>Example</title><style>p { color: red }</style></head>
    <body><h1>Hello</h1><p>First   paragraph &amp; more.</p><script>alert(1)</script><p>Second</p></body></html>"""
    assert html_to_text(html) == "Example\nHello\nFirst paragraph & more.\nSecond"


def test_fetch_url_refuses_this_machine():
    with pytest.raises(ValueError, match="non-public address"):
        fetch_url(url="http://localhost:11434/api/tags")


def test_fetch_url_refuses_other_schemes():
    with pytest.raises(ValueError, match="http"):
        fetch_url(url="file:///etc/passwd")
