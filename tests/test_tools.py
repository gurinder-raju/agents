from datetime import datetime

import pytest

from agent_v3.tools import current_time, fetch_url, find_links, html_to_text


def test_current_time_returns_iso_timestamp_in_the_given_zone():
    stamp = current_time(timezone="Asia/Kolkata")
    assert datetime.fromisoformat(stamp).utcoffset().total_seconds() == 5.5 * 3600


def test_current_time_has_a_tool_spec_built_from_the_docstring():
    spec = current_time.tool_spec
    assert spec["name"] == "current_time"
    assert "timezone" in spec["inputSchema"]["json"]["properties"]


def test_html_to_text_keeps_visible_text_one_block_per_line():
    html = """<html><head><title>Release notes</title><style>p { color: red }</style></head>
    <body><h1>Version 2.1</h1><p>Faster   startup &amp; a new config file.</p><script>alert(1)</script><p>Upgrade with uv.</p></body></html>"""
    assert html_to_text(html) == "Release notes\nVersion 2.1\nFaster startup & a new config file.\nUpgrade with uv."


def test_fetch_url_refuses_this_machine():
    with pytest.raises(ValueError, match="non-public address"):
        fetch_url(url="http://localhost:11434/api/tags")


def test_fetch_url_refuses_other_schemes():
    with pytest.raises(ValueError, match="http"):
        fetch_url(url="file:///etc/passwd")


def test_find_links_returns_matching_links_in_page_order():
    html = """<a href="/downloads/release/python-3152/">Python 3.15.2</a>
    <a href="/downloads/release/python-3139/">Python 3.13.9</a> <a href="/ftp/python/3.13.9/">Download</a>
    <a href="/downloads/release/python-3138/"><span>Python</span> 3.13.8</a> <a href="javascript:void(0)">3.13</a>"""
    found = find_links(html, "https://www.python.org/downloads/", "python 3.13")
    assert found.splitlines() == [
        "Python 3.13.9: https://www.python.org/downloads/release/python-3139/",
        "Download: https://www.python.org/ftp/python/3.13.9/",
        "Python 3.13.8: https://www.python.org/downloads/release/python-3138/",
    ]
    assert "No links" in find_links(html, "https://www.python.org/downloads/", "3.99")
