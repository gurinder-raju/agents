import fcntl
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from agent_v3 import tools
from agent_v3.tools import parse_feed, read_feed

NOW = datetime.now(UTC)
RECENT = (NOW - timedelta(hours=2)).isoformat()
OLD = (NOW - timedelta(hours=30)).isoformat()

# Made-up posts shaped like Reddit's feeds: Atom, with the post body as HTML in <content>.
ATOM = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>newest submissions : learnpython</title>
  <entry>
    <author><name>/u/river_stone</name></author>
    <content type="html">&lt;div class="md"&gt;&lt;p&gt;Should I use venv &amp;amp; pip or uv?&lt;/p&gt;&lt;/div&gt;</content>
    <id>t3_1abc23</id>
    <link href="https://www.reddit.com/r/learnpython/comments/1abc23/question_about_virtual_environments/"/>
    <published>{RECENT}</published>
    <title>Question about virtual environments</title>
  </entry>
  <entry>
    <author><name>/u/quiet_maple</name></author>
    <content type="html">&lt;p&gt;Share what you are reading this week.&lt;/p&gt;</content>
    <id>t3_1abc01</id>
    <link href="https://www.reddit.com/r/learnpython/comments/1abc01/weekly_resources_thread/"/>
    <published>{OLD}</published>
    <title>Weekly resources thread</title>
  </entry>
</feed>"""

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Release notes</title>
  <item><title>Version 2.1</title><link>https://example.com/releases/2.1</link><guid>release-2-1</guid>
    <pubDate>Mon, 26 Oct 2026 16:52:54 GMT</pubDate><description>&lt;p&gt;Faster startup and a new config file.&lt;/p&gt;</description></item>
</channel></rss>"""


def test_parse_atom_feed():
    title, entries = parse_feed(ATOM)
    assert title == "newest submissions : learnpython"
    first = entries[0]
    assert first["id"] == "t3_1abc23"
    assert first["author"] == "/u/river_stone"
    assert first["link"] == "https://www.reddit.com/r/learnpython/comments/1abc23/question_about_virtual_environments/"
    assert first["text"] == "Should I use venv & pip or uv?"


def test_parse_rss_feed():
    title, [item] = parse_feed(RSS)
    assert title == "Release notes"
    assert item["id"] == "release-2-1"
    assert item["published"] == datetime(2026, 10, 26, 16, 52, 54, tzinfo=UTC)
    assert item["text"] == "Faster startup and a new config file."


def test_read_feed_keeps_only_recent_entries(monkeypatch):
    monkeypatch.setattr(tools, "http_get", lambda url: httpx.Response(200, text=ATOM))
    result = json.loads(read_feed(url="https://www.reddit.com/r/learnpython/new/.rss", since_hours=24))
    assert [e["id"] for e in result["entries"]] == ["t3_1abc23"]
    assert result["omitted"] == 0


def test_rate_limits_are_shared_through_a_file_per_host(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "RATE_LIMIT_DIR", tmp_path)
    exhausted = httpx.Response(200, headers={"x-ratelimit-remaining": "0.0", "x-ratelimit-reset": "30"})
    with tools._host_turn("www.reddit.com") as turn:
        tools._remember_rate_limit(exhausted, turn)
        # While one agent holds a host's turn, another can't take it.
        with open(tmp_path / "www.reddit.com") as other, pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with tools._host_turn("example.com") as turn:
        tools._remember_rate_limit(httpx.Response(200, headers={"x-ratelimit-remaining": "57"}), turn)

    # The next agent to take reddit.com's turn waits out the limit; example.com had requests left.
    slept = []
    monkeypatch.setattr(tools.time, "sleep", slept.append)
    with tools._host_turn("www.reddit.com") as turn:
        tools._wait_for("www.reddit.com", turn)
    with tools._host_turn("example.com") as turn:
        tools._wait_for("example.com", turn)
    assert len(slept) == 1 and 29 < slept[0] <= 31
