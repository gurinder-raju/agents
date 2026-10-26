import json
from datetime import UTC, datetime, timedelta

import httpx

from hello_agent import tools
from hello_agent.tools import parse_feed, read_feed

NOW = datetime.now(UTC)
RECENT = (NOW - timedelta(hours=2)).isoformat()
OLD = (NOW - timedelta(hours=30)).isoformat()

# Shaped like Reddit's feeds: Atom, the post body as HTML in <content>.
ATOM = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>newest submissions : example</title>
  <entry>
    <author><name>/u/alice</name></author>
    <content type="html">&lt;div class="md"&gt;&lt;p&gt;Recent &amp;amp; relevant&lt;/p&gt;&lt;/div&gt;</content>
    <id>t3_new</id>
    <link href="https://www.reddit.com/r/example/comments/new/a_recent_post/"/>
    <published>{RECENT}</published>
    <title>A recent post</title>
  </entry>
  <entry>
    <author><name>/u/bob</name></author>
    <content type="html">&lt;p&gt;Old&lt;/p&gt;</content>
    <id>t3_old</id>
    <link href="https://www.reddit.com/r/example/comments/old/an_old_post/"/>
    <published>{OLD}</published>
    <title>An old post</title>
  </entry>
</feed>"""

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Example blog</title>
  <item><title>Hello</title><link>https://example.com/hello</link><guid>hello-1</guid>
    <pubDate>Mon, 26 Oct 2026 16:52:54 GMT</pubDate><description>&lt;p&gt;Hi there&lt;/p&gt;</description></item>
</channel></rss>"""


def test_parse_atom_feed():
    title, entries = parse_feed(ATOM)
    assert title == "newest submissions : example"
    first = entries[0]
    assert first["id"] == "t3_new"
    assert first["author"] == "/u/alice"
    assert first["link"] == "https://www.reddit.com/r/example/comments/new/a_recent_post/"
    assert first["text"] == "Recent & relevant"


def test_parse_rss_feed():
    title, [item] = parse_feed(RSS)
    assert title == "Example blog"
    assert item["id"] == "hello-1"
    assert item["published"] == datetime(2026, 10, 26, 16, 52, 54, tzinfo=UTC)
    assert item["text"] == "Hi there"


def test_read_feed_keeps_only_recent_entries(monkeypatch):
    monkeypatch.setattr(tools, "http_get", lambda url: httpx.Response(200, text=ATOM))
    result = json.loads(read_feed(url="https://www.reddit.com/r/example/new/.rss", since_hours=24))
    assert [e["id"] for e in result["entries"]] == ["t3_new"]
    assert result["omitted"] == 0


def test_rate_limit_headers_delay_the_next_request(monkeypatch):
    monkeypatch.setattr(tools, "_next_request_at", {})
    exhausted = httpx.Response(200, headers={"x-ratelimit-remaining": "0.0", "x-ratelimit-reset": "30"})
    tools._remember_rate_limit("www.reddit.com", exhausted)
    wait = tools._next_request_at["www.reddit.com"] - tools.time.monotonic()
    assert 29 < wait <= 31

    tools._remember_rate_limit("example.com", httpx.Response(200, headers={"x-ratelimit-remaining": "57"}))
    assert "example.com" not in tools._next_request_at
