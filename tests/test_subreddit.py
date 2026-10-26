import json
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest
from strands.agent.state import AgentState

from agent_v3 import tools
from agent_v3.tools import read_subreddit
from test_feeds import ATOM, NOW, OLD, RECENT

POST_LINK = "https://www.reddit.com/r/learnpython/comments/1abc23/question_about_virtual_environments/"
LISTING_URL = "https://www.reddit.com/r/learnpython/new/.rss?limit=100"
COMMENTS_URL = "https://www.reddit.com/r/learnpython/comments/.rss?limit=100"


def comment(comment_id, post_id, author, text, hours_ago):
    published = (NOW - timedelta(hours=hours_ago)).isoformat()
    return f"""<entry><id>t1_{comment_id}</id><author><name>{author}</name></author>
    <link href="https://www.reddit.com/r/learnpython/comments/{post_id}/slug/{comment_id}/"/>
    <published>{published}</published><title>{author} on a post</title>
    <content type="html">&lt;p&gt;{text}&lt;/p&gt;</content></entry>"""


def comment_feed(*entries):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>newest comments : learnpython</title>{"".join(entries)}</feed>"""


# The subreddit's comment feed, newest first: two comments on the recent post, one on another post,
# then one older than the 24-hour window, which tells read_subreddit it can stop.
COMMENTS = comment_feed(
    comment("c2", "1abc23", "/u/slow_tide", "venv and pip still work fine.", 1),
    comment("c9", "1zzz99", "/u/far_hill", "On another post.", 1.5),
    comment("c1", "1abc23", "/u/pine_needle", "uv creates the venv for you.", 1.8),
    comment("c0", "1abc01", "/u/old_oak", "On the weekly thread.", 29),
)


class Requests(list):
    """The URLs requested, in order, plus the pages the fake Reddit serves, which a test can change."""


@pytest.fixture
def fake_reddit(monkeypatch):
    requested = Requests()
    pages = {LISTING_URL: ATOM, COMMENTS_URL: COMMENTS}

    def http_get(url):
        requested.append(url)
        if url in pages:
            return httpx.Response(200, text=pages[url])
        raise AssertionError(f"unexpected request: {url}")

    monkeypatch.setattr(tools, "http_get", http_get)
    requested.pages = pages
    return requested


def test_reads_recent_posts_and_their_comments(fake_reddit):
    result = json.loads(read_subreddit(subreddit="r/learnpython"))

    [post] = result["posts"]  # the 30-hour-old post is outside the 24-hour window
    assert post["title"] == "Question about virtual environments"
    assert post["comment_count"] == 2
    assert post["comments"] == [
        {"author": "/u/pine_needle", "text": "uv creates the venv for you."},
        {"author": "/u/slow_tide", "text": "venv and pip still work fine."},
    ]
    assert result["omitted"] == {"posts": 0, "comments": 0}
    assert result["comments_complete"] is True
    assert fake_reddit == [LISTING_URL, COMMENTS_URL]  # two requests, however many posts there are


def test_pages_through_the_comment_feed_until_the_cutoff(fake_reddit):
    full_page = comment_feed(*(comment(f"p{n}", "1abc23", "/u/busy", f"Comment {n}.", 0.5) for n in range(100)))
    fake_reddit.pages[COMMENTS_URL] = full_page
    fake_reddit.pages[COMMENTS_URL + "&after=t1_p99"] = COMMENTS
    result = json.loads(read_subreddit(subreddit="learnpython"))

    assert result["posts"][0]["comment_count"] == 102
    assert fake_reddit[-1] == COMMENTS_URL + "&after=t1_p99"


def test_says_when_the_page_limit_stopped_it_early(fake_reddit, monkeypatch):
    monkeypatch.setattr(tools, "MAX_COMMENT_PAGES", 1)
    fake_reddit.pages[COMMENTS_URL] = comment_feed(*(comment(f"p{n}", "1abc23", "/u/busy", "Hi.", 0.5) for n in range(100)))
    result = json.loads(read_subreddit(subreddit="learnpython"))

    assert result["comments_complete"] is False
    assert result["posts"][0]["comment_count"] == 100


def test_sorts_by_comment_count(fake_reddit):
    # Make the weekly thread recent and put it first, as the newest post with no comments in the window.
    head, first, second = ATOM.replace(OLD, RECENT).split("<entry>")
    fake_reddit.pages[LISTING_URL] = head + "<entry>" + second.replace("</feed>", "") + "<entry>" + first + "</feed>"
    newest_first = json.loads(read_subreddit(subreddit="learnpython"))["posts"]
    most_commented_first = json.loads(read_subreddit(subreddit="learnpython", sort="comments"))["posts"]

    assert [post["comment_count"] for post in newest_first] == [0, 2]
    assert [post["comment_count"] for post in most_commented_first] == [2, 0]


def test_finds_the_page_a_link_post_points_to(fake_reddit):
    link_post = ATOM.replace(
        "&lt;p&gt;Should I use venv &amp;amp; pip or uv?&lt;/p&gt;",
        "&lt;span&gt;&lt;a href=&quot;https://example.com/uv?a=1&amp;amp;b=2&quot;&gt;[link]&lt;/a&gt;&lt;/span&gt;",
    )
    fake_reddit.pages[LISTING_URL] = link_post
    [post] = json.loads(read_subreddit(subreddit="learnpython"))["posts"]

    assert post["url"] == "https://example.com/uv?a=1&b=2"


def test_trims_comments_to_fit_the_budget(fake_reddit, monkeypatch):
    full_size = len(read_subreddit(subreddit="learnpython"))
    monkeypatch.setattr(tools, "MIN_COMMENTS_PER_POST", 1)
    monkeypatch.setattr(tools, "MAX_SUBREDDIT_CHARS", full_size - 1)
    result = json.loads(read_subreddit(subreddit="learnpython"))

    assert [c["author"] for c in result["posts"][0]["comments"]] == ["/u/pine_needle"]
    assert result["posts"][0]["comment_count"] == 2
    assert result["omitted"] == {"posts": 0, "comments": 1}


def test_rejects_names_that_are_not_subreddits(fake_reddit):
    with pytest.raises(ValueError, match="not a valid subreddit name"):
        read_subreddit(subreddit="learnpython/../../api")
    assert fake_reddit == []


def test_remembers_posts_for_write_report(fake_reddit):
    fake_reddit.pages[LISTING_URL] = ATOM.replace(
        "&lt;p&gt;Should I use venv &amp;amp; pip or uv?&lt;/p&gt;",
        "&lt;p&gt;Should I use venv or uv?&lt;/p&gt; submitted by &lt;a href=&quot;https://www.reddit.com/user/river_stone&quot;&gt;"
        " /u/river_stone &lt;/a&gt; &lt;br/&gt; &lt;span&gt;&lt;a href=&quot;" + POST_LINK + "&quot;&gt;[link]&lt;/a&gt;&lt;/span&gt;"
        " &lt;span&gt;&lt;a href=&quot;" + POST_LINK + "&quot;&gt;[comments]&lt;/a&gt;&lt;/span&gt;",
    )
    agent = SimpleNamespace(state=AgentState())
    [post] = json.loads(read_subreddit(subreddit="learnpython", agent=agent))["posts"]

    assert post["text"] == "Should I use venv or uv?"  # without Reddit's "submitted by ... [link] [comments]"
    assert post["url"] is None  # a text post's [link] is the post itself
    [source] = agent.state.get(tools.SOURCES).values()
    assert source["title"] == "Question about virtual environments"
    assert source["comment_count"] == 2
    assert "uv creates the venv for you." in source["text"]


def test_says_what_it_read_and_keeps_real_characters(fake_reddit):
    fake_reddit.pages[LISTING_URL] = ATOM.replace("Question about virtual environments", "Question about venvs 🐍")
    agent = SimpleNamespace(state=AgentState())
    result = read_subreddit(subreddit="learnpython", agent=agent)

    assert "🐍" in result  # not \\ud83d\\udc0d, which the model would copy into its quotes
    assert agent.state.get(tools.COVERAGE) == ["r/learnpython: 1 posts and 2 comments from the last 24 hours."]
