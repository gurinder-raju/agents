import json

import httpx
import pytest

from agent_v3 import tools
from agent_v3.tools import read_subreddit
from test_feeds import ATOM

POST_LINK = "https://www.reddit.com/r/learnpython/comments/1abc23/question_about_virtual_environments/"

# A post's comment feed: the post itself first, then one entry per comment.
THREAD = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Question about virtual environments : learnpython</title>
  <entry><id>t3_1abc23</id><author><name>/u/river_stone</name></author><title>Question about virtual environments</title>
    <content type="html">&lt;p&gt;Should I use venv &amp;amp; pip or uv?&lt;/p&gt;</content></entry>
  <entry><id>t1_c1</id><author><name>/u/pine_needle</name></author><title>/u/pine_needle on Question</title>
    <content type="html">&lt;p&gt;uv creates the venv for you.&lt;/p&gt;</content></entry>
  <entry><id>t1_c2</id><author><name>/u/slow_tide</name></author><title>/u/slow_tide on Question</title>
    <content type="html">&lt;p&gt;venv and pip still work fine.&lt;/p&gt;</content></entry>
</feed>"""


@pytest.fixture
def fake_reddit(monkeypatch):
    requested = []

    def http_get(url):
        requested.append(url)
        if url == "https://www.reddit.com/r/learnpython/new/.rss?limit=100":
            return httpx.Response(200, text=ATOM)
        if url == POST_LINK + ".rss":
            return httpx.Response(200, text=THREAD)
        raise AssertionError(f"unexpected request: {url}")

    monkeypatch.setattr(tools, "http_get", http_get)
    return requested


def test_reads_recent_posts_and_their_comments(fake_reddit):
    result = json.loads(read_subreddit(subreddit="r/learnpython"))

    [post] = result["posts"]  # the 30-hour-old post is outside the 24-hour window
    assert post["title"] == "Question about virtual environments"
    assert post["comments"] == [
        {"author": "/u/pine_needle", "text": "uv creates the venv for you."},
        {"author": "/u/slow_tide", "text": "venv and pip still work fine."},
    ]
    assert result["omitted"] == {"posts": 0, "comments": 0}
    assert len(fake_reddit) == 2  # the listing, then one comment feed


def test_trims_comments_to_fit_the_budget(fake_reddit, monkeypatch):
    full_size = len(read_subreddit(subreddit="learnpython"))
    monkeypatch.setattr(tools, "MIN_COMMENTS_PER_POST", 1)
    monkeypatch.setattr(tools, "MAX_SUBREDDIT_CHARS", full_size - 1)
    result = json.loads(read_subreddit(subreddit="learnpython"))

    assert [c["author"] for c in result["posts"][0]["comments"]] == ["/u/pine_needle"]
    assert result["omitted"] == {"posts": 0, "comments": 1}


def test_rejects_names_that_are_not_subreddits(fake_reddit):
    with pytest.raises(ValueError, match="not a valid subreddit name"):
        read_subreddit(subreddit="learnpython/../../api")
    assert fake_reddit == []
