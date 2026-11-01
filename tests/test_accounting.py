import pytest

from fetcher import tools
from fetcher.accounting import source_key
from fetcher.report import ReportError, canonical_link, contains_words
from fetcher.tools import requested_sources, write_report
from test_report import report_dir, tool_context  # noqa: F401  (report_dir is a fixture)

PATTERN = "Go through r/Python. Link each post to https://example.org/search?q=<name>."
LINKS = "Find posts in r/Python about uv."
READ = {"source": "r/Python", "status": "complete", "line": "r/Python: read in full."}


def link_of(number):
    return f"https://www.reddit.com/r/Python/comments/{number}/post_{number}/"


def post(number, *groups, shown=True):
    """What read_subreddit remembers about a post. Each group is what a local tool marked in it."""
    words = " ".join(group["text"] for group in groups)
    return {
        "id": f"Python-{number}",
        "title": f"Post {number}",
        "comment_count": number,
        "text": f"Post {number}\nsome words {words}",
        "read_from": "r/Python",
        "shown": shown,
        "groups": list(groups),
        "link": link_of(number),
    }


def group(text, shown=True):
    return {"text": text, "lines": text.split("\n"), "field": "text", "start": 0, "end": len(text), "shown": shown}


def context_for(prompt, posts, reads=(READ,), **settings):
    context = tool_context(prompt, tool_output=" ".join(p["link"] for p in posts), **settings)
    state = context.agent.state
    state.set(tools.SOURCES, {canonical_link(p["link"]): p for p in posts})
    state.set(tools.POST_IDS, {p["id"]: p["link"] for p in posts})
    state.set(tools.COVERAGE, {read["source"].lower(): read for read in reads})
    return context


def report_of(prompt, posts, items=(), left_out=(), reads=(READ,)):
    context = context_for(prompt, posts, reads)
    reply = write_report(title="t", criteria="c", items=list(items), left_out=list(left_out), tool_context=context)
    return context.agent.state.get(tools.REPORT), reply


def item(number, fill="value", quote="some words"):
    return {"post": f"Python-{number}", "fill": fill, "description": "d", "quote": quote}


def adds_up(tally):
    return (
        tally["found"] == tally["truncated"] + tally["assessed"] + tally["unaccounted"]
        and tally["assessed"] == tally["rendered"] + tally["rejected"]
    )


def test_everything_found_ends_up_as_exactly_one_thing(report_dir):
    posts = [post(1), post(2), post(3), post(4, shown=False), post(5)]
    items = [item(1, fill="some"), {"post": "Python-2", "fill": "", "description": "No name here.", "quote": "some words"}]
    report, _ = report_of(PATTERN, posts, items, [{"source": "Python-3", "reason": "A repost."}])

    [row] = report["sources"]
    assert row["posts"] == {"found": 5, "truncated": 1, "assessed": 3, "submitted": 2, "rejected": 2, "rendered": 1, "unaccounted": 1}
    assert adds_up(row["posts"])
    assert row["unaccounted"]["posts"] == ["Python-5"]
    assert report["warnings"] == ["unaccounted_candidates", "no_value", "links_not_checked"]


def test_a_row_for_a_source_covers_the_rest_of_its_posts(report_dir):
    posts = [post(1), post(2), post(3)]
    report, reply = report_of(PATTERN, posts, [item(1, fill="some")], [{"source": "r/Python", "reason": "The others have no name."}])

    [row] = report["sources"]
    assert (row["posts"]["rendered"], row["posts"]["rejected"], row["posts"]["unaccounted"]) == (1, 2, 0)
    assert report["unexplained"] == [] and "but" not in reply


@pytest.mark.parametrize("name", ["r/Python", "Python", "python", "/r/python/"])
def test_a_source_can_be_named_however_the_model_writes_it(report_dir, name):
    report, _ = report_of(PATTERN, [post(1)], left_out=[{"source": name, "reason": "No name."}])
    assert report["unexplained"] == []


def test_what_the_model_never_got_needs_no_reason_and_counts_apart(report_dir):
    short = {"source": "r/Python", "status": "incomplete", "line": "r/Python: 1 posts didn't fit."}
    report, _ = report_of(PATTERN, [post(1), post(2, shown=False)], left_out=[{"source": "r/Python", "reason": "No name."}], reads=[short])

    [row] = report["sources"]
    assert (row["posts"]["rejected"], row["posts"]["truncated"], row["posts"]["unaccounted"]) == (1, 1, 0)
    assert "unaccounted_candidates" not in report["warnings"]  # that's input truncation, reported on its own:
    assert "source_incomplete" in report["warnings"]


def test_a_post_with_no_row_is_flagged_for_the_model_and_at_the_top_of_the_page(report_dir):
    report, reply = report_of(PATTERN, [post(1), post(2)], [item(1, fill="some")])

    assert report["unexplained"] == ["r/Python: Python-2"]
    assert "no item or left_out row covers these posts: r/Python: Python-2" in reply and "call write_report again" in reply
    html = open(report["path"]).read()
    assert "Partial report" in html and "No item or left_out row covers these posts: r/Python: Python-2." in html


def test_a_long_list_of_posts_is_cut_short_but_counted_in_full(report_dir):
    report, _ = report_of(PATTERN, [post(number) for number in range(1, 21)])
    [row] = report["sources"]
    assert row["posts"]["unaccounted"] == len(row["unaccounted"]["posts"]) == 20
    assert report["unexplained"][0].endswith("and 8 more")


def test_a_source_with_no_report_rows_needs_a_reason(report_dir):
    report, _ = report_of(PATTERN, [post(1)])
    assert report["warnings"][0] == "unaccounted_candidates" and report["sources"][0]["reason"] is None

    report, _ = report_of(PATTERN, [post(1)], left_out=[{"source": "r/Python", "reason": "Nothing had a name."}])
    assert report["unexplained"] == [] and report["sources"][0]["reason"] == "Nothing had a name."


def test_a_source_with_nothing_in_it_explains_itself(report_dir):
    empty = {"source": "r/Python", "status": "complete", "line": "r/Python: 0 posts and 0 comments from the last 24 hours."}
    report, _ = report_of(PATTERN, [], reads=[empty])
    assert report["unexplained"] == [] and report["sources"][0]["reason"] == empty["line"]


def test_a_failed_read_needs_no_row_and_stays_a_failed_read(report_dir):
    failed = {"source": "r/Python", "status": "failed", "line": "r/Python: could not be read (timeout)."}
    report, _ = report_of(PATTERN, [], reads=[failed])
    assert report["unexplained"] == [] and "source_failed" in report["warnings"]


def test_a_requested_source_nobody_read_is_flagged_until_a_row_says_why(report_dir):
    prompt = "Go through r/Python and r/rust. Link each post to https://example.org/search?q=<name>."
    report, reply = report_of(prompt, [])
    [_, unread] = report["sources"]
    assert (unread["source"], unread["status"], unread["requested"]) == ("r/rust", "not_read", True)
    assert "source_not_read" in report["warnings"] and "asked for but not read" in reply and report["unexplained"] == ["r/rust: not read"]

    report, _ = report_of(prompt, [], left_out=[{"source": "r/rust", "reason": "The user's other job covers it."}])
    assert report["unexplained"] == [] and "source_not_read" not in report["warnings"]
    assert report["sources"][1]["reason"] == "The user's other job covers it."


def test_a_source_the_prompt_did_not_name_is_still_accounted_for(report_dir):
    report, _ = report_of("Find the Python subreddits' posts about uv.", [post(1)])
    [row] = report["sources"]
    assert row["requested"] is False and row["posts"]["unaccounted"] == 1


def test_a_page_fetched_along_the_way_is_not_a_candidate(report_dir):
    page = {"source": "https://example.com/docs", "status": "complete", "line": "https://example.com/docs: read in full."}
    context = context_for(LINKS, [], [READ, page])
    context.agent.state.set(tools.SOURCES, {canonical_link(page["source"]): {"text": "docs"}})
    write_report(title="t", criteria="c", items=[], tool_context=context)
    report = context.agent.state.get(tools.REPORT)
    assert report["unexplained"] == [] and [row["posts"]["found"] for row in report["sources"]] == [0, 0]


def test_a_page_the_prompt_named_is_a_candidate(report_dir):
    url = "https://example.com/changelog"
    page = {"source": url, "status": "complete", "line": f"{url}: read in full."}
    context = context_for(f"Check {url} for uv.", [], [page])
    context.agent.state.set(tools.SOURCES, {canonical_link(url): {"text": "uv changes"}})
    write_report(title="t", criteria="c", items=[], tool_context=context)
    report = context.agent.state.get(tools.REPORT)
    assert report["sources"][0]["posts"]["unaccounted"] == 1 and report["unexplained"] == [f"{url}: {canonical_link(url)}"]


def test_a_row_that_names_nothing_read_is_flagged(report_dir):
    report, reply = report_of(PATTERN, [], left_out=[{"source": "Python-99", "reason": "Lost it."}])
    assert "unknown_left_out" in report["warnings"] and "don't name a source or post your tools read: Python-99" in reply


def test_a_group_with_no_row_is_flagged_even_with_a_row_for_its_source(report_dir):
    report, reply = report_of(PATTERN, [post(1, group("Ada"))], left_out=[{"source": "r/Python", "reason": "Nothing here."}])

    [row] = report["sources"]
    assert row["posts"]["unaccounted"] == 0  # the post is covered, but what's in it has to be named
    assert row["groups"]["unaccounted"] == 1 and row["unaccounted"]["groups"] == ["Python-1: Ada"]
    assert "unaccounted_values" in report["warnings"] and "no item or left_out row covers these values: Python-1: Ada" in reply
    assert report["unexplained"] == ["Python-1: Ada"]


@pytest.mark.parametrize("row", [
    {"source": "Python-1", "value": "Ada", "reason": "A character in a story."},
    {"source": "r/Python", "value": "Ada", "reason": "A character in a story."},
    {"source": "Python-1", "reason": "The whole post is a story."},
])
def test_a_group_is_explained_by_naming_it_or_its_post(report_dir, row):
    report, _ = report_of(PATTERN, [post(1, group("Ada"))], left_out=[row, {"source": "r/Python", "reason": "Nothing else."}])
    [source] = report["sources"]
    assert source["groups"] == {"found": 1, "truncated": 0, "assessed": 1, "submitted": 0, "rejected": 1, "rendered": 0, "unaccounted": 0}
    assert report["unexplained"] == []


def test_a_row_for_one_value_leaves_the_others_unexplained(report_dir):
    posts = [post(1, group("Ada"), group("Bob"))]
    report, _ = report_of(PATTERN, posts, left_out=[{"source": "Python-1", "value": "Ada", "reason": "A story."}])
    assert report["sources"][0]["unaccounted"]["groups"] == ["Python-1: Bob"]


def test_every_group_in_a_post_has_to_be_reported_or_explained(report_dir):
    posts = [post(1, group("Ada"), group("Bob"), group("Cy"))]
    items = [item(1, fill="Ada"), item(1, fill="Bob")]
    report, _ = report_of(PATTERN, posts, items)

    [row] = report["sources"]
    assert row["groups"] == {"found": 3, "truncated": 0, "assessed": 2, "submitted": 2, "rejected": 0, "rendered": 2, "unaccounted": 1}
    assert adds_up(row["groups"])
    assert report["unexplained"] == ["Python-1: Cy"]


def test_every_decoded_line_of_a_group_needs_a_disposition(report_dir):
    report, _ = report_of(PATTERN, [post(1, group("Ada\n1234"))], [item(1, fill="Ada")])
    [row] = report["sources"]
    assert row["groups"]["unaccounted"] == 1 and report["unexplained"] == ["Python-1: 1234"]
    assert report["items"][0]["groups"][0]["lines"] == ["Ada", "1234"]


def test_remaining_decoded_lines_can_be_reported_or_explicitly_rejected(report_dir):
    posts = [post(1, group("Ada\n1234"))]
    for items, left_out in [
        ([item(1, fill="Ada"), item(1, fill="1234")], []),
        ([item(1, fill="Ada")], [{"source": "Python-1", "value": "1234", "reason": "Not a target value."}]),
    ]:
        report, _ = report_of(PATTERN, posts, items, left_out)
        assert report["sources"][0]["groups"]["rendered"] == 1
        assert report["unexplained"] == []


def test_post_ids_in_the_excluded_value_field_are_actionable_validation_errors(report_dir):
    context = context_for(PATTERN, [post(1), post(2)])
    with pytest.raises(ReportError) as caught:
        write_report(title="t", criteria="c", items=[], left_out=[
            {"source": "r/Python", "value": "Python-1", "reason": "No match."},
            {"source": "r/Python", "value": "Python-2", "reason": "No match."},
        ], tool_context=context)
    assert len(caught.value.errors) == 2
    assert all(f"left_out[{i}].value:" in error and "put the post id in source" in error for i, error in enumerate(caught.value.errors))
    assert list(report_dir.iterdir()) == []


def test_a_post_id_can_still_be_a_literal_value_in_recorded_source_text(report_dir):
    report, _ = report_of(PATTERN, [post(1, group("Python-2")), post(2)], left_out=[
        {"source": "r/Python", "value": "Python-2", "reason": "A literal identifier, not a target."},
        {"source": "r/Python", "reason": "No other matches."},
    ])
    assert report["unexplained"] == []


@pytest.mark.parametrize("exclusion", [
    {"source": "Python-1", "reason": "Exclude the post."},
    {"source": "Python-1", "value": "Ada", "reason": "Exclude the value."},
    {"source": "r/Python", "value": "Ada", "reason": "Exclude the value."},
])
def test_reporting_and_explicitly_excluding_the_same_post_or_value_is_rejected(report_dir, exclusion):
    context = context_for(PATTERN, [post(1, group("Ada"))])
    with pytest.raises(ReportError, match=r"left_out\[0\].source: this exclusion also appears in items"):
        write_report(title="t", criteria="c", items=[item(1, fill="Ada")], left_out=[exclusion], tool_context=context)


def test_wrong_post_values_point_to_recorded_alternatives_without_reassigning_them(report_dir):
    report, reply = report_of(PATTERN, [post(1), post(2, group("Ada"))], [item(1, fill="Ada")])
    assert report["items"][0]["post"] == "Python-1"
    assert report["items"][0]["possible_sources"] == ["Python-2"]
    assert "wrong_post" in report["warnings"] and "Python-1: check Python-2" in reply


def test_generated_search_urls_in_quotes_are_identified_as_output_not_evidence(report_dir):
    url = "https://example.org/search?q=Ada"
    report, reply = report_of(PATTERN, [post(1, group("Ada"))], [item(1, fill="Ada", quote=url)])
    assert report["items"][0]["evidence"][0]["generated_link_quote"]
    assert "generated_link_quote" in report["warnings"] and "copy source text instead" in reply


def test_a_quote_url_that_really_occurs_in_its_source_is_valid(report_dir):
    url = "https://example.org/search?q=Ada"
    report, _ = report_of(PATTERN, [post(1, group("Ada"), group(url))], [item(1, fill="Ada", quote=url)])
    assert not report["items"][0]["evidence"][0].get("generated_link_quote")
    assert "quote_not_found" not in report["warnings"]


def test_an_empty_fill_is_no_answer_for_a_value_that_is_there(report_dir):
    items = [{"post": "Python-1", "fill": "", "description": "No value here.", "quote": "some words"}]
    report, _ = report_of(PATTERN, [post(1, group("Ada"))], items)
    [row] = report["sources"]
    assert row["posts"]["rejected"] == 1 and row["unaccounted"]["groups"] == ["Python-1: Ada"]


def test_a_group_the_model_never_got_is_cut_off_not_unexplained(report_dir):
    report, _ = report_of(PATTERN, [post(1, group("Ada", shown=False))], left_out=[{"source": "r/Python", "reason": "Nothing here."}])
    [row] = report["sources"]
    assert row["groups"]["truncated"] == 1 and row["groups"]["unaccounted"] == 0 and report["unexplained"] == []


def test_groups_only_count_in_a_job_with_a_link_pattern(report_dir):
    report, _ = report_of(LINKS, [post(1, group("Ada"))], left_out=[{"source": "r/Python", "reason": "None are about uv."}])
    [row] = report["sources"]
    assert row["groups"]["found"] == 0 and report["unexplained"] == []


def test_counts_what_each_source_put_in_the_report(report_dir):
    other = post(7) | {"read_from": "r/rust", "id": "rust-1", "link": "https://www.reddit.com/r/rust/comments/7/x/"}
    rust = {"source": "r/rust", "status": "complete", "line": "r/rust: read in full."}
    items = [
        item(1, fill="some"),
        item(1, fill="words"),
        item(1, fill="words", quote="Post 1"),
    ]
    report, _ = report_of(PATTERN, [post(1), other], items, [{"source": "r/rust", "reason": "No names."}], reads=[READ, rust])

    python, rust_row = report["sources"]
    assert python["report"] == {"posts": 1, "values": 2, "evidence": 3}
    assert rust_row["report"] == {"posts": 0, "values": 0, "evidence": 0} and rust_row["posts"]["rejected"] == 1
    assert report["counts"] == {"posts": 1, "values": 2, "evidence": 3}


def test_the_page_shows_the_counts_and_why_things_were_left_out(report_dir):
    report, _ = report_of(PATTERN, [post(1), post(2)], [item(1, fill="some")], [{"source": "Python-2", "value": "Ada", "reason": "A story & more."}])
    html = open(report["path"]).read()
    assert "<li>r/Python: 2 candidates found, 0 cut off before the model read them, 2 assessed, 1 submitted, 1 rejected, 1 rendered, 0 unaccounted.</li>" in html
    assert f'<a href="{link_of(2)}" rel="noopener noreferrer">Post 2</a> <span class="value">Ada</span>: A story &amp; more.' in html


def test_left_out_rows_are_checked_with_everything_else(report_dir):
    bad = [{"source": "r/Python"}, {"source": "Python-1", "reason": "", "note": "x"}, "r/Python", {"source": "Python-1", "reason": "r", "value": "a\nb"}]
    with pytest.raises(ReportError) as caught:
        write_report(title="t", criteria="c", items=[], left_out=bad, tool_context=tool_context())
    assert caught.value.errors == [
        "left_out[0].reason: missing, expected a string",
        "left_out[1].note: unknown field, expected only source, value, reason",
        "left_out[1].reason: expected a non-empty string, got an empty one",
        "left_out[2]: expected an object with source, value, reason, got a string",
        "left_out[3].value: expected one value on one line, got 'a\\nb'",
    ]
    assert list(report_dir.iterdir()) == []


@pytest.mark.parametrize("left_out", [None, "r/Python", {"source": "r/Python", "reason": "r"}])
def test_left_out_has_to_be_a_list(report_dir, left_out):
    with pytest.raises(ReportError, match="left_out: expected a list of objects"):
        write_report(title="t", criteria="c", items=[], left_out=left_out, tool_context=tool_context())


@pytest.mark.parametrize("prompt, sources", [
    ("Go through r/Python and /r/rust, then R/golang.", ["r/Python", "r/rust", "r/golang"]),
    ("Read https://www.reddit.com/r/learnpython/new/ today.", ["r/learnpython"]),
    ("Read https://example.com/feed.xml, and https://example.org/page.", ["https://example.com/feed.xml", "https://example.org/page"]),
    ("Link each to https://example.org/search?q=<name> or https://example.net/?q={x}.", []),
    ("Compare and/or contrast, in r/w mode.", []),
])
def test_finds_the_sources_a_prompt_names(prompt, sources):
    assert requested_sources(prompt) == sources


def test_a_source_is_the_same_source_however_it_is_written():
    assert len({source_key(name) for name in ("r/Python", "python", "R/PYTHON", "/r/python/")}) == 1
    assert source_key("https://www.example.com/feed/") == source_key("links on http://example.com/feed")


def test_a_group_is_matched_by_its_words_not_by_slashes():
    assert contains_words("AC/DC\n1234", "1234") and contains_words("AC/DC\n1234", "AC/DC")
