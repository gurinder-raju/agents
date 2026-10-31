import json
import sys
from pathlib import Path

import httpx
import pytest
from scripted import ScriptedModel
from strands.agent.state import AgentState
from test_feeds import ATOM
from test_subreddit import COMMENTS, COMMENTS_URL, LISTING_URL

from fetcher import agent as agent_module
from fetcher import cli, tools
from fetcher.outcome import job_outcome

READ = {"source": "r/Python", "status": "complete", "line": "r/Python: 2 posts."}
SHORT = {"source": "r/rust", "status": "incomplete", "line": "r/rust: 9 posts didn't fit."}
FAILED = {"source": "r/go", "status": "failed", "line": "r/go: could not be read (timeout)."}
PATH = "/reports/run-t.html"


def report(*warnings, **fields):
    return {"status": "saved", "path": PATH, "matches": 0, "warnings": list(warnings), "skipped": {}} | fields


def state(report=None, failure=None, reads=()):
    values = {tools.COVERAGE: {read["source"]: read for read in reads}}
    if report:
        values[tools.REPORT] = report
    if failure:
        values[tools.REPORT_FAILURE] = failure
    return AgentState(values)


def test_zero_matches_from_sources_read_in_full_is_complete():
    outcome = job_outcome(state(report(), reads=[READ]), require_report=True)
    assert (outcome["outcome"], outcome["exit_code"], outcome["reasons"]) == ("complete", 0, [])
    assert outcome["artifact"]["status"] == "saved"
    assert outcome["sources"] == {"r/Python": "complete"}


def test_a_question_that_needs_no_report_is_complete_without_one():
    outcome = job_outcome(state())
    assert outcome["outcome"] == "complete"
    assert outcome["artifact"] == {"status": "not_saved"}


def test_warnings_about_how_a_report_was_made_do_not_make_it_partial():
    saved = report("merged_items", "legacy_item", "links_not_checked")
    assert job_outcome(state(saved, reads=[READ]))["outcome"] == "complete"


@pytest.mark.parametrize("warning", [
    "unknown_post", "no_value", "no_sources_read", "quote_not_found", "value_not_found",
    "link_not_from_tools", "link_missing", "link_unreachable", "a_warning_added_later",
])
def test_anything_else_makes_a_saved_report_partial(warning):
    outcome = job_outcome(state(report(warning), reads=[READ]), require_report=True)
    assert (outcome["outcome"], outcome["exit_code"], outcome["reasons"]) == ("partial", 3, [warning])
    assert outcome["artifact"]["status"] == "saved"  # the file is fine: it's the job that isn't finished


def test_a_failed_or_cut_short_source_makes_the_job_partial():
    outcome = job_outcome(state(report("source_failed", "source_incomplete"), reads=[READ, SHORT, FAILED]))
    assert outcome["outcome"] == "partial"
    assert outcome["reasons"] == [SHORT["line"], FAILED["line"]]
    assert outcome["sources"] == {"r/Python": "complete", "r/rust": "incomplete", "r/go": "failed"}


def test_a_saved_report_can_not_rescue_a_job_whose_sources_all_failed():
    outcome = job_outcome(state(report("source_failed"), reads=[FAILED]), require_report=True)
    assert (outcome["outcome"], outcome["exit_code"]) == ("failed", 1)
    assert outcome["artifact"]["status"] == "saved"


def test_a_required_report_that_was_never_attempted():
    outcome = job_outcome(state(reads=[READ]), require_report=True)
    assert (outcome["outcome"], outcome["exit_code"]) == ("no_report", 4)
    assert outcome["artifact"] == {"status": "not_saved"}


@pytest.mark.parametrize("stage", ["validation", "render", "write"])
def test_an_attempt_that_failed_is_no_report_even_if_the_job_did_not_require_one(stage):
    outcome = job_outcome(state(failure={"stage": stage, "errors": ["items: expected a list"]}, reads=[READ]))
    assert (outcome["outcome"], outcome["exit_code"]) == ("no_report", 4)
    assert outcome["artifact"] == {"status": f"{stage}_failed"}
    assert outcome["reasons"] == ["items: expected a list"]


def test_a_failed_retry_leaves_the_saved_report_standing():
    outcome = job_outcome(state(report(), failure={"stage": "validation", "errors": ["e"]}, reads=[READ]), require_report=True)
    assert (outcome["outcome"], outcome["artifact"]["status"]) == ("complete", "saved")


def test_a_run_that_stopped_failed_whatever_became_of_the_report():
    outcome = job_outcome(state(report(), reads=[READ]), error=ConnectionError("Ollama isn't running"))
    assert (outcome["outcome"], outcome["exit_code"]) == ("failed", 1)
    assert outcome["reasons"] == ["the run stopped: Ollama isn't running"]
    assert outcome["artifact"]["status"] == "saved"


@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    """Run the real command line and agent loop on a scripted model. Returns the exit status, output and trace."""
    monkeypatch.setattr(tools, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(cli, "unload_model", lambda agent: None)
    pages = {LISTING_URL: ATOM, COMMENTS_URL: COMMENTS}

    def http_get(url):
        if url in pages:
            return httpx.Response(200, text=pages[url])
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(tools, "http_get", http_get)

    def run_cli(*replies, flags=(), prompt="Find posts in r/learnpython about uv"):
        model = ScriptedModel(*replies)
        monkeypatch.setattr(agent_module, "OllamaModel", lambda **config: model)
        monkeypatch.setattr(sys, "argv", ["fetcher", prompt, "--trace-dir", str(tmp_path), "--no-verify-links", *flags])
        code = cli.main()
        [trace] = tmp_path.glob("*.jsonl")
        steps = [json.loads(line) for line in trace.read_text().splitlines()]
        trace.unlink()
        return code, capsys.readouterr().out, steps, model

    return run_cli


READ_SOURCE = ("read_subreddit", {"subreddit": "learnpython"})
EMPTY_REPORT = ("write_report", {"title": "uv posts", "criteria": "Posts about uv.", "items": []})


def test_a_clean_run_exits_zero_and_prints_the_report(run):
    code, out, _, _ = run(READ_SOURCE, EMPTY_REPORT, "No posts matched.")
    assert code == 0
    assert "outcome: complete\n" in out
    report_line = next(line for line in out.splitlines() if line.startswith("report: "))
    path = Path(report_line.removeprefix("report: "))
    assert path.is_absolute() and "Nothing matched the criteria." in path.read_text()


def test_an_unreadable_source_with_a_report_exits_one_and_the_report_says_why(run):
    code, out, _, _ = run(("read_subreddit", {"subreddit": "golang"}), EMPTY_REPORT, "I couldn't read it.")
    assert code == 1
    assert "outcome: failed (r/golang: could not be read (no route to host).)" in out
    path = Path(next(line for line in out.splitlines() if line.startswith("report: ")).removeprefix("report: "))
    assert "No source could be read, so nothing was checked." in path.read_text()


def test_one_unreadable_source_of_two_exits_three(run):
    code, out, _, _ = run(READ_SOURCE, ("read_subreddit", {"subreddit": "golang"}), EMPTY_REPORT, "Done.")
    assert code == 3
    assert "outcome: partial (r/golang: could not be read" in out


def test_an_invalid_report_that_is_never_fixed_exits_four(run):
    bad = ("write_report", {"title": "t", "criteria": "c", "items": [{"post": "learnpython-1", "description": 3}]})
    code, out, _, _ = run(READ_SOURCE, bad, "I tried.")
    assert code == 4
    assert "outcome: no_report (items[0].description: expected a string, got a number)" in out
    assert "report: " not in out


def test_a_model_that_goes_away_mid_run_exits_one(run):
    code, out, _, _ = run(READ_SOURCE, ConnectionError("Ollama isn't running"))
    assert code == 1
    assert "outcome: failed (the run stopped: Ollama isn't running)" in out


def test_a_question_needs_no_report_and_exits_zero(run):
    code, out, _, _ = run("It is noon in Tokyo.", prompt="What time is it in Tokyo?")
    assert code == 0
    assert "outcome: complete\n" in out
