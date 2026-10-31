import json
import re
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
from fetcher.outcome import FINISH_ACCOUNTING, FINISH_REPORT, finalization_request, job_outcome, report_finished
from fetcher.tools import saved_report

READ = {"source": "r/Python", "status": "complete", "line": "r/Python: 2 posts."}
SHORT = {"source": "r/rust", "status": "incomplete", "line": "r/rust: 9 posts didn't fit."}
FAILED = {"source": "r/go", "status": "failed", "line": "r/go: could not be read (timeout)."}
RUN = "20261031-060002-d07486"


@pytest.fixture
def report(tmp_path):
    """A record of a saved report, with its file there unless a test says otherwise."""
    def make(*warnings, run_id=RUN, on_disk=True):
        path = tmp_path / f"{run_id}-t.html"
        if on_disk:
            path.write_text("<!doctype html>")
        return {"status": "saved", "run_id": run_id, "path": str(path), "matches": 0, "warnings": list(warnings), "skipped": {}}

    return make


def state(report=None, failure=None, reads=()):
    values = {tools.COVERAGE: {read["source"]: read for read in reads}}
    if report:
        values[tools.REPORT] = report
    if failure:
        values[tools.REPORT_FAILURE] = failure
    return AgentState(values)


def test_zero_matches_from_sources_read_in_full_is_complete(report):
    outcome = job_outcome(state(report(), reads=[READ]), RUN, report_required=True)
    assert (outcome["outcome"], outcome["exit_code"], outcome["reasons"]) == ("complete", 0, [])
    assert outcome["artifact"]["status"] == "saved"
    assert outcome["sources"] == {"r/Python": "complete"}


def test_a_question_that_needs_no_report_is_complete_without_one():
    outcome = job_outcome(state(), RUN)
    assert outcome["outcome"] == "complete"
    assert outcome["artifact"] == {"status": "not_saved"}


def test_warnings_about_how_a_report_was_made_do_not_make_it_partial(report):
    saved = report("duplicate_items", "legacy_item", "links_not_checked", "no_value")
    assert job_outcome(state(saved, reads=[READ]), RUN)["outcome"] == "complete"


@pytest.mark.parametrize("warning", [
    "unknown_post", "no_sources_read", "quote_not_found", "value_not_found", "link_not_from_tools", "link_missing",
    "link_unreachable", "unaccounted_candidates", "unaccounted_values", "source_not_read", "unknown_left_out",
    "a_warning_added_later",
])
def test_anything_else_makes_a_saved_report_partial(report, warning):
    outcome = job_outcome(state(report(warning), reads=[READ]), RUN, report_required=True)
    assert (outcome["outcome"], outcome["exit_code"], outcome["reasons"]) == ("partial", 3, [warning])
    assert outcome["artifact"]["status"] == "saved"  # the file is fine: it's the job that isn't finished


def test_a_failed_or_cut_short_source_makes_the_job_partial(report):
    outcome = job_outcome(state(report("source_failed", "source_incomplete"), reads=[READ, SHORT, FAILED]), RUN)
    assert outcome["outcome"] == "partial"
    assert outcome["reasons"] == [SHORT["line"], FAILED["line"]]
    assert outcome["sources"] == {"r/Python": "complete", "r/rust": "incomplete", "r/go": "failed"}


def test_a_saved_report_can_not_rescue_a_job_whose_sources_all_failed(report):
    outcome = job_outcome(state(report("source_failed"), reads=[FAILED]), RUN, report_required=True)
    assert (outcome["outcome"], outcome["exit_code"]) == ("failed", 1)
    assert outcome["artifact"]["status"] == "saved"


def test_a_required_report_that_was_never_attempted():
    outcome = job_outcome(state(reads=[READ]), RUN, report_required=True)
    assert (outcome["outcome"], outcome["exit_code"]) == ("report_not_created", 4)
    assert outcome["artifact"] == {"status": "not_saved"}


@pytest.mark.parametrize("stage", ["validation", "render", "write"])
def test_an_attempt_that_failed_is_no_report_even_if_the_job_did_not_require_one(stage):
    outcome = job_outcome(state(failure={"stage": stage, "errors": ["items: expected a list"]}, reads=[READ]), RUN)
    assert (outcome["outcome"], outcome["exit_code"]) == ("report_not_created", 4)
    assert outcome["artifact"] == {"status": f"{stage}_failed"}
    assert outcome["reasons"] == ["items: expected a list"]


def test_a_failed_retry_leaves_the_saved_report_standing(report):
    outcome = job_outcome(state(report(), failure={"stage": "validation", "errors": ["e"]}, reads=[READ]), RUN, report_required=True)
    assert (outcome["outcome"], outcome["artifact"]["status"]) == ("complete", "saved")


def test_a_run_that_stopped_failed_whatever_became_of_the_report(report):
    outcome = job_outcome(state(report(), reads=[READ]), RUN, error=ConnectionError("Ollama isn't running"))
    assert (outcome["outcome"], outcome["exit_code"]) == ("failed", 1)
    assert outcome["reasons"] == ["the run stopped: Ollama isn't running"]
    assert outcome["artifact"]["status"] == "saved"


def test_a_report_only_counts_if_its_file_is_there(report):
    gone = report(on_disk=False)
    assert saved_report(state(gone), RUN) is None
    outcome = job_outcome(state(gone, reads=[READ]), RUN, report_required=True)
    assert (outcome["outcome"], outcome["exit_code"]) == ("report_not_created", 4)
    assert outcome["artifact"]["status"] == "not_found"
    assert outcome["reasons"] == [f"this run has no report at {gone['path']}"]


def test_a_report_only_counts_for_the_run_that_saved_it(report):
    other = report(run_id="20261030-060005-9d814b")
    assert saved_report(state(other), RUN) is None
    assert saved_report(state(other), "20261030-060005-9d814b") == other
    assert job_outcome(state(other, reads=[READ]), RUN, report_required=True)["outcome"] == "report_not_created"


def test_a_report_with_something_unexplained_is_not_finished(report):
    assert not report_finished(state(), RUN)
    assert not report_finished(state(report(on_disk=False)), RUN)
    assert report_finished(state(report()), RUN)
    assert not report_finished(state(report("unaccounted_candidates") | {"unexplained": ["r/Python: Python-2"]}), RUN)


def test_says_what_is_unexplained_when_asking_to_finish(report):
    saved = report() | {"unexplained": ["r/Python: Python-2", "Python-3: Ada"]}
    assert finalization_request(state(saved)) == f"{FINISH_ACCOUNTING}\n- r/Python: Python-2\n- Python-3: Ada"


def test_a_failed_retry_is_not_told_that_no_report_was_saved(report):
    saved = report() | {"unexplained": ["r/Python: Python-2"]}
    request = finalization_request(state(saved, failure={"stage": "validation", "errors": ["items: expected a list"]}))
    assert "no report has been saved" not in request
    assert request.startswith(FINISH_ACCOUNTING) and "r/Python: Python-2" in request and "items: expected a list" in request


def test_a_failed_attempt_with_no_report_says_so(report):
    request = finalization_request(state(failure={"stage": "validation", "errors": ["items: expected a list"]}))
    assert request.startswith(FINISH_REPORT) and request.endswith("- items: expected a list")


@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    """Run the real command line and agent loop on a scripted model. Returns the exit status, output and trace."""
    monkeypatch.setattr(tools, "REPORT_DIR", tmp_path / "reports")
    unloaded = []
    monkeypatch.setattr(cli, "unload_model", unloaded.append)
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
        assert len(unloaded) == 1  # the model is freed however the run ends
        unloaded.clear()
        [trace] = tmp_path.glob("*.jsonl")
        steps = [json.loads(line) for line in trace.read_text().splitlines()]
        trace.unlink()
        return code, capsys.readouterr().out, steps, model

    return run_cli


READ_SOURCE = ("read_subreddit", {"subreddit": "learnpython"})
NOT_ABOUT_UV = [{"source": "r/learnpython", "reason": "No post is about uv."}]
EMPTY_REPORT = ("write_report", {"title": "uv posts", "criteria": "Posts about uv.", "items": [], "left_out": NOT_ABOUT_UV})


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
    assert "outcome: report_not_created (items[0].description: expected a string, got a number)" in out
    assert "report: " not in out


def test_a_model_that_goes_away_mid_run_exits_one(run):
    code, out, _, _ = run(READ_SOURCE, ConnectionError("Ollama isn't running"))
    assert code == 1
    assert "outcome: failed (the run stopped: Ollama isn't running)" in out


def test_a_question_needs_no_report_and_exits_zero(run):
    code, out, _, _ = run("It is noon in Tokyo.", prompt="What time is it in Tokyo?")
    assert code == 0
    assert "outcome: complete\n" in out


def kinds(steps, kind):
    return [step for step in steps if step["kind"] == kind]


def user_turns(model):
    """The text of each user message in the model's last request, leaving out tool results."""
    return [block["text"] for message in model.requests[-1]["messages"] if message["role"] == "user" for block in message["content"] if "text" in block]


def test_asks_the_model_to_finish_when_it_stops_without_the_report(run):
    code, out, steps, model = run(READ_SOURCE, "Here are the matches in my reply.", EMPTY_REPORT, "Saved.", flags=["--report-required"])
    assert code == 0
    assert "outcome: complete\n" in out
    assert user_turns(model)[1:] == [FINISH_REPORT]
    assert [step["report_required"] for step in kinds(steps, "invocation_start")] == [True, True]  # the second is the request

    [saved] = kinds(steps, "artifact_saved")
    assert saved["run_id"] == steps[0]["run_id"] and Path(saved["path"]).name.startswith(saved["run_id"])  # still this run


def test_asks_only_once_then_records_report_not_created(run):
    code, out, steps, model = run(READ_SOURCE, "One.", "Two.", "Never reached.", flags=["--report-required"])
    assert code == 4
    assert "outcome: report_not_created (the agent stopped without saving a report)" in out
    assert model.replies == ["Never reached."]
    [outcome] = kinds(steps, "job_outcome")
    assert outcome["outcome"] == "report_not_created" and outcome["artifact"] == {"status": "not_saved"}


def test_the_request_to_finish_repeats_what_was_wrong_with_the_last_call(run):
    bad = ("write_report", {"title": "t", "criteria": "c", "items": [{"post": "learnpython-1", "description": 3}]})
    code, _, steps, model = run(READ_SOURCE, bad, "I tried.", EMPTY_REPORT, "Saved.", flags=["--report-required"])
    assert code == 0
    assert "items[0].description: expected a string, got a number" in user_turns(model)[-1]

    [failed] = kinds(steps, "artifact_failed")
    assert failed["stage"] == "validation" and failed["errors"] == ["items[0].description: expected a string, got a number"]


def test_a_save_that_left_no_file_is_not_a_saved_report(run, monkeypatch):
    monkeypatch.setattr(tools, "_save_atomically", lambda path, text: None)
    code, out, _, model = run(READ_SOURCE, EMPTY_REPORT, "Saved.", EMPTY_REPORT, "Saved again.", "Never reached.", flags=["--report-required"])
    assert code == 4
    assert "outcome: report_not_created (this run has no report at " in out
    assert model.replies == ["Never reached."]


def test_a_write_that_keeps_failing_ends_as_report_not_created(run, monkeypatch):
    def disk_full(path, text):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(tools, "_save_atomically", disk_full)
    code, out, steps, _ = run(READ_SOURCE, EMPTY_REPORT, "The disk is full.", EMPTY_REPORT, "Still full.", flags=["--report-required"])
    assert code == 4
    assert "No space left on device" in out
    assert [step["stage"] for step in kinds(steps, "artifact_failed")] == ["write", "write"]
    assert kinds(steps, "job_outcome")[0]["artifact"] == {"status": "write_failed"}


def test_does_not_ask_again_after_a_run_that_crashed(run):
    code, _, _, model = run(ConnectionError("Ollama isn't running"), "Never reached.", flags=["--report-required"])
    assert code == 1
    assert model.replies == ["Never reached."]


def test_a_crash_while_finishing_the_report_still_frees_the_model(run):
    code, out, _, _ = run("Here are the matches.", ConnectionError("Ollama went away"), flags=["--report-required"])
    assert code == 1
    assert "outcome: failed (the run stopped: Ollama went away)" in out


def test_a_job_that_does_not_need_a_report_is_not_asked_for_one(run):
    code, _, _, model = run("It is noon in Tokyo.", "Never reached.", prompt="What time is it in Tokyo?")
    assert code == 0
    assert model.replies == ["Never reached."]


SILENT_REPORT = ("write_report", {"title": "uv posts", "criteria": "Posts about uv.", "items": []})


def test_asks_the_model_to_explain_what_it_read_and_left_out(run):
    code, out, steps, model = run(READ_SOURCE, SILENT_REPORT, "Saved.", EMPTY_REPORT, "Saved.", flags=["--report-required"])
    assert code == 0
    assert "outcome: complete\n" in out
    assert user_turns(model)[-1] == f"{FINISH_ACCOUNTING}\n- r/learnpython: learnpython-1"
    first, second = kinds(steps, "artifact_saved")
    assert first["warnings"] == ["unaccounted_candidates", "links_not_checked"] and first["path"] == second["path"]
    assert second["warnings"] == ["links_not_checked"]


def test_asks_only_once_then_ends_partial_with_the_report_still_there(run):
    code, out, _, model = run(READ_SOURCE, SILENT_REPORT, "Saved.", "That's all I have.", "Never reached.", flags=["--report-required"])
    assert code == 3
    assert "outcome: partial (unaccounted_candidates)" in out and "report: " in out
    assert model.replies == ["Never reached."]


def test_an_unexplained_omission_is_flagged_without_asking_if_no_report_was_required(run):
    code, out, _, model = run(READ_SOURCE, SILENT_REPORT, "Saved.", "Never reached.")
    assert code == 3
    assert "outcome: partial (unaccounted_candidates)" in out
    assert model.replies == ["Never reached."]


def find_marked(text):
    return [
        {"text": match.group(1), "lines": match.group(1).split("\n"), "start": match.start(), "end": match.end()}
        for match in re.finditer(r"\[found: ([^\]]*)\]", text)
    ]


def test_a_group_with_no_row_is_caught_whole_run(run, monkeypatch):
    monkeypatch.setattr(agent_module, "load_local_tools", lambda: ([], [lambda text: text.replace("venv", "[found: Ada]")], [find_marked]))
    prompt = "Find posts in r/learnpython. Link each to https://example.org/search?q=<name>."
    report = ("write_report", {"title": "names", "criteria": "c", "items": [], "left_out": NOT_ABOUT_UV})
    fixed = ("write_report", {"title": "names", "criteria": "c", "left_out": NOT_ABOUT_UV, "items": [
        {"post": "learnpython-1", "fill": "Ada", "description": "Named in the post.", "quote": "[found: Ada]"},
    ]})
    code, out, steps, model = run(READ_SOURCE, report, "Saved.", fixed, "Saved.", prompt=prompt, flags=["--report-required"])

    assert "learnpython-1: Ada" in user_turns(model)[-1]
    assert code == 0 and "outcome: complete\n" in out
    saved = kinds(steps, "artifact_saved")[-1]
    [source] = saved["sources"]
    assert source["groups"]["rendered"] >= 1 and source["groups"]["unaccounted"] == 0


def test_the_trace_and_the_data_file_hold_the_same_per_value_records(run):
    prompt = "Find posts in r/learnpython. Link each to https://example.org/search?q=<name>."
    report = ("write_report", {"title": "names", "criteria": "c", "items": [
        {"post": "learnpython-1", "fill": "uv", "description": "Named in the post.", "quote": "pip or uv"},
        {"post": "learnpython-1", "fill": "pip", "description": "Also named.", "quote": "venv & pip"},
        {"post": "learnpython-1", "fill": "uv", "description": "In a comment.", "quote": "uv creates the venv"},
    ]})
    code, _, steps, _ = run(READ_SOURCE, report, "Saved.", prompt=prompt, flags=["--report-json"])
    assert code == 0

    [saved] = kinds(steps, "artifact_saved")
    data = json.loads(Path(saved["data_path"]).read_text())
    assert saved["schema_version"] == data["schema_version"] == 2
    assert saved["counts"] == data["counts"] == {"posts": 1, "values": 2, "evidence": 3}
    assert saved["items"] == data["items"] and saved["sources"] == data["sources"] == kinds(steps, "job_outcome")[0]["artifact"]["sources"]
    assert data["sources"][0]["posts"] == {"found": 1, "truncated": 0, "assessed": 1, "submitted": 1, "rejected": 0, "rendered": 1, "unaccounted": 0}
    assert [(record["fill"], len(record["evidence"])) for record in data["items"]] == [("uv", 2), ("pip", 1)]


def test_the_trace_has_the_tool_requested_and_the_tool_run(run):
    call = ("link-report", {"title": "uv posts", "criteria": "Posts about uv.", "items": [], "left_out": NOT_ABOUT_UV})
    code, _, steps, _ = run(READ_SOURCE, call, ("no_such_tool", {"x": "y"}), "Saved.")
    assert code == 0
    results = kinds(steps, "tool_result")
    assert [(r["requested_tool"], r["executed_tool"]) for r in results] == [
        ("read_subreddit", "read_subreddit"),
        ("link-report", "write_report"),
        ("no_such_tool", None),
    ]
    assert all(result["tool_use_id"].startswith("tooluse_") for result in results)
    assert [call["tool"] for call in kinds(steps, "tool_call")] == ["read_subreddit", "link-report", "no_such_tool"]
    assert kinds(steps, "tool_call")[1]["tool_use_id"] == results[1]["tool_use_id"]


def test_the_trace_ends_with_the_saved_report_and_the_jobs_outcome(run):
    _, _, steps, _ = run(READ_SOURCE, ("read_subreddit", {"subreddit": "golang"}), EMPTY_REPORT, "Done.", flags=["--report-required"])
    [saved] = kinds(steps, "artifact_saved")
    assert saved["status"] == "saved" and saved["warnings"] == ["source_failed", "links_not_checked"]
    assert saved["schema_version"] == 2

    assert steps[-1]["kind"] == "job_outcome"
    assert steps[-1]["outcome"] == "partial" and steps[-1]["exit_code"] == 3
    assert steps[-1]["artifact"]["path"] == saved["path"]
    assert steps[-1]["sources"] == {"r/learnpython": "complete", "r/golang": "failed"}


def test_the_model_never_sees_the_runs_settings_or_schema_version(run):
    _, _, _, model = run(READ_SOURCE, EMPTY_REPORT, "Done.", flags=["--report-json"])
    [spec] = [spec for spec in model.requests[0]["tool_specs"] if spec["name"] == "write_report"]
    assert list(spec["inputSchema"]["json"]["properties"]) == ["title", "criteria", "items", "sort", "left_out"]


def test_saves_the_reports_data_on_request(run):
    item = {"post": "learnpython-1", "description": "Asks about uv.", "quote": "use venv & pip or uv"}
    _, _, steps, _ = run(READ_SOURCE, ("write_report", {"title": "uv posts", "criteria": "c", "items": [item]}), "Done.", flags=["--report-json"])
    [saved] = kinds(steps, "artifact_saved")
    data = json.loads(Path(saved["data_path"]).read_text())

    assert data["schema_version"] == 2 and data["run_id"] == saved["run_id"] and data["path"] == saved["path"]
    assert data["criteria"] == "c" and data["prompt"] == "Find posts in r/learnpython about uv"
    [source] = data["sources"]
    assert source["source"] == "r/learnpython" and source["status"] == "complete" and source["requested"] is True
    assert source["line"] == "r/learnpython: 1 posts and 2 comments from the last 24 hours."
    [entry] = data["items"]
    assert entry["post"] == "learnpython-1"
    assert entry["title"] == "Question about virtual environments" and entry["comment_count"] == 2
    assert entry["links"][0]["check"] == "unverified"


def test_saves_no_data_file_unless_asked(run):
    _, _, steps, _ = run(READ_SOURCE, EMPTY_REPORT, "Done.")
    [saved] = kinds(steps, "artifact_saved")
    assert "data_path" not in saved and not Path(saved["path"]).with_suffix(".json").exists()
