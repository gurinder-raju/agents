"""Whether a job did what it was for, and what its exit status says.

Two separate questions. A report's status says what happened to the file. The job's outcome says whether
the job is done: a report can be saved for a job that failed, because every source was down and the
report says so.

The policy for partial reports:
- A call that fails validation saves nothing. The model gets every problem back and can call again.
- A valid call is always saved, whatever else is wrong, because some of the answer beats none of it.
  The report says at the top what makes it partial, and write_report tells the model.
- It's partial if a source failed or was cut short, if no source was read, if an item was left out, or
  if something in it couldn't be confirmed: see NOTES for the only warnings that don't count.
- An empty report is a complete answer only if every source was read in full. Otherwise it says
  that nothing matched in what could be read, or that nothing could be.

A job that needs a report says so with --report-required. Its instructions then ask for one. When the
agent returns, the command line looks for this run's file, and if it isn't there, asks the model once
more to finish. If it's still not there, the job's outcome is report_not_created.
"""

from typing import Any

from .tools import COVERAGE, REPORT, REPORT_FAILURE, saved_report

FINISH_REPORT = (
    "The job isn't finished: no report has been saved, and a list in your reply doesn't count. Call write_report "
    "now with title, criteria and items, one item per match, or an empty items list if nothing matched."
)

# 2 is taken: argparse exits with it when it can't read the command line.
EXIT_CODES = {"complete": 0, "failed": 1, "partial": 3, "report_not_created": 4}

# Warnings that say how a report was made, not what it covers or whether it's right. Any other warning
# makes a report partial, so a new kind of warning counts unless it's added here.
NOTES = {"links_not_checked", "merged_items", "legacy_item"}
# What the job's list of reasons already says in words, from the reads themselves.
_READ_WARNINGS = {"source_failed", "source_incomplete"}


def finalization_request(state: Any) -> str:
    """What to tell a model that stopped without saving the report, including why its last call didn't."""
    failure = state.get(REPORT_FAILURE)
    if failure is None:
        return FINISH_REPORT
    return "\n- ".join([f"{FINISH_REPORT}\nThe last call saved nothing, because of these:", *failure["errors"]])


def job_outcome(state: Any, run_id: str, report_required: bool = False, error: Exception | None = None) -> dict[str, Any]:
    """Work out how the job went, from what the tools recorded in the agent's state.

    `error` is whatever stopped the run. "complete" means it did all it set out to, which can be zero
    matches. "partial" is the policy above. "report_not_created" means it needed a report and this run's
    file isn't there: because the job requires one, or because it tried and failed, or because the file
    has gone. "failed" means the run stopped, or none of its sources could be read, whatever became of
    the report.
    """
    report = saved_report(state, run_id)
    recorded = state.get(REPORT)
    failure = state.get(REPORT_FAILURE)
    reads = list((state.get(COVERAGE) or {}).values())
    gaps = [read["line"] for read in reads if read["status"] != "complete"]
    warnings = [code for code in (report or {}).get("warnings", []) if code not in NOTES]

    if error is not None:
        outcome, reasons = "failed", [f"the run stopped: {error}"]
    elif report is None and recorded:
        outcome, reasons = "report_not_created", [f"this run has no report at {recorded['path']}"]
    elif report is None and (report_required or failure):
        outcome = "report_not_created"
        reasons = failure["errors"] if failure else ["the agent stopped without saving a report"]
    elif reads and all(read["status"] == "failed" for read in reads):
        outcome, reasons = "failed", gaps
    elif warnings or gaps:
        outcome, reasons = "partial", gaps + [code for code in warnings if code not in _READ_WARNINGS]
    else:
        outcome, reasons = "complete", []
    if recorded and not report:
        artifact = recorded | {"status": "not_found"}
    else:
        artifact = report or {"status": f"{failure['stage']}_failed" if failure else "not_saved"}
    return {
        "outcome": outcome,
        "exit_code": EXIT_CODES[outcome],
        "reasons": reasons,
        "artifact": artifact,
        "sources": {read["source"]: read["status"] for read in reads},
    }
