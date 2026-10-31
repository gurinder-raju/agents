"""Command line entry point: uv run fetcher "your prompt". See outcome.py for what its exit status means."""

import argparse
import sys
import traceback

from .agent import build_agent, unload_model
from .outcome import finalization_request, job_outcome, report_finished
from .recorder import TraceRecorder


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent once.")
    parser.add_argument("prompt", help="what to ask the agent")
    parser.add_argument("--trace-dir", default="traces", help="where to write the step trace (default: traces/)")
    parser.add_argument(
        "--report-required", action="store_true", help="the job must save a report: ask the model once more if it hasn't"
    )
    parser.add_argument("--no-verify-links", action="store_true", help="don't check a report's links over the network")
    parser.add_argument("--report-json", action="store_true", help="also save the report's data as JSON beside its HTML")
    args = parser.parse_args()

    recorder = TraceRecorder(args.trace_dir)
    agent = build_agent(recorder, args.prompt, args.report_required)
    # invocation_state reaches the hooks and tools that ask for a ToolContext, and never the model.
    # write_report names its file after the run.
    settings = {
        "run_id": recorder.run_id,
        "report_required": args.report_required,
        "verify_links": not args.no_verify_links,
        "report_json": args.report_json,
    }
    error = None
    try:
        # Strands' default callback handler streams the reply to the terminal.
        agent(args.prompt, invocation_state=settings)
        if args.report_required and not report_finished(agent.state, recorder.run_id):
            print("the report isn't finished; asking the model to finish it", file=sys.stderr, flush=True)
            agent(finalization_request(agent.state), invocation_state=settings)
    except Exception as failure:  # the outcome below reports it
        traceback.print_exc()
        error = failure
    finally:
        # The model holds most of the Mac's memory while it's loaded, so free it as soon as the run ends.
        unload_model(agent)

    outcome = job_outcome(agent.state, recorder.run_id, args.report_required, error)
    recorder.record_outcome(outcome)
    print()
    if path := outcome["artifact"].get("path"):
        print(f"\nreport: {path}")
    reasons = outcome["reasons"]
    print(f"\noutcome: {outcome['outcome']}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    print(f"trace: {recorder.path}")
    return outcome["exit_code"]
