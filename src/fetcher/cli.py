"""Command line entry point: uv run fetcher "your prompt". See outcome.py for what its exit status means."""

import argparse
import traceback

from .agent import build_agent, unload_model
from .outcome import job_outcome
from .recorder import TraceRecorder


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent once.")
    parser.add_argument("prompt", help="what to ask the agent")
    parser.add_argument("--trace-dir", default="traces", help="where to write the step trace (default: traces/)")
    parser.add_argument("--no-verify-links", action="store_true", help="don't check a report's links over the network")
    args = parser.parse_args()

    recorder = TraceRecorder(args.trace_dir)
    agent = build_agent(recorder, args.prompt)
    error = None
    try:
        # Strands' default callback handler streams the reply to the terminal. invocation_state reaches
        # tools that ask for a ToolContext, and never the model: write_report names its file after the run.
        agent(args.prompt, invocation_state={"run_id": recorder.run_id, "verify_links": not args.no_verify_links})
    except Exception as failure:  # the outcome below reports it
        traceback.print_exc()
        error = failure
    finally:
        # The model holds most of the Mac's memory while it's loaded, so free it as soon as the run ends.
        unload_model(agent)

    outcome = job_outcome(agent.state, error=error)
    print()
    if path := outcome["artifact"].get("path"):
        print(f"\nreport: {path}")
    reasons = outcome["reasons"]
    print(f"\noutcome: {outcome['outcome']}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    print(f"trace: {recorder.path}")
    return outcome["exit_code"]
