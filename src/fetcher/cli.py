"""Command line entry point: uv run fetcher "your prompt"."""

import argparse

from .agent import build_agent, unload_model
from .recorder import TraceRecorder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent once.")
    parser.add_argument("prompt", help="what to ask the agent")
    parser.add_argument("--trace-dir", default="traces", help="where to write the step trace (default: traces/)")
    parser.add_argument("--no-verify-links", action="store_true", help="don't check a report's links over the network")
    args = parser.parse_args()

    recorder = TraceRecorder(args.trace_dir)
    agent = build_agent(recorder, args.prompt)
    try:
        # Strands' default callback handler streams the reply to the terminal. invocation_state reaches
        # tools that ask for a ToolContext, and never the model: write_report names its file after the run.
        agent(args.prompt, invocation_state={"run_id": recorder.run_id, "verify_links": not args.no_verify_links})
    finally:
        # The model holds most of the Mac's memory while it's loaded, so free it as soon as the run ends.
        unload_model(agent)
    print(f"\n\ntrace: {recorder.path}")
