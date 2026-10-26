"""Command line entry point: uv run agent-v3 "your prompt"."""

import argparse

from .agent import build_agent
from .recorder import TraceRecorder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent once.")
    parser.add_argument("prompt", help="what to ask the agent")
    parser.add_argument("--trace-dir", default="traces", help="where to write the step trace (default: traces/)")
    args = parser.parse_args()

    recorder = TraceRecorder(args.trace_dir)
    agent = build_agent(recorder)
    agent(args.prompt)  # Strands' default callback handler streams the reply to the terminal
    print(f"\n\ntrace: {recorder.path}")
