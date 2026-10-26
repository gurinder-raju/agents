# agents

An experimental, portable agent design: one agent that runs locally on Ollama or in the cloud on AWS Bedrock and AgentCore.

The agent here is **agent-v3**, built on the Strands Harness SDK one layer at a time: agent, harness,
inference, gateway and tools. It runs locally first, then on AWS.

The architecture diagrams live in [agent-stack.html](agent-stack.html).

This repository holds only the generic agent. Real jobs, traces, outputs and sessions are gitignored.

## Run it

Needs [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/download) with a model that supports tools.

```bash
ollama pull gemma4:e2b
uv run agent-v3 "What time is it in Tokyo?"
uv run agent-v3 "Go through the new posts in r/Python. For each one, tell me which Python libraries the post or its comments mention."
uv run pytest
```

Skills live in `skills/<name>/SKILL.md`; put your own in `jobs/skills/`, which git ignores.

Ask for a report and the agent saves an HTML page of links, with your prompt and its criteria at the top,
to `outputs/reports/<run_id>-<title>.html`. Like traces, reports are gitignored.

Each run writes its steps to `traces/<run_id>.jsonl`. Set `AGENT_MODEL` to try another Ollama model, and `OLLAMA_HOST` if Ollama isn't on `localhost:11434`.
