# agents

A generic hello-world agent, built one layer at a time to learn how agents work end to end:
agent, harness, inference, gateway and tools. Local first (Ollama), then AWS (Strands, Bedrock, AgentCore).

The architecture diagrams live in [agent-stack.html](agent-stack.html).

This repository holds only the generic agent. Real jobs, traces, outputs and sessions are gitignored.

## Run it

Needs [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/download) with a model that supports tools.

```bash
ollama pull gemma4:e2b
uv run hello-agent "What time is it in Tokyo?"
uv run pytest
```

Each run writes its steps to `traces/<run_id>.jsonl`. Set `AGENT_MODEL` to try another Ollama model, and `OLLAMA_HOST` if Ollama isn't on `localhost:11434`.
