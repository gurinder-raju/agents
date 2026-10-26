"""The agent: which model it calls, which tools it has, and how it is told to behave."""

import os

from strands import Agent
from strands.models.ollama import OllamaModel

from .recorder import TraceRecorder
from .tools import current_time, fetch_url

# Generic on purpose. Instructions for a specific job belong in job files under jobs/, which git ignores.
SYSTEM_PROMPT = """You are a helpful assistant.
Use your tools when they help you answer. If you can't do something, say so plainly."""


def build_agent(recorder: TraceRecorder) -> Agent:
    model = OllamaModel(
        host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        model_id=os.environ.get("AGENT_MODEL", "gemma4:e2b"),
        temperature=0.2,
        # Ollama's context window: prompt plus reply, in tokens. Set explicitly so a fetched page fits.
        # Bigger windows use more RAM, so raise this only if the model starts losing track of a page.
        options={"num_ctx": 8192},
    )
    return Agent(
        model=model,
        tools=[current_time, fetch_url],
        system_prompt=SYSTEM_PROMPT,
        hooks=[recorder],
    )
