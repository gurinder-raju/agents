"""The agent: which model it calls, which tools it has, and how it is told to behave."""

import os
import sys
from pathlib import Path

import ollama
from strands import Agent
from strands.models.ollama import OllamaModel
from strands.vended_plugins.skills import AgentSkills

from .recorder import TraceRecorder
from .reply_retry import ReplyRetry
from .skill_redirect import SkillNameRedirect
from .tools import current_time, fetch_url, read_feed, read_subreddit, write_report

# Generic on purpose. Instructions for a specific job belong in job files under jobs/, which git ignores.
SYSTEM_PROMPT = """You are a helpful assistant.
Use your tools when they help you answer. If you can't do something, say so plainly.
When the user asks for a report, save it with write_report, even when nothing matched.
Skills listed under available_skills are not tools. If one fits the task, call the `skills` tool with
its name before anything else, then follow the instructions it returns."""

# Each skill is a folder with a SKILL.md. skills/ holds generic ones and is committed;
# jobs/skills/ is for your own, and git ignores it. Paths are relative to where you run the agent.
SKILL_DIRS = ["skills", "jobs/skills"]


def build_agent(recorder: TraceRecorder) -> Agent:
    model = OllamaModel(
        host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        # gemma4:e4b made no false matches in a benchmark where gemma4:e2b made one to four a run. It
        # peaks at about 11 GB on a 16 GB Mac; for less memory and more mistakes, set AGENT_MODEL=gemma4:e2b.
        model_id=os.environ.get("AGENT_MODEL", "gemma4:e4b"),
        temperature=0.2,
        # A cap on each reply, thinking included. Normal replies stay under about 5,000 tokens; without
        # a cap, gemma4:12b once looped on the same text for 60,000 tokens and 30 minutes. ReplyRetry
        # asks again when a reply hits it.
        max_tokens=8192,
        # Ollama's context window: prompt plus reply, in tokens. 32k fits a day of a subreddit's posts and
        # comments, or a week of a quiet one, and it is part of the memory a loaded model takes. Ollama only
        # applies this when it loads the model: if `ollama ps` shows a smaller CONTEXT, run `ollama stop <model>`.
        options={"num_ctx": 32768},
    )
    # The plugin lists each skill's name and description in the system prompt, and adds a
    # `skills` tool the model calls to load a skill's full instructions when it needs them.
    skills = AgentSkills(skills=[d for d in SKILL_DIRS if Path(d).is_dir()])
    return Agent(
        model=model,
        tools=[current_time, fetch_url, read_feed, read_subreddit, write_report],
        system_prompt=SYSTEM_PROMPT,
        # Registered after the recorder, so it runs first after a model call and the trace shows its retries.
        hooks=[recorder, SkillNameRedirect(skills), ReplyRetry()],
        plugins=[skills],
    )


def unload_model(agent: Agent) -> None:
    """Ask Ollama to drop the model from memory now, instead of after its default 5 idle minutes.

    gemma4:e4b holds about 11 GB while it's loaded, most of a 16 GB Mac. This is what `ollama stop` does.
    """
    model = agent.model
    try:
        ollama.Client(model.host, **model.client_args).generate(model=model.config["model_id"], keep_alive=0)
    except Exception as error:  # Ollama isn't running, or the model is already gone: nothing to free
        print(f"could not unload {model.config['model_id']}: {error}", file=sys.stderr)
