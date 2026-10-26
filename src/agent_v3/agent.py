"""The agent: which model it calls, which tools it has, and how it is told to behave."""

import os
from pathlib import Path

from strands import Agent
from strands.models.ollama import OllamaModel
from strands.vended_plugins.skills import AgentSkills

from .recorder import TraceRecorder
from .skill_redirect import SkillNameRedirect
from .tools import current_time, fetch_url, read_feed, read_subreddit, write_report

# Generic on purpose. Instructions for a specific job belong in job files under jobs/, which git ignores.
SYSTEM_PROMPT = """You are a helpful assistant.
Use your tools when they help you answer. If you can't do something, say so plainly.
Skills listed under available_skills are not tools. To use one, call the `skills` tool with its name,
then follow the instructions it returns."""

# Each skill is a folder with a SKILL.md. skills/ holds generic ones and is committed;
# jobs/skills/ is for your own, and git ignores it. Paths are relative to where you run the agent.
SKILL_DIRS = ["skills", "jobs/skills"]


def build_agent(recorder: TraceRecorder) -> Agent:
    model = OllamaModel(
        host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        model_id=os.environ.get("AGENT_MODEL", "gemma4:e2b"),
        temperature=0.2,
        # Ollama's context window: prompt plus reply, in tokens. 32k fits a day of subreddit posts and
        # comments; on gemma4:e2b it barely changed memory use on a 16 GB Mac. Ollama only applies
        # this when it loads the model: if `ollama ps` shows a smaller CONTEXT, run `ollama stop <model>`.
        options={"num_ctx": 32768},
    )
    # The plugin lists each skill's name and description in the system prompt, and adds a
    # `skills` tool the model calls to load a skill's full instructions when it needs them.
    skills = AgentSkills(skills=[d for d in SKILL_DIRS if Path(d).is_dir()])
    return Agent(
        model=model,
        tools=[current_time, fetch_url, read_feed, read_subreddit, write_report],
        system_prompt=SYSTEM_PROMPT,
        hooks=[recorder, SkillNameRedirect(skills)],
        plugins=[skills],
    )
