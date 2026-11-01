"""The agent: which model it calls, which tools it has, and how it is told to behave."""

import hashlib
import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ollama
from strands import Agent
from strands.models.ollama import OllamaModel
from strands.vended_plugins.skills import AgentSkills

from .recorder import TraceRecorder
from .reply_retry import ReplyRetry
from .skill_redirect import SkillNameRedirect
from .tools import current_time, fetch_url, read_feed, read_subreddit, report_tool

# Generic on purpose. Instructions for a specific job belong in job files under jobs/, which git ignores.
SYSTEM_PROMPT = """You are a helpful assistant.
Use your tools when they help you answer. If you can't do something, say so plainly.
When the user asks for a report, save it with write_report, even when nothing matched. A list in
your reply isn't a report.
Skills listed under available_skills are not tools. If one fits the task, call the `skills` tool with
its name before anything else, then follow the instructions it returns."""

# Added for a job that must end with a saved report, which the command line then makes sure of.
REPORT_REQUIRED = """

This job must end with a saved report. When you've finished with the sources, call write_report once,
with every match, before your final reply. Call it even if nothing matched or a source couldn't be
read: the report shows the user what was checked."""

# Each skill is a folder with a SKILL.md. skills/ holds generic ones and is committed;
# jobs/skills/ is for your own, and git ignores it. Paths are relative to where you run the agent.
SKILL_DIRS = ["skills", "jobs/skills"]
# Your own tools, which git also ignores. See load_local_tools.
LOCAL_TOOL_DIR = Path("jobs/tools")


def build_agent(recorder: TraceRecorder, prompt: str = "", report_required: bool = False) -> Agent:
    """Build the agent for one run. `prompt` picks the report items its model is shown: see report_tool."""
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
    local_tools, text_transforms, group_finders = load_local_tools()
    agent = Agent(
        model=model,
        tools=[current_time, fetch_url, read_feed, read_subreddit, report_tool(prompt), *local_tools],
        system_prompt=SYSTEM_PROMPT + (REPORT_REQUIRED if report_required else ""),
        # Registered after the recorder, so it runs first after a model call and the trace shows its retries.
        hooks=[recorder, SkillNameRedirect(skills), ReplyRetry()],
        plugins=[skills],
    )
    # The read tools run these on every post, comment and feed entry before the model reads them.
    agent.text_transforms = text_transforms
    agent.group_finders = group_finders
    agent.loaded_skill_instructions = []
    if report_required:
        report_skill = next((skill for skill in skills.get_available_skills(agent) if skill.name == "link-report"), None)
        if report_skill is not None:
            agent.system_prompt += (
                "\n\nThe link-report skill is already loaded below; follow its instructions.\n"
                f"<loaded_skill name=\"link-report\">\n{report_skill.instructions}\n</loaded_skill>"
            )
            agent.loaded_skill_instructions.append({
                "name": report_skill.name,
                "sha256": hashlib.sha256(report_skill.instructions.encode("utf-8")).hexdigest(),
            })
    return agent


def load_local_tools(folder: Path = LOCAL_TOOL_DIR) -> tuple[list[Any], list[Callable[[str], str]], list[Callable[[str], list[dict]]]]:
    """Load your own tools from jobs/tools/, which git ignores, so they never reach the repository.

    Each .py file there may define TOOLS, a list of @tool functions to give the model;
    transform_text(text), which rewrites every post, comment and feed entry before the model reads
    it: for example, decoding text that a community hides in a code the model can't read reliably;
    and find_groups(text), which lists what the transform marked in a rewritten text, so a job with a
    link pattern can account for each one. Each group is a dict with text, lines (the text's own
    line breaks, as a list), and start and end offsets.
    Files whose names start with _ or test_ are skipped. A file that fails to load is skipped with a warning.
    """
    tools: list[Any] = []
    transforms: list[Callable[[str], str]] = []
    finders: list[Callable[[str], list[dict]]] = []
    for path in sorted(folder.glob("*.py")) if folder.is_dir() else []:
        if path.name.startswith(("_", "test_")):
            continue
        spec = importlib.util.spec_from_file_location(f"fetcher_local_tools.{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            print(f"skipping local tool file {path}: {error}", file=sys.stderr)
            continue
        tools += list(getattr(module, "TOOLS", []))
        if callable(transform := getattr(module, "transform_text", None)):
            transforms.append(transform)
        if callable(finder := getattr(module, "find_groups", None)):
            finders.append(finder)
    return tools, transforms, finders


def unload_model(agent: Agent) -> None:
    """Ask Ollama to drop the model from memory now, instead of after its default 5 idle minutes.

    gemma4:e4b holds about 11 GB while it's loaded, most of a 16 GB Mac. This is what `ollama stop` does.
    """
    model = agent.model
    try:
        ollama.Client(model.host, **model.client_args).generate(model=model.config["model_id"], keep_alive=0)
    except Exception as error:  # Ollama isn't running, or the model is already gone: nothing to free
        print(f"could not unload {model.config['model_id']}: {error}", file=sys.stderr)
