"""Harness fix-up: when the model calls a skill's name as if it were a tool, load the skill instead.

Small models see `reddit-new-posts` listed under available_skills and often call it directly, which
fails with "Unknown tool". A BeforeToolCallEvent hook may rewrite a tool call before it runs, so this
turns `reddit-new-posts(...)` into `skills(skill_name="reddit-new-posts")`.
"""

from typing import Any

from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry
from strands.vended_plugins.skills import AgentSkills


class SkillNameRedirect(HookProvider):
    def __init__(self, skills: AgentSkills) -> None:
        self._skills = skills

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self.redirect)

    def redirect(self, event: BeforeToolCallEvent) -> None:
        name = event.tool_use["name"]
        if event.selected_tool is not None:
            return  # a real tool by that name exists
        if name in {skill.name for skill in self._skills.get_available_skills(event.agent)}:
            event.tool_use = {**event.tool_use, "name": "skills", "input": {"skill_name": name}}
