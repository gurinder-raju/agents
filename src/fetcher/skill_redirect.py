"""Harness fix-up: when the model calls a skill's name as if it were a tool, do what it meant.

Small models see `reddit-new-posts` listed under available_skills and often call it directly, which
fails with "Unknown tool". A BeforeToolCallEvent hook may rewrite a tool call before it runs, so this
turns `reddit-new-posts(...)` into `skills(skill_name="reddit-new-posts")`.

Sometimes the model has a tool in mind and only borrows a skill's name: gemma4:e4b once called
`link-report(title=..., criteria=..., items=[...])` to save its report. When the arguments are exactly
one tool's, that tool runs with them. Otherwise the skill loads, and if the call had arguments, a note
on the result says they went unused, so the model doesn't mistake the skill's instructions for its
finished work.
"""

from typing import Any

from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent, HookProvider, HookRegistry
from strands.vended_plugins.skills import AgentSkills


class SkillNameRedirect(HookProvider):
    def __init__(self, skills: AgentSkills) -> None:
        self._skills = skills
        self._unused = {}  # toolUseId -> the skill name a call with unused arguments was made to

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self.redirect)
        registry.add_callback(AfterToolCallEvent, self.note_unused_arguments)

    def redirect(self, event: BeforeToolCallEvent) -> None:
        name = event.tool_use["name"]
        if event.selected_tool is not None:
            return  # a real tool by that name exists
        if name not in {skill.name for skill in self._skills.get_available_skills(event.agent)}:
            return
        arguments = event.tool_use.get("input")
        if tool := _tool_taking(event.agent, arguments):
            event.tool_use = {**event.tool_use, "name": tool}
            return
        if arguments:
            self._unused[event.tool_use["toolUseId"]] = name
        event.tool_use = {**event.tool_use, "name": "skills", "input": {"skill_name": name}}

    def note_unused_arguments(self, event: AfterToolCallEvent) -> None:
        name = self._unused.pop(event.tool_use["toolUseId"], None)
        if name:
            note = (
                f"{name} is a skill, not a tool, so the arguments you gave it were not used and nothing has "
                "been done with them yet. Its instructions are above: follow them, calling the tools they name."
            )
            event.result = {**event.result, "content": [*event.result["content"], {"text": note}]}


def _tool_taking(agent: Any, arguments: Any) -> str | None:
    """The one tool these arguments fit: all the parameters it requires, and none it doesn't have."""
    if not isinstance(arguments, dict) or not arguments:
        return None
    fits = []
    for spec in agent.tool_registry.get_all_tool_specs():
        schema = spec.get("inputSchema", {}).get("json", {})
        if set(schema.get("required", [])) <= set(arguments) <= set(schema.get("properties", {})):
            fits.append(spec["name"])
    return fits[0] if len(fits) == 1 else None
