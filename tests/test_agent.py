from strands.vended_plugins.skills import Skill

from hello_agent.agent import build_agent
from hello_agent.recorder import TraceRecorder

# These build the real agent and read the real skill files without calling the model.
# Run pytest from the repo folder so skills/ is found.


def test_agent_has_the_tools_and_the_skills_tool(tmp_path):
    agent = build_agent(TraceRecorder(tmp_path))
    assert {"current_time", "fetch_url", "skills"} <= set(agent.tool_registry.registry)


def test_summarize_page_skill_parses(tmp_path):
    skill = Skill.from_file("skills/summarize-page", strict=True)
    assert skill.name == "summarize-page"
    assert skill.allowed_tools == ["fetch_url"]
    assert "fetch_url" in skill.instructions
