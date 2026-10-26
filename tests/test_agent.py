from strands.vended_plugins.skills import Skill

from agent_v3.agent import build_agent
from agent_v3.recorder import TraceRecorder

# These build the real agent and read the real skill files without calling the model.
# Run pytest from the repo folder so skills/ is found.


def test_agent_has_the_tools_and_the_skills_tool(tmp_path):
    agent = build_agent(TraceRecorder(tmp_path))
    assert {"current_time", "fetch_url", "read_feed", "read_subreddit", "write_report", "skills"} <= set(agent.tool_registry.registry)


def test_reddit_skill_parses(tmp_path):
    skill = Skill.from_file("skills/reddit-new-posts", strict=True)
    assert skill.name == "reddit-new-posts"
    assert skill.allowed_tools == ["read_subreddit"]
    assert "read_subreddit" in skill.instructions



def test_link_report_skill_parses():
    skill = Skill.from_file("skills/link-report", strict=True)
    assert skill.name == "link-report"
    assert "write_report" in skill.allowed_tools
    assert "write_report" in skill.instructions
