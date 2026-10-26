from types import SimpleNamespace

from strands.vended_plugins.skills import AgentSkills, Skill

from fetcher.skill_redirect import SkillNameRedirect

SKILLS = AgentSkills(skills=[Skill(name="reddit-new-posts", description="Read new posts.", instructions="Steps.")])


def tool_call(name, selected_tool=None):
    return SimpleNamespace(
        agent=None,
        selected_tool=selected_tool,
        tool_use={"toolUseId": "tooluse_1", "name": name, "input": {"subreddit": "Python"}},
    )


def test_a_skill_called_as_a_tool_becomes_a_skills_call():
    event = tool_call("reddit-new-posts")
    SkillNameRedirect(SKILLS).redirect(event)
    assert event.tool_use == {"toolUseId": "tooluse_1", "name": "skills", "input": {"skill_name": "reddit-new-posts"}}


def test_unknown_names_that_are_not_skills_are_left_alone():
    event = tool_call("search_web")
    SkillNameRedirect(SKILLS).redirect(event)
    assert event.tool_use["name"] == "search_web"


def test_real_tools_are_left_alone():
    event = tool_call("reddit-new-posts", selected_tool=object())
    SkillNameRedirect(SKILLS).redirect(event)
    assert event.tool_use["name"] == "reddit-new-posts"
