from types import SimpleNamespace

from strands.vended_plugins.skills import AgentSkills, Skill

from fetcher.skill_redirect import SkillNameRedirect
from fetcher.tools import fetch_url, read_feed, read_subreddit, write_report

SKILLS = AgentSkills(
    skills=[
        Skill(name="reddit-new-posts", description="Read new posts.", instructions="Steps."),
        Skill(name="link-report", description="Build a link report.", instructions="Steps."),
    ]
)


class Agent:
    """Stands in for the agent, whose tool specs the redirect reads. A class, so it can be a dict key."""

    def __init__(self, *tools):
        self.tool_registry = SimpleNamespace(get_all_tool_specs=lambda: [tool.tool_spec for tool in tools])


AGENT = Agent(fetch_url, read_feed, read_subreddit, write_report)


def tool_call(name, arguments=None, selected_tool=None):
    return SimpleNamespace(
        agent=AGENT,
        selected_tool=selected_tool,
        tool_use={"toolUseId": "tooluse_1", "name": name, "input": arguments or {}},
    )


def run(hook, event, text="Skill instructions."):
    """Redirect the call, then return the result the model gets back from it."""
    hook.redirect(event)
    after = SimpleNamespace(tool_use=event.tool_use, result={"toolUseId": "tooluse_1", "status": "success", "content": [{"text": text}]})
    hook.note_unused_arguments(after)
    return after.result


def test_a_skill_called_as_a_tool_becomes_a_skills_call():
    event = tool_call("reddit-new-posts")
    SkillNameRedirect(SKILLS).redirect(event)
    assert event.tool_use == {"toolUseId": "tooluse_1", "name": "skills", "input": {"skill_name": "reddit-new-posts"}}


def test_a_skill_name_given_one_tools_arguments_runs_that_tool():
    report = {"title": "T", "criteria": "C", "items": [{"post": "Python-1"}], "sort": "comments"}
    event = tool_call("link-report", report)
    SkillNameRedirect(SKILLS).redirect(event)
    assert event.tool_use == {"toolUseId": "tooluse_1", "name": "write_report", "input": report}


def test_arguments_that_fit_no_tool_or_several_load_the_skill():
    for arguments in ({"topic": "Python"}, {"url": "https://example.com/feed"}, {"title": "T", "items": []}):
        event = tool_call("link-report", arguments)
        SkillNameRedirect(SKILLS).redirect(event)
        assert event.tool_use["name"] == "skills", arguments
        assert event.tool_use["input"] == {"skill_name": "link-report"}


def test_the_skill_says_when_the_calls_arguments_went_unused():
    result = run(SkillNameRedirect(SKILLS), tool_call("link-report", {"title": "T", "items": []}))
    assert result["content"][0] == {"text": "Skill instructions."}
    assert "were not used" in result["content"][1]["text"]


def test_a_plain_skill_load_gets_no_note():
    result = run(SkillNameRedirect(SKILLS), tool_call("link-report"))
    assert result["content"] == [{"text": "Skill instructions."}]


def test_unknown_names_that_are_not_skills_are_left_alone():
    event = tool_call("search_web", {"subreddit": "Python"})
    SkillNameRedirect(SKILLS).redirect(event)
    assert event.tool_use["name"] == "search_web"


def test_real_tools_are_left_alone():
    event = tool_call("reddit-new-posts", {"subreddit": "Python"}, selected_tool=object())
    SkillNameRedirect(SKILLS).redirect(event)
    assert event.tool_use["name"] == "reddit-new-posts"
