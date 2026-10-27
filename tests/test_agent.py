from strands.vended_plugins.skills import Skill

from fetcher import agent as agent_module
from fetcher.agent import build_agent, unload_model
from fetcher.recorder import TraceRecorder

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


def test_unloading_asks_ollama_to_drop_the_model_now(tmp_path, monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, host, **kwargs):
            calls.append(("host", host))

        def generate(self, **request):
            calls.append(("generate", request))

    monkeypatch.setattr(agent_module.ollama, "Client", FakeClient)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    unload_model(build_agent(TraceRecorder(tmp_path)))
    assert calls[-1] == ("generate", {"model": "gemma4:e4b", "keep_alive": 0})


def test_unloading_when_ollama_is_down_only_warns(tmp_path, monkeypatch, capsys):
    class DownClient:
        def __init__(self, host, **kwargs):
            pass

        def generate(self, **request):
            raise ConnectionError("Ollama isn't running")

    monkeypatch.setattr(agent_module.ollama, "Client", DownClient)
    unload_model(build_agent(TraceRecorder(tmp_path)))
    assert "could not unload" in capsys.readouterr().err


def test_loads_local_tools_and_text_transforms(tmp_path):
    (tmp_path / "shout.py").write_text(
        "from strands import tool\n\n"
        "@tool\ndef shout(text: str) -> str:\n    \"\"\"Upper-case text.\n\n    Args:\n        text: Any text.\n    \"\"\"\n    return text.upper()\n\n"
        "TOOLS = [shout]\n\n"
        "def transform_text(text):\n    return text.replace('a', 'A')\n"
    )
    (tmp_path / "broken.py").write_text("raise ImportError('missing dependency')\n")
    (tmp_path / "test_shout.py").write_text("raise AssertionError('tests are not tools')\n")

    tools, transforms = agent_module.load_local_tools(tmp_path)
    assert [t.tool_name for t in tools] == ["shout"]
    assert [transform("banana") for transform in transforms] == ["bAnAnA"]


def test_no_local_tools_folder_means_no_local_tools(tmp_path):
    assert agent_module.load_local_tools(tmp_path / "missing") == ([], [])
