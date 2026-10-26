from datetime import datetime

from hello_agent.tools import current_time


def test_current_time_returns_iso_timestamp_in_the_given_zone():
    stamp = current_time(timezone="Asia/Kolkata")
    assert datetime.fromisoformat(stamp).utcoffset().total_seconds() == 5.5 * 3600


def test_current_time_has_a_tool_spec_built_from_the_docstring():
    spec = current_time.tool_spec
    assert spec["name"] == "current_time"
    assert "timezone" in spec["inputSchema"]["json"]["properties"]
