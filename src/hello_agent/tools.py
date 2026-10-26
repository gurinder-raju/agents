"""Tools the agent can call.

Strands turns each @tool function into a tool spec for the model: the function name becomes the
tool name, the docstring becomes its description, and the typed arguments become its JSON schema.
The model never runs these. It asks for a call, the harness runs the function and sends back the result.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from strands import tool


@tool
def current_time(timezone: str = "UTC") -> str:
    """Get the current date and time.

    Args:
        timezone: IANA timezone name, for example "America/Los_Angeles". Defaults to UTC.
    """
    return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")
