"""A stand-in for Ollama, so a whole run can go through the real agent loop with no model and no network."""

import json
from typing import Any

from strands.models import Model


class ScriptedModel(Model):
    """Each model call plays the next reply: text, a (tool name, arguments) call, or an exception to raise."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError
        yield

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.requests.append({"messages": list(messages), "tool_specs": tool_specs, "system_prompt": system_prompt})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        yield {"messageStart": {"role": "assistant"}}
        if isinstance(reply, str):
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": reply}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        else:
            name, arguments = reply
            yield {"contentBlockStart": {"start": {"toolUse": {"name": name, "toolUseId": f"tooluse_{len(self.requests)}"}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(arguments)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}}
