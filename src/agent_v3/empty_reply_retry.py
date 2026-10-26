"""Harness fix-up: when the model stops with an empty reply, ask it again.

When a small model breaks the JSON of a tool call, Ollama can't parse it and drops it without an
error ("gemma4 tool call parsing failed" in Ollama's server log). The reply arrives with no text and
no tool call, and the run would end with nothing to show for it. An AfterModelCallEvent hook may set
`retry`, which discards the reply and calls the model again.
"""

import sys
from typing import Any

from strands.hooks import AfterModelCallEvent, HookProvider, HookRegistry
from strands.types.content import Message

MAX_RETRIES = 2


class EmptyReplyRetry(HookProvider):
    def __init__(self) -> None:
        self._retries = 0  # in a row: a reply with something in it resets the count

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(AfterModelCallEvent, self.retry_if_empty)

    def retry_if_empty(self, event: AfterModelCallEvent) -> None:
        if event.stop_response is None:
            return  # the call failed: Strands reports that
        if is_empty(event.stop_response.message) and self._retries < MAX_RETRIES:
            self._retries += 1
            print("model replied with nothing; asking again", file=sys.stderr, flush=True)
            event.retry = True
        else:
            self._retries = 0


def is_empty(message: Message) -> bool:
    return not any("toolUse" in block or block.get("text", "").strip() for block in message["content"])
