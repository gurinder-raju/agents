"""Harness fix-up: when the model's reply is empty or cut off, ask it again.

Two ways a small local model's turn can come back with nothing usable:
- It breaks the JSON of a tool call. Ollama can't parse it and drops it without an error ("gemma4
  tool call parsing failed" in Ollama's server log), so the reply has no text and no tool call.
- It loops, writing the same text over and over, until it hits the reply cap (max_tokens in
  agent.py). gemma4:12b once did this for 60,000 tokens and 30 minutes before the cap existed.

An AfterModelCallEvent hook may set `retry`, which discards the reply and calls the model again.
Strands runs these hooks before it acts on a cut-off reply, so a retry here replaces it.
"""

import sys
from typing import Any

from strands.hooks import AfterModelCallEvent, HookProvider, HookRegistry
from strands.types.content import Message

MAX_RETRIES = 2


class ReplyRetry(HookProvider):
    def __init__(self) -> None:
        self._retries = 0  # in a row: a usable reply resets the count

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(AfterModelCallEvent, self.retry_if_unusable)

    def retry_if_unusable(self, event: AfterModelCallEvent) -> None:
        if event.stop_response is None:
            return  # the call failed: Strands reports that
        cut_off = event.stop_response.stop_reason == "max_tokens"
        if (cut_off or is_empty(event.stop_response.message)) and self._retries < MAX_RETRIES:
            self._retries += 1
            reason = "was cut off at the reply limit" if cut_off else "replied with nothing"
            print(f"model {reason}; asking again", file=sys.stderr, flush=True)
            event.retry = True
        else:
            self._retries = 0


def is_empty(message: Message) -> bool:
    return not any("toolUse" in block or block.get("text", "").strip() for block in message["content"])
