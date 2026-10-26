"""Step recorder: writes every step of a run to traces/<run_id>.jsonl, one JSON object per line.

Strands calls these hooks as its agent loop runs, so the trace shows the order things happen in:
the prompt arriving, each model call and what came back, each tool call and its result, and the end.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from strands.hooks import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)


class TraceRecorder(HookProvider):
    def __init__(self, trace_dir: str | Path = "traces") -> None:
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.path = Path(trace_dir) / f"{self.run_id}.jsonl"
        self._step = 0
        self._model_started = 0.0

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeInvocationEvent, self.on_invocation_start)
        registry.add_callback(BeforeModelCallEvent, self.on_model_request)
        registry.add_callback(AfterModelCallEvent, self.on_model_response)
        registry.add_callback(BeforeToolCallEvent, self.on_tool_call)
        registry.add_callback(AfterToolCallEvent, self.on_tool_result)
        registry.add_callback(AfterInvocationEvent, self.on_invocation_end)

    def on_invocation_start(self, event: BeforeInvocationEvent) -> None:
        self._write("invocation_start", input=event.messages)

    def on_model_request(self, event: BeforeModelCallEvent) -> None:
        # The model keeps no memory between calls: the harness sends the whole conversation every time.
        # The system prompt and tool specs are read here, not at invocation start, because plugins
        # such as AgentSkills can change them in their own invocation-start hooks.
        self._model_started = time.perf_counter()
        self._write(
            "model_request",
            system_prompt=event.agent.system_prompt,
            tool_specs=event.agent.tool_registry.get_all_tool_specs(),
            message_count=len(event.agent.messages),
            messages=event.agent.messages,
        )

    def on_model_response(self, event: AfterModelCallEvent) -> None:
        ms = round((time.perf_counter() - self._model_started) * 1000)
        if event.stop_response is None:
            self._write("model_error", ms=ms, error=repr(event.exception))
            return
        self._write(
            "model_response",
            ms=ms,
            stop_reason=event.stop_response.stop_reason,
            message=event.stop_response.message,
            retried=event.retry,  # true when a hook discarded this reply and asked the model again
        )

    def on_tool_call(self, event: BeforeToolCallEvent) -> None:
        self._write(
            "tool_call",
            tool=event.tool_use["name"],
            tool_use_id=event.tool_use["toolUseId"],
            input=event.tool_use["input"],
        )

    def on_tool_result(self, event: AfterToolCallEvent) -> None:
        self._write(
            "tool_result",
            tool=event.tool_use["name"],
            tool_use_id=event.tool_use["toolUseId"],
            ms=round(event.duration * 1000) if event.duration is not None else None,
            status=event.result["status"],
            content=event.result["content"],
        )

    def on_invocation_end(self, event: AfterInvocationEvent) -> None:
        result = event.result
        self._write(
            "invocation_end",
            stop_reason=result.stop_reason if result else None,
            usage=result.metrics.accumulated_usage if result else None,
        )

    def _write(self, kind: str, **fields: Any) -> None:
        self._step += 1
        record = {
            "run_id": self.run_id,
            "step": self._step,
            "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "kind": kind,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
