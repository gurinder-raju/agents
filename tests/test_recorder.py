import json
from types import SimpleNamespace

from fetcher.recorder import TraceRecorder

# Stand-ins for the Strands hook events: the recorder only reads these attributes.
TOOL_USE = {"name": "current_time", "toolUseId": "tooluse_1", "input": {"timezone": "UTC"}}


def test_writes_one_numbered_line_per_step(tmp_path):
    recorder = TraceRecorder(tmp_path)
    agent = SimpleNamespace(
        system_prompt="You are a helpful assistant.",
        tool_registry=SimpleNamespace(get_all_tool_specs=lambda: [{"name": "current_time"}]),
        messages=[{"role": "user", "content": [{"text": "What time is it in Tokyo?"}]}],
    )

    recorder.on_model_request(SimpleNamespace(agent=agent))
    recorder.on_model_response(
        SimpleNamespace(
            stop_response=SimpleNamespace(stop_reason="tool_use", message={"role": "assistant", "content": []}),
            retry=False,
            exception=None,
        )
    )
    recorder.on_tool_call(SimpleNamespace(tool_use=TOOL_USE))
    recorder.on_tool_result(
        SimpleNamespace(
            tool_use=TOOL_USE,
            duration=0.0021,
            result={"toolUseId": "tooluse_1", "status": "success", "content": [{"text": "2026-10-26T17:00:00+00:00"}]},
        )
    )

    lines = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [line["kind"] for line in lines] == ["model_request", "model_response", "tool_call", "tool_result"]
    assert [line["step"] for line in lines] == [1, 2, 3, 4]
    assert {line["run_id"] for line in lines} == {recorder.run_id}
    assert lines[0]["message_count"] == 1
    assert lines[3]["ms"] == 2


def test_records_model_errors(tmp_path):
    recorder = TraceRecorder(tmp_path)
    recorder.on_model_response(SimpleNamespace(stop_response=None, exception=ConnectionError("ollama not running")))

    line = json.loads(recorder.path.read_text())
    assert line["kind"] == "model_error"
    assert "ollama not running" in line["error"]
