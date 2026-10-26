from types import SimpleNamespace

from agent_v3.empty_reply_retry import MAX_RETRIES, EmptyReplyRetry


def reply(*content):
    return SimpleNamespace(stop_response=SimpleNamespace(message={"role": "assistant", "content": list(content)}), retry=False)


def test_asks_again_after_an_empty_reply_but_not_forever():
    hook = EmptyReplyRetry()
    retries = []
    for _ in range(MAX_RETRIES + 1):
        event = reply()
        hook.retry_if_empty(event)
        retries.append(event.retry)
    assert retries == [True] * MAX_RETRIES + [False]


def test_keeps_replies_with_text_or_a_tool_call():
    hook = EmptyReplyRetry()
    for event in (reply({"text": "Done."}), reply({"toolUse": {"name": "write_report"}}), reply({"text": "  "}, {"toolUse": {}})):
        hook.retry_if_empty(event)
        assert event.retry is False


def test_whitespace_alone_is_empty():
    event = reply({"text": " \n"})
    EmptyReplyRetry().retry_if_empty(event)
    assert event.retry is True
