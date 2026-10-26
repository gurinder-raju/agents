from types import SimpleNamespace

from fetcher.reply_retry import MAX_RETRIES, ReplyRetry


def reply(*content, stop_reason="end_turn"):
    message = {"role": "assistant", "content": list(content)}
    return SimpleNamespace(stop_response=SimpleNamespace(message=message, stop_reason=stop_reason), retry=False)


def test_asks_again_after_an_empty_reply_but_not_forever():
    hook = ReplyRetry()
    retries = []
    for _ in range(MAX_RETRIES + 1):
        event = reply()
        hook.retry_if_unusable(event)
        retries.append(event.retry)
    assert retries == [True] * MAX_RETRIES + [False]


def test_asks_again_when_a_reply_is_cut_off_at_the_limit():
    event = reply({"text": "and so on and so on and so on"}, stop_reason="max_tokens")
    ReplyRetry().retry_if_unusable(event)
    assert event.retry is True


def test_keeps_replies_with_text_or_a_tool_call():
    hook = ReplyRetry()
    for event in (reply({"text": "Done."}), reply({"toolUse": {"name": "write_report"}}), reply({"text": "  "}, {"toolUse": {}})):
        hook.retry_if_unusable(event)
        assert event.retry is False


def test_whitespace_alone_is_empty():
    event = reply({"text": " \n"})
    ReplyRetry().retry_if_unusable(event)
    assert event.retry is True
