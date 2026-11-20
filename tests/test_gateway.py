import io
import json
import stat
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Thread

import httpx
import pytest
from botocore.response import StreamingBody

from fetcher import agent, cli, credentials, gateway, gateway_transport, tools
from fetcher.credentials import CredentialUnavailable, KeychainStore
from fetcher.gateway import GatewayError, load_gateway
from fetcher.gateway_transport import lambda_handler, local_gateway, read_at_lambda, read_at_socket
from scripted import ScriptedModel


SOURCE = "Example"
TOKEN = "devvit_at_synthetic_test_token"
ENDPOINT = "https://example-aaa123-external.devvit.net/external/v1/read"
EVIDENCE = "The example comment mentions a formatter."


def snapshot(*, comments_complete=True):
    return {
        "schema_version": 1,
        "posts_complete": True,
        "comments_complete": comments_complete,
        "posts": [{
            "id": "upstream-id",
            "title": "Example post",
            "link": "https://example.invalid/posts/one",
            "published": (datetime.now(UTC) - timedelta(minutes=5)).isoformat(),
            "text": "A discussion about tools.",
            "comments": [{"text": EVIDENCE}],
        }],
    }


def event(source=SOURCE, hours=24):
    return {"schema_version": 1, "source": source, "since_hours": hours}


def fixture_config(tmp_path, data=None):
    (tmp_path / "fixture.json").write_text(json.dumps(data if data is not None else snapshot()))
    config = tmp_path / "gateway.toml"
    config.write_text('[sources.Example]\nprovider = "fixture"\npath = "fixture.json"\nmax_hours = 24\n')
    return config


def devvit_config(tmp_path, endpoint=ENDPOINT):
    config = tmp_path / "gateway.toml"
    config.write_text(
        'credential_store = "secretsmanager"\n'
        '[sources.Example]\nprovider = "devvit"\ncredential = "test-reference"\n'
        f'endpoint = "{endpoint}"\nmax_hours = 24\n'
    )
    return config


def mock_provider(monkeypatch, handler):
    def build_client(**settings):
        assert settings["follow_redirects"] is False
        return real_client(transport=httpx.MockTransport(handler), **settings)

    real_client = httpx.Client
    monkeypatch.setattr(gateway.httpx, "Client", build_client)


def test_allowlist_and_time_limit_apply_before_provider_access(tmp_path, monkeypatch):
    configured = load_gateway(devvit_config(tmp_path))

    def unexpected(*args, **kwargs):
        pytest.fail("A denied request must never look up a secret")

    monkeypatch.setattr(configured.credentials, "get", unexpected)
    assert configured.read(event("Other"))["error"] == "source_denied"
    assert configured.read(event(hours=25))["error"] == "invalid_request"
    assert configured.read(event() | {"endpoint": ENDPOINT})["error"] == "invalid_request"
    assert configured.read(event() | {"credential": "other"})["error"] == "invalid_request"


@pytest.mark.parametrize("endpoint", [
    "https://example-aaa123-external.devvit.net.example.invalid/external/v1/read",
    "https://example-aaa123-external.devvit.net/api/read",
    "http://example-aaa123-external.devvit.net/external/v1/read",
    "https://example-aaa123-external.devvit.net/external/v1/read?key=value",
])
def test_refuses_unapproved_credential_destinations(tmp_path, endpoint):
    with pytest.raises(ValueError, match="HTTPS Devvit external endpoint"):
        load_gateway(devvit_config(tmp_path, endpoint))


def test_fixture_filters_window_and_preserves_incomplete_coverage(tmp_path):
    data = snapshot(comments_complete=False)
    data["posts"].append({
        **data["posts"][0], "link": "https://example.invalid/posts/old",
        "published": (datetime.now(UTC) - timedelta(days=2)).isoformat(),
    })
    reply = load_gateway(fixture_config(tmp_path, data)).read(event("example"))
    assert [post["link"] for post in reply["posts"]] == [data["posts"][0]["link"]]
    assert reply["posts_complete"] is True and reply["comments_complete"] is False
    assert reply["posts"][0]["comment_count"] == 1
    assert "id" not in reply["posts"][0]


@pytest.mark.parametrize(("status", "error"), [
    (403, "provider_denied"),
    (429, "rate_limited"),
    (404, "source_unavailable"),
    (307, "source_unavailable"),
    (200, "invalid_response"),
])
def test_provider_errors_and_echoed_credentials_stay_out_of_replies(tmp_path, monkeypatch, status, error):
    configured = load_gateway(devvit_config(tmp_path))
    monkeypatch.setattr(configured.credentials, "get", lambda name: TOKEN)
    data = snapshot()
    data["posts"][0]["text"] = TOKEN
    requests = []

    def provider(request):
        requests.append(request.url)
        return httpx.Response(status, json=data, headers={"Location": "https://example.invalid/"})

    mock_provider(monkeypatch, provider)
    assert configured.read(event()) == {"schema_version": 1, "error": error}
    assert requests == [httpx.URL(ENDPOINT)]


def test_keychain_errors_are_safe_for_the_gateway(monkeypatch):
    def security(command, **options):
        assert command[-5:] == ["-s", "fetcher.gateway", "-a", "test-reference", "-w"]
        assert options["capture_output"] is True and TOKEN not in command
        return subprocess.CompletedProcess(command, 0, stdout=TOKEN + "\n")

    monkeypatch.setattr(credentials.subprocess, "run", security)
    store = KeychainStore("fetcher.gateway")
    assert store.get("test-reference") == TOKEN

    def locked(command, **options):
        raise subprocess.CalledProcessError(1, command, stderr=TOKEN)

    monkeypatch.setattr(credentials.subprocess, "run", locked)
    with pytest.raises(CredentialUnavailable) as raised:
        store.get("test-reference")
    assert TOKEN not in str(raised.value)


def test_lambda_round_trip_uses_gateway_credential_and_returns_only_data(tmp_path, monkeypatch):
    monkeypatch.setenv("FETCHER_GATEWAY_CONFIG", str(devvit_config(tmp_path)))

    def provider(request):
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        body = json.loads(request.content)
        assert body["source"] == SOURCE
        assert datetime.fromisoformat(body["end_at"]) - datetime.fromisoformat(body["start_at"]) == timedelta(hours=24)
        return httpx.Response(200, json=snapshot() | {"extra": TOKEN})

    mock_provider(monkeypatch, provider)
    calls = []

    class AWS:
        def close(self):
            pass

        def invoke(self, **kwargs):
            assert kwargs["FunctionName"] == "test-gateway"
            assert kwargs["InvocationType"] == "RequestResponse"
            request = json.loads(kwargs["Payload"])
            assert request == event()
            payload = json.dumps(lambda_handler(request, None)).encode()
            return {"StatusCode": 200, "Payload": StreamingBody(io.BytesIO(payload), len(payload))}

        def get_secret_value(self, **kwargs):
            assert kwargs == {"SecretId": "test-reference"}
            return {"SecretString": TOKEN}

    def get_client(service):
        calls.append(service)
        return AWS()

    monkeypatch.setattr(credentials, "aws_client", get_client)
    monkeypatch.setattr(gateway_transport, "aws_client", get_client)
    reply = read_at_lambda("test-gateway", SOURCE, 24)
    assert reply["posts"][0]["comments"][0]["text"] == EVIDENCE
    assert calls == ["lambda", "secretsmanager"]
    assert TOKEN not in json.dumps(reply)


@pytest.mark.parametrize(("requested", "comments_complete", "expected_exit"), [
    (SOURCE, True, 0), (SOURCE, False, 3), ("Unconfigured", True, 1),
])
def test_unix_gateway_through_real_agent_cli(tmp_path, monkeypatch, requested, comments_complete, expected_exit):
    gateway_service = load_gateway(fixture_config(tmp_path, snapshot(comments_complete=comments_complete)))

    def no_direct_network(*args, **kwargs):
        pytest.fail("Gateway mode must not use direct fetch or report link checking")

    monkeypatch.setattr(tools, "http_get", no_direct_network)
    monkeypatch.setattr(tools, "verify_links", no_direct_network)
    monkeypatch.setattr(tools, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(cli, "unload_model", lambda agent: None)
    items = [{
        "post": f"{SOURCE}-1", "description": "It names a kind of tool.", "quote": EVIDENCE,
    }] if requested == SOURCE else []
    model = ScriptedModel(
        ("read_subreddit", {"subreddit": f"r/{requested}"}),
        ("write_report", {
            "title": "Example report", "criteria": "A mention of tools in the comments.",
            "items": items,
        }),
        "Report saved.",
    )
    monkeypatch.setattr(agent, "OllamaModel", lambda **settings: model)
    with tempfile.TemporaryDirectory(prefix="fetcher-gw-", dir="/tmp") as directory:
        path = Path(directory) / "g.sock"
        with local_gateway(path, gateway_service) as server:
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
            worker.start()
            try:
                with pytest.raises(GatewayError, match="source_denied"):
                    read_at_socket(path, "Other", 24)
                monkeypatch.setattr(sys, "argv", [
                    "fetcher", f"r/{requested}", "--gateway-socket", str(path), "--report-required",
                    "--report-json", "--trace-dir", str(tmp_path / "traces"),
                ])
                assert cli.main() == expected_exit
            finally:
                server.shutdown()
                worker.join()
        assert not path.exists()

    report = next((tmp_path / "reports").glob("*.json"))
    artifact = json.loads(report.read_text())
    tool_names = {spec["name"] for spec in model.requests[0]["tool_specs"]}
    assert {"read_subreddit", "write_report"} <= tool_names
    assert not {"fetch_url", "read_feed"} & tool_names
    if requested == SOURCE:
        assert artifact["sources"][0]["status"] == ("complete" if comments_complete else "incomplete")
        assert artifact["items"][0]["evidence"][0]["quote"] == EVIDENCE
    else:
        assert artifact["sources"][0]["status"] == "failed"
        assert "source_denied" in artifact["sources"][0]["line"]
        assert artifact["items"] == []
