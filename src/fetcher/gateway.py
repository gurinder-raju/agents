"""Read approved sources for the harness without exposing provider credentials."""

import json
import re
import time
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .credentials import CredentialStore, CredentialUnavailable, KeychainStore, SecretsManagerStore
from .report import canonical_link, url_problem

SCHEMA_VERSION = 1
SUBREDDIT_NAME = re.compile(r"[A-Za-z0-9_]{2,21}")
MAX_HOURS = 168
MAX_REQUEST_BYTES = 4096
MAX_RESPONSE_BYTES = 2_000_000

ERRORS = {
    "invalid_request": "invalid gateway request",
    "source_denied": "the source is not configured in the gateway",
    "credential_unavailable": "the gateway could not load a usable provider credential",
    "provider_denied": "the provider rejected the gateway credential or denied access",
    "rate_limited": "the provider is rate limiting reads",
    "source_unavailable": "the provider or its source endpoint is unavailable",
    "invalid_response": "the source returned invalid or oversized data",
    "gateway_unavailable": "the gateway could not complete the read",
}


class GatewayError(RuntimeError):
    """One of a fixed set of messages safe for the tool result and trace."""

    def __init__(self, code: str) -> None:
        self.code = code if code in ERRORS else "gateway_unavailable"
        super().__init__(f"{self.code}: {ERRORS[self.code]}")


@dataclass(frozen=True)
class Source:
    name: str
    max_hours: float
    fixture: Path | None = None
    endpoint: str | None = None
    credential: str | None = None


class Gateway:
    def __init__(self, sources: dict[str, Source], credentials: CredentialStore | None = None) -> None:
        self.sources = sources
        self.credentials = credentials

    def read(self, request: Any) -> dict[str, Any]:
        try:
            return self._read(request)
        except GatewayError as error:
            return error_reply(error.code)
        except Exception:
            # Upstream responses and SDK or credential exceptions may contain secrets.
            return error_reply("gateway_unavailable")

    def _read(self, request: Any) -> dict[str, Any]:
        if not _has_version(request) or request.keys() != {"schema_version", "source", "since_hours"}:
            raise GatewayError("invalid_request")
        name, hours = request["source"], request["since_hours"]
        if not isinstance(name, str) or not SUBREDDIT_NAME.fullmatch(name):
            raise GatewayError("invalid_request")
        source = self.sources.get(name.lower())
        if source is None:
            raise GatewayError("source_denied")
        if not _valid_hours(hours, source.max_hours):
            raise GatewayError("invalid_request")

        end = datetime.now(UTC)
        start = end - timedelta(hours=hours)
        if source.fixture is not None:
            try:
                with source.fixture.open("rb") as file:
                    data = decode_reply(file.read(MAX_RESPONSE_BYTES + 1))
            except OSError:
                raise GatewayError("source_unavailable") from None
            result = normalize_snapshot(data)
        else:
            result = self._read_devvit(source, start, end)
        result["posts"] = [
            post for post in result["posts"]
            if start <= datetime.fromisoformat(post["published"]) <= end
        ]
        result["posts"].sort(key=lambda post: post["published"], reverse=True)
        return result

    def _read_devvit(self, source: Source, start: datetime, end: datetime) -> dict[str, Any]:
        if self.credentials is None or source.credential is None or source.endpoint is None:
            raise GatewayError("credential_unavailable")
        try:
            token = self.credentials.get(source.credential)
        except CredentialUnavailable:
            raise GatewayError("credential_unavailable") from None
        if not re.fullmatch(r"devvit_at_[-A-Za-z0-9._~+/]+=*", token):
            raise GatewayError("credential_unavailable")

        deadline = time.monotonic() + 20
        try:
            with httpx.Client(timeout=httpx.Timeout(10, connect=5), follow_redirects=False, trust_env=False) as client:
                with client.stream(
                    "POST",
                    source.endpoint,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/json",
                        "Accept-Encoding": "identity",
                    },
                    json={
                        "schema_version": SCHEMA_VERSION, "source": source.name,
                        "start_at": start.isoformat(), "end_at": end.isoformat(),
                    },
                ) as response:
                    if response.status_code in (401, 403):
                        raise GatewayError("provider_denied")
                    if response.status_code == 429:
                        raise GatewayError("rate_limited")
                    if response.status_code != 200:
                        raise GatewayError("source_unavailable")
                    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                        raise GatewayError("invalid_response")
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        if time.monotonic() > deadline:
                            raise GatewayError("source_unavailable")
                        if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                            raise GatewayError("invalid_response")
                        body.extend(chunk)
        except httpx.HTTPError:
            raise GatewayError("source_unavailable") from None
        result = normalize_snapshot(decode_reply(body))
        if token in json.dumps(result, ensure_ascii=False):
            raise GatewayError("invalid_response")
        return result


def load_gateway(path: Path) -> Gateway:
    """Load trusted operator settings. Credential references identify items in the selected store."""
    try:
        with path.open("rb") as file:
            config = tomllib.load(file)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        raise ValueError("Cannot read gateway configuration. Check the path and TOML syntax.") from None
    if config.keys() - {"credential_store", "keychain_service", "keychain_path", "sources"}:
        raise ValueError("Unrecognized gateway configuration field.")
    store = config.get("credential_store")
    if store not in (None, "keychain", "secretsmanager"):
        raise ValueError("Use keychain or secretsmanager for credential_store.")
    credentials: CredentialStore | None = None
    if store == "keychain":
        service = config.get("keychain_service", "fetcher.gateway")
        keychain = config.get("keychain_path")
        if not _nonempty_string(service) or (keychain is not None and not _nonempty_string(keychain)):
            raise ValueError("Keychain settings must be nonempty strings.")
        credentials = KeychainStore(service, path.parent / Path(keychain).expanduser() if keychain else None)
    elif store == "secretsmanager":
        credentials = SecretsManagerStore()

    configured = config.get("sources")
    if not isinstance(configured, dict) or not configured:
        raise ValueError("Configure at least one named source.")
    sources = {}
    for name, entry in configured.items():
        if not SUBREDDIT_NAME.fullmatch(name) or name.lower() in sources or not isinstance(entry, dict):
            raise ValueError("Use unique subreddit names for source tables.")
        max_hours = entry.get("max_hours", MAX_HOURS)
        if not _valid_hours(max_hours):
            raise ValueError("Source max_hours must be greater than 0 and at most 168.")
        provider = entry.get("provider")
        if provider == "fixture" and not entry.keys() - {"provider", "max_hours", "path"}:
            if not _nonempty_string(entry.get("path")):
                raise ValueError("A fixture source needs a local path.")
            source = Source(name, max_hours, fixture=path.parent / Path(entry["path"]).expanduser())
        elif provider == "devvit" and not entry.keys() - {"provider", "max_hours", "endpoint", "credential"}:
            if credentials is None or not _nonempty_string(entry.get("credential")):
                raise ValueError("A Devvit source needs a credential reference and a credential store.")
            if not _devvit_endpoint(entry.get("endpoint")):
                raise ValueError("Configure an HTTPS Devvit external endpoint for the source.")
            source = Source(name, max_hours, endpoint=entry["endpoint"], credential=entry["credential"])
        else:
            raise ValueError("Configure a fixture or devvit source with its documented fields.")
        sources[name.lower()] = source
    return Gateway(sources, credentials)


def _devvit_endpoint(value: Any) -> bool:
    if not isinstance(value, str) or url_problem(value):
        return False
    url = urlsplit(value)
    return (
        url.scheme == "https"
        and re.fullmatch(r"[a-z0-9][a-z0-9-]*-external\.devvit\.net", url.hostname) is not None
        and url.port in (None, 443)
        and url.username is None
        and not (url.query or url.fragment)
        and re.fullmatch(r"/external/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*/?", url.path) is not None
    )


def _valid_hours(value: Any, maximum: float = MAX_HOURS) -> bool:
    return type(value) in (int, float) and 0 < value <= maximum


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and "\x00" not in value


def _has_version(value: Any) -> bool:
    return isinstance(value, dict) and type(value.get("schema_version")) is int and value["schema_version"] == SCHEMA_VERSION


def error_reply(code: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "error": code}


def decode_reply(data: bytes | bytearray) -> Any:
    if len(data) > MAX_RESPONSE_BYTES:
        raise GatewayError("invalid_response")
    try:
        return json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError):
        raise GatewayError("invalid_response") from None


def read_reply(data: Any) -> dict[str, Any]:
    if _has_version(data) and isinstance(data.get("error"), str):
        raise GatewayError(data["error"])
    return normalize_snapshot(data)


def normalize_snapshot(data: Any) -> dict[str, Any]:
    """Project provider data onto the fields present_posts uses and check both completeness flags."""
    if (
        not _has_version(data) or "error" in data or not isinstance(data.get("posts"), list)
        or type(data.get("posts_complete")) is not bool or type(data.get("comments_complete")) is not bool
    ):
        raise GatewayError("invalid_response")
    posts = []
    links = set()
    for row in data["posts"]:
        post = _normalize_post(row)
        key = canonical_link(post["link"])
        if key in links:
            raise GatewayError("invalid_response")
        links.add(key)
        posts.append(post)
    return {
        "schema_version": SCHEMA_VERSION,
        "posts": posts,
        "posts_complete": data["posts_complete"],
        "comments_complete": data["comments_complete"],
    }


def _normalize_post(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict) or not isinstance(row.get("comments"), list):
        raise GatewayError("invalid_response")
    title, text, link, published = (_text(row.get(key)) for key in ("title", "text", "link", "published"))
    linked_url = _optional_text(row.get("url"))
    if url_problem(link) or (linked_url is not None and url_problem(linked_url)):
        raise GatewayError("invalid_response")
    try:
        timestamp = datetime.fromisoformat(published)
        if timestamp.tzinfo is None:
            raise ValueError
        published = timestamp.astimezone(UTC).isoformat(timespec="microseconds")
    except (ValueError, OverflowError):
        raise GatewayError("invalid_response") from None
    comments = []
    for comment in row["comments"]:
        if not isinstance(comment, dict):
            raise GatewayError("invalid_response")
        comments.append({"author": _optional_text(comment.get("author")), "text": _text(comment.get("text"))})
    return {
        "title": title, "author": _optional_text(row.get("author")), "text": text,
        "published": published, "link": link, "url": linked_url,
        "comment_count": len(comments), "comments": comments,
    }


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise GatewayError("invalid_response")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise GatewayError("invalid_response") from None
    return value


def _optional_text(value: Any) -> str | None:
    return None if value is None else _text(value)
