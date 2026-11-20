"""Use the same gateway protocol over a local Unix socket or synchronous Lambda invocation."""

import argparse
import grp
import json
import os
import signal
import socket
import socketserver
import stat
from collections.abc import Iterator
from contextlib import closing, contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from .credentials import aws_client
from .gateway import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    SCHEMA_VERSION,
    Gateway,
    GatewayError,
    decode_reply,
    error_reply,
    load_gateway,
    read_reply,
)


def read_at_socket(path: Path, source: str, since_hours: float) -> dict[str, Any]:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(30)
            connection.connect(str(path.expanduser()))
            connection.sendall(_request(source, since_hours) + b"\n")
            with connection.makefile("rb") as stream:
                response = stream.readline(MAX_RESPONSE_BYTES + 2)
    except OSError:
        raise GatewayError("gateway_unavailable") from None
    if not response.endswith(b"\n"):
        raise GatewayError("invalid_response")
    return read_reply(decode_reply(response[:-1]))


def read_at_lambda(function: str, source: str, since_hours: float) -> dict[str, Any]:
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        with closing(aws_client("lambda")) as client:
            result = client.invoke(FunctionName=function, InvocationType="RequestResponse", Payload=_request(source, since_hours))
            with closing(result["Payload"]) as stream:
                if result.get("FunctionError") or result.get("StatusCode") != 200:
                    raise GatewayError("gateway_unavailable")
                response = stream.read(MAX_RESPONSE_BYTES + 1)
    except (BotoCoreError, ClientError):
        raise GatewayError("gateway_unavailable") from None
    return read_reply(decode_reply(response))


def _request(source: str, since_hours: float) -> bytes:
    try:
        data = json.dumps(
            {"schema_version": SCHEMA_VERSION, "source": source, "since_hours": since_hours}, allow_nan=False
        ).encode()
    except (TypeError, ValueError):
        raise GatewayError("invalid_request") from None
    if len(data) > MAX_REQUEST_BYTES:
        raise GatewayError("invalid_request")
    return data


def _response(data: dict[str, Any]) -> bytes:
    encoded = json.dumps(data).encode()
    if len(encoded) > MAX_RESPONSE_BYTES:
        return json.dumps(error_reply("invalid_response")).encode()
    return encoded


class _RequestHandler(socketserver.StreamRequestHandler):
    timeout = 5

    def handle(self) -> None:
        try:
            data = self.rfile.readline(MAX_REQUEST_BYTES + 2)
            if not data.endswith(b"\n") or len(data) > MAX_REQUEST_BYTES + 1:
                result = error_reply("invalid_request")
            else:
                try:
                    event = json.loads(data.decode("utf-8"))
                except (ValueError, RecursionError):
                    result = error_reply("invalid_request")
                else:
                    result = self.server.gateway.read(event)
            self.wfile.write(_response(result) + b"\n")
        except OSError:
            # A caller that disconnects or times out has no reply to receive.
            return


class _LocalServer(socketserver.UnixStreamServer):
    def __init__(self, path: Path, gateway: Gateway) -> None:
        self.gateway = gateway
        super().__init__(str(path), _RequestHandler)


@contextmanager
def local_gateway(path: Path, gateway: Gateway, group: str | None = None) -> Iterator[_LocalServer]:
    """Bind within a directory owned by the gateway. An explicit group can share access with the harness."""
    path = path.expanduser().absolute()
    directory = path.parent.lstat()
    if not stat.S_ISDIR(directory.st_mode) or directory.st_uid != os.geteuid():
        raise ValueError("The gateway socket directory must be owned by the gateway account.")
    gid = None
    if group is None:
        if directory.st_mode & 0o077:
            raise ValueError("Use a private socket directory with mode 0700.")
    else:
        gid = grp.getgrnam(group).gr_gid
        if directory.st_gid != gid or directory.st_mode & 0o027 or not directory.st_mode & 0o010:
            raise ValueError("Use a socket directory with mode 0750 and the configured socket group.")

    old_umask = os.umask(0o177)
    try:
        server = _LocalServer(path, gateway)
    finally:
        os.umask(old_umask)
    inode = path.lstat()
    try:
        with server:
            if gid is not None:
                os.chown(path, -1, gid)
                path.chmod(0o660)
            yield server
    finally:
        try:
            current = path.lstat()
            if (inode.st_dev, inode.st_ino) == (current.st_dev, current.st_ino):
                path.unlink()
        except FileNotFoundError:
            pass


@lru_cache(maxsize=1)
def _lambda_gateway(config: str) -> Gateway:
    return load_gateway(Path(config))


def lambda_handler(event: Any, context: Any) -> dict[str, Any]:
    """Deploy with an IAM role that can read the configured Secrets Manager entries."""
    try:
        gateway = _lambda_gateway(os.environ.get("FETCHER_GATEWAY_CONFIG", "gateway.toml"))
    except (OSError, ValueError):
        return error_reply("gateway_unavailable")
    return json.loads(_response(gateway.read(event)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve an allowlisted source gateway on a local Unix socket.")
    parser.add_argument("--config", type=Path, required=True, help="operator-owned gateway TOML file")
    parser.add_argument("--socket", type=Path, required=True, help="socket path inside a private directory")
    parser.add_argument("--socket-group", help="share the socket with this local group")
    args = parser.parse_args()

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        gateway = load_gateway(args.config.expanduser())
        with local_gateway(args.socket, gateway, args.socket_group) as server:
            print("Gateway ready.", flush=True)
            server.serve_forever()
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
