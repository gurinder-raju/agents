"""Credential access for the gateway process."""

import subprocess
from contextlib import closing
from pathlib import Path
from typing import Protocol


class CredentialUnavailable(RuntimeError):
    """The gateway identity could not retrieve a usable secret."""


class CredentialStore(Protocol):
    def get(self, reference: str) -> str: ...


class KeychainStore:
    def __init__(self, service: str, keychain: Path | None = None) -> None:
        self.service = service
        self.keychain = keychain

    def get(self, reference: str) -> str:
        command = ["/usr/bin/security", "find-generic-password", "-s", self.service, "-a", reference, "-w"]
        if self.keychain is not None:
            command.append(str(self.keychain))
        try:
            result = subprocess.run(
                command, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True, timeout=5
            )
        except (OSError, UnicodeError, subprocess.SubprocessError):
            raise CredentialUnavailable("Keychain credential unavailable") from None
        secret = result.stdout.rstrip("\r\n")
        if not secret:
            raise CredentialUnavailable("Keychain credential unavailable")
        return secret


class SecretsManagerStore:
    def get(self, reference: str) -> str:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            with closing(aws_client("secretsmanager")) as client:
                secret = client.get_secret_value(SecretId=reference).get("SecretString")
        except (BotoCoreError, ClientError):
            raise CredentialUnavailable("Secrets Manager credential unavailable") from None
        if not isinstance(secret, str) or not secret:
            raise CredentialUnavailable("Secrets Manager credential unavailable")
        return secret


def aws_client(service: str):
    import boto3
    from botocore.config import Config

    return boto3.client(
        service,
        config=Config(connect_timeout=4, read_timeout=35 if service == "lambda" else 5, retries={"total_max_attempts": 1}),
    )
