from __future__ import annotations

from collections.abc import Iterable

import pytest

from voiceprobe.telephony.ami import (
    AMIAuthenticationError,
    AMIConnectionStateError,
    AMIMessage,
    AMIProtocolError,
    AsteriskAMIClient,
    AsteriskAMIConfig,
    parse_ami_message,
)

SECRET = "super-secret-test-value"


class FakeSocket:
    def __init__(
        self,
        chunks: Iterable[bytes],
    ) -> None:
        self.chunks = list(chunks)
        self.sent = bytearray()
        self.closed = False
        self.timeout: float | None = None

    def recv(
        self,
        size: int,
    ) -> bytes:
        del size

        if not self.chunks:
            return b""

        return self.chunks.pop(0)

    def sendall(
        self,
        data: bytes,
    ) -> None:
        self.sent.extend(data)

    def settimeout(
        self,
        value: float | None,
    ) -> None:
        self.timeout = value

    def close(self) -> None:
        self.closed = True


def config() -> AsteriskAMIConfig:
    return AsteriskAMIConfig(
        username="voiceprobe",
        secret=SECRET,
    )


def factory_for(
    fake_socket: FakeSocket,
):
    def factory(
        address: tuple[str, int],
        timeout: float,
    ) -> FakeSocket:
        assert address == (
            "127.0.0.1",
            5038,
        )
        assert timeout == 5.0

        return fake_socket

    return factory


def success_messages() -> list[bytes]:
    return [
        b"Asterisk Call ",
        b"Manager/11.0.0\r\n",
        (b"Response: Success\r\nMessage: Authentication accepted\r\n\r\n"),
        (b"Response: Success\r\nPing: Pong\r\n\r\n"),
        (b"Response: Goodbye\r\nMessage: Thanks\r\n\r\n"),
    ]


def test_config_rejects_remote_ami_host() -> None:
    with pytest.raises(
        ValueError,
        match="127.0.0.1",
    ):
        AsteriskAMIConfig(
            username="voiceprobe",
            secret=SECRET,
            host="192.0.2.20",
        )


def test_config_repr_never_contains_secret() -> None:
    value = config()

    assert SECRET not in repr(value)


def test_config_rejects_header_injection() -> None:
    with pytest.raises(
        ValueError,
        match="line breaks",
    ):
        AsteriskAMIConfig(
            username=("voiceprobe\r\nAction: Originate"),
            secret=SECRET,
        )


def test_message_lookup_is_case_insensitive() -> None:
    message = AMIMessage(
        headers=(
            (
                "Response",
                "Success",
            ),
            (
                "PING",
                "Pong",
            ),
        )
    )

    assert message.get("response") == "Success"
    assert message.get("ping") == "Pong"


def test_parse_ami_message() -> None:
    message = parse_ami_message(b"Response: Success\r\nPing: Pong")

    assert message.response == "Success"
    assert message.get("Ping") == "Pong"


def test_connect_accepts_fragmented_banner() -> None:
    fake = FakeSocket(
        [
            b"Asterisk ",
            b"Call Manager/",
            b"11.0.0\r\n",
        ]
    )

    client = AsteriskAMIClient(
        config(),
        connection_factory=(factory_for(fake)),
    )

    banner = client.connect()

    assert banner == ("Asterisk Call Manager/11.0.0")
    assert client.connected is True

    client.close()

    assert fake.closed is True


def test_ping_requires_authentication() -> None:
    fake = FakeSocket(
        [
            b"Asterisk Call Manager/11.0.0\r\n",
        ]
    )

    client = AsteriskAMIClient(
        config(),
        connection_factory=(factory_for(fake)),
    )

    client.connect()

    with pytest.raises(
        AMIConnectionStateError,
        match="authenticate",
    ):
        client.ping()


def test_login_and_ping_succeed() -> None:
    fake = FakeSocket(success_messages())

    client = AsteriskAMIClient(
        config(),
        connection_factory=(factory_for(fake)),
    )

    client.connect()
    client.login()

    response = client.ping()

    assert client.authenticated is True
    assert response.get("Ping") == "Pong"

    sent = bytes(fake.sent)

    assert b"Action: Login\r\n" in sent
    assert b"Events: off\r\n" in sent
    assert b"Action: Ping\r\n" in sent

    client.logoff()

    assert fake.closed is True


def test_failed_login_does_not_expose_secret() -> None:
    fake = FakeSocket(
        [
            b"Asterisk Call Manager/11.0.0\r\n",
            (b"Response: Error\r\nMessage: Authentication failed\r\n\r\n"),
        ]
    )

    client = AsteriskAMIClient(
        config(),
        connection_factory=(factory_for(fake)),
    )

    client.connect()

    with pytest.raises(
        AMIAuthenticationError,
    ) as captured:
        client.login()

    assert SECRET not in str(captured.value)


def test_invalid_banner_is_rejected_and_socket_closed() -> None:
    fake = FakeSocket(
        [
            b"Not Asterisk\r\n",
        ]
    )

    client = AsteriskAMIClient(
        config(),
        connection_factory=(factory_for(fake)),
    )

    with pytest.raises(
        AMIProtocolError,
        match="banner",
    ):
        client.connect()

    assert fake.closed is True


def test_context_manager_logs_off_and_closes() -> None:
    fake = FakeSocket(
        [
            b"Asterisk Call Manager/11.0.0\r\n",
            (b"Response: Success\r\nMessage: Authentication accepted\r\n\r\n"),
            (b"Response: Goodbye\r\nMessage: Thanks\r\n\r\n"),
        ]
    )

    with AsteriskAMIClient(
        config(),
        connection_factory=(factory_for(fake)),
    ) as client:
        client.login()

    assert fake.closed is True
    assert b"Action: Logoff\r\n" in bytes(fake.sent)
