from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

import pytest

from voiceprobe.telephony.ami import (
    AMIConnectionStateError,
    AMIOriginateError,
    AsteriskAMIClient,
    AsteriskAMIConfig,
)

SECRET = "synthetic-secret"
DESTINATION = "+12025550101"
CALL_ID = UUID("550e8400-e29b-41d4-a716-446655440000")


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
    fake: FakeSocket,
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

        return fake

    return factory


def ids(
    *values: str,
):
    iterator = iter(values)

    def next_id() -> str:
        return next(iterator)

    return next_id


def test_originate_requires_authentication() -> None:
    fake = FakeSocket(
        [
            b"Asterisk Call Manager/11.0.0\r\n",
        ]
    )

    client = AsteriskAMIClient(
        config(),
        connection_factory=factory_for(fake),
    )

    client.connect()

    with pytest.raises(
        AMIConnectionStateError,
        match="authenticate",
    ):
        client.originate_audiosocket(DESTINATION)


def test_originate_requires_call_events() -> None:
    fake = FakeSocket(
        [
            b"Asterisk Call Manager/11.0.0\r\n",
            (b"Response: Success\r\nActionID: login-1\r\n\r\n"),
        ]
    )

    client = AsteriskAMIClient(
        config(),
        connection_factory=factory_for(fake),
        action_id_factory=ids(
            "login-1",
        ),
    )

    client.connect()
    client.login()

    with pytest.raises(
        AMIConnectionStateError,
        match="events='call'",
    ):
        client.originate_audiosocket(DESTINATION)


def test_successful_originate_sends_exact_protocol() -> None:
    fake = FakeSocket(
        [
            b"Asterisk Call Manager/11.0.0\r\n",
            (b"Response: Success\r\nActionID: login-1\r\n\r\n"),
            (
                b"Response: Success\r\n"
                b"ActionID: originate-1\r\n"
                b"Message: Originate successfully queued\r\n"
                b"\r\n"
            ),
            (
                b"Event: OriginateResponse\r\n"
                b"ActionID: originate-1\r\n"
                b"Response: Success\r\n"
                b"Channel: Local/+12025550101@voiceprobe-test-00000001;1\r\n"
                b"Reason: 4\r\n"
                b"Uniqueid: 1750000000.123\r\n"
                b"\r\n"
            ),
        ]
    )

    client = AsteriskAMIClient(
        config(),
        connection_factory=factory_for(fake),
        action_id_factory=ids(
            "login-1",
            "originate-1",
        ),
    )

    client.connect()
    client.login(events="call")

    result = client.originate_audiosocket(
        DESTINATION,
        call_id=CALL_ID,
        timeout_ms=30_000,
    )

    sent = bytes(fake.sent)

    assert b"Events: call\r\n" in sent
    assert b"Action: Originate\r\n" in sent
    assert b"Channel: Local/+12025550101@voiceprobe-test\r\n" in sent
    assert b"Application: AudioSocket\r\n" in sent
    assert (b"Data: 550e8400-e29b-41d4-a716-446655440000,127.0.0.1:9019\r\n") in sent
    assert b"Timeout: 30000\r\n" in sent
    assert b"Async: true\r\n" in sent
    assert b"ActionID: originate-1\r\n" in sent

    assert result.action_id == "originate-1"
    assert result.audiosocket_call_id == CALL_ID
    assert result.asterisk_unique_id == "1750000000.123"
    assert result.response == "Success"


def test_originate_ignores_unrelated_events() -> None:
    fake = FakeSocket(
        [
            b"Asterisk Call Manager/11.0.0\r\n",
            (b"Response: Success\r\nActionID: login-1\r\n\r\n"),
            (
                b"Event: OriginateResponse\r\n"
                b"ActionID: wrong-action\r\n"
                b"Response: Failure\r\n"
                b"Reason: 0\r\n"
                b"Uniqueid: <null>\r\n"
                b"\r\n"
            ),
            (b"Response: Success\r\nActionID: originate-1\r\n\r\n"),
            (b"Event: Newchannel\r\nChannel: unrelated\r\n\r\n"),
            (
                b"Event: OriginateResponse\r\n"
                b"ActionID: originate-1\r\n"
                b"Response: Success\r\n"
                b"Channel: Local/test\r\n"
                b"Reason: 4\r\n"
                b"Uniqueid: 123.456\r\n"
                b"\r\n"
            ),
        ]
    )

    client = AsteriskAMIClient(
        config(),
        connection_factory=factory_for(fake),
        action_id_factory=ids(
            "login-1",
            "originate-1",
        ),
    )

    client.connect()
    client.login(events="call")

    result = client.originate_audiosocket(
        DESTINATION,
        call_id=CALL_ID,
    )

    assert result.asterisk_unique_id == "123.456"


def test_failed_originate_response_is_rejected() -> None:
    fake = FakeSocket(
        [
            b"Asterisk Call Manager/11.0.0\r\n",
            (b"Response: Success\r\nActionID: login-1\r\n\r\n"),
            (b"Response: Success\r\nActionID: originate-1\r\n\r\n"),
            (
                b"Event: OriginateResponse\r\n"
                b"ActionID: originate-1\r\n"
                b"Response: Failure\r\n"
                b"Reason: 0\r\n"
                b"Uniqueid: <null>\r\n"
                b"\r\n"
            ),
        ]
    )

    client = AsteriskAMIClient(
        config(),
        connection_factory=factory_for(fake),
        action_id_factory=ids(
            "login-1",
            "originate-1",
        ),
    )

    client.connect()
    client.login(events="call")

    with pytest.raises(
        AMIOriginateError,
        match="reason=0",
    ):
        client.originate_audiosocket(
            DESTINATION,
            call_id=CALL_ID,
        )


def test_immediate_originate_rejection_is_rejected() -> None:
    fake = FakeSocket(
        [
            b"Asterisk Call Manager/11.0.0\r\n",
            (b"Response: Success\r\nActionID: login-1\r\n\r\n"),
            (
                b"Response: Error\r\n"
                b"ActionID: originate-1\r\n"
                b"Message: Permission denied\r\n"
                b"\r\n"
            ),
        ]
    )

    client = AsteriskAMIClient(
        config(),
        connection_factory=factory_for(fake),
        action_id_factory=ids(
            "login-1",
            "originate-1",
        ),
    )

    client.connect()
    client.login(events="call")

    with pytest.raises(
        AMIOriginateError,
        match="rejected",
    ):
        client.originate_audiosocket(
            DESTINATION,
            call_id=CALL_ID,
        )


def test_generated_audiosocket_id_is_uuid() -> None:
    fake = FakeSocket(
        [
            b"Asterisk Call Manager/11.0.0\r\n",
            (b"Response: Success\r\nActionID: login-1\r\n\r\n"),
            (b"Response: Success\r\nActionID: originate-1\r\n\r\n"),
            (
                b"Event: OriginateResponse\r\n"
                b"ActionID: originate-1\r\n"
                b"Response: Success\r\n"
                b"Channel: Local/test\r\n"
                b"Uniqueid: 123.456\r\n"
                b"\r\n"
            ),
        ]
    )

    client = AsteriskAMIClient(
        config(),
        connection_factory=factory_for(fake),
        action_id_factory=ids(
            "login-1",
            "originate-1",
        ),
    )

    client.connect()
    client.login(events="call")

    result = client.originate_audiosocket(DESTINATION)

    assert isinstance(
        result.audiosocket_call_id,
        UUID,
    )


def test_originate_rejects_header_injection_destination() -> None:
    fake = FakeSocket(
        [
            b"Asterisk Call Manager/11.0.0\r\n",
            (b"Response: Success\r\nActionID: login-1\r\n\r\n"),
        ]
    )

    client = AsteriskAMIClient(
        config(),
        connection_factory=factory_for(fake),
        action_id_factory=ids(
            "login-1",
        ),
    )

    client.connect()
    client.login(events="call")

    with pytest.raises(
        ValueError,
        match="line breaks",
    ):
        client.originate_audiosocket("+12025550101\r\nApplication: System")


def test_originate_rejects_non_nanp_destination() -> None:
    fake = FakeSocket(
        [
            b"Asterisk Call Manager/11.0.0\r\n",
            (b"Response: Success\r\nActionID: login-1\r\n\r\n"),
        ]
    )

    client = AsteriskAMIClient(
        config(),
        connection_factory=factory_for(fake),
        action_id_factory=ids(
            "login-1",
        ),
    )

    client.connect()
    client.login(events="call")

    with pytest.raises(
        ValueError,
        match="NANP",
    ):
        client.originate_audiosocket("+442071234567")


def test_originate_timeout_is_bounded() -> None:
    fake = FakeSocket(
        [
            b"Asterisk Call Manager/11.0.0\r\n",
            (b"Response: Success\r\nActionID: login-1\r\n\r\n"),
        ]
    )

    client = AsteriskAMIClient(
        config(),
        connection_factory=factory_for(fake),
        action_id_factory=ids(
            "login-1",
        ),
    )

    client.connect()
    client.login(events="call")

    with pytest.raises(
        ValueError,
        match="180000",
    ):
        client.originate_audiosocket(
            DESTINATION,
            timeout_ms=180_001,
        )
