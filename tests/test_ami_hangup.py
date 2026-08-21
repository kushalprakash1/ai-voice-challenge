from voiceprobe.telephony.ami import (
    AsteriskAMIClient,
    AsteriskAMIConfig,
)


class ScriptedSocket:
    def __init__(
        self,
        incoming: bytes,
    ) -> None:
        self._incoming = bytearray(incoming)
        self.sent: list[bytes] = []
        self.closed = False

    def recv(
        self,
        size: int,
    ) -> bytes:
        if not self._incoming:
            return b""

        chunk = bytes(self._incoming[:size])
        del self._incoming[:size]

        return chunk

    def sendall(
        self,
        data: bytes,
    ) -> None:
        self.sent.append(data)

    def settimeout(
        self,
        value: float | None,
    ) -> None:
        del value

    def close(self) -> None:
        self.closed = True


def test_wait_for_hangup_skips_unrelated_event_and_correlates() -> None:
    incoming = (
        b"Asterisk Call Manager/5.0\r\n"
        b"Response: Success\r\n"
        b"ActionID: action-test\r\n"
        b"Message: Authentication accepted\r\n"
        b"\r\n"
        b"Event: Hangup\r\n"
        b"Channel: PJSIP/unrelated-00000001\r\n"
        b"Uniqueid: unrelated.1\r\n"
        b"Linkedid: unrelated.1\r\n"
        b"Cause: 16\r\n"
        b"Cause-txt: Normal Clearing\r\n"
        b"\r\n"
        b"Event: Hangup\r\n"
        b"Channel: Local/+12025550100@voiceprobe-test-00000001;1\r\n"
        b"Uniqueid: 1786788695.112\r\n"
        b"Linkedid: 1786788695.112\r\n"
        b"Cause: 16\r\n"
        b"Cause-txt: Normal Clearing\r\n"
        b"\r\n"
    )

    sock = ScriptedSocket(incoming)

    client = AsteriskAMIClient(
        AsteriskAMIConfig(
            username="voiceprobe",
            secret="synthetic-test-secret",
        ),
        connection_factory=(lambda address, timeout: sock),
        action_id_factory=(lambda: "action-test"),
    )

    client.connect()
    client.login(events="call")

    result = client.wait_for_hangup(
        unique_id="1786788695.112",
        channel=("Local/+12025550100@voiceprobe-test-00000001;1"),
    )

    assert result.unique_id == "1786788695.112"
    assert result.channel == ("Local/+12025550100@voiceprobe-test-00000001;1")
    assert result.linked_id == "1786788695.112"
    assert result.cause == 16
    assert result.cause_text == "Normal Clearing"
    assert result.tech_cause is None

    client.close()

    assert sock.closed
