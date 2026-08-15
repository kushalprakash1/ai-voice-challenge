from __future__ import annotations

import threading

import pytest

from voiceprobe import autonomous_phone as phone
from voiceprobe.tts.telephony import (
    AUDIOSOCKET_PCM8K_TYPE,
    FRAME_BYTES,
)


class StopAfterFramesConnection:
    def __init__(
        self,
        *,
        stop: threading.Event,
        frame_limit: int,
    ) -> None:
        self.stop = stop
        self.frame_limit = frame_limit
        self.sent: list[bytes] = []

    def sendall(
        self,
        payload: bytes,
    ) -> None:
        self.sent.append(payload)

        if len(self.sent) >= self.frame_limit:
            self.stop.set()


def test_idle_silence_sends_valid_zero_pcm_frames() -> None:
    stop = threading.Event()

    connection = StopAfterFramesConnection(
        stop=stop,
        frame_limit=3,
    )

    lock = threading.Lock()

    phone.send_idle_silence(
        connection,  # type: ignore[arg-type]
        stop=stop,
        send_lock=lock,
    )

    assert len(connection.sent) == 3

    for packet in connection.sent:
        assert packet[0] == AUDIOSOCKET_PCM8K_TYPE

        assert (
            int.from_bytes(
                packet[1:3],
                byteorder="big",
            )
            == FRAME_BYTES
        )

        assert packet[3:] == bytes(FRAME_BYTES)


def test_idle_silence_sends_nothing_when_already_stopped() -> None:
    stop = threading.Event()
    stop.set()

    connection = StopAfterFramesConnection(
        stop=stop,
        frame_limit=1,
    )

    phone.send_idle_silence(
        connection,  # type: ignore[arg-type]
        stop=stop,
        send_lock=threading.Lock(),
    )

    assert connection.sent == []


def test_patient_speech_holds_shared_media_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = threading.Lock()

    observed_lock_state: list[bool] = []

    def fake_send_audio(
        connection: object,
        pcm16: bytes,
        *,
        recorder: object,
    ) -> None:
        del connection, pcm16, recorder

        observed_lock_state.append(
            lock.locked()
        )

    monkeypatch.setattr(
        phone,
        "send_audio",
        fake_send_audio,
    )

    phone.send_audio_synchronized(
        object(),  # type: ignore[arg-type]
        b"\x00\x00",
        send_lock=lock,
        recorder=None,
    )

    assert observed_lock_state == [True]
