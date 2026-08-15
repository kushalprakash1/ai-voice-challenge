from __future__ import annotations

import queue
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import voiceprobe.autonomous_phone as phone
from voiceprobe.agents.brain import CommunicationKind
from voiceprobe.conversation.turns import CompletedTurn


class FakeConnection:
    def __init__(
        self,
        *,
        fail_send: bool = False,
    ) -> None:
        self.fail_send = fail_send
        self.sent: list[bytes] = []
        self.shutdown_modes: list[int] = []

    def sendall(
        self,
        payload: bytes,
    ) -> None:
        if self.fail_send:
            raise BrokenPipeError("synthetic disconnected socket")

        self.sent.append(payload)

    def shutdown(
        self,
        how: int,
    ) -> None:
        self.shutdown_modes.append(how)


class FakeRecorder:
    def __init__(
        self,
        sequence: list[str],
    ) -> None:
        self.sequence = sequence

    def record_event(
        self,
        event_name: str,
        **fields: object,
    ) -> None:
        del fields
        self.sequence.append(event_name)

    def record_turn_metrics(
        self,
        metrics: object,
    ) -> None:
        del metrics

    def record_transcript_turn(
        self,
        *,
        speaker: str,
        **fields: object,
    ) -> None:
        del fields

        if speaker == "patient":
            self.sequence.append("patient_transcript")


class FakeSession:
    def __init__(
        self,
        kind: CommunicationKind,
    ) -> None:
        self.kind = kind

    def handle_agent_turn(
        self,
        agent_turn: str,
    ) -> object:
        del agent_turn

        return SimpleNamespace(
            patient_text=(
                "Okay, thank you. Bye."
                if self.kind is CommunicationKind.END_CONVERSATION
                else ("" if self.kind is CommunicationKind.WAIT else "Okay.")
            ),
            decision=SimpleNamespace(
                kind=self.kind,
                facts_to_communicate=(),
                probe=None,
            ),
            meaning=SimpleNamespace(),
            progress=SimpleNamespace(
                objective_complete=False,
            ),
            timings=SimpleNamespace(
                interpreter_seconds=0.0,
                decision_seconds=0.0,
                verbalizer_seconds=0.0,
                state_update_seconds=0.0,
            ),
        )


def make_turn() -> CompletedTurn:
    now = time.monotonic()

    return CompletedTurn(
        text="Okay, bye.",
        lines=("Okay, bye.",),
        started_at=now - 0.2,
        completed_at=now,
    )


def install_fast_audio(
    monkeypatch: pytest.MonkeyPatch,
    sequence: list[str],
) -> None:
    monkeypatch.setattr(
        phone,
        "synthesize",
        lambda **kwargs: np.zeros(
            160,
            dtype=np.float32,
        ),
    )

    monkeypatch.setattr(
        phone,
        "resample_to_telephony",
        lambda audio: audio,
    )

    monkeypatch.setattr(
        phone,
        "float_audio_to_pcm16",
        lambda audio: b"\x00\x00" * len(audio),
    )

    def fake_send_audio(
        connection: object,
        pcm16: bytes,
        *,
        recorder: object,
    ) -> None:
        del connection, pcm16, recorder
        sequence.append("audio_sent")

    monkeypatch.setattr(
        phone,
        "send_audio",
        fake_send_audio,
    )

    monkeypatch.setattr(
        phone.time,
        "sleep",
        lambda seconds: None,
    )


def test_terminate_audiosocket_sends_packet_and_shutdown() -> None:
    connection = FakeConnection()

    packet_sent = phone.terminate_audiosocket_connection(
        connection  # type: ignore[arg-type]
    )

    assert packet_sent is True
    assert connection.sent == [b"\x00\x00\x00"]
    assert connection.shutdown_modes == [socket.SHUT_RDWR]


def test_terminate_audiosocket_still_shutdowns_when_send_fails() -> None:
    connection = FakeConnection(fail_send=True)

    packet_sent = phone.terminate_audiosocket_connection(
        connection  # type: ignore[arg-type]
    )

    assert packet_sent is False
    assert connection.shutdown_modes == [socket.SHUT_RDWR]


def test_end_conversation_hangup_occurs_after_final_playback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence: list[str] = []
    install_fast_audio(
        monkeypatch,
        sequence,
    )

    def fake_terminate(
        connection: object,
    ) -> bool:
        del connection
        sequence.append("terminate")
        return True

    monkeypatch.setattr(
        phone,
        "terminate_audiosocket_connection",
        fake_terminate,
    )

    turns: queue.Queue[CompletedTurn | None] = queue.Queue()

    turns.put(make_turn())

    busy = threading.Event()
    busy.set()

    stop = threading.Event()

    phone.process_turns(
        turns=turns,
        connection=object(),  # type: ignore[arg-type]
        session=FakeSession(CommunicationKind.END_CONVERSATION),  # type: ignore[arg-type]
        pipeline=object(),  # type: ignore[arg-type]
        voice="synthetic",
        busy=busy,
        stop=stop,
        recorder=FakeRecorder(sequence),  # type: ignore[arg-type]
    )

    assert stop.is_set()
    assert sequence.index("audio_sent") < sequence.index("patient_transcript")
    assert sequence.index("patient_transcript") < sequence.index("playback_finished")
    assert sequence.index("playback_finished") < sequence.index(
        "local_hangup_requested"
    )
    assert sequence.index("local_hangup_requested") < sequence.index("terminate")
    assert sequence.index("terminate") < sequence.index("local_hangup_signaled")


def test_normal_response_does_not_terminate_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence: list[str] = []
    install_fast_audio(
        monkeypatch,
        sequence,
    )

    def unexpected_terminate(
        connection: object,
    ) -> bool:
        del connection
        raise AssertionError("Normal response attempted to terminate AudioSocket.")

    monkeypatch.setattr(
        phone,
        "terminate_audiosocket_connection",
        unexpected_terminate,
    )

    turns: queue.Queue[CompletedTurn | None] = queue.Queue()

    turns.put(make_turn())
    turns.put(None)

    busy = threading.Event()
    busy.set()

    stop = threading.Event()

    phone.process_turns(
        turns=turns,
        connection=object(),  # type: ignore[arg-type]
        session=FakeSession(CommunicationKind.ANSWER),  # type: ignore[arg-type]
        pipeline=object(),  # type: ignore[arg-type]
        voice="synthetic",
        busy=busy,
        stop=stop,
        recorder=FakeRecorder(sequence),  # type: ignore[arg-type]
    )

    assert stop.is_set() is False
    assert "audio_sent" in sequence
    assert "local_hangup_requested" not in sequence


def test_wait_skips_tts_playback_and_patient_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence: list[str] = []

    def unexpected_synthesize(**kwargs: object) -> np.ndarray:
        del kwargs
        raise AssertionError("WAIT attempted to invoke TTS.")

    def unexpected_terminate(connection: object) -> bool:
        del connection
        raise AssertionError("WAIT attempted to terminate AudioSocket.")

    monkeypatch.setattr(
        phone,
        "synthesize",
        unexpected_synthesize,
    )
    monkeypatch.setattr(
        phone,
        "terminate_audiosocket_connection",
        unexpected_terminate,
    )

    turns: queue.Queue[CompletedTurn | None] = queue.Queue()
    turns.put(make_turn())
    turns.put(None)

    busy = threading.Event()
    busy.set()

    stop = threading.Event()

    phone.process_turns(
        turns=turns,
        connection=object(),  # type: ignore[arg-type]
        session=FakeSession(CommunicationKind.WAIT),  # type: ignore[arg-type]
        pipeline=object(),  # type: ignore[arg-type]
        voice="synthetic",
        busy=busy,
        stop=stop,
        recorder=FakeRecorder(sequence),  # type: ignore[arg-type]
    )

    assert stop.is_set() is False
    assert "patient_wait" in sequence
    assert "patient_response_generated" not in sequence
    assert "audio_sent" not in sequence
    assert "patient_transcript" not in sequence
    assert "playback_started" not in sequence
    assert "local_hangup_requested" not in sequence
