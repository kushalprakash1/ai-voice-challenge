import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import voiceprobe.autonomous_phone as phone
from voiceprobe.agents.brain import CommunicationKind


CRITICAL_TEXT = "No, I need an appointment."


class FakeRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def record_event(
        self,
        event: str,
        **fields: object,
    ) -> None:
        self.events.append((event, fields))

    def record_turn_metrics(self, metrics: object) -> None:
        del metrics

    def record_transcript_turn(
        self,
        **fields: object,
    ) -> None:
        del fields


class FakeSession:
    def handle_agent_turn(self, agent_turn: str):
        del agent_turn

        return SimpleNamespace(
            patient_text=CRITICAL_TEXT,
            decision=SimpleNamespace(
                kind=CommunicationKind.DECLINE_WORKFLOW,
                facts_to_communicate=(),
                probe=None,
            ),
            progress=SimpleNamespace(
                objective_complete=False,
            ),
            timings=SimpleNamespace(
                interpreter_seconds=0.0,
                decision_seconds=0.0,
                verbalizer_seconds=0.0,
                state_update_seconds=0.0,
            ),
            meaning=SimpleNamespace(),
        )


def test_pre_render_builder_produces_telephony_pcm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synth_calls: list[str] = []

    def fake_synthesize(
        *,
        pipeline: object,
        voice: str,
        text: str,
    ) -> np.ndarray:
        del pipeline, voice
        synth_calls.append(text)

        return np.ones(
            240,
            dtype=np.float32,
        )

    monkeypatch.setattr(
        phone,
        "synthesize",
        fake_synthesize,
    )

    monkeypatch.setattr(
        phone,
        "resample_to_telephony",
        lambda audio: np.ones(
            80,
            dtype=np.float32,
        ),
    )

    monkeypatch.setattr(
        phone,
        "float_audio_to_pcm16",
        lambda audio: b"\x01\x02" * len(audio),
    )

    cache = phone.build_pre_rendered_tts_cache(
        pipeline=object(),  # type: ignore[arg-type]
        voice="synthetic",
        texts=(CRITICAL_TEXT,),
    )

    assert synth_calls == [CRITICAL_TEXT]
    assert cache == {
        CRITICAL_TEXT: b"\x01\x02" * 80,
    }


def test_process_turns_cache_hit_never_invokes_kokoro(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached_pcm = b"\x01\x02" * 8_000
    sent: list[bytes] = []

    def forbidden_synthesize(**kwargs: object) -> np.ndarray:
        del kwargs
        raise AssertionError(
            "Cache hit unexpectedly invoked Kokoro."
        )

    def forbidden_resample(audio: np.ndarray) -> np.ndarray:
        del audio
        raise AssertionError(
            "Cache hit unexpectedly invoked resampling."
        )

    def fake_send_audio(
        connection: object,
        pcm16: bytes,
        *,
        recorder: object,
    ) -> None:
        del connection, recorder
        sent.append(pcm16)

    monkeypatch.setattr(
        phone,
        "synthesize",
        forbidden_synthesize,
    )

    monkeypatch.setattr(
        phone,
        "resample_to_telephony",
        forbidden_resample,
    )

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

    turns: queue.Queue[object] = queue.Queue()

    turns.put(
        SimpleNamespace(
            text="Have a great day.",
            completed_at=time.monotonic(),
        )
    )

    turns.put(None)

    busy = threading.Event()
    busy.set()

    stop = threading.Event()
    recorder = FakeRecorder()

    phone.process_turns(
        turns=turns,  # type: ignore[arg-type]
        connection=object(),  # type: ignore[arg-type]
        session=FakeSession(),  # type: ignore[arg-type]
        pipeline=object(),  # type: ignore[arg-type]
        voice="synthetic",
        busy=busy,
        stop=stop,
        recorder=recorder,  # type: ignore[arg-type]
        tts_pcm_cache={
            CRITICAL_TEXT: cached_pcm,
        },
    )

    assert sent == [cached_pcm]

    event_names = [
        event
        for event, _ in recorder.events
    ]

    assert "tts_cache_hit" in event_names
    assert "tts_cache_miss" not in event_names
    assert busy.is_set() is False
    assert stop.is_set() is False
