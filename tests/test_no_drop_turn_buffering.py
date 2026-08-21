# Regression tests for no-drop live turn buffering.

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

from voiceprobe.agents.brain import CommunicationKind
from voiceprobe.autonomous_phone import (
    process_turns,
    queue_completed_turn,
    should_forward_inbound_audio,
)
from voiceprobe.conversation.turns import CompletedTurn


def make_turn(
    text: str,
    *,
    started_at: float = 1.0,
    completed_at: float = 2.0,
) -> CompletedTurn:
    return CompletedTurn(
        text=text,
        lines=(text,),
        started_at=started_at,
        completed_at=completed_at,
    )


def test_finalized_turn_is_buffered_when_worker_is_busy() -> None:
    turns: queue.Queue[CompletedTurn | None] = queue.Queue()
    busy = threading.Event()
    stop = threading.Event()

    busy.set()

    turn = make_turn(
        "Would you like to check Friday, August 28th?"
    )

    disposition = queue_completed_turn(
        turn=turn,
        turns=turns,
        busy=busy,
        stop=stop,
    )

    assert disposition == "buffered"
    assert turns.get_nowait() is turn


def test_new_turn_claims_busy_window_before_queue_publish() -> None:
    turns: queue.Queue[CompletedTurn | None] = queue.Queue()
    busy = threading.Event()
    stop = threading.Event()

    turn = make_turn("What insurance do you have?")

    disposition = queue_completed_turn(
        turn=turn,
        turns=turns,
        busy=busy,
        stop=stop,
    )

    assert disposition == "queued"
    assert busy.is_set()
    assert turns.get_nowait() is turn


def test_shutdown_does_not_accept_new_turn() -> None:
    turns: queue.Queue[CompletedTurn | None] = queue.Queue()
    busy = threading.Event()
    stop = threading.Event()

    stop.set()

    disposition = queue_completed_turn(
        turn=make_turn("Friday afternoon?"),
        turns=turns,
        busy=busy,
        stop=stop,
    )

    assert disposition == "stopped"
    assert turns.empty()


def test_reasoning_does_not_mute_inbound_asr() -> None:
    playback_active = threading.Event()

    assert should_forward_inbound_audio(
        playback_active=playback_active,
    )

    playback_active.set()

    assert not should_forward_inbound_audio(
        playback_active=playback_active,
    )


class FakeSession:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def handle_agent_turn(self, text: str):
        self.seen.append(text)

        return SimpleNamespace(
            patient_text="",
            decision=SimpleNamespace(
                kind=CommunicationKind.WAIT,
                facts_to_communicate=(),
                probe=None,
            ),
            timings=SimpleNamespace(
                interpreter_seconds=0.0,
                decision_seconds=0.0,
                verbalizer_seconds=0.0,
                state_update_seconds=0.0,
            ),
            progress=SimpleNamespace(
                objective_complete=False,
            ),
            meaning={},
        )


class FakeRecorder:
    def record_turn_metrics(self, payload) -> None:
        pass

    def record_event(self, *args, **kwargs) -> None:
        pass


def test_worker_processes_multiple_buffered_turns_sequentially() -> None:
    turns: queue.Queue[CompletedTurn | None] = queue.Queue()

    first = make_turn(
        "There are no Friday afternoon openings.",
        started_at=1.0,
        completed_at=2.0,
    )
    second = make_turn(
        "Would you like the following Friday?",
        started_at=2.1,
        completed_at=3.0,
    )

    turns.put(first)
    turns.put(second)
    turns.put(None)

    busy = threading.Event()
    playback_active = threading.Event()
    stop = threading.Event()
    session = FakeSession()

    process_turns(
        turns=turns,
        connection=object(),
        session=session,
        pipeline=object(),
        voice="test",
        busy=busy,
        playback_active=playback_active,
        stop=stop,
        recorder=FakeRecorder(),
        audiosocket_send_lock=None,
        tts_pcm_cache=None,
    )

    assert session.seen == [
        first.text,
        second.text,
    ]
    assert not playback_active.is_set()
