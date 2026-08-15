from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from voiceprobe.conversation.turns import CompletedTurn
from voiceprobe.media.live_asr import ConsoleTranscriptListener


def event(text: str) -> SimpleNamespace:
    return SimpleNamespace(line=SimpleNamespace(text=text))


def test_continuing_speech_combines_slowly_finalized_lines() -> None:
    """Reproduce the real 'Great. Your book...' phone failure."""
    turns: list[CompletedTurn] = []

    listener = ConsoleTranscriptListener(
        on_turn=turns.append,
    )

    try:
        # The ASR lines finish >2 seconds apart even though the second
        # line starts while the first pending turn is still alive.
        with patch(
            "voiceprobe.media.live_asr.time.monotonic",
            side_effect=[10.0, 12.5],
        ):
            listener.on_line_completed(event("Great."))

            # Real Moonshine emitted this start/partial activity before
            # completing the second line.
            listener.on_line_started(event("Your book for"))

            listener.on_line_text_changed(event("Your book for Friday at 2.30 p.m."))

            listener.on_line_completed(event("Your book for Friday at 2.30 p.m."))

        # Nothing should have been prematurely emitted merely because
        # finalization of line two took > TURN_GAP_SECONDS.
        assert turns == []

        listener.flush()

        assert len(turns) == 1
        assert turns[0].lines == (
            "Great.",
            "Your book for Friday at 2.30 p.m.",
        )
        assert turns[0].text == ("Great. Your book for Friday at 2.30 p.m.")

    finally:
        listener.close()


def test_timer_flush_still_separates_genuine_turns() -> None:
    """A real endpoint still emits one turn before later speech."""
    turns: list[CompletedTurn] = []

    listener = ConsoleTranscriptListener(
        on_turn=turns.append,
    )

    try:
        with patch(
            "voiceprobe.media.live_asr.time.monotonic",
            return_value=20.0,
        ):
            listener.on_line_completed(event("What insurance do you have?"))

        # Explicitly simulate the endpoint timer firing before a later,
        # separate receptionist utterance begins.
        listener.flush()

        assert len(turns) == 1
        assert turns[0].text == ("What insurance do you have?")

        with patch(
            "voiceprobe.media.live_asr.time.monotonic",
            return_value=30.0,
        ):
            listener.on_line_completed(event("What day works best?"))

        listener.flush()

        assert len(turns) == 2
        assert turns[1].text == ("What day works best?")

    finally:
        listener.close()
