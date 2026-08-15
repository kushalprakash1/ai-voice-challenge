"""Endpoint policy regression tests for short live-ASR turns."""

from voiceprobe.media.live_asr import (
    COMPLETE_TURN_FLUSH_SECONDS,
    INCOMPLETE_TURN_FLUSH_SECONDS,
    SHORT_TURN_FLUSH_SECONDS,
    ConsoleTranscriptListener,
)


def delay(text: str) -> float:
    return ConsoleTranscriptListener._flush_delay_for_text(text)


def test_long_complete_question_keeps_proven_fast_endpoint() -> None:
    assert delay("What insurance do you have?") == COMPLETE_TURN_FLUSH_SECONDS


def test_short_self_contained_day_uses_middle_endpoint() -> None:
    assert delay("Friday.") == SHORT_TURN_FLUSH_SECONDS


def test_short_goodbye_uses_middle_endpoint() -> None:
    assert delay("Okay, bye.") == SHORT_TURN_FLUSH_SECONDS


def test_short_insurance_confirmation_uses_middle_endpoint() -> None:
    assert delay("Blue Cross?") == SHORT_TURN_FLUSH_SECONDS


def test_short_time_confirmation_uses_middle_endpoint() -> None:
    assert delay("4:30 PM?") == SHORT_TURN_FLUSH_SECONDS


def test_bare_interrogative_keeps_fragment_protection() -> None:
    assert delay("What?") == INCOMPLETE_TURN_FLUSH_SECONDS


def test_connective_fragment_keeps_fragment_protection() -> None:
    assert delay("They and") == INCOMPLETE_TURN_FLUSH_SECONDS


def test_explicit_continuation_keeps_fragment_protection() -> None:
    assert delay("How about...") == INCOMPLETE_TURN_FLUSH_SECONDS


def test_auxiliary_pronoun_fragment_keeps_fragment_protection() -> None:
    assert delay("Can you?") == INCOMPLETE_TURN_FLUSH_SECONDS


def test_auxiliary_fragment_without_terminal_punctuation_stays_slow() -> None:
    assert delay("Would Friday") == INCOMPLETE_TURN_FLUSH_SECONDS


def test_short_complete_auxiliary_question_can_finish() -> None:
    assert delay("Would Friday?") == SHORT_TURN_FLUSH_SECONDS
