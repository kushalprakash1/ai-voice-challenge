from dataclasses import dataclass

from voiceprobe.conversation.turns import CompletedTurn
from voiceprobe.media.live_asr import ConsoleTranscriptListener


@dataclass
class FakeLine:
    text: str


@dataclass
class FakeEvent:
    line: FakeLine


def test_listener_emits_completed_turn_to_callback() -> None:
    turns: list[CompletedTurn] = []

    listener = ConsoleTranscriptListener(
        on_turn=turns.append,
    )

    listener.on_line_completed(
        FakeEvent(line=FakeLine(text="What insurance do you have?"))
    )

    listener.flush()
    listener.close()

    assert len(turns) == 1
    assert turns[0].text == "What insurance do you have?"


def test_endpoint_fast_for_complete_question() -> None:
    assert (
        ConsoleTranscriptListener._flush_delay_for_text("What insurance do you have?")
        == 0.65
    )


def test_endpoint_waits_for_single_word_fragment() -> None:
    assert ConsoleTranscriptListener._flush_delay_for_text("What?") == 1.8


def test_endpoint_waits_for_two_word_fragment() -> None:
    assert ConsoleTranscriptListener._flush_delay_for_text("They and") == 1.8


def test_endpoint_waits_for_explicitly_incomplete_phrase() -> None:
    assert ConsoleTranscriptListener._flush_delay_for_text("How about...") == 1.8


def test_complete_line_emits_prefetch_candidate() -> None:
    candidates: list[str] = []

    listener = ConsoleTranscriptListener(
        on_candidate=candidates.append,
    )

    listener.on_line_completed(
        FakeEvent(line=FakeLine(text="What insurance do you have?"))
    )

    listener.close()

    assert candidates == ["What insurance do you have?"]


def test_tiny_fragment_does_not_prefetch() -> None:
    candidates: list[str] = []

    listener = ConsoleTranscriptListener(
        on_candidate=candidates.append,
    )

    listener.on_line_completed(FakeEvent(line=FakeLine(text="What?")))

    listener.close()

    assert candidates == []
