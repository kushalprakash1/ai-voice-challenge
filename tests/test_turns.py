import pytest

from voiceprobe.conversation.turns import TurnAssembler


def test_combines_finalized_lines_into_one_turn() -> None:
    assembler = TurnAssembler(max_gap_seconds=1.0)

    assert (
        assembler.add_line(
            "I would like an appointment.",
            completed_at=1.0,
        )
        is None
    )

    assert (
        assembler.add_line(
            "Friday afternoon.",
            completed_at=1.4,
        )
        is None
    )

    turn = assembler.flush(completed_at=2.0)

    assert turn is not None
    assert turn.text == "I would like an appointment. Friday afternoon."
    assert turn.lines == (
        "I would like an appointment.",
        "Friday afternoon.",
    )


def test_large_gap_finishes_previous_turn() -> None:
    assembler = TurnAssembler(max_gap_seconds=0.9)

    assembler.add_line(
        "What is your date of birth?",
        completed_at=1.0,
    )

    previous = assembler.add_line(
        "And what insurance do you have?",
        completed_at=3.0,
    )

    assert previous is not None
    assert previous.text == "What is your date of birth?"

    current = assembler.flush()

    assert current is not None
    assert current.text == "And what insurance do you have?"


def test_ignores_blank_lines() -> None:
    assembler = TurnAssembler()

    result = assembler.add_line(
        "   ",
        completed_at=1.0,
    )

    assert result is None
    assert not assembler.has_pending_turn


def test_normalizes_internal_whitespace() -> None:
    assembler = TurnAssembler()

    assembler.add_line(
        "Friday     afternoon.",
        completed_at=1.0,
    )

    turn = assembler.flush()

    assert turn is not None
    assert turn.text == "Friday afternoon."


def test_rejects_non_positive_gap() -> None:
    with pytest.raises(ValueError):
        TurnAssembler(max_gap_seconds=0)


def test_gap_splitting_remains_default_behavior() -> None:
    assembler = TurnAssembler(
        max_gap_seconds=2.0,
    )

    assert (
        assembler.add_line(
            "First line.",
            completed_at=10.0,
        )
        is None
    )

    previous = assembler.add_line(
        "Second line.",
        completed_at=12.5,
    )

    assert previous is not None
    assert previous.text == "First line."

    remaining = assembler.flush()

    assert remaining is not None
    assert remaining.text == "Second line."


def test_gap_splitting_can_be_disabled_for_live_endpointing() -> None:
    assembler = TurnAssembler(
        max_gap_seconds=2.0,
        split_on_gap=False,
    )

    assert (
        assembler.add_line(
            "Great.",
            completed_at=10.0,
        )
        is None
    )

    assert (
        assembler.add_line(
            "Your book for Friday at 2.30 p.m.",
            completed_at=12.5,
        )
        is None
    )

    turn = assembler.flush()

    assert turn is not None
    assert turn.text == ("Great. Your book for Friday at 2.30 p.m.")


def test_suppresses_rapid_duplicate_finalized_line() -> None:
    assembler = TurnAssembler(
        max_gap_seconds=2.0,
        split_on_gap=False,
        duplicate_window_seconds=0.75,
    )

    assembler.add_line(
        "Friday.",
        completed_at=1.0,
    )
    assembler.add_line(
        "Friday.",
        completed_at=1.2,
    )

    turn = assembler.flush()

    assert turn is not None
    assert turn.text == "Friday."
    assert turn.lines == ("Friday.",)


def test_preserves_intentional_repeat_outside_duplicate_window() -> None:
    assembler = TurnAssembler(
        max_gap_seconds=2.0,
        split_on_gap=False,
        duplicate_window_seconds=0.5,
    )

    assembler.add_line(
        "Friday.",
        completed_at=1.0,
    )
    assembler.add_line(
        "Friday.",
        completed_at=1.75,
    )

    turn = assembler.flush()

    assert turn is not None
    assert turn.text == "Friday. Friday."
    assert turn.lines == (
        "Friday.",
        "Friday.",
    )
