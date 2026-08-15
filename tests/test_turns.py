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
