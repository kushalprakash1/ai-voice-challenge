from voiceprobe.autonomous_phone import (
    IncompleteTurnBuffer,
    is_incomplete_turn_fragment,
)
from voiceprobe.conversation.turns import CompletedTurn


def make_turn(
    text: str,
    *,
    started_at: float,
    completed_at: float,
) -> CompletedTurn:
    return CompletedTurn(
        text=text,
        lines=(text,),
        started_at=started_at,
        completed_at=completed_at,
    )


def test_exact_live_fragment_is_never_actionable_alone() -> None:
    assert is_incomplete_turn_fragment("Would any...")


def test_complete_short_turns_remain_actionable() -> None:
    assert not is_incomplete_turn_fragment("Friday.")
    assert not is_incomplete_turn_fragment("Blue Cross?")
    assert not is_incomplete_turn_fragment("Thank you.")
    assert not is_incomplete_turn_fragment("One moment.")


def test_fragment_is_held_instead_of_sent_to_reasoning() -> None:
    buffer = IncompleteTurnBuffer()

    turn, disposition, discarded = buffer.ingest(
        make_turn(
            "Would any...",
            started_at=1.0,
            completed_at=1.5,
        )
    )

    assert turn is None
    assert disposition == "held_fragment"
    assert discarded is None


def test_quick_clause_continuation_merges() -> None:
    buffer = IncompleteTurnBuffer()

    first, _, _ = buffer.ingest(
        make_turn(
            "Would any...",
            started_at=1.0,
            completed_at=1.5,
        )
    )

    assert first is None

    merged, disposition, discarded = buffer.ingest(
        make_turn(
            "of these times work for you?",
            started_at=2.0,
            completed_at=2.8,
        )
    )

    assert disposition == "merged_fragment"
    assert discarded is None
    assert merged is not None
    assert merged.text == "Would any of these times work for you?"
    assert merged.lines == (
        "Would any...",
        "of these times work for you?",
    )


def test_independent_sentence_is_not_blindly_attached() -> None:
    buffer = IncompleteTurnBuffer()

    buffer.ingest(
        make_turn(
            "Would any...",
            started_at=1.0,
            completed_at=1.5,
        )
    )

    next_turn, disposition, discarded = buffer.ingest(
        make_turn(
            "There are no Friday afternoon openings.",
            started_at=2.0,
            completed_at=3.0,
        )
    )

    assert disposition == "fragment_discarded"
    assert next_turn is not None
    assert next_turn.text == (
        "There are no Friday afternoon openings."
    )
    assert discarded is not None
    assert discarded.text == "Would any..."


def test_stale_fragment_does_not_poison_later_complete_turn() -> None:
    buffer = IncompleteTurnBuffer(
        merge_window_seconds=4.0,
    )

    buffer.ingest(
        make_turn(
            "Would any...",
            started_at=1.0,
            completed_at=1.5,
        )
    )

    next_turn, disposition, discarded = buffer.ingest(
        make_turn(
            "There are no Friday afternoon openings.",
            started_at=8.0,
            completed_at=9.0,
        )
    )

    assert disposition == "fragment_discarded"
    assert next_turn is not None
    assert discarded is not None
    assert discarded.text == "Would any..."


def test_pending_fragment_can_be_taken_at_shutdown() -> None:
    buffer = IncompleteTurnBuffer()

    buffer.ingest(
        make_turn(
            "Would any...",
            started_at=1.0,
            completed_at=1.5,
        )
    )

    pending = buffer.take_pending()

    assert pending is not None
    assert pending.text == "Would any..."
    assert buffer.take_pending() is None
