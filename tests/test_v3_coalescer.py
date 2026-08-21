from voiceprobe.v3.coalescer import ConversationBurstCoalescer
from voiceprobe.v3.models import DecisionKind


def test_latest_actionable_request_wins_over_stale_status() -> None:
    result = ConversationBurstCoalescer().coalesce(
        [
            "Thanks, Alex.",
            "Let me check available appointments for you on Friday afternoon.",
            "Thanks for confirming your date of birth, Alex.",
            "What is the reason for your visit?",
        ]
    )

    assert result.actionable_turn == "What is the reason for your visit?"
    assert result.decision.kind == DecisionKind.ANSWER_COMPLAINT
    assert "shoulder" in result.decision.text.casefold()


def test_acknowledgement_only_burst_waits() -> None:
    result = ConversationBurstCoalescer().coalesce(
        [
            "Thanks, Alex.",
            "Welcome to Pivot Point.",
        ]
    )

    assert result.actionable_turn is None
    assert result.decision.kind == DecisionKind.WAIT


def test_incomplete_fragment_does_not_fall_through_to_model() -> None:
    result = ConversationBurstCoalescer().coalesce(
        ["Would any..."]
    )

    assert result.actionable_turn is None
    assert result.decision.kind == DecisionKind.HOLD


def test_provider_request_beats_prior_availability_status() -> None:
    result = ConversationBurstCoalescer().coalesce(
        [
            "We have openings on Friday afternoon with two providers.",
            (
                "Would you prefer to see Dr. Zygmunt-Lukowski or "
                "Dr. Kelly Noble or is the first available okay?"
            ),
        ]
    )

    assert result.decision.kind == DecisionKind.ANSWER_PROVIDER_PREFERENCE
    assert result.decision.text == "First available is fine."


def test_direct_question_owns_trailing_illustrative_fragment() -> None:
    result = ConversationBurstCoalescer().coalesce(
        [
            "Can you tell me the reason for your visit?",
            (
                "For example, is this a routine checkup, a follow-up, "
                "or something urgent?"
            ),
        ]
    )

    assert result.actionable_turn == "Can you tell me the reason for your visit?"
    assert result.decision.kind == DecisionKind.ANSWER_COMPLAINT
    assert result.decision.text == "I have right shoulder pain."
    assert "consultation" not in result.decision.text.casefold()
