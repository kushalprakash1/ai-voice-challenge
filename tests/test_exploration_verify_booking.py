from voiceprobe.agents.brain import (
    CommunicationDecision,
    CommunicationKind,
)
from voiceprobe.conversation.exploration_policy import (
    apply_exploration_policy,
)
from voiceprobe.conversation.grounding import (
    ground_turn_meaning,
)
from voiceprobe.conversation.meaning import (
    TurnMeaning,
)
from voiceprobe.conversation.objective import (
    AppointmentProgress,
)
from voiceprobe.scenarios.catalog import (
    get_scenario,
)


def test_exploration_preserves_verify_booking() -> None:
    scenario = get_scenario(
        "autonomous-phone-diagnostic"
    )

    meaning = TurnMeaning(
        topic="conversation ending",
        conversation_end_requested=True,
    )

    grounded = ground_turn_meaning(
        scenario=scenario,
        meaning=meaning,
    )

    progress = AppointmentProgress(
        offered_day="Friday",
        offered_time="2:30 PM",
        offer_accepted=True,
        booking_confirmed=False,
    )

    base = CommunicationDecision(
        kind=CommunicationKind.VERIFY_BOOKING,
        offered_day="Friday",
        offered_time="2:30 PM",
    )

    result = apply_exploration_policy(
        scenario=scenario,
        grounded=grounded,
        progress=progress,
        agent_turn="Okay, goodbye.",
        base_decision=base,
    )

    assert (
        result.kind
        is CommunicationKind.VERIFY_BOOKING
    )

    assert result.offered_day == "Friday"
    assert result.offered_time == "2:30 PM"


def test_exploration_still_restates_objective_before_any_offer() -> None:
    scenario = get_scenario(
        "autonomous-phone-diagnostic"
    )

    meaning = TurnMeaning(
        topic="conversation ending",
        conversation_end_requested=True,
    )

    grounded = ground_turn_meaning(
        scenario=scenario,
        meaning=meaning,
    )

    progress = AppointmentProgress()

    base = CommunicationDecision(
        kind=CommunicationKind.DECLINE_WORKFLOW,
    )

    result = apply_exploration_policy(
        scenario=scenario,
        grounded=grounded,
        progress=progress,
        agent_turn="Okay, goodbye.",
        base_decision=base,
    )

    assert (
        result.kind
        is CommunicationKind.ANSWER
    )

    assert result.state_objective is True
