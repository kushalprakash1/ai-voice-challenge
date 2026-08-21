import asyncio

from voiceprobe.v3.fast_policy import RoutineSchedulingPolicy
from voiceprobe.v3.flow_state import SchedulingFlowTracker
from voiceprobe.v3.models import DecisionKind, PolicyDecision
from voiceprobe.v3.runtime import VoiceProbeV3Runtime
from voiceprobe.v3.semantic_router import (
    SemanticClassification,
    SemanticIntent,
    V31SemanticRouter,
    _complex_scheduling_action,
)


RUN5_SPLIT_BRANCH = (
    "There are no Friday afternoon openings available. "
    "Would you like to look at afternoon slots on another day?",
    "or check for the next available Friday.",
)

RUN5_TUESDAY_BRANCH = (
    "Monday's available times are also in the morning, nine, "
    "nine forty five, and ten thirty AM with Averger. "
    "Would you like to check afternoon slots on Tuesday "
    "or another specific day?"
)

RUN5_SLOT_OFFER = (
    "The first available weekday afternoon slot is Tuesday, "
    "August twenty fifth. You can book at two fifteen PM, "
    "three PM, or three forty five PM with Doogie Hauser at "
    "Pivot Point Orthopedics. Would you like one of these times?"
)


def test_fallback_receives_complete_stabilized_run5_burst():
    observed = []

    async def resolver(agent_turn, snapshot):
        del snapshot
        observed.append(agent_turn)
        return PolicyDecision(
            DecisionKind.WAIT,
            reason="run5_capture",
        )

    async def scenario():
        runtime = VoiceProbeV3Runtime(
            fallback_resolver=resolver,
        )

        result = await runtime.process_turns(
            RUN5_SPLIT_BRANCH,
        )

        assert result.route.value == "fallback"

    asyncio.run(scenario())

    assert observed == [
        " ".join(RUN5_SPLIT_BRANCH)
    ]


def test_grounding_prefers_proposed_tuesday_over_monday_morning():
    tracker = SchedulingFlowTracker()
    tracker.relax_day_constraint_for_afternoon()

    decision = _complex_scheduling_action(
        RUN5_TUESDAY_BRANCH,
        tracker.snapshot(),
        0.90,
    )

    assert decision.kind == (
        DecisionKind.SEARCH_ALTERNATE_DAY_AFTERNOON
    )
    assert decision.text == "Please check Tuesday afternoon."


def test_existing_run4_example_still_selects_first_grounded_monday():
    turn = (
        "There are no Friday afternoon openings available. "
        "Would you like to look at afternoon slots on another day "
        "such as Monday or Tuesday next week?"
    )

    decision = _complex_scheduling_action(
        turn,
        SchedulingFlowTracker().snapshot(),
        0.90,
    )

    assert decision.text == "Please check Monday afternoon."


def test_relaxed_date_time_semantic_turn_selects_tuesday_branch():
    class DateTimeRouter(V31SemanticRouter):
        async def classify(self, agent_turn, snapshot):
            del agent_turn, snapshot
            return SemanticClassification(
                intent=SemanticIntent.DATE_TIME_PREFERENCE_REQUEST,
                confidence=0.90,
                source="run5_regression",
                score=0.90,
                margin=0.15,
            )

    async def scenario():
        tracker = SchedulingFlowTracker()
        tracker.relax_day_constraint_for_afternoon()

        router = DateTimeRouter(
            use_embeddings=False,
        )

        decision = await router.resolve(
            RUN5_TUESDAY_BRANCH,
            tracker.snapshot(),
        )

        assert decision.kind == (
            DecisionKind.SEARCH_ALTERNATE_DAY_AFTERNOON
        )

        assert decision.text == (
            "Please check Tuesday afternoon."
        )

        assert decision.kind != DecisionKind.STATE_OBJECTIVE

    asyncio.run(scenario())


def test_multi_slot_offer_explicitly_names_first_selected_slot():
    policy = RoutineSchedulingPolicy()
    policy.relax_day_constraint_for_afternoon()

    decision = policy.decide(RUN5_SLOT_OFFER)

    assert decision.kind == DecisionKind.GRANT_PERMISSION
    assert decision.reason == (
        "compatible_concrete_slot_offered"
    )

    assert decision.text == (
        "Yes, please book the two fifteen PM slot."
    )
