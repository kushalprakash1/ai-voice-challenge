from voiceprobe.v3.flow_state import (
    SchedulingFlowTracker,
)
from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.production import (
    DEFAULT_PRODUCTION_FLUX_CONFIG,
)
from voiceprobe.v3.semantic_router import (
    _complex_scheduling_action,
)

LIVE_RUN4_TURN = (
    "There are no Friday afternoon openings available. "
    "Would you like to look at afternoon slots on another day "
    "such as Monday or Tuesday next week?"
)


def test_live_run4_complex_turn_produces_action_not_clarify():
    tracker = SchedulingFlowTracker()

    decision = _complex_scheduling_action(
        LIVE_RUN4_TURN,
        tracker.snapshot(),
        0.90,
    )

    assert decision.kind == (
        DecisionKind.SEARCH_ALTERNATE_DAY_AFTERNOON
    )

    assert decision.text == (
        "Please check Monday afternoon."
    )

    assert decision.reason == (
        "semantic_v31:choose_alternate_day_afternoon"
    )

    assert decision.kind != DecisionKind.CLARIFY
    assert "restate" not in decision.text.casefold()


def test_live_run4_action_durably_relaxes_only_day_constraint():
    tracker = SchedulingFlowTracker()

    decision = _complex_scheduling_action(
        LIVE_RUN4_TURN,
        tracker.snapshot(),
        0.90,
    )

    after = tracker.apply_decision(decision)

    assert after.allow_earlier_week_afternoons is True


def test_complex_turn_without_specific_day_still_never_repeats():
    tracker = SchedulingFlowTracker()

    decision = _complex_scheduling_action(
        "Would you like me to broaden the appointment search?",
        tracker.snapshot(),
        0.90,
    )

    assert decision.kind == (
        DecisionKind.SEARCH_ALTERNATE_DAY_AFTERNOON
    )

    assert decision.kind != DecisionKind.CLARIFY
    assert "restate" not in decision.text.casefold()
    assert "afternoon" in decision.text.casefold()


def test_live_production_uses_3000ms_continuation_grace():
    assert (
        DEFAULT_PRODUCTION_FLUX_CONFIG.continuation_grace_ms
        == 3000.0
    )


def test_live_run4_full_resolve_path_carries_snapshot():
    import asyncio

    from voiceprobe.v3.semantic_router import (
        SemanticClassification,
        SemanticIntent,
        V31SemanticRouter,
    )

    class Run4Router(V31SemanticRouter):
        async def classify(self, agent_turn, snapshot):
            return SemanticClassification(
                intent=SemanticIntent.SCHEDULING_COMPLEX,
                confidence=0.90,
                source="run4_regression",
                score=0.90,
                margin=0.15,
            )

    async def scenario():
        router = Run4Router(use_embeddings=False)
        tracker = SchedulingFlowTracker()

        decision = await router.resolve(
            LIVE_RUN4_TURN,
            tracker.snapshot(),
        )

        assert decision.kind == (
            DecisionKind.SEARCH_ALTERNATE_DAY_AFTERNOON
        )
        assert decision.text == (
            "Please check Monday afternoon."
        )
        assert decision.kind != DecisionKind.CLARIFY

    asyncio.run(scenario())


def test_live_run4_runtime_resolve_then_accepts_pm_slot():
    import asyncio

    from voiceprobe.v3.runtime import VoiceProbeV3Runtime
    from voiceprobe.v3.semantic_router import (
        SemanticClassification,
        SemanticIntent,
        V31SemanticRouter,
    )

    class Run4Router(V31SemanticRouter):
        async def classify(self, agent_turn, snapshot):
            return SemanticClassification(
                intent=SemanticIntent.SCHEDULING_COMPLEX,
                confidence=0.90,
                source="run4_regression",
                score=0.90,
                margin=0.15,
            )

    async def scenario():
        router = Run4Router(use_embeddings=False)

        runtime = VoiceProbeV3Runtime(
            fallback_resolver=router.resolve,
        )

        branch = await runtime.process_turns(
            [LIVE_RUN4_TURN]
        )

        assert branch.decision.kind == (
            DecisionKind.SEARCH_ALTERNATE_DAY_AFTERNOON
        )
        assert branch.after.allow_earlier_week_afternoons

        slot = await runtime.process_turns([
            (
                "I have Monday at 2:30 PM. "
                "Would that work for you?"
            )
        ])

        assert slot.decision.kind == (
            DecisionKind.GRANT_PERMISSION
        )
        assert slot.decision.reason == (
            "compatible_concrete_slot_offered"
        )

    asyncio.run(scenario())


def test_morning_scenario_never_uses_afternoon_relaxation_action():
    import asyncio

    from voiceprobe.v3.models import PatientFacts
    from voiceprobe.v3.semantic_router import (
        SemanticClassification,
        SemanticIntent,
        V31SemanticRouter,
    )

    class MorningRouter(V31SemanticRouter):
        async def classify(self, agent_turn, snapshot):
            return SemanticClassification(
                intent=SemanticIntent.SCHEDULING_COMPLEX,
                confidence=0.90,
                source="morning_guard_regression",
                score=0.90,
                margin=0.15,
            )

    async def scenario():
        router = MorningRouter(
            facts=PatientFacts(
                preferred_day="Tuesday",
                preferred_time="morning",
            ),
            use_embeddings=False,
        )

        tracker = SchedulingFlowTracker()

        decision = await router.resolve(
            (
                "There are no Tuesday morning openings. "
                "Would you like me to check another day?"
            ),
            tracker.snapshot(),
        )

        assert (
            decision.kind
            != DecisionKind.SEARCH_ALTERNATE_DAY_AFTERNOON
        )

        assert (
            tracker.snapshot().allow_earlier_week_afternoons
            is False
        )

    asyncio.run(scenario())
