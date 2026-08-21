from voiceprobe.v3.flow_state import (
    FlowStage,
    SchedulingFlowTracker,
)
from voiceprobe.v3.models import (
    DecisionKind,
    PatientFacts,
)

from voiceprobe.v32.semantic_adapter import (
    resolve_semantic_turn,
)
from voiceprobe.v32.semantic_frame import Focus
from voiceprobe.v32.semantic_policy import (
    RoutedSemanticTurn,
    SemanticRoute,
)


FACTS = PatientFacts()


def resolve(route, focus=Focus.NONE):
    return resolve_semantic_turn(
        RoutedSemanticTurn(
            route=route,
            fact_focus=focus,
        ),
        facts=FACTS,
    )


def test_reschedule_explanation_has_zero_durable_mutation():
    tracker = SchedulingFlowTracker()
    before = tracker.snapshot()

    resolution = resolve(
        SemanticRoute.ANSWER_RESCHEDULE_REASON
    )

    assert resolution.decision is not None
    assert (
        resolution.decision.kind
        is DecisionKind.CONTEXTUAL_ANSWER
    )

    after = tracker.apply_decision(
        resolution.decision
    )

    assert after == before
    assert after.accepted_slot_text is None
    assert after.booking_confirmation_text is None
    assert not after.complete


def test_transaction_gate_never_creates_policy_decision():
    tracker = SchedulingFlowTracker()
    before = tracker.snapshot()

    resolution = resolve(
        SemanticRoute.TRANSACTION_GATE
    )

    assert resolution.decision is None
    assert resolution.delegates_transaction
    assert tracker.snapshot() == before


def test_wait_and_hold_have_zero_durable_mutation():
    for route in (
        SemanticRoute.WAIT,
        SemanticRoute.HOLD,
    ):
        tracker = SchedulingFlowTracker()
        before = tracker.snapshot()

        resolution = resolve(route)

        assert resolution.decision is not None

        after = tracker.apply_decision(
            resolution.decision
        )

        assert after == before


def test_visit_reason_marks_only_visit_reason():
    tracker = SchedulingFlowTracker()

    resolution = resolve(
        SemanticRoute.ANSWER_FACT,
        Focus.COMPLAINT,
    )

    assert resolution.decision is not None
    assert (
        resolution.decision.kind
        is DecisionKind.ANSWER_COMPLAINT
    )
    assert resolution.decision.text == (
        "I have right shoulder pain."
    )

    after = tracker.apply_decision(
        resolution.decision
    )

    assert after.communicated == frozenset({
        FlowStage.VISIT_REASON,
    })

    assert after.accepted_slot_text is None
    assert not after.complete


def test_provider_preference_marks_only_provider():
    tracker = SchedulingFlowTracker()

    resolution = resolve(
        SemanticRoute.ANSWER_FACT,
        Focus.PROVIDER_PREFERENCE,
    )

    assert resolution.decision is not None

    after = tracker.apply_decision(
        resolution.decision
    )

    assert after.communicated == frozenset({
        FlowStage.PROVIDER,
    })

    assert after.accepted_slot_text is None
    assert not after.complete


def test_insurance_uses_python_fact_and_marks_only_insurance():
    tracker = SchedulingFlowTracker()

    resolution = resolve(
        SemanticRoute.ANSWER_FACT,
        Focus.INSURANCE,
    )

    assert resolution.decision is not None
    assert resolution.decision.text == "Blue Cross."

    after = tracker.apply_decision(
        resolution.decision
    )

    assert after.communicated == frozenset({
        FlowStage.INSURANCE,
    })

    assert after.accepted_slot_text is None
    assert not after.complete


def test_ambiguous_name_grounding_fails_closed():
    resolution = resolve(
        SemanticRoute.ANSWER_FACT,
        Focus.NAME,
    )

    assert resolution.route is SemanticRoute.UNKNOWN
    assert resolution.decision is None
