
import asyncio

from voiceprobe.v32.semantic_parser import SemanticParser
from voiceprobe.v32.semantic_policy import (
    SemanticRoute,
    route_semantic_frame,
)


class FakeBackend:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def generate_json(self, **kwargs):
        self.calls += 1
        return self.response


def parse(response, text="example"):
    backend = FakeBackend(response)

    trace = asyncio.run(
        SemanticParser(
            backend=backend,
        ).parse(
            remote_turn=text,
            recent_dialogue=(
                "PGAI: You have an appointment Tuesday.",
                "PATIENT: I'd like Friday afternoon.",
            ),
        )
    )

    assert backend.calls == 1

    return trace


def test_reschedule_reason_routes_to_answer():
    trace = parse({
        "speech_act": "ask",
        "operation": "reschedule",
        "focus": "reschedule_reason",
        "commitment": "informational",
        "certainty": "high",
    })

    routed = route_semantic_frame(trace.frame)

    assert (
        routed.route
        is SemanticRoute.ANSWER_RESCHEDULE_REASON
    )


def test_explicit_insurance_overrides_prior_reschedule_context():
    trace = parse({
        "speech_act": "ask",
        "operation": "none",
        "focus": "insurance",
        "commitment": "informational",
        "certainty": "high",
    })

    routed = route_semantic_frame(trace.frame)

    assert routed.route is SemanticRoute.ANSWER_FACT
    assert routed.fact_focus.value == "insurance"


def test_booking_permission_is_transaction_gated():
    trace = parse({
        "speech_act": "ask",
        "operation": "book",
        "focus": "appointment_status",
        "commitment": "permission_request",
        "certainty": "high",
    })

    assert (
        route_semantic_frame(trace.frame).route
        is SemanticRoute.TRANSACTION_GATE
    )


def test_acknowledgement_waits():
    trace = parse({
        "speech_act": "acknowledge",
        "operation": "none",
        "focus": "none",
        "commitment": "none",
        "certainty": "high",
    })

    assert (
        route_semantic_frame(trace.frame).route
        is SemanticRoute.WAIT
    )


def test_slot_intro_holds():
    trace = parse({
        "speech_act": "inform",
        "operation": "list_slots",
        "focus": "slot_options_intro",
        "commitment": "informational",
        "certainty": "high",
    })

    assert (
        route_semantic_frame(trace.frame).route
        is SemanticRoute.HOLD
    )


def test_invalid_model_object_fails_closed():
    trace = parse({
        "speech_act": "banana",
        "operation": "none",
        "focus": "insurance",
        "commitment": "informational",
        "certainty": "high",
    })

    assert trace.validation_error is not None

    assert (
        route_semantic_frame(trace.frame).route
        is SemanticRoute.UNKNOWN
    )



def test_visit_reason_and_reschedule_reason_are_distinct():
    """Reason for care and reason for changing a visit are different intents."""

    visit = parse({
        "speech_act": "ask",
        "operation": "none",
        "focus": "visit_reason",
        "commitment": "informational",
        "certainty": "high",
    })

    visit_route = route_semantic_frame(
        visit.frame
    )

    assert visit_route.route is SemanticRoute.ANSWER_FACT
    assert visit_route.fact_focus.value == "complaint"

    reschedule = parse({
        "speech_act": "ask",
        "operation": "reschedule",
        "focus": "reschedule_reason",
        "commitment": "informational",
        "certainty": "high",
    })

    assert (
        route_semantic_frame(reschedule.frame).route
        is SemanticRoute.ANSWER_RESCHEDULE_REASON
    )


def test_visit_reason_never_becomes_transaction_gate():
    trace = parse({
        "speech_act": "ask",
        "operation": "none",
        "focus": "visit_reason",
        "commitment": "informational",
        "certainty": "high",
    })

    routed = route_semantic_frame(trace.frame)

    assert routed.route is SemanticRoute.ANSWER_FACT
    assert routed.fact_focus.value == "complaint"


def test_reschedule_reason_remains_non_transactional():
    trace = parse({
        "speech_act": "ask",
        "operation": "reschedule",
        "focus": "reschedule_reason",
        "commitment": "informational",
        "certainty": "high",
    })

    assert (
        route_semantic_frame(trace.frame).route
        is SemanticRoute.ANSWER_RESCHEDULE_REASON
    )
