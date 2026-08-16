import asyncio

from voiceprobe.v3.flow_state import FlowStage
from voiceprobe.v3.models import DecisionKind, PolicyDecision
from voiceprobe.v3.runtime import (
    DecisionRoute,
    VoiceProbeV3Runtime,
)


def run(coro):
    return asyncio.run(coro)


def test_runtime_processes_routine_turn_without_fallback() -> None:
    runtime = VoiceProbeV3Runtime()

    result = run(
        runtime.process_turns(
            ["What is the reason for your visit?"]
        )
    )

    assert result.route == DecisionRoute.DETERMINISTIC
    assert result.decision.kind == DecisionKind.ANSWER_COMPLAINT
    assert result.decision.text == "I have right shoulder pain."
    assert result.policy_latency_ms >= 0.0
    assert runtime.metrics.fallback_decisions == 0


def test_runtime_busy_burst_discards_stale_non_actionable_work() -> None:
    emitted = []
    runtime = VoiceProbeV3Runtime(
        on_decision=emitted.append,
        continuation_grace_ms=0,
    )

    class FakeFlux:
        def __init__(self):
            self.handlers = {}

        def event_handler(self, name):
            def decorator(func):
                self.handlers[name] = func
                return func
            return decorator

    service = FakeFlux()
    runtime.attach_flux(service)
    runtime.mark_response_started()

    async def scenario():
        for turn in (
            "Thanks, Alex.",
            "Let me check available appointments for you on Friday afternoon.",
            "Thanks for confirming your date of birth, Alex.",
            "What is the reason for your visit?",
        ):
            await service.handlers["on_end_of_turn"](
                service,
                turn,
            )

        assert emitted == []
        await runtime.mark_response_finished()

    run(scenario())

    assert len(emitted) == 1
    assert emitted[0].decision.kind == DecisionKind.ANSWER_COMPLAINT
    assert emitted[0].actionable_turn == "What is the reason for your visit?"
    assert len(emitted[0].source_turns) == 4


def test_runtime_updates_flow_state_from_profile_sequence() -> None:
    runtime = VoiceProbeV3Runtime()

    first = run(
        runtime.process_turns(
            [
                (
                    "Would you like to create a demo patient profile? "
                    "I just need your first and last name."
                )
            ]
        )
    )

    assert first.decision.kind == DecisionKind.CREATE_PROFILE
    assert FlowStage.PROFILE in first.after.communicated

    second = run(
        runtime.process_turns(
            ["Your demo patient profile is set up."]
        )
    )

    assert FlowStage.PROFILE in second.after.confirmed
    assert FlowStage.IDENTITY in second.after.confirmed


def test_runtime_provider_choice_uses_fast_policy() -> None:
    runtime = VoiceProbeV3Runtime()

    result = run(
        runtime.process_turns(
            [
                (
                    "We have openings on Friday afternoon. "
                    "Would you prefer Dr. A or Dr. B, "
                    "or is the first available okay?"
                )
            ]
        )
    )

    assert result.decision.kind == DecisionKind.ANSWER_PROVIDER_PREFERENCE
    assert result.decision.text == "First available is fine."
    assert FlowStage.PROVIDER in result.after.communicated


def test_runtime_fallback_is_injected_and_cannot_mutate_state_directly() -> None:
    calls = []

    async def fallback(turn, snapshot):
        calls.append((turn, snapshot))
        return PolicyDecision(
            kind=DecisionKind.STATE_OBJECTIVE,
            text="I need to schedule an appointment for Friday afternoon.",
            reason="test_fallback",
        )

    runtime = VoiceProbeV3Runtime(
        fallback_resolver=fallback,
    )

    result = run(
        runtime.process_turns(
            ["Could you unpack what brings you into the clinic today?"]
        )
    )

    assert result.route == DecisionRoute.FALLBACK
    assert len(calls) == 1
    assert result.decision.kind == DecisionKind.STATE_OBJECTIVE
    assert FlowStage.DATE_TIME in result.after.communicated
    assert runtime.metrics.fallback_decisions == 1


def test_runtime_without_fallback_resolver_keeps_fallback_explicit() -> None:
    runtime = VoiceProbeV3Runtime()

    result = run(
        runtime.process_turns(
            ["Could you unpack what brings you into the clinic today?"]
        )
    )

    assert result.route == DecisionRoute.FALLBACK
    assert result.decision.kind == DecisionKind.FALLBACK
    assert result.requires_response
    assert not result.response_ready


def test_runtime_metrics_count_wait_hold_and_response() -> None:
    runtime = VoiceProbeV3Runtime()

    run(runtime.process_turns(["Thanks, Alex."]))
    run(runtime.process_turns(["Would any..."]))
    run(runtime.process_turns(["What is the reason for your visit?"]))

    metrics = runtime.metrics

    assert metrics.total_decisions == 3
    assert metrics.waits == 1
    assert metrics.holds == 1
    assert metrics.response_required == 1
    assert metrics.deterministic_decisions == 3
    assert metrics.average_policy_latency_ms >= 0.0


def test_runtime_does_not_false_complete_from_slot_status() -> None:
    runtime = VoiceProbeV3Runtime()

    result = run(
        runtime.process_turns(
            ["I found an appointment Friday at 2:30 PM."]
        )
    )

    assert not result.after.complete
    assert FlowStage.CONFIRMATION not in result.after.confirmed
