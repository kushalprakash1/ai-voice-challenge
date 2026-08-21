import asyncio

import pytest

from voiceprobe.v3.ingress import (
    FluxIngressController,
    RemoteSpeechBurstBuffer,
)
from voiceprobe.v3.models import DecisionKind


def test_idle_end_of_turn_is_decided_immediately() -> None:
    ingress = RemoteSpeechBurstBuffer()

    result = ingress.ingest_end_of_turn(
        "What is the reason for your visit?"
    )

    assert result is not None
    assert result.decision.kind == DecisionKind.ANSWER_COMPLAINT
    assert result.decision.text == "I have right shoulder pain."
    assert result.source_turns == (
        "What is the reason for your visit?",
    )
    assert result.emission_reason == "immediate_end_of_turn"


def test_suppressed_candidate_recoalesces_with_illustrative_continuation() -> None:
    ingress = RemoteSpeechBurstBuffer()
    first = ingress.ingest_end_of_turn(
        "Can you tell me the reason for your visit?"
    )
    assert first is not None

    ingress.mark_response_started()
    assert ingress.ingest_end_of_turn(
        "For example, is this a routine checkup, a follow-up, or urgent?"
    ) is None

    result = ingress.mark_response_suppressed()

    assert result is not None
    assert result.source_turns == (
        "Can you tell me the reason for your visit?",
        "For example, is this a routine checkup, a follow-up, or urgent?",
    )
    assert result.decision.kind == DecisionKind.ANSWER_COMPLAINT
    assert result.decision.text == "I have right shoulder pain."
    assert result.emission_reason == "suppressed_candidate_recoalesced"


def test_suppressed_candidate_without_continuation_is_not_reprocessed() -> None:
    ingress = RemoteSpeechBurstBuffer()
    first = ingress.ingest_end_of_turn(
        "Just to confirm, is this a new patient consultation?"
    )
    assert first is not None
    ingress.mark_response_started()

    assert ingress.mark_response_suppressed() is None


def test_busy_remote_speech_is_coalesced_not_fifo_replayed() -> None:
    ingress = RemoteSpeechBurstBuffer()
    ingress.mark_response_started()

    assert ingress.ingest_end_of_turn("Thanks, Alex.") is None
    assert (
        ingress.ingest_end_of_turn(
            "Let me check available appointments for you on Friday afternoon."
        )
        is None
    )
    assert (
        ingress.ingest_end_of_turn(
            "Thanks for confirming your date of birth, Alex."
        )
        is None
    )
    assert (
        ingress.ingest_end_of_turn(
            "What is the reason for your visit?"
        )
        is None
    )

    result = ingress.mark_response_finished()

    assert result is not None
    assert result.decision.kind == DecisionKind.ANSWER_COMPLAINT
    assert result.actionable_turn == "What is the reason for your visit?"
    assert result.decision.text == "I have right shoulder pain."
    assert len(result.source_turns) == 4
    assert result.buffered_turn_count == 3
    assert result.emission_reason == "buffered_burst_drained"


def test_latest_actionable_turn_wins_inside_busy_burst() -> None:
    ingress = RemoteSpeechBurstBuffer()
    ingress.mark_response_started()

    ingress.ingest_end_of_turn(
        "What is the reason for your visit?"
    )
    ingress.ingest_end_of_turn(
        (
            "We have openings on Friday afternoon with two providers. "
            "Would you prefer Dr. A or Dr. B, "
            "or is the first available okay?"
        )
    )

    result = ingress.mark_response_finished()

    assert result is not None
    assert (
        result.decision.kind
        == DecisionKind.ANSWER_PROVIDER_PREFERENCE
    )
    assert result.decision.text == "First available is fine."


def test_incomplete_turn_remains_hold() -> None:
    ingress = RemoteSpeechBurstBuffer()

    result = ingress.ingest_end_of_turn(
        "Would any..."
    )

    assert result is not None
    assert result.decision.kind == DecisionKind.HOLD
    assert not result.requires_response


def test_empty_transcript_is_ignored() -> None:
    ingress = RemoteSpeechBurstBuffer()

    assert ingress.ingest_end_of_turn("   ") is None


def test_pending_turns_can_be_cleared_on_call_teardown() -> None:
    ingress = RemoteSpeechBurstBuffer()
    ingress.mark_response_started()

    ingress.ingest_end_of_turn("Thanks, Alex.")
    ingress.ingest_end_of_turn("What is the reason for your visit?")

    assert ingress.clear_pending() == (
        "Thanks, Alex.",
        "What is the reason for your visit?",
    )
    assert ingress.pending_turns == ()


class FakeFluxService:
    def __init__(self) -> None:
        self.handlers = {}

    def event_handler(self, event_name):
        def decorator(func):
            self.handlers[event_name] = func
            return func

        return decorator


def test_controller_registers_documented_flux_events() -> None:
    service = FakeFluxService()
    emitted = []

    async def run() -> None:
        controller = FluxIngressController(
            on_decision=emitted.append,
            continuation_grace_ms=0,
        )
        controller.attach(service)

        assert set(service.handlers) == {
            "on_start_of_turn",
            "on_turn_resumed",
            "on_end_of_turn",
        }

        await service.handlers["on_start_of_turn"](
            service,
            "What is the reason",
        )
        await service.handlers["on_turn_resumed"](
            service,
        )
        await service.handlers["on_end_of_turn"](
            service,
            "What is the reason for your visit?",
        )

        assert controller.turn_resumed_count == 1

    asyncio.run(run())

    assert len(emitted) == 1
    assert emitted[0].decision.kind == DecisionKind.ANSWER_COMPLAINT


def test_controller_drains_one_decision_after_response_finishes() -> None:
    service = FakeFluxService()
    emitted = []

    async def run() -> None:
        controller = FluxIngressController(
            on_decision=emitted.append,
            continuation_grace_ms=0,
        )
        controller.attach(service)
        controller.mark_response_started()

        await service.handlers["on_end_of_turn"](
            service,
            "Thanks, Alex.",
        )
        await service.handlers["on_end_of_turn"](
            service,
            "What is the reason for your visit?",
        )

        assert emitted == []

        result = await controller.mark_response_finished()

        assert result is not None

    asyncio.run(run())

    assert len(emitted) == 1
    assert emitted[0].decision.kind == DecisionKind.ANSWER_COMPLAINT


def test_double_attach_is_rejected() -> None:
    controller = FluxIngressController(continuation_grace_ms=0)
    controller.attach(FakeFluxService())

    with pytest.raises(RuntimeError):
        controller.attach(FakeFluxService())
