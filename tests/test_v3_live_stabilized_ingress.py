import asyncio

from voiceprobe.v3.ingress import (
    FluxIngressController,
    RemoteSpeechBurstBuffer,
)
from voiceprobe.v3.models import DecisionKind


class FakeFluxService:
    def __init__(self) -> None:
        self.handlers = {}

    def event_handler(self, event_name):
        def decorator(func):
            self.handlers[event_name] = func
            return func

        return decorator


def test_burst_buffer_accepts_multiple_stabilized_turns() -> None:
    buffer = RemoteSpeechBurstBuffer()

    result = buffer.ingest_turns(
        (
            (
                "We have openings on Friday, August twenty first. "
                "The available times are nine AM and ten thirty AM. "
                "Would any of these work for your Friday afternoon?"
            ),
            "preference, or would you like to look at later dates or times?",
        ),
        emission_reason="stabilized_end_of_turn",
    )

    assert result is not None
    assert result.decision.kind == DecisionKind.DECLINE_INCOMPATIBLE_OFFER
    assert result.buffered_turn_count == 1
    assert result.emission_reason == "stabilized_end_of_turn"


def test_live_controller_does_not_emit_eot_before_stabilization_flush() -> None:
    service = FakeFluxService()
    emitted = []

    async def scenario() -> None:
        controller = FluxIngressController(
            on_decision=emitted.append,
            continuation_grace_ms=60_000,
        )
        controller.attach(service)

        await service.handlers["on_end_of_turn"](
            service,
            "What is the reason for your visit?",
        )

        assert emitted == []
        assert controller.pending_stabilized_turns == (
            "What is the reason for your visit?",
        )

        result = await controller.flush_stabilized_pending()

        assert result is not None
        assert result.decision.kind == DecisionKind.ANSWER_COMPLAINT

    asyncio.run(scenario())

    assert len(emitted) == 1


def test_start_of_turn_within_grace_merges_next_eot() -> None:
    service = FakeFluxService()
    emitted = []

    first = (
        "We have openings for new patient consultation on Friday, "
        "August twenty first. The available times are nine AM, "
        "nine forty five AM, and ten thirty AM. "
        "Would any of these work for your Friday afternoon?"
    )
    continuation = (
        "preference, or would you like to look at later dates or times?"
    )

    async def scenario() -> None:
        controller = FluxIngressController(
            on_decision=emitted.append,
            continuation_grace_ms=60_000,
        )
        controller.attach(service)

        await service.handlers["on_end_of_turn"](service, first)
        await service.handlers["on_start_of_turn"](
            service,
            "preference",
        )

        assert emitted == []
        assert controller.pending_stabilized_turns == (first,)

        await service.handlers["on_end_of_turn"](
            service,
            continuation,
        )

        result = await controller.flush_stabilized_pending()

        assert result is not None
        assert result.source_turns == (first, continuation)
        assert result.decision.kind == DecisionKind.DECLINE_INCOMPATIBLE_OFFER
        assert result.decision.text == (
            "Those times don't work for me. "
            "Do you have anything Friday afternoon?"
        )

    asyncio.run(scenario())

    assert len(emitted) == 1


def test_turn_resumed_cancels_pending_stabilization_timer() -> None:
    service = FakeFluxService()
    emitted = []

    async def scenario() -> None:
        controller = FluxIngressController(
            on_decision=emitted.append,
            continuation_grace_ms=60_000,
        )
        controller.attach(service)

        await service.handlers["on_end_of_turn"](
            service,
            "Would you like to look at later dates",
        )
        await service.handlers["on_turn_resumed"](service)

        assert controller.turn_resumed_count == 1
        assert controller.pending_stabilized_turns == (
            "Would you like to look at later dates",
        )
        assert emitted == []

        controller.clear_pending()

    asyncio.run(scenario())


def test_busy_response_still_coalesces_after_stabilization() -> None:
    service = FakeFluxService()
    emitted = []

    async def scenario() -> None:
        controller = FluxIngressController(
            on_decision=emitted.append,
            continuation_grace_ms=60_000,
        )
        controller.attach(service)
        controller.mark_response_started()

        await service.handlers["on_end_of_turn"](
            service,
            "Thanks, Alex.",
        )
        await controller.flush_stabilized_pending()

        await service.handlers["on_end_of_turn"](
            service,
            "What is the reason for your visit?",
        )
        await controller.flush_stabilized_pending()

        assert emitted == []

        result = await controller.mark_response_finished()

        assert result is not None
        assert result.decision.kind == DecisionKind.ANSWER_COMPLAINT
        assert len(result.source_turns) == 2

    asyncio.run(scenario())

    assert len(emitted) == 1


def test_clear_pending_discards_stabilized_and_busy_turns() -> None:
    service = FakeFluxService()

    async def scenario() -> None:
        controller = FluxIngressController(
            continuation_grace_ms=60_000,
        )
        controller.attach(service)

        await service.handlers["on_end_of_turn"](
            service,
            "What is the reason for your visit?",
        )

        cleared = controller.clear_pending()

        assert cleared == (
            "What is the reason for your visit?",
        )
        assert controller.pending_stabilized_turns == ()
        assert controller.burst_buffer.pending_turns == ()

    asyncio.run(scenario())


def test_zero_grace_preserves_immediate_controller_mode_for_unit_tests() -> None:
    service = FakeFluxService()
    emitted = []

    async def scenario() -> None:
        controller = FluxIngressController(
            on_decision=emitted.append,
            continuation_grace_ms=0,
        )
        controller.attach(service)

        await service.handlers["on_end_of_turn"](
            service,
            "What is the reason for your visit?",
        )

    asyncio.run(scenario())

    assert len(emitted) == 1
    assert emitted[0].decision.kind == DecisionKind.ANSWER_COMPLAINT
