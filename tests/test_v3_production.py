import asyncio
from dataclasses import replace

import pytest

from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.production import (
    DEFAULT_PRODUCTION_FLUX_CONFIG,
    PipecatRuntimeBridge,
    ProductionFluxConfig,
)


class FakeSpeechFrame:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeWorker:
    def __init__(self) -> None:
        self.frames = []

    async def queue_frames(self, frames) -> None:
        self.frames.extend(frames)


class FakeFlux:
    def __init__(self) -> None:
        self.handlers = {}

    def event_handler(self, name):
        def decorator(func):
            self.handlers[name] = func
            return func

        return decorator


def make_bridge(*, grace_ms: float = 0.0):
    config = replace(
        DEFAULT_PRODUCTION_FLUX_CONFIG,
        continuation_grace_ms=grace_ms,
    )
    bridge = PipecatRuntimeBridge(
        config=config,
        tts_frame_factory=FakeSpeechFrame,
    )
    worker = FakeWorker()
    bridge.bind_worker(worker)
    return bridge, worker


def test_production_flux_config_is_frozen_to_proven_settings() -> None:
    config = DEFAULT_PRODUCTION_FLUX_CONFIG

    assert config.model == "flux-general-en"
    assert config.sample_rate == 8000
    assert config.eot_threshold == 0.85
    assert config.eot_timeout_ms == 5000
    assert config.eager_eot_threshold is None
    assert config.continuation_grace_ms == 600.0
    assert config.flux_encoding == "linear16"
    assert "Pivot Point" in config.keyterms
    assert "Blue Cross" in config.keyterms


def test_production_config_rejects_eager_eot() -> None:
    config = ProductionFluxConfig(eager_eot_threshold=0.5)

    with pytest.raises(ValueError):
        config.validate()


def test_bridge_queues_only_response_ready_decisions() -> None:
    async def scenario():
        bridge, worker = make_bridge()

        wait = await bridge.runtime.process_turns(["Thanks, Alex."])
        assert wait.decision.kind == DecisionKind.WAIT
        assert worker.frames == []

        answer = await bridge.runtime.process_turns(
            ["What is the reason for your visit?"]
        )
        assert answer.decision.kind == DecisionKind.ANSWER_COMPLAINT
        assert len(worker.frames) == 1
        assert worker.frames[0].text == "I have right shoulder pain."
        assert bridge.queued_speech_count == 1

    asyncio.run(scenario())


def test_bridge_does_not_speak_unresolved_fallback() -> None:
    async def scenario():
        bridge, worker = make_bridge()

        result = await bridge.runtime.process_turns(
            ["Could you unpack the metaphysics of this appointment?"]
        )

        assert result.decision.kind == DecisionKind.FALLBACK
        assert result.response_ready is False
        assert worker.frames == []

    asyncio.run(scenario())


def test_bridge_marks_busy_before_tts_and_drains_after_tts_stops() -> None:
    async def scenario():
        bridge, worker = make_bridge()
        stt = FakeFlux()
        bridge.attach_flux(stt)

        await bridge.runtime.process_turns(
            ["What is the reason for your visit?"]
        )
        assert len(worker.frames) == 1

        # While the first response is waiting for TTS completion, the remote
        # provider speaks again. It must be preserved, not answered in parallel.
        await stt.handlers["on_end_of_turn"](
            stt,
            (
                "We have openings on Friday afternoon with two providers. "
                "Would you prefer either provider or is first available okay?"
            ),
        )

        assert len(worker.frames) == 1
        assert bridge.runtime.ingress.burst_buffer.pending_turns

        await bridge.on_tts_stopped()

        assert len(worker.frames) == 2
        assert worker.frames[1].text == "First available is fine."

    asyncio.run(scenario())


def test_previous_call_short_continuation_is_one_live_decision() -> None:
    async def scenario():
        bridge, worker = make_bridge(grace_ms=60_000)
        stt = FakeFlux()
        bridge.attach_flux(stt)

        first = (
            "We have openings for new patient consultation on Friday, "
            "August twenty first. The available times are nine AM, "
            "nine forty five AM, and ten thirty AM. "
            "Would any of these work for your Friday afternoon?"
        )
        continuation = (
            "preference, or would you like to look at later dates or times?"
        )

        await stt.handlers["on_end_of_turn"](stt, first)
        await stt.handlers["on_start_of_turn"](stt, "preference")
        await stt.handlers["on_end_of_turn"](stt, continuation)

        assert worker.frames == []

        await bridge.runtime.ingress.flush_stabilized_pending()

        assert len(worker.frames) == 1
        assert worker.frames[0].text == (
            "Those times don't work for me. "
            "Do you have anything Friday afternoon?"
        )

    asyncio.run(scenario())


def test_bridge_rejects_worker_without_queue_frames() -> None:
    bridge = PipecatRuntimeBridge(
        tts_frame_factory=FakeSpeechFrame,
    )

    with pytest.raises(TypeError):
        bridge.bind_worker(object())


def test_clear_pending_returns_stabilized_text() -> None:
    async def scenario():
        bridge, _ = make_bridge(grace_ms=60_000)
        stt = FakeFlux()
        bridge.attach_flux(stt)

        await stt.handlers["on_end_of_turn"](
            stt,
            "What is the reason for your visit?",
        )

        cleared = bridge.clear_pending()

        assert cleared == ("What is the reason for your visit?",)
        assert bridge.runtime.ingress.pending_stabilized_turns == ()

    asyncio.run(scenario())
