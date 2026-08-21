from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import ClassVar

import pytest

from voiceprobe.v3 import production
from voiceprobe.v3.ingress import FluxIngressController
from voiceprobe.v3.production import DEFAULT_PRODUCTION_FLUX_CONFIG


class FakeFlux:
    def __init__(self) -> None:
        self.handlers = {}

    def event_handler(self, name):
        def decorate(handler):
            self.handlers[name] = handler
            return handler
        return decorate


class FakeQwen:
    instances: ClassVar[list[FakeQwen]] = []

    def __init__(self) -> None:
        self.calls = 0
        self.last_observation = {}
        self.__class__.instances.append(self)

    async def resolve(self, *_):
        self.calls += 1
        raise AssertionError("initial profile prerequisite must not invoke Qwen")


class Sink:
    def __init__(self) -> None:
        self.frames = []
        self.quiet_overrides = []
        self.queued_at = None

    def set_next_outbound_commit_quiet_ms(self, value):
        self.quiet_overrides.append(value)

    async def queue_frames(self, frames):
        self.queued_at = time.perf_counter()
        self.frames.extend(frames)


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv("VOICEPROBE_SCENARIO", "doctor-specialist-directory")
    monkeypatch.setattr(production, "DoctorDirectoryQwenRouter", FakeQwen)
    FakeQwen.instances.clear()
    instance = production.PipecatRuntimeBridge(
        config=replace(DEFAULT_PRODUCTION_FLUX_CONFIG, continuation_grace_ms=3000),
        tts_frame_factory=lambda text: type("Frame", (), {"text": text})(),
    )
    sink = Sink()
    flux = FakeFlux()
    instance.bind_frame_sink(sink)
    instance.attach_flux(flux)
    return instance, sink, flux


@pytest.mark.asyncio
async def test_exact_failed_greeting_beats_remote_patience_and_commits_on_delivery(bridge):
    instance, sink, flux = bridge
    target = (
        "Would you like to create a demo patient profile? "
        "I just need your first and last name."
    )
    target_end = time.perf_counter()
    await flux.handlers["on_end_of_turn"](flux, target)

    await asyncio.sleep(0.32)
    latency_ms = (sink.queued_at - target_end) * 1000
    assert latency_ms < 700
    assert len(sink.frames) == 1
    assert sink.frames[0].text == "Yes, please. my name is Gyeong-hyeon Gwak."
    assert sink.quiet_overrides == []
    assert FakeQwen.instances[0].calls == 0
    assert instance.runtime.decisions[0].decision.kind.value == "create_profile"
    assert instance.scenario_metadata["profile_consent_spoken"] is False

    await instance.on_tts_stopped()
    assert instance.scenario_metadata["profile_consent_spoken"] is True


@pytest.mark.asyncio
async def test_partial_profile_question_does_not_speak_before_continuation(bridge):
    instance, sink, flux = bridge
    await flux.handlers["on_end_of_turn"](flux, "Would you like to create a demo...")
    await asyncio.sleep(0.32)
    assert sink.frames == []
    await flux.handlers["on_start_of_turn"](flux, "patient profile")
    await flux.handlers["on_end_of_turn"](
        flux, "patient profile? I just need your first and last name."
    )
    await asyncio.sleep(0.32)
    assert sink.frames == []
    assert instance.runtime.ingress.continuation_grace_ms == 3000
    instance.clear_pending()


@pytest.mark.asyncio
async def test_remote_resume_during_short_hold_suppresses_candidate(bridge):
    instance, sink, flux = bridge
    await flux.handlers["on_end_of_turn"](
        flux, "Do you want to make a demo patient profile?"
    )
    await asyncio.sleep(0.05)
    await flux.handlers["on_turn_resumed"](flux)
    await asyncio.sleep(0.30)
    assert sink.frames == []
    assert instance.scenario_metadata["profile_consent_spoken"] is False
    instance.clear_pending()


@pytest.mark.asyncio
async def test_duplicate_flux_eot_queues_initial_response_exactly_once(bridge):
    instance, sink, flux = bridge
    target = "Do you want to make a demo patient profile?"
    await flux.handlers["on_end_of_turn"](flux, target)
    await asyncio.sleep(0.30)
    await flux.handlers["on_end_of_turn"](flux, target)
    await instance.on_tts_stopped()
    await asyncio.sleep(0.05)
    assert len(sink.frames) == 1


@pytest.mark.asyncio
async def test_unrelated_initial_greeting_keeps_global_grace():
    flux = FakeFlux()
    emitted = []
    controller = FluxIngressController(
        on_decision=emitted.append,
        continuation_grace_ms=3000,
        fast_stabilization_predicate=lambda text: "profile" in text.casefold(),
    )
    controller.attach(flux)
    await flux.handlers["on_end_of_turn"](flux, "How may I help you today?")
    await asyncio.sleep(0.32)
    assert emitted == []
    assert controller.continuation_grace_ms == 3000
    controller.clear_pending()


@pytest.mark.asyncio
async def test_fast_profile_response_uses_global_grace_without_commit_override(
    monkeypatch,
):
    monkeypatch.setenv("VOICEPROBE_SCENARIO", "doctor-specialist-directory")
    monkeypatch.setattr(production, "DoctorDirectoryQwenRouter", FakeQwen)
    instance = production.PipecatRuntimeBridge(
        config=replace(DEFAULT_PRODUCTION_FLUX_CONFIG, continuation_grace_ms=3000),
        tts_frame_factory=lambda text: type("Frame", (), {"text": text})(),
    )
    sink = Sink()
    flux = FakeFlux()
    instance.bind_frame_sink(sink)
    instance.attach_flux(flux)

    target_end = time.perf_counter()
    await flux.handlers["on_end_of_turn"](
        flux, "Do you want to make a demo patient profile?"
    )
    await asyncio.sleep(0.27)
    assert len(sink.frames) == 1
    assert (sink.queued_at - target_end) * 1000 < 700
    assert sink.quiet_overrides == []
    assert instance.scenario_metadata["profile_consent_spoken"] is False
    await instance.on_tts_stopped()
    assert instance.scenario_metadata["profile_consent_spoken"] is True
