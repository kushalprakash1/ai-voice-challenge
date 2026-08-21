import asyncio

import pytest

import voiceprobe.v3.production as production
from voiceprobe.v3.models import DecisionKind
from voiceprobe.v32.runtime_fallback import (
    V32SemanticFallbackResolver,
)
from voiceprobe.v32.semantic_parser import SemanticParser


class FakeSpeechFrame:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeWorker:
    def __init__(self) -> None:
        self.frames = []

    async def queue_frames(self, frames) -> None:
        self.frames.extend(frames)


class FakeBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def generate_json(self, **kwargs):
        del kwargs

        self.calls += 1
        return self.responses.pop(0)


def clear_v32_env(monkeypatch):
    for key in (
        "VOICEPROBE_V32_SEMANTIC",
        "VOICEPROBE_V32_OLLAMA_ENDPOINT",
        "VOICEPROBE_V32_MODEL",
    ):
        monkeypatch.delenv(
            key,
            raising=False,
        )


def test_default_production_mode_remains_v31(
    monkeypatch,
):
    clear_v32_env(monkeypatch)

    bridge = production.PipecatRuntimeBridge(
        tts_frame_factory=FakeSpeechFrame,
    )

    assert bridge.semantic_mode == "v31"
    assert bridge.v32_semantic_resolver is None


def test_v32_requires_explicit_endpoint(
    monkeypatch,
):
    clear_v32_env(monkeypatch)

    monkeypatch.setenv(
        "VOICEPROBE_V32_SEMANTIC",
        "1",
    )

    with pytest.raises(
        ValueError,
        match="VOICEPROBE_V32_OLLAMA_ENDPOINT",
    ):
        production.PipecatRuntimeBridge(
            tts_frame_factory=FakeSpeechFrame,
        )


def test_invalid_feature_flag_fails_closed(
    monkeypatch,
):
    clear_v32_env(monkeypatch)

    monkeypatch.setenv(
        "VOICEPROBE_V32_SEMANTIC",
        "maybe",
    )

    with pytest.raises(
        ValueError,
        match="VOICEPROBE_V32_SEMANTIC",
    ):
        production.PipecatRuntimeBridge(
            tts_frame_factory=FakeSpeechFrame,
        )


def test_v32_feature_flag_selects_v32(
    monkeypatch,
):
    clear_v32_env(monkeypatch)

    monkeypatch.setenv(
        "VOICEPROBE_V32_SEMANTIC",
        "1",
    )
    monkeypatch.setenv(
        "VOICEPROBE_V32_OLLAMA_ENDPOINT",
        "http://gpu-node:11434/api/chat",
    )
    monkeypatch.setenv(
        "VOICEPROBE_V32_MODEL",
        "qwen3.5:4b",
    )

    backend = FakeBackend([])
    resolver = V32SemanticFallbackResolver(
        parser=SemanticParser(
            backend=backend,
        )
    )

    monkeypatch.setattr(
        production,
        "_build_v32_semantic_fallback",
        lambda: resolver,
    )

    bridge = production.PipecatRuntimeBridge(
        tts_frame_factory=FakeSpeechFrame,
    )

    assert bridge.semantic_mode == "v32"
    assert bridge.v32_semantic_resolver is resolver


def test_v32_production_bridge_speaks_grounded_fallback(
    monkeypatch,
):
    async def scenario():
        clear_v32_env(monkeypatch)

        monkeypatch.setenv(
            "VOICEPROBE_V32_SEMANTIC",
            "1",
        )
        monkeypatch.setenv(
            "VOICEPROBE_V32_OLLAMA_ENDPOINT",
            "http://gpu-node:11434/api/chat",
        )

        backend = FakeBackend([
            {
                "speech_act": "ask",
                "operation": "reschedule",
                "focus": "reschedule_reason",
                "commitment": "informational",
                "certainty": "high",
            }
        ])

        resolver = V32SemanticFallbackResolver(
            parser=SemanticParser(
                backend=backend,
            )
        )

        monkeypatch.setattr(
            production,
            "_build_v32_semantic_fallback",
            lambda: resolver,
        )

        bridge = production.PipecatRuntimeBridge(
            tts_frame_factory=FakeSpeechFrame,
        )

        worker = FakeWorker()
        bridge.bind_worker(worker)

        # Deterministic WAIT: no LLM call and no queued speech.
        wait = await bridge.runtime.process_turns([
            "No problem."
        ])

        assert wait.decision.kind is DecisionKind.WAIT
        assert backend.calls == 0
        assert worker.frames == []

        # Novel ordinary language falls through to v3.2.
        result = await bridge.runtime.process_turns([
            "What is the reason for changing your appointment?"
        ])

        assert (
            result.decision.kind
            is DecisionKind.CONTEXTUAL_ANSWER
        )

        assert backend.calls == 1

        assert len(worker.frames) == 1
        assert (
            "Friday afternoon"
            in worker.frames[0].text
        )

        assert (
            result.after.accepted_slot_text
            is None
        )
        assert (
            result.after.booking_confirmation_text
            is None
        )
        assert not result.after.complete

    asyncio.run(scenario())


def test_explicit_custom_fallback_still_wins(
    monkeypatch,
):
    clear_v32_env(monkeypatch)

    monkeypatch.setenv(
        "VOICEPROBE_V32_SEMANTIC",
        "1",
    )

    def custom(agent_turn, snapshot):
        del agent_turn, snapshot

        return production.PolicyDecision(
            DecisionKind.CLARIFY,
            text="custom",
            reason="custom",
        )

    bridge = production.PipecatRuntimeBridge(
        fallback_resolver=custom,
        tts_frame_factory=FakeSpeechFrame,
    )

    assert bridge.semantic_mode == "custom"
    assert bridge.v32_semantic_resolver is None
