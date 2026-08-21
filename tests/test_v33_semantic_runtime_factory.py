from __future__ import annotations

from voiceprobe.v33.reasoner import (
    OllamaV33Reasoner,
    build_v33_reasoner,
)


def test_reasoner_factory_defaults_to_existing_ollama(monkeypatch):
    monkeypatch.delenv("VOICEPROBE_V33_LEVEL2_RUNTIME_CANDIDATE", raising=False)
    reasoner = build_v33_reasoner(
        endpoint="http://127.0.0.1:11434/api/chat",
        model="qwen3.5:4b",
        timeout_seconds=2.5,
    )
    assert isinstance(reasoner, OllamaV33Reasoner)


def test_reasoner_factory_requires_exact_one(monkeypatch):
    monkeypatch.setenv("VOICEPROBE_V33_LEVEL2_RUNTIME_CANDIDATE", "true")
    reasoner = build_v33_reasoner(
        endpoint="http://127.0.0.1:11434/api/chat",
    )
    assert isinstance(reasoner, OllamaV33Reasoner)


def test_reasoner_factory_selects_local_candidate_without_initializing_models(
    monkeypatch,
):
    import voiceprobe.v33.semantic_runtime_v2 as local_runtime

    sentinel = object()
    monkeypatch.setattr(
        local_runtime.SemanticLabV2Reasoner,
        "shared",
        classmethod(lambda cls: sentinel),
    )
    monkeypatch.setenv("VOICEPROBE_V33_LEVEL2_RUNTIME_CANDIDATE", "1")

    reasoner = build_v33_reasoner(
        endpoint="http://127.0.0.1:11434/api/chat",
    )
    assert reasoner is sentinel
