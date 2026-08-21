"""Test-suite isolation from local VoiceProbe runtime configuration."""

import pytest

_RUNTIME_ENV_KEYS = (
    "VOICEPROBE_ACCENT_MODE",
    "VOICEPROBE_BACKGROUND_MODE",
    "VOICEPROBE_BACKGROUND_SNR_DB",
    "VOICEPROBE_LIVE_MONITOR",
    "VOICEPROBE_PERSONA",
    "VOICEPROBE_PERSONA_SEED",
    "VOICEPROBE_PERSONA_SEQUENCE",
    "VOICEPROBE_SCENARIO",
    "VOICEPROBE_TURN_MODE",
    "VOICEPROBE_V3_LIVE",
    "VOICEPROBE_V3_QWEN_FALLBACK",
    "VOICEPROBE_V32_MODEL",
    "VOICEPROBE_V32_OLLAMA_ENDPOINT",
    "VOICEPROBE_V32_SEMANTIC",
)


@pytest.fixture(autouse=True)
def isolate_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require tests to opt in to runtime modes explicitly."""
    for key in _RUNTIME_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
