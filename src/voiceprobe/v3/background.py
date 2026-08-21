"""Deterministic, semantic-free outbound office ambience for assessment calls."""

from __future__ import annotations

import math
import os
import random
import struct
from dataclasses import dataclass


BACKGROUND_ENV = "VOICEPROBE_BACKGROUND_MODE"
BACKGROUND_SNR_ENV = "VOICEPROBE_BACKGROUND_SNR_DB"
BACKGROUND_NONE = "none"
BACKGROUND_OFFICE_CHATTER_LOUD = "office_chatter_loud"
BACKGROUND_SEED = 2026081904
SAMPLE_RATE = 8_000


def background_mode_from_environment() -> str:
    value = os.getenv(BACKGROUND_ENV, BACKGROUND_NONE).strip().casefold() or BACKGROUND_NONE
    if value not in {BACKGROUND_NONE, BACKGROUND_OFFICE_CHATTER_LOUD}:
        raise ValueError(f"{BACKGROUND_ENV} must be none or office_chatter_loud")
    return value


def background_snr_from_environment() -> float:
    value = float(os.getenv(BACKGROUND_SNR_ENV, "10"))
    if not -5.0 <= value <= 40.0:
        raise ValueError(f"{BACKGROUND_SNR_ENV} must be between -5 and 40")
    return value


@dataclass(frozen=True, slots=True)
class BackgroundMixResult:
    pcm: bytes
    metadata: dict[str, object]


def deterministic_office_chatter(*, seconds: int = 12, seed: int = BACKGROUND_SEED) -> bytes:
    """Create loopable non-linguistic babble/noise; it contains no recordings."""
    rng = random.Random(seed)
    count = SAMPLE_RATE * seconds
    phases = [rng.random() * math.tau for _ in range(7)]
    frequencies = [113.0, 147.0, 181.0, 223.0, 271.0, 337.0, 419.0]
    samples: list[int] = []
    filtered = 0.0
    for index in range(count):
        t = index / SAMPLE_RATE
        # Slowly varying, overlapping formant-like murmur without phonemes.
        envelope = 0.35 + 0.18 * math.sin(math.tau * 0.71 * t)
        murmur = sum(
            math.sin(math.tau * frequency * t + phases[offset])
            for offset, frequency in enumerate(frequencies)
        ) / len(frequencies)
        filtered = 0.82 * filtered + 0.18 * rng.uniform(-1.0, 1.0)
        desk = 0.0
        if index % 6137 in {0, 1, 2, 3}:
            desk = (1.0 - (index % 6137) / 4.0) * rng.choice((-0.45, 0.45))
        value = max(-0.95, min(0.95, envelope * murmur + 0.22 * filtered + desk))
        samples.append(round(value * 12_000))
    return struct.pack(f"<{count}h", *samples)


def _rms(samples: tuple[int, ...]) -> float:
    return math.sqrt(sum(value * value for value in samples) / max(1, len(samples)))


def mix_background(
    speech_pcm: bytes,
    *,
    mode: str,
    target_snr_db: float = 10.0,
    seed: int = BACKGROUND_SEED,
) -> BackgroundMixResult:
    if mode == BACKGROUND_NONE:
        return BackgroundMixResult(speech_pcm, {
            "background_mode": BACKGROUND_NONE,
            "background_applied": False,
            "background_snr_db": None,
            "background_seed": None,
        })
    if mode != BACKGROUND_OFFICE_CHATTER_LOUD:
        raise ValueError(f"unsupported background mode: {mode}")
    if not speech_pcm or len(speech_pcm) % 2:
        raise ValueError("speech must be non-empty PCM16")
    speech = struct.unpack(f"<{len(speech_pcm) // 2}h", speech_pcm)
    asset = deterministic_office_chatter(seed=seed)
    noise_all = struct.unpack(f"<{len(asset) // 2}h", asset)
    offset = seed % len(noise_all)
    noise = tuple(noise_all[(offset + index) % len(noise_all)] for index in range(len(speech)))
    speech_rms = _rms(speech)
    noise_rms = _rms(noise)
    if speech_rms == 0 or noise_rms == 0:
        raise ValueError("speech and background must contain signal")
    scale = speech_rms / (noise_rms * (10 ** (target_snr_db / 20.0)))
    raw = [speech[index] + noise[index] * scale for index in range(len(speech))]
    peak = max(abs(value) for value in raw)
    headroom_scale = min(1.0, 32_000.0 / peak) if peak else 1.0
    mixed = tuple(round(value * headroom_scale) for value in raw)
    pcm = struct.pack(f"<{len(mixed)}h", *mixed)
    effective_snr = 20.0 * math.log10(
        (speech_rms * headroom_scale) / (noise_rms * scale * headroom_scale)
    )
    return BackgroundMixResult(pcm, {
        "background_mode": mode,
        "background_applied": True,
        "background_snr_db": target_snr_db,
        "effective_snr_db": round(effective_snr, 3),
        "background_seed": seed,
        "background_asset": "deterministic_procedural_office_chatter_pcm16_8khz_v1",
        "background_peak_pcm16": max(abs(value) for value in mixed),
        "background_clipped_samples": sum(abs(value) >= 32767 for value in mixed),
    })


def validate_background_asset() -> dict[str, object]:
    pcm = deterministic_office_chatter()
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    return {
        "valid": bool(pcm) and len(pcm) % 2 == 0 and max(abs(v) for v in samples) < 32767,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "sample_width": 2,
        "seed": BACKGROUND_SEED,
        "music_present": False,
        "semantic_content_present": False,
    }
