"""Telephony audio conversion and AudioSocket packet framing."""

from __future__ import annotations

import re
from collections.abc import Iterator

import numpy as np
import soxr

KOKORO_SAMPLE_RATE = 24_000
TELEPHONY_SAMPLE_RATE = 8_000

AUDIOSOCKET_PCM8K_TYPE = 0x10


_TIME_WITH_DOT_RE = re.compile(
    r"\b(\d{1,2})\.(\d{2})\s*([ap])\.?m\.?(?![A-Za-z])",
    re.IGNORECASE,
)

_COMPACT_TIME_RE = re.compile(
    r"\b(\d{1,2})(\d{2})\s*([ap])\.?m\.?(?![A-Za-z])",
    re.IGNORECASE,
)

_STANDARD_TIME_RE = re.compile(
    r"\b(\d{1,2}):(\d{2})\s*([ap])\.?m\.?(?![A-Za-z])",
    re.IGNORECASE,
)


def normalize_text_for_tts(text: str) -> str:
    """Convert written/ASR clock forms into speech-friendly text.

    This changes only the string sent to TTS. Conversation meaning,
    appointment state, transcripts, and evidence remain untouched.
    """

    def format_time(match: re.Match[str]) -> str:
        hour, minute, meridiem = match.groups()

        formatted = f"{hour}:{minute} {meridiem.upper()}M"

        # In forms such as "11:30 a.m.", the final period may also be
        # acting as sentence punctuation. Preserve it only when the
        # matched time reaches the end of the complete utterance.
        if match.group(0).endswith(".") and match.end() == len(match.string):
            formatted += "."

        return formatted

    normalized = _TIME_WITH_DOT_RE.sub(
        format_time,
        text,
    )

    normalized = _COMPACT_TIME_RE.sub(
        format_time,
        normalized,
    )

    normalized = _STANDARD_TIME_RE.sub(
        format_time,
        normalized,
    )

    return normalized


FRAME_DURATION_SECONDS = 0.020
FRAME_SAMPLES = int(TELEPHONY_SAMPLE_RATE * FRAME_DURATION_SECONDS)
FRAME_BYTES = FRAME_SAMPLES * 2


def resample_to_telephony(
    audio: np.ndarray,
    *,
    source_rate: int = KOKORO_SAMPLE_RATE,
) -> np.ndarray:
    """Convert mono floating-point speech to 8 kHz float32 audio."""
    samples = np.asarray(
        audio,
        dtype=np.float32,
    ).reshape(-1)

    if samples.size == 0:
        raise ValueError("Audio cannot be empty.")

    if source_rate <= 0:
        raise ValueError("source_rate must be greater than zero.")

    if source_rate == TELEPHONY_SAMPLE_RATE:
        return samples.copy()

    resampled = soxr.resample(
        samples,
        source_rate,
        TELEPHONY_SAMPLE_RATE,
        quality="HQ",
    )

    return np.asarray(
        resampled,
        dtype=np.float32,
    )


def float_audio_to_pcm16(
    audio: np.ndarray,
) -> bytes:
    """Encode normalized floating-point samples as little-endian PCM16."""
    samples = np.asarray(
        audio,
        dtype=np.float32,
    ).reshape(-1)

    if samples.size == 0:
        raise ValueError("Audio cannot be empty.")

    clipped = np.clip(
        samples,
        -1.0,
        1.0,
    )

    pcm = np.rint(clipped * 32767.0).astype("<i2")

    return pcm.tobytes()


def iter_pcm_frames(
    pcm16: bytes,
    *,
    pad_final: bool = True,
) -> Iterator[bytes]:
    """Split PCM16 into 20 ms telephony frames."""
    if not pcm16:
        raise ValueError("PCM payload cannot be empty.")

    for offset in range(
        0,
        len(pcm16),
        FRAME_BYTES,
    ):
        frame = pcm16[offset : offset + FRAME_BYTES]

        if len(frame) < FRAME_BYTES and pad_final:
            frame += bytes(FRAME_BYTES - len(frame))

        yield frame


def build_audiosocket_terminate_packet() -> bytes:
    """Build the AudioSocket zero-length termination frame."""
    return bytes((0x00, 0x00, 0x00))


def build_audiosocket_packet(
    payload: bytes,
    *,
    message_type: int = AUDIOSOCKET_PCM8K_TYPE,
) -> bytes:
    """Wrap one payload in an Asterisk AudioSocket frame."""
    if not payload:
        raise ValueError("AudioSocket payload cannot be empty.")

    if not 0 <= message_type <= 255:
        raise ValueError("AudioSocket message type must fit one byte.")

    if len(payload) > 65_535:
        raise ValueError("AudioSocket payload exceeds 16-bit length field.")

    header = bytes((message_type,)) + len(payload).to_bytes(
        2,
        byteorder="big",
        signed=False,
    )

    return header + payload
