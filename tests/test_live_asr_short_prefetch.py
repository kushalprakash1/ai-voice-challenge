"""Delayed semantic-prefetch behavior for short live-ASR turns."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from voiceprobe.media.live_asr import (
    SHORT_TURN_PREFETCH_DELAY_SECONDS,
    ConsoleTranscriptListener,
)


def line_event(text: str) -> SimpleNamespace:
    return SimpleNamespace(line=SimpleNamespace(text=text))


def test_long_complete_turn_still_prefetches_immediately() -> None:
    candidates: list[str] = []
    listener = ConsoleTranscriptListener(on_candidate=candidates.append)
    try:
        listener.on_line_completed(line_event("What insurance do you have?"))
        assert candidates == ["What insurance do you have?"]
    finally:
        listener.close()


def test_short_complete_turn_prefetches_before_endpoint() -> None:
    candidate_seen = threading.Event()
    candidates: list[str] = []

    def on_candidate(text: str) -> None:
        candidates.append(text)
        candidate_seen.set()

    listener = ConsoleTranscriptListener(on_candidate=on_candidate)
    try:
        listener.on_line_completed(line_event("Friday."))
        assert not candidate_seen.wait(0.20)
        assert candidate_seen.wait(SHORT_TURN_PREFETCH_DELAY_SECONDS + 0.5)
        assert candidates == ["Friday."]
    finally:
        listener.close()


def test_short_prefetch_trigger_is_cancelled_when_speech_resumes() -> None:
    candidate_seen = threading.Event()
    listener = ConsoleTranscriptListener(
        on_candidate=lambda _text: candidate_seen.set(),
    )
    try:
        listener.on_line_completed(line_event("Friday."))
        time.sleep(0.15)
        listener.on_line_started(line_event("Friday afternoon"))
        assert not candidate_seen.wait(SHORT_TURN_PREFETCH_DELAY_SECONDS + 0.25)
    finally:
        listener.close()


def test_fragment_does_not_start_prefetch() -> None:
    candidate_seen = threading.Event()
    listener = ConsoleTranscriptListener(
        on_candidate=lambda _text: candidate_seen.set(),
    )
    try:
        listener.on_line_completed(line_event("They and"))
        assert not candidate_seen.wait(SHORT_TURN_PREFETCH_DELAY_SECONDS + 0.25)
    finally:
        listener.close()
