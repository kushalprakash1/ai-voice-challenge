"""Deterministic real-time budget simulation for staged calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SimulatedClock:
    now: float = 0.0

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("Cannot move simulated time backwards")
        self.now += seconds
        return self.now


@dataclass(frozen=True, slots=True)
class RealtimeBudget:
    first_audio_deadline_seconds: float = 2.0
    remote_repeat_threshold_seconds: float = 3.5
    max_call_duration_seconds: float = 180.0

    def audio_start_ok(self, seconds: float) -> bool:
        return seconds <= self.first_audio_deadline_seconds
