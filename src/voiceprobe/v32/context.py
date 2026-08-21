"""Compact projection of authoritative VoiceProbe state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from voiceprobe.v3.flow_state import FlowSnapshot
from voiceprobe.v3.models import PatientFacts


@dataclass(frozen=True, slots=True)
class ReasoningContext:
    facts: PatientFacts
    snapshot: FlowSnapshot
    recent_dialogue: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, object]:
        return {
            "authoritative_patient_facts": asdict(self.facts),
            "dialogue_state": {
                "current_stage": self.snapshot.current_stage.value,
                "complete": self.snapshot.complete,
                "communicated": sorted(
                    x.value
                    for x in self.snapshot.communicated
                ),
                "confirmed": sorted(
                    x.value
                    for x in self.snapshot.confirmed
                ),
                "accepted_slot_text": (
                    self.snapshot.accepted_slot_text
                ),
                "booking_confirmation_text": (
                    self.snapshot.booking_confirmation_text
                ),
                "allow_earlier_week_afternoons": (
                    self.snapshot.allow_earlier_week_afternoons
                ),
            },
            "recent_dialogue": list(
                self.recent_dialogue[-8:]
            ),
        }


def normalize_history(
    history: Iterable[str] | None,
) -> tuple[str, ...]:
    if history is None:
        return ()

    return tuple(
        " ".join(item.split())
        for item in history
        if item and item.strip()
    )
