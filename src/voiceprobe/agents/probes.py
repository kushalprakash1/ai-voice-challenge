"""Deterministic behavioral probes layered over ordinary PatientBrain decisions.

Most scenario targets are passive observations: the patient should correct a
wrong fact or reject an incompatible offer only if the tested agent actually
creates that condition. This module contains only the small set of realistic
patient-driven probes that VoiceProbe itself can safely enact.
"""

from __future__ import annotations

from dataclasses import dataclass

from voiceprobe.agents.brain import CommunicationDecision, CommunicationKind
from voiceprobe.conversation.objective import AppointmentProgress
from voiceprobe.scenarios.models import PatientScenario, ProbeKind


@dataclass(frozen=True, slots=True)
class ProbeProgress:
    """Per-call record of patient-driven probes already enacted."""

    fired: frozenset[ProbeKind] = frozenset()

    def has_fired(self, probe: ProbeKind) -> bool:
        """Return whether this probe has already changed patient behavior."""
        return probe in self.fired

    def mark_fired(self, probe: ProbeKind) -> ProbeProgress:
        """Return new immutable progress with one probe marked fired."""
        return ProbeProgress(fired=self.fired | {probe})


def apply_probe_policy(
    *,
    scenario: PatientScenario,
    appointment: AppointmentProgress,
    probe_progress: ProbeProgress,
    prior_agent_turn_count: int,
    base_decision: CommunicationDecision,
    booking_confirmed_this_turn: bool = False,
) -> tuple[CommunicationDecision, ProbeProgress]:
    """Overlay a narrowly scoped patient-driven experiment on a normal decision."""
    enabled = set(scenario.probes)

    repeat_probe = ProbeKind.REQUEST_AGENT_REPEAT_ONCE

    if (
        repeat_probe in enabled
        and not probe_progress.has_fired(repeat_probe)
        and prior_agent_turn_count >= 1
        and base_decision.kind is CommunicationKind.ANSWER
    ):
        return (
            CommunicationDecision(
                kind=CommunicationKind.ASK_AGENT_TO_REPEAT,
                probe=repeat_probe,
            ),
            probe_progress.mark_fired(repeat_probe),
        )

    booking_probe = ProbeKind.VERIFY_BOOKING_BEFORE_END

    if (
        booking_probe in enabled
        and not probe_progress.has_fired(booking_probe)
        and appointment.offer_accepted
        and not appointment.booking_confirmed
        and not booking_confirmed_this_turn
        and base_decision.kind
        in {
            CommunicationKind.END_CONVERSATION,
            CommunicationKind.CLARIFY,
        }
    ):
        return (
            CommunicationDecision(
                kind=CommunicationKind.VERIFY_BOOKING,
                offered_day=appointment.offered_day,
                offered_time=appointment.offered_time,
                probe=booking_probe,
            ),
            probe_progress.mark_fired(booking_probe),
        )

    return base_decision, probe_progress
