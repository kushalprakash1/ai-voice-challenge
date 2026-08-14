"""Outbound call planning."""

from dataclasses import dataclass

from voiceprobe.policy import CallPolicy
from voiceprobe.safety import validate_destination


@dataclass(frozen=True, slots=True)
class OutboundCallPlan:
    """A validated description of an outbound assessment call."""

    originating_number: str
    destination: str
    max_duration_seconds: int
    dry_run: bool


def build_call_plan(policy: CallPolicy) -> OutboundCallPlan:
    """Create a call plan without contacting a telephony provider."""
    destination = validate_destination(policy.destination)

    return OutboundCallPlan(
        originating_number=policy.originating_number,
        destination=destination,
        max_duration_seconds=policy.max_call_duration_seconds,
        dry_run=policy.dry_run,
    )
