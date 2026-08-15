"""Deterministic scheduling-objective state for VoiceProbe.

Language models may interpret conversation meaning, but they do not
decide whether a scheduling objective has actually been completed.
Python owns that transition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class AppointmentProgress:
    """Verified progress toward completing an appointment."""

    preferred_day_shared: bool = False
    preferred_time_shared: bool = False

    offered_day: str | None = None
    offered_time: str | None = None

    offer_accepted: bool = False
    booking_confirmed: bool = False

    @property
    def has_offer(self) -> bool:
        """Return whether the tested agent has offered a concrete slot."""
        return self.offered_day is not None or self.offered_time is not None

    @property
    def objective_complete(self) -> bool:
        """Only a confirmed accepted booking completes the objective."""
        return self.has_offer and self.offer_accepted and self.booking_confirmed


def record_preferences_shared(
    progress: AppointmentProgress,
    *,
    day_shared: bool = False,
    time_shared: bool = False,
) -> AppointmentProgress:
    """Record patient scheduling preferences already communicated."""
    return replace(
        progress,
        preferred_day_shared=(progress.preferred_day_shared or day_shared),
        preferred_time_shared=(progress.preferred_time_shared or time_shared),
    )


def record_slot_offer(
    progress: AppointmentProgress,
    *,
    day: str | None,
    time: str | None,
) -> AppointmentProgress:
    """Record a concrete appointment slot offered by the tested agent."""
    if day is None and time is None:
        raise ValueError("An appointment offer requires a day or time.")

    return replace(
        progress,
        offered_day=day,
        offered_time=time,
        offer_accepted=False,
        booking_confirmed=False,
    )


def record_offer_accepted(
    progress: AppointmentProgress,
) -> AppointmentProgress:
    """Record the patient's acceptance of the current offered slot."""
    if not progress.has_offer:
        raise ValueError("Cannot accept an appointment before a slot is offered.")

    return replace(
        progress,
        offer_accepted=True,
    )


def record_booking_confirmed(
    progress: AppointmentProgress,
) -> AppointmentProgress:
    """Record explicit confirmation that the booking was completed."""
    if not progress.has_offer:
        raise ValueError("Cannot confirm a booking before a slot is offered.")

    if not progress.offer_accepted:
        raise ValueError("Cannot confirm a booking before the offer is accepted.")

    return replace(
        progress,
        booking_confirmed=True,
    )
