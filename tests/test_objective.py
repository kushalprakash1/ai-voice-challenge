import pytest

from voiceprobe.conversation.objective import (
    AppointmentProgress,
    record_booking_confirmed,
    record_offer_accepted,
    record_preferences_shared,
    record_slot_offer,
)


def test_initial_progress_is_incomplete() -> None:
    progress = AppointmentProgress()

    assert not progress.has_offer
    assert not progress.objective_complete


def test_records_preferences_without_completing_objective() -> None:
    progress = record_preferences_shared(
        AppointmentProgress(),
        day_shared=True,
        time_shared=True,
    )

    assert progress.preferred_day_shared
    assert progress.preferred_time_shared
    assert not progress.objective_complete


def test_slot_offer_does_not_complete_objective() -> None:
    progress = record_slot_offer(
        AppointmentProgress(),
        day="Friday",
        time="2:30 PM",
    )

    assert progress.has_offer
    assert progress.offered_day == "Friday"
    assert progress.offered_time == "2:30 PM"
    assert not progress.objective_complete


def test_acceptance_alone_does_not_complete_objective() -> None:
    progress = record_slot_offer(
        AppointmentProgress(),
        day="Friday",
        time="2:30 PM",
    )

    progress = record_offer_accepted(progress)

    assert progress.offer_accepted
    assert not progress.objective_complete


def test_confirmed_accepted_booking_completes_objective() -> None:
    progress = record_slot_offer(
        AppointmentProgress(),
        day="Friday",
        time="2:30 PM",
    )

    progress = record_offer_accepted(progress)
    progress = record_booking_confirmed(progress)

    assert progress.booking_confirmed
    assert progress.objective_complete


def test_cannot_accept_without_offer() -> None:
    with pytest.raises(
        ValueError,
        match="before a slot is offered",
    ):
        record_offer_accepted(AppointmentProgress())


def test_cannot_confirm_without_acceptance() -> None:
    progress = record_slot_offer(
        AppointmentProgress(),
        day="Friday",
        time="2:30 PM",
    )

    with pytest.raises(
        ValueError,
        match="before the offer is accepted",
    ):
        record_booking_confirmed(progress)


def test_new_offer_resets_previous_acceptance() -> None:
    progress = record_slot_offer(
        AppointmentProgress(),
        day="Friday",
        time="2:30 PM",
    )

    progress = record_offer_accepted(progress)

    progress = record_slot_offer(
        progress,
        day="Friday",
        time="4:00 PM",
    )

    assert progress.offered_time == "4:00 PM"
    assert not progress.offer_accepted
    assert not progress.booking_confirmed
