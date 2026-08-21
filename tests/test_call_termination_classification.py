from voiceprobe.telephony.asterisk_adapter import (
    AsteriskTerminationStatus,
    _classify_termination,
    _termination_failure_reason,
)


def test_objective_completion_is_normal_termination() -> None:
    status = _classify_termination(
        objective_complete=True,
        max_duration_reached=False,
    )

    assert status is AsteriskTerminationStatus.NORMAL_COMPLETION

    assert (
        _termination_failure_reason(
            status=status,
            booking_confirmed=True,
            offer_accepted=True,
            offered_day="Friday",
            offered_time="2:30 PM",
        )
        is None
    )


def test_incomplete_unsolicited_disconnect_is_premature_remote() -> None:
    status = _classify_termination(
        objective_complete=False,
        max_duration_reached=False,
    )

    assert (
        status
        is AsteriskTerminationStatus.PREMATURE_REMOTE_TERMINATION
    )

    reason = _termination_failure_reason(
        status=status,
        booking_confirmed=False,
        offer_accepted=True,
        offered_day="Friday",
        offered_time="2:30 PM",
    )

    assert reason is not None
    assert "premature_remote_termination" in reason
    assert "booking_confirmed=False" in reason
    assert "offer_accepted=True" in reason
    assert "Friday" in reason
    assert "2:30 PM" in reason


def test_incomplete_deadline_disconnect_is_max_duration() -> None:
    status = _classify_termination(
        objective_complete=False,
        max_duration_reached=True,
    )

    assert (
        status
        is AsteriskTerminationStatus.MAX_DURATION_TERMINATION
    )

    reason = _termination_failure_reason(
        status=status,
        booking_confirmed=False,
        offer_accepted=False,
        offered_day=None,
        offered_time=None,
    )

    assert reason is not None
    assert "max_duration_termination" in reason


def test_completed_objective_wins_over_deadline_race() -> None:
    status = _classify_termination(
        objective_complete=True,
        max_duration_reached=True,
    )

    assert status is AsteriskTerminationStatus.NORMAL_COMPLETION
