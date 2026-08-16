from __future__ import annotations

from dataclasses import dataclass

import pytest

from voiceprobe.v3.asterisk_live import project_v3_flow_snapshot
from voiceprobe.v3.flow_state import FlowSnapshot, FlowStage


@dataclass(frozen=True)
class _ProjectionCase:
    complete: bool
    slot: str | None
    confirmation: str | None


def _snapshot(case: _ProjectionCase) -> FlowSnapshot:
    confirmed = (
        frozenset({FlowStage.CONFIRMATION})
        if case.complete
        else frozenset()
    )
    return FlowSnapshot(
        communicated=frozenset(),
        confirmed=confirmed,
        current_stage=(
            FlowStage.COMPLETE if case.complete else FlowStage.CONFIRMATION
        ),
        complete=case.complete,
        accepted_slot_text=case.slot,
        booking_confirmation_text=case.confirmation,
    )


def test_projection_does_not_invent_legacy_day_or_time() -> None:
    result = project_v3_flow_snapshot(
        _snapshot(
            _ProjectionCase(
                complete=False,
                slot="2:30 PM",
                confirmation=None,
            )
        )
    )

    assert result.objective_complete is False
    assert result.booking_confirmed is False
    assert result.offer_accepted is True
    assert result.offered_day is None
    assert result.offered_time is None
    assert result.accepted_slot_text == "2:30 PM"


def test_projection_treats_v3_complete_as_authoritative_booking_confirmation() -> None:
    result = project_v3_flow_snapshot(
        _snapshot(
            _ProjectionCase(
                complete=True,
                slot="2:30 PM",
                confirmation="You're booked Friday at 2:30 PM.",
            )
        )
    )

    assert result.objective_complete is True
    assert result.booking_confirmed is True
    assert result.offer_accepted is True
    assert result.booking_confirmation_text == "You're booked Friday at 2:30 PM."


def test_adapter_v3_environment_switch_is_explicit(monkeypatch) -> None:
    from voiceprobe.telephony.asterisk_adapter import (
        v3_live_enabled_from_environment,
    )

    monkeypatch.delenv("VOICEPROBE_V3_LIVE", raising=False)
    assert v3_live_enabled_from_environment() is False

    monkeypatch.setenv("VOICEPROBE_V3_LIVE", "1")
    assert v3_live_enabled_from_environment() is True

    monkeypatch.setenv("VOICEPROBE_V3_LIVE", "off")
    assert v3_live_enabled_from_environment() is False

    monkeypatch.setenv("VOICEPROBE_V3_LIVE", "definitely")
    with pytest.raises(ValueError):
        v3_live_enabled_from_environment()
