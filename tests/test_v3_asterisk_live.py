from __future__ import annotations

from dataclasses import dataclass

import pytest

from voiceprobe.v3.asterisk_live import (
    project_v3_flow_snapshot,
    scenario_termination_failure_reason,
)
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


def test_medication_projection_and_failure_are_scenario_specific() -> None:
    snapshot = _snapshot(_ProjectionCase(False, None, None))
    projection = project_v3_flow_snapshot(
        snapshot,
        scenario_id="medication-refill-correction",
        scenario_metadata={
            "scenario_stage": "correct_dose",
            "dose_correction_spoken": True,
            "dose_acknowledged": False,
            "objective_complete": False,
        },
    )

    reason = scenario_termination_failure_reason(
        status="max_duration_termination", projection=projection
    )
    assert projection.objective_complete is False
    assert projection.booking_confirmed is False
    assert "medication-refill-correction objective" in reason
    assert "dose_correction_spoken=True" in reason
    assert "scheduling objective" not in reason
    assert "booking_confirmed" not in reason
    assert "offered_day" not in reason


def test_self_pay_projection_uses_its_own_objective_state() -> None:
    snapshot = _snapshot(_ProjectionCase(False, None, None))
    projection = project_v3_flow_snapshot(
        snapshot,
        scenario_id="self-pay-location-switch",
        scenario_metadata={"objective_complete": True},
    )
    assert projection.objective_complete is True
    assert projection.booking_confirmed is False


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


def test_recording_bridge_persists_remote_and_patient_turns() -> None:
    import asyncio

    from voiceprobe.v3.asterisk_live import _RecordingPipecatRuntimeBridge

    class Recorder:
        def __init__(self) -> None:
            self.transcript = []
            self.events = []
            self.metrics = []

        def record_transcript_turn(self, *, speaker, text, **metadata) -> None:
            self.transcript.append((speaker, text, metadata))

        def record_event(self, event, **details) -> None:
            self.events.append((event, details))

        def record_turn_metrics(self, metrics) -> None:
            self.metrics.append(dict(metrics))

    class Sink:
        def __init__(self) -> None:
            self.frames = []

        async def queue_frames(self, frames) -> None:
            self.frames.extend(frames)

    recorder = Recorder()
    sink = Sink()
    bridge = _RecordingPipecatRuntimeBridge(
        recorder=recorder,
        tts_frame_factory=lambda text: ("tts", text),
    )
    bridge.bind_frame_sink(sink)

    result = asyncio.run(
        bridge.runtime.process_turns(
            ["What insurance do you have?"],
            ingress_reason="test_flux_eot",
        )
    )

    assert result.response_ready
    assert recorder.transcript[0][0] == "agent"
    assert recorder.transcript[0][1] == "What insurance do you have?"
    assert recorder.transcript[0][2]["source"] == "deepgram_flux_eot"

    assert recorder.transcript[1][0] == "patient"
    assert "Blue Cross" in recorder.transcript[1][1]
    assert recorder.transcript[1][2]["delivery_status"] == "queued_for_tts"

    assert sink.frames == [("tts", result.decision.text)]
    assert recorder.events[-1][0] == "v3_runtime_decision"
    assert recorder.events[-1][1]["decision_kind"] == result.decision.kind.value
    assert recorder.metrics[-1]["decision_kind"] == result.decision.kind.value
    assert recorder.metrics[-1]["response_ready"] is True


def test_recording_bridge_persists_wait_turn_without_patient_speech() -> None:
    import asyncio

    from voiceprobe.v3.asterisk_live import _RecordingPipecatRuntimeBridge

    class Recorder:
        def __init__(self) -> None:
            self.transcript = []
            self.events = []
            self.metrics = []

        def record_transcript_turn(self, *, speaker, text, **metadata) -> None:
            self.transcript.append((speaker, text, metadata))

        def record_event(self, event, **details) -> None:
            self.events.append((event, details))

        def record_turn_metrics(self, metrics) -> None:
            self.metrics.append(dict(metrics))

    recorder = Recorder()
    bridge = _RecordingPipecatRuntimeBridge(
        recorder=recorder,
        tts_frame_factory=lambda text: ("tts", text),
    )

    result = asyncio.run(
        bridge.runtime.process_turns(
            ["Thanks."],
            ingress_reason="test_flux_eot",
        )
    )

    assert not result.response_ready
    assert [(speaker, text) for speaker, text, _ in recorder.transcript] == [
        ("agent", "Thanks.")
    ]
    assert recorder.events[-1][0] == "v3_runtime_decision"
    assert recorder.metrics[-1]["decision_kind"] == "wait"
    assert recorder.metrics[-1]["response_ready"] is False
