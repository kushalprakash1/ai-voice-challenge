import asyncio
import json
from pathlib import Path

from voiceprobe.v3.flow_state import FlowStage, SchedulingFlowTracker
from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.runtime import VoiceProbeV3Runtime

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "v3_calls"
    / "flux_latest_eot_085.jsonl"
)


def load_fixture():
    return [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_composite_disclaimer_does_not_swallow_profile_request() -> None:
    row = load_fixture()[0]
    runtime = VoiceProbeV3Runtime()
    result = asyncio.run(runtime.process_turns([row["text"]]))
    assert result.decision.kind == DecisionKind.CREATE_PROFILE
    assert result.decision.text == (
        "Yes, please. My name is Chitragupta Subramnian Singh."
    )


def test_real_audio_short_ack_is_not_fallback() -> None:
    runtime = VoiceProbeV3Runtime()
    result = asyncio.run(runtime.process_turns(["Great, Alex."]))
    assert result.decision.kind == DecisionKind.WAIT
    assert not result.response_ready


def test_reason_plus_routine_checkup_example_answers_both_fields() -> None:
    row = load_fixture()[10]
    runtime = VoiceProbeV3Runtime()
    result = asyncio.run(runtime.process_turns([row["text"]]))
    assert result.decision.kind == DecisionKind.ANSWER_VISIT_DETAILS
    assert "right shoulder pain" in result.decision.text
    assert "new patient consultation" in result.decision.text


def test_real_audio_trailing_function_word_is_hold_not_fallback() -> None:
    row = load_fixture()[11]
    runtime = VoiceProbeV3Runtime()
    result = asyncio.run(runtime.process_turns([row["text"]]))
    assert result.decision.kind == DecisionKind.HOLD
    assert not result.response_ready


def test_flux_spoken_number_dob_confirms_flow_state() -> None:
    tracker = SchedulingFlowTracker()
    snapshot = tracker.observe_remote_turn(
        "Thanks for confirming your date of birth as April twelfth nineteen ninety eight."
    )
    assert FlowStage.DOB in snapshot.confirmed


def test_complete_latest_flux_085_fixture_has_zero_fallbacks() -> None:
    async def scenario():
        runtime = VoiceProbeV3Runtime()
        for row in load_fixture():
            result = await runtime.process_turns(
                [row["text"]],
                ingress_reason="frozen_flux_085_regression",
            )
            assert result.decision.kind.value == row["expected_kind"]

        assert runtime.metrics.fallback_decisions == 0
        assert runtime.metrics.holds == 2
        assert FlowStage.DOB in runtime.flow_controller.tracker.snapshot().confirmed

    asyncio.run(scenario())
