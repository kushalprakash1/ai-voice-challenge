from __future__ import annotations

import math
import struct

import pytest

from voiceprobe.v3.background import BACKGROUND_OFFICE_CHATTER_LOUD, mix_background, validate_background_asset
from voiceprobe.v3.flow_state import SchedulingFlowTracker
from voiceprobe.v3.self_pay_location import SelfPayLocationSwitchScenario


def obs(action: str, **updates):
    value = {
        "self_pay_location_action": action, "extracted_locations": (),
        "extracted_location": "", "target_acknowledged_location": "",
        "target_asserts_active_location": False,
        "extracted_insurance_status": "none", "extracted_insurer": "",
        "target_acknowledges_self_pay": False, "office_hours": "",
        "hours_location": "", "hours_context_changed": False,
        "requires_response": True,
    }
    value.update(updates)
    return value


class Qwen:
    def __init__(self, outputs): self.outputs, self.last_observation = list(outputs), {}
    async def resolve(self, *_): self.last_observation = self.outputs.pop(0)


@pytest.mark.asyncio
async def test_cross_domain_milestones_and_grounded_switch():
    outputs = [
        obs("provide_self_pay", extracted_insurance_status="insured", extracted_insurer="Plan A"),
        obs("ask_locations", target_acknowledges_self_pay=True, extracted_insurance_status="self_pay"),
        obs("locations_offered", extracted_locations=("Downtown", "North First")),
        obs("location_acknowledgement", extracted_location="Downtown", target_acknowledged_location="Downtown"),
        obs("confirm_location", office_hours="Monday through Friday until five"),
        obs("clarify"),
        obs("location_acknowledgement", extracted_location="North First", target_acknowledged_location="North First"),
        obs("confirm_location", office_hours="closes at four"),
        obs("confirm_location", office_hours="Saturday until two"),
        obs("provide_self_pay", target_acknowledges_self_pay=True, extracted_insurance_status="self_pay"),
    ]
    tracker = SchedulingFlowTracker()
    scenario = SelfPayLocationSwitchScenario(tracker=tracker, qwen=Qwen(outputs))
    turns = [
        "We accept Plan A.", "Okay, self-pay is fine.",
        "We have Downtown and North First.", "We will use Downtown.",
        "That office is open Monday through Friday until five.",
        "Yes, self-pay patients can be seen there.", "We will use North First.",
        "That office closes at four.", "It is open Saturday until two.",
        "Yes, you are still self-pay.",
    ]
    decisions = []
    for turn in turns:
        decision = await scenario.resolve(turn, tracker.snapshot())
        decisions.append(decision)
        scenario.mark_decision_spoken(decision)
    assert "first location" in decisions[2].text
    assert "second location" in decisions[5].text
    assert decisions[6].text == "What time does that office close?"
    metadata = scenario.metadata()
    assert metadata["location_a"] == "Downtown"
    assert metadata["location_b"] == "North First"
    assert metadata["objective_complete"] is True
    assert tracker.snapshot().caller_truth["insurance_status"].value == "self_pay"


def test_background_mixer_is_10db_nonclipping_and_audio_only():
    speech = tuple(round(8000 * math.sin(2 * math.pi * 220 * i / 8000)) for i in range(8000))
    pcm = struct.pack("<8000h", *speech)
    result = mix_background(pcm, mode=BACKGROUND_OFFICE_CHATTER_LOUD, target_snr_db=10)
    assert result.pcm != pcm and len(result.pcm) == len(pcm)
    assert result.metadata["effective_snr_db"] == pytest.approx(10.0, abs=0.01)
    assert result.metadata["background_clipped_samples"] == 0
    assert validate_background_asset()["valid"] is True
