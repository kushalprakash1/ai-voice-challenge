from __future__ import annotations

import pytest

from voiceprobe.v3.flow_controller import SchedulingFlowController
from voiceprobe.v3.flow_state import SchedulingFlowTracker
from voiceprobe.v3.medication_refill import (
    CORRECTION_DOSE,
    MedicationRefillCorrectionScenario,
)
from voiceprobe.v3.qwen_v3_fallback import QwenV3FallbackRouter


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    async def generate_json(self, **kwargs):
        del kwargs
        return self.outputs.pop(0)


def observation(action, *, field="none", value="", ack=False, response=True):
    return {
        "kind": "other",
        "requested_fact": "none",
        "choice_options": [],
        "remote_dob_claim": False,
        "remote_existing_appointment_claim": False,
        "requested_day_change": False,
        "requested_time_change": False,
        "requested_provider_change": False,
        "afternoon_constraint_retained": False,
        "requires_response": response,
        "medication_action": action,
        "extracted_target_field": field,
        "extracted_target_value": value,
        "target_acknowledges_correction": ack,
    }


def scenario(*outputs):
    tracker = SchedulingFlowTracker()
    qwen = QwenV3FallbackRouter(backend=FakeBackend(outputs))
    return tracker, MedicationRefillCorrectionScenario(tracker=tracker, qwen=qwen)


@pytest.mark.asyncio
async def test_open_intent_uses_scenario_owned_dose():
    tracker, refill = scenario(observation("request_refill"))
    result = await refill.resolve("How can I help you today?", tracker.snapshot())
    assert result.reason == "medication_refill:request_refill"
    assert "refill" in result.text
    assert tracker.snapshot().caller_truth["dose"].value == "10 mg"


@pytest.mark.asyncio
async def test_target_dose_then_explicit_correction_keeps_authorities_separate():
    tracker, refill = scenario(
        observation("confirm_or_correct_target_claim", field="dose", value="20 mg"),
        observation("confirm_or_correct_target_claim", field="dose", value="20 mg"),
    )
    first = await refill.resolve("I have twenty milligrams here.", tracker.snapshot())
    assert tracker.snapshot().target_observations["dose"].value == "20 mg"
    assert tracker.snapshot().caller_truth["dose"].value == CORRECTION_DOSE
    assert first.reason == "medication_refill:correct_dose"

    await refill.resolve("I still have twenty milligrams.", tracker.snapshot())
    assert tracker.snapshot().caller_truth["dose"].value == CORRECTION_DOSE
    assert not refill.oracle.correction_retention_failure


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "turn",
    ("What strength?", "What dosage?", "How many milligrams?"),
)
async def test_reworded_dose_questions_ask_target_for_dose_on_file(turn):
    tracker, refill = scenario(observation("ask_dose_on_file"))
    result = await refill.resolve(turn, tracker.snapshot())
    assert result.reason == "medication_refill:ask_dose_on_file"
    assert "10 mg" in result.text
    assert tracker.snapshot().caller_truth["dose"].value == "10 mg"


@pytest.mark.asyncio
async def test_pharmacy_claim_remains_target_observed_only():
    tracker, refill = scenario(
        observation(
            "confirm_or_correct_target_claim",
            field="pharmacy",
            value="Main Street Pharmacy",
        )
    )
    await refill.resolve("I have Main Street Pharmacy here.", tracker.snapshot())
    snapshot = tracker.snapshot()
    assert snapshot.target_observations["pharmacy"].value == "Main Street Pharmacy"
    assert "pharmacy" not in snapshot.caller_truth


@pytest.mark.asyncio
async def test_ungrounded_model_extraction_cannot_become_any_state():
    tracker, refill = scenario(
        observation("confirm_or_correct_target_claim", field="prescriber", value="Dr. Invented")
    )
    await refill.resolve("Who is the prescriber?", tracker.snapshot())
    snapshot = tracker.snapshot()
    assert "prescriber" not in snapshot.target_observations
    assert "prescriber" not in snapshot.caller_truth


@pytest.mark.asyncio
async def test_oracle_requires_acknowledgement_before_old_dose_returns():
    tracker, refill = scenario(
        observation("confirm_or_correct_target_claim", field="dose", value="20 mg"),
        observation(
            "confirm_or_correct_target_claim", field="dose", value="10 mg", ack=True
        ),
        observation("confirm_or_correct_target_claim", field="dose", value="20 mg"),
    )
    await refill.resolve("I have twenty milligrams here.", tracker.snapshot())
    await refill.resolve("So you're correcting that to ten?", tracker.snapshot())
    assert tracker.snapshot().committed_dialogue["dose"].value == "10 mg"
    await refill.resolve("I still have twenty milligrams.", tracker.snapshot())
    assert refill.oracle.correction_retention_failure
    assert tracker.snapshot().caller_truth["dose"].value == "10 mg"


def test_existing_selected_slot_grounding_still_matches_spoken_slot():
    result = SchedulingFlowController().decide_burst(
        ["I have Friday at 2:30 PM. Would that work for you?"]
    )
    assert result.after.committed_dialogue["selected_slot"].value == "2:30 PM"
    assert result.after.accepted_slot_text in result.decision.text
