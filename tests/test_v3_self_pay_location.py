from __future__ import annotations

import pytest

from voiceprobe.v3.flow_state import SchedulingFlowTracker
from voiceprobe.v3.self_pay_location import SelfPayLocationSwitchScenario
from voiceprobe.v3.prerequisite import PrerequisiteOverlay


def observation(action, **updates):
    base = {
        "self_pay_location_action": action,
        "extracted_locations": (),
        "extracted_location": "",
        "target_acknowledged_location": "",
        "target_asserts_active_location": False,
        "extracted_insurance_status": "none",
        "extracted_insurer": "",
        "target_acknowledges_self_pay": False,
        "office_hours": "",
        "hours_location": "",
        "hours_context_changed": False,
        "requires_response": True,
    }
    base.update(updates)
    return base


class ScriptedQwen:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.last_observation = {}

    async def resolve(self, turn, snapshot):
        del turn, snapshot
        self.last_observation = self.outputs.pop(0)


def build(*outputs):
    tracker = SchedulingFlowTracker()
    scenario = SelfPayLocationSwitchScenario(
        tracker=tracker, qwen=ScriptedQwen(outputs)
    )
    return tracker, scenario


@pytest.mark.asyncio
async def test_profile_setup_acknowledgment_yields_then_insurance_routes_self_pay():
    tracker, scenario = build(
        observation("establish_self_pay"),
        observation("provide_self_pay"),
    )
    overlay = PrerequisiteOverlay(
        scenario_id=scenario.scenario_id,
        tracker=tracker,
        domain_resolver=scenario.resolve,
    )
    first = await overlay.resolve(
        "Thanks, Alex. Your demo profile is set up. How can I help you?",
        tracker.snapshot(),
    )
    second = await overlay.resolve("What insurance do you have?", tracker.snapshot())
    assert not first.reason.startswith("prerequisite:")
    assert second.reason == "self_pay_location:establish_self_pay"
    assert "self-pay" in second.text


@pytest.mark.asyncio
async def test_self_pay_truth_exists_only_after_caller_states_it():
    tracker, scenario = build(observation("provide_self_pay"))
    assert "insurance_status" not in tracker.snapshot().caller_truth
    result = await scenario.resolve("What insurance do you carry?", tracker.snapshot())
    assert "self-pay" in result.text
    assert "insurance_status" not in tracker.snapshot().caller_truth
    scenario.mark_decision_spoken(result)
    assert tracker.snapshot().caller_truth["insurance_status"].value == "self_pay"


@pytest.mark.asyncio
async def test_dynamic_locations_select_then_switch_only_to_grounded_values():
    tracker, scenario = build(
        observation(
            "locations_offered",
            extracted_locations=("North Clinic", "South Clinic"),
        ),
        observation(
            "location_acknowledgement",
            extracted_location="North Clinic",
            target_acknowledged_location="North Clinic",
        ),
        observation(
            "location_acknowledgement",
            extracted_location="South Clinic",
            target_acknowledged_location="South Clinic",
        ),
    )
    # This legacy unit isolates location mechanics; establish the prior
    # insurance milestones as already delivered.
    scenario._insurance_exercised = True
    scenario._self_pay_stated = True
    first = await scenario.resolve(
        "We have North Clinic and South Clinic.", tracker.snapshot()
    )
    assert "first location" in first.text
    scenario.mark_decision_spoken(first)
    assert tracker.snapshot().target_observations["offered_locations"].value == (
        "North Clinic", "South Clinic"
    )
    second = await scenario.resolve("Okay, North Clinic.", tracker.snapshot())
    assert second.reason == "self_pay_location:ask_location_hours"
    scenario.mark_decision_spoken(second)
    # Complete the contextual self-pay probe before the milestone-driven switch.
    scenario.qwen.outputs.insert(0, observation("confirm_location", extracted_location="North Clinic", office_hours="closes at five", hours_location="North Clinic"))
    hours = await scenario.resolve("North Clinic closes at five.", tracker.snapshot())
    scenario.mark_decision_spoken(hours)
    scenario.qwen.outputs.insert(0, observation("clarify"))
    contextual = await scenario.resolve("Yes, self-pay patients can be seen there.", tracker.snapshot())
    assert contextual.reason == "self_pay_location:switch_location"
    scenario.mark_decision_spoken(contextual)
    third = await scenario.resolve("Okay, we'll use South Clinic.", tracker.snapshot())
    assert "that office close" in third.text
    scenario.mark_decision_spoken(third)
    snapshot = tracker.snapshot()
    assert snapshot.caller_truth["selected_location"].value == "South Clinic"
    assert snapshot.committed_dialogue["selected_location"].value == "South Clinic"


@pytest.mark.asyncio
async def test_choice_question_uses_previously_grounded_location_offer():
    tracker, scenario = build(
        observation(
            "locations_offered",
            extracted_locations=("North Clinic", "South Clinic"),
        ),
        observation("ask_locations"),
    )
    scenario._insurance_exercised = True
    scenario._self_pay_stated = True
    offered = await scenario.resolve(
        "We have North Clinic and South Clinic.", tracker.snapshot()
    )
    scenario.mark_decision_spoken(offered)
    result = await scenario.resolve("Which one do you prefer?", tracker.snapshot())
    assert result.reason == "self_pay_location:ask_location_hours"
    assert tracker.snapshot().caller_truth["selected_location"].value == "North Clinic"


@pytest.mark.asyncio
async def test_location_reversion_and_that_office_mismatch_require_acknowledged_switch():
    tracker, scenario = build(
        observation("locations_offered", extracted_locations=("North Clinic", "South Clinic")),
        observation("location_acknowledgement", extracted_location="North Clinic", target_acknowledged_location="North Clinic"),
        observation("location_acknowledgement", extracted_location="South Clinic", target_acknowledged_location="South Clinic"),
        observation("confirm_location", extracted_location="North Clinic", target_asserts_active_location=True, office_hours="closes at six", hours_location="North Clinic"),
    )
    scenario._insurance_exercised = True
    scenario._self_pay_stated = True
    first = await scenario.resolve("We have North Clinic and South Clinic.", tracker.snapshot()); scenario.mark_decision_spoken(first)
    second = await scenario.resolve("Okay, North Clinic.", tracker.snapshot()); scenario.mark_decision_spoken(second)
    scenario._self_pay_contextual_exercised = True
    switch = await scenario.resolve("Okay, we'll use South Clinic.", tracker.snapshot()); scenario.mark_decision_spoken(switch)
    scenario._switch_acknowledged = True
    scenario._hours_followup_pending = True
    await scenario.resolve("The North Clinic closes at six.", tracker.snapshot())
    names = {item.oracle_name for item in scenario.oracle_evidence}
    assert "location_switch_retention_failure" in names
    assert "active_location_context_mismatch" in names
    assert tracker.snapshot().caller_truth["selected_location"].value == "South Clinic"


@pytest.mark.asyncio
async def test_benign_historical_location_mention_does_not_trigger_oracle():
    tracker, scenario = build(
        observation("locations_offered", extracted_locations=("North Clinic", "South Clinic")),
        observation("location_acknowledgement", extracted_location="North Clinic"),
    )
    await scenario.resolve("North Clinic and South Clinic are available.", tracker.snapshot())
    await scenario.resolve("North Clinic is one of those offices.", tracker.snapshot())
    assert scenario.oracle_evidence == ()


@pytest.mark.asyncio
async def test_self_pay_regression_is_conservative_and_requires_acknowledgment():
    tracker, scenario = build(
        observation("provide_self_pay"),
        observation("ask_locations", extracted_insurance_status="self_pay", target_acknowledges_self_pay=True),
        observation("clarify", extracted_insurance_status="specific_insurer", extracted_insurer="Acme Plan"),
        observation("clarify", extracted_insurance_status="specific_insurer", extracted_insurer="Acme Plan"),
    )
    for turn in (
        "Who is your insurance provider?",
        "Okay, we can treat you as self-pay. Which office?",
        "I still see Acme Plan here.",
        "Your insurer is Acme Plan.",
    ):
        decision = await scenario.resolve(turn, tracker.snapshot())
        if decision.requires_response:
            scenario.mark_decision_spoken(decision)
    assert any(
        item.oracle_name == "self_pay_state_regression"
        for item in scenario.oracle_evidence
    )
    assert tracker.snapshot().caller_truth["insurance_status"].value == "self_pay"


@pytest.mark.asyncio
async def test_same_location_hours_contradiction_requires_stable_context():
    tracker, scenario = build(
        observation("locations_offered", extracted_locations=("South Clinic",)),
        observation("location_acknowledgement", extracted_location="South Clinic", target_acknowledged_location="South Clinic"),
        observation("confirm_location", extracted_location="South Clinic", office_hours="closes at five", hours_location="South Clinic"),
        observation("confirm_location", extracted_location="South Clinic", office_hours="closes at seven", hours_location="South Clinic"),
    )
    for turn in (
        "South Clinic is available.",
        "Okay, we'll use South Clinic.",
        "South Clinic closes at five.",
        "South Clinic closes at seven.",
    ):
        await scenario.resolve(turn, tracker.snapshot())
    assert any(
        item.oracle_name == "office_hours_internal_contradiction"
        for item in scenario.oracle_evidence
    )


@pytest.mark.asyncio
async def test_different_location_hours_are_a_benign_control():
    tracker, scenario = build(
        observation("confirm_location", extracted_location="North Clinic", office_hours="closes at five", hours_location="North Clinic"),
        observation("confirm_location", extracted_location="South Clinic", office_hours="closes at two", hours_location="South Clinic"),
    )
    await scenario.resolve("North Clinic closes at five.", tracker.snapshot())
    await scenario.resolve("South Clinic closes at two.", tracker.snapshot())
    assert not any(
        item.oracle_name == "office_hours_internal_contradiction"
        for item in scenario.oracle_evidence
    )


@pytest.mark.asyncio
async def test_ungrounded_location_extraction_is_rejected_without_state_mutation():
    tracker, scenario = build(
        observation("locations_offered", extracted_locations=("Invented Clinic",))
    )
    await scenario.resolve("Which office would you like?", tracker.snapshot())
    assert "offered_locations" not in tracker.snapshot().target_observations
    assert "selected_location" not in tracker.snapshot().caller_truth
    assert scenario.rejected_extractions
