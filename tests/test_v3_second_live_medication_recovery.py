from __future__ import annotations

import pytest

from voiceprobe.telephony.asterisk_adapter import AsteriskTerminationStatus
from voiceprobe.v3 import production
from voiceprobe.v3.asterisk_live import (
    project_v3_flow_snapshot,
    scenario_termination_failure_reason,
)
from voiceprobe.v3.flow_state import SchedulingFlowTracker, StateProvenance
from voiceprobe.v3.medication_refill import MedicationRefillCorrectionScenario
from voiceprobe.v3.prerequisite import PrerequisiteOverlay
from voiceprobe.v3.qwen_v3_fallback import QwenV3FallbackRouter


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    async def generate_json(self, **kwargs):
        del kwargs
        return self.outputs.pop(0)


def semantic(
    action="none",
    *,
    outcome="none",
    escalation=False,
    setup=False,
    field="none",
    value="",
    ack=False,
    response=True,
):
    return {
        "requires_response": response,
        "medication_action": action,
        "medication_outcome": outcome,
        "offers_human_escalation": escalation,
        "offers_medication_list_setup": setup,
        "extracted_target_field": field,
        "extracted_target_value": value,
        "target_acknowledges_correction": ack,
    }


def build(*outputs):
    tracker = SchedulingFlowTracker()
    qwen = QwenV3FallbackRouter(
        backend=FakeBackend(outputs), semantic_domain="medication"
    )
    scenario = MedicationRefillCorrectionScenario(tracker=tracker, qwen=qwen)
    overlay = PrerequisiteOverlay(
        scenario_id=scenario.scenario_id,
        tracker=tracker,
        domain_resolver=scenario.resolve,
    )
    return tracker, scenario, overlay


@pytest.mark.asyncio
async def test_profile_acknowledgment_yields_to_objective_router_without_consent():
    tracker, scenario, overlay = build(semantic("request_refill"))
    result = await overlay.resolve(
        "Thanks, Alex. Your demo patient profile is set up. How can I help you today?",
        tracker.snapshot(),
    )
    assert result.reason == "medication_refill:request_refill"
    assert result.text != "Yes, please."
    snapshot = tracker.snapshot()
    observed = snapshot.target_observations["profile_created_acknowledged"]
    assert observed.value is True
    assert observed.provenance is StateProvenance.TARGET_OBSERVED
    assert overlay.metadata()["deterministic_prerequisite_hits"] == 0
    assert scenario.metadata()["caller_truth"]["date_of_birth"] == "April 12, 1998"


@pytest.mark.asyncio
async def test_actionable_profile_request_keeps_combined_consent_and_name():
    tracker, _, overlay = build()
    result = await overlay.resolve(
        "Would you like to create a demo patient profile? I just need your first and last name to get started.",
        tracker.snapshot(),
    )
    assert result.reason == "prerequisite:provide_first_name_and_last_name"
    assert result.text == "Yes, please. my name is Chitragupta Subramnian Singh."


@pytest.mark.asyncio
async def test_exact_live_unavailability_recovers_then_escalates_once_then_blocks():
    turns = (
        "I don't see any medications on your chart that can be refilled right now. If you'd like to speak with someone about this, just let me know.",
        "There aren't any medications on your chart that I can refill at this time. If you want to talk to someone from our team about this, just let me know.",
        "I don't see any medications on your chart that can be refilled right now. If you'd like to speak with someone from our team about this, just let me know.",
    )
    outputs = [
        semantic(outcome="refill_unavailable", escalation=True),
        semantic(outcome="refill_unavailable", escalation=True),
        semantic(outcome="refill_unavailable", escalation=True),
    ]
    tracker, scenario, overlay = build(*outputs)
    results = []
    for turn in turns:
        result = await overlay.resolve(turn, tracker.snapshot())
        results.append(result)
        scenario.mark_decision_spoken(result)
    assert [item.reason for item in results] == [
        "medication_refill:ask_medication_list_setup",
        "medication_refill:repeat_medication_list_setup",
        "medication_refill:accept_human_escalation",
    ]
    assert sum("connect me" in item.text for item in results) == 1
    assert all(item.reason != "medication_refill:request_refill" for item in results)
    metadata = scenario.metadata()
    assert metadata["medication_setup_probe_unanswered"] is True
    assert tracker.snapshot().target_observations["refill_availability"].value == "unavailable"
    assert "refill_availability" not in tracker.snapshot().caller_truth


@pytest.mark.asyncio
async def test_add_medication_branch_resumes_with_scenario_owned_identity():
    tracker, scenario, overlay = build(
        semantic(outcome="refill_unavailable"),
        semantic("provide_medication", outcome="medication_list_setup_supported"),
    )
    first = await overlay.resolve("There are no medications on your chart.", tracker.snapshot())
    second = await overlay.resolve("Sure. What medication would you like me to add?", tracker.snapshot())
    assert first.reason == "medication_refill:ask_medication_list_setup"
    assert second.reason == "medication_refill:provide_medication"
    assert second.text == "The medication is lisinopril."
    assert scenario.metadata()["medication_list_setup_supported"] is True


@pytest.mark.asyncio
async def test_rejected_setup_uses_staff_then_domain_can_resume():
    tracker, scenario, overlay = build(
        semantic(outcome="refill_unavailable"),
        semantic(outcome="medication_list_setup_rejected", escalation=True),
        semantic("provide_medication"),
    )
    results = []
    for turn in (
        "No medications are listed.",
        "I can't add medications here, but I can connect you with our team.",
        "Before I connect you, what medication is this about?",
    ):
        result = await overlay.resolve(turn, tracker.snapshot())
        results.append(result)
        scenario.mark_decision_spoken(result)
    assert [result.reason for result in results] == [
        "medication_refill:ask_medication_list_setup",
        "medication_refill:ask_alternate_medication_setup",
        "medication_refill:provide_medication",
    ]
    assert scenario.metadata()["experiment_status"] == "in_progress"


@pytest.mark.asyncio
async def test_newer_capability_supersedes_unavailability_without_truth_contamination():
    tracker, _, overlay = build(
        semantic(outcome="refill_unavailable"),
        semantic("confirm_or_correct_target_claim", field="dose", value="20 mg"),
    )
    await overlay.resolve("Nothing is refillable right now.", tracker.snapshot())
    result = await overlay.resolve("I see twenty milligrams listed.", tracker.snapshot())
    snapshot = tracker.snapshot()
    assert result.reason == "medication_refill:correct_dose"
    assert snapshot.target_observations["refill_availability"].value == "available"
    assert snapshot.target_observations["dose"].value == "20 mg"
    assert snapshot.caller_truth["dose"].value == "10 mg"


@pytest.mark.asyncio
async def test_escalation_acknowledgment_is_observed_and_waits():
    tracker, scenario, overlay = build(
        semantic(outcome="escalation_acknowledged", response=False)
    )
    result = await overlay.resolve(
        "Okay, I'll connect you with someone who can help.", tracker.snapshot()
    )
    assert result.reason == "medication_refill:escalation_acknowledged"
    assert scenario.metadata()["human_escalation_acknowledged"] is True
    assert tracker.snapshot().target_observations[
        "human_escalation_acknowledged"
    ].provenance is StateProvenance.TARGET_OBSERVED


@pytest.mark.asyncio
async def test_unknown_clinical_fact_is_not_invented_and_requests_verification():
    tracker, _, overlay = build(semantic("handle_unknown_clinical_fact"))
    result = await overlay.resolve(
        "What is the prescription number and how many refills were authorized?",
        tracker.snapshot(),
    )
    assert result.reason == "medication_refill:request_setup_without_unknown_fact"
    assert "don't have" in result.text
    truth = tracker.snapshot().caller_truth
    assert "prescription_number" not in truth
    assert "refill_count" not in truth


@pytest.mark.asyncio
async def test_target_outcome_normalization_overrides_contradictory_refill_action():
    tracker, _, overlay = build(
        semantic("request_refill", outcome="refill_unavailable")
    )
    result = await overlay.resolve(
        "There are no medications listed.", tracker.snapshot()
    )
    assert result.reason == "medication_refill:ask_medication_list_setup"


@pytest.mark.asyncio
async def test_compound_added_acknowledgment_preserves_dose_question_and_lifecycle():
    tracker, scenario, overlay = build(semantic("ask_dose_on_file"))
    result = await overlay.resolve(
        "Lisinopril has been added. What strength?", tracker.snapshot()
    )
    assert result.reason == "medication_refill:ask_dose_on_file"
    assert scenario.metadata()["medication_list_setup_acknowledged"] is True


@pytest.mark.asyncio
async def test_setup_probe_beats_compound_transfer_offer_and_delivery_is_explicit():
    tracker, scenario, overlay = build(
        semantic(outcome="refill_unavailable", escalation=True),
        semantic(outcome="refill_unavailable", escalation=True),
    )
    first = await overlay.resolve(
        "I don't see any medications. I can connect you with someone.",
        tracker.snapshot(),
    )
    assert first.reason == "medication_refill:ask_medication_list_setup"
    assert "lisinopril" in first.text.casefold()
    assert scenario.metadata()["medication_setup_probe_proposed"] is True
    assert scenario.metadata()["medication_setup_probe_spoken"] is False

    scenario.mark_decision_suppressed(first)
    second = await overlay.resolve(
        "I still don't see any medication and can transfer you.", tracker.snapshot()
    )
    assert second.reason == "medication_refill:ask_medication_list_setup"
    assert scenario.metadata()["human_escalation_requested"] is False
    scenario.mark_decision_spoken(second)
    assert scenario.metadata()["medication_setup_probe_spoken"] is True


@pytest.mark.asyncio
async def test_supported_setup_identity_dose_ack_then_refill_and_success_progression():
    tracker, scenario, overlay = build(
        semantic(outcome="refill_unavailable"),
        semantic("provide_medication", outcome="medication_list_setup_supported"),
        semantic("ask_dose_on_file"),
        semantic(outcome="medication_added", response=False),
        semantic("provide_pharmacy_preference"),
    )
    turns = (
        "There are no medications on your chart.",
        "Sure. What medication should I add?",
        "What strength?",
        "I've added lisinopril ten milligrams to your profile.",
        "Which pharmacy should I use for lisinopril?",
    )
    reasons = []
    for turn in turns:
        decision = await overlay.resolve(turn, tracker.snapshot())
        reasons.append(decision.reason)
        scenario.mark_decision_spoken(decision)
    assert reasons == [
        "medication_refill:ask_medication_list_setup",
        "medication_refill:provide_medication",
        "medication_refill:ask_dose_on_file",
        "medication_refill:request_refill_after_setup",
        "medication_refill:provide_pharmacy_preference",
    ]
    metadata = scenario.metadata()
    assert metadata["medication_identity_spoken"] is True
    assert metadata["medication_dose_spoken"] is True
    assert metadata["medication_persistence_verified"] is True
    assert not scenario.oracle.medication_state_persistence_failure


@pytest.mark.asyncio
async def test_added_then_absent_after_spoken_refill_fires_persistence_oracle():
    tracker, scenario, overlay = build(
        semantic(outcome="refill_unavailable"),
        semantic(outcome="medication_added", response=False),
        semantic(outcome="refill_unavailable"),
    )
    setup = await overlay.resolve("No medications are listed.", tracker.snapshot())
    scenario.mark_decision_spoken(setup)
    refill = await overlay.resolve(
        "Okay, that's now on the medication list.", tracker.snapshot()
    )
    scenario.mark_decision_spoken(refill)
    later = await overlay.resolve(
        "I don't see any medications on your chart.", tracker.snapshot()
    )
    assert later.reason == "medication_refill:target_capability_blocked"
    assert scenario.oracle.medication_state_persistence_failure
    evidence = next(
        item for item in scenario.oracle.evidence
        if item.oracle_name == "medication_state_persistence_failure"
    )
    assert len(evidence.evidence_turns) == 4
    assert "now on the medication list" in evidence.evidence_turns[1]
    assert "later target absence" in evidence.evidence_turns[3]


@pytest.mark.asyncio
async def test_rejected_setup_gets_one_alternate_question_then_escalation():
    tracker, scenario, overlay = build(
        semantic(outcome="refill_unavailable"),
        semantic(outcome="medication_list_setup_rejected", escalation=True),
        semantic(outcome="medication_list_setup_rejected", escalation=True),
    )
    reasons = []
    for turn in (
        "No medications are listed.",
        "I can't add medications here, but I can connect you.",
        "There is no other way to add one here, but I can connect you.",
    ):
        decision = await overlay.resolve(turn, tracker.snapshot())
        reasons.append(decision.reason)
        scenario.mark_decision_spoken(decision)
    assert reasons == [
        "medication_refill:ask_medication_list_setup",
        "medication_refill:ask_alternate_medication_setup",
        "medication_refill:accept_human_escalation",
    ]


@pytest.mark.asyncio
async def test_wrong_added_dose_is_corrected_without_changing_caller_truth():
    tracker, scenario, overlay = build(
        semantic(outcome="refill_unavailable"),
        semantic(
            "confirm_or_correct_target_claim",
            outcome="medication_added",
            field="dose",
            value="20 mg",
        ),
    )
    setup = await overlay.resolve("No medications are listed.", tracker.snapshot())
    scenario.mark_decision_spoken(setup)
    correction = await overlay.resolve(
        "I've added lisinopril twenty milligrams.", tracker.snapshot()
    )
    assert correction.reason == "medication_refill:correct_dose"
    assert "ten milligrams" in correction.text.casefold()
    assert tracker.snapshot().caller_truth["dose"].value == "10 mg"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("turn", "output", "reason"),
    (
        (
            "What pharmacy do you use?",
            semantic("provide_pharmacy_preference"),
            "medication_refill:provide_pharmacy_preference",
        ),
        (
            "I'm unable to connect you or process this request.",
            semantic(outcome="workflow_blocked"),
            "medication_refill:target_capability_blocked",
        ),
        (
            "I can help with a refill if you tell me the medication.",
            semantic("provide_medication"),
            "medication_refill:provide_medication",
        ),
    ),
)
async def test_forward_recovery_branches(turn, output, reason):
    tracker, _, overlay = build(output)
    result = await overlay.resolve(turn, tracker.snapshot())
    assert result.reason == reason


def test_capability_block_has_distinct_scenario_result_not_success():
    tracker, scenario, _ = build()
    scenario._experiment_status = "target_capability_blocked"
    scenario._productive_recovery_paths_exhausted = True
    scenario._workflow_terminal = True
    scenario._blocking_turn = "I cannot proceed."
    projection = project_v3_flow_snapshot(
        tracker.snapshot(),
        scenario_id="medication-refill-correction",
        scenario_metadata=scenario.metadata(),
    )
    reason = scenario_termination_failure_reason(
        status=AsteriskTerminationStatus.PREMATURE_REMOTE_TERMINATION,
        projection=projection,
    )
    assert projection.objective_complete is False
    assert reason.startswith("target_capability_blocked:")
    assert "I cannot proceed" in reason


class ProductionSemantic:
    def __init__(self):
        self.last_observation = {}

    async def resolve(self, target_turn, snapshot):
        del snapshot
        text = target_turn.casefold()
        if "medications on your chart" in text:
            self.last_observation = semantic(
                outcome="refill_unavailable", escalation=True
            )
        elif "what medication" in text:
            self.last_observation = semantic("provide_medication")
        elif "twenty" in text:
            self.last_observation = semantic(
                "confirm_or_correct_target_claim", field="dose", value="20 mg"
            )
        else:
            self.last_observation = semantic("request_refill")


class Sink:
    def __init__(self):
        self.frames = []

    async def queue_frames(self, frames):
        self.frames.extend(frames)


def test_closest_production_path_speaks_recovery_and_resumed_domain(monkeypatch):
    async def run():
        monkeypatch.setenv("VOICEPROBE_SCENARIO", "medication-refill-correction")
        monkeypatch.setattr(production, "QwenV3FallbackRouter", ProductionSemantic)
        bridge = production.PipecatRuntimeBridge(tts_frame_factory=lambda text: text)
        sink = Sink()
        bridge.bind_frame_sink(sink)
        turns = (
            "Thanks, Alex. Your demo patient profile is set up. How can I help you today?",
            "I don't see any medications on your chart. If you'd like to speak with someone, let me know.",
            "Before I connect you, what medication is this about?",
            "I have twenty milligrams here.",
        )
        results = []
        for turn in turns:
            result = await bridge.runtime.process_turns([turn])
            results.append(result)
            if result.response_ready:
                await bridge.on_tts_stopped()
        assert [result.decision.reason for result in results] == [
            "medication_refill:request_refill",
            "medication_refill:ask_medication_list_setup",
            "medication_refill:provide_medication",
            "medication_refill:correct_dose",
        ]
        assert sink.frames == [result.decision.text for result in results]
        assert bridge.runtime.flow_controller.tracker.snapshot().caller_truth[
            "dose"
        ].value == "10 mg"

    import asyncio

    asyncio.run(run())


def test_production_suppressed_setup_probe_remains_eligible(monkeypatch):
    async def run():
        monkeypatch.setenv("VOICEPROBE_SCENARIO", "medication-refill-correction")
        monkeypatch.setattr(production, "QwenV3FallbackRouter", ProductionSemantic)
        bridge = production.PipecatRuntimeBridge(tts_frame_factory=lambda text: text)
        bridge.bind_frame_sink(Sink())
        first = await bridge.runtime.process_turns(
            ["I don't see any medications on your chart."]
        )
        assert first.decision.reason == "medication_refill:ask_medication_list_setup"
        await bridge.on_tts_suppressed()
        assert bridge.scenario_metadata["medication_setup_probe_spoken"] is False
        second = await bridge.runtime.process_turns(
            ["I still don't see any medications on your chart."]
        )
        assert second.decision.reason == "medication_refill:ask_medication_list_setup"

    import asyncio

    asyncio.run(run())


@pytest.mark.parametrize("persistence_failure", [False, True])
def test_no_telephony_production_medication_setup_replays(monkeypatch, persistence_failure):
    outputs = [
        semantic("request_refill"),
        semantic(outcome="refill_unavailable", escalation=True),
        semantic("provide_medication", outcome="medication_list_setup_supported"),
        semantic("ask_dose_on_file"),
        semantic(outcome="medication_added", response=False),
        (
            semantic(outcome="refill_unavailable")
            if persistence_failure
            else semantic("provide_pharmacy_preference")
        ),
    ]

    class ScriptedProductionSemantic:
        def __init__(self):
            self.backend = FakeBackend(outputs)
            self.last_observation = {}

        async def resolve(self, target_turn, snapshot):
            del target_turn, snapshot
            self.last_observation = await self.backend.generate_json()

    async def run():
        monkeypatch.setenv("VOICEPROBE_SCENARIO", "medication-refill-correction")
        monkeypatch.setattr(
            production, "QwenV3FallbackRouter", ScriptedProductionSemantic
        )
        bridge = production.PipecatRuntimeBridge(tts_frame_factory=lambda text: text)
        bridge.bind_frame_sink(Sink())
        turns = (
            "Your demo profile is set up. How can I help?",
            "I don't see any medications. I can connect you with staff.",
            "Sure. What medication should I add?",
            "What strength?",
            "I've added lisinopril ten milligrams to your profile.",
            (
                "I don't see any medications on your chart."
                if persistence_failure
                else "Which pharmacy should I use to continue the refill?"
            ),
        )
        reasons = []
        for turn in turns:
            result = await bridge.runtime.process_turns([turn])
            reasons.append(result.decision.reason)
            if result.response_ready:
                await bridge.on_tts_stopped()
        assert reasons[:5] == [
            "medication_refill:request_refill",
            "medication_refill:ask_medication_list_setup",
            "medication_refill:provide_medication",
            "medication_refill:ask_dose_on_file",
            "medication_refill:request_refill_after_setup",
        ]
        if persistence_failure:
            assert bridge.scenario_metadata["oracle_candidate"] is True
            assert bridge.scenario_metadata["oracle_evidence"][0]["oracle_name"] == (
                "medication_state_persistence_failure"
            )
        else:
            assert reasons[-1] == "medication_refill:provide_pharmacy_preference"
            assert bridge.scenario_metadata["medication_persistence_verified"] is True

    import asyncio

    asyncio.run(run())
