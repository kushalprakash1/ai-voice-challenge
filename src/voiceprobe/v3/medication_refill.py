"""Opt-in synthetic medication-refill correction assessment for VoiceProbe v3."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .flow_state import FlowSnapshot, SchedulingFlowTracker
from .models import DecisionKind, PolicyDecision
from .oracle_evidence import OracleEvidence
from .qwen_v3_fallback import QwenV3FallbackRouter


SCENARIO_ID = "medication-refill-correction"
MEDICATION = "lisinopril"
CORRECTION_DOSE = "10 mg"
PHARMACY_PREFERENCE = "the pharmacy on file"


@dataclass(frozen=True, slots=True)
class MedicationOracle:
    correction_retention_failure: bool = False
    medication_state_persistence_failure: bool = False
    evidence: tuple[OracleEvidence, ...] = ()


class MedicationRefillCorrectionScenario:
    """Python-owned grounding/state around the existing Qwen whole-turn path."""

    scenario_id = SCENARIO_ID

    def __init__(
        self,
        *,
        tracker: SchedulingFlowTracker,
        qwen: QwenV3FallbackRouter,
    ) -> None:
        self.tracker = tracker
        self.qwen = qwen
        self.tracker.establish_caller_truth(
            "medication", MEDICATION, evidence=f"{SCENARIO_ID} fixture"
        )
        self.tracker.establish_caller_truth(
            "dose", CORRECTION_DOSE, evidence=f"{SCENARIO_ID} fixture"
        )
        self.tracker.establish_caller_truth(
            "pharmacy_preference",
            PHARMACY_PREFERENCE,
            evidence=f"{SCENARIO_ID} fixture",
        )
        self._old_dose: str | None = None
        self._correction_asserted = False
        self._correction_acknowledged = False
        self._oracle = MedicationOracle()
        self._turn_index = 0
        self._correction_turn: str | None = None
        self._acknowledgment_turn: str | None = None
        self._old_claim_turn: str | None = None
        self.last_semantic_action = "none"
        self._refill_requested = False
        self._medication_provided = False
        self._medication_identity_spoken = False
        self._medication_dose_spoken = False
        self._pharmacy_handled = False
        self._workflow_terminal = False
        self._experiment_status = "in_progress"
        self._blocking_turn: str | None = None
        self._refill_unavailable_observed = False
        self._medication_setup_probe_required = False
        self._medication_setup_probe_proposed = False
        self._medication_setup_probe_spoken = False
        self._medication_setup_response_observed = False
        self._medication_setup_followup_proposed = False
        self._medication_setup_followup_spoken = False
        self._medication_setup_probe_unanswered = False
        self._alternate_setup_question_proposed = False
        self._alternate_setup_question_spoken = False
        self._medication_setup_unavailable = False
        self._medication_list_setup_supported = False
        self._medication_list_setup_rejected = False
        self._medication_list_setup_acknowledged = False
        self._medication_added_turn: str | None = None
        self._post_add_refill_proposed = False
        self._post_add_refill_spoken = False
        self._post_add_refill_turn: str | None = None
        self._medication_persistence_verified = False
        self._unknown_fact_followup_spoken = False
        self._human_escalation_offered = False
        self._human_escalation_requested = False
        self._human_escalation_acknowledged = False
        self._productive_recovery_paths_exhausted = False

    @property
    def oracle(self) -> MedicationOracle:
        return self._oracle

    @property
    def objective_complete(self) -> bool:
        """Conservative completion of the dose-correction experiment."""
        return (
            self._old_dose is not None
            and self._correction_asserted
            and self._correction_acknowledged
            and self._pharmacy_handled
        )

    @property
    def scenario_terminal(self) -> bool:
        return self.objective_complete or self._experiment_status == "target_capability_blocked"

    def mark_decision_spoken(self, decision: PolicyDecision) -> None:
        """Commit delivery-sensitive experiment state after playback completes."""
        if decision.reason == "medication_refill:ask_medication_list_setup":
            self._medication_setup_probe_spoken = True
        elif decision.reason == "medication_refill:repeat_medication_list_setup":
            self._medication_setup_followup_spoken = True
        elif decision.reason == "medication_refill:ask_alternate_medication_setup":
            self._alternate_setup_question_spoken = True
        elif decision.reason == "medication_refill:request_refill_after_setup":
            self._post_add_refill_spoken = True
            self._post_add_refill_turn = (
                f"turn={self._turn_index}; caller requested refill after setup"
            )
        elif decision.reason == "medication_refill:provide_medication":
            self._medication_identity_spoken = True
        elif decision.reason == "medication_refill:ask_dose_on_file":
            self._medication_dose_spoken = True
        elif decision.reason == "medication_refill:request_setup_without_unknown_fact":
            self._unknown_fact_followup_spoken = True

    def mark_decision_suppressed(self, decision: PolicyDecision) -> None:
        """A proposed but unsent setup action remains eligible for delivery."""
        if decision.reason == "medication_refill:ask_medication_list_setup":
            self._medication_setup_probe_proposed = False
        elif decision.reason == "medication_refill:repeat_medication_list_setup":
            self._medication_setup_followup_proposed = False
        elif decision.reason == "medication_refill:ask_alternate_medication_setup":
            self._alternate_setup_question_proposed = False
        elif decision.reason == "medication_refill:request_refill_after_setup":
            self._post_add_refill_proposed = False

    async def resolve(
        self,
        target_turn: str,
        snapshot: FlowSnapshot,
    ) -> PolicyDecision:
        self._turn_index += 1
        await self.qwen.resolve(target_turn, snapshot)
        observation = self.qwen.last_observation
        action = observation["medication_action"]
        outcome = observation.get("medication_outcome", "none")
        self.last_semantic_action = action

        self._observe_target_outcome(
            outcome=outcome,
            offers_human_escalation=bool(
                observation.get("offers_human_escalation", False)
            ),
            offers_medication_list_setup=bool(
                observation.get("offers_medication_list_setup", False)
            ),
            target_turn=target_turn,
        )

        self._advance_persistence_oracle(outcome=outcome, target_turn=target_turn)
        if (
            self._post_add_refill_spoken
            and outcome != "refill_unavailable"
            and action in {
                "provide_medication",
                "ask_dose_on_file",
                "provide_pharmacy_preference",
                "confirm_or_correct_target_claim",
                "finish",
            }
        ):
            self._medication_persistence_verified = True

        field_name = observation["extracted_target_field"]
        value = observation["extracted_target_value"]
        grounded_value = _grounded_target_value(target_turn, field_name, value)

        if grounded_value is not None:
            self.tracker.observe_target_value(
                field_name, grounded_value, evidence=target_turn
            )
            self._advance_oracle(
                field_name=field_name,
                value=grounded_value,
                target_turn=target_turn,
                acknowledged=observation["target_acknowledges_correction"],
            )

        return self._patient_decision(action, outcome, grounded_value, target_turn)

    def _patient_decision(
        self,
        action: str,
        outcome: str,
        grounded_value: str | None,
        target_turn: str,
    ) -> PolicyDecision:
        # Explicit domain questions always resume the workflow, even after an
        # earlier unavailable observation or while escalation is pending.
        if action in {
            "provide_medication",
            "ask_dose_on_file",
            "provide_pharmacy_preference",
            "confirm_or_correct_target_claim",
        }:
            self.tracker.observe_target_value(
                "refill_availability", "available", evidence=target_turn
            )

        if outcome == "escalation_acknowledged":
            return PolicyDecision(
                DecisionKind.WAIT,
                reason="medication_refill:escalation_acknowledged",
            )

        if (
            action
            not in {
                "provide_medication",
                "ask_dose_on_file",
                "provide_pharmacy_preference",
                "confirm_or_correct_target_claim",
            }
            and outcome
            in {"refill_unavailable", "medication_list_setup_rejected", "workflow_blocked"}
        ):
            recovery = self._recovery_decision(outcome, target_turn)
            if recovery is not None:
                return recovery

        if action == "request_refill":
            self._refill_requested = True
            return PolicyDecision(
                DecisionKind.STATE_OBJECTIVE,
                text="I'm calling to request a medication refill.",
                reason="medication_refill:request_refill",
            )
        if action == "provide_medication":
            self._medication_provided = True
            return PolicyDecision(
                DecisionKind.ANSWER_FACT,
                text=f"The medication is {MEDICATION}.",
                reason="medication_refill:provide_medication",
            )
        if action == "ask_dose_on_file":
            dose = self.tracker.snapshot().caller_truth.get("dose")
            text = (
                f"The correct dose is {dose.value}."
                if dose is not None
                else "What dose or strength do you have on file?"
            )
            return PolicyDecision(
                DecisionKind.CONTEXTUAL_ANSWER,
                text=text,
                reason="medication_refill:ask_dose_on_file",
            )
        if action == "provide_pharmacy_preference":
            self._pharmacy_handled = True
            return PolicyDecision(
                DecisionKind.ANSWER_FACT,
                text="Please use the pharmacy on file.",
                reason="medication_refill:provide_pharmacy_preference",
            )
        if action == "handle_unknown_clinical_fact":
            if not self._unknown_fact_followup_spoken:
                return PolicyDecision(
                    DecisionKind.CLARIFY,
                    text=(
                        "I don't have that information. Can the medication still "
                        "be added without it, or is there another setup route?"
                    ),
                    reason="medication_refill:request_setup_without_unknown_fact",
                )
            if self._human_escalation_offered and not self._human_escalation_requested:
                self._human_escalation_requested = True
                return PolicyDecision(
                    DecisionKind.GRANT_PERMISSION,
                    text="Yes, please connect me with someone who can help verify that information.",
                    reason="medication_refill:accept_human_escalation",
                )
            return PolicyDecision(
                DecisionKind.CLARIFY,
                text=(
                    "I don't have that information. Can it be verified, or can "
                    "someone help me with the refill?"
                ),
                reason="medication_refill:request_fact_verification",
            )
        if action == "finish":
            self._workflow_terminal = True
            return PolicyDecision(
                DecisionKind.CONTEXTUAL_ANSWER,
                text="No, that's all. Thank you.",
                reason="medication_refill:finish",
            )
        if action == "confirm_or_correct_target_claim" and grounded_value:
            field_name = self.qwen.last_observation["extracted_target_field"]
            if field_name == "dose":
                if grounded_value != CORRECTION_DOSE:
                    self.tracker.establish_caller_truth(
                        "dose",
                        CORRECTION_DOSE,
                        evidence="explicit synthetic caller correction",
                    )
                    self._correction_asserted = True
                    self._correction_turn = (
                        f"turn={self._turn_index}; caller correction: {CORRECTION_DOSE}"
                    )
                    if self._old_dose is None:
                        self._old_dose = grounded_value
                        self._old_claim_turn = (
                            f"turn={self._turn_index}; initial target claim: {target_turn}"
                        )
                    return PolicyDecision(
                        DecisionKind.CORRECT_FACT,
                        text=(
                            "No, that's not right. It should be ten "
                            "milligrams."
                        ),
                        reason="medication_refill:correct_dose",
                    )

                if self._correction_asserted:
                    self.tracker.commit_dialogue_value(
                        "dose",
                        CORRECTION_DOSE,
                        evidence=target_turn,
                        grounded_in_target="dose",
                    )
                    self._correction_acknowledged = True
                    self._acknowledgment_turn = (
                        f"turn={self._turn_index}; target acknowledgment: {target_turn}"
                    )
                return PolicyDecision(
                    DecisionKind.GRANT_PERMISSION,
                    text="Yes, ten milligrams is correct.",
                    reason="medication_refill:confirm_or_correct_target_claim",
                )

        if (
            outcome == "medication_added"
            and action == "none"
            and not self._post_add_refill_spoken
        ):
            self._post_add_refill_proposed = True
            return PolicyDecision(
                DecisionKind.STATE_OBJECTIVE,
                text="Great. I'd like to refill that medication.",
                reason="medication_refill:request_refill_after_setup",
            )

        if not self.qwen.last_observation["requires_response"]:
            return PolicyDecision(
                DecisionKind.WAIT, reason="medication_refill:no_response_required"
            )
        return PolicyDecision(
            DecisionKind.CLARIFY,
            text="Could you clarify what refill information you need?",
            reason="medication_refill:clarify",
        )

    def _observe_target_outcome(
        self,
        *,
        outcome: str,
        offers_human_escalation: bool,
        offers_medication_list_setup: bool,
        target_turn: str,
    ) -> None:
        if outcome == "refill_unavailable":
            self._refill_unavailable_observed = True
            self._medication_setup_probe_required = True
            self.tracker.observe_target_value(
                "refill_availability", "unavailable", evidence=target_turn
            )
        elif outcome == "medication_list_setup_supported":
            self._medication_list_setup_supported = True
            self._medication_setup_response_observed = self._medication_setup_probe_spoken
            self.tracker.observe_target_value(
                "medication_list_setup_supported", True, evidence=target_turn
            )
        elif outcome == "medication_list_setup_rejected":
            self._medication_list_setup_rejected = True
            self._medication_setup_response_observed = self._medication_setup_probe_spoken
            self.tracker.observe_target_value(
                "medication_list_setup_supported", False, evidence=target_turn
            )
        elif outcome == "medication_added":
            self._medication_list_setup_acknowledged = True
            self._medication_setup_response_observed = self._medication_setup_probe_spoken
            self._medication_added_turn = (
                f"turn={self._turn_index}; target acknowledged addition: {target_turn}"
            )
            self.tracker.observe_target_value(
                "medication_list_setup_acknowledged", True, evidence=target_turn
            )
            self.tracker.observe_target_value(
                "refill_availability", "available", evidence=target_turn
            )
        elif outcome == "escalation_acknowledged":
            self._human_escalation_acknowledged = True
            self.tracker.observe_target_value(
                "human_escalation_acknowledged", True, evidence=target_turn
            )

        if offers_human_escalation:
            self._human_escalation_offered = True
            self.tracker.observe_target_value(
                "human_escalation_offered", True, evidence=target_turn
            )
        if offers_medication_list_setup:
            self.tracker.observe_target_value(
                "medication_list_setup_offered", True, evidence=target_turn
            )

    def _recovery_decision(
        self, outcome: str, target_turn: str
    ) -> PolicyDecision | None:
        if not self._medication_setup_probe_spoken and outcome != "workflow_blocked":
            self._medication_setup_probe_proposed = True
            self.last_semantic_action = "ask_medication_list_setup"
            return PolicyDecision(
                DecisionKind.CONTEXTUAL_ANSWER,
                text=(
                    "Can you add lisinopril to my demo profile "
                    "so I can continue with the refill request?"
                ),
                reason="medication_refill:ask_medication_list_setup",
            )

        if self._medication_list_setup_rejected and not self._medication_setup_unavailable:
            if not self._alternate_setup_question_spoken:
                self._alternate_setup_question_proposed = True
                return PolicyDecision(
                    DecisionKind.CLARIFY,
                    text="Is there another way to add a medication to this demo profile?",
                    reason="medication_refill:ask_alternate_medication_setup",
                )
            self._medication_setup_unavailable = True

        unanswered_redirect = (
            self._medication_setup_probe_spoken
            and not self._medication_setup_response_observed
            and outcome == "refill_unavailable"
        )
        if unanswered_redirect and not self._medication_setup_followup_spoken:
            self._medication_setup_followup_proposed = True
            return PolicyDecision(
                DecisionKind.CLARIFY,
                text="Before transferring me, can you add lisinopril to this demo profile?",
                reason="medication_refill:repeat_medication_list_setup",
            )
        if unanswered_redirect and self._medication_setup_followup_spoken:
            self._medication_setup_probe_unanswered = True

        if self._human_escalation_offered and not self._human_escalation_requested:
            self._human_escalation_requested = True
            self.last_semantic_action = "accept_human_escalation"
            return PolicyDecision(
                DecisionKind.GRANT_PERMISSION,
                text="Yes, please connect me with someone who can help with the refill.",
                reason="medication_refill:accept_human_escalation",
            )

        # A repeated unavailability after chart setup was attempted and an
        # offered escalation was accepted is grounded exhaustion, not a retry.
        if outcome == "workflow_blocked" or (
            self._medication_setup_probe_spoken
            and (
                self._medication_setup_unavailable
                or self._medication_setup_probe_unanswered
                or self._human_escalation_requested
                or not self._human_escalation_offered
            )
        ):
            self._productive_recovery_paths_exhausted = True
            self._experiment_status = "target_capability_blocked"
            self._blocking_turn = target_turn
            self._workflow_terminal = True
            self.last_semantic_action = "target_capability_blocked"
            self.tracker.observe_target_value(
                "target_capability_blocked", True, evidence=target_turn
            )
            return PolicyDecision(
                DecisionKind.WAIT,
                reason="medication_refill:target_capability_blocked",
            )
        return None

    def _advance_oracle(
        self,
        *,
        field_name: str,
        value: str,
        target_turn: str,
        acknowledged: bool,
    ) -> None:
        if field_name != "dose":
            return
        if not self._correction_asserted:
            self._old_dose = value
            self._old_claim_turn = (
                f"turn={self._turn_index}; initial target claim: {target_turn}"
            )
            return
        if value == CORRECTION_DOSE and acknowledged:
            self._correction_acknowledged = True
            self._acknowledgment_turn = (
                f"turn={self._turn_index}; target acknowledgment: {target_turn}"
            )
            return
        if (
            self._correction_acknowledged
            and self._old_dose is not None
            and value == self._old_dose
        ):
            retained = tuple(
                item for item in self._oracle.evidence
                if item.oracle_name != "correction_retention_failure"
            )
            self._oracle = MedicationOracle(
                correction_retention_failure=True,
                medication_state_persistence_failure=(
                    self._oracle.medication_state_persistence_failure
                ),
                evidence=retained + (
                    OracleEvidence(
                        oracle_name="correction_retention_failure",
                        status="candidate",
                        scenario=SCENARIO_ID,
                        field="dose",
                        expected_value=CORRECTION_DOSE,
                        observed_value=value,
                        evidence_turns=tuple(
                            item
                            for item in (
                                self._old_claim_turn,
                                self._correction_turn,
                                self._acknowledgment_turn,
                                f"turn={self._turn_index}; later target claim: {target_turn}",
                            )
                            if item is not None
                        ),
                        relevant_provenance=(
                            "target_observed",
                            "caller_scenario",
                            "committed_validated",
                            "target_observed",
                        ),
                        reason=(
                            "Target returned to the old dose after the caller "
                            "correction was acknowledged."
                        ),
                    ),
                ),
            )

    def _advance_persistence_oracle(self, *, outcome: str, target_turn: str) -> None:
        if not (
            outcome == "refill_unavailable"
            and self._medication_setup_probe_spoken
            and self._medication_list_setup_acknowledged
            and self._post_add_refill_spoken
            and self._medication_added_turn is not None
            and self._post_add_refill_turn is not None
        ):
            return
        persistence = OracleEvidence(
            oracle_name="medication_state_persistence_failure",
            status="candidate",
            scenario=SCENARIO_ID,
            field="medication_list",
            expected_value=MEDICATION,
            observed_value="absent_or_not_refillable",
            evidence_turns=(
                "caller setup request was delivered: lisinopril",
                self._medication_added_turn,
                self._post_add_refill_turn,
                f"turn={self._turn_index}; later target absence claim: {target_turn}",
            ),
            relevant_provenance=(
                "caller_scenario_spoken",
                "target_observed",
                "caller_scenario_spoken",
                "target_observed",
            ),
            reason=(
                "The target explicitly acknowledged adding the medication, then "
                "reported no medication after a delivered refill request."
            ),
        )
        retained = tuple(
            item for item in self._oracle.evidence
            if item.oracle_name != "medication_state_persistence_failure"
        )
        self._oracle = MedicationOracle(
            correction_retention_failure=self._oracle.correction_retention_failure,
            medication_state_persistence_failure=True,
            evidence=retained + (persistence,),
        )

    def metadata(self) -> dict[str, object]:
        snapshot = self.tracker.snapshot()
        return {
            "scenario": SCENARIO_ID,
            "scenario_stage": self.last_semantic_action,
            "caller_truth": {k: v.value for k, v in snapshot.caller_truth.items()},
            "target_observations": {
                k: v.value for k, v in snapshot.target_observations.items()
            },
            "committed_state": {
                k: v.value for k, v in snapshot.committed_dialogue.items()
            },
            "rejected_extractions": (),
            "oracle_evidence": tuple(asdict(item) for item in self._oracle.evidence),
            "refill_requested": self._refill_requested,
            "medication_provided": self._medication_provided,
            "medication_identity_spoken": self._medication_identity_spoken,
            "medication_dose_spoken": self._medication_dose_spoken,
            "target_old_dose_observed": self._old_dose is not None,
            "dose_correction_spoken": self._correction_asserted,
            "dose_acknowledged": self._correction_acknowledged,
            "pharmacy_handled": self._pharmacy_handled,
            "workflow_terminal": self._workflow_terminal,
            "experiment_status": (
                "experiment_completed" if self.objective_complete else self._experiment_status
            ),
            "scenario_terminal": self.scenario_terminal,
            "blocking_target_statement": self._blocking_turn,
            "refill_unavailable_observed": self._refill_unavailable_observed,
            "medication_setup_probe_required": self._medication_setup_probe_required,
            "medication_setup_probe_proposed": self._medication_setup_probe_proposed,
            "medication_setup_probe_spoken": self._medication_setup_probe_spoken,
            "medication_setup_response_observed": self._medication_setup_response_observed,
            "medication_setup_followup_spoken": self._medication_setup_followup_spoken,
            "medication_setup_probe_unanswered": self._medication_setup_probe_unanswered,
            "medication_setup_unavailable": self._medication_setup_unavailable,
            "medication_list_setup_supported": self._medication_list_setup_supported,
            "medication_list_setup_rejected": self._medication_list_setup_rejected,
            "medication_list_setup_acknowledged": self._medication_list_setup_acknowledged,
            "medication_persistence_verified": self._medication_persistence_verified,
            "human_escalation_offered": self._human_escalation_offered,
            "human_escalation_requested": self._human_escalation_requested,
            "human_escalation_acknowledged": self._human_escalation_acknowledged,
            "productive_recovery_paths_exhausted": self._productive_recovery_paths_exhausted,
            "objective_complete": self.objective_complete,
        }


_DOSE_WORDS = {"ten": "10 mg", "twenty": "20 mg"}


def _grounded_target_value(
    target_turn: str,
    field_name: str,
    extracted_value: str,
) -> str | None:
    """Reject Qwen values that have no lexical evidence in target speech."""

    if field_name == "none" or not extracted_value.strip():
        return None
    normalized_turn = " ".join(target_turn.casefold().split())
    normalized_value = " ".join(extracted_value.casefold().split())
    if field_name == "dose":
        digit = re.search(r"\b(\d+)\s*(?:mg|milligrams?)\b", normalized_turn)
        if digit is not None:
            grounded = f"{int(digit.group(1))} mg"
            return grounded if grounded == normalized_value else None
        for word, grounded in _DOSE_WORDS.items():
            if re.search(rf"\b{word}\s+(?:mg|milligrams?)\b", normalized_turn):
                return grounded if grounded == normalized_value else None
            if re.search(rf"\b(?:say|mean|dose|dosage|strength)\s+{word}\b", normalized_turn):
                return grounded if grounded == normalized_value else None
            if re.search(rf"\bcorrect(?:ing|ed)?\b.*\bto\s+{word}\b", normalized_turn):
                return grounded if grounded == normalized_value else None
            if re.search(
                rf"\b(?:have|show|listed)\b.*\b{word}\b(?:\s+listed)?",
                normalized_turn,
            ):
                return grounded if grounded == normalized_value else None
        return None
    return extracted_value.strip() if normalized_value in normalized_turn else None
