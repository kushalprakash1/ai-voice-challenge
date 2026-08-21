"""Opt-in self-pay and dynamic-location adversarial scenario for VoiceProbe v3."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .flow_state import FlowSnapshot, SchedulingFlowTracker
from .models import DecisionKind, PolicyDecision
from .oracle_evidence import OracleEvidence
from .qwen_v3_fallback import QwenV3FallbackRouter


SCENARIO_ID = "office-hours-location-insurance"
SELF_PAY = "self_pay"


class SelfPayLocationSwitchScenario:
    scenario_id = SCENARIO_ID

    def __init__(self, *, tracker: SchedulingFlowTracker, qwen: QwenV3FallbackRouter, scenario_id: str = SCENARIO_ID) -> None:
        self.scenario_id = scenario_id
        self.tracker = tracker
        self.qwen = qwen
        self.last_semantic_action = "none"
        self.rejected_extractions: list[dict[str, object]] = []
        self._turn_index = 0
        self._self_pay_stated = False
        self._self_pay_acknowledged = False
        self._insurance_exercised = False
        self._self_pay_contextual_exercised = False
        self._self_pay_revisited = False
        self._locations_discovered = False
        self._location_discovery_exercised = False
        self._location_switch_exercised = False
        self._hours_exercised = False
        self._contextual_hours_exercised = False
        self._weekend_hours_exercised = False
        self._hours_queries_completed = 0
        self._transfer_declined = False
        self._offered_locations: list[str] = []
        self._selected_location: str | None = None
        self._selected_locations: list[str] = []
        self._selected_acknowledged = False
        self._prior_location: str | None = None
        self._switched_location: str | None = None
        self._switch_acknowledged = False
        self._switch_turn: str | None = None
        self._self_pay_ack_turn: str | None = None
        self._hours_followup_pending = False
        self._hours_question_turn: str | None = None
        self._hours_by_location: dict[str, str] = {}
        self._hours_turn_by_location: dict[str, str] = {}
        self._evidence: list[OracleEvidence] = []
        self._last_location_mismatch = False
        self._pending_location_by_text: dict[str, str] = {}

    def mark_decision_spoken(self, decision: PolicyDecision) -> None:
        """Commit caller-owned state only after outbound playback completes."""
        reason = decision.reason
        if reason == "self_pay_location:establish_self_pay":
            self._self_pay_stated = True
            self.tracker.establish_caller_truth(
                "insurance_status", SELF_PAY, evidence=decision.text
            )
        elif reason == "self_pay_location:select_location":
            choice = self._pending_location_by_text.pop(decision.text, None)
            if choice:
                self._selected_location = choice
                if choice not in self._selected_locations:
                    self._selected_locations.append(choice)
                self._selected_acknowledged = False
                self.tracker.establish_caller_truth("selected_location", choice, evidence=decision.text)
        elif reason == "self_pay_location:ask_location_hours":
            self._hours_exercised = True; self._hours_queries_completed += 1
        elif reason == "self_pay_location:ask_self_pay_at_location":
            self._self_pay_contextual_exercised = True
        elif reason == "self_pay_location:switch_location":
            choice = self._pending_location_by_text.pop(decision.text, None)
            if choice:
                self._prior_location = self._selected_location
                self._switched_location = choice; self._selected_location = choice
                if choice not in self._selected_locations: self._selected_locations.append(choice)
                self._selected_acknowledged = False; self._location_switch_exercised = True
                self._switch_turn = f"turn={self._turn_index}; caller switch: {decision.text}"
                self.tracker.establish_caller_truth("selected_location", choice, evidence=decision.text)
        elif reason == "self_pay_location:ask_active_location_hours":
            self._contextual_hours_exercised = True; self._hours_queries_completed += 1
        elif reason == "self_pay_location:ask_saturday_hours":
            self._weekend_hours_exercised = True; self._hours_queries_completed += 1
        elif reason == "self_pay_location:revisit_self_pay":
            self._self_pay_revisited = True
        elif reason == "self_pay_location:decline_transfer_continue":
            self._transfer_declined = True

    def mark_decision_suppressed(self, decision: PolicyDecision) -> None:
        self._pending_location_by_text.pop(decision.text, None)

    @property
    def oracle_evidence(self) -> tuple[OracleEvidence, ...]:
        return tuple(self._evidence)

    @property
    def objective_complete(self) -> bool:
        """Complete only after the reachable cross-domain milestones."""
        location_goal = (
            self._location_switch_exercised and self._contextual_hours_exercised
            if len(self._offered_locations) >= 2
            else self._hours_exercised
        )
        return (
            self._insurance_exercised
            and self._self_pay_stated
            and self._locations_discovered
            and self._hours_exercised
            and self._self_pay_contextual_exercised
            and self._weekend_hours_exercised
            and self._self_pay_revisited
            and location_goal
        )

    async def resolve(self, target_turn: str, snapshot: FlowSnapshot) -> PolicyDecision:
        self._turn_index += 1
        await self.qwen.resolve(target_turn, snapshot)
        observation = self.qwen.last_observation
        action = observation["self_pay_location_action"]
        self.last_semantic_action = action
        if action in {"establish_self_pay", "provide_self_pay"} or observation["extracted_insurance_status"] != "none" or observation["extracted_insurer"]:
            self._insurance_exercised = True

        locations = self._ground_locations(
            target_turn, observation["extracted_locations"]
        )
        location = self._ground_scalar(
            target_turn, "location", observation["extracted_location"]
        )
        if location and location not in locations:
            locations.append(location)
        if locations:
            self._observe_locations(locations, target_turn)
            self._locations_discovered = True
            self._location_discovery_exercised = True

        insurance = observation["extracted_insurance_status"]
        insurer = self._ground_scalar(
            target_turn, "insurer", observation["extracted_insurer"]
        )
        if insurance != "none" and self._insurance_grounded(target_turn, insurance, insurer):
            self.tracker.observe_target_value(
                "insurance_status", insurance, evidence=target_turn
            )
            if insurer:
                self.tracker.observe_target_value("insurer", insurer, evidence=target_turn)
            self._evaluate_self_pay_regression(insurance, insurer, target_turn)
        elif insurance != "none":
            self._reject("insurance_status", insurance, target_turn)

        if observation["target_acknowledges_self_pay"] and self._self_pay_stated:
            self._self_pay_acknowledged = True
            self._self_pay_ack_turn = (
                f"turn={self._turn_index}; target self-pay acknowledgment: {target_turn}"
            )
            self.tracker.commit_dialogue_value(
                "insurance_status", SELF_PAY, evidence=target_turn
            )

        acknowledged_location = self._ground_scalar(
            target_turn,
            "acknowledged_location",
            observation["target_acknowledged_location"],
        )
        if acknowledged_location:
            self.tracker.observe_target_value(
                "target_acknowledged_location", acknowledged_location, evidence=target_turn
            )
            self._acknowledge_location(acknowledged_location, target_turn)

        self._evaluate_location_reversion(
            location,
            target_turn,
            observation["target_asserts_active_location"],
        )
        self._observe_hours(target_turn, observation)
        return self._patient_decision(action, target_turn)

    def _patient_decision(self, action: str, target_turn: str) -> PolicyDecision:
        lower = target_turn.casefold()
        if ("transfer" in lower or "connect" in lower or "someone" in lower) and not self._transfer_declined:
            return PolicyDecision(DecisionKind.CONTEXTUAL_ANSWER, text="No thanks, I just have another question. What insurance plans do you accept?", reason="self_pay_location:decline_transfer_continue")
        if not self._insurance_exercised:
            return PolicyDecision(DecisionKind.CONTEXTUAL_ANSWER, text="What insurance plans do you accept?", reason="self_pay_location:ask_insurance")
        if not self._self_pay_stated:
            return PolicyDecision(
                DecisionKind.ANSWER_FACT,
                text="I don't have insurance. I'll be self-pay.",
                reason="self_pay_location:establish_self_pay",
            )
        if not self._locations_discovered:
            return PolicyDecision(DecisionKind.CONTEXTUAL_ANSWER, text="What office locations do you have?", reason="self_pay_location:ask_locations")
        if self._selected_location is None:
            choice = self._next_location_choice()
            if choice is not None:
                text = "I'd like the first location you mentioned."
                reason = "self_pay_location:select_location"
                self._pending_location_by_text[text] = choice
                return PolicyDecision(DecisionKind.CONTEXTUAL_ANSWER, text=text, reason=reason)
        if self._selected_location and not self._hours_exercised:
            self._hours_followup_pending = True
            return PolicyDecision(DecisionKind.CONTEXTUAL_ANSWER, text="What are the hours for that location?", reason="self_pay_location:ask_location_hours")
        if not self._self_pay_contextual_exercised:
            return PolicyDecision(DecisionKind.CONTEXTUAL_ANSWER, text="Can I still be seen at that location if I'm self-pay?", reason="self_pay_location:ask_self_pay_at_location")
        choice = self._next_location_choice()
        if choice is not None and not self._location_switch_exercised:
            text = "Actually, I'd like to switch to the second location you mentioned."
            self._pending_location_by_text[text] = choice
            return PolicyDecision(DecisionKind.CONTEXTUAL_ANSWER, text=text, reason="self_pay_location:switch_location")
        if self._location_switch_exercised and not self._contextual_hours_exercised:
            self._hours_followup_pending = True
            self._hours_question_turn = f"turn={self._turn_index}; caller question: What time does that office close?"
            return PolicyDecision(DecisionKind.CONTEXTUAL_ANSWER, text="What time does that office close?", reason="self_pay_location:ask_active_location_hours")
        if not self._weekend_hours_exercised:
            self._hours_followup_pending = True
            return PolicyDecision(DecisionKind.CONTEXTUAL_ANSWER, text="What about Saturday?", reason="self_pay_location:ask_saturday_hours")
        if not self._self_pay_revisited:
            return PolicyDecision(DecisionKind.CONTEXTUAL_ANSWER, text="And you still have me as self-pay, correct?", reason="self_pay_location:revisit_self_pay")
        if self.objective_complete or action == "finish":
            return PolicyDecision(
                DecisionKind.CONTEXTUAL_ANSWER,
                text="No, that's all. Thank you.",
                reason="self_pay_location:finish",
            )
        if action == "confirm_location" and self._last_location_mismatch:
            return PolicyDecision(
                DecisionKind.CORRECT_FACT,
                text=f"No, I switched to {self._switched_location}.",
                reason="self_pay_location:correct_active_location",
            )
        if not self.qwen.last_observation["requires_response"]:
            return PolicyDecision(DecisionKind.WAIT, reason="self_pay_location:wait")
        return PolicyDecision(
            DecisionKind.CLARIFY,
            text="Could you clarify which insurance or location detail you need?",
            reason="self_pay_location:clarify",
        )

    def _next_location_choice(self) -> str | None:
        for location in self._offered_locations:
            if location not in self._selected_locations:
                return location
        return None

    def _observe_locations(self, locations: list[str], turn: str) -> None:
        for location in locations:
            if location not in self._offered_locations:
                self._offered_locations.append(location)
        self.tracker.observe_target_value(
            "offered_locations", tuple(self._offered_locations), evidence=turn
        )

    def _acknowledge_location(self, location: str, turn: str) -> None:
        if location != self._selected_location:
            return
        self._selected_acknowledged = True
        if self._switched_location == location:
            self._switch_acknowledged = True
        self.tracker.commit_dialogue_value("selected_location", location, evidence=turn)

    def _observe_hours(self, turn: str, observation: dict[str, Any]) -> None:
        interval = self._ground_scalar(turn, "office_hours", observation["office_hours"])
        if not interval:
            return
        explicit_location = self._ground_scalar(
            turn, "hours_location", observation["hours_location"]
        )
        location = explicit_location or (
            self._selected_location if self._hours_followup_pending else None
        )
        if location is None:
            self._reject("office_hours", interval, turn)
            return
        day = next((name for name in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday") if name in turn.casefold()), "general")
        hours_key = f"{location}|{day}"
        key = f"office_hours:{hours_key}"
        prior = self._hours_by_location.get(hours_key)
        self.tracker.observe_target_value(key, interval, evidence=turn)
        if prior and prior != interval and not observation["hours_context_changed"]:
            self._add_evidence(
                "office_hours_internal_contradiction",
                "office_hours",
                prior,
                interval,
                (
                    self._hours_turn_by_location.get(
                        hours_key, f"prior hours: {prior}"
                    ),
                    self._turn("contradiction", turn),
                ),
                "Same grounded location received incompatible hours without a context change.",
            )
        self._hours_by_location[hours_key] = interval
        self._hours_turn_by_location[hours_key] = self._turn("target hours", turn)
        self._hours_followup_pending = False

    def _evaluate_location_reversion(
        self, location: str | None, turn: str, asserts_active: bool
    ) -> None:
        self._last_location_mismatch = False
        if not (
            asserts_active
            and location
            and self._switch_acknowledged
            and self._prior_location
        ):
            return
        if location != self._prior_location:
            return
        self._last_location_mismatch = True
        active = self._switched_location
        self._add_evidence(
            "location_switch_retention_failure",
            "selected_location",
            active,
            location,
            (
                self._switch_turn or f"caller switch: {active}",
                self._turn("reversion", turn),
            ),
            "Target reverted to the prior location after acknowledging the switch.",
        )
        if self._hours_followup_pending:
            self._add_evidence(
                "active_location_context_mismatch",
                "selected_location",
                active,
                location,
                (
                    self._hours_question_turn or f"that-office question: {active}",
                    self._turn("answer", turn),
                ),
                "Target answered the active-location follow-up using the prior location.",
            )

    def _evaluate_self_pay_regression(
        self, status: str, insurer: str | None, turn: str
    ) -> None:
        if not self._self_pay_acknowledged:
            return
        lower = turn.casefold()
        contradictory = (status in {"insured", "specific_insurer"} or bool(insurer)) and any(phrase in lower for phrase in ("your insurance is", "you have insurance", "you are insured", "your insurer is"))
        if contradictory:
            self._add_evidence(
                "self_pay_state_regression",
                "insurance_status",
                SELF_PAY,
                insurer or status,
                (
                    self._self_pay_ack_turn or f"self-pay acknowledgment: {SELF_PAY}",
                    self._turn("regression", turn),
                ),
                "Target asserted an incompatible insurance state after acknowledging self-pay.",
            )

    def _add_evidence(
        self,
        name: str,
        field: str,
        expected: object,
        observed: object,
        turns: tuple[str, ...],
        reason: str,
    ) -> None:
        if any(item.oracle_name == name for item in self._evidence):
            return
        self._evidence.append(
            OracleEvidence(
                oracle_name=name,
                status="candidate",
                scenario=self.scenario_id,
                field=field,
                expected_value=expected,
                observed_value=observed,
                evidence_turns=turns,
                relevant_provenance=(
                    "caller_scenario", "target_observed", "committed_validated"
                ),
                reason=reason,
            )
        )

    def _ground_locations(self, turn: str, candidates: tuple[str, ...]) -> list[str]:
        grounded: list[str] = []
        for value in candidates:
            result = self._ground_scalar(turn, "location", value)
            if result and result not in grounded:
                grounded.append(result)
        return grounded

    def _ground_scalar(self, turn: str, field: str, value: str) -> str | None:
        candidate = " ".join(value.split())
        if not candidate:
            return None
        if candidate.casefold() in " ".join(turn.casefold().split()):
            return candidate
        self._reject(field, candidate, turn)
        return None

    def _insurance_grounded(
        self, turn: str, status: str, insurer: str | None
    ) -> bool:
        text = turn.casefold()
        if status == "self_pay":
            return "self-pay" in text or "self pay" in text or "out of pocket" in text
        if status == "insured":
            return "insured" in text or "insurance" in text
        return insurer is not None

    def _reject(self, field: str, value: object, turn: str) -> None:
        self.rejected_extractions.append(
            {"field": field, "value": value, "source_turn": turn}
        )

    def _turn(self, label: str, text: str) -> str:
        return f"turn={self._turn_index}; {label}: {text}"

    def metadata(self) -> dict[str, object]:
        snapshot = self.tracker.snapshot()
        return {
            "scenario": self.scenario_id,
            "scenario_stage": self.last_semantic_action,
            "selected_location": self._selected_location,
            "switch_acknowledged": self._switch_acknowledged,
            "caller_truth": {k: v.value for k, v in snapshot.caller_truth.items()},
            "target_observations": {
                k: v.value for k, v in snapshot.target_observations.items()
            },
            "committed_state": {
                k: v.value for k, v in snapshot.committed_dialogue.items()
            },
            "rejected_extractions": tuple(self.rejected_extractions),
            "oracle_evidence": tuple(asdict(item) for item in self._evidence),
            "self_pay_stated": self._self_pay_stated,
            "self_pay_acknowledged": self._self_pay_acknowledged,
            "insurance_exercised": self._insurance_exercised,
            "self_pay_established": self._self_pay_stated,
            "locations_discovered": self._locations_discovered,
            "location_a": self._offered_locations[0] if self._offered_locations else None,
            "location_b": self._offered_locations[1] if len(self._offered_locations) > 1 else None,
            "location_discovery_exercised": self._location_discovery_exercised,
            "location_switch_exercised": self._location_switch_exercised,
            "hours_exercised": self._hours_exercised,
            "contextual_hours_exercised": self._contextual_hours_exercised,
            "hours_queries_completed": self._hours_queries_completed,
            "contextual_hours_completed": self._contextual_hours_exercised and self._weekend_hours_exercised,
            "oracle_candidates": tuple(item.oracle_name for item in self._evidence),
            "objective_complete": self.objective_complete,
        }
