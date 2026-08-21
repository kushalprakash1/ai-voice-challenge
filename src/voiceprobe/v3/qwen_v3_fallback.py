"""Qwen 4B whole-burst semantic fallback for VoiceProbe v3.

Ownership:
- Existing deterministic SchedulingFlowController remains first authority.
- Existing FlowTracker remains durable workflow state.
- Existing slot resolver/grounding remains sole concrete-slot authority.
- Qwen sees ONLY complete unresolved clinic speech and returns structured semantics.
- Qwen never receives patient truth values and never writes patient dialogue.
- Python PatientFacts + current FlowSnapshot choose the patient response.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from typing import Any

from voiceprobe.v3.models import DecisionKind, PatientFacts, PolicyDecision
from voiceprobe.v33.reasoner import OllamaBackend, OllamaConfig


_FLAG = "VOICEPROBE_V3_QWEN_FALLBACK"


def qwen_v3_fallback_enabled_from_environment() -> bool:
    value = os.environ.get(_FLAG)
    if value in (None, "", "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{_FLAG} must be exactly '0' or '1'.")


def _enum_or_default(kind_name: str, fallback: DecisionKind) -> DecisionKind:
    return getattr(DecisionKind, kind_name, fallback)


class QwenV3FallbackRouter:
    """Perception-only Qwen router for unresolved complete PGAI bursts."""

    def __init__(
        self,
        *,
        facts: PatientFacts | None = None,
        endpoint: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        semantic_domain: str | None = None,
        backend: Any | None = None,
    ) -> None:
        self.facts = facts or PatientFacts()
        if semantic_domain not in {None, "medication", "self_pay_location"}:
            raise ValueError(f"Unsupported Qwen V3 semantic domain: {semantic_domain}")
        self.semantic_domain = semantic_domain

        self.endpoint = (
            endpoint
            or os.environ.get(
                "VOICEPROBE_OLLAMA_URL",
                "http://127.0.0.1:11434/api/chat",
            )
        )
        self.model = (
            model
            or os.environ.get(
                "VOICEPROBE_REASONING_V2_EDGE_MODEL",
                "qwen3.5:4b",
            )
        )
        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.environ.get(
                "VOICEPROBE_V3_QWEN_TIMEOUT_SECONDS",
                "3.5",
            )
        )
        self.startup_timeout_seconds = float(
            os.environ.get(
                "VOICEPROBE_V3_QWEN_STARTUP_TIMEOUT_SECONDS",
                "20.0",
            )
        )

        self._backend = backend or OllamaBackend(
            OllamaConfig(
                endpoint=self.endpoint,
                model=self.model,
                timeout_seconds=self.timeout_seconds,
                keep_alive="15m",
                num_ctx=768,
                temperature=0.0,
            )
        )

        self.last_observation: dict[str, Any] = {}
        self.last_raw_observation: dict[str, Any] = {}
        self.last_complete_turn = ""

        # A cold qwen3.5:4b load can take several seconds, while warm inference
        # is comfortably inside the 2.5 s live-turn budget. Production-owned
        # routers therefore load the model once at startup with a separate,
        # generous timeout. Injected test backends are never auto-warmed.
        if backend is None:
            self.warmup_sync()

    def _startup_backend(self) -> OllamaBackend:
        return OllamaBackend(
            OllamaConfig(
                endpoint=self.endpoint,
                model=self.model,
                timeout_seconds=self.startup_timeout_seconds,
                keep_alive="15m",
                num_ctx=768,
                temperature=0.0,
            )
        )

    async def warmup(self) -> None:
        backend = self._startup_backend()
        await backend.generate_json(
            system="Return the required JSON object only.",
            prompt='{"warmup":true}',
            schema={
                "type": "object",
                "properties": {
                    "ok": {"type": "boolean"},
                },
                "required": ["ok"],
                "additionalProperties": False,
            },
        )

    def warmup_sync(self) -> None:
        """Cold-load Qwen before telephony while preserving live 2.5 s timeout."""
        error: list[BaseException] = []

        def runner() -> None:
            try:
                asyncio.run(self.warmup())
            except BaseException as exc:
                error.append(exc)

        thread = threading.Thread(
            target=runner,
            name="voiceprobe-qwen-startup-warmup",
            daemon=True,
        )
        thread.start()
        thread.join()

        if error:
            raise error[0]

    async def resolve(
        self,
        agent_turn: str,
        snapshot: Any,
    ) -> PolicyDecision:
        complete_turn = " ".join(str(agent_turn).split())
        self.last_complete_turn = complete_turn

        if not complete_turn:
            return PolicyDecision(
                DecisionKind.WAIT,
                reason="qwen_v3_blank_burst",
            )

        raw = await self._backend.generate_json(
            system=self._system_prompt(self.semantic_domain),
            prompt=json.dumps(
                {
                    "clinic_turn": complete_turn,
                    "grounded_target_context": self._grounded_target_context(snapshot),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            schema=self._schema(self.semantic_domain),
        )

        self.last_raw_observation = dict(raw) if isinstance(raw, dict) else {"raw": raw}
        observation = self._normalize_observation(raw)
        if self.semantic_domain == "medication":
            self._ground_medication_outcome(complete_turn, observation)
        if (
            self.semantic_domain == "medication"
            and observation["extracted_target_field"] == "dose"
            and observation["extracted_target_value"]
        ):
            observation["medication_action"] = "confirm_or_correct_target_claim"
        if (
            self.semantic_domain == "self_pay_location"
            and observation["extracted_insurer"]
        ):
            observation["extracted_insurance_status"] = "specific_insurer"
            observation["self_pay_location_action"] = "provide_self_pay"
        if (
            observation["target_acknowledges_self_pay"]
            and observation["self_pay_location_action"]
            in {"establish_self_pay", "provide_self_pay"}
        ):
            observation["self_pay_location_action"] = "ask_locations"
        self.last_observation = observation

        return self._decision_for_observation(
            observation=observation,
            snapshot=snapshot,
        )

    @staticmethod
    def _ground_medication_outcome(
        complete_turn: str, observation: dict[str, Any]
    ) -> None:
        """Validate target lifecycle semantics against literal source structure."""
        text = complete_turn.casefold()
        outcome = observation["medication_outcome"]
        explicit_setup_rejection = bool(
            re.search(
                r"\b(?:cannot|can't|unable to)\s+(?:add|update)\b[^.!?]*"
                r"\b(?:medication|medications|medicine|prescription)\b",
                text,
            )
            or re.search(
                r"\bno\s+(?:alternate|other)\s+(?:mechanism|way|route)\b"
                r"[^.!?]*\badd\b[^.!?]*\bmedications?\b",
                text,
            )
        )
        # A compound rejection + staff offer is still a setup rejection. The
        # transfer clause must not erase the direct answer to the setup probe.
        if explicit_setup_rejection:
            outcome = "medication_list_setup_rejected"
            observation["medication_outcome"] = outcome
        if outcome == "none":
            if re.search(
                r"\b(?:medication|medicine|prescription)\b[^.!?]*"
                r"\b(?:has been|was|is)\s+(?:added|updated)\b",
                text,
            ) or re.search(
                r"\b(?:has been|was|is)\s+(?:added|updated)\b[^.!?]*"
                r"\b(?:medication|medicine|prescription)\b",
                text,
            ):
                outcome = "medication_added"
            elif (
                observation["medication_action"]
                in {"ask_dose_on_file", "provide_medication"}
                and re.search(r"\b(?:has been|was|is)\s+(?:added|updated)\b", text)
            ):
                outcome = "medication_added"
            elif explicit_setup_rejection:
                outcome = "medication_list_setup_rejected"
            observation["medication_outcome"] = outcome

        # Target state and caller action are separate axes. An outcome-only
        # target statement cannot direct the caller to request the refill again.
        if (
            observation["medication_action"] == "request_refill"
            and outcome
            in {
                "refill_unavailable",
                "medication_added",
                "medication_list_setup_rejected",
                "escalation_acknowledged",
                "workflow_blocked",
            }
        ):
            observation["medication_action"] = "none"

    @staticmethod
    def _grounded_target_context(snapshot: Any) -> dict[str, Any]:
        """Expose only clinic-sourced observations needed to resolve references."""
        observations = getattr(snapshot, "target_observations", {}) or {}
        context: dict[str, Any] = {}
        offered = observations.get("offered_locations")
        value = getattr(offered, "value", ()) if offered is not None else ()
        if isinstance(value, (list, tuple)):
            locations = [str(item).strip() for item in value if str(item).strip()]
            if locations:
                context["offered_locations"] = locations
        dose = observations.get("dose")
        dose_value = getattr(dose, "value", "") if dose is not None else ""
        if str(dose_value).strip():
            context["last_observed_dose"] = str(dose_value).strip()
        return context

    @staticmethod
    def _schema(domain: str | None = None) -> dict[str, Any]:
        if domain == "self_pay_location":
            return {
                "type": "object",
                "properties": {
                    "location_action": {
                        "type": "string",
                        "enum": [
                            "asks_insurance",
                            "asks_self_pay_status",
                            "acknowledges_self_pay",
                            "offers_locations",
                            "asks_location_choice",
                            "acknowledges_location",
                            "states_location",
                            "asks_about_switch",
                            "states_office_hours",
                            "asks_office_hours",
                            "finish",
                            "other",
                        ],
                    },
                    "insurer": {"type": "string"},
                    "locations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 4,
                    },
                    "acknowledged_location": {"type": "string"},
                    "states_active_location": {"type": "boolean"},
                    "office_hours": {"type": "string"},
                    "requires_response": {"type": "boolean"},
                },
                "required": [
                    "location_action",
                    "insurer",
                    "locations",
                    "acknowledged_location",
                    "states_active_location",
                    "office_hours",
                    "requires_response",
                ],
                "additionalProperties": False,
            }
        schema = {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [
                        "open_intent",
                        "fact_request",
                        "visit_type_request",
                        "provider_preference_request",
                        "appointment_choice",
                        "reschedule_confirmation",
                        "reschedule_reason_request",
                        "availability_fallback",
                        "presence_check",
                        "acknowledgement",
                        "transaction_confirmation",
                        "other",
                    ],
                },
                "requested_fact": {
                    "type": "string",
                    "enum": [
                        "none",
                        "full_name",
                        "dob",
                        "insurance",
                        "complaint",
                        "symptom_duration",
                        "appointment_type",
                        "provider_preference",
                    ],
                },
                "choice_options": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "new_appointment",
                            "reschedule",
                            "cancel",
                            "keep",
                        ],
                    },
                    "maxItems": 4,
                },
                "remote_dob_claim": {
                    "type": "boolean",
                },
                "remote_existing_appointment_claim": {
                    "type": "boolean",
                },
                "requested_day_change": {
                    "type": "boolean",
                },
                "requested_time_change": {
                    "type": "boolean",
                },
                "requested_provider_change": {
                    "type": "boolean",
                },
                "afternoon_constraint_retained": {
                    "type": "boolean",
                },
                "requires_response": {
                    "type": "boolean",
                },
                "medication_action": {
                    "type": "string",
                    "enum": [
                        "none",
                        "request_refill",
                        "provide_medication",
                        "ask_dose_on_file",
                        "correct_dose",
                        "provide_pharmacy_preference",
                        "confirm_or_correct_target_claim",
                        "handle_unknown_clinical_fact",
                        "clarify",
                        "finish",
                    ],
                },
                "medication_outcome": {
                    "type": "string",
                    "enum": [
                        "none",
                        "refill_unavailable",
                        "medication_list_setup_supported",
                        "medication_list_setup_rejected",
                        "medication_added",
                        "escalation_acknowledged",
                        "workflow_blocked",
                    ],
                },
                "offers_human_escalation": {"type": "boolean"},
                "offers_medication_list_setup": {"type": "boolean"},
                "extracted_target_field": {
                    "type": "string",
                    "enum": [
                        "none",
                        "medication",
                        "dose",
                        "pharmacy",
                        "prescriber",
                        "insurance",
                        "location",
                        "appointment_slot",
                    ],
                },
                "extracted_target_value": {"type": "string"},
                "target_acknowledges_correction": {"type": "boolean"},
                "self_pay_location_action": {
                    "type": "string",
                    "enum": [
                        "none",
                        "establish_self_pay",
                        "provide_self_pay",
                        "ask_locations",
                        "locations_offered",
                        "select_location",
                        "location_acknowledgement",
                        "ask_active_location_hours",
                        "confirm_location",
                        "finish",
                        "clarify",
                    ],
                },
                "extracted_locations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 4,
                },
                "extracted_location": {"type": "string"},
                "target_acknowledged_location": {"type": "string"},
                "target_asserts_active_location": {"type": "boolean"},
                "extracted_insurance_status": {
                    "type": "string",
                    "enum": ["none", "self_pay", "insured", "specific_insurer"],
                },
                "extracted_insurer": {"type": "string"},
                "target_acknowledges_self_pay": {"type": "boolean"},
                "office_hours": {"type": "string"},
                "hours_location": {"type": "string"},
                "hours_context_changed": {"type": "boolean"},
            },
            "required": [
                "kind",
                "requested_fact",
                "choice_options",
                "remote_dob_claim",
                "remote_existing_appointment_claim",
                "requested_day_change",
                "requested_time_change",
                "requested_provider_change",
                "afternoon_constraint_retained",
                "requires_response",
                "medication_action",
                "extracted_target_field",
                "extracted_target_value",
                "target_acknowledges_correction",
                "self_pay_location_action",
                "extracted_locations",
                "extracted_location",
                "target_acknowledged_location",
                "target_asserts_active_location",
                "extracted_insurance_status",
                "extracted_insurer",
                "target_acknowledges_self_pay",
                "office_hours",
                "hours_location",
                "hours_context_changed",
            ],
            "additionalProperties": False,
        }
        domain_fields = {
            "medication": (
                "requires_response",
                "medication_action",
                "medication_outcome",
                "offers_human_escalation",
                "offers_medication_list_setup",
                "extracted_target_field",
                "extracted_target_value",
                "target_acknowledges_correction",
            ),
            "self_pay_location": (
                "requires_response",
                "self_pay_location_action",
                "extracted_locations",
                "extracted_location",
                "target_acknowledged_location",
                "target_asserts_active_location",
                "extracted_insurance_status",
                "extracted_insurer",
                "target_acknowledges_self_pay",
                "office_hours",
                "hours_location",
                "hours_context_changed",
            ),
        }
        fields = domain_fields.get(domain)
        if fields is not None:
            schema["properties"] = {
                key: schema["properties"][key] for key in fields
            }
            schema["required"] = list(fields)
        return schema

    @staticmethod
    def _system_prompt(domain: str | None = None) -> str:
        if domain == "medication":
            return """Interpret only the clinic's complete latest medication-refill turn into the required JSON. Do not write patient dialogue or invent facts.

Choose exactly one medication_action:
- request_refill: general refill intake or open help.
- provide_medication: asks which medication or prescription the caller needs refilled. This action is only about medication identity, never pharmacy identity or pharmacy choice.
- ask_dose_on_file: asks strength, dosage, or milligrams without stating a dose.
- provide_pharmacy_preference: any question asking which pharmacy to use, asking for pharmacy preference, or confirming whether to use a pharmacy already on file. This is mutually exclusive with provide_medication.
- confirm_or_correct_target_claim: the clinic states or asks the caller to verify a medication, dose, or pharmacy value. Any explicit numeric strength/dose claim (including spoken numbers with mg/milligrams) uses extracted_target_field=dose, a normalized value such as "20 mg", and this action—not provide_medication or ask_dose_on_file.
- handle_unknown_clinical_fact: asks for prescriber authorization, prescription number, refill count, or another clinical/authorization fact not owned by the synthetic caller fixture.
- finish: asks whether anything else is needed after the refill workflow.
- clarify: a medication-refill request is genuinely ambiguous.
- none: no medication-refill action fits.

Represent the clinic's outcome separately from what it asks the caller to do:
- refill_unavailable: no medication/prescription is present, active, eligible, or refillable, or the clinic cannot process a refill from this profile.
- medication_list_setup_supported: the clinic says a medication may be added/updated on the demo profile.
- medication_list_setup_rejected: the clinic says it cannot add/update medication information.
- medication_added: the clinic confirms the medication was added/updated.
- escalation_acknowledged: the clinic confirms it will connect/transfer/get staff.
- workflow_blocked: the clinic explicitly says neither it nor another offered route can proceed.
- none: no such target outcome is stated.
Set offers_human_escalation=true for an explicit offer to connect or speak with staff/team/someone. Set offers_medication_list_setup=true for an explicit offer to add/update medication information. These booleans are independent: preserve both in compound turns. A refill-unavailable statement is never request_refill merely because it contains refill vocabulary.

Action and outcome are independent. Always preserve an explicit outcome even when
the same turn asks a question:
- can/may add or update medication information => medication_list_setup_supported
- has added or updated the medication => medication_added
- cannot add or update medication information => medication_list_setup_rejected
An offer or statement that medication information can be added is not itself
provide_medication. Use provide_medication only when the clinic directly asks the
caller to identify the medication.

Extraction is evidence, not an answer slot:
- extracted_target_field identifies a concrete value explicitly stated by the clinic, otherwise none.
- A question such as "which pharmacy" contains no concrete pharmacy value: use extracted_target_field=none and extracted_target_value="".
- A reference such as "pharmacy on file" is a preference question, not a concrete target value.
- Normalize spoken dose values to digits plus mg. grounded_target_context.last_observed_dose may resolve a short numeric dose follow-up.
- target_acknowledges_correction is true only when the clinic accepts/restates the corrected value.
- Empty strings mean absent; never emit the word none in free-text fields.
"""
        if domain == "self_pay_location":
            return """Interpret only the clinic's latest self-pay/location utterance. Qwen perceives literal clinic speech; Python owns authoritative location state, selection, switching, pronoun resolution, and oracles.

Choose exactly one location_action:
- asks_insurance: asks what insurance, coverage, carrier, or provider the caller has.
- asks_self_pay_status: asks the yes/no question whether the caller is self-pay, uninsured, paying personally, or out of pocket.
- acknowledges_self_pay: explicitly accepts or confirms that the caller will be self-pay.
- offers_locations: explicitly presents one or more named offices as choices.
- asks_location_choice: asks which office/location the caller wants without presenting named choices.
- acknowledges_location: explicitly accepts/proceeds with a named selected office.
- states_location: states a named office, its availability, or which named office is active, without offering a new choice.
- asks_about_switch: asks whether the caller wants a different named/grounded office.
- states_office_hours: supplies office hours or a closing time, with either a named office or a contextual phrase such as "that office".
- asks_office_hours: asks about hours/closing time.
- finish: asks whether anything else is needed after this workflow.
- other: none fits.

Boundary rules:
- An offer to transfer, connect, or hand the caller to an office/person is other. It is not asks_location_choice merely because the word office appears.
- A request that the caller repeat or clarify which location they previously meant is other. asks_location_choice is only a new request to choose among available offices.

Extraction rules:
- insurer is a literal insurer name in the latest utterance, otherwise "". Never emit "unknown" or copy context.
- locations contains only literal office/location names in the latest utterance. Never copy grounded_target_context.
- A contextual pronoun such as it, there, that office, or that location never licenses copying names from grounded_target_context into locations.
- acknowledged_location is a literal named office the clinic explicitly accepts/proceeds with, otherwise "".
- states_active_location is true only when the clinic explicitly calls a named office active/current/selected.
- office_hours preserves the literal hours phrase supplied by the clinic, otherwise "". For questions, it is empty.
- "that office" has no extracted location. Python resolves it from committed active state.
"""
        return """Classify the clinic's COMPLETE latest scheduling turn into the required JSON. Do not write patient dialogue or invent patient facts/slots.

kind:
open_intent = general "how can I help / tell me your request";
fact_request = directly asks name/DOB/insurance/complaint/duration;
visit_type_request = asks appointment type;
provider_preference_request = asks provider preference;
appointment_choice = asks new vs reschedule/change vs cancel/keep;
reschedule_confirmation = asks whether a stated appointment is the one to reschedule;
reschedule_reason_request = asks WHY it must be rescheduled;
availability_fallback = requested availability failed and clinic asks what day/time/provider may change;
presence_check = asks if caller is there;
acknowledgement = no response needed;
transaction_confirmation = says transaction completed;
other = none fit.

Rules:
- Read the ENTIRE turn, not only its final clause.
- "How can I assist...let me know your request" => open_intent, requested_fact=none.
- "new appointment, reschedule, or cancel" => appointment_choice and include ALL explicit choice_options.
- "Is this the appointment you want to reschedule?" => reschedule_confirmation.
- A stated DOB followed by open-intent => open_intent + remote_dob_claim=true.
- requested_fact is non-none only when directly requested.
- If another day is proposed while afternoon stays fixed: requested_day_change=true and afternoon_constraint_retained=true.
- requires_response=true for questions/requests/choices/yes-no confirmations/fallback offers.

Medication-refill extension:
- medication_action describes the grounded caller action requested by this turn.
- General refill intake/open help => request_refill.
- Asking which medication/prescription, including what medication the caller is calling about => provide_medication.
- Asking strength/dosage/how many milligrams, without stating a dose => ask_dose_on_file.
- Asking which pharmacy or whether to use the pharmacy on file => provide_pharmacy_preference.
- A target-stated medication/dose/pharmacy value => confirm_or_correct_target_claim.
- A short follow-up asking whether the caller means a number is a dose claim when grounded_target_context contains last_observed_dose; normalize that spoken number as the new dose.
- Any direct pharmacy question => provide_pharmacy_preference, never provide_medication.
- A later statement that a dose is still listed => confirm_or_correct_target_claim even if the unit is omitted and prior dose context resolves it.
- "Anything else?" after the refill request => finish.
- Extract a target value only when the clinic explicitly says it. Never infer one.
- Normalize spoken dose words to digits plus mg (for example "twenty milligrams" => "20 mg").
- target_acknowledges_correction=true only when the clinic accepts/restates the corrected value.

Self-pay/location extension:
- Asking insurance/provider => provide_self_pay. Open help for this scenario => establish_self_pay.
- Asking whether the caller pays out of pocket => establish_self_pay.
- A clinic claim naming an insurer => provide_self_pay and extract the exact insurer name with specific_insurer status.
- Acknowledging out-of-pocket/self-pay => ask_locations and target_acknowledges_self_pay=true.
- Asking which office the caller wants, with no options => ask_locations.
- Explicitly offering one or more offices => locations_offered and extract every exact location name.
- A follow-up asking which/what one is preferred after grounded_target_context lists offered_locations => locations_offered. Do not copy context locations into extraction fields; they were not spoken in the latest turn.
- Accepting/using a caller-selected office => location_acknowledgement and copy the exact grounded name to target_acknowledged_location.
- target_asserts_active_location=true only when the target says an explicit office is the currently selected/active office, or uses it to answer a pending "that office" question. Historical mentions are false.
- A statement/question about an explicit office => extracted_location is that exact name.
- A statement giving office hours, with either an explicit office or a contextual pronoun, => confirm_location.
- A question asking the closing time of the active/that office => ask_active_location_hours.
- A statement that another named office remains available, without offering it as a new choice, => confirm_location.
- "Anything else" after the location workflow => finish.
- Extract office_hours exactly as spoken; set hours_location only when the office is explicit.
- Preserve the exact words of an hours value (for example "closes at five"); do not convert it to clock notation.
- For "that office/location", leave hours_location empty; Python owns active context.
- hours_context_changed=true only for an explicit different day/date/schedule context.
- Never invent a location, insurer, or hours value. Empty strings/arrays mean absent; never put the word "none" in a free-text extraction field.
"""

    @staticmethod
    def _normalize_observation(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise TypeError(
                "Qwen V3 semantic backend returned non-object JSON."
            )

        if "location_action" in raw:
            semantic = str(raw.get("location_action", "other")).strip()
            action_map = {
                "asks_insurance": "provide_self_pay",
                "asks_self_pay_status": "establish_self_pay",
                "acknowledges_self_pay": "ask_locations",
                "offers_locations": "locations_offered",
                "asks_location_choice": "locations_offered",
                "acknowledges_location": "location_acknowledgement",
                "states_location": "confirm_location",
                "asks_about_switch": "locations_offered",
                "states_office_hours": "confirm_location",
                "asks_office_hours": "ask_active_location_hours",
                "finish": "finish",
                "other": "none",
            }
            locations = tuple(
                str(item).strip()
                for item in (raw.get("locations") or ())
                if str(item).strip()
                and str(item).strip().casefold() not in {"none", "unknown"}
                and str(item).strip().casefold()
                not in {"that office", "that location", "this office", "this location"}
            )
            insurer = str(raw.get("insurer", "")).strip()
            if insurer.casefold() in {"none", "unknown"}:
                insurer = ""
            acknowledged = str(raw.get("acknowledged_location", "")).strip()
            if acknowledged.casefold() in {"none", "unknown"}:
                acknowledged = ""
            hours = str(raw.get("office_hours", "")).strip()
            if hours.casefold() in {"none", "unknown"}:
                hours = ""
            explicit_location = locations[0] if len(locations) == 1 else ""
            return {
                "kind": "other",
                "requested_fact": "none",
                "choice_options": (),
                "remote_dob_claim": False,
                "remote_existing_appointment_claim": False,
                "requested_day_change": False,
                "requested_time_change": False,
                "requested_provider_change": False,
                "afternoon_constraint_retained": False,
                "requires_response": bool(raw.get("requires_response", True)),
                "medication_action": "none",
                "extracted_target_field": "none",
                "extracted_target_value": "",
                "target_acknowledges_correction": False,
                "self_pay_location_action": action_map.get(semantic, "none"),
                "extracted_locations": locations,
                "extracted_location": explicit_location,
                "target_acknowledged_location": acknowledged,
                "target_asserts_active_location": bool(
                    raw.get("states_active_location", False)
                ),
                "extracted_insurance_status": (
                    "self_pay"
                    if semantic == "acknowledges_self_pay"
                    else ("specific_insurer" if insurer else "none")
                ),
                "extracted_insurer": insurer,
                "target_acknowledges_self_pay": semantic
                == "acknowledges_self_pay",
                "office_hours": hours,
                "hours_location": (
                    explicit_location
                    if semantic == "states_office_hours"
                    else ""
                ),
                "hours_context_changed": False,
            }

        def free_text(key: str) -> str:
            value = str(raw.get(key, "")).strip()
            return "" if value.casefold() in {"none", "unknown"} else value

        observation = {
            "kind": str(raw.get("kind", "other")).strip(),
            "requested_fact": str(
                raw.get("requested_fact", "none")
            ).strip(),
            "choice_options": tuple(
                str(x).strip()
                for x in (raw.get("choice_options") or [])
                if str(x).strip()
            ),
            "remote_dob_claim": bool(
                raw.get("remote_dob_claim", False)
            ),
            "remote_existing_appointment_claim": bool(
                raw.get(
                    "remote_existing_appointment_claim",
                    False,
                )
            ),
            "requested_day_change": bool(
                raw.get("requested_day_change", False)
            ),
            "requested_time_change": bool(
                raw.get("requested_time_change", False)
            ),
            "requested_provider_change": bool(
                raw.get("requested_provider_change", False)
            ),
            "afternoon_constraint_retained": bool(
                raw.get(
                    "afternoon_constraint_retained",
                    False,
                )
            ),
            "requires_response": bool(
                raw.get("requires_response", True)
            ),
            "medication_action": str(
                raw.get("medication_action", "none")
            ).strip(),
            "medication_outcome": str(
                raw.get("medication_outcome", "none")
            ).strip(),
            "offers_human_escalation": bool(
                raw.get("offers_human_escalation", False)
            ),
            "offers_medication_list_setup": bool(
                raw.get("offers_medication_list_setup", False)
            ),
            "extracted_target_field": str(
                raw.get("extracted_target_field", "none")
            ).strip(),
            "extracted_target_value": str(
                raw.get("extracted_target_value", "")
            ).strip(),
            "target_acknowledges_correction": bool(
                raw.get("target_acknowledges_correction", False)
            ),
            "self_pay_location_action": str(
                raw.get("self_pay_location_action", "none")
            ).strip(),
            "extracted_locations": tuple(
                str(item).strip()
                for item in (raw.get("extracted_locations") or [])
                if str(item).strip() and str(item).strip().casefold() != "none"
            ),
            "extracted_location": free_text("extracted_location"),
            "target_acknowledged_location": free_text(
                "target_acknowledged_location"
            ),
            "target_asserts_active_location": bool(
                raw.get("target_asserts_active_location", False)
            ),
            "extracted_insurance_status": str(
                raw.get("extracted_insurance_status", "none")
            ).strip(),
            "extracted_insurer": free_text("extracted_insurer"),
            "target_acknowledges_self_pay": bool(
                raw.get("target_acknowledges_self_pay", False)
            ),
            "office_hours": free_text("office_hours"),
            "hours_location": free_text("hours_location"),
            "hours_context_changed": bool(raw.get("hours_context_changed", False)),
        }
        return observation

    def _state_objective(
        self,
        *,
        reason: str,
    ) -> PolicyDecision:
        f = self.facts
        return PolicyDecision(
            DecisionKind.STATE_OBJECTIVE,
            text=(
                f"I need to schedule a {f.appointment_type} "
                f"for {f.preferred_day} {f.preferred_time}."
            ),
            reason=reason,
        )

    def _decision_for_observation(
        self,
        *,
        observation: dict[str, Any],
        snapshot: Any,
    ) -> PolicyDecision:
        f = self.facts
        kind = observation["kind"]
        requested_fact = observation["requested_fact"]
        options = set(observation["choice_options"])

        if kind == "transaction_confirmation":
            return PolicyDecision(
                DecisionKind.WAIT,
                reason="qwen_v3_remote_transaction_confirmation",
            )

        if kind == "acknowledgement":
            return PolicyDecision(
                DecisionKind.WAIT,
                reason="qwen_v3_acknowledgement",
            )

        if kind == "presence_check":
            return self._state_objective(
                reason="qwen_v3_presence_recovery",
            )

        if kind == "reschedule_confirmation":
            return PolicyDecision(
                DecisionKind.GRANT_PERMISSION,
                text="Yes, that's the appointment I want to reschedule.",
                reason="qwen_v3_reschedule_confirmation",
            )

        if kind == "reschedule_reason_request":
            return PolicyDecision(
                DecisionKind.ANSWER_COMPLAINT,
                text=f"I have {f.complaint}.",
                reason="qwen_v3_reschedule_reason_requested",
            )

        fact_answers = {
            "full_name": (
                DecisionKind.ANSWER_FACT,
                f"{f.first_name} {f.last_name}.",
            ),
            "dob": (
                DecisionKind.ANSWER_FACT,
                f"{f.dob}.",
            ),
            "insurance": (
                DecisionKind.ANSWER_FACT,
                f"{f.insurance}.",
            ),
            "complaint": (
                DecisionKind.ANSWER_COMPLAINT,
                f"I have {f.complaint}.",
            ),
            "symptom_duration": (
                DecisionKind.ANSWER_FACT,
                f"{f.symptom_duration}.",
            ),
            "appointment_type": (
                DecisionKind.ANSWER_APPOINTMENT_TYPE,
                f"A {f.appointment_type}.",
            ),
            "provider_preference": (
                DecisionKind.ANSWER_PROVIDER_PREFERENCE,
                "First available is fine.",
            ),
        }

        if kind == "fact_request" and requested_fact in fact_answers:
            decision_kind, text = fact_answers[requested_fact]
            return PolicyDecision(
                decision_kind,
                text=text,
                reason=f"qwen_v3_{requested_fact}_requested",
            )

        if kind == "visit_type_request":
            return PolicyDecision(
                DecisionKind.ANSWER_APPOINTMENT_TYPE,
                text=f"A {f.appointment_type}.",
                reason="qwen_v3_appointment_type_requested",
            )

        if kind == "provider_preference_request":
            return PolicyDecision(
                DecisionKind.ANSWER_PROVIDER_PREFERENCE,
                text="First available is fine.",
                reason="qwen_v3_provider_preference_requested",
            )

        if kind == "open_intent":
            if observation["remote_dob_claim"]:
                combined_kind = _enum_or_default(
                    "CORRECT_AND_STATE_OBJECTIVE",
                    DecisionKind.STATE_OBJECTIVE,
                )
                return PolicyDecision(
                    combined_kind,
                    text=(
                        f"Actually, my date of birth is {f.dob}. "
                        f"I need to schedule a {f.appointment_type} "
                        f"for {f.preferred_day} {f.preferred_time}."
                    ),
                    reason="qwen_v3_correct_dob_then_state_objective",
                )

            return self._state_objective(
                reason="qwen_v3_open_intent",
            )

        if kind == "appointment_choice":
            if observation["remote_existing_appointment_claim"]:
                return PolicyDecision(
                    DecisionKind.STATE_OBJECTIVE,
                    text=(
                        "I'd like to reschedule it. "
                        f"I'm looking for {f.preferred_day} "
                        f"{f.preferred_time}."
                    ),
                    reason="qwen_v3_existing_appointment_reschedule",
                )

            if "new_appointment" in options:
                return PolicyDecision(
                    DecisionKind.STATE_OBJECTIVE,
                    text="I'd like to make a new appointment.",
                    reason="qwen_v3_new_appointment_choice",
                )

            if "reschedule" in options:
                return PolicyDecision(
                    DecisionKind.STATE_OBJECTIVE,
                    text=(
                        "I'd like to reschedule my existing appointment. "
                        f"I'm looking for {f.preferred_day} "
                        f"{f.preferred_time}."
                    ),
                    reason="qwen_v3_reschedule_choice",
                )

        if kind == "availability_fallback":
            if (
                observation["requested_day_change"]
                and observation["afternoon_constraint_retained"]
            ):
                return PolicyDecision(
                    DecisionKind.SEARCH_ALTERNATE_DAY_AFTERNOON,
                    text="Please check another weekday afternoon.",
                    reason="qwen_v3_alternate_day_afternoon",
                )

            if observation["requested_day_change"]:
                return PolicyDecision(
                    DecisionKind.SEARCH_ALTERNATE_DAY_AFTERNOON,
                    text="Please check another weekday afternoon.",
                    reason="qwen_v3_alternate_day_search",
                )

        if not observation["requires_response"]:
            return PolicyDecision(
                DecisionKind.WAIT,
                reason="qwen_v3_no_response_required",
            )

        # Do not re-enter the old V31 "please rephrase" loop. The model has
        # already determined a response is required but no specialized branch
        # owns it, so recover by restating the grounded mission.
        return self._state_objective(
            reason="qwen_v3_mission_recovery",
        )
