"""Call #5: grounded doctor/specialist directory assessment.

The model is perception-only.  Python owns milestones, caller identity,
grounding, contextual references, and every oracle.
"""
from __future__ import annotations

import re
import json
import asyncio
import os
import threading
from difflib import SequenceMatcher
from dataclasses import asdict
from typing import Any

from .flow_state import FlowSnapshot, SchedulingFlowTracker
from .models import DecisionKind, PolicyDecision
from .oracle_evidence import OracleEvidence
from voiceprobe.v33.reasoner import OllamaBackend, OllamaConfig

SCENARIO_ID = "doctor-specialist-directory"
FULL_NAME = "Gyeong-hyeon Gwak"
FIRST_NAME = "Gyeong-hyeon"
LAST_NAME = "Gwak"
DEFAULT_DOCTOR_QWEN_TIMEOUT_SECONDS = 6.0
DEFAULT_DOCTOR_QWEN_STARTUP_TIMEOUT_SECONDS = 20.0


class DoctorDirectoryQwenRouter:
    """Bounded specialist-directory perception; receives no caller truth."""
    def __init__(self, *, backend: Any | None = None) -> None:
        self.endpoint = os.environ.get("VOICEPROBE_OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
        self.model = os.environ.get("VOICEPROBE_REASONING_V2_EDGE_MODEL", "qwen3.5:4b")
        self.timeout_seconds = float(os.environ.get(
            "VOICEPROBE_V3_QWEN_TIMEOUT_SECONDS",
            str(DEFAULT_DOCTOR_QWEN_TIMEOUT_SECONDS),
        ))
        self.startup_timeout_seconds = float(os.environ.get(
            "VOICEPROBE_V3_QWEN_STARTUP_TIMEOUT_SECONDS",
            str(DEFAULT_DOCTOR_QWEN_STARTUP_TIMEOUT_SECONDS),
        ))
        self._backend = backend or OllamaBackend(OllamaConfig(
            endpoint=self.endpoint,
            model=self.model,
            timeout_seconds=self.timeout_seconds,
            keep_alive="15m", num_ctx=1024, temperature=0.0))
        self.last_observation: dict[str, Any] = {}
        self.last_raw_observation: dict[str, Any] = {}
        if backend is None:
            self.warmup_sync()

    def _startup_backend(self) -> OllamaBackend:
        return OllamaBackend(OllamaConfig(
            endpoint=self.endpoint,
            model=self.model,
            timeout_seconds=self.startup_timeout_seconds,
            keep_alive="15m", num_ctx=1024, temperature=0.0))

    async def warmup(self) -> None:
        await self._startup_backend().generate_json(
            system="Return the required JSON object only.",
            prompt='{"warmup":true}',
            schema={"type": "object", "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"], "additionalProperties": False},
        )

    def warmup_sync(self) -> None:
        """Load and verify Qwen while bridge construction is still pre-dial."""
        errors: list[BaseException] = []

        def runner() -> None:
            try:
                asyncio.run(self.warmup())
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=runner, name="voiceprobe-doctor-qwen-warmup", daemon=True)
        thread.start()
        thread.join()
        if errors:
            raise errors[0]

    async def resolve(self, target_turn: str, snapshot: FlowSnapshot) -> PolicyDecision:
        context = {
            "offered_doctors": [d for d in snapshot.target_observations.get("offered_doctors", ()).value]
            if "offered_doctors" in snapshot.target_observations else [],
            "selected_doctor": (
                snapshot.committed_dialogue["active_doctor"].value
                if "active_doctor" in snapshot.committed_dialogue
                else snapshot.committed_dialogue["selected_doctor_initial"].value
                if "selected_doctor_initial" in snapshot.committed_dialogue else ""
            ),
            "active_location": snapshot.committed_dialogue.get("active_doctor_location").value
            if "active_doctor_location" in snapshot.committed_dialogue else "",
        }
        raw = await self._backend.generate_json(system=self.system_prompt(), prompt=json.dumps(
            {"clinic_turn": " ".join(target_turn.split()), "grounded_target_context": context},
            ensure_ascii=False), schema=self.schema())
        if not isinstance(raw, dict):
            raise TypeError("Doctor-directory Qwen output must be an object")
        self.last_raw_observation = dict(raw)
        self.last_observation = self.normalize(raw)
        self.last_observation["doctor_action"] = self._calibrate_action(target_turn, self.last_observation)
        return PolicyDecision(DecisionKind.WAIT, reason="doctor_directory:qwen_perception_only")

    @staticmethod
    def _calibrate_action(text: str, observation: dict[str, Any]) -> str:
        """Conservative literal boundary rules for recurring Qwen confusions."""
        t = " ".join(text.casefold().split())
        if re.search(r"\b(?:how may i help|how can i help)\b", t): return "other"
        if re.search(r"\bprofile name\b|\bspelled\b", t): return "reports_profile_name"
        if re.search(r"\bprofile\b[^.!?]*\b(?:registered|created|ready|set up)\b", t): return "profile_registered"
        if re.search(r"\b(?:female|male|nonbinary)\s+(?:doctor|physician)\b|\bis (?:explicitly )?(?:female|male|nonbinary)\b", t): return "states_gender"
        if re.search(r"\b(?:more than one|multiple)\s+(?:doctor|physician)|\b(?:only|just)\b[^.!?]{0,24}\bone doctor\b", t): return "states_multiple_doctor_capability"
        if re.search(r"\b(?:why|reason)\b[^.!?]*\b(?:switch|different doctor|change doctor)\b", t): return "asks_switch_reason"
        if re.search(r"\b(?:switched|changed|updated)\b[^.!?]*\b(?:doctor|physician|provider)\b|\b(?:doctor|dr\.?)\s+[^.!?]+\s+is now (?:your|the)\b", t): return "switch_acknowledged"
        if re.search(r"\b(?:cannot|can't|unable|not able)\b[^.!?]*\b(?:switch|change)\b", t): return "switch_denied"
        if re.search(r"\b(?:hours?|from\s+\w+\s+to\s+\w+|nine to (?:four|five))\b", t): return "states_hours"
        if re.search(r"\b(?:specializes? in|specialty is)\b", t): return "states_specialty"
        if re.search(r"\bworks? at\b", t): return "states_location"
        if re.search(r"\b(?:available|we have|we offer)\b", t) and re.search(r"\bdr\.?\b|\bdoctor\b", t): return "offers_doctors"
        return str(observation.get("doctor_action", "other"))

    @staticmethod
    def schema() -> dict[str, Any]:
        p = {
            "requires_response": {"type": "boolean"},
            "doctor_action": {"type": "string", "enum": ["profile_registered", "reports_profile_name", "offers_doctors", "states_specialty", "states_gender", "gender_unavailable", "states_location", "states_hours", "states_multiple_doctor_capability", "asks_switch_reason", "switch_acknowledged", "switch_denied", "finish", "other"]},
            "reported_profile_name": {"type": "string"}, "reported_profile_spelling": {"type": "string"},
            "doctors": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {"name": {"type": "string"}, "specialty": {"type": "string"}}, "required": ["name", "specialty"], "additionalProperties": False}},
            "doctor_name": {"type": "string"}, "specialty": {"type": "string"},
            "explicit_gender": {"type": "string", "enum": ["", "male", "female", "nonbinary", "unavailable"]},
            "locations": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
            "hours": {"type": "string"}, "hours_location": {"type": "string"}, "day": {"type": "string"},
            "multiple_doctors_capability": {"type": "string", "enum": ["", "yes", "no", "unavailable"]},
        }
        return {"type": "object", "properties": p, "required": list(p), "additionalProperties": False}

    @staticmethod
    def normalize(raw: dict[str, Any]) -> dict[str, Any]:
        clean = dict(raw)
        for key in ("reported_profile_name", "reported_profile_spelling", "doctor_name", "specialty", "hours", "hours_location", "day"):
            value = " ".join(str(clean.get(key, "")).split())
            clean[key] = "" if value.casefold() in {"none", "unknown", "n/a"} else value
        clean["doctors"] = tuple({"name": " ".join(str(x.get("name", "")).split()), "specialty": " ".join(str(x.get("specialty", "")).split())} for x in clean.get("doctors", ()) if isinstance(x, dict) and str(x.get("name", "")).strip())
        clean["locations"] = tuple(" ".join(str(x).split()) for x in clean.get("locations", ()) if str(x).strip() and str(x).casefold() not in {"that location", "that office", "unknown"})
        gender = str(clean.get("explicit_gender", "")).casefold()
        clean["explicit_gender"] = gender if gender in {"male", "female", "nonbinary", "unavailable"} else ""
        capability = str(clean.get("multiple_doctors_capability", "")).casefold()
        clean["multiple_doctors_capability"] = capability if capability in {"yes", "no", "unavailable"} else ""
        return clean

    @staticmethod
    def system_prompt() -> str:
        return """Interpret only the clinic's complete latest doctor-directory turn. Return the schema JSON only. Never invent a doctor, specialty, gender, location, hours, profile name, or spelling; extraction fields contain only literal words from clinic_turn. grounded_target_context may resolve 'that doctor/location' but must never be copied into extraction fields.
Actions: profile_registered only for an explicit created/registered/ready profile acknowledgement; reports_profile_name when the clinic repeats or spells the registered patient name; offers_doctors for named available doctors/specialists; states_specialty for an explicit doctor specialty; states_gender only for explicit male/female/nonbinary information; gender_unavailable when the clinic says gender is unavailable; states_location for one or more explicit doctor workplaces; states_hours for explicit working hours; states_multiple_doctor_capability for an explicit yes/no about seeing multiple doctors; asks_switch_reason when the clinic asks why the caller wants to switch; switch_acknowledged only when the clinic establishes the requested doctor as active; switch_denied for an explicit refusal; finish for anything-else closure; otherwise other.
reported_profile_name is the repeated name. reported_profile_spelling preserves the clinic's spelled letters as one string. Pronouns never establish gender. doctors lists only explicitly offered names and any specialty stated in the same turn. doctor_name is nonempty only when explicitly spoken in this turn. locations contains every explicitly named location; multiple legitimate locations are retained. hours preserves the literal hours claim. hours_location is nonempty only if explicitly spoken. day is the explicit day or empty. multiple_doctors_capability is yes/no only for a literal capability statement. Empty strings/arrays mean absent."""


def normalize_person_name(value: str) -> str:
    """Ignore case/spacing/punctuation while retaining letter identity."""
    return "".join(ch for ch in value.casefold() if ch.isalpha())


def registered_name_materially_incompatible(value: str) -> bool:
    candidate = normalize_person_name(value)
    truth = normalize_person_name(FULL_NAME)
    return bool(candidate) and candidate != truth


def explicitly_acknowledges_profile_created(value: str) -> bool:
    """Recognize only unambiguous successful profile-registration claims."""
    text = " ".join(value.casefold().split())
    return any(
        re.search(pattern, text)
        for pattern in (
            r"\b(?:your\s+)?(?:demo\s+)?(?:patient\s+)?profile\s+(?:has been|was|is|'s)\s+(?:successfully\s+)?(?:created|registered|set up|ready|all set)\b",
            r"\b(?:i|we)(?:'ve| have)?\s+(?:successfully\s+)?(?:created|registered|set up)\s+(?:your\s+)?(?:demo\s+)?(?:patient\s+)?profile\b",
        )
    )


def target_addressed_name(value: str) -> str | None:
    """Extract a direct salutation as supporting, non-authoritative evidence."""
    match = re.search(
        r"\bthanks(?:\s+you)?\s*,\s*([A-Za-z][A-Za-z'\-]*)\b",
        value,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def normalize_claim(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def same_grounded_doctor(left: str, right: str) -> bool:
    """Tolerate ASR variants without merging substantively different providers."""
    a, b = normalize_person_name(left), normalize_person_name(right)
    if not a or not b:
        return False
    if a == b:
        return True
    aw, bw = re.findall(r"[a-z]+", left.casefold()), re.findall(r"[a-z]+", right.casefold())
    surname_close = bool(aw and bw and SequenceMatcher(None, aw[-1], bw[-1]).ratio() >= .75)
    return surname_close and SequenceMatcher(None, a, b).ratio() >= .60


class DoctorSpecialistDirectoryScenario:
    scenario_id = SCENARIO_ID

    def __init__(self, *, tracker: SchedulingFlowTracker, qwen: Any) -> None:
        self.tracker, self.qwen = tracker, qwen
        for key, value in (("full_name", FULL_NAME), ("first_name", FIRST_NAME), ("last_name", LAST_NAME)):
            tracker.establish_caller_truth(key, value, evidence=f"{SCENARIO_ID} synthetic caller fixture")
        self.turn = 0
        self.profile_registered = False
        self.name_probe_spoken = False
        self.name_verified = False
        self.offered_doctors: list[dict[str, str]] = []
        self.selected_doctor: str | None = None  # compatibility: initial selection
        self.selected_doctor_initial: str | None = None
        self.initial_doctor_selection_acknowledged = False
        self.active_doctor: str | None = None
        self.switch_requested_doctor: str | None = None
        self.switch_reason_spoken = False
        self.switch_acknowledged = False
        self.multiple_doctors_capability_observed: str | None = None
        self.specialty: str | None = None
        self.gender_claims: list[tuple[str, str, str]] = []
        self.gender_probe_complete: set[str] = set()
        self.locations: list[str] = []
        self.selected_location: str | None = None
        self.hours_claims: list[dict[str, str | int]] = []
        self.context_followup = False
        self.post_switch_specialty: str | None = None
        self.post_switch_locations: list[str] = []
        self.post_switch_hours: list[dict[str, str | int]] = []
        self.final_reported_active_doctor: str | None = None
        self.last_spoken_reason: str | None = None
        self.evidence: list[OracleEvidence] = []
        self.rejected_extractions: list[dict[str, object]] = []
        self.semantic_failures: list[dict[str, str]] = []

    async def resolve(self, target_turn: str, snapshot: FlowSnapshot) -> PolicyDecision:
        try:
            await self.qwen.resolve(target_turn, snapshot)
        except Exception as exc:
            self.semantic_failures.append({
                "event": "doctor_directory_semantic_failure",
                "kind": (
                    "timeout" if isinstance(exc, TimeoutError)
                    else "connection" if isinstance(exc, (OSError, ConnectionError))
                    else "backend"
                ),
                "error": f"{type(exc).__name__}: {exc}",
            })
            return PolicyDecision(
                DecisionKind.CLARIFY,
                text="Could you please repeat that question?",
                reason="doctor_directory:semantic_failure",
            )
        self.turn += 1
        o = self.qwen.last_observation
        action = str(o.get("doctor_action", "other"))
        profile_created_acknowledged = explicitly_acknowledges_profile_created(target_turn)
        if profile_created_acknowledged:
            # This transition is grounded directly in literal target language;
            # it must not depend on Qwen choosing the matching action label.
            action = "profile_registered"
            self.tracker.observe_target_value(
                "profile_created_acknowledged", True, evidence=target_turn
            )
        if action == "profile_registered":
            self.profile_registered = True
            self.tracker.observe_target_value("profile_registered", True, evidence=target_turn)

        addressed_name = target_addressed_name(target_turn)
        if addressed_name:
            self.tracker.observe_target_value(
                "reported_profile_name_candidate", addressed_name, evidence=target_turn
            )

        reported = self._literal(target_turn, o.get("reported_profile_name", ""), "reported_profile_name")
        spelling = self._literal(target_turn, o.get("reported_profile_spelling", ""), "reported_profile_spelling")
        if reported or spelling:
            value = spelling or reported or ""
            self.tracker.observe_target_value("reported_profile_name", reported or "", evidence=target_turn)
            self.tracker.observe_target_value("reported_profile_spelling", spelling or "", evidence=target_turn)
            self.name_verified = True
            if registered_name_materially_incompatible(value):
                self._oracle("profile_name_registration_mismatch", "full_name", FULL_NAME, value,
                             (f"caller truth: {FULL_NAME}", self._turn(target_turn)),
                             "Target-reported registered name is materially incompatible after punctuation-insensitive normalization.")

        doctors = []
        for raw in o.get("doctors", ()):
            name = self._literal(target_turn, raw.get("name", ""), "doctor")
            specialty = self._literal(target_turn, raw.get("specialty", ""), "specialty")
            if name:
                item = {"name": name, "specialty": specialty or ""}
                if not any(d["name"].casefold() == name.casefold() for d in self.offered_doctors):
                    self.offered_doctors.append(item)
                doctors.append(item)
        if doctors:
            self.tracker.observe_target_value("offered_doctors", tuple(self.offered_doctors), evidence=target_turn)

        claimed_doctor = self._literal(target_turn, o.get("doctor_name", ""), "doctor_name")
        active = self.active_doctor or self.selected_doctor
        if self.selected_doctor_initial and not self.initial_doctor_selection_acknowledged and action in {
            "states_specialty", "states_gender", "gender_unavailable", "states_location", "states_hours"
        } and (not claimed_doctor or same_grounded_doctor(claimed_doctor, self.selected_doctor_initial)):
            self.initial_doctor_selection_acknowledged = True
            self.active_doctor = self.selected_doctor_initial
            active = self.active_doctor
            self.tracker.commit_dialogue_value("active_doctor", active, evidence=target_turn)
        if self.last_spoken_reason == "doctor_directory:final_active_doctor_check" and claimed_doctor:
            self.final_reported_active_doctor = claimed_doctor
            if active and not same_grounded_doctor(claimed_doctor, active):
                self._oracle("doctor_switch_retention_failure", "active_doctor", active, claimed_doctor,
                             (f"committed doctor: {active}", self._turn(target_turn)),
                             "Final active-doctor report reverted to an incompatible provider.")
        if active and claimed_doctor and not same_grounded_doctor(claimed_doctor, active) and action in {"states_specialty", "states_location", "states_hours"}:
            oracle = "doctor_switch_retention_failure" if self.switch_acknowledged else "doctor_identity_context_mismatch"
            self._oracle(oracle, "active_doctor", active, claimed_doctor,
                         (f"committed doctor: {active}", self._turn(target_turn)),
                         "A contextual question about the active doctor was answered using another doctor without explanation.")

        specialty = self._literal(target_turn, o.get("specialty", ""), "specialty")
        if active and specialty:
            prior = self.specialty
            self.tracker.observe_target_value("doctor_specialty", specialty, evidence=target_turn)
            if prior and not self.switch_acknowledged and normalize_person_name(prior) != normalize_person_name(specialty):
                self._oracle("specialist_identity_or_specialty_inconsistency", "specialty", prior, specialty,
                             (f"prior target specialty: {prior}", self._turn(target_turn)),
                             "Target explicitly supplied incompatible specialties for the same selected doctor.")
            self.specialty = specialty
            if self.switch_acknowledged:
                self.post_switch_specialty = specialty

        gender = str(o.get("explicit_gender", "")).strip().casefold()
        if gender in {"male", "female", "nonbinary"} and active and (not claimed_doctor or same_grounded_doctor(claimed_doctor, active)):
            doctor_key = normalize_person_name(active)
            self.gender_claims.append((active, gender, target_turn))
            self.gender_probe_complete.add(doctor_key)
            self.tracker.observe_target_value("doctor_gender", gender, evidence=target_turn)
            previous = next((g for d, g, _ in self.gender_claims[:-1] if same_grounded_doctor(d, active) and g != gender), None)
            if previous:
                self._oracle("doctor_gender_explicit_contradiction", "gender", previous, gender,
                             (next(t for d, g, t in self.gender_claims if same_grounded_doctor(d, active) and g == previous), self._turn(target_turn)),
                             "Two explicit incompatible target gender claims exist; pronouns are not evidence.")

        for raw in o.get("locations", ()) if action in {"states_location", "states_hours"} else ():
            location = self._literal(target_turn, raw, "location")
            if location and location not in self.locations:
                self.locations.append(location)
        if self.locations:
            self.tracker.observe_target_value("doctor_locations", tuple(self.locations), evidence=target_turn)
            if self.selected_location is None:
                self.selected_location = self.locations[0]
                self.tracker.commit_dialogue_value("active_doctor_location", self.selected_location, evidence=target_turn)
            if self.switch_acknowledged:
                self.post_switch_locations = list(self.locations)

        hours = self._literal(target_turn, o.get("hours", ""), "hours")
        day = str(o.get("day", "")).strip().casefold()
        hours_location = self._literal(target_turn, o.get("hours_location", ""), "hours_location") or self.selected_location
        if hours and active and hours_location:
            claim = {"doctor": active, "location": hours_location, "day": day, "hours": hours, "source_turn": self.turn}
            for prior in self.hours_claims:
                same = (str(prior["doctor"]).casefold(), str(prior["location"]).casefold(), prior["day"]) == (active.casefold(), hours_location.casefold(), day)
                if same and normalize_claim(str(prior["hours"])) != normalize_claim(hours):
                    self._oracle("doctor_hours_internal_contradiction", "doctor_location_day_hours", prior["hours"], hours,
                                 (f"turn={prior['source_turn']}; {prior['hours']}", self._turn(target_turn)),
                                 "Incompatible hours were explicitly claimed for the same doctor, location, and day.")
            self.hours_claims.append(claim)
            self.tracker.observe_target_value("doctor_hours", tuple(self.hours_claims), evidence=target_turn)
            if self.switch_acknowledged:
                self.post_switch_hours.append(claim)

        capability = str(o.get("multiple_doctors_capability", "")).casefold()
        if capability in {"yes", "no", "unavailable"}:
            prior = self.multiple_doctors_capability_observed
            if prior in {"yes", "no"} and capability in {"yes", "no"} and prior != capability:
                self._oracle("multiple_doctor_capability_contradiction", "multiple_doctors", prior, capability,
                             (f"prior target capability: {prior}", self._turn(target_turn)),
                             "Target gave explicit incompatible multiple-doctor capability statements.")
            self.multiple_doctors_capability_observed = capability
            self.tracker.observe_target_value("multiple_doctors_capability_observed", capability, evidence=target_turn)

        if action == "asks_switch_reason" and self.switch_requested_doctor:
            return self._say("I'd prefer a different doctor.", "switch_reason")
        if action == "switch_acknowledged" and self.switch_requested_doctor:
            if claimed_doctor and not same_grounded_doctor(claimed_doctor, self.switch_requested_doctor):
                self._oracle("doctor_switch_acknowledgment_failure", "switch_requested_doctor", self.switch_requested_doctor, claimed_doctor,
                             (f"switch requested: {self.switch_requested_doctor}", self._turn(target_turn)),
                             "Target accepted the switch but immediately established an incompatible doctor.")
            else:
                self.switch_acknowledged = True
                self.active_doctor = self.switch_requested_doctor
                self.locations = []
                self.selected_location = None
                self.tracker.commit_dialogue_value("active_doctor", self.active_doctor, evidence=target_turn)

        return self._next(action)

    def _next(self, action: str) -> PolicyDecision:
        if self.profile_registered and not self.name_probe_spoken:
            return self._say("Could you repeat the name you have on my profile and spell it for me?", "verify_registered_name")
        if not self.profile_registered:
            return PolicyDecision(DecisionKind.WAIT, reason="doctor_directory:await_profile_registration")
        if not self.name_verified:
            return self._say("Could you repeat the name you have on my profile and spell it for me?", "verify_registered_name")
        if not self.offered_doctors:
            return self._say("What doctors or specialists are available?", "discover_specialists")
        if self.selected_doctor is None:
            self.selected_doctor = self.offered_doctors[0]["name"]
            self.selected_doctor_initial = self.selected_doctor
            self.active_doctor = self.selected_doctor
            self.initial_doctor_selection_acknowledged = False
            self.tracker.commit_dialogue_value("selected_doctor_initial", self.selected_doctor, grounded_in_target=None, evidence="selected from target offer")
            offered_specialty = self.offered_doctors[0].get("specialty", "")
            if offered_specialty:
                self.specialty = offered_specialty
                return self._say(f"I'd like Doctor {self.selected_doctor} as my doctor.", "select_grounded_doctor")
            return self._say(f"I'd like Doctor {self.selected_doctor} as my doctor. What specialty does that doctor handle?", "select_grounded_doctor")
        if not self.specialty:
            return self._say("What specialty is that doctor?", "ask_specialty")
        doctor_key = normalize_person_name(self.active_doctor or self.selected_doctor or "")
        if action == "gender_unavailable":
            self.gender_probe_complete.add(doctor_key)
        if not self.switch_acknowledged and doctor_key not in self.gender_probe_complete:
            return self._say("Is that doctor a male or female doctor?", "ask_gender")
        if not self.locations and not self.hours_claims:
            return self._say("Which location does that doctor work at, and what hours does that doctor work there?", "ask_location_hours")
        if not self.locations:
            return self._say("Which location does that doctor work at?", "ask_location")
        if not any(same_grounded_doctor(str(c["doctor"]), self.active_doctor or "") for c in self.hours_claims):
            return self._say("What hours does that doctor work at that location?", "ask_hours")
        if self.multiple_doctors_capability_observed is None:
            return self._say("Can I see more than one doctor if I need to, or do I have to choose just one?", "ask_multiple_doctor_capability")
        if not self.switch_requested_doctor:
            alternatives = [d["name"] for d in self.offered_doctors if not same_grounded_doctor(d["name"], self.selected_doctor or "")]
            if not alternatives:
                return self._say("No, that's all. Thank you.", "finish_no_switch_candidate")
            self.switch_requested_doctor = alternatives[0]
            self.tracker.commit_dialogue_value("switch_requested_doctor", self.switch_requested_doctor, evidence="selected from target offer")
            return self._say(f"Actually, I'd like to switch to Doctor {self.switch_requested_doctor}.", "request_switch")
        if not self.switch_acknowledged:
            return PolicyDecision(DecisionKind.WAIT, reason="doctor_directory:await_switch_acknowledgment")
        if not self.post_switch_specialty:
            return self._say("What specialty does my doctor handle?", "post_switch_specialty")
        if not self.post_switch_locations or not self.post_switch_hours:
            return self._say("Which location does my doctor work at, and what hours does that doctor work there?", "post_switch_location_hours")
        if not self.final_reported_active_doctor:
            return self._say("Which doctor do I have now?", "final_active_doctor_check")
        return self._say("No, that's all. Thank you.", "finish")

    def mark_decision_spoken(self, decision: PolicyDecision) -> None:
        self.last_spoken_reason = decision.reason
        if decision.reason.endswith("verify_registered_name"):
            self.name_probe_spoken = True
        if decision.reason.endswith("switch_reason"):
            self.switch_reason_spoken = True

    def mark_decision_suppressed(self, decision: PolicyDecision) -> None:
        """Suppressed speech must not advance a spoken milestone."""
        return None

    @property
    def objective_complete(self) -> bool:
        return self.name_verified and bool(self.selected_doctor_initial and self.multiple_doctors_capability_observed and self.switch_requested_doctor and self.switch_acknowledged and self.post_switch_specialty and self.post_switch_locations and self.post_switch_hours)

    def _say(self, text: str, reason: str) -> PolicyDecision:
        return PolicyDecision(DecisionKind.CONTEXTUAL_ANSWER, text=text, reason=f"doctor_directory:{reason}")

    def _literal(self, source: str, value: object, field: str) -> str | None:
        candidate = " ".join(str(value or "").split())
        if not candidate:
            return None
        compact_source = re.sub(r"[^a-z0-9]+", "", source.casefold())
        compact_value = re.sub(r"[^a-z0-9]+", "", candidate.casefold())
        if compact_value and compact_value in compact_source:
            return candidate
        self.rejected_extractions.append({"field": field, "value": candidate, "source_turn": source})
        return None

    def _turn(self, text: str) -> str:
        return f"turn={self.turn}; target: {text}"

    def _oracle(self, name: str, field: str, expected: object, observed: object, turns: tuple[str, ...], reason: str) -> None:
        if any(e.oracle_name == name for e in self.evidence):
            return
        self.evidence.append(OracleEvidence(name, "candidate", SCENARIO_ID, field, expected, observed, turns,
                                            ("caller_scenario", "target_observed", "committed_validated"), reason))

    def metadata(self) -> dict[str, object]:
        s = self.tracker.snapshot()
        return {"scenario": SCENARIO_ID, "objective_complete": self.objective_complete,
                "profile_registered": self.profile_registered, "name_verification_complete": self.name_verified,
                "selected_doctor": self.selected_doctor, "specialty": self.specialty,
                "selected_doctor_initial": self.selected_doctor_initial, "initial_doctor_selection_acknowledged": self.initial_doctor_selection_acknowledged,
                "multiple_doctors_capability_observed": self.multiple_doctors_capability_observed,
                "switch_requested_doctor": self.switch_requested_doctor, "switch_reason_spoken": self.switch_reason_spoken,
                "switch_acknowledged": self.switch_acknowledged, "active_doctor": self.active_doctor,
                "gender_probe_complete": tuple(sorted(self.gender_probe_complete)),
                "explicit_gender_claims": tuple(g for _, g, _ in self.gender_claims), "doctor_locations": tuple(self.locations),
                "hours_claims": tuple(self.hours_claims), "context_followup": self.context_followup,
                "post_switch_specialty": self.post_switch_specialty, "post_switch_locations": tuple(self.post_switch_locations),
                "post_switch_hours": tuple(self.post_switch_hours), "final_reported_active_doctor": self.final_reported_active_doctor,
                "caller_truth": {k: v.value for k, v in s.caller_truth.items()},
                "target_observations": {k: v.value for k, v in s.target_observations.items()},
                "committed_state": {k: v.value for k, v in s.committed_dialogue.items()},
                "rejected_extractions": tuple(self.rejected_extractions), "oracle_evidence": tuple(asdict(e) for e in self.evidence)}
