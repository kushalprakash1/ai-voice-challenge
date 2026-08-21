"""Semantic-only Ollama interpreter for VoiceProbe v3.3 semantic-planner v0.17.

Qwen no longer generates or ranks patient actions. It performs one job:
convert the latest clinic utterance into a compact RemoteObservation. Python
then generates, validates, scores, and verbalizes available actions from state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from voiceprobe.v32.ollama_backend import OllamaBackend, OllamaConfig

from .action_generator import StrategicActionGenerator
from .actions import ActionPlan
from .mind import AgentMind
from .world_model import ObservationKind, RemoteObservation


FACT_KEYS = (
    "first_name",
    "last_name",
    "full_name",
    "dob",
    "insurance",
    "complaint",
    "visit_type",
    "reschedule_reason",
)

REMOTE_CLAIMS = (
    "profile_exists",
    "profile_missing",
    "appointment_exists",
    "appointment_missing",
)

TRANSACTION_OPS = (
    "none",
    "book",
    "reschedule",
    "cancel",
    "keep",
    "search",
)

QUESTION_TARGETS = (
    "none",
    "reschedule_reason",
    "patient_fact",
    "search_preference",
    "option_choice",
    "transaction_permission",
    "profile",
    "presence",
    "open_intent",
    "clarification",
)

RECORD_STATUSES = (
    "none",
    "exists",
    "missing",
)

FALLBACK_TARGETS = (
    "none",
    "clock_time_or_daypart",
    "calendar_date_or_day",
    "either_time_or_calendar",
    "provider",
    "unspecified_preference",
)

AVAILABILITY_FAILURES = (
    "none",
    "clock_time_or_daypart",
    "calendar_date_or_day",
    "provider",
    "combination",
)

RECORD_CLAIMS_V12 = (
    "none",
    "profile_exists",
    "profile_missing",
    "appointment_exists",
    "appointment_missing",
)


_ACTIONABLE_KINDS = {
    ObservationKind.PROFILE_REQUEST,
    ObservationKind.IDENTITY_CONFIRMATION,
    ObservationKind.FACT_REQUEST,
    ObservationKind.RESCHEDULE_REASON_REQUEST,
    ObservationKind.OPEN_INTENT,
    ObservationKind.VISIT_TYPE_REQUEST,
    ObservationKind.PROVIDER_PREFERENCE_REQUEST,
    ObservationKind.PROVIDER_NAME_REQUEST,
    ObservationKind.AVAILABILITY_RESULT,
    ObservationKind.ALTERNATIVE_SEARCH_OFFER,
    ObservationKind.OPTION_OFFER,
    ObservationKind.TRANSACTION_PERMISSION_REQUEST,
    ObservationKind.PRESENCE_CHECK,
    ObservationKind.CLARIFICATION_REQUEST,
}


class Reasoner(Protocol):
    async def propose(
        self,
        *,
        mind: AgentMind,
        remote_turn: str,
    ) -> tuple[RemoteObservation, tuple[ActionPlan, ...]]:
        ...


@dataclass(slots=True)
class OllamaV33Reasoner:
    backend: OllamaBackend
    generator: StrategicActionGenerator = field(default_factory=StrategicActionGenerator)

    @classmethod
    def from_endpoint(
        cls,
        *,
        endpoint: str,
        model: str = "qwen3.5:4b",
        timeout_seconds: float = 2.5,
    ) -> "OllamaV33Reasoner":
        return cls(
            backend=OllamaBackend(
                OllamaConfig(
                    endpoint=endpoint,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    keep_alive="15m",
                    num_ctx=768,
                    temperature=0.0,
                )
            ),
            generator=StrategicActionGenerator(),
        )

    async def warmup(self) -> None:
        await self.backend.generate_json(
            system="Return the required JSON object only.",
            prompt='{"warmup":true}',
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        )

    def _schema(self, preference_keys: tuple[str, ...]) -> dict[str, Any]:
        # v0.10 keeps semantic perception independent from mission implementation
        # axis names. The model reports semantic features; Python maps them to
        # day/time_of_day/provider/combination after inference.
        _ = preference_keys
        return {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [x.value for x in ObservationKind],
                    "description": (
                        "Coarse speech act only. Python may derive a more specific "
                        "alternative-search observation from fallback_target. option_offer "
                        "is only for concrete appointment choices/slots, not a broad daypart "
                        "or a request to search another constraint."
                    ),
                },
                "respond": {"type": "boolean"},
                "fact": {
                    "type": "string",
                    "enum": ["none", *FACT_KEYS],
                },
                "offers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 4,
                    "description": (
                        "Concrete appointment choices only, such as an actual dated/timed slot. "
                        "A daypart category, another day/time, or a search direction belongs in "
                        "fallback_target and must not be placed here."
                    ),
                },
                "availability_failure": {
                    "type": "string",
                    "enum": list(AVAILABILITY_FAILURES),
                    "description": (
                        "What the latest utterance explicitly says is unavailable. "
                        "clock_time_or_daypart = a clock time or within-day bucket failed; "
                        "calendar_date_or_day = a date/day/week/calendar window failed; "
                        "provider = a clinician/provider failed; combination = only the current set of constraints has no match "
                        "or multiple axes fail together; none = no availability failure is stated."
                    ),
                },
                "fallback_target": {
                    "type": "string",
                    "enum": list(FALLBACK_TARGETS),
                    "description": (
                        "Exactly one category describing what the clinic explicitly proposes varying next. "
                        "clock_time_or_daypart = keep the calendar identity and vary clock time/daypart/earlier-later opening; "
                        "calendar_date_or_day = vary date/day/week/month/calendar horizon; "
                        "either_time_or_calendar = explicitly offers changing either time-of-day OR calendar as alternatives; "
                        "provider = vary clinician/provider; unspecified_preference = asks which constraint/preference can change without naming one; "
                        "none = no fallback search is proposed. Choose one category only."
                    ),
                },
                "record_claim": {
                    "type": "string",
                    "enum": list(RECORD_CLAIMS_V12),
                    "description": (
                        "One explicit clinic claim about profile or existing-appointment record status, otherwise none. "
                        "Availability language such as no openings, no slots, or no matching appointments is never a missing-record claim."
                    ),
                },
                "selected": {"type": "string"},
                "operation": {
                    "type": "string",
                    "enum": list(TRANSACTION_OPS),
                },
            },
            "required": [
                "kind",
                "respond",
                "fact",
                "offers",
                "availability_failure",
                "fallback_target",
                "record_claim",
                "selected",
                "operation",
            ],
            "additionalProperties": False,
        }

    async def propose(
        self,
        *,
        mind: AgentMind,
        remote_turn: str,
    ) -> tuple[RemoteObservation, tuple[ActionPlan, ...]]:
        preference_keys = tuple(pref.key for pref in mind.mission.preferences)

        # Semantic perception must not see patient goal/preference values. Those
        # belong to Python planning and previously contaminated model output
        # (for example leaking provider into unrelated calendar fallbacks).
        payload = {
            "state": {
                "profile": mind.world.profile_status,
                "appointment": mind.world.remote_existing_appointment,
                "selected": mind.world.selected_option,
                "verified": mind.world.selection_verified,
                "authorized": mind.world.transaction_authorized,
            },
            "recent": mind.world.history[-2:],
            "turn": " ".join(remote_turn.split()),
        }

        system = """You are the semantic interpreter inside a simulated patient voice agent.

Interpret ONLY the clinic's latest utterance. State/recent turns may resolve references, but are not evidence for facts absent from the latest utterance. Patient goals and preference values are hidden. Never choose patient actions, write patient dialogue, or invent patient facts.

The output fields have separate jobs. Do not make them agree by copying labels between fields:
- kind is the coarse speech act.
- fallback_target says what the clinic explicitly proposes varying next.
- availability_failure says what the clinic explicitly reports as unavailable.
- fact is an explicitly requested authoritative patient fact.
- record_claim is only explicit profile/existing-appointment RECORD state.

Dominance rule: a direct fallback/search question is an alternative-search interaction even if the proposed fallback is phrased like an option. Python will use fallback_target as authoritative evidence for that distinction.

kind rules:
- reschedule_reason_request: asks for the CAUSE/explanation for moving an existing appointment.
- alternative_search_offer: asks which fallback/search direction to try because availability/preferences did not match.
- option_offer: presents one or more CONCRETE appointment choices/slots for selection. A broad daypart such as mornings, a different day/time, or a request to search a category is not a concrete option_offer.
- availability_result: reports availability/unavailability without asking what fallback to search and without presenting a concrete slot choice.
- profile_request: asks whether to create/start a patient profile.
- presence_check: asks whether the caller is still there.
- transaction_permission_request: asks permission to actually book/reschedule/cancel/keep.
- open_intent: asks generally how it can help.
- acknowledgement: short acknowledgement requiring no answer.
- other: only when no defined kind fits.

Critical distinction:
- Asking WHY the patient needs to reschedule => reschedule_reason_request.
- Asking WHAT alternative to search or whether another constraint would work => alternative_search_offer.
Never infer reschedule_reason merely because the workflow involves rescheduling.

fallback_target is ONE mutually exclusive answer to: WHAT does the clinic explicitly propose varying next?
- clock_time_or_daypart: vary clock time/daypart/earlier-later opening while calendar identity stays the same or no new calendar identity is proposed. A named weekday used only as an anchor does not make this a calendar change.
- calendar_date_or_day: explicitly vary date/day/weekday/week/month/calendar horizon.
- either_time_or_calendar: explicitly offers changing either time-of-day OR calendar as separate alternatives.
- provider: explicitly proposes a different clinician/provider/person.
- unspecified_preference: asks which constraint/preference can change but names no axis.
- none: no fallback search is proposed.
Do not infer provider or unspecified fallback from generic unavailability.

availability_failure is ONE mutually exclusive answer to: WHAT does the clinic explicitly say failed?
- clock_time_or_daypart: a clock time/daypart is unavailable.
- calendar_date_or_day: a date/day/calendar horizon is unavailable.
- provider: a clinician/provider is unavailable.
- combination: the current combination/set of constraints has no match, or multiple dimensions fail together without one clean single axis.
- none: no availability failure is explicitly stated.
Failure describes what failed; fallback_target describes what the clinic proposes varying next. A cross-axis fallback is possible, so never copy one into the other automatically.

record_claim is only an explicit statement about profile/existing-appointment RECORD existence or absence. Availability language such as no openings, no slots, or no matching appointments is never a missing-record claim.

offers contains only concrete appointment slots/choices explicitly offered in the latest turn. Broad categories and search directions belong only in fallback_target. selected is only a concrete appointment explicitly stated as selected. respond=true for a direct question/request/presence check/choice/permission/fallback offer.

Return the required JSON only."""

        raw = await self.backend.generate_json(
            system=system,
            prompt=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            schema=self._schema(preference_keys),
        )
        raw = await self._adjudicate_temporal_disagreement(
            raw=raw,
            remote_turn=remote_turn,
        )

        observation = self._observation_from_raw(raw, remote_turn)
        candidates = self.generator.generate(mind=mind, observation=observation)
        return observation, candidates

    @staticmethod
    def _needs_temporal_adjudication(raw: dict[str, Any]) -> bool:
        """Return True only when the first pass leaves temporal change ambiguous.

        This is semantic-structure based: it never inspects raw clinic phrases.
        A narrow second read is used for either of two situations:
        1. the first pass says both time and calendar are proposed fallbacks; or
        2. it says one temporal axis failed but a different single axis is proposed.

        The adjudicator exists to distinguish a dimension that is merely mentioned
        as fixed/retained from a dimension the clinic actually proposes varying.
        """

        # Legacy fixtures deliberately bypass the live v0.14 schema so older
        # scripted backends never acquire a hidden extra model call.
        if "question_target" in raw:
            return False

        failure = str(raw.get("availability_failure") or "none")
        target = str(raw.get("fallback_target") or "none")

        if target == "either_time_or_calendar":
            return True

        temporal = {"clock_time_or_daypart", "calendar_date_or_day"}
        return failure in temporal and target in temporal and failure != target

    async def _adjudicate_temporal_disagreement(
        self,
        *,
        raw: dict[str, Any],
        remote_turn: str,
    ) -> dict[str, Any]:
        if not self._needs_temporal_adjudication(raw):
            return raw

        resolved = await self.backend.generate_json(
            system=(
                "Resolve one semantic ambiguity in the clinic's latest utterance. "
                "Judge ONLY what the clinic proposes CHANGING in the fallback search, "
                "not what failed and not dimensions merely mentioned as fixed/retained. "
                "A dimension explicitly kept the same is NOT a proposed fallback axis. "
                "calendar_change=no means only clock time/daypart changes while calendar "
                "identity is retained, or an earlier/later opening is proposed without a "
                "new calendar identity. calendar_change=yes means the proposed fallback "
                "changes date/day/weekday/week/month/calendar horizon while time is retained "
                "or unspecified. calendar_change=both means the clinic genuinely offers "
                "time/daypart change OR calendar change as separate alternative search "
                "directions. Mere mention of both dimensions is not enough for both. "
                "Return the required JSON only."
            ),
            prompt=json.dumps(
                {"turn": " ".join(remote_turn.split())},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            schema={
                "type": "object",
                "properties": {
                    "calendar_change": {
                        "type": "string",
                        "enum": ["no", "yes", "both"],
                    }
                },
                "required": ["calendar_change"],
                "additionalProperties": False,
            },
        )

        calendar_change = str(resolved.get("calendar_change") or "")
        target_map = {
            "no": "clock_time_or_daypart",
            "yes": "calendar_date_or_day",
            "both": "either_time_or_calendar",
        }
        target = target_map.get(calendar_change)
        if target is None:
            raise ValueError(
                f"Invalid temporal adjudication result: {calendar_change!r}"
            )

        updated = dict(raw)
        updated["fallback_target"] = target
        return updated

    @staticmethod
    def _record_claims_from_raw(raw: dict[str, Any]) -> tuple[str, ...]:
        # v0.12 uses one mutually-exclusive record claim so the model cannot
        # independently hallucinate profile and appointment status together.
        if "record_claim" in raw:
            claim = str(raw.get("record_claim") or "none")
            return () if claim == "none" else (claim,)

        # Legacy v0.9-v0.11 fixture compatibility.
        if "profile_record_status" in raw or "appointment_record_status" in raw:
            claims: list[str] = []
            profile = str(raw.get("profile_record_status") or "none")
            appointment = str(raw.get("appointment_record_status") or "none")
            if profile in {"exists", "missing"}:
                claims.append(f"profile_{profile}")
            if appointment in {"exists", "missing"}:
                claims.append(f"appointment_{appointment}")
            return tuple(claims)
        return tuple(str(x) for x in raw.get("claims", ()))

    def _observation_from_raw(
        self,
        raw: dict[str, Any],
        remote_turn: str,
    ) -> RemoteObservation:
        kind = ObservationKind(str(raw["kind"]))
        fact = str(raw.get("fact") or "")
        if fact == "none":
            fact = ""

        question_target = str(raw.get("question_target") or "none")

        def temporal_axes(scope: str) -> tuple[str, ...]:
            if scope == "within_day":
                return ("time_of_day",)
            if scope == "calendar":
                return ("day",)
            if scope == "mixed":
                return ("time_of_day", "day")
            return ()

        v12_semantic_contract = any(
            key in raw for key in {"fallback_target", "availability_failure", "record_claim"}
        )
        v11_semantic_contract = (
            not v12_semantic_contract
            and "fallback_temporal_change" in raw
        )
        v10_semantic_contract = (
            not v11_semantic_contract
            and "fallback_temporal_relation" in raw
        )
        v09_semantic_contract = (
            not v11_semantic_contract
            and not v10_semantic_contract
            and any(
                key in raw
                for key in {
                    "failed_time_of_day",
                    "failed_calendar",
                    "fallback_time_of_day",
                    "fallback_calendar",
                    "profile_record_status",
                    "appointment_record_status",
                }
            )
        )
        v08_semantic_contract = (
            not v11_semantic_contract
            and not v10_semantic_contract
            and not v09_semantic_contract
            and any(
                key in raw
                for key in {
                    "failed_temporal_scope",
                    "provider_unavailable",
                    "combination_unavailable",
                    "provider_fallback",
                    "unspecified_relaxation",
                }
            )
        )

        if v12_semantic_contract:
            failure = str(raw.get("availability_failure") or "none")
            failure_map = {
                "clock_time_or_daypart": ("time_of_day",),
                "calendar_date_or_day": ("day",),
                "provider": ("provider",),
                "combination": ("combination",),
                "none": (),
            }
            unavailable = failure_map.get(failure, ())

            target = str(raw.get("fallback_target") or "none")
            target_map = {
                "clock_time_or_daypart": ("time_of_day",),
                "calendar_date_or_day": ("day",),
                "either_time_or_calendar": ("time_of_day", "day"),
                "provider": ("provider",),
                "unspecified_preference": ("combination",),
                "none": (),
            }
            search_constraints = target_map.get(target, ())
        elif v11_semantic_contract:
            unavailable_parts: list[str] = []
            if bool(raw.get("failed_time_of_day")):
                unavailable_parts.append("time_of_day")
            if bool(raw.get("failed_calendar")):
                unavailable_parts.append("day")
            if bool(raw.get("provider_unavailable")):
                unavailable_parts.append("provider")
            if bool(raw.get("combination_unavailable")):
                unavailable_parts.append("combination")
            unavailable = tuple(dict.fromkeys(unavailable_parts))

            change = str(raw.get("fallback_temporal_change") or "none")
            search_parts: list[str] = []
            if change == "time_of_day":
                search_parts.append("time_of_day")
            elif change == "calendar":
                search_parts.append("day")
            elif change == "time_or_calendar":
                search_parts.extend(("time_of_day", "day"))
            if bool(raw.get("provider_fallback")):
                search_parts.append("provider")
            if bool(raw.get("unspecified_relaxation")):
                search_parts.append("combination")
            search_constraints = tuple(dict.fromkeys(search_parts))
        elif v10_semantic_contract:
            unavailable_parts: list[str] = []
            if bool(raw.get("failed_time_of_day")):
                unavailable_parts.append("time_of_day")
            if bool(raw.get("failed_calendar")):
                unavailable_parts.append("day")
            if bool(raw.get("provider_unavailable")):
                unavailable_parts.append("provider")
            if bool(raw.get("combination_unavailable")):
                unavailable_parts.append("combination")
            unavailable = tuple(dict.fromkeys(unavailable_parts))

            relation = str(raw.get("fallback_temporal_relation") or "none")
            search_parts: list[str] = []
            if relation == "retains_calendar_changes_time":
                search_parts.append("time_of_day")
            elif relation == "changes_calendar":
                search_parts.append("day")
            elif relation == "offers_both":
                search_parts.extend(("time_of_day", "day"))
            if bool(raw.get("provider_fallback")):
                search_parts.append("provider")
            if bool(raw.get("unspecified_relaxation")):
                search_parts.append("combination")
            search_constraints = tuple(dict.fromkeys(search_parts))
        elif v09_semantic_contract:
            # Compatibility path for v0.9 fixtures/test doubles.
            unavailable_parts: list[str] = []
            if bool(raw.get("failed_time_of_day")):
                unavailable_parts.append("time_of_day")
            if bool(raw.get("failed_calendar")):
                unavailable_parts.append("day")
            if bool(raw.get("provider_unavailable")):
                unavailable_parts.append("provider")
            if bool(raw.get("combination_unavailable")):
                unavailable_parts.append("combination")
            unavailable = tuple(dict.fromkeys(unavailable_parts))

            search_parts: list[str] = []
            if bool(raw.get("fallback_time_of_day")):
                search_parts.append("time_of_day")
            if bool(raw.get("fallback_calendar")):
                search_parts.append("day")
            if bool(raw.get("provider_fallback")):
                search_parts.append("provider")
            if bool(raw.get("unspecified_relaxation")):
                search_parts.append("combination")
            search_constraints = tuple(dict.fromkeys(search_parts))
        elif v08_semantic_contract:
            # Compatibility path for v0.8 fixtures/test doubles.
            failed_scope = str(raw.get("failed_temporal_scope") or "none")
            unavailable_parts = list(temporal_axes(failed_scope))
            if bool(raw.get("provider_unavailable")):
                unavailable_parts.append("provider")
            if bool(raw.get("combination_unavailable")):
                unavailable_parts.append("combination")
            unavailable = tuple(dict.fromkeys(unavailable_parts))

            proposed_scope = str(raw.get("temporal_scope") or "none")
            search_parts = list(temporal_axes(proposed_scope))
            if bool(raw.get("provider_fallback")):
                search_parts.append("provider")
            if bool(raw.get("unspecified_relaxation")):
                search_parts.append("combination")
            search_constraints = tuple(dict.fromkeys(search_parts))
        else:
            # Compatibility path for v0.7 and older fixtures/test doubles only.
            unavailable = tuple(
                "day" if str(x) == "date_range" else str(x)
                for x in raw.get("unavailable", ())
                if str(x) != "none"
            )
            if "temporal_scope" in raw:
                proposed_scope = str(raw.get("temporal_scope") or "none")
                non_temporal = [
                    str(x)
                    for x in raw.get("search_constraints", ())
                    if str(x) not in {"none", "day", "time_of_day", "date_range"}
                ]
                derived = temporal_axes(proposed_scope)
                if not derived:
                    derived = tuple(
                        dict.fromkeys(
                            "day" if str(x) == "date_range" else str(x)
                            for x in raw.get("search_constraints", ())
                            if str(x) in {"day", "time_of_day", "date_range"}
                        )
                    )
                search_constraints = tuple(dict.fromkeys((*non_temporal, *derived)))
            else:
                search_constraints = tuple(
                    dict.fromkeys(
                        "day" if str(x) == "date_range" else str(x)
                        for x in raw.get("search_constraints", ())
                        if str(x) != "none"
                    )
                )

        operation = str(raw.get("operation") or "")
        if operation == "none":
            operation = ""

        # v0.13 live contract removes question_target. fallback_target already
        # answers whether the clinic is proposing a search relaxation, so Python
        # derives the dominant observation from that structured semantic fact.
        # This eliminates contradictory kind/question_target voting without ever
        # matching raw clinic phrases. Legacy fixtures keep the older repair path.
        live_v13_contract = (
            "fallback_target" in raw
            and "availability_failure" in raw
            and "record_claim" in raw
            and "question_target" not in raw
        )

        if live_v13_contract:
            if search_constraints:
                kind = ObservationKind.ALTERNATIVE_SEARCH_OFFER
                fact = ""
                operation = "search"
            elif kind is ObservationKind.ALTERNATIVE_SEARCH_OFFER:
                fact = ""
                operation = "search"
                if not search_constraints:
                    search_constraints = ("combination",)
            elif kind is ObservationKind.RESCHEDULE_REASON_REQUEST:
                fact = "reschedule_reason"
            elif fact == "reschedule_reason":
                # The fact field alone cannot transform another speech act into
                # a causal reschedule question.
                fact = ""
        elif question_target == "search_preference":
            kind = ObservationKind.ALTERNATIVE_SEARCH_OFFER
            fact = ""
            operation = "search"
            if not search_constraints:
                search_constraints = ("combination",)
        elif question_target == "reschedule_reason":
            kind = ObservationKind.RESCHEDULE_REASON_REQUEST
            fact = "reschedule_reason"
        elif kind is ObservationKind.ALTERNATIVE_SEARCH_OFFER:
            fact = ""
            operation = "search"
        elif kind is ObservationKind.RESCHEDULE_REASON_REQUEST:
            fact = "reschedule_reason"
        elif fact == "reschedule_reason":
            fact = ""

        remote_claims = self._record_claims_from_raw(raw)
        requires_response = (
            bool(raw.get("respond"))
            or kind in _ACTIONABLE_KINDS
            or bool(remote_claims)
        )

        return RemoteObservation(
            kind=kind,
            raw_text=remote_turn,
            requires_response=requires_response,
            requested_fact=fact,
            offered_options=tuple(
                " ".join(str(x).split())
                for x in raw.get("offers", ())
                if str(x).strip()
            ),
            unavailable_constraints=unavailable,
            search_constraints=search_constraints,
            remote_claims=remote_claims,
            selected_option=" ".join(str(raw.get("selected") or "").split()),
            transaction_operation=operation,
        )


# ---------------------------------------------------------------------------
# Feature-flagged reasoner construction.
#
# Default behavior is intentionally unchanged. SemanticLab-v2 is selected only
# when VOICEPROBE_V33_LEVEL2_RUNTIME_CANDIDATE is exactly "1".
# ---------------------------------------------------------------------------

def build_v33_reasoner(
    *,
    endpoint: str,
    model: str = "qwen3.5:4b",
    timeout_seconds: float = 2.5,
) -> Reasoner:
    import os

    if os.environ.get("VOICEPROBE_V33_LEVEL2_RUNTIME_CANDIDATE") == "1":
        from .semantic_runtime_v2 import SemanticLabV2Reasoner
        return SemanticLabV2Reasoner.shared()

    return OllamaV33Reasoner.from_endpoint(
        endpoint=endpoint,
        model=model,
        timeout_seconds=timeout_seconds,
    )
