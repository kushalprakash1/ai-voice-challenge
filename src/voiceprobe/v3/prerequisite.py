"""Deterministic, interruptible caller-prerequisite handling.

This module intentionally owns only shared caller facts.  It does not run the
scheduling controller and cannot interpret medication, insurance, location, or
slot state supplied by the target.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Awaitable, Callable

from .flow_state import FlowSnapshot, SchedulingFlowTracker
from .models import DecisionKind, PatientFacts, PolicyDecision


ScenarioResolver = Callable[
    [str, FlowSnapshot], PolicyDecision | Awaitable[PolicyDecision]
]
CompoundFactProvider = Callable[[str], str | None]


@dataclass(slots=True)
class PrerequisiteState:
    profile_consent_spoken: bool = False
    identity_fields_spoken: tuple[str, ...] = ()
    dob_spoken: bool = False
    last_action: str = "none"
    deterministic_hits: int = 0
    profile_created_acknowledged: bool = False


class PrerequisiteOverlay:
    """Answer a current prerequisite request, then yield on the next turn."""

    def __init__(
        self,
        *,
        scenario_id: str,
        tracker: SchedulingFlowTracker,
        domain_resolver: ScenarioResolver,
        facts: PatientFacts | None = None,
        compound_fact_provider: CompoundFactProvider | None = None,
    ) -> None:
        self.scenario_id = scenario_id
        self.tracker = tracker
        self.domain_resolver = domain_resolver
        self.compound_fact_provider = compound_fact_provider
        self.facts = facts or PatientFacts()
        self.state = PrerequisiteState()
        for key, value in (
            ("first_name", self.facts.first_name),
            ("last_name", self.facts.last_name),
            ("full_name", f"{self.facts.first_name} {self.facts.last_name}"),
            ("date_of_birth", self.facts.dob),
        ):
            self.tracker.establish_caller_truth(
                key, value, evidence=f"{scenario_id} synthetic patient fixture"
            )

    async def resolve(
        self, target_turn: str, snapshot: FlowSnapshot
    ) -> PolicyDecision:
        if _acknowledges_profile_created(target_turn):
            self.state.profile_created_acknowledged = True
            self.tracker.observe_target_value(
                "profile_created_acknowledged",
                True,
                evidence=target_turn,
            )
        prerequisite = self.resolve_prerequisite(target_turn)
        if prerequisite is not None:
            self.state.deterministic_hits += 1
            return prerequisite
        self.state.last_action = "none"
        result = self.domain_resolver(target_turn, snapshot)
        if isinstance(result, Awaitable):
            return await result
        return result

    def resolve_prerequisite(self, target_turn: str) -> PolicyDecision | None:
        text = " ".join(target_turn.casefold().split())
        if re.search(r"\b(?:are you (?:still )?there|can you hear me)\b", text):
            self.state.last_action = "presence_check"
            return PolicyDecision(
                DecisionKind.ANSWER_FACT,
                text="Yes, I'm here.",
                reason="prerequisite:presence_check",
            )
        profile = _is_current_profile_request(text)
        first, last, full = _requested_name_fields(text)
        dob = _is_current_dob_request(text)

        # A blocking profile workflow commonly states the identity requirement
        # rather than putting a question mark after every field.  Treat that as
        # current only when profile creation/progression is itself requested.
        if profile and not (first or last or full):
            full = _mentions_name_requirement(text)

        if not (profile or first or last or full or dob):
            return None

        fields: list[str] = []
        parts: list[str] = []
        if profile:
            parts.append("Yes, please")
        if full or (first and last):
            fields.extend(("first_name", "last_name"))
            parts.append(f"my name is {self.facts.first_name} {self.facts.last_name}")
        elif first:
            fields.append("first_name")
            parts.append(f"my first name is {self.facts.first_name}")
        elif last:
            fields.append("last_name")
            parts.append(f"my last name is {self.facts.last_name}")
        if dob:
            fields.append("date_of_birth")
            parts.append(f"my date of birth is {self.facts.dob}")

        # Preserve both explicitly requested known facts for simple compound
        # prerequisite/domain prompts without creating a general planner.
        provider = self.compound_fact_provider
        if provider is not None and _currently_requests_medication(text):
            medication = provider("medication")
            if medication:
                parts.append(f"the medication is {medication}")
                fields.append("medication")
        if provider is not None and _currently_requests_pharmacy(text):
            pharmacy = provider("pharmacy_preference")
            if pharmacy:
                parts.append(f"please use {pharmacy}")
                fields.append("pharmacy_preference")

        self.state.last_action = "provide_prerequisite"
        return PolicyDecision(
            DecisionKind.CREATE_PROFILE if profile else DecisionKind.ANSWER_FACT,
            text=_join_response_parts(parts),
            reason="prerequisite:provide_" + "_and_".join(fields or ["profile_consent"]),
        )

    def mark_decision_spoken(self, decision: PolicyDecision) -> None:
        """Commit caller disclosures only after PCM delivery completes."""
        if not decision.reason.startswith("prerequisite:provide_"):
            return
        reason = decision.reason
        if decision.kind == DecisionKind.CREATE_PROFILE:
            self.state.profile_consent_spoken = True
        fields = tuple(
            field for field in ("first_name", "last_name") if field in reason
        )
        self.state.identity_fields_spoken = tuple(
            dict.fromkeys((*self.state.identity_fields_spoken, *fields))
        )
        if "date_of_birth" in reason:
            self.state.dob_spoken = True

    def metadata(self) -> dict[str, object]:
        return {
            "prerequisite_action": self.state.last_action,
            "profile_consent_spoken": self.state.profile_consent_spoken,
            "prerequisite_fields_provided": self.state.identity_fields_spoken
            + (("date_of_birth",) if self.state.dob_spoken else ()),
            "deterministic_prerequisite_hits": self.state.deterministic_hits,
            "profile_created_acknowledged": self.state.profile_created_acknowledged,
        }


def _is_current_profile_request(text: str) -> bool:
    # Ownership is clause-local: an unrelated "How can I help?" must not turn
    # a preceding "profile is set up" acknowledgment into a consent request.
    for clause in re.split(r"[?.!;]+", text):
        if not re.search(r"\b(?:demo\s+)?(?:patient\s+)?profile\b", clause):
            continue
        if re.search(
            r"\b(?:would you like (?:me|us)?\s*to|do you want (?:me|us)?\s*to|may i|can (?:i|we)|"
            r"shall (?:i|we))\b[^.!?]*\b(?:create|set up|make|start)\b",
            clause,
        ):
            return True
        if re.search(
            r"\b(?:need|have|required)\b[^.!?]*\b(?:create|set up|complete)\b",
            clause,
        ):
            return True
    return False


def is_high_confidence_initial_profile_prerequisite(text: str) -> bool:
    """True only for a semantically complete profile/name prerequisite.

    This deliberately rejects incomplete eager/partial text such as
    ``Would you like...``.  It is used only to select a shorter post-Flux-EOT
    stabilization hold; the normal prerequisite resolver still owns the
    response and caller facts.
    """
    normalized = " ".join(text.casefold().split())
    profile = _is_current_profile_request(normalized)
    first, last, full = _requested_name_fields(normalized)
    if profile:
        return True
    return full or (first and last)


def _acknowledges_profile_created(target_turn: str) -> bool:
    text = " ".join(target_turn.casefold().split())
    return bool(
        re.search(
            r"\b(?:profile|demo patient profile)\b\s+"
            r"(?:has been|was|is|'s)\s+(?:created|set up|ready|all set)\b",
            text,
        )
        or re.search(
            r"\b(?:created|set up|completed)\s+(?:your|the)\s+"
            r"(?:demo\s+)?(?:patient\s+)?profile\b",
            text,
        )
    )


def _requested_name_fields(text: str) -> tuple[bool, bool, bool]:
    request = r"(?:can|could|may|would) (?:i|you|we)|please|need|provide|give|tell|what(?:'s| is)|start with"
    full = bool(
        re.search(rf"\b(?:{request})\b[^?.!]*\b(?:full name|first and last name|first name and last name|your name)\b", text)
        or re.search(r"\b(?:full name|first and last name|first name and last name)\s*,?\s*please\b", text)
    )
    first = full or bool(re.search(rf"\b(?:{request})\b[^?.!]*\bfirst name\b", text))
    last = full or bool(re.search(rf"\b(?:{request})\b[^?.!]*\blast name\b", text))
    return first, last, full


def _mentions_name_requirement(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:need|require|required|start with)\b[^.!?]*\b"
            r"(?:name|first and last name|first name and last name)\b",
            text,
        )
    )


def _is_current_dob_request(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:(?:what(?:'s| is)|provide|give|tell|need|may i have|can i have)\b"
            r"[^?.!]*\b(?:date of birth|dob)|(?:date of birth|dob)\s*,?\s*please)\b",
            text,
        )
    )


def _currently_requests_medication(text: str) -> bool:
    # Future narration ("after that I'll ask...") and historical thanks do
    # not own the current turn.  Only a direct request in a compound prompt does.
    return bool(
        re.search(
            r"\b(?:and|then)\s+(?:tell|give|provide|what|the)\b[^?.!]*"
            r"\b(?:medication|prescription)\b",
            text,
        )
        or re.search(r"\band\s+the\s+(?:medication|prescription)\b", text)
    ) and not bool(re.search(r"\bafter (?:that|i|get|we)\b[^.!?]*\bask\b", text))


def _currently_requests_pharmacy(text: str) -> bool:
    return bool(re.search(r"\band\b[^?.!]*\b(?:which|what|your|the)\s+pharmacy\b", text))


def _join_response_parts(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0][0].upper() + parts[0][1:] + "."
    return parts[0][0].upper() + parts[0][1:] + ". " + ", and ".join(parts[1:]) + "."
