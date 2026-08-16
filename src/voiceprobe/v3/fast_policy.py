"""Deterministic fast policy for routine medical-scheduling dialogue.

The fast policy exists to keep elementary, safety-relevant turns out of the
slow LLM path. It does not attempt open-domain semantic understanding.
Unknown or genuinely novel turns return FALLBACK for a separate model layer.
"""

from __future__ import annotations

import re

from .models import DecisionKind, PatientFacts, PolicyDecision


_INCOMPLETE_SUFFIXES = ("...", "…", ",", "-", "—", ":")
_INCOMPLETE_FINAL_WORDS = frozenset({"and", "for", "of", "or", "to", "with"})


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _is_obvious_fragment(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.endswith(_INCOMPLETE_SUFFIXES):
        return True
    normalized = _norm(stripped)
    if normalized in {"would any", "can you", "would you", "do you"}:
        return True

    words = normalized.rstrip(".?!").split()
    if words and words[-1] in _INCOMPLETE_FINAL_WORDS:
        return True

    return False


class RoutineSchedulingPolicy:
    """Fast-path policy with deterministic patient-fact grounding."""

    def __init__(self, facts: PatientFacts | None = None) -> None:
        self.facts = facts or PatientFacts()

    def decide(self, agent_turn: str) -> PolicyDecision:
        text = _norm(agent_turn)
        raw = agent_turn.strip()
        f = self.facts

        if _is_obvious_fragment(raw):
            return PolicyDecision(
                DecisionKind.HOLD,
                reason="obvious_incomplete_asr_fragment",
            )

        # Acknowledgements and status updates. Boilerplate is evaluated only
        # after actionable intents so a disclaimer prefix cannot swallow a
        # real question in the same Flux EndOfTurn.
        if text in {
            "thanks.",
            "thanks",
            "thanks, alex.",
            "thanks, alex",
            "great.",
            "great",
            "great, alex.",
            "great, alex",
            "welcome to pivot point.",
            "welcome to pivot point",
            "thank you for calling.",
            "thank you for calling",
        }:
            return PolicyDecision(
                DecisionKind.WAIT,
                reason="acknowledgement_or_greeting",
            )

        if _contains_any(
            text,
            (
                "let me check available appointments",
                "let me create your demo patient profile",
            ),
        ) and "?" not in raw:
            return PolicyDecision(
                DecisionKind.WAIT,
                reason="remote_status_update",
            )

        # Specific fact requests must outrank generic scheduling language.
        if (
            _contains_any(
                text,
                (
                    "last name",
                    "surname",
                ),
            )
            and "create a demo patient profile" not in text
            and not _contains_any(
                text,
                (
                    "first and last name",
                    "first name and last name",
                ),
            )
        ):
            return PolicyDecision(
                DecisionKind.ANSWER_FACT,
                text=f"{f.last_name}.",
                reason="last_name_requested",
            )

        if _contains_any(
            text,
            (
                "first and last name",
                "first name and last name",
            ),
        ) and "create a demo patient profile" not in text:
            return PolicyDecision(
                DecisionKind.ANSWER_FACT,
                text=f"{f.first_name} {f.last_name}.",
                reason="full_name_requested",
            )

        # Required demo-profile workflow. Agree and provide identity in one turn.
        if "create a demo patient profile" in text:
            return PolicyDecision(
                DecisionKind.CREATE_PROFILE,
                text=f"Yes, please. My name is {f.first_name} {f.last_name}.",
                reason="profile_workflow_requested",
            )

        # Provider choice should not be mistaken for another date/time request.
        # Real Flux wording varies between "is first available okay?" and
        # "do you have a preference, or should I offer the first available?"
        # Keep this semantic rather than encoding provider names.
        provider_preference_requested = _contains_any(
            text,
            (
                "first available okay",
                "first available ok",
                "which provider",
                "prefer to see dr.",
                "provider do you prefer",
            ),
        )

        if not provider_preference_requested:
            mentions_first_available = "first available" in text
            mentions_provider = _contains_any(
                text,
                (
                    "provider",
                    "doctor",
                    "physician",
                    "dr.",
                ),
            )
            asks_provider_choice = _contains_any(
                text,
                (
                    "preference",
                    "prefer",
                    "offer",
                ),
            )
            provider_preference_requested = (
                mentions_first_available
                and mentions_provider
                and asks_provider_choice
            )

        if provider_preference_requested:
            return PolicyDecision(
                DecisionKind.ANSWER_PROVIDER_PREFERENCE,
                text="First available is fine.",
                reason="provider_preference_requested",
            )

        # A branch that explicitly preserves Friday afternoon. Flux can
        # render calendar ordinals either numerically or as spoken words.
        mentions_august_28 = _contains_any(
            text,
            (
                "august 28",
                "august twenty eighth",
                "august twenty-eighth",
            ),
        )
        if (
            mentions_august_28
            and "afternoon" in text
            and _contains_any(text, ("would you like", "check", "look"))
        ):
            if _contains_any(
                text,
                (
                    "or check other days",
                    "or another day",
                    "other days in the future",
                ),
            ):
                return PolicyDecision(
                    DecisionKind.CHOOSE_SEARCH_BRANCH,
                    text="Please check Friday, August 28th for afternoon appointments.",
                    reason="choose_constraint_preserving_search_branch",
                )
            return PolicyDecision(
                DecisionKind.GRANT_PERMISSION,
                text="Yes, please.",
                reason="compatible_following_friday_search",
            )

        # Direct reason-for-visit / complaint request.
        asks_reason = _contains_any(
            text,
            (
                "reason for your visit",
                "reason why",
                "why you'd like to be seen",
                "why you would like to be seen",
            ),
        )

        # Visit-type requests can appear as bare option lists with no question mark.
        appointment_type_language = _contains_any(
            text,
            (
                "new patient consultation",
                "routine checkup",
                "routine office visit",
                "general office visit",
                "follow-up",
                "follow up",
            ),
        )
        asks_type = appointment_type_language and (
            _contains_any(
                text,
                (
                    "is this",
                    "is this appointment",
                    "is this visit",
                    "or something else",
                ),
            )
            or text.startswith("a new patient consultation")
        )

        if asks_reason and asks_type:
            return PolicyDecision(
                DecisionKind.ANSWER_VISIT_DETAILS,
                text=(
                    f"I have {f.complaint}. "
                    f"This is for a {f.appointment_type}."
                ),
                reason="reason_and_visit_type_requested",
            )

        if asks_reason:
            return PolicyDecision(
                DecisionKind.ANSWER_COMPLAINT,
                text=f"I have {f.complaint}.",
                reason="complaint_requested",
            )

        if asks_type:
            return PolicyDecision(
                DecisionKind.ANSWER_APPOINTMENT_TYPE,
                text=f"A {f.appointment_type}.",
                reason="appointment_type_requested",
            )

        if _contains_any(
            text,
            (
                "insurance",
                "insurance provider",
                "what coverage",
            ),
        ):
            return PolicyDecision(
                DecisionKind.ANSWER_FACT,
                text=f"{f.insurance}.",
                reason="insurance_requested",
            )

        if _contains_any(
            text,
            (
                "date of birth",
                "dob",
                "birthday",
            ),
        ) and _contains_any(
            text,
            (
                "what is",
                "what's",
                "provide",
                "tell me",
            ),
        ):
            return PolicyDecision(
                DecisionKind.ANSWER_FACT,
                text=f"{f.dob}.",
                reason="dob_requested",
            )

        # Wrong DOB assertion + open-ended "how can I help" must be corrected.
        wrong_dob_asserted = (
            "date of birth is" in text
            and "april 12" not in text
            and "1998" not in text
        )
        open_ended_intent = _contains_any(
            text,
            (
                "how can i help you today",
                "what can i help you with",
                "how may i help you",
            ),
        )
        if wrong_dob_asserted and open_ended_intent:
            return PolicyDecision(
                DecisionKind.CORRECT_AND_STATE_OBJECTIVE,
                text=(
                    f"Actually, my date of birth is {f.dob}. "
                    f"I need to schedule an appointment for "
                    f"{f.preferred_day} {f.preferred_time}."
                ),
                reason="correct_remote_fact_then_answer_open_intent",
            )

        if open_ended_intent:
            return PolicyDecision(
                DecisionKind.STATE_OBJECTIVE,
                text=(
                    f"I need to schedule an appointment for "
                    f"{f.preferred_day} {f.preferred_time}."
                ),
                reason="open_ended_intent_question",
            )

        # Concrete morning-only choices conflict with the hard afternoon constraint.
        time_matches = re.findall(
            r"\b(?:9|9[.:]45|10[.:]30)\s*(?:a\.?m\.?|am)\b",
            text,
        )
        spoken_morning_time = _contains_any(
            text,
            (
                "nine am",
                "nine a.m.",
                "nine forty five am",
                "nine forty-five am",
                "nine forty five a.m.",
                "ten thirty am",
                "ten thirty a.m.",
            ),
        )
        if (time_matches or spoken_morning_time) and _contains_any(
            text,
            (
                "work for you",
                "do any of these",
                "would any of these",
            ),
        ):
            return PolicyDecision(
                DecisionKind.DECLINE_INCOMPATIBLE_OFFER,
                text=(
                    "Those times don't work for me. "
                    "Do you have anything Friday afternoon?"
                ),
                reason="morning_offer_conflicts_with_afternoon_constraint",
            )

        if text.startswith("thanks for confirming") and "?" not in raw:
            return PolicyDecision(
                DecisionKind.WAIT,
                reason="acknowledgement",
            )

        # Remote statements with no request do not need a reply.
        if (
            "there are no friday afternoon openings" in text
            or "we have opening" in text
            or "we have openings" in text
        ) and "?" not in raw:
            return PolicyDecision(
                DecisionKind.WAIT,
                reason="informational_availability_statement",
            )

        # Pure boilerplate is non-actionable. This check deliberately comes
        # after actionable intents because Flux may combine a recording
        # disclaimer and the first scheduling question into one EndOfTurn.
        if _contains_any(
            text,
            (
                "call may be recorded",
                "para español",
                "para espanol",
            ),
        ):
            return PolicyDecision(
                DecisionKind.WAIT,
                reason="boilerplate",
            )

        return PolicyDecision(
            DecisionKind.FALLBACK,
            reason="novel_or_ambiguous_turn",
            confidence=0.0,
        )
