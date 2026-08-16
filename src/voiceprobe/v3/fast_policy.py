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


_DIGIT_PM_RE = re.compile(
    r"\b(?:1[0-2]|0?[1-9])"
    r"(?::[0-5]\d|\.[0-5]\d)?"
    r"\s*(?:p\.?m\.?|pm)\b",
    flags=re.IGNORECASE,
)

_SPOKEN_PM_RE = re.compile(
    r"\b(?:one|two|three|four|five)"
    r"(?:\s+(?:fifteen|thirty|forty[\s-]?five))?"
    r"\s+(?:p\.?m\.?|pm)\b",
    flags=re.IGNORECASE,
)

_OTHER_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "saturday",
    "sunday",
)

_EARLIER_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
)


def _extract_offered_pm_slot(text: str) -> str | None:
    digit = _DIGIT_PM_RE.search(text)

    if digit is not None:
        return digit.group(0)

    spoken = _SPOKEN_PM_RE.search(text)

    if spoken is not None:
        return spoken.group(0)

    return None


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
        self._allow_earlier_week_afternoons = False

    @property
    def allow_earlier_week_afternoons(self) -> bool:
        """Whether Friday-only has been explicitly relaxed to Mon-Thu PM."""

        return self._allow_earlier_week_afternoons

    def relax_day_constraint_for_afternoon(self) -> None:
        """Allow earlier-week afternoons after an explicit fallback branch."""

        self._allow_earlier_week_afternoons = True

    def should_relax_day_constraint_for_afternoon(
        self,
        agent_turn: str,
    ) -> bool:
        """Recognize a remote fallback prompt that permits controlled relaxation."""

        text = _norm(agent_turn)

        offers_day_or_provider = (
            "afternoon" in text
            and _contains_any(
                text,
                (
                    "different day",
                    "another day",
                    "other day",
                ),
            )
            and _contains_any(
                text,
                (
                    "different provider",
                    "another provider",
                    "other provider",
                ),
            )
        )
        asks_earlier_week = (
            "afternoon" in text
            and _contains_any(
                text,
                (
                    "earlier in the week",
                    "earlier this week",
                    "earlier in week",
                ),
            )
            and _contains_any(
                text,
                (
                    "check",
                    "look",
                    "see",
                    "options",
                    "openings",
                    "availability",
                ),
            )
        )

        return offers_day_or_provider or asks_earlier_week

    def decide(self, agent_turn: str) -> PolicyDecision:
        text = _norm(agent_turn)
        raw = agent_turn.strip()
        f = self.facts

        # Live regression 2026-08-16: Flux emitted a complete, actionable
        # wrong-DOB + open-intent turn with a trailing comma. The generic
        # fragment gate must not suppress that correction.
        early_open_ended_intent = _contains_any(
            text,
            (
                "how can i help you today",
                "what can i help you with",
                "how may i help you",
                "can i help you today",
            ),
        )
        early_wrong_dob_asserted = (
            "date of birth is" in text
            and "april 12" not in text
            and "1998" not in text
        )

        if early_wrong_dob_asserted and early_open_ended_intent:
            return PolicyDecision(
                DecisionKind.CORRECT_AND_STATE_OBJECTIVE,
                text=(
                    f"Actually, my date of birth is {f.dob}. "
                    f"I need to schedule an appointment for "
                    f"{f.preferred_day} {f.preferred_time}."
                ),
                reason="correct_remote_fact_then_answer_open_intent",
            )

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

        # Live 2026-08-16 fallback: the scheduler offered afternoon options
        # on another day versus another provider, then asked whether to check
        # afternoon options earlier in the week. A bare "Yes, please" is
        # ambiguous here. Choose the day branch explicitly while preserving PM.
        offers_day_or_provider = (
            "afternoon" in text
            and _contains_any(
                text,
                (
                    "different day",
                    "another day",
                    "other day",
                ),
            )
            and _contains_any(
                text,
                (
                    "different provider",
                    "another provider",
                    "other provider",
                ),
            )
        )
        asks_earlier_week = (
            "afternoon" in text
            and _contains_any(
                text,
                (
                    "earlier in the week",
                    "earlier this week",
                    "earlier in week",
                ),
            )
            and _contains_any(
                text,
                (
                    "check",
                    "look",
                    "see",
                    "options",
                    "openings",
                    "availability",
                ),
            )
        )

        if offers_day_or_provider:
            return PolicyDecision(
                DecisionKind.CHOOSE_SEARCH_BRANCH,
                text="Please check afternoon options earlier in the week.",
                reason="choose_earlier_week_afternoon_search",
            )

        if asks_earlier_week:
            return PolicyDecision(
                DecisionKind.GRANT_PERMISSION,
                text=(
                    "Yes, please. Please check afternoon options "
                    "earlier in the week."
                ),
                reason="allow_earlier_week_afternoon_search",
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
                "can i help you today",
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

        if _contains_any(
            text,
            (
                "are you still there",
                "are you there",
                "hello, are you there",
            ),
        ):
            return PolicyDecision(
                DecisionKind.STATE_OBJECTIVE,
                text=(
                    "Yes, I'm here. "
                    f"I need to schedule an appointment for "
                    f"{f.preferred_day} {f.preferred_time}."
                ),
                reason="presence_check_restate_objective",
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
            follow_up = (
                "Do you have anything in the afternoon earlier in the week?"
                if self._allow_earlier_week_afternoons
                else "Do you have anything Friday afternoon?"
            )
            return PolicyDecision(
                DecisionKind.DECLINE_INCOMPATIBLE_OFFER,
                text=f"Those times don't work for me. {follow_up}",
                reason="morning_offer_conflicts_with_afternoon_constraint",
            )

        # Concrete slot acceptance was not exercised by the two historical
        # recordings because both ended before booking. Treat it as a first-class
        # scheduling primitive rather than a generic yes/no response.
        offered_pm_slot = _extract_offered_pm_slot(raw)
        slot_offer_cue = _contains_any(
            text,
            (
                "does that work",
                "would that work",
                "work for you",
                "would you like",
                "can book",
                "can schedule",
                "available at",
                "opening at",
                "appointment at",
                "slot at",
            ),
        )

        offered_earlier_weekday = _contains_any(text, _EARLIER_WEEKDAYS)
        offered_other_weekday = _contains_any(text, _OTHER_WEEKDAYS)

        if (
            offered_pm_slot is not None
            and offered_other_weekday
            and slot_offer_cue
            and not (
                self._allow_earlier_week_afternoons
                and offered_earlier_weekday
            )
        ):
            if self._allow_earlier_week_afternoons:
                decline_text = (
                    "That day doesn't work for me. "
                    "Please check an afternoon earlier in the week."
                )
                reason = "offer_outside_relaxed_earlier_week_window"
            else:
                decline_text = (
                    "That day doesn't work for me. "
                    "I need a Friday afternoon appointment."
                )
                reason = "non_friday_offer_conflicts_with_day_constraint"

            return PolicyDecision(
                DecisionKind.DECLINE_INCOMPATIBLE_OFFER,
                text=decline_text,
                reason=reason,
            )

        booking_confirmation = (
            offered_pm_slot is not None
            and _contains_any(
                text,
                (
                    "scheduled",
                    "booked",
                    "appointment is confirmed",
                    "appointment has been confirmed",
                    "you're confirmed",
                    "you are confirmed",
                    "reserved",
                ),
            )
        )

        if booking_confirmation:
            return PolicyDecision(
                DecisionKind.WAIT,
                reason="booking_confirmation",
            )

        if offered_pm_slot is not None and slot_offer_cue:
            return PolicyDecision(
                DecisionKind.GRANT_PERMISSION,
                text=(
                    f"Yes, please book the {offered_pm_slot} slot."
                ),
                reason="compatible_concrete_slot_offered",
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
