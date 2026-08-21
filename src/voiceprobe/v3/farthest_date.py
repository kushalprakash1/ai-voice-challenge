"""Thin farthest-date policy for the production scheduling controller."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime

from .flow_state import FlowSnapshot, extract_concrete_slot
from .models import DecisionKind, PolicyDecision

SCENARIO_ID = "farthest-date-scheduling"
OBJECTIVE_TEXT = "I'd like to schedule a new patient consultation. What's the furthest date in the future that you can currently book?"
FIRST_VISIT_RESPONSE = "Yes, this is my first visit and I'd like to establish care."
RESCHEDULE_RESPONSE = "I'd like to reschedule it. I'm looking for the furthest date in the future that you can currently book."
REASON_RESPONSE = "I'd prefer a later appointment."
DATE_PREFERENCE_RESPONSE = "I don't have a specific day. I want the latest date you can currently book."
TIME_PREFERENCE_RESPONSE = "I don't have a time preference."
VERIFICATION_TEXT = "Is that the latest date you can currently book?"
FALLBACK_TEXT = "What's the latest appointment date your scheduling system can currently see?"
BOOKING_ACCEPTANCE = "Yes, please book that appointment."
RESCHEDULE_CONFIRMATION_RESPONSE = (
    "Yes, that's the appointment I'd like to reschedule. Please move it to the "
    "furthest date in the future that you can currently book."
)
ALTERNATIVE_SEARCH_RESPONSE = (
    "Yes, please check a different day or time. I don't have a day or time "
    "preference. I want the latest date you can currently book."
)
CONTINUE_LATEST_SEARCH_RESPONSE = (
    "I'd like to look for a date further in the future. What's the latest "
    "date you can currently book?"
)

_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_DATE_RE = re.compile(
    rf"\b(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b",
    re.IGNORECASE,
)


class FarthestDatePolicy:
    """Scenario wording/selection overlay; scheduling remains controller-owned."""

    scenario_id = SCENARIO_ID

    def __init__(self) -> None:
        self.objective_stated = False
        self._objective_attempts_prepared = 0
        self.verification_asked = False
        self.fallback_used = False
        self.latest_offered_date: str | None = None
        self.latest_offered_slot: str | None = None
        self.booking_accepted = False

    def __call__(
        self,
        source_turns: tuple[str, ...],
        actionable_turn: str | None,
        decision: PolicyDecision,
        snapshot: FlowSnapshot,
    ) -> PolicyDecision:
        del snapshot
        # Flux may finalize the offer and each alternative as separate
        # transcript fragments.  The overlay owns the complete scheduling
        # choice, so reason over the preserved burst rather than its tail.
        raw = " ".join(source_turns) or actionable_turn or ""
        text = " ".join(raw.casefold().split())

        # Call #6 asks these after the objective is known.  Keep them ahead of
        # the generic appointment-type route and the semantic fallback.
        if self._asks_first_visit(text):
            return PolicyDecision(DecisionKind.CONTEXTUAL_ANSWER, FIRST_VISIT_RESPONSE, "call6:first_visit")
        if self._offers_keep_reschedule_cancel(text):
            return PolicyDecision(DecisionKind.CONTEXTUAL_ANSWER, RESCHEDULE_RESPONSE, "call6:reschedule_choice")
        # This confirms which existing appointment is being moved. It is not a
        # candidate slot, even when the clinic repeats its date and time.
        if self._asks_to_reschedule_existing_appointment(text):
            return PolicyDecision(
                DecisionKind.CONTEXTUAL_ANSWER,
                RESCHEDULE_CONFIRMATION_RESPONSE,
                "call6:confirm_existing_appointment_to_reschedule",
            )
        # Call #6 has no day/daypart constraint to preserve. Continue the
        # availability search instead of accepting a transfer.
        if self._offers_alternative_search_or_transfer(text):
            return PolicyDecision(
                DecisionKind.CHOOSE_SEARCH_BRANCH,
                ALTERNATIVE_SEARCH_RESPONSE,
                "call6:continue_unconstrained_latest_search",
            )

        if (
            (
                decision.reason in {"open_ended_intent_question", "correct_remote_fact_then_answer_open_intent"}
                or self._is_generic_help_prompt(text)
            )
            and not self.objective_stated
            and self._objective_attempts_prepared < 2
        ):
            self._objective_attempts_prepared += 1
            return PolicyDecision(DecisionKind.STATE_OBJECTIVE, OBJECTIVE_TEXT, "call6:objective")
        if decision.reason == "existing_appointment_reschedule":
            return replace(decision, text=RESCHEDULE_RESPONSE, reason="call6:reschedule")
        if decision.reason in {"reschedule_reason_requested", "complaint_requested"} and "reschedul" in text:
            return PolicyDecision(DecisionKind.CONTEXTUAL_ANSWER, REASON_RESPONSE, "call6:later_reason")
        if decision.reason == "appointment_type_requested":
            return decision
        dates = self._offered_dates(raw)
        if dates:
            latest = max(dates, key=self._date_key)
            self.latest_offered_date = latest
            self.latest_offered_slot = extract_concrete_slot(raw)
            if len(dates) > 1:
                return PolicyDecision(
                    DecisionKind.CHOOSE_SEARCH_BRANCH,
                    "The later of those dates works for me. What times are available?",
                    "call6:choose_latest_grounded_date",
                )
            if not self.verification_asked:
                return PolicyDecision(DecisionKind.CLARIFY, VERIFICATION_TEXT, "call6:verify_latest_once")

        if self._asks_preferred_date(text):
            return PolicyDecision(DecisionKind.STATE_OBJECTIVE, DATE_PREFERENCE_RESPONSE, "call6:date_preference")
        if self._asks_preferred_time(text):
            return PolicyDecision(DecisionKind.STATE_OBJECTIVE, TIME_PREFERENCE_RESPONSE, "call6:time_preference")
        if self._cannot_search_absolute_latest(text) and not self.fallback_used:
            return PolicyDecision(DecisionKind.CLARIFY, FALLBACK_TEXT, "call6:latest_visible_once")

        if self.verification_asked and self._confirms_latest(text):
            if self.latest_offered_slot:
                return PolicyDecision(
                    DecisionKind.GRANT_PERMISSION,
                    BOOKING_ACCEPTANCE,
                    "compatible_concrete_slot_offered",
                )
            return PolicyDecision(DecisionKind.GRANT_PERMISSION, BOOKING_ACCEPTANCE, "call6:accept_date")
        return decision

    def decide_before_shared_policy(
        self,
        source_turns: tuple[str, ...],
        snapshot: FlowSnapshot,
    ) -> PolicyDecision | None:
        """Own Call #6 choice offers before Friday-oriented shared policy."""
        del snapshot
        raw = " ".join(source_turns)
        text = " ".join(raw.casefold().split())
        if self._offers_later_date_search(text):
            return PolicyDecision(
                DecisionKind.CHOOSE_SEARCH_BRANCH,
                CONTINUE_LATEST_SEARCH_RESPONSE,
                "call6:continue_latest_search",
            )
        if self._is_latest_final_slot_offer(text, raw):
            self.latest_offered_slot = extract_concrete_slot(raw)
            return PolicyDecision(
                DecisionKind.GRANT_PERMISSION,
                "Yes, please book the offered slot.",
                "compatible_concrete_slot_offered",
            )
        return None

    def mark_decision_spoken(self, decision: PolicyDecision) -> None:
        if decision.reason == "call6:objective":
            self.objective_stated = True
        elif decision.reason == "call6:verify_latest_once":
            self.verification_asked = True
        elif decision.reason == "call6:latest_visible_once":
            self.fallback_used = True
        elif decision.reason in {"compatible_concrete_slot_offered", "call6:accept_date"}:
            self.booking_accepted = True

    def mark_decision_suppressed(self, decision: PolicyDecision) -> None:
        del decision

    @property
    def objective_complete(self) -> bool:
        return self.booking_accepted

    def metadata(self) -> dict[str, object]:
        return {
            "scenario": self.scenario_id,
            "stable_scheduler_used": True,
            "objective_stated": self.objective_stated,
            "verification_asked": self.verification_asked,
            "fallback_used": self.fallback_used,
            "latest_offered_date": self.latest_offered_date,
            "booking_accepted": self.booking_accepted,
            "objective_complete": self.objective_complete,
        }

    def grounded_slot_for_acceptance(self) -> str | None:
        """Return the slot retained from the target's verified offer."""
        return self.latest_offered_slot

    @staticmethod
    def _asks_preferred_date(text: str) -> bool:
        return bool(re.search(r"\b(?:what|which|preferred|prefer)\b[^?.!]{0,45}\b(?:date|day)\b", text))

    @staticmethod
    def _asks_first_visit(text: str) -> bool:
        return bool(
            re.search(r"\bfirst\s+time\s+visiting\b|\bfirst\s+visit\b|\bestablish\s+care\b", text)
        )

    @staticmethod
    def _offers_keep_reschedule_cancel(text: str) -> bool:
        return all(choice in text for choice in ("keep", "reschedule", "cancel"))

    @staticmethod
    def _asks_to_reschedule_existing_appointment(text: str) -> bool:
        return (
            "appointment" in text
            and "reschedul" in text
            and bool(re.search(r"\b(?:is|would|do)\b", text))
        ) or (
            "appointment" in text
            and "move" in text
            and "later date" in text
            and bool(re.search(r"\b(?:would|do)\b", text))
        )

    @staticmethod
    def _offers_alternative_search_or_transfer(text: str) -> bool:
        different_day_or_time = bool(
            re.search(r"\b(?:different|another)\s+(?:day|time)\b", text)
            or re.search(r"\bother\s+(?:days?|times?|openings?)\b", text)
        )
        search_question = "?" in text or bool(
            re.search(r"\b(?:would|do) you (?:like|want)\b", text)
        )
        return different_day_or_time and search_question

    @staticmethod
    def _offers_later_date_search(text: str) -> bool:
        search_question = "?" in text or bool(
            re.search(r"\b(?:would|do) you (?:like|want|prefer)\b", text)
        )
        later_date = bool(
            re.search(r"\b(?:later|another)\s+date\b", text)
            or re.search(r"\b(?:further|farther)\b[^?.!]{0,35}\b(?:future|out|date|dates)\b", text)
            or re.search(r"\bdates?\s+(?:further|farther)\s+out\b", text)
        )
        explicit_search_choice = bool(
            re.search(r"\b(?:look|try|search|check)\b[^?.!]{0,55}\b(?:later|further|farther|future)\b", text)
            or re.search(r"\bprefer\b[^?.!]{0,55}\b(?:later|further|farther|future)\b", text)
            or (
                "one of these times" in text
                and bool(re.search(r"\bor\b[^?.!]{0,55}\b(?:later|another)\s+date\b", text))
            )
            or (
                bool(re.search(r"\b(?:openings?|times?)\b", text))
                and bool(re.search(r"\bor\b[^?.!]{0,55}\b(?:later|another)\s+date\b", text))
            )
        )
        return search_question and later_date and explicit_search_choice

    @staticmethod
    def _is_latest_final_slot_offer(text: str, raw: str) -> bool:
        asserts_latest = bool(
            re.search(
                r"\b(?:latest|furthest|farthest)\s+date\b[^.!?]{0,55}\b(?:can|could)\s+(?:currently\s+)?book\b",
                text,
            )
        )
        offers_transfer = bool(
            re.search(r"\b(?:speak|talk|transfer(?:red)?)\b[^.!?]{0,55}\b(?:clinic|someone|person|options?)\b", text)
        )
        return asserts_latest and offers_transfer and extract_concrete_slot(raw) is not None

    @staticmethod
    def _is_generic_help_prompt(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:how may i help|what would you like to do(?: next)?|"
                r"i can help with appointments\b)",
                text,
            )
        )

    @staticmethod
    def _asks_preferred_time(text: str) -> bool:
        return bool(re.search(r"\b(?:what|which|preferred|prefer)\b[^?.!]{0,45}\btime\b|morning or afternoon", text))

    @staticmethod
    def _cannot_search_absolute_latest(text: str) -> bool:
        return bool(re.search(r"\b(?:can't|cannot|unable|not able)\b[^.!?]{0,80}\b(?:latest|furthest|farthest|absolute)\b", text))

    @staticmethod
    def _confirms_latest(text: str) -> bool:
        return not re.search(r"\b(?:no|not)\b", text) and bool(re.search(r"\b(?:yes|correct|right|latest|furthest|farthest)\b", text))

    @staticmethod
    def _offered_dates(text: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(match.group(0) for match in _DATE_RE.finditer(text)))

    @staticmethod
    def _date_key(value: str) -> datetime:
        normalized = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", value.casefold()).replace(",", "")
        for fmt in ("%B %d %Y", "%B %d"):
            try:
                parsed = datetime.strptime(normalized, fmt)
                return parsed.replace(year=parsed.year if "%Y" in fmt else 2000)
            except ValueError:
                continue
        return datetime.min


def farthest_date_phrase_inventory() -> tuple[str, ...]:
    """Exact bounded phrases the current deterministic Call #6 can speak."""
    return tuple(sorted({
        "Chitragupta Subramnian Singh.",
        "Yes, please. My name is Chitragupta Subramnian Singh.",
        "A new patient consultation.",
        "First available is fine.",
        OBJECTIVE_TEXT,
        FIRST_VISIT_RESPONSE,
        RESCHEDULE_RESPONSE,
        REASON_RESPONSE,
        DATE_PREFERENCE_RESPONSE,
        TIME_PREFERENCE_RESPONSE,
        VERIFICATION_TEXT,
        FALLBACK_TEXT,
        BOOKING_ACCEPTANCE,
        RESCHEDULE_CONFIRMATION_RESPONSE,
        ALTERNATIVE_SEARCH_RESPONSE,
        CONTINUE_LATEST_SEARCH_RESPONSE,
        "The later of those dates works for me. What times are available?",
        # The current stable Call #6 scheduler offers this fixed three-slot
        # choice. The controller preserves the selected source spelling.
        "Yes, please book the ten thirty AM slot.",
        "Yes, please book the eleven fifteen AM slot.",
        "Yes, please book the twelve PM slot.",
    }))
