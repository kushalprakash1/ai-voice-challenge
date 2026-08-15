"""Deterministic cleanup of semantic interpreter output.

The semantic model may represent scheduling details in more than one
structurally valid way. This module canonicalizes those representations
before grounding and PatientBrain reasoning.
"""

from __future__ import annotations

import re

from voiceprobe.conversation.meaning import (
    AppointmentOffer,
    FactAssertion,
    TurnMeaning,
)
from voiceprobe.conversation.scheduling import (
    time_matches_preference,
)
from voiceprobe.conversation.state import FactKey


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _first_asserted_value(
    meaning: TurnMeaning,
    fact: FactKey,
) -> str | None:
    for assertion in meaning.stated_facts:
        if assertion.fact == fact:
            return assertion.value

    return None


# Narrow recognition of a real telephony-ASR corruption observed in
# testing: "you're booked for ..." -> "your book for ...".
#
# Unlike _BOOKING_CONFIRMATION_PATTERNS, this signal is NOT sufficient on
# its own. It may only recover a confirmation when deterministic session
# state proves that the extracted slot matches a slot the patient already
# accepted.
_ASR_BOOKING_CONFIRMATION_PATTERNS = (
    re.compile(
        r"\byour\s+book\s+for\b",
        re.IGNORECASE,
    ),
)


_BOOKING_CONFIRMATION_PATTERNS = (
    re.compile(
        r"\b(?:you're|you are|you've been|you have been)\s+"
        r"(?:now\s+)?(?:booked|scheduled|confirmed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:you're|you are)\s+all\s+set\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi(?:'ve| have)\s+(?:got\s+)?you\s+"
        r"(?:booked|scheduled|confirmed|down)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi\s+have\s+you\s+down\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\byour\s+appointment\s+"
        r"(?:is|has been)\s+"
        r"(?:booked|scheduled|confirmed)\b",
        re.IGNORECASE,
    ),
)


def _recover_explicit_booking_confirmation(
    meaning: TurnMeaning,
    *,
    agent_turn: str | None,
) -> TurnMeaning:
    """Recover only unmistakable booking-completion language."""
    if meaning.booking_confirmed:
        return meaning

    if agent_turn is None:
        return meaning

    normalized_turn = " ".join(agent_turn.replace("’", "'").split())

    if not any(
        pattern.search(normalized_turn) for pattern in _BOOKING_CONFIRMATION_PATTERNS
    ):
        return meaning

    return meaning.model_copy(
        update={
            "booking_confirmed": True,
        }
    )


def recover_asr_booking_confirmation(
    meaning: TurnMeaning,
    *,
    agent_turn: str | None,
    accepted_offer_matches: bool,
) -> TurnMeaning:
    """Recover a narrow ASR-corrupted confirmation using trusted state.

    The lexical corruption alone is never authoritative. Recovery is
    allowed only after deterministic session logic has established that
    the extracted slot matches a slot the patient already accepted.
    """
    if meaning.booking_confirmed:
        return meaning

    if not accepted_offer_matches:
        return meaning

    if meaning.appointment_offer is None:
        return meaning

    if agent_turn is None:
        return meaning

    normalized_turn = " ".join(agent_turn.replace("’", "'").split())

    if not any(
        pattern.search(normalized_turn)
        for pattern in _ASR_BOOKING_CONFIRMATION_PATTERNS
    ):
        return meaning

    return meaning.model_copy(
        update={
            "booking_confirmed": True,
        }
    )


def _promote_booking_details(
    meaning: TurnMeaning,
) -> TurnMeaning:
    """Recover booking-slot details stored as scheduling assertions."""
    if not meaning.booking_confirmed:
        return meaning

    if meaning.appointment_offer is not None:
        return meaning

    day = _first_asserted_value(
        meaning,
        "preferred_day",
    )
    time = _first_asserted_value(
        meaning,
        "preferred_time",
    )

    if day is None and time is None:
        return meaning

    return meaning.model_copy(
        update={
            "appointment_offer": AppointmentOffer(
                day=day,
                time=time,
            ),
        }
    )


def _matches_offered_day(
    assertion: FactAssertion,
    *,
    offered_day: str | None,
) -> bool:
    if assertion.fact != "preferred_day":
        return False

    if offered_day is None:
        return False

    return _normalize_text(assertion.value) == _normalize_text(offered_day)


def _matches_offered_time(
    assertion: FactAssertion,
    *,
    offered_time: str | None,
) -> bool:
    if assertion.fact != "preferred_time":
        return False

    if offered_time is None:
        return False

    return time_matches_preference(
        preferred=assertion.value,
        offered=offered_time,
    ) or time_matches_preference(
        preferred=offered_time,
        offered=assertion.value,
    )


def normalize_turn_meaning(
    meaning: TurnMeaning,
    *,
    agent_turn: str | None = None,
) -> TurnMeaning:
    """Canonicalize scheduling semantics before grounding."""
    normalized = _recover_explicit_booking_confirmation(
        meaning,
        agent_turn=agent_turn,
    )
    normalized = _promote_booking_details(normalized)

    offer = normalized.appointment_offer

    if offer is None:
        return normalized

    # An offered or confirmed scheduling slot is not automatically a
    # statement about what the patient prefers. Remove duplicate
    # scheduling assertions once those values are represented by the
    # canonical appointment_offer field.
    filtered_assertions = tuple(
        assertion
        for assertion in normalized.stated_facts
        if not (
            _matches_offered_day(
                assertion,
                offered_day=offer.day,
            )
            or _matches_offered_time(
                assertion,
                offered_time=offer.time,
            )
        )
    )

    if filtered_assertions == normalized.stated_facts:
        return normalized

    return normalized.model_copy(
        update={
            "stated_facts": filtered_assertions,
        }
    )
