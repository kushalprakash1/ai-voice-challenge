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


_STANDALONE_DAY_REPLY = re.compile(
    r"^(?:on\s+)?"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"\s*[.!?]?$",
    re.IGNORECASE,
)

_STANDALONE_TIME_REPLY = re.compile(
    r"^(?:at\s+)?"
    r"(?P<hour>1[0-2]|0?[1-9])"
    r"(?::(?P<minute>[0-5]\d))?"
    r"\s*"
    r"(?P<meridiem>a\.?\s*m\.?|p\.?\s*m\.?)"
    r"\s*[.!?]?$",
    re.IGNORECASE,
)

_STANDALONE_DAYPART_REPLY = re.compile(
    r"^(?:in\s+the\s+)?"
    r"(morning|afternoon|evening)"
    r"\s*[.!?]?$",
    re.IGNORECASE,
)


def _standalone_day_reply(agent_turn: str) -> str | None:
    normalized = " ".join(agent_turn.split())
    match = _STANDALONE_DAY_REPLY.fullmatch(normalized)

    if match is None:
        return None

    return match.group(1).title()


def _standalone_time_reply(agent_turn: str) -> str | None:
    normalized = " ".join(agent_turn.split())

    time_match = _STANDALONE_TIME_REPLY.fullmatch(normalized)
    if time_match is not None:
        hour = str(int(time_match.group("hour")))
        minute = time_match.group("minute")
        meridiem = (
            "AM" if time_match.group("meridiem").casefold().startswith("a") else "PM"
        )

        if minute is None:
            return f"{hour} {meridiem}"

        return f"{hour}:{minute} {meridiem}"

    daypart_match = _STANDALONE_DAYPART_REPLY.fullmatch(normalized)
    if daypart_match is not None:
        return daypart_match.group(1).casefold()

    return None


def _complete_pending_appointment_offer(
    meaning: TurnMeaning,
    *,
    agent_turn: str | None,
    pending_offer: AppointmentOffer | None,
) -> TurnMeaning:
    """Complete one trusted partial slot from one narrow elliptical reply."""
    if pending_offer is None:
        return meaning

    # Context completion is valid only when exactly one slot dimension is
    # already known. A complete or completely empty pending slot is not
    # elliptical context.
    pending_has_day = pending_offer.day is not None
    pending_has_time = pending_offer.time is not None

    if pending_has_day == pending_has_time:
        return meaning

    if meaning.conversation_end_requested or meaning.requests_repetition:
        return meaning

    offer = meaning.appointment_offer

    # First prefer structured semantics if the interpreter already extracted
    # the complementary dimension.
    if offer is not None:
        if offer.day is not None and offer.time is not None:
            return meaning

        if (
            pending_offer.day is None
            and pending_offer.time is not None
            and offer.day is not None
            and offer.time is None
        ):
            return meaning.model_copy(
                update={
                    "appointment_offer": AppointmentOffer(
                        day=offer.day,
                        time=pending_offer.time,
                    ),
                    "unclear": False,
                }
            )

        if (
            pending_offer.time is None
            and pending_offer.day is not None
            and offer.time is not None
            and offer.day is None
        ):
            return meaning.model_copy(
                update={
                    "appointment_offer": AppointmentOffer(
                        day=pending_offer.day,
                        time=offer.time,
                    ),
                    "unclear": False,
                }
            )

        # A partial semantic offer that does not supply the missing dimension
        # represents a new/changed partial offer, not completion of the old one.
        return meaning

    if agent_turn is None:
        return meaning

    # The raw-text fallback is intentionally narrow. It handles only a
    # standalone weekday, clock time, or daypart when trusted state proves
    # exactly which dimension is missing.
    if pending_offer.day is None:
        day = _standalone_day_reply(agent_turn)

        if day is None:
            return meaning

        completed_offer = AppointmentOffer(
            day=day,
            time=pending_offer.time,
        )

    else:
        time_value = _standalone_time_reply(agent_turn)

        if time_value is None:
            return meaning

        completed_offer = AppointmentOffer(
            day=pending_offer.day,
            time=time_value,
        )

    return meaning.model_copy(
        update={
            "appointment_offer": completed_offer,
            "unclear": False,
        }
    )


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
    pending_offer: AppointmentOffer | None = None,
) -> TurnMeaning:
    """Canonicalize scheduling semantics before grounding."""
    normalized = _recover_explicit_booking_confirmation(
        meaning,
        agent_turn=agent_turn,
    )
    normalized = _promote_booking_details(normalized)

    normalized = _complete_pending_appointment_offer(
        normalized,
        agent_turn=agent_turn,
        pending_offer=pending_offer,
    )

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
