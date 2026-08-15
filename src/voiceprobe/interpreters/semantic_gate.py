"""High-confidence deterministic semantic classification.

The gate handles speech acts that can be classified conservatively from
surface language without an LLM. Ambiguous language returns None and is
delegated to the normal Ollama semantic interpreter.

Important design rule:
The gate extracts what the tested agent said. It does not decide whether
the patient should cooperate. PatientBrain remains authoritative over the
scenario objective and appointment progress.
"""

from __future__ import annotations

import re

from voiceprobe.conversation.meaning import (
    AppointmentOffer,
    QuestionKind,
    ResponseExpectation,
    TurnMeaning,
    WorkflowDirection,
    WorkflowRelation,
)
from voiceprobe.scenarios.models import PatientScenario


def _normalize(text: str) -> str:
    """Normalize whitespace and case without destroying punctuation."""
    return " ".join(text.casefold().split())


# ------------------------------------------------------------------
# Workflow-permission grammar
# ------------------------------------------------------------------

_PERMISSION_RE = re.compile(
    r"\b(?:"
    r"would you like me to|"
    r"would you like to|"
    r"do you want me to|"
    r"should i|"
    r"shall i|"
    r"may i|"
    r"can i"
    r")\b"
)

_STOP_ACTION_RE = re.compile(
    r"\b(?:"
    r"cancel|"
    r"stop|"
    r"end|"
    r"terminate|"
    r"abandon|"
    r"discontinue"
    r")\b"
)

_SCHEDULING_ACTION_RE = re.compile(
    r"\b(?:"
    r"schedule|"
    r"scheduling|"
    r"book|"
    r"booking|"
    r"reserve|"
    r"appointment|"
    r"appointments|"
    r"availability|"
    r"available slots?|"
    r"time slots?"
    r")\b"
)

_SIDE_WORKFLOW_RE = re.compile(
    r"\b(?:"
    r"profile|"
    r"account|"
    r"preferences|"
    r"registration|"
    r"register|"
    r"enroll|"
    r"enrollment|"
    r"demo setup|"
    r"temporary setup"
    r")\b"
)

_REQUIRED_RE = re.compile(
    r"\b(?:"
    r"before i can|"
    r"before we can|"
    r"need to|"
    r"have to|"
    r"must|"
    r"required|"
    r"necessary|"
    r"in order to|"
    r"to continue|"
    r"so i can|"
    r"so we can"
    r")\b"
)

_GENERIC_CONTINUE_RE = re.compile(
    r"\b(?:continue|proceed|go ahead|move forward)\b"
)

_SCHEDULING_OBJECTIVE_RE = re.compile(
    r"\b(?:schedule|scheduling|appointment|book|booking)\b"
)


# ------------------------------------------------------------------
# Appointment-slot grammar
# ------------------------------------------------------------------

_DAY_RE = re.compile(
    r"\b("
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r")\b"
)

_TIME_RE = re.compile(
    r"\b"
    r"(?P<hour>0?[1-9]|1[0-2])"
    r"(?:[:.](?P<minute>[0-5]\d))?"
    r"\s*"
    r"(?P<meridiem>a\.?m\.?|p\.?m\.?)"
    r"(?=\s|[?.!,;:]|$)"
)

_COMPACT_TIME_RE = re.compile(
    r"\b"
    r"(?P<compact>[1-9]\d{2}|1[0-2]\d{2})"
    r"\s*"
    r"(?P<meridiem>a\.?m\.?|p\.?m\.?)"
    r"(?=\s|[?.!,;:]|$)"
)

_OFFER_CUE_RE = re.compile(
    r"\b(?:"
    r"available|"
    r"availability|"
    r"how about|"
    r"i have|"
    r"we have|"
    r"i can get you in|"
    r"we can get you in|"
    r"can get you in|"
    r"fit you in|"
    r"opening|"
    r"open slot|"
    r"time slot|"
    r"would that work|"
    r"would this work|"
    r"would it work|"
    r"does that work|"
    r"does this work|"
    r"work for you|"
    r"come in"
    r")\b"
)

_BOOKING_CONFIRMATION_RE = re.compile(
    r"\b(?:"
    r"you(?:'re| are) (?:booked|scheduled|confirmed)|"
    r"you(?:'re| are) all set|"
    r"your .{0,80}? appointment "
    r"(?:is|has been) (?:booked|scheduled|confirmed)|"
    r"(?:the )?appointment "
    r"(?:is|has been) (?:booked|scheduled|confirmed)|"
    r"i(?:'ve| have) (?:booked|scheduled|confirmed) "
    r"(?:your|the) appointment"
    r")\b"
)


_DECLARATIVE_NAME_REQUEST_RE = re.compile(
    r"\b(?:"
    r"i|we"
    r")\s+"
    r"(?:just\s+)?"
    r"(?:need|require)\s+"
    r"(?:your\s+)?"
    r"(?:"
    r"first\s+and\s+last\s+name|"
    r"full\s+name|"
    r"name"
    r")\b"
)


_VALUE_ENTRY_QUESTION_RE = re.compile(
    r"\b(?:"
    r"what|which"
    r")\s+"
    r"(?:value\s+)?"
    r"should\s+i\s+"
    r"(?:"
    r"enter|"
    r"put(?:\s+down)?|"
    r"use|"
    r"record|"
    r"write(?:\s+down)?|"
    r"list"
    r")\b"
)

_END_RE = re.compile(
    r"(?:"
    r"\bgoodbye\b|"
    r"\bbye\b|"
    r"\bhave a (?:good|great|nice|wonderful) day\b|"
    r"\btake care\b"
    r")"
)



# Open-ended receptionist prompts have an obvious response:
# state the immutable scheduling objective.
_OPEN_GOAL_RE = re.compile(
    r"\b(?:"
    r"how may i help you(?: today)?|"
    r"how can i help you(?: today)?|"
    r"what can i help you with|"
    r"what would you like help with|"
    r"what would you like to (?:do|try|ask)"
    r"(?: or ask)?(?: about)?(?: next)?"
    r")\b"
)


# Legal notices, language-selection prompts, and greetings do not
# require a patient response.
_NON_ACTIONABLE_RE = re.compile(
    r"\b(?:"
    r"this call may be recorded|"
    r"call may be recorded|"
    r"para espa(?:ñ|n)ol|"
    r"thank you for calling"
    r")\b"
)


def _normalized_clock_time(match: re.Match[str]) -> str:
    """Return a stable human-readable clock representation."""
    hour = int(match.group("hour"))
    minute = match.group("minute")
    meridiem = match.group("meridiem").replace(".", "").upper()

    if minute is None:
        return f"{hour} {meridiem}"

    return f"{hour}:{minute} {meridiem}"


def _normalized_compact_time(match: re.Match[str]) -> str:
    """Normalize forms such as 230 PM or 1130 AM."""
    compact = match.group("compact")
    meridiem = match.group("meridiem").replace(".", "").upper()

    if len(compact) == 3:
        hour = int(compact[0])
        minute = compact[1:]
    else:
        hour = int(compact[:2])
        minute = compact[2:]

    if not 1 <= hour <= 12:
        raise ValueError("Compact clock hour is outside 12-hour range.")

    return f"{hour}:{minute} {meridiem}"


def _extract_concrete_slot(
    text: str,
) -> tuple[str, str] | None:
    """Extract one unambiguous weekday plus one clock time.

    Multiple distinct days or times are intentionally rejected so the LLM
    fallback can interpret more complicated alternatives.
    """
    days = {
        match.group(1).capitalize()
        for match in _DAY_RE.finditer(text)
    }

    times = {
        _normalized_clock_time(match)
        for match in _TIME_RE.finditer(text)
    }

    if not times:
        for match in _COMPACT_TIME_RE.finditer(text):
            try:
                times.add(_normalized_compact_time(match))
            except ValueError:
                return None

    if len(days) != 1 or len(times) != 1:
        return None

    return next(iter(days)), next(iter(times))


# ------------------------------------------------------------------
# Patient-fact request grammar
# ------------------------------------------------------------------

_FACT_REQUEST_CUE_RE = re.compile(
    r"\b(?:"
    r"what(?:'s| is) your|"
    r"what are your|"
    r"can you (?:confirm|provide|state|repeat|tell me)|"
    r"could you (?:confirm|provide|state|repeat|tell me)|"
    r"please (?:confirm|provide|state|repeat|tell me)|"
    r"confirm your|"
    r"verify your|"
    r"provide your|"
    r"state your|"
    r"repeat your|"
    r"tell me your|"
    r"i (?:just )?need your|"
    r"we (?:just )?need your|"
    r"i need to verify your|"
    r"we need to verify your|"
    r"i need to confirm your|"
    r"we need to confirm your|"
    r"who am i speaking with|"
    r"who are you covered through|"
    r"which day works|"
    r"what day works|"
    r"which time works|"
    r"what time works|"
    r"how long|"
    r"when did|"
    r"what brought you in|"
    r"what brings you in|"
    r"reason for (?:the )?(?:visit|call|calling)"
    r")\b"
)

_FACT_MENTION_PATTERNS: tuple[
    tuple[str, re.Pattern[str]],
    ...
] = (
    (
        "first_name",
        re.compile(
            r"\bfirst\s+name\b"
        ),
    ),
    (
        "last_name",
        re.compile(
            r"\b(?:last\s+name|surname|family\s+name)\b"
        ),
    ),
    (
        "name",
        re.compile(
            r"\b(?:"
            r"name|"
            r"full name|"
            r"who am i speaking with"
            r")\b"
        ),
    ),
    (
        "complaint",
        re.compile(
            r"\b(?:"
            r"what brought you in|"
            r"what brings you in|"
            r"reason for (?:the )?(?:visit|call|calling)|"
            r"symptoms?|"
            r"medical problem"
            r")\b"
        ),
    ),
    (
        "duration",
        re.compile(
            r"\b(?:"
            r"how long|"
            r"when did .{0,35}(?:start|begin)"
            r")\b"
        ),
    ),
    (
        "date_of_birth",
        re.compile(
            r"\b(?:"
            r"date of birth|"
            r"dob|"
            r"birthday"
            r")\b"
        ),
    ),
    (
        "insurance",
        re.compile(
            r"\b(?:"
            r"insurance|"
            r"coverage|"
            r"insurer|"
            r"carrier|"
            r"covered through"
            r")\b"
        ),
    ),
    (
        "preferred_day",
        re.compile(
            r"\b(?:"
            r"preferred day|"
            r"preferred date|"
            r"which day works|"
            r"what day works"
            r")\b"
        ),
    ),
    (
        "preferred_time",
        re.compile(
            r"\b(?:"
            r"preferred time|"
            r"which time works|"
            r"what time works|"
            r"time of day"
            r")\b"
        ),
    ),
)


def _requested_facts(text: str) -> tuple[str, ...]:
    """Return supported patient facts explicitly being requested."""
    if _FACT_REQUEST_CUE_RE.search(text) is None:
        return ()

    facts: list[str] = []

    for fact, pattern in _FACT_MENTION_PATTERNS:
        if pattern.search(text) is not None:
            facts.append(fact)

    # "first name" and "last name" naturally contain the generic token "name".
    # Prefer the more precise semantic field rather than returning both.
    if "first_name" in facts or "last_name" in facts:
        facts = [
            fact
            for fact in facts
            if fact != "name"
        ]

    return tuple(dict.fromkeys(facts))


def _workflow_relation(
    *,
    scenario: PatientScenario,
    text: str,
    direction: WorkflowDirection,
) -> WorkflowRelation:
    """Relate an obvious workflow request to the scenario objective."""
    objective = _normalize(scenario.objective)

    scheduling_objective = (
        _SCHEDULING_OBJECTIVE_RE.search(objective) is not None
    )

    if direction is WorkflowDirection.STOP:
        if scheduling_objective:
            return WorkflowRelation.OPPOSES_OBJECTIVE

        return WorkflowRelation.UNCERTAIN

    side_workflow = _SIDE_WORKFLOW_RE.search(text) is not None
    required = _REQUIRED_RE.search(text) is not None
    scheduling_context = _SCHEDULING_ACTION_RE.search(text) is not None

    if side_workflow and not required:
        return WorkflowRelation.NONE

    if (
        scheduling_objective
        and scheduling_context
        and not side_workflow
    ):
        return WorkflowRelation.ADVANCES_OBJECTIVE

    if (
        scheduling_objective
        and scheduling_context
        and required
    ):
        return WorkflowRelation.ADVANCES_OBJECTIVE

    if _GENERIC_CONTINUE_RE.search(text) is not None:
        return WorkflowRelation.UNCERTAIN

    return WorkflowRelation.UNCERTAIN


def _routine_intake_fast_facts(
    text: str,
) -> tuple[str, ...]:
    """Resolve obvious routine intake slots without invoking an LLM."""
    text = _normalize(text)

    if not text:
        return ()

    request_like = (
        "?" in text
        or "need your" in text
        or "please provide" in text
        or "please tell me" in text
        or "what should i enter" in text
        or "what should i put" in text
        or "what should i record" in text
    )

    if not request_like:
        return ()

    # Provider/clinician preference is separate from patient identity.
    provider_question = (
        "provider" in text
        and (
            "which provider" in text
            or "what provider" in text
            or "provider would you like" in text
            or "provider do you prefer" in text
            or "provider you prefer" in text
            or "provider preference" in text
            or "name of the provider" in text
            or "available provider" in text
            or "any provider" in text
            or "open to" in text
        )
    )

    if provider_question:
        return ("provider_preference",)

    # Appointment type must outrank generic "new patient" wording.
    if (
        "type of appointment" in text
        or "appointment type" in text
        or (
            "new patient consultation" in text
            and (
                "follow-up" in text
                or "general office visit" in text
                or "something else" in text
            )
        )
        or (
            "routine checkup" in text
            and (
                "follow-up" in text
                or "urgent concern" in text
                or "specific procedure" in text
            )
        )
    ):
        return ("appointment_type",)

    # TRUNCATED FIRST-NAME ASR FAST PATH
    #
    # Streaming ASR can finalize:
    #
    #     "I just need your first"
    #
    # before the remote speaker's trailing "name" is retained.
    # Keep this deliberately narrow: only an utterance ending in
    # "need your first" is interpreted as a first-name request.
    truncated_first_name_request = (
        re.search(
            r"\bneed\s+your\s+first[\s.!?]*$",
            text,
        )
        is not None
    )

    if truncated_first_name_request:
        return ("first_name",)

    combined_name = (
        "first and last name" in text
        or "first & last name" in text
        or "first and last names" in text
    )

    first_name = (
        "first name" in text
        or combined_name
    )

    last_name = (
        "last name" in text
        or "surname" in text
        or "family name" in text
        or combined_name
    )

    if first_name and last_name:
        return (
            "first_name",
            "last_name",
        )

    if first_name:
        return ("first_name",)

    if last_name:
        return ("last_name",)

    asks_visited_before = (
        "visited us before" in text
        or "visited before" in text
        or "been here before" in text
        or "seen us before" in text
        or "seen you before" in text
    )

    asks_patient_status = (
        "are you a patient" in text
        or "new patient" in text
        or "existing patient" in text
        or "returning patient" in text
    )

    if asks_patient_status or asks_visited_before:
        facts: list[str] = []

        if asks_patient_status:
            facts.append("patient_status")

        if asks_visited_before:
            facts.append("visited_before")

        return tuple(facts)

    return ()


def deterministic_turn_meaning(
    *,
    scenario: PatientScenario,
    agent_turn: str,
) -> TurnMeaning | None:
    """Return high-confidence meaning, or None to delegate to Ollama."""
    text = _normalize(agent_turn)

    if not text:
        return None

    end_requested = _END_RE.search(text) is not None
    slot = _extract_concrete_slot(text)

    # --------------------------------------------------------------
    # 1. Explicit booking confirmations.
    #
    # Confirmation is checked before generic slot offers because both
    # may contain the same weekday/time tokens.
    # --------------------------------------------------------------
    if _BOOKING_CONFIRMATION_RE.search(text) is not None:
        offer = (
            AppointmentOffer(
                day=slot[0],
                time=slot[1],
            )
            if slot is not None
            else None
        )

        return TurnMeaning(
            response_expectation=ResponseExpectation.NONE,
            workflow_relation=WorkflowRelation.ADVANCES_OBJECTIVE,
            question_kind=QuestionKind.NONE,
            workflow_direction=WorkflowDirection.NONE,
            topic="booking confirmation",
            requested_facts=(),
            appointment_offer=offer,
            booking_confirmed=True,
            conversation_end_requested=end_requested,
        )

    # --------------------------------------------------------------
    # 2. Concrete appointment offers.
    # --------------------------------------------------------------
    if (
        slot is not None
        and _OFFER_CUE_RE.search(text) is not None
    ):
        return TurnMeaning(
            response_expectation=ResponseExpectation.YES_NO,
            workflow_relation=WorkflowRelation.ADVANCES_OBJECTIVE,
            question_kind=QuestionKind.WORKFLOW_PERMISSION,
            workflow_direction=WorkflowDirection.CONTINUE,
            topic="appointment offer",
            requested_facts=(),
            appointment_offer=AppointmentOffer(
                day=slot[0],
                time=slot[1],
            ),
            conversation_end_requested=end_requested,
        )

    # --------------------------------------------------------------
    # 3. Open-ended call-purpose prompt.
    #
    # "How may I help you?" is not ambiguous. The patient should state
    # the immutable scheduling objective.
    # --------------------------------------------------------------
    if _OPEN_GOAL_RE.search(text) is not None:
        return TurnMeaning(
            response_expectation=ResponseExpectation.FREEFORM,
            workflow_relation=WorkflowRelation.ADVANCES_OBJECTIVE,
            question_kind=QuestionKind.OTHER,
            workflow_direction=WorkflowDirection.NONE,
            topic="call purpose",
            requested_facts=(),
            conversation_end_requested=end_requested,
        )

    # --------------------------------------------------------------
    # 4. Explicit side workflow.
    #
    # This is intentionally evaluated before fact extraction so:
    #
    # "I can create a demo patient profile. I need your name."
    #
    # cannot disclose the patient's name into the unwanted workflow.
    # --------------------------------------------------------------
    # A value-entry question asks the caller for information.
    #
    # Example:
    # "I just need your first name. What should I enter for you?"
    #
    # Although "should I" appears in the sentence, this is not permission.
    # It asks the caller which factual value the remote agent should enter.
    value_entry_facts = _requested_facts(text)

    if (
        value_entry_facts
        and _VALUE_ENTRY_QUESTION_RE.search(text) is not None
    ):
        return TurnMeaning(
            response_expectation=ResponseExpectation.FACT,
            workflow_relation=WorkflowRelation.NONE,
            question_kind=QuestionKind.PATIENT_ATTRIBUTE,
            workflow_direction=WorkflowDirection.NONE,
            topic="patient information",
            requested_facts=value_entry_facts,
            conversation_end_requested=end_requested,
        )

    side_workflow = _SIDE_WORKFLOW_RE.search(text) is not None
    required = _REQUIRED_RE.search(text) is not None
    scheduling_context = (
        _SCHEDULING_ACTION_RE.search(text) is not None
    )

    if (
        side_workflow
        and not (
            required
            and scheduling_context
        )
    ):
        return TurnMeaning(
            response_expectation=ResponseExpectation.YES_NO,
            workflow_relation=WorkflowRelation.NONE,
            question_kind=QuestionKind.WORKFLOW_PERMISSION,
            workflow_direction=WorkflowDirection.CONTINUE,
            topic="side workflow",
            requested_facts=(),
            conversation_end_requested=end_requested,
        )

    # --------------------------------------------------------------
    # 5. Explicit workflow-permission grammar.
    # --------------------------------------------------------------
    # DAYPART-ONLY APPOINTMENT OFFER
    #
    # Telephony ASR may finalize the day and the offered daypart
    # as separate turns. A phrase such as "book a morning slot"
    # is still a high-confidence appointment offer and must be
    # compared against the patient's authoritative preference.
    daypart_offer = next(
        (
            daypart
            for daypart in (
                "morning",
                "afternoon",
                "evening",
            )
            if re.search(
                rf"\b{daypart}\b",
                text,
            )
            is not None
        ),
        None,
    )

    scheduling_offer_language = (
        "slot" in text
        or "appointment" in text
    )

    # A daypart becomes an appointment offer only when the remote
    # agent is actually proposing to book/reserve/schedule it.
    #
    # "Would you like me to check Friday afternoon appointments?"
    # is permission to search and must remain a normal workflow
    # permission, not an appointment offer.
    booking_action = (
        re.search(
            r"\b(?:book|booking|schedule|scheduling|reserve|reserving)\b",
            text,
        )
        is not None
    )

    if (
        slot is None
        and daypart_offer is not None
        and scheduling_offer_language
        and booking_action
    ):
        return TurnMeaning(
            response_expectation=ResponseExpectation.YES_NO,
            workflow_relation=WorkflowRelation.ADVANCES_OBJECTIVE,
            question_kind=QuestionKind.WORKFLOW_PERMISSION,
            workflow_direction=WorkflowDirection.CONTINUE,
            topic="appointment offer",
            requested_facts=(),
            appointment_offer=AppointmentOffer(
                day=None,
                time=daypart_offer,
            ),
            conversation_end_requested=end_requested,
        )

    if (
        _PERMISSION_RE.search(text) is not None
        and _routine_intake_fast_facts(text)
        != ("provider_preference",)
    ):
        direction = (
            WorkflowDirection.STOP
            if _STOP_ACTION_RE.search(text) is not None
            else WorkflowDirection.CONTINUE
        )

        relation = _workflow_relation(
            scenario=scenario,
            text=text,
            direction=direction,
        )

        return TurnMeaning(
            response_expectation=ResponseExpectation.YES_NO,
            workflow_relation=relation,
            question_kind=QuestionKind.WORKFLOW_PERMISSION,
            workflow_direction=direction,
            topic="workflow permission",
            requested_facts=(),
            conversation_end_requested=end_requested,
        )

    # --------------------------------------------------------------
    # 4. High-confidence supported patient-fact requests.
    # --------------------------------------------------------------
    facts = _routine_intake_fast_facts(text)

    if not facts:
        if _DECLARATIVE_NAME_REQUEST_RE.search(text) is not None:
            facts = ("name",)
        else:
            facts = _requested_facts(text)

    if facts:
        return TurnMeaning(
            response_expectation=ResponseExpectation.FACT,
            workflow_relation=WorkflowRelation.NONE,
            question_kind=QuestionKind.PATIENT_ATTRIBUTE,
            workflow_direction=WorkflowDirection.NONE,
            topic="patient information",
            requested_facts=facts,
            conversation_end_requested=end_requested,
        )

    # --------------------------------------------------------------
    # 5. Plain conversation termination.
    #
    # The gate records the termination signal only. PatientBrain decides
    # whether objective state permits ending the conversation.
    # --------------------------------------------------------------
    if end_requested:
        return TurnMeaning(
            response_expectation=ResponseExpectation.NONE,
            workflow_relation=WorkflowRelation.NONE,
            question_kind=QuestionKind.NONE,
            workflow_direction=WorkflowDirection.NONE,
            topic="conversation ending",
            requested_facts=(),
            conversation_end_requested=True,
        )

    # Non-actionable greeting / legal / language-selection
    # speech belongs in history but does not require a patient reply.
    if _NON_ACTIONABLE_RE.search(text) is not None:
        return TurnMeaning()

    # Everything genuinely unknown remains an LLM responsibility.
    return None
