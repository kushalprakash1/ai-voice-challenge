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

_END_RE = re.compile(
    r"(?:"
    r"\bgoodbye\b|"
    r"\bbye\b|"
    r"\bhave a (?:good|nice) day\b|"
    r"\btake care\b"
    r")"
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
    r"i need your|"
    r"we need your|"
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

    return tuple(facts)


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
    # 3. Explicit workflow-permission grammar.
    # --------------------------------------------------------------
    if _PERMISSION_RE.search(text) is not None:
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

    # Everything else remains an LLM responsibility.
    return None
