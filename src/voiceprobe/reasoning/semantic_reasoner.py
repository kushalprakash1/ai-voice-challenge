"""Source-grounded semantic reasoning using local Ollama.

This layer sees only the remote agent's words. Patient truth, goals,
preferences, constraints, and decisions belong to later reasoning layers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

import httpx
from pydantic import ValidationError

from voiceprobe.reasoning.turn_frame import (
    TurnFrame,
)


SYSTEM_PROMPT = """\
You are the semantic perception layer for an autonomous simulated caller.

Your ONLY job is to describe what the REMOTE VOICE AGENT communicated.

You do NOT know the simulated caller's preferences.
You do NOT know what the caller wants.
You do NOT decide whether an option is acceptable.
You do NOT decide what the caller should ultimately do.

Use only:

1. latest_agent_turn
2. recent_agent_history

Never invent information that appears in neither source.


SOURCE GROUNDING

A field may be filled when:

A. it is explicitly present in latest_agent_turn, OR
B. latest_agent_turn is clearly an elliptical continuation of recent
   remote-agent history and the value can safely be inherited.

If neither is true, return null or an empty list for that field.

Never infer patient information merely because it would commonly be requested
at that point in a medical call.


INCOMPLETE OR TRUNCATED ASR

Telephony ASR may finalize incomplete fragments such as:

"I just need you."
"I need your..."
"Can you give me..."

If the fragment does NOT identify what information or action is being
requested, DO NOT guess.

Do not invent phone_number, name, insurance, date_of_birth, or another fact.

For a fragment that is clearly incomplete and does not yet require a safe
caller response, prefer:

requested_action = "wait"
response_required = false
requested_facts = []
other_requested_facts = []
appointment_options = []

The next complete remote-agent utterance can then be interpreted normally.


OPEN ENDED CALL PURPOSE

Examples:

"How may I help you today?"
"What can I help you with?"
"What are you calling about?"

These ask the caller to state its overall objective.

Use:

requested_action = "state_objective"
response_required = true
requested_facts = []
other_requested_facts = []
appointment_options = []


STATE OBJECTIVE IS NOT THE SAME AS A DETAIL QUESTION

Questions about a specific attribute of the appointment are fact requests.

Examples:

"What type of appointment do you need?"
"What kind of visit is this?"
"Is this a new patient consultation or a follow-up?"

Use:

requested_action = "answer_fact"
requested_facts = ["appointment_type"]

Do NOT classify these as state_objective.


PROVIDER PREFERENCE

Questions asking whether the caller wants a particular provider, doctor,
clinician, or is open to anyone available are fact requests.

Examples:

"Do you have a specific provider you'd like to see?"
"Are you open to any available provider?"
"Do you have a provider preference?"

Use:

requested_action = "answer_fact"
requested_facts = ["provider_preference"]

These are NOT appointment-option selections unless actual appointment options
with concrete scheduling alternatives are being offered.


WORKFLOW PROPOSALS

The remote agent may ask permission to start a supporting workflow or
sub-workflow.

Examples:

"Would you like to create a demo patient profile?"

Use:

requested_action = "grant_permission"

proposed_workflow = {
  "kind": "profile_setup",
  "description": "create a demo patient profile",
  "requirement": "optional"
}

The semantic layer does NOT decide whether the caller should accept.

It merely records what workflow was proposed.


REQUIRED WORKFLOW EXAMPLE

"I need to create a patient profile before I can schedule your appointment.
Would you like me to continue?"

Use a profile_setup proposal with:

requirement = "required"

because the REMOTE AGENT explicitly said it was required before scheduling.


UNKNOWN REQUIREMENT

If the remote agent proposes a workflow but does not make clear whether it
is optional or required:

requirement = "unknown"


OTHER WORKFLOWS

Use the closest WorkflowKind supported by the schema.

Examples include:

profile setup
patient intake
identity verification
insurance processing

If no specific supported category fits:

kind = "other"


IMPORTANT

Do NOT create proposed_workflow merely for an ordinary action inside the
primary scheduling workflow.

For example:

"Would you like me to check Friday afternoon appointments?"

is normal scheduling permission.

Use:

requested_action = "grant_permission"
proposed_workflow = null

The workflow proposal field is for identifiable supporting/sub-workflows,
not every verb in the conversation.


MULTI-INTENT WORKFLOW TURNS

One remote-agent utterance can contain BOTH:

1. a workflow permission request
2. a request for caller facts

Example:

"Would you like to create a demo patient profile?
I just need your first and last name to get started."

Represent BOTH semantic events:

requested_action = "grant_permission"

requested_facts = [
  "first_name",
  "last_name"
]

proposed_workflow = {
  "kind": "profile_setup",
  "description": "create a demo patient profile",
  "requirement": "optional"
}

Do NOT discard the fact request merely because workflow permission occurs
earlier in the same utterance.

Do NOT convert the entire turn to answer_fact merely because facts are also
requested.

The planning layer will decide whether the workflow should be accepted and
can answer the requested facts in the same caller response.


WORKFLOW PERMISSION IS NOT A FACT REQUEST

A workflow proposal such as:

"Would you like to create a demo patient profile?"

has:

requested_facts = []
other_requested_facts = []

It is not asking for patient information yet.


SEARCH PERMISSION IS NOT A FACT REQUEST

Example:

"Would you like me to check Friday afternoon appointments?"

This is:

requested_action = "grant_permission"
response_required = true
requested_facts = []
other_requested_facts = []
appointment_options = []

The sentence itself must NEVER appear in requested_facts.


FACT REQUESTS

Examples:

"What insurance do you have?"
requested_action = "answer_fact"
requested_facts = ["insurance"]

"Can I get your first and last name?"
requested_action = "answer_fact"
requested_facts = ["first_name", "last_name"]

Use other_requested_facts only when the requested fact genuinely does not fit
the canonical RequestedFact enum.


SPLIT / ELLIPTICAL APPOINTMENT OFFERS

A remote agent may communicate one appointment option across multiple
consecutive turns.

Example:

recent_agent_history:
"How about 4:30 PM?"

latest_agent_turn:
"Friday."

If the latest turn clearly supplies a missing component of the immediately
preceding appointment offer or choice request, treat it as a CONTINUATION
of that same offer.

Merge only information that is safely established by the relevant recent
offer.

For the example above:

requested_action = "choose_option"
response_required = true

appointment_options = [
  {
    "day": "Friday",
    "time": "4:30 PM"
  }
]

Do NOT classify the completing fragment as a passive requested_action="none"
merely because the latest utterance is short.

The same rule applies when the pieces are reversed:

recent_agent_history:
"How about Friday?"

latest_agent_turn:
"At 4:30 PM."

The result is one Friday 4:30 PM appointment option.

IMPORTANT:

Only inherit from an immediately relevant scheduling offer.

Do NOT combine unrelated dates, times, providers, or appointment details
from distant conversation history.


BOOKING CONFIRMATION COMPLETENESS

If booking_confirmed = true, confirmed_appointment MUST identify the
confirmed slot.

Example:

latest_agent_turn:
"Okay, you're booked for Friday at 4:30 PM."

Use:

booking_confirmed = true

confirmed_appointment = {
  "day": "Friday",
  "time": "4:30 PM"
}

Do not return booking_confirmed=true with confirmed_appointment=null when
the booked slot is present in the latest utterance.

If the confirmation is elliptical, such as:

"You're all set."

a confirmed slot may be inherited only when the immediately relevant recent
scheduling context establishes it unambiguously.

Never invent a missing slot.


GENERAL PRESENTED CHOICES

The remote agent may ask the caller to choose between actions, searches,
workflow branches, or other alternatives that are NOT concrete bookable
appointment slots.

Example:

"There are no Friday afternoon openings on August 21st.
Would you like to look at afternoon options on another day,
or check the following Friday, August 28th?"

Use:

requested_action = "choose_presented_choice"
response_required = true
appointment_options = []

presented_choices = [
  {
    "label": "look at afternoon options on another day",
    "kind": "search_availability",
    "day": "another day",
    "date_text": null,
    "time": null,
    "daypart": "afternoon",
    "provider": null,
    "appointment_type": null
  },
  {
    "label": "check the following Friday, August 28th",
    "kind": "search_availability",
    "day": "Friday",
    "date_text": "August 28th",
    "time": null,
    "daypart": null,
    "provider": null,
    "appointment_type": null
  }
]

Do not use patient preferences to populate presented_choices.
Do not decide which branch is best in the semantic layer.
Concrete booking-slot choices remain requested_action = "choose_option".
If alternatives are merely informational, do not manufacture a choice.

SEMANTIC ONTOLOGY BOUNDARIES

Keep these categories strictly separate.

1. CALLER / PROFILE ASSERTION

Example:

"Your date of birth is July 4th, 2000."

This belongs in stated_facts.

2. APPOINTMENT OFFER

Example:

"I have Friday at 2:30 PM available. Would you like that?"

This belongs in appointment_options.

3. BOOKING CONFIRMATION

Examples:

"You are booked for Friday at 2:30 PM."
"Great, you're booked Friday at 2:30."

Use:

booking_confirmed = true

confirmed_appointment = {
  "day": "Friday",
  "time": "2:30 PM"
}

appointment_options = []

unless the same utterance separately contains additional choices.

Do NOT represent a booked slot as:

preferred_day
preferred_time
appointment_type

Merely scheduling or booking Friday does NOT assert that the caller's
preferred_day is Friday.

Merely scheduling or booking 2:30 PM does NOT assert that the caller's
preferred_time is 2:30 PM.

The words:

book
booked
booking
appointment

are NOT appointment_type values.

appointment_type is only an actual visit category such as:

new patient consultation
follow-up
routine visit

A phrase such as:

"Friday at 2:30 PM"

is NEVER an appointment_type.


CALLER PREFERENCES AS ASSERTIONS

Only create a stated_fact for preferred_day or preferred_time when the
remote agent explicitly attributes a PREFERENCE to the caller.

Examples:

"You said you prefer Friday."
"Your preferred time is afternoon."

Those may be stated_facts.

But:

"You are booked Friday at 2:30 PM."

is booking state, not preference state.


REMOTE FACT ASSERTIONS

IMPORTANT SOURCE RULE FOR stated_facts

stated_facts is stricter than ordinary contextual interpretation.

A stated_fact means the REMOTE AGENT ASSERTED THAT FACT IN
latest_agent_turn.

Therefore:

- stated_facts MUST be supported by latest_agent_turn itself
- NEVER inherit a stated_fact from recent_agent_history
- NEVER copy a stated_fact from an example in this prompt
- NEVER manufacture an assertion because it would be plausible
- recent history may help interpret other conversational fields, but
  it cannot create a new assertion in the current turn

Example:

recent history:
"Your date of birth is July 4th, 2000."

latest turn:
"Great. You're booked Friday at 2:30 PM."

The latest turn has:

stated_facts = []

The previous DOB statement MUST NOT reappear as a current assertion.

The remote agent may state a fact ABOUT THE CALLER.

Examples:

"Your date of birth is July 4th, 2000."
stated_facts = [
  {
    "fact": "date_of_birth",
    "value": "July 4th, 2000"
  }
]

"I have your insurance as Blue Cross."
stated_facts = [
  {
    "fact": "insurance",
    "value": "Blue Cross"
  }
]

"You are a returning patient."
stated_facts = [
  {
    "fact": "patient_status",
    "value": "a returning patient"
  }
]

IMPORTANT:

stated_facts records what the REMOTE AGENT CLAIMED.

You do not know whether the assertion is true.

Never change, suppress, or rewrite an asserted value to match what you
think the caller might want.

A single remote utterance may contain BOTH an assertion and a request.

Example:

"Your date of birth is July 4th, 2000. How may I help you today?"

should contain:

stated_facts:
  date_of_birth = July 4th, 2000

AND:

requested_action = "state_objective"

Do not discard one semantic event merely because another occurs later
in the same utterance.

If no caller-related fact was asserted:

stated_facts = []

STATUS / WAIT

Examples:

"One moment."
"Let me check availability."
"I'm searching for openings now."

Normally:

speech_act = "status"
requested_action = "wait"
response_required = false
agent_is_still_working = true


APPOINTMENT OFFERS

Extract EVERY concrete appointment option actually communicated.

Example:

"We have Friday at 9 AM, 9:45 AM and 10:30 AM with Becker."

Return THREE SlotOption objects.

Never collapse multiple times into one option.

Do not convert these into a yes/no patient decision.
The policy layer will evaluate them later.


SEARCH PERMISSION VS OFFER

"Would you like me to check Friday afternoon appointments?"

is permission to SEARCH.

It contains ZERO appointment_options.

"We have Friday at 2:30 PM. Would that work?"

is an actual appointment OFFER.


CHOICE QUESTIONS

Only use requested_action = "choose_option" when concrete alternatives are
actually identifiable.

Example:

"Which time would you like: 9 AM, 9:45 AM or 10:30 AM?"

requested_action = "choose_option"

appointment_options contains all three times.

If there are zero concrete alternatives, NEVER emit choose_option.

For example:

"Do you have a specific provider you'd like to see?"

is NOT choose_option because no concrete appointment alternatives were
offered.


CONVERSATIONAL INHERITANCE

Example:

recent history:
"Friday has 9 AM and 10 AM available."

latest:
"Which one would you like?"

Friday and the two options may be inherited because the latest utterance
clearly refers to those previously offered options.

But if there is NO relevant recent history and latest says:

"Which time would you like, 9 AM or 10 AM?"

then the option day MUST be null.


ASR NORMALIZATION

Streaming telephony ASR may render equivalent times as:

9.45 a.m.
9:45 AM
10.30 a.
10:30 a.m.

Preserve or normalize obvious intended clock meaning without inventing
missing scheduling facts.


CONFIDENCE

confidence measures confidence in the semantic extraction.

It does NOT measure whether the remote agent is correct.

Return only schema-valid structured output.
"""


_ORDINAL_SUFFIX_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)\b",
    flags=re.IGNORECASE,
)

_MERIDIEM_PUNCT_RE = re.compile(
    r"\b([ap])\s*\.?\s*m\.?(?=\s|$|[,.!?;:])",
    flags=re.IGNORECASE,
)

_NON_ALNUM_RE = re.compile(
    r"[^a-z0-9]+",
)

_EVIDENCE_NOISE_TOKENS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "my",
    "your",
}


def _normalize_evidence_text(
    value: object,
) -> str:
    """Normalize harmless speech/text differences for source checks."""

    text = str(
        value
    ).casefold()

    text = _ORDINAL_SUFFIX_RE.sub(
        r"\1",
        text,
    )

    # Canonicalize telephony meridiem variants before removing
    # punctuation so "p.m." and "PM" both become "pm".
    text = _MERIDIEM_PUNCT_RE.sub(
        r"\1m",
        text,
    )

    text = _NON_ALNUM_RE.sub(
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def _asserted_value_has_current_turn_evidence(
    *,
    value: object,
    agent_turn: str,
) -> bool:
    """Require assertion values to be evidenced by THIS remote turn.

    This deliberately prefers dropping an uncertain assertion over
    fabricating a patient correction.

    It is not semantic reasoning. It is a final provenance guard.
    """

    source = _normalize_evidence_text(
        agent_turn
    )

    candidate = _normalize_evidence_text(
        value
    )

    if not source or not candidate:
        return False

    # Strongest case: normalized value occurs directly in source.
    if candidate in source:
        return True

    # Permit harmless surrounding-word differences such as:
    #
    #   value  = "a returning patient"
    #   source = "You are a returning patient."
    #
    # Every meaningful value token still has to occur in THIS turn.
    tokens = [
        token
        for token in candidate.split()
        if token not in _EVIDENCE_NOISE_TOKENS
    ]

    if not tokens:
        return False

    source_tokens = set(
        source.split()
    )

    return all(
        token in source_tokens
        for token in tokens
    )


_WEEKDAY_RE = re.compile(
    r"\b("
    r"monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday"
    r")\b",
    flags=re.IGNORECASE,
)

_CLOCK_TIME_RE = re.compile(
    r"""
    \b
    (?P<hour>1[0-2]|0?[1-9])
    (?:
        [:.]
        (?P<minute>[0-5][0-9])
    )?
    \s*
    (?P<meridiem>
        a(?:\.?\s*m\.?)?
        |
        p(?:\.?\s*m\.?)?
    )
    (?=\s|$|[,.!?;:])
    """,
    flags=(
        re.IGNORECASE
        | re.VERBOSE
    ),
)


def _extract_explicit_weekday(
    text: str,
) -> str | None:
    match = _WEEKDAY_RE.search(
        text
    )

    if match is None:
        return None

    return match.group(
        1
    ).capitalize()


def _extract_explicit_clock_time(
    text: str,
) -> str | None:
    """Extract an explicitly spoken 12-hour clock time.

    Examples:

    4.30 p.m. -> 4:30 PM
    9:45 AM   -> 9:45 AM
    4 PM      -> 4:00 PM
    """

    match = _CLOCK_TIME_RE.search(
        text
    )

    if match is None:
        return None

    hour = int(
        match.group(
            "hour"
        )
    )

    minute_text = match.group(
        "minute"
    )

    minute = (
        int(minute_text)
        if minute_text is not None
        else 0
    )

    raw_meridiem = (
        match.group(
            "meridiem"
        )
        .casefold()
        .replace(
            ".",
            "",
        )
        .replace(
            " ",
            "",
        )
    )

    meridiem = (
        "PM"
        if raw_meridiem.startswith(
            "p"
        )
        else "AM"
    )

    return (
        f"{hour}:{minute:02d} "
        f"{meridiem}"
    )


def _slot_field_has_source_evidence(
    *,
    value: object,
    agent_turn: str,
    recent_history: Sequence[str],
) -> bool:
    """Check appointment-slot provenance against real remote speech.

    Booking confirmations may inherit from nearby actual conversation
    history, but never from system-prompt examples or model memory.
    """

    if _asserted_value_has_current_turn_evidence(
        value=value,
        agent_turn=agent_turn,
    ):
        return True

    nearby_history = " ".join(
        item
        for item in recent_history[-2:]
        if item.strip()
    )

    if not nearby_history:
        return False

    return _asserted_value_has_current_turn_evidence(
        value=value,
        agent_turn=nearby_history,
    )


_BOOKING_CONFIRMATION_RE = re.compile(
    r"""
    (?:
        \b(?:you(?:'re| are)|your)\s+
        (?:now\s+)?
        (?:booked|scheduled|confirmed|book)\b
    )
    |
    (?:
        \bi\s+(?:have|'ve)\s+you\s+
        (?:booked|scheduled)\b
    )
    |
    (?:
        \bappointment\s+
        (?:is|has\s+been)\s+
        (?:booked|scheduled|confirmed)\b
    )
    |
    (?:
        \b(?:booked|scheduled|confirmed)\s+for\b
    )
    |
    (?:
        \byou(?:'re| are)\s+all\s+set\b
    )
    """,
    flags=(
        re.IGNORECASE
        | re.VERBOSE
    ),
)


def _has_current_booking_confirmation_evidence(
    agent_turn: str,
) -> bool:
    """Return whether THIS remote turn communicates booking completion.

    booking_confirmed is a current-turn semantic event. Previous scheduling
    history may fill details of an elliptical confirmation, but it cannot
    manufacture a new confirmation on a later goodbye or acknowledgement.
    """

    return (
        _BOOKING_CONFIRMATION_RE.search(
            agent_turn
        )
        is not None
    )


def source_repair_semantic_payload(
    *,
    payload: object,
    agent_turn: str,
    recent_history: Sequence[str],
) -> object:
    """Apply deterministic provenance repair before TurnFrame validation.

    Qwen proposes semantic structure. Explicit scheduling values in actual
    remote speech are authoritative over model-generated slot values.

    This is deliberately narrow: it repairs booking-confirmation provenance,
    not patient truth or policy decisions.
    """

    if not isinstance(
        payload,
        dict,
    ):
        return payload

    repaired = dict(
        payload
    )

    if repaired.get(
        "booking_confirmed"
    ) is not True:
        return repaired

    # booking_confirmed is a semantic event in the CURRENT remote turn.
    #
    # Nearby history may later supply details for an elliptical current
    # confirmation such as "You're all set", but history must never turn
    # "Okay, bye" or another passive utterance into a new confirmation.
    if not _has_current_booking_confirmation_evidence(
        agent_turn
    ):
        repaired[
            "booking_confirmed"
        ] = False

        repaired[
            "confirmed_appointment"
        ] = None

        return repaired

    raw_slot = repaired.get(
        "confirmed_appointment"
    )

    if isinstance(
        raw_slot,
        dict,
    ):
        slot = dict(
            raw_slot
        )
    else:
        slot = {
            "day": None,
            "date_text": None,
            "time": None,
            "daypart": None,
            "provider": None,
            "appointment_type": None,
        }

    # Prefer values explicitly present in the CURRENT booking statement.
    explicit_day = (
        _extract_explicit_weekday(
            agent_turn
        )
    )

    explicit_time = (
        _extract_explicit_clock_time(
            agent_turn
        )
    )

    if explicit_day is not None:
        slot[
            "day"
        ] = explicit_day

    if explicit_time is not None:
        slot[
            "time"
        ] = explicit_time

    # If the current confirmation is elliptical, nearby real conversation
    # may safely supply a missing day/time.
    nearby = " ".join(
        [
            *[
                item
                for item
                in recent_history[-2:]
                if item.strip()
            ],
            agent_turn,
        ]
    )

    if slot.get(
        "day"
    ) is None:
        inherited_day = (
            _extract_explicit_weekday(
                nearby
            )
        )

        if inherited_day is not None:
            slot[
                "day"
            ] = inherited_day

    if slot.get(
        "time"
    ) is None:
        inherited_time = (
            _extract_explicit_clock_time(
                nearby
            )
        )

        if inherited_time is not None:
            slot[
                "time"
            ] = inherited_time

    # Remove model-added slot attributes that have no provenance in the
    # actual remote speech.
    for field in (
        "date_text",
        "daypart",
        "provider",
        "appointment_type",
    ):
        value = slot.get(
            field
        )

        if value is None:
            continue

        if not _slot_field_has_source_evidence(
            value=value,
            agent_turn=agent_turn,
            recent_history=recent_history,
        ):
            slot[
                field
            ] = None

    # Day/time proposed by Qwen also need provenance if we were unable to
    # replace them with explicit source values above.
    if (
        explicit_day is None
        and slot.get(
            "day"
        ) is not None
        and not _slot_field_has_source_evidence(
            value=slot[
                "day"
            ],
            agent_turn=agent_turn,
            recent_history=recent_history,
        )
    ):
        slot[
            "day"
        ] = None

    if (
        explicit_time is None
        and slot.get(
            "time"
        ) is not None
        and not _slot_field_has_source_evidence(
            value=slot[
                "time"
            ],
            agent_turn=agent_turn,
            recent_history=recent_history,
        )
    ):
        slot[
            "time"
        ] = None

    if any(
        slot.get(
            field
        )
        is not None
        for field in (
            "day",
            "date_text",
            "time",
            "daypart",
            "provider",
            "appointment_type",
        )
    ):
        repaired[
            "confirmed_appointment"
        ] = slot
    else:
        repaired[
            "confirmed_appointment"
        ] = None

    return repaired


def source_ground_turn_frame(
    *,
    frame: TurnFrame,
    agent_turn: str,
) -> TurnFrame:
    """Remove unsupported remote-agent fact assertions.

    Qwen may use recent context to understand elliptical dialogue, but
    AgentFactAssertion has stricter provenance semantics:

        the remote agent asserted this fact in THIS turn.

    Unsupported assertions are removed before patient truth grounding.
    """

    supported = [
        assertion
        for assertion in frame.stated_facts
        if _asserted_value_has_current_turn_evidence(
            value=assertion.value,
            agent_turn=agent_turn,
        )
    ]

    if len(supported) == len(
        frame.stated_facts
    ):
        return frame

    return frame.model_copy(
        update={
            "stated_facts": supported,
        }
    )


class StructuredTurnReasoner:
    """Convert arbitrary remote-agent speech into typed semantics."""

    def __init__(
        self,
        *,
        model: str,
        url: str,
        client: httpx.Client | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.model = model
        self.url = url

        self._owns_client = client is None

        self._client = (
            client
            if client is not None
            else httpx.Client(
                timeout=timeout_seconds,
            )
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def interpret(
        self,
        *,
        agent_turn: str,
        recent_history: Sequence[str] = (),
    ) -> TurnFrame:
        """Return source-grounded structured meaning.

        One automatic repair attempt is allowed when Qwen returns output that
        violates the semantic schema. The schema error is fed back to the
        model rather than crashing the caller immediately.
        """

        normalized_turn = " ".join(
            agent_turn.split()
        )

        if not normalized_turn:
            raise ValueError(
                "agent_turn cannot be blank."
            )

        context = {
            "recent_agent_history": [
                " ".join(item.split())
                for item in recent_history[-4:]
                if item.strip()
            ],
            "latest_agent_turn": normalized_turn,
        }

        schema = TurnFrame.model_json_schema()

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + "\n\nOUTPUT JSON SCHEMA:\n"
                    + json.dumps(
                        schema,
                        separators=(",", ":"),
                    )
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    context,
                    separators=(",", ":"),
                ),
            },
        ]

        last_error: ValidationError | None = None

        for attempt in range(2):

            response = self._client.post(
                self.url,
                json={
                    "model": self.model,
                    "stream": False,
                    "think": False,
                    "format": schema,
                    "options": {
                        "temperature": 0,
                    },
                    "messages": messages,
                },
            )

            response.raise_for_status()

            payload = response.json()

            try:
                content = payload["message"]["content"]
            except (KeyError, TypeError) as error:
                raise RuntimeError(
                    "Ollama response did not contain message.content."
                ) from error

            if not isinstance(
                content,
                str,
            ):
                raise RuntimeError(
                    "Ollama message.content must be text."
                )

            try:
                try:
                    raw_payload = json.loads(
                        content
                    )
                except json.JSONDecodeError:
                    # Preserve Pydantic's normal structured validation
                    # behavior for malformed JSON.
                    frame = TurnFrame.model_validate_json(
                        content
                    )
                else:
                    grounded_payload = (
                        source_repair_semantic_payload(
                            payload=raw_payload,
                            agent_turn=normalized_turn,
                            recent_history=(
                                context[
                                    "recent_agent_history"
                                ]
                            ),
                        )
                    )

                    frame = TurnFrame.model_validate(
                        grounded_payload
                    )

                return source_ground_turn_frame(
                    frame=frame,
                    agent_turn=agent_turn,
                )

            except ValidationError as error:
                last_error = error

                if attempt == 1:
                    break

                # Give Qwen its invalid answer plus deterministic validator
                # feedback and allow one correction.
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

                repair_guidance = (
                    "Your previous structured output was invalid.\n\n"
                    "Validation error:\n"
                    f"{error}\n\n"
                    "Repair the STRUCTURED INTERPRETATION, not the source "
                    "meaning.\n\n"
                    "Use only latest_agent_turn and recent_agent_history "
                    "already supplied in this conversation.\n\n"
                    "GENERAL REPAIR RULES:\n"
                    "- Preserve fields that are clearly supported by the "
                    "source.\n"
                    "- Fix the field or dependent structure that caused the "
                    "schema violation.\n"
                    "- Do not erase a source-grounded semantic event merely "
                    "to make validation pass.\n"
                    "- Do not invent facts, options, dates, times, providers, "
                    "or patient information.\n\n"
                    "BOOKING CONFIRMATION REPAIR RULE:\n"
                    "If the validation error says that "
                    "booking_confirmed=true requires confirmed_appointment, "
                    "re-read the source before repairing.\n"
                    "If the remote agent clearly says the appointment IS "
                    "booked/confirmed, KEEP booking_confirmed=true.\n"
                    "Then populate confirmed_appointment from the booked slot "
                    "that is explicitly stated in latest_agent_turn or "
                    "unambiguously established by the immediately relevant "
                    "recent scheduling context.\n"
                    "Do NOT change booking_confirmed to false merely to "
                    "satisfy the schema when the remote agent explicitly "
                    "confirmed the booking.\n"
                    "If no booked slot can actually be grounded in the "
                    "supplied speech/history, do not invent one.\n\n"
                    "Return a completely new schema-valid JSON result."
                )

                messages.append(
                    {
                        "role": "user",
                        "content": repair_guidance,
                    }
                )

        assert last_error is not None

        raise last_error
