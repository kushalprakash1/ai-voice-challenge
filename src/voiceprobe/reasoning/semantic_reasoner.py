"""Source-grounded semantic reasoning using local Ollama.

IMPORTANT ARCHITECTURAL RULE:

This module DOES NOT receive the patient profile.

The semantic interpreter determines only what the remote agent communicated.
Patient truth, goals, preferences, constraints, and decisions belong to later
reasoning layers.
"""

from __future__ import annotations

import json
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


REMOTE FACT ASSERTIONS

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
                return TurnFrame.model_validate_json(
                    content
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

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous structured output was invalid.\n\n"
                            "Validation error:\n"
                            f"{error}\n\n"
                            "Correct the semantic interpretation using only "
                            "the supplied agent speech/history. Do not invent "
                            "missing facts or appointment options. Return a "
                            "new schema-valid result."
                        ),
                    }
                )

        assert last_error is not None

        raise last_error
