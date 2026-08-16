"""Source-grounded semantic reasoning using local Ollama.

IMPORTANT ARCHITECTURAL RULE:

This module DOES NOT receive the patient profile.

The semantic interpreter's only job is to determine what the remote
agent communicated. Patient truth, goals, preferences, constraints,
and decisions belong to later reasoning layers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import httpx

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

SOURCE-GROUNDING RULES

A field may be filled when:

A. it is explicitly present in latest_agent_turn, OR
B. latest_agent_turn is clearly an elliptical continuation of recent
   remote-agent history and the value can safely be inherited.

If neither is true, return null for that field.

Example:

recent history:
"Friday has 9 AM and 10 AM available."

latest:
"Which one would you like?"

Friday may be inherited because the latest utterance clearly refers
to those previously offered options.

But if there is NO recent history and latest says:

"Which time would you like, 9 AM or 10 AM?"

then day MUST be null.

Never infer Friday merely because a simulated caller might prefer Friday.

STATUS / WAIT

Examples:

"One moment."
"Let me check availability."
"I'm searching for openings now."

Normally:

speech_act = status
requested_action = wait
response_required = false
agent_is_still_working = true

SEARCH PERMISSION IS NOT A FACT REQUEST

Example:

"Would you like me to check Friday afternoon appointments?"

This is:

requested_action = grant_permission
response_required = true
requested_facts = []
other_requested_facts = []
appointment_options = []

The sentence itself must NEVER appear in requested_facts.

FACT REQUESTS

Examples:

"What is your insurance?"
requested_action = answer_fact
requested_facts = ["insurance"]

"Can I get your first and last name?"
requested_action = answer_fact
requested_facts = ["first_name", "last_name"]

Use other_requested_facts only when the requested fact genuinely does
not fit the canonical RequestedFact enum.

APPOINTMENT OFFERS

Extract EVERY concrete option actually communicated.

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

"Which time would you like: 9 AM, 9:45 AM or 10:30 AM?"

requested_action = choose_option

and appointment_options contains all three times.

If no day is stated and no relevant recent history supplies a day,
every option's day must be null.

ASR NORMALIZATION

Streaming telephony ASR may render equivalent times as:

9.45 a.m.
9:45 AM
10.30 a.
10:30 a.m.

Preserve or normalize obvious intended clock meaning without inventing
missing scheduling facts.

CONFIDENCE

confidence measures how confident you are in the semantic extraction.

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
        """Return source-grounded structured meaning."""

        normalized_turn = " ".join(
            agent_turn.split()
        )

        if not normalized_turn:
            raise ValueError(
                "agent_turn cannot be blank."
            )

        context = {
            # IMPORTANT:
            # No patient/scenario/objective data belongs here.
            "recent_agent_history": [
                " ".join(item.split())
                for item in recent_history[-4:]
                if item.strip()
            ],
            "latest_agent_turn": normalized_turn,
        }

        schema = TurnFrame.model_json_schema()

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
                "messages": [
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
                ],
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

        if not isinstance(content, str):
            raise RuntimeError(
                "Ollama message.content must be text."
            )

        return TurnFrame.model_validate_json(
            content
        )
