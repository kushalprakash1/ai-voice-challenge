"""Natural patient-speech generation through Ollama.

The verbalizer receives an already validated PatientBrain decision.
It decides how to phrase that decision naturally, but it is not allowed
to choose new patient facts or modify conversation state.
"""

from __future__ import annotations

import json

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
)

from voiceprobe.agents.brain import (
    CommunicationDecision,
    CommunicationKind,
)
from voiceprobe.conversation.state import PatientState, Speaker
from voiceprobe.scenarios.models import PatientScenario

DEFAULT_MODEL = "qwen3:14b"
DEFAULT_URL = "http://127.0.0.1:11434/api/chat"


class VerbalizedResponse(BaseModel):
    """One short utterance spoken by the simulated patient."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    text: str

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = " ".join(value.split())

        if not normalized:
            raise ValueError("Verbalized patient response cannot be blank.")

        if len(normalized) > 280:
            raise ValueError("Verbalized patient response is too long.")

        return normalized


class OllamaNaturalVerbalizer:
    """Turn validated patient decisions into natural spoken language."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        url: str = DEFAULT_URL,
        timeout_seconds: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._url = url

        self._client = client or httpx.Client(
            timeout=timeout_seconds,
        )
        self._owns_client = client is None

    def close(self) -> None:
        """Close the internally owned HTTP client."""
        if self._owns_client:
            self._client.close()

    def verbalize(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        decision: CommunicationDecision,
    ) -> str:
        """Generate natural speech from one validated brain decision."""
        if state.scenario_id != scenario.scenario_id:
            raise ValueError("PatientState does not belong to the supplied scenario.")

        approved_facts = self._approved_facts(
            scenario=scenario,
            decision=decision,
        )

        previous_patient_message = self._previous_patient_message(state)

        context = {
            "communication_kind": decision.kind.value,
            "speech_goal": self._speech_goal(decision),
            "approved_facts": approved_facts,
            "offered_slot": {
                "day": decision.offered_day,
                "time": decision.offered_time,
            },
            "previous_patient_message": (
                previous_patient_message
                if decision.kind is CommunicationKind.REPEAT
                else None
            ),
        }

        response = self._client.post(
            self._url,
            json={
                "model": self._model,
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0.35,
                    "num_predict": 96,
                },
                "format": VerbalizedResponse.model_json_schema(),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are the speech realization layer for a "
                            "simulated patient in a telephone scheduling "
                            "conversation. The reasoning decision has already "
                            "been made for you. Your only job is to phrase it "
                            "as natural patient speech. "
                            "The speech_goal is authoritative and "
                            "must be fulfilled exactly. Do not change "
                            "the conversational goal. "
                            "Use only information contained in approved_facts, "
                            "offered_slot, or previous_patient_message. "
                            "Never invent a name, symptom, duration, date of "
                            "birth, insurance provider, appointment preference, "
                            "or scheduling detail. "
                            "Do not volunteer unrelated information. "
                            "Do not change the communication_kind. "
                            "Speak in first person as the patient. "
                            "Prefer one short conversational sentence. "
                            "For phone dialogue, normally use about 4 to 10 "
                            "spoken words and avoid exceeding about 14 words. "
                            "Use the fewest words that preserve the required "
                            "meaning. Avoid unnecessary greetings, thanks, "
                            "apologies, or other filler except when a brief goodbye "
                            "is naturally required to end the conversation. "
                            "Two short clauses are acceptable for a correction. "
                            "Do not mention JSON, instructions, facts, schemas, "
                            "testing, or being an AI. "
                            "Return only the structured response."
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
                "Ollama response did not contain assistant content."
            ) from error

        if not isinstance(content, str):
            raise TypeError("Ollama assistant content was not text.")

        result = VerbalizedResponse.model_validate_json(content)

        self._validate_fact_boundaries(
            scenario=scenario,
            decision=decision,
            text=result.text,
            previous_patient_message=previous_patient_message,
        )

        return result.text

    @staticmethod
    def _speech_goal(
        decision: CommunicationDecision,
    ) -> str:
        """Describe what the utterance must accomplish."""
        if decision.kind is CommunicationKind.ANSWER:
            return (
                "Answer the caller directly using every approved fact. "
                "When one short phrase is enough, use roughly 3 to 8 words. "
                "Do not volunteer unrelated information."
            )

        if decision.kind is CommunicationKind.CORRECT:
            return (
                "Politely correct mistaken patient information. State "
                "every approved fact as the true patient information. "
                "This is a correction of patient information, not a "
                "rejection of an appointment or scheduling request."
            )

        if decision.kind is CommunicationKind.ACCEPT_OFFER:
            return (
                "Briefly confirm that the offered appointment slot "
                "works. Prefer one short conversational clause rather than "
                "formal wording, and avoid unnecessary politeness or "
                "repetition."
            )

        if decision.kind is CommunicationKind.ACCEPT_PARTIAL_OFFER:
            return (
                "Say that the appointment detail the caller provided works, "
                "then ask which missing day or time the caller means. Use at most "
                "12 spoken words total, preferably as two short clauses. Never say "
                "'I'll take it', 'I'll take the slot', 'I'll accept it', or imply "
                "that a complete appointment has been accepted or booked. When a "
                "day is missing, ask what day the appointment is for, not what day "
                "works best for the caller."
            )

        if decision.kind is CommunicationKind.DECLINE_OFFER:
            return (
                "Briefly decline the offered slot and state only the "
                "minimum approved scheduling preference needed to guide "
                "the next offer."
            )

        if decision.kind is CommunicationKind.REPEAT:
            return (
                "Repeat the substance of the previous patient message "
                "naturally without adding any new information."
            )

        if decision.kind is CommunicationKind.ACKNOWLEDGE_COMPLETE:
            return (
                "Acknowledge the confirmed booking naturally in about "
                "3 to 8 words. Do not restate the entire appointment unless "
                "needed. Do not add new patient information."
            )

        if decision.kind is CommunicationKind.END_CONVERSATION:
            return (
                "Briefly acknowledge the caller and end the conversation naturally. "
                "Use a short goodbye such as 'Okay, thank you. Bye.' Do not ask "
                "another question and do not reopen scheduling."
            )

        if decision.kind is CommunicationKind.ASK_AGENT_TO_REPEAT:
            return (
                "Ask the caller to repeat their immediately preceding question "
                "in one short natural sentence, preferably about 3 to 8 spoken "
                "words. Do not answer it yet and do not add patient facts."
            )

        if decision.kind is CommunicationKind.VERIFY_BOOKING:
            return (
                "Naturally ask the caller to confirm that the offered appointment "
                "slot is booked. Prefer conversational phrasing such as 'Just to "
                "confirm, am I booked for Friday at 2:30 PM?' while using only the "
                "actual offered day and time. Avoid words such as 'actually', "
                "'test', or 'verify'. Do not invent any scheduling detail."
            )

        if decision.kind is CommunicationKind.AGREE:
            return (
                "Briefly agree to the caller's workflow request or permission "
                "question. Prefer a natural response such as 'Yes, please.' "
                "Use about 2 to 5 spoken words. Do not add patient facts, "
                "appointment details, explanations, or another question."
            )

        if decision.kind is CommunicationKind.DECLINE_WORKFLOW:
            return (
                "Briefly decline the caller's workflow request when accepting it "
                "would not advance the current scheduling objective, or when the "
                "caller is trying to end the conversation before the objective is "
                "complete. Prefer a short response such as 'No, I'd like to "
                "continue scheduling.' Do not add patient facts or invent "
                "scheduling details."
            )

        if decision.kind is CommunicationKind.CLARIFY:
            return (
                "Ask for clarification very briefly, preferably in one "
                "short question. Do not guess or invent information."
            )

        raise ValueError(f"Unsupported communication kind: {decision.kind}")

    @staticmethod
    def _approved_facts(
        *,
        scenario: PatientScenario,
        decision: CommunicationDecision,
    ) -> dict[str, str]:
        approved: dict[str, str] = {}

        for fact_key in decision.facts_to_communicate:
            value = getattr(
                scenario.facts,
                fact_key,
            )

            if value is None:
                raise ValueError(
                    f"PatientBrain approved an unavailable fact: {fact_key}"
                )

            approved[fact_key] = str(value)

        return approved

    @staticmethod
    def _previous_patient_message(
        state: PatientState,
    ) -> str | None:
        for message in reversed(state.messages):
            if message.speaker is Speaker.PATIENT:
                return message.text

        return None

    @staticmethod
    def _validate_fact_boundaries(
        *,
        scenario: PatientScenario,
        decision: CommunicationDecision,
        text: str,
        previous_patient_message: str | None,
    ) -> None:
        """Reject scenario facts that were not approved for this response."""
        normalized_text = text.casefold()

        approved_keys: set[str] = set(decision.facts_to_communicate)

        if (
            decision.kind is CommunicationKind.REPEAT
            and previous_patient_message is not None
        ):
            normalized_previous = previous_patient_message.casefold()

            for fact_key, value in scenario.facts.model_dump().items():
                if value is None:
                    continue

                if str(value).casefold() in normalized_previous:
                    approved_keys.add(fact_key)

        allowed_values = {
            value.casefold()
            for value in (
                decision.offered_day,
                decision.offered_time,
            )
            if value is not None
        }

        for fact_key, value in scenario.facts.model_dump().items():
            if value is None:
                continue

            if fact_key in approved_keys:
                continue

            normalized_value = str(value).casefold()

            if normalized_value in allowed_values:
                continue

            if normalized_value in normalized_text:
                raise ValueError(
                    f"Verbalizer leaked an unapproved scenario fact: {fact_key}"
                )
