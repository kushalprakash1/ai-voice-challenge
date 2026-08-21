"""SemanticFrame-native Qwen interpreter for SemanticLab v2.

Offline candidate only. It is not wired into the production planner/runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from voiceprobe.v32.ollama_backend import OllamaBackend, OllamaConfig

from .semantic_frame import (
    AmbiguityKind,
    ConstraintAxis,
    RecordClaim,
    ReferenceKind,
    SemanticAmbiguity,
    SemanticFrame,
    SemanticTopic,
    SpeechAct,
    TransactionOperation,
    TransactionSignal,
)


FACT_KEYS = (
    "first_name",
    "last_name",
    "full_name",
    "dob",
    "insurance",
    "complaint",
    "visit_type",
    "reschedule_reason",
)


@dataclass(slots=True)
class OllamaSemanticFrameInterpreter:
    backend: OllamaBackend

    @classmethod
    def from_endpoint(
        cls,
        *,
        endpoint: str,
        model: str = "qwen3.5:4b",
        timeout_seconds: float = 10.0,
    ) -> "OllamaSemanticFrameInterpreter":
        return cls(
            backend=OllamaBackend(
                OllamaConfig(
                    endpoint=endpoint,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    keep_alive="15m",
                    num_ctx=1024,
                    temperature=0.0,
                )
            )
        )

    async def warmup(self) -> None:
        await self.backend.generate_json(
            system="Return the required JSON object only.",
            prompt='{"warmup":true}',
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        )

    @staticmethod
    def schema() -> dict[str, Any]:
        axes = [axis.value for axis in ConstraintAxis]
        return {
            "type": "object",
            "properties": {
                "speech_act": {
                    "type": "string",
                    "enum": [value.value for value in SpeechAct],
                },
                "topic": {
                    "type": "string",
                    "enum": [value.value for value in SemanticTopic],
                },
                "requested_fact": {
                    "type": "string",
                    "enum": ["none", *FACT_KEYS],
                },
                "failed_constraints": {
                    "type": "array",
                    "items": {"type": "string", "enum": axes},
                    "uniqueItems": True,
                    "maxItems": 3,
                },
                "proposed_changes": {
                    "type": "array",
                    "items": {"type": "string", "enum": axes},
                    "uniqueItems": True,
                    "maxItems": 3,
                },
                "retained_constraints": {
                    "type": "array",
                    "items": {"type": "string", "enum": axes},
                    "uniqueItems": True,
                    "maxItems": 3,
                },
                "offered_options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 4,
                },
                "selected_option": {"type": "string"},
                "record_claims": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [value.value for value in RecordClaim],
                    },
                    "uniqueItems": True,
                    "maxItems": 2,
                },
                "transaction_operation": {
                    "type": "string",
                    "enum": [value.value for value in TransactionOperation],
                },
                "transaction_signal": {
                    "type": "string",
                    "enum": [value.value for value in TransactionSignal],
                },
                "reference": {
                    "type": "string",
                    "enum": [value.value for value in ReferenceKind],
                },
                "ambiguity": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": [value.value for value in AmbiguityKind],
                        },
                        "candidates": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 4,
                        },
                        "detail": {"type": "string"},
                    },
                    "required": ["kind", "candidates", "detail"],
                    "additionalProperties": False,
                },
            },
            "required": [
                "speech_act",
                "topic",
                "requested_fact",
                "failed_constraints",
                "proposed_changes",
                "retained_constraints",
                "offered_options",
                "selected_option",
                "record_claims",
                "transaction_operation",
                "transaction_signal",
                "reference",
                "ambiguity",
            ],
            "additionalProperties": False,
        }

    async def interpret(
        self,
        *,
        remote_turn: str,
        recent_remote_turns: tuple[str, ...] = (),
    ) -> tuple[SemanticFrame, dict[str, Any]]:
        payload = {
            "recent_remote_turns": list(recent_remote_turns[-2:]),
            "turn": " ".join(remote_turn.split()),
        }

        system = """You are the semantic perception layer inside a simulated patient voice agent.

Extract ONLY observable meaning from the clinic's latest utterance. Recent clinic turns may be used only to resolve references such as "that one", "same time", "the earlier one", "yes", pronouns, or ellipsis. Do not copy old meanings into the latest turn unless the latest turn refers to them.

Never infer or choose patient goals, patient preferences, patient facts, patient actions, strategic decisions, or patient dialogue.

The fields are independent semantic features. Preserve compound meaning instead of forcing the utterance into one mutually-exclusive intent.

speech_act = the primary conversational act of the LATEST utterance.
topic = the primary subject of the latest utterance. Other independent fields still preserve secondary meaning.

requested_fact = an authoritative patient fact explicitly requested in the latest turn, otherwise none.
"What are you being seen for?" requests complaint.
"What type of visit do you need?" requests visit_type.
"Why are you rescheduling?" requests reschedule_reason.

failed_constraints = EVERY scheduling dimension explicitly reported unavailable:
day, time_of_day, provider.
If "Friday afternoon" is unavailable, include BOTH day and time_of_day.

proposed_changes = EVERY scheduling dimension the clinic explicitly proposes varying/searching next.
retained_constraints = EVERY scheduling dimension the clinic explicitly says to keep fixed.
A dimension cannot be both proposed and retained.
"keep Friday and try another time" => proposed_changes=[time_of_day], retained_constraints=[day].
"keep the same time and try another day" => proposed_changes=[day], retained_constraints=[time_of_day].
"another day or time" => proposed_changes=[day,time_of_day].

offered_options = concrete appointment choices explicitly PRESENTED IN THE LATEST turn. Never copy choices from context into offered_options.
selected_option = a concrete option selected/accepted/resolved by the latest turn. It MAY be resolved from context, including "Wednesday", "the later one", "yes", or "go with her".

record_claims = explicit actual profile or appointment RECORD state:
profile_exists, profile_missing, appointment_exists, appointment_missing.
Availability language such as no openings is never a missing-record claim.
Questions and hypotheticals are not record claims.

transaction_operation = none, book, reschedule, cancel, keep, create_profile, search.
search is read-only.

transaction_signal applies only to STATE-CHANGING operations:
proposed, permission_request, confirmed, none.
A question asking permission merely to CHECK/SEARCH availability has transaction_signal=none.

reference identifies what the latest turn points back to:
none, prior_option, prior_day, prior_time, prior_provider, prior_entity, ambiguous.

ambiguity represents genuine unresolved semantic uncertainty AFTER using context.
kind=none requires candidates=[] and detail="".
Otherwise provide at least two plausible candidates.
Do not invent ambiguity when context resolves the meaning.

ASR text may contain fillers, missing punctuation, transcription mistakes, or fragments. Interpret recoverable observable meaning.

Return the required JSON object only."""

        raw = await self.backend.generate_json(
            system=system,
            prompt=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            schema=self.schema(),
        )

        requested_fact = str(raw.get("requested_fact") or "none")
        if requested_fact == "none":
            requested_fact = ""

        ambiguity_raw = dict(raw.get("ambiguity") or {})
        ambiguity_kind = AmbiguityKind(
            str(ambiguity_raw.get("kind") or AmbiguityKind.NONE.value)
        )
        ambiguity_candidates = tuple(
            str(value) for value in ambiguity_raw.get("candidates", ())
        )
        ambiguity_detail = str(ambiguity_raw.get("detail") or "")
        if ambiguity_kind is AmbiguityKind.NONE:
            ambiguity_candidates = ()
            ambiguity_detail = ""

        frame = SemanticFrame(
            raw_text=remote_turn,
            speech_act=SpeechAct(str(raw["speech_act"])),
            topic=SemanticTopic(str(raw["topic"])),
            requested_fact=requested_fact,
            failed_constraints=tuple(
                ConstraintAxis(str(value))
                for value in raw.get("failed_constraints", ())
            ),
            proposed_changes=tuple(
                ConstraintAxis(str(value))
                for value in raw.get("proposed_changes", ())
            ),
            retained_constraints=tuple(
                ConstraintAxis(str(value))
                for value in raw.get("retained_constraints", ())
            ),
            offered_options=tuple(
                str(value) for value in raw.get("offered_options", ())
            ),
            selected_option=str(raw.get("selected_option") or ""),
            record_claims=tuple(
                RecordClaim(str(value))
                for value in raw.get("record_claims", ())
            ),
            transaction_operation=TransactionOperation(
                str(raw.get("transaction_operation") or "none")
            ),
            transaction_signal=TransactionSignal(
                str(raw.get("transaction_signal") or "none")
            ),
            reference=ReferenceKind(str(raw.get("reference") or "none")),
            ambiguity=SemanticAmbiguity(
                kind=ambiguity_kind,
                candidates=ambiguity_candidates,
                detail=ambiguity_detail,
            ),
        )
        return frame, raw
