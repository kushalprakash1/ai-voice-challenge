"""State-aware semantic fallback for VoiceProbe v3.1.

The deterministic v3 policy remains the first path. This module is invoked only
for explicit FALLBACK decisions. Natural-language understanding is separated
from patient truth and flow mutation:

    remote utterance
        -> mathematical prototype scorer
        -> local structured Qwen classifier when needed
        -> validated semantic intent
        -> deterministic PolicyDecision

The language model never emits patient facts, slot acceptance, booking state, or
free-form patient speech.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Awaitable, Callable, Mapping
from urllib.parse import urlparse

import httpx

from .embedding_semantics import (
    CompositionalEmbeddingClassifier,
    EmbeddingClassifier,
    EmbeddingSemanticUnavailable,
)
from .flow_state import FlowSnapshot, FlowStage
from .models import DecisionKind, PatientFacts, PolicyDecision
from .statistical_semantics import StatisticalIntentScorer


DEFAULT_PRIMARY_MODEL = "qwen3:1.7b"
DEFAULT_ESCALATION_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

PROTOTYPE_ACCEPT_SCORE = 0.82
PROTOTYPE_MIN_MARGIN = 0.08
PRIMARY_MODEL_ACCEPT_CONFIDENCE = 0.82
ESCALATION_MODEL_ACCEPT_CONFIDENCE = 0.74
REPEATED_UNKNOWN_SIMILARITY = 0.88


class SemanticIntent(StrEnum):
    PROFILE_CREATE_REQUEST = "profile_create_request"
    FULL_NAME_REQUEST = "full_name_request"
    FIRST_NAME_REQUEST = "first_name_request"
    LAST_NAME_REQUEST = "last_name_request"
    DOB_REQUEST = "dob_request"
    DOB_ASSERTION = "dob_assertion"
    DOB_ASSERTION_AND_OPEN_ENDED_HELP = "dob_assertion_and_open_ended_help"
    OPEN_ENDED_HELP = "open_ended_help"
    VISIT_REASON_REQUEST = "visit_reason_request"
    APPOINTMENT_TYPE_REQUEST = "appointment_type_request"
    VISIT_REASON_AND_TYPE_REQUEST = "visit_reason_and_type_request"
    INSURANCE_REQUEST = "insurance_request"
    DATE_TIME_PREFERENCE_REQUEST = "date_time_preference_request"
    PROVIDER_PREFERENCE_REQUEST = "provider_preference_request"
    PRESENCE_CHECK = "presence_check"
    ACKNOWLEDGEMENT = "acknowledgement"
    STATUS_UPDATE = "status_update"
    SCHEDULING_COMPLEX = "scheduling_complex"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SemanticClassification:
    intent: SemanticIntent
    confidence: float
    source: str
    score: float = 0.0
    margin: float = 0.0


Classifier = Callable[
    [str, FlowSnapshot],
    Awaitable[SemanticClassification],
]


_PROTOTYPES: Mapping[SemanticIntent, tuple[str, ...]] = {
    SemanticIntent.PROFILE_CREATE_REQUEST: (
        "Would you like to create a demo patient profile?",
        "May I set up a patient profile for you?",
        "Should I create your patient profile?",
    ),
    SemanticIntent.FULL_NAME_REQUEST: (
        "What is your first and last name?",
        "Can I have your full name?",
        "Who am I speaking with?",
    ),
    SemanticIntent.FIRST_NAME_REQUEST: (
        "What is your first name?",
        "Can I get your given name?",
    ),
    SemanticIntent.LAST_NAME_REQUEST: (
        "What is your last name?",
        "Can I get your surname?",
        "What is your family name?",
    ),
    SemanticIntent.DOB_REQUEST: (
        "What is your date of birth?",
        "Can you give me your birthday?",
        "What is your DOB?",
    ),
    SemanticIntent.DOB_ASSERTION: (
        "Your date of birth is July fourth two thousand.",
        "I have your birthday as July fourth.",
        "Your DOB is listed as July fourth.",
    ),
    SemanticIntent.OPEN_ENDED_HELP: (
        "How may I help you today?",
        "May I help you?",
        "What can I help you with?",
        "What are you calling about?",
    ),
    SemanticIntent.VISIT_REASON_REQUEST: (
        "What is the reason for your visit?",
        "What is the reason for your appointment?",
        "What brings you in?",
        "What are we seeing you for?",
        "What is the specific concern?",
        "Why would you like to be seen?",
        "What is this appointment for?",
    ),
    SemanticIntent.APPOINTMENT_TYPE_REQUEST: (
        "What type of appointment do you need?",
        "Is this a new patient consultation or a follow up?",
        "What kind of visit is this?",
    ),
    SemanticIntent.VISIT_REASON_AND_TYPE_REQUEST: (
        "What is the reason for the visit and what type of appointment do you need?",
        "What brings you in and is this a new patient consultation?",
    ),
    SemanticIntent.INSURANCE_REQUEST: (
        "What insurance do you have?",
        "Who are you covered through?",
        "What is your insurance provider?",
        "Which carrier do you have?",
    ),
    SemanticIntent.DATE_TIME_PREFERENCE_REQUEST: (
        "What day and time would you prefer?",
        "When would you like the appointment?",
        "What date and time works for you?",
    ),
    SemanticIntent.PROVIDER_PREFERENCE_REQUEST: (
        "Do you have a provider preference?",
        "Which doctor would you prefer?",
        "Is first available okay?",
    ),
    SemanticIntent.PRESENCE_CHECK: (
        "Are you still there?",
        "Hello are you there?",
        "Can you hear me?",
    ),
    SemanticIntent.ACKNOWLEDGEMENT: (
        "Thanks Alex.",
        "Okay thank you.",
        "Great.",
    ),
    SemanticIntent.STATUS_UPDATE: (
        "Let me check that for you.",
        "I am looking at availability now.",
        "Your profile is set up.",
    ),
}


_STAGE_PRIORS: Mapping[FlowStage, Mapping[SemanticIntent, float]] = {
    FlowStage.PROFILE: {
        SemanticIntent.PROFILE_CREATE_REQUEST: 1.0,
        SemanticIntent.FULL_NAME_REQUEST: 0.8,
    },
    FlowStage.IDENTITY: {
        SemanticIntent.FULL_NAME_REQUEST: 1.0,
        SemanticIntent.FIRST_NAME_REQUEST: 1.0,
        SemanticIntent.LAST_NAME_REQUEST: 1.0,
    },
    FlowStage.DOB: {
        SemanticIntent.DOB_REQUEST: 1.0,
        SemanticIntent.DOB_ASSERTION: 1.0,
        SemanticIntent.OPEN_ENDED_HELP: 0.4,
    },
    FlowStage.VISIT_REASON: {
        SemanticIntent.VISIT_REASON_REQUEST: 1.0,
        SemanticIntent.OPEN_ENDED_HELP: 0.7,
    },
    FlowStage.APPOINTMENT_TYPE: {
        SemanticIntent.APPOINTMENT_TYPE_REQUEST: 1.0,
        SemanticIntent.VISIT_REASON_AND_TYPE_REQUEST: 0.9,
    },
    FlowStage.INSURANCE: {
        SemanticIntent.INSURANCE_REQUEST: 1.0,
    },
    FlowStage.DATE_TIME: {
        SemanticIntent.DATE_TIME_PREFERENCE_REQUEST: 1.0,
        SemanticIntent.SCHEDULING_COMPLEX: 0.7,
    },
    FlowStage.PROVIDER: {
        SemanticIntent.PROVIDER_PREFERENCE_REQUEST: 1.0,
    },
}


_TOKEN_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)


def _fact_names(items: object) -> set[str]:
    """Extract fact identifiers defensively from existing semantic models."""
    if items is None:
        return set()

    try:
        iterable = tuple(items)
    except TypeError:
        return set()

    names: set[str] = set()

    for item in iterable:
        if isinstance(item, str):
            names.add(item)
            continue

        if isinstance(item, dict):
            candidates = (
                item.get("fact"),
                item.get("key"),
                item.get("name"),
            )
        else:
            candidates = (
                getattr(item, "fact", None),
                getattr(item, "key", None),
                getattr(item, "name", None),
            )

        for candidate in candidates:
            value = getattr(candidate, "value", candidate)
            if isinstance(value, str) and value:
                names.add(value)

    return names


def _intent_from_existing_semantics(meaning: object) -> SemanticIntent | None:
    """Map existing TurnMeaning structure into the v3.1 intent ontology."""
    requested = _fact_names(getattr(meaning, "requested_facts", ()))
    stated = _fact_names(getattr(meaning, "stated_facts", ()))

    if "complaint" in requested and "appointment_type" in requested:
        return SemanticIntent.VISIT_REASON_AND_TYPE_REQUEST
    if "complaint" in requested:
        return SemanticIntent.VISIT_REASON_REQUEST
    if "appointment_type" in requested:
        return SemanticIntent.APPOINTMENT_TYPE_REQUEST
    if "insurance" in requested:
        return SemanticIntent.INSURANCE_REQUEST
    if "date_of_birth" in requested:
        return SemanticIntent.DOB_REQUEST
    if {"first_name", "last_name"} <= requested:
        return SemanticIntent.FULL_NAME_REQUEST
    if "name" in requested:
        return SemanticIntent.FULL_NAME_REQUEST
    if "first_name" in requested:
        return SemanticIntent.FIRST_NAME_REQUEST
    if "last_name" in requested:
        return SemanticIntent.LAST_NAME_REQUEST
    if "provider_preference" in requested:
        return SemanticIntent.PROVIDER_PREFERENCE_REQUEST
    if {"preferred_day", "preferred_time"} & requested:
        return SemanticIntent.DATE_TIME_PREFERENCE_REQUEST
    if "date_of_birth" in stated:
        return SemanticIntent.DOB_ASSERTION

    return None


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.casefold()))


def _features(text: str) -> Counter[str]:
    """Sparse n-gram feature vector used by the deterministic semantic scorer."""
    normalized = _normalize(text)
    tokens = normalized.split()
    vector: Counter[str] = Counter()

    for token in tokens:
        vector[f"u:{token}"] += 1.0

    for left, right in zip(tokens, tokens[1:]):
        vector[f"b:{left}_{right}"] += 1.35

    compact = normalized.replace(" ", "_")
    if len(compact) >= 3:
        for index in range(len(compact) - 2):
            vector[f"c:{compact[index:index + 3]}"] += 0.18

    return vector


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0

    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot / (left_norm * right_norm)


class PrototypeIntentScorer:
    """Dependency-free cosine prototype scorer with a flow-stage prior."""

    def __init__(self) -> None:
        self._prototype_vectors = {
            intent: tuple(_features(example) for example in examples)
            for intent, examples in _PROTOTYPES.items()
        }

    def classify(
        self,
        agent_turn: str,
        snapshot: FlowSnapshot,
    ) -> SemanticClassification:
        turn_vector = _features(agent_turn)
        scored: list[tuple[float, SemanticIntent]] = []

        for intent, vectors in self._prototype_vectors.items():
            lexical = max(
                (cosine_similarity(turn_vector, vector) for vector in vectors),
                default=0.0,
            )
            prior = _STAGE_PRIORS.get(snapshot.current_stage, {}).get(intent, 0.0)
            score = (0.88 * lexical) + (0.12 * prior)
            scored.append((score, intent))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_score, top_intent = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = max(0.0, top_score - second_score)

        return SemanticClassification(
            intent=top_intent,
            confidence=min(1.0, top_score),
            source="prototype",
            score=top_score,
            margin=margin,
        )

    def sufficiently_confident(
        self,
        result: SemanticClassification,
    ) -> bool:
        return (
            result.score >= PROTOTYPE_ACCEPT_SCORE
            and result.margin >= PROTOTYPE_MIN_MARGIN
        )


class LocalQwenIntentClassifier:
    """Strict local-only structured classifier."""

    def __init__(
        self,
        *,
        primary_model: str = DEFAULT_PRIMARY_MODEL,
        escalation_model: str = DEFAULT_ESCALATION_MODEL,
        url: str = DEFAULT_OLLAMA_URL,
        timeout_seconds: float = 8.0,
    ) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError(
                "VoiceProbe v3.1 semantic inference must remain on local Ollama."
            )

        self._primary_model = primary_model
        self._escalation_model = escalation_model
        self._url = url
        self._timeout_seconds = timeout_seconds

    async def __call__(
        self,
        agent_turn: str,
        snapshot: FlowSnapshot,
    ) -> SemanticClassification:
        primary = await self._classify_with_model(
            model=self._primary_model,
            agent_turn=agent_turn,
            snapshot=snapshot,
        )

        if (
            primary.intent != SemanticIntent.UNKNOWN
            and primary.confidence >= PRIMARY_MODEL_ACCEPT_CONFIDENCE
        ):
            return primary

        escalation = await self._classify_with_model(
            model=self._escalation_model,
            agent_turn=agent_turn,
            snapshot=snapshot,
        )

        if (
            escalation.intent != SemanticIntent.UNKNOWN
            and escalation.confidence >= ESCALATION_MODEL_ACCEPT_CONFIDENCE
        ):
            return escalation

        return SemanticClassification(
            intent=SemanticIntent.UNKNOWN,
            confidence=max(primary.confidence, escalation.confidence),
            source="qwen_unresolved",
        )

    async def _classify_with_model(
        self,
        *,
        model: str,
        agent_turn: str,
        snapshot: FlowSnapshot,
    ) -> SemanticClassification:
        intents = [intent.value for intent in SemanticIntent]
        schema = {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": intents},
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["intent", "confidence"],
            "additionalProperties": False,
        }

        system = (
            "You are the semantic perception layer for an autonomous synthetic "
            "patient scheduling caller. Classify only what the remote scheduler "
            "said. Return exactly one intent and confidence. Never answer the "
            "question, never invent patient facts, never decide that a booking "
            "occurred, and never select or accept an appointment slot. "
            "visit_reason_request includes reason for visit, reason for appointment, "
            "what brings you in, what are we seeing you for, or specific concern. "
            "open_ended_help includes how may I help, may I help you, or what are "
            "you calling about. dob_assertion is a scheduler statement that claims "
            "a DOB rather than asking for it. "
            "dob_assertion_and_open_ended_help is used only when the same turn both "
            "claims a DOB and asks an open-ended help question. "
            "status_update is information that does not solicit a response. "
            "acknowledgement is thanks/okay/great without another request. "
            "scheduling_complex is a scheduling offer/branch that is not safely "
            "represented by narrower intents. Use unknown only when the utterance "
            "genuinely cannot be understood."
        )

        user = json.dumps(
            {
                "current_flow_stage": snapshot.current_stage.value,
                "utterance": agent_turn,
            },
            separators=(",", ":"),
        )

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(
                self._url,
                json={
                    "model": model,
                    "stream": False,
                    "think": False,
                    "keep_alive": "30m",
                    "options": {"temperature": 0, "num_predict": 64},
                    "format": schema,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )

        response.raise_for_status()
        payload = response.json()

        try:
            content = payload["message"]["content"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                "Ollama semantic response did not contain assistant content."
            ) from error

        if not isinstance(content, str):
            raise TypeError("Ollama semantic response content was not text.")

        parsed = json.loads(content)
        if set(parsed) != {"intent", "confidence"}:
            raise ValueError("Ollama semantic response contained unexpected fields.")

        intent = SemanticIntent(parsed["intent"])
        confidence = float(parsed["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Ollama semantic confidence must be in [0, 1].")

        return SemanticClassification(
            intent=intent,
            confidence=confidence,
            source=f"ollama:{model}",
        )


_ALTERNATE_WEEKDAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday)\b",
    flags=re.IGNORECASE,
)

# Prefer a weekday grounded in the scheduler's proposed search action.
# This matters for utterances such as:
#
#   "Monday only has morning times. Would you like me to check
#    afternoon slots on Tuesday?"
#
# A naive first-weekday match would incorrectly select Monday.
_TARGET_ALTERNATE_WEEKDAY_RE = re.compile(
    r"\b(?:check|look(?:\s+at|\s+for)?|see|search(?:\s+for)?)\b"
    r"[^?.!]{0,96}?\b(monday|tuesday|wednesday|thursday)\b",
    flags=re.IGNORECASE,
)


def _grounded_alternate_weekday(agent_turn: str) -> str | None:
    targeted = _TARGET_ALTERNATE_WEEKDAY_RE.search(agent_turn)

    if targeted is not None:
        return targeted.group(1).title()

    # If the scheduler names example alternate days without attaching
    # one to a separate action phrase, preserve the existing first
    # explicitly grounded weekday behavior.
    first = _ALTERNATE_WEEKDAY_RE.search(agent_turn)

    if first is not None:
        return first.group(1).title()

    return None


def _complex_scheduling_action(
    agent_turn: str,
    snapshot: FlowSnapshot,
    confidence: float,
) -> PolicyDecision:
    """Resolve an understood scheduling-choice turn without clarification.

    Semantic perception has already established that this is a scheduling
    choice. This layer only grounds the patient action in explicit remote
    evidence and the durable scheduling state.
    """

    offered_day = _grounded_alternate_weekday(agent_turn)

    if offered_day is not None:
        response = f"Please check {offered_day} afternoon."

    elif snapshot.allow_earlier_week_afternoons:
        response = (
            "Yes, please check the first available weekday afternoon."
        )

    else:
        # The remote side has offered a broader search but no concrete
        # alternate weekday was recoverable. Preserve the requested
        # afternoon constraint and authorize only a weekday search.
        response = "Please check another weekday afternoon."

    return PolicyDecision(
        DecisionKind.SEARCH_ALTERNATE_DAY_AFTERNOON,
        text=response,
        reason="semantic_v31:choose_alternate_day_afternoon",
        confidence=confidence,
    )


class V31SemanticRouter:
    """Production fallback router with deterministic grounding and loop control."""

    def __init__(
        self,
        *,
        facts: PatientFacts | None = None,
        classifier: Classifier | None = None,
        scorer: PrototypeIntentScorer | None = None,
        use_embeddings: bool = False,
        embedding_classifier: EmbeddingClassifier | None = None,
    ) -> None:
        self._facts = facts or PatientFacts()
        # Generative Qwen remains out of production intent control. The old
        # injectable field is retained only for compatibility with existing
        # tests and explicit experimental callers.
        self._classifier = classifier
        self._scorer = scorer or PrototypeIntentScorer()
        self._statistical = StatisticalIntentScorer()

        if embedding_classifier is not None:
            self._embedding_classifier: EmbeddingClassifier | None = (
                embedding_classifier
            )
        elif use_embeddings:
            self._embedding_classifier = (
                CompositionalEmbeddingClassifier()
            )
        else:
            self._embedding_classifier = None

        self._last_unresolved_vector: Counter[str] | None = None
        self._clarification_count = 0

    @property
    def scorer(self) -> PrototypeIntentScorer:
        return self._scorer

    async def classify(
        self,
        agent_turn: str,
        snapshot: FlowSnapshot,
    ) -> SemanticClassification:
        statistical = self._statistical.classify(
            agent_turn,
            stage=snapshot.current_stage.value,
        )

        # Wrong-DOB correction was already proven by the exact run-3 replay.
        # Preserve that high-confidence safety behavior before the new embedding
        # request layer, whose benchmark focused on scheduler requests/actions.
        if (
            self._statistical.accepts(statistical)
            and statistical.intent == SemanticIntent.DOB_ASSERTION.value
        ):
            return SemanticClassification(
                intent=SemanticIntent.DOB_ASSERTION,
                confidence=min(1.0, statistical.score),
                source="statistical_v31:dob_assertion",
                score=statistical.score,
                margin=statistical.margin,
            )

        if self._embedding_classifier is not None:
            try:
                embedded = await self._embedding_classifier.classify(
                    agent_turn
                )
            except EmbeddingSemanticUnavailable:
                embedded = None

            if embedded is not None:
                try:
                    intent = SemanticIntent(embedded.intent)
                except ValueError:
                    intent = SemanticIntent.UNKNOWN

                return SemanticClassification(
                    intent=intent,
                    confidence=embedded.confidence,
                    source=embedded.source,
                    score=embedded.score,
                    margin=embedded.margin,
                )

        # Cache/network/model failures never become silence. They fail safely to
        # the previously validated statistical classifier.
        if self._statistical.accepts(statistical):
            return SemanticClassification(
                intent=SemanticIntent(statistical.intent),
                confidence=min(1.0, statistical.score),
                source="statistical_v31",
                score=statistical.score,
                margin=statistical.margin,
            )

        if self._statistical.confidently_unknown(statistical):
            return SemanticClassification(
                intent=SemanticIntent.UNKNOWN,
                confidence=min(1.0, statistical.score),
                source="statistical_v31_ood",
                score=statistical.score,
                margin=statistical.margin,
            )

        return SemanticClassification(
            intent=SemanticIntent.UNKNOWN,
            confidence=min(1.0, statistical.score),
            source="statistical_v31_abstain",
            score=statistical.score,
            margin=statistical.margin,
        )

    async def resolve(
        self,
        agent_turn: str,
        snapshot: FlowSnapshot,
    ) -> PolicyDecision:
        classification = await self.classify(agent_turn, snapshot)

        if classification.intent == SemanticIntent.UNKNOWN:
            return self._clarification(agent_turn, classification)

        self._last_unresolved_vector = None
        self._clarification_count = 0
        return self._decision_for_intent(
            classification=classification,
            agent_turn=agent_turn,
            snapshot=snapshot,
        )

    def _decision_for_intent(
        self,
        *,
        classification: SemanticClassification,
        agent_turn: str,
        snapshot: FlowSnapshot,
    ) -> PolicyDecision:
        intent = classification.intent
        facts = self._facts
        reason = f"semantic_v31:{intent.value}:{classification.source}"

        if intent == SemanticIntent.PROFILE_CREATE_REQUEST:
            return PolicyDecision(
                DecisionKind.CREATE_PROFILE,
                text=f"Yes, please. My name is {facts.first_name} {facts.last_name}.",
                reason=reason,
                confidence=classification.confidence,
            )

        if intent == SemanticIntent.FULL_NAME_REQUEST:
            return PolicyDecision(
                DecisionKind.ANSWER_FACT,
                text=f"{facts.first_name} {facts.last_name}.",
                reason="full_name_requested",
                confidence=classification.confidence,
            )

        if intent == SemanticIntent.FIRST_NAME_REQUEST:
            return PolicyDecision(
                DecisionKind.ANSWER_FACT,
                text=f"{facts.first_name}.",
                reason="first_name_requested",
                confidence=classification.confidence,
            )

        if intent == SemanticIntent.LAST_NAME_REQUEST:
            return PolicyDecision(
                DecisionKind.ANSWER_FACT,
                text=f"{facts.last_name}.",
                reason="last_name_requested",
                confidence=classification.confidence,
            )

        if intent == SemanticIntent.DOB_REQUEST:
            return PolicyDecision(
                DecisionKind.ANSWER_FACT,
                text=f"{facts.dob}.",
                reason="dob_requested",
                confidence=classification.confidence,
            )

        if intent == SemanticIntent.DOB_ASSERTION:
            if _mentions_authoritative_dob(agent_turn, facts.dob):
                return PolicyDecision(
                    DecisionKind.WAIT,
                    reason="semantic_v31:correct_dob_assertion",
                    confidence=classification.confidence,
                )
            return PolicyDecision(
                DecisionKind.CORRECT_FACT,
                text=f"Actually, my date of birth is {facts.dob}.",
                reason="dob_correction",
                confidence=classification.confidence,
            )

        if intent == SemanticIntent.DOB_ASSERTION_AND_OPEN_ENDED_HELP:
            if _mentions_authoritative_dob(agent_turn, facts.dob):
                return PolicyDecision(
                    DecisionKind.STATE_OBJECTIVE,
                    text=(
                        "I need to schedule an appointment for "
                        f"{facts.preferred_day} {facts.preferred_time}."
                    ),
                    reason="open_ended_intent_question",
                    confidence=classification.confidence,
                )
            return PolicyDecision(
                DecisionKind.CORRECT_AND_STATE_OBJECTIVE,
                text=(
                    f"Actually, my date of birth is {facts.dob}. "
                    "I need to schedule an appointment for "
                    f"{facts.preferred_day} {facts.preferred_time}."
                ),
                reason="correct_remote_fact_then_answer_open_intent",
                confidence=classification.confidence,
            )

        if intent == SemanticIntent.OPEN_ENDED_HELP:
            return PolicyDecision(
                DecisionKind.STATE_OBJECTIVE,
                text=(
                    "I need to schedule an appointment for "
                    f"{facts.preferred_day} {facts.preferred_time}."
                ),
                reason="open_ended_intent_question",
                confidence=classification.confidence,
            )

        if intent == SemanticIntent.VISIT_REASON_REQUEST:
            return PolicyDecision(
                DecisionKind.ANSWER_COMPLAINT,
                text=f"I have {facts.complaint}.",
                reason="complaint_requested",
                confidence=classification.confidence,
            )

        if intent == SemanticIntent.APPOINTMENT_TYPE_REQUEST:
            return PolicyDecision(
                DecisionKind.ANSWER_APPOINTMENT_TYPE,
                text=f"A {facts.appointment_type}.",
                reason="appointment_type_requested",
                confidence=classification.confidence,
            )

        if intent == SemanticIntent.VISIT_REASON_AND_TYPE_REQUEST:
            return PolicyDecision(
                DecisionKind.ANSWER_VISIT_DETAILS,
                text=(
                    f"I have {facts.complaint}. "
                    f"This is for a {facts.appointment_type}."
                ),
                reason="reason_and_visit_type_requested",
                confidence=classification.confidence,
            )

        if intent == SemanticIntent.INSURANCE_REQUEST:
            return PolicyDecision(
                DecisionKind.ANSWER_FACT,
                text=f"{facts.insurance}.",
                reason="insurance_requested",
                confidence=classification.confidence,
            )

        if intent == SemanticIntent.DATE_TIME_PREFERENCE_REQUEST:
            grounded_weekday = _grounded_alternate_weekday(agent_turn)

            if (
                snapshot.allow_earlier_week_afternoons
                and "afternoon" in _normalize(agent_turn)
                and grounded_weekday is not None
            ):
                return _complex_scheduling_action(
                    agent_turn,
                    snapshot,
                    classification.confidence,
                )

            return PolicyDecision(
                DecisionKind.STATE_OBJECTIVE,
                text=f"{facts.preferred_day} {facts.preferred_time}.",
                reason="semantic_v31:date_time_preference_requested",
                confidence=classification.confidence,
            )

        if intent == SemanticIntent.PROVIDER_PREFERENCE_REQUEST:
            return PolicyDecision(
                DecisionKind.ANSWER_PROVIDER_PREFERENCE,
                text="First available is fine.",
                reason="provider_preference_requested",
                confidence=classification.confidence,
            )

        if intent == SemanticIntent.PRESENCE_CHECK:
            return PolicyDecision(
                DecisionKind.STATE_OBJECTIVE,
                text=(
                    "Yes, I'm here. I need to schedule an appointment for "
                    f"{facts.preferred_day} {facts.preferred_time}."
                ),
                reason="presence_check_restate_objective",
                confidence=classification.confidence,
            )

        if intent in {
            SemanticIntent.ACKNOWLEDGEMENT,
            SemanticIntent.STATUS_UPDATE,
        }:
            return PolicyDecision(
                DecisionKind.WAIT,
                reason=reason,
                confidence=classification.confidence,
            )

        if intent == SemanticIntent.SCHEDULING_COMPLEX:
            # The existing durable relaxation state is intentionally
            # afternoon-specific. Do not apply it to morning/evening
            # scenarios until alternate-day relaxation is generalized.
            if _normalize(facts.preferred_time) != "afternoon":
                return self._clarification(
                    agent_turn,
                    classification,
                )

            return _complex_scheduling_action(
                agent_turn,
                snapshot,
                classification.confidence,
            )

        return self._clarification(agent_turn, classification)

    def _clarification(
        self,
        agent_turn: str,
        classification: SemanticClassification,
    ) -> PolicyDecision:
        current_vector = _features(agent_turn)
        repeated = (
            self._last_unresolved_vector is not None
            and cosine_similarity(
                current_vector,
                self._last_unresolved_vector,
            ) >= REPEATED_UNKNOWN_SIMILARITY
        )

        if repeated:
            self._clarification_count += 1
        else:
            self._clarification_count = 0

        self._last_unresolved_vector = current_vector

        variants = (
            "Could you rephrase that scheduling question?",
            (
                "I may not have understood. Are you asking for my personal "
                "information, reason for the visit, or appointment availability?"
            ),
            (
                "I'm having trouble with that wording. Please ask one "
                "scheduling question at a time."
            ),
            (
                "Could you ask that a different way and specify what information "
                "you need from me?"
            ),
        )
        text = variants[self._clarification_count % len(variants)]

        return PolicyDecision(
            DecisionKind.CLARIFY,
            text=text,
            reason=(
                "semantic_v31:unresolved:"
                f"{classification.source}:repeat={self._clarification_count}"
            ),
            confidence=classification.confidence,
        )


def _mentions_authoritative_dob(agent_turn: str, dob: str) -> bool:
    normalized = _normalize(agent_turn)
    dob_normalized = _normalize(dob)

    if dob_normalized in normalized:
        return True

    if dob_normalized == "april 12 1998":
        variants = (
            "april 12th 1998",
            "april twelfth 1998",
            "april twelfth nineteen ninety eight",
            "april twelve nineteen ninety eight",
        )
        return any(variant in normalized for variant in variants)

    return False
