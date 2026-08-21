"""Calibrated local embedding semantics for VoiceProbe v3.1.

This module is a perception layer only. It never writes patient facts, accepts
appointment slots, mutates booking state, or generates patient speech.

Production path:
    remote utterance
      -> clause segmentation
      -> local qwen3-embedding vectors
      -> cosine similarity against validated prototypes / centroids
      -> calibrated speech-act overrides
      -> one atomic fact request per semantic clause
      -> optional compound reason+appointment-type intent
      -> deterministic v3 PolicyDecision in semantic_router.py

Prototype embeddings are prepared offline and stored under ~/.cache/voiceprobe.
A missing/invalid cache or unavailable local Ollama endpoint raises a typed
availability error so the v3 router can fail safely to its statistical fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence
from urllib.parse import urlparse

import httpx

from .statistical_semantics import _CORPUS


DEFAULT_MODEL = "qwen3-embedding:0.6b"
DEFAULT_EMBED_URL = "http://127.0.0.1:11434/api/embed"
CACHE_VERSION = 1

FACT_HEADS = (
    "visit_reason_request",
    "appointment_type_request",
    "insurance_request",
    "dob_request",
    "full_name_request",
    "first_name_request",
    "last_name_request",
    "provider_preference_request",
    "date_time_preference_request",
)

SPEECH_ACT_HEADS = (
    "open_ended_help",
    "presence_check",
    "profile_create_request",
    "scheduling_complex",
    "acknowledgement",
    "status_update",
    "unknown",
)

COMPOUND_PROTOTYPES = (
    "what is the reason for your appointment and what type of visit is this",
    "what brings you in and is this a new patient consultation or follow up",
    "what is the reason for the visit is it routine follow up or a specific concern",
    "tell me why you need the appointment and whether this is a new patient visit",
    "what are we seeing you for and what kind of appointment do you need",
    "what is this appointment for and is it a consultation or follow up",
    "what is the concern and what appointment type should i schedule",
    "why are you coming in is this routine follow up or for a specific concern",
    "what is the reason for your appointment for example routine follow up or a specific concern",
    "what brings you in is this for a routine visit a follow up or a specific concern",
    "tell me the reason for the appointment and the kind of visit",
    "what are you being seen for and is this a new patient consultation",
)

FACT_THRESHOLDS: Mapping[str, float] = {
    "provider_preference_request": 0.60,
}
DEFAULT_FACT_THRESHOLD = 0.68

SPEECH_OVERRIDE_RULES: Mapping[str, tuple[float, float]] = {
    "profile_create_request": (0.80, 0.08),
    "presence_check": (0.78, 0.08),
    "open_ended_help": (0.80, 0.10),
    "scheduling_complex": (0.72, 0.05),
    "unknown": (0.68, 0.05),
}

GENERIC_HELP_TIE_WINDOW = 0.025

INTRO_PREFIX_RE = re.compile(
    r"^\s*(?:uh+|um+|okay|ok|alright|all right|so)\b[\s,:\-–—]*",
    flags=re.IGNORECASE,
)
CONTEXT_PREFIX_RE = re.compile(
    r"^\s*(?:before|once|after|while|since|given that|now that)\b",
    flags=re.IGNORECASE,
)
SPLIT_RE = re.compile(
    r"\s*(?:"
    r"[;]"
    r"|\?\s+"
    r"|[.!]\s+(?=(?:what|which|who|when|where|why|how|is|are|do|does|can|could|would|should|may|tell|give|please)\b)"
    r"|,\s+(?=and\b|but\b)"
    r"|\s+and\s+(?=(?:what|which|who|when|where|why|how|whether|is|are|do|does|can|could|would|should|may|tell|give|please)\b)"
    r")\s*",
    flags=re.IGNORECASE,
)


class EmbeddingSemanticUnavailable(RuntimeError):
    """The local embedding semantic layer cannot safely classify this turn."""


@dataclass(frozen=True, slots=True)
class EmbeddingSemanticResult:
    intent: str
    confidence: float
    score: float
    margin: float
    source: str
    clauses: tuple[str, ...]


class EmbeddingClassifier(Protocol):
    async def classify(self, text: str) -> EmbeddingSemanticResult:
        ...


def default_cache_path(
    model: str = DEFAULT_MODEL,
) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)
    return (
        Path.home()
        / ".cache"
        / "voiceprobe"
        / f"v31_semantic_{safe_model}.json"
    )


def _training_corpus() -> dict[str, tuple[str, ...]]:
    corpus = {
        label: tuple(examples)
        for label, examples in _CORPUS.items()
    }
    corpus["visit_reason_and_type_request"] = COMPOUND_PROTOTYPES
    return corpus


def training_items() -> tuple[tuple[str, str], ...]:
    items: list[tuple[str, str]] = []
    for label, examples in sorted(_training_corpus().items()):
        for text in examples:
            items.append((label, text))
    return tuple(items)


def corpus_sha256(
    model: str = DEFAULT_MODEL,
) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "model": model,
        "corpus": _training_corpus(),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_loopback_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http":
        raise ValueError("Embedding endpoint must use local HTTP.")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Embedding endpoint must be loopback-only.")
    if parsed.path.rstrip("/") != "/api/embed":
        raise ValueError("Embedding endpoint must be Ollama /api/embed.")


def normalize_clause(text: str) -> str:
    text = INTRO_PREFIX_RE.sub("", text.strip())
    return " ".join(text.split()).strip(" ,;:-–—")


def split_clauses(text: str) -> tuple[str, ...]:
    cleaned = normalize_clause(text)
    if not cleaned:
        return ()

    raw_parts = [
        normalize_clause(part)
        for part in SPLIT_RE.split(cleaned)
        if normalize_clause(part)
    ]

    if len(raw_parts) <= 1 and "," in cleaned:
        comma_parts = [
            normalize_clause(part)
            for part in cleaned.split(",")
            if normalize_clause(part)
        ]
        if len(comma_parts) > 1:
            raw_parts = comma_parts

    if len(raw_parts) > 1:
        kept: list[str] = []
        for index, part in enumerate(raw_parts):
            if (
                index < len(raw_parts) - 1
                and CONTEXT_PREFIX_RE.match(part)
                and "?" not in part
            ):
                continue
            kept.append(part)
        raw_parts = kept or raw_parts

    return tuple(raw_parts)


def _normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 0.0:
        raise EmbeddingSemanticUnavailable(
            "Embedding vector had invalid norm."
        )
    return tuple(value / norm for value in values)


def _dot(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    return sum(a * b for a, b in zip(left, right))


def _normalized_mean(
    vectors: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    if not vectors:
        raise EmbeddingSemanticUnavailable(
            "Embedding prototype class was empty."
        )
    width = len(vectors[0])
    if width == 0:
        raise EmbeddingSemanticUnavailable(
            "Embedding vectors were empty."
        )
    if any(len(vector) != width for vector in vectors):
        raise EmbeddingSemanticUnavailable(
            "Embedding prototype dimensions were inconsistent."
        )

    mean = tuple(
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(width)
    )
    return _normalize_vector(mean)


def _semantic_score(
    vector: Sequence[float],
    *,
    prototypes: Sequence[Sequence[float]],
    centroid: Sequence[float],
) -> float:
    nearest = max(
        _dot(vector, prototype)
        for prototype in prototypes
    )
    center = _dot(vector, centroid)
    return 0.74 * nearest + 0.26 * center


@dataclass(frozen=True, slots=True)
class _CacheData:
    prototypes: Mapping[str, tuple[tuple[float, ...], ...]]
    centroids: Mapping[str, tuple[float, ...]]
    dimension: int


def build_cache_payload(
    *,
    model: str,
    embeddings: Sequence[Sequence[float]],
) -> dict[str, object]:
    items = training_items()
    if len(embeddings) != len(items):
        raise ValueError(
            "Embedding count does not match semantic prototype count."
        )

    by_label: dict[str, list[list[float]]] = defaultdict(list)
    dimension: int | None = None

    for (label, _), raw_vector in zip(items, embeddings):
        vector = list(_normalize_vector(raw_vector))
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError(
                "Embedding prototype dimensions were inconsistent."
            )
        by_label[label].append(vector)

    if dimension is None:
        raise ValueError("Semantic prototype corpus was empty.")

    return {
        "cache_version": CACHE_VERSION,
        "model": model,
        "corpus_sha256": corpus_sha256(model),
        "dimension": dimension,
        "prototype_count": len(items),
        "prototypes": dict(by_label),
    }


def write_cache_payload(
    payload: Mapping[str, object],
    *,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_cache(
    *,
    path: Path,
    model: str,
) -> _CacheData:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except FileNotFoundError as error:
        raise EmbeddingSemanticUnavailable(
            f"Embedding semantic cache is missing: {path}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise EmbeddingSemanticUnavailable(
            "Embedding semantic cache could not be read."
        ) from error

    if payload.get("cache_version") != CACHE_VERSION:
        raise EmbeddingSemanticUnavailable(
            "Embedding semantic cache version mismatch."
        )
    if payload.get("model") != model:
        raise EmbeddingSemanticUnavailable(
            "Embedding semantic cache model mismatch."
        )
    if payload.get("corpus_sha256") != corpus_sha256(model):
        raise EmbeddingSemanticUnavailable(
            "Embedding semantic cache corpus fingerprint mismatch."
        )

    try:
        dimension = int(payload["dimension"])
        raw_prototypes = payload["prototypes"]
    except (KeyError, TypeError, ValueError) as error:
        raise EmbeddingSemanticUnavailable(
            "Embedding semantic cache metadata was invalid."
        ) from error

    if not isinstance(raw_prototypes, dict):
        raise EmbeddingSemanticUnavailable(
            "Embedding semantic cache prototypes were invalid."
        )

    required = set(FACT_HEADS) | set(SPEECH_ACT_HEADS)
    missing = required - set(raw_prototypes)
    if missing:
        raise EmbeddingSemanticUnavailable(
            "Embedding semantic cache is missing classes: "
            + ", ".join(sorted(missing))
        )

    prototypes: dict[
        str,
        tuple[tuple[float, ...], ...],
    ] = {}
    centroids: dict[str, tuple[float, ...]] = {}

    for label, raw_vectors in raw_prototypes.items():
        if not isinstance(raw_vectors, list) or not raw_vectors:
            raise EmbeddingSemanticUnavailable(
                f"Embedding class {label!r} was empty."
            )

        normalized = tuple(
            _normalize_vector(vector)
            for vector in raw_vectors
        )
        if any(len(vector) != dimension for vector in normalized):
            raise EmbeddingSemanticUnavailable(
                "Embedding semantic cache dimension mismatch."
            )

        prototypes[str(label)] = normalized
        centroids[str(label)] = _normalized_mean(normalized)

    return _CacheData(
        prototypes=prototypes,
        centroids=centroids,
        dimension=dimension,
    )


class CompositionalEmbeddingClassifier:
    """Closed-domain semantic classifier backed by local sentence embeddings."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        url: str = DEFAULT_EMBED_URL,
        timeout_seconds: float = 5.0,
        cache_path: Path | None = None,
    ) -> None:
        validate_loopback_url(url)
        if timeout_seconds <= 0:
            raise ValueError(
                "Embedding timeout_seconds must be positive."
            )

        self._model = model
        self._url = url
        self._timeout_seconds = timeout_seconds
        self._cache_path = cache_path or default_cache_path(model)
        self._cache: _CacheData | None = None

    @property
    def cache_path(self) -> Path:
        return self._cache_path

    def _ensure_cache(self) -> _CacheData:
        if self._cache is None:
            self._cache = _load_cache(
                path=self._cache_path,
                model=self._model,
            )
        return self._cache

    async def _embed(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        if not texts:
            raise EmbeddingSemanticUnavailable(
                "Cannot embed an empty semantic turn."
            )

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
            ) as client:
                response = await client.post(
                    self._url,
                    json={
                        "model": self._model,
                        "input": list(texts),
                        "truncate": True,
                        "keep_alive": "30m",
                    },
                )
            response.raise_for_status()
            payload = response.json()
            raw_embeddings = payload.get("embeddings")
        except (
            httpx.HTTPError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            raise EmbeddingSemanticUnavailable(
                "Local embedding request failed."
            ) from error

        if (
            not isinstance(raw_embeddings, list)
            or len(raw_embeddings) != len(texts)
        ):
            raise EmbeddingSemanticUnavailable(
                "Local embedding response had unexpected shape."
            )

        try:
            return tuple(
                _normalize_vector(vector)
                for vector in raw_embeddings
            )
        except (TypeError, ValueError) as error:
            raise EmbeddingSemanticUnavailable(
                "Local embedding response contained invalid vectors."
            ) from error

    def _scores(
        self,
        vector: Sequence[float],
        labels: Sequence[str],
        cache: _CacheData,
    ) -> dict[str, float]:
        if len(vector) != cache.dimension:
            raise EmbeddingSemanticUnavailable(
                "Live embedding dimension did not match prototype cache."
            )

        return {
            label: _semantic_score(
                vector,
                prototypes=cache.prototypes[label],
                centroid=cache.centroids[label],
            )
            for label in labels
        }

    @staticmethod
    def _compose(
        *,
        clause_fact_scores: Sequence[Mapping[str, float]],
        clause_speech_scores: Sequence[Mapping[str, float]],
    ) -> tuple[str, float, float]:
        aggregate_facts: dict[str, float] = defaultdict(float)
        aggregate_speech: dict[str, float] = defaultdict(float)

        for scores in clause_fact_scores:
            for label, score in scores.items():
                aggregate_facts[label] = max(
                    aggregate_facts[label],
                    score,
                )
        for scores in clause_speech_scores:
            for label, score in scores.items():
                aggregate_speech[label] = max(
                    aggregate_speech[label],
                    score,
                )

        # Strong high-specificity speech acts override incidental similarity to
        # supported patient fact questions.
        speech_candidates: list[tuple[float, str]] = []

        for fact_scores, speech_scores in zip(
            clause_fact_scores,
            clause_speech_scores,
        ):
            best_fact = max(
                fact_scores.values(),
                default=0.0,
            )

            for label, (
                threshold,
                lead,
            ) in SPEECH_OVERRIDE_RULES.items():
                score = speech_scores.get(label, 0.0)
                if (
                    score >= threshold
                    and score - best_fact >= lead
                ):
                    speech_candidates.append(
                        (score, label)
                    )

        if speech_candidates:
            speech_candidates.sort(reverse=True)
            for preferred in (
                "scheduling_complex",
                "profile_create_request",
                "presence_check",
                "unknown",
                "open_ended_help",
            ):
                matching = [
                    item
                    for item in speech_candidates
                    if item[1] == preferred
                ]
                if matching:
                    score = max(
                        item[0]
                        for item in matching
                    )
                    second = max(
                        (
                            value
                            for value in (
                                list(aggregate_facts.values())
                                + list(aggregate_speech.values())
                            )
                            if value < score
                        ),
                        default=0.0,
                    )
                    return (
                        preferred,
                        score,
                        max(0.0, score - second),
                    )

        # One atomic requested fact per semantic clause.
        selected_by_clause: list[
            tuple[str, float]
        ] = []

        for scores in clause_fact_scores:
            ranked = sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if not ranked:
                continue

            top_label, top_score = ranked[0]
            threshold = FACT_THRESHOLDS.get(
                top_label,
                DEFAULT_FACT_THRESHOLD,
            )
            if top_score >= threshold:
                selected_by_clause.append(
                    (top_label, top_score)
                )

        selected_labels = {
            label
            for label, _ in selected_by_clause
        }

        reason = (
            "visit_reason_request"
            in selected_labels
        )
        appointment_type = (
            "appointment_type_request"
            in selected_labels
        )

        if reason and appointment_type:
            reason_score = max(
                score
                for label, score
                in selected_by_clause
                if label == "visit_reason_request"
            )
            type_score = max(
                score
                for label, score
                in selected_by_clause
                if label
                == "appointment_type_request"
            )
            confidence = min(
                reason_score,
                type_score,
            )
            return (
                "visit_reason_and_type_request",
                confidence,
                0.0,
            )

        if selected_by_clause:
            predicted, score = max(
                selected_by_clause,
                key=lambda item: item[1],
            )

            if predicted == "visit_reason_request":
                open_score = aggregate_speech.get(
                    "open_ended_help",
                    0.0,
                )
                if (
                    open_score >= DEFAULT_FACT_THRESHOLD
                    and score - open_score
                    <= GENERIC_HELP_TIE_WINDOW
                ):
                    return (
                        "open_ended_help",
                        open_score,
                        max(
                            0.0,
                            open_score - score,
                        ),
                    )

            runner = max(
                (
                    value
                    for label, value
                    in aggregate_facts.items()
                    if label != predicted
                ),
                default=0.0,
            )
            return (
                predicted,
                score,
                max(0.0, score - runner),
            )

        ranked_speech = sorted(
            aggregate_speech.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked_speech:
            label, score = ranked_speech[0]
            if score >= DEFAULT_FACT_THRESHOLD:
                runner = (
                    ranked_speech[1][1]
                    if len(ranked_speech) > 1
                    else 0.0
                )
                return (
                    label,
                    score,
                    max(0.0, score - runner),
                )

        best_score = max(
            (
                list(aggregate_facts.values())
                + list(aggregate_speech.values())
            ),
            default=0.0,
        )
        return (
            "unknown",
            best_score,
            0.0,
        )

    async def classify(
        self,
        text: str,
    ) -> EmbeddingSemanticResult:
        cache = self._ensure_cache()
        clauses = split_clauses(text)
        if not clauses:
            clauses = (normalize_clause(text),)
        if not clauses or not clauses[0]:
            return EmbeddingSemanticResult(
                intent="unknown",
                confidence=0.0,
                score=0.0,
                margin=0.0,
                source=f"embedding_v31:{self._model}",
                clauses=(),
            )

        vectors = await self._embed(clauses)
        fact_scores = [
            self._scores(
                vector,
                FACT_HEADS,
                cache,
            )
            for vector in vectors
        ]
        speech_scores = [
            self._scores(
                vector,
                SPEECH_ACT_HEADS,
                cache,
            )
            for vector in vectors
        ]

        intent, score, margin = self._compose(
            clause_fact_scores=fact_scores,
            clause_speech_scores=speech_scores,
        )

        return EmbeddingSemanticResult(
            intent=intent,
            confidence=min(
                1.0,
                max(0.0, score),
            ),
            score=score,
            margin=margin,
            source=f"embedding_v31:{self._model}",
            clauses=clauses,
        )
