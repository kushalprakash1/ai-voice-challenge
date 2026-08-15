"""Small, reproducible ASR benchmark for VoiceProbe.

The benchmark intentionally separates ordinary word error rate from
fact-critical recognition. In a medical scheduling conversation,
mishearing "the" is much less important than mishearing "knee",
a patient's name, an appointment time, or another scenario fact.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_RESULTS_PATH = Path("artifacts/asr/benchmark_results.jsonl")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Metrics for one ASR model on one audio sample."""

    model: str
    audio_file: str
    reference_text: str
    hypothesis: str
    word_error_rate: float
    reference_words: int
    word_errors: int
    critical_terms_expected: tuple[str, ...]
    critical_terms_correct: tuple[str, ...]
    critical_term_accuracy: float
    audio_duration_seconds: float
    processing_seconds: float
    real_time_factor: float
    created_at: str


def normalize_text(text: str) -> str:
    """Normalize text before word-level comparison."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Return word-level Levenshtein distance."""
    previous = list(range(len(hypothesis) + 1))

    for ref_index, ref_word in enumerate(reference, start=1):
        current = [ref_index]

        for hyp_index, hyp_word in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_word == hyp_word else 1

            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + substitution_cost,
                )
            )

        previous = current

    return previous[-1]


def calculate_wer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """Calculate word error rate and return supporting counts."""
    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()

    if not reference_words:
        raise ValueError("Reference text cannot be empty.")

    errors = edit_distance(reference_words, hypothesis_words)
    wer = errors / len(reference_words)

    return wer, errors, len(reference_words)


def score_critical_terms(
    hypothesis: str,
    critical_terms: tuple[str, ...],
) -> tuple[tuple[str, ...], float]:
    """Measure whether scenario-critical terms survived transcription."""
    normalized_hypothesis = normalize_text(hypothesis)

    correct = tuple(
        term for term in critical_terms if normalize_text(term) in normalized_hypothesis
    )

    accuracy = len(correct) / len(critical_terms) if critical_terms else 1.0

    return correct, accuracy


def build_result(
    *,
    model: str,
    audio_file: str,
    reference_text: str,
    hypothesis: str,
    critical_terms: tuple[str, ...],
    audio_duration_seconds: float,
    processing_seconds: float,
) -> BenchmarkResult:
    """Calculate all metrics for one transcription."""
    if audio_duration_seconds <= 0:
        raise ValueError("Audio duration must be greater than zero.")

    if processing_seconds < 0:
        raise ValueError("Processing time cannot be negative.")

    wer, errors, reference_word_count = calculate_wer(
        reference_text,
        hypothesis,
    )

    correct_terms, critical_accuracy = score_critical_terms(
        hypothesis,
        critical_terms,
    )

    return BenchmarkResult(
        model=model,
        audio_file=audio_file,
        reference_text=reference_text,
        hypothesis=hypothesis,
        word_error_rate=wer,
        reference_words=reference_word_count,
        word_errors=errors,
        critical_terms_expected=critical_terms,
        critical_terms_correct=correct_terms,
        critical_term_accuracy=critical_accuracy,
        audio_duration_seconds=audio_duration_seconds,
        processing_seconds=processing_seconds,
        real_time_factor=processing_seconds / audio_duration_seconds,
        created_at=datetime.now(UTC).isoformat(),
    )


def save_result(
    result: BenchmarkResult,
    path: Path = DEFAULT_RESULTS_PATH,
) -> None:
    """Append one benchmark result as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(result)) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one VoiceProbe ASR transcription."
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument(
        "--critical-term",
        action="append",
        default=[],
        help="Repeat this argument for each important term.",
    )
    parser.add_argument("--audio-seconds", required=True, type=float)
    parser.add_argument("--processing-seconds", required=True, type=float)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = build_result(
        model=args.model,
        audio_file=args.audio,
        reference_text=args.reference,
        hypothesis=args.hypothesis,
        critical_terms=tuple(args.critical_term),
        audio_duration_seconds=args.audio_seconds,
        processing_seconds=args.processing_seconds,
    )

    save_result(result, args.results)

    print(f"Model: {result.model}")
    print(
        f"WER: {result.word_error_rate:.2%} "
        f"({result.word_errors}/{result.reference_words} word errors)"
    )
    print(f"Critical-term accuracy: {result.critical_term_accuracy:.2%}")
    print(
        "Critical terms correct:",
        ", ".join(result.critical_terms_correct) or "none",
    )
    print(f"Processing time: {result.processing_seconds:.3f}s")
    print(f"Real-time factor: {result.real_time_factor:.3f}")
    print(f"Saved: {args.results}")


if __name__ == "__main__":
    main()
