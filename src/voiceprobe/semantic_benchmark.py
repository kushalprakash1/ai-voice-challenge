"""Semantic-interpreter benchmark for VoiceProbe.

This benchmark measures whether natural scheduling utterances are mapped
to the intended semantic representation before PatientBrain is allowed
to consume them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

import httpx
from pydantic import BaseModel, ConfigDict

from voiceprobe.conversation.grounding import ground_turn_meaning
from voiceprobe.conversation.normalization import normalize_turn_meaning
from voiceprobe.conversation.state import FactKey, build_initial_state
from voiceprobe.interpreters.ollama import OllamaConversationInterpreter
from voiceprobe.scenarios.models import PatientFacts, PatientScenario

DEFAULT_CASES = Path("tests/data/semantic_cases.jsonl")
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_URL = "http://127.0.0.1:11434/api/chat"


class ExpectedOffer(BaseModel):
    """Expected appointment slot extracted from one benchmark case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    day: str | None = None
    time: str | None = None


class SemanticCase(BaseModel):
    """One labeled semantic benchmark utterance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    utterance: str

    expected_requested_facts: tuple[FactKey, ...] = ()
    expected_stated_fact_keys: tuple[FactKey, ...] = ()
    expected_conflicts: tuple[FactKey, ...] = ()

    expected_appointment_offer: ExpectedOffer | None = None

    expected_booking_confirmed: bool = False
    expected_requests_repetition: bool = False
    expected_unclear: bool = False


def normalize(value: str | None) -> str | None:
    """Normalize optional semantic text for benchmark comparison."""
    if value is None:
        return None

    return " ".join(value.lower().split())


def load_cases(path: Path) -> tuple[SemanticCase, ...]:
    """Load JSONL benchmark cases."""
    cases: list[SemanticCase] = []

    for line_number, raw_line in enumerate(
        path.read_text().splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on {path}:{line_number}") from error

        cases.append(SemanticCase.model_validate(payload))

    if not cases:
        raise ValueError("Semantic benchmark contains no cases.")

    return tuple(cases)


def build_scenario() -> PatientScenario:
    """Build the fixed scenario used by the semantic benchmark."""
    return PatientScenario(
        scenario_id="semantic-benchmark",
        objective="Schedule an appointment for Friday afternoon.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            date_of_birth="January 12, 1998",
            insurance="Blue Cross",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
    )


def compare_case(
    *,
    case: SemanticCase,
    meaning: object,
    grounded: object,
) -> tuple[str, ...]:
    """Return human-readable mismatches for one benchmark case."""
    from voiceprobe.conversation.grounding import GroundedTurnMeaning
    from voiceprobe.conversation.meaning import TurnMeaning

    if not isinstance(meaning, TurnMeaning):
        raise TypeError("meaning must be TurnMeaning.")

    if not isinstance(grounded, GroundedTurnMeaning):
        raise TypeError("grounded must be GroundedTurnMeaning.")

    failures: list[str] = []

    if set(meaning.requested_facts) != set(case.expected_requested_facts):
        failures.append(
            "requested_facts "
            f"expected={sorted(case.expected_requested_facts)!r} "
            f"actual={sorted(meaning.requested_facts)!r}"
        )

    stated_keys = {assertion.fact for assertion in meaning.stated_facts}

    if stated_keys != set(case.expected_stated_fact_keys):
        failures.append(
            "stated_fact_keys "
            f"expected={sorted(case.expected_stated_fact_keys)!r} "
            f"actual={sorted(stated_keys)!r}"
        )

    conflict_keys = {conflict.fact for conflict in grounded.conflicts}

    if conflict_keys != set(case.expected_conflicts):
        failures.append(
            "conflicts "
            f"expected={sorted(case.expected_conflicts)!r} "
            f"actual={sorted(conflict_keys)!r}"
        )

    expected_offer = case.expected_appointment_offer
    actual_offer = meaning.appointment_offer

    if expected_offer is None:
        if actual_offer is not None:
            failures.append(
                f"appointment_offer expected=None actual={actual_offer.model_dump()!r}"
            )
    elif actual_offer is None:
        failures.append(
            f"appointment_offer expected={expected_offer.model_dump()!r} actual=None"
        )
    else:
        expected_day = normalize(expected_offer.day)
        expected_time = normalize(expected_offer.time)
        actual_day = normalize(actual_offer.day)
        actual_time = normalize(actual_offer.time)

        if expected_day != actual_day or expected_time != actual_time:
            failures.append(
                "appointment_offer "
                f"expected={(expected_day, expected_time)!r} "
                f"actual={(actual_day, actual_time)!r}"
            )

    if meaning.booking_confirmed is not case.expected_booking_confirmed:
        failures.append(
            "booking_confirmed "
            f"expected={case.expected_booking_confirmed} "
            f"actual={meaning.booking_confirmed}"
        )

    if meaning.requests_repetition is not case.expected_requests_repetition:
        failures.append(
            "requests_repetition "
            f"expected={case.expected_requests_repetition} "
            f"actual={meaning.requests_repetition}"
        )

    if meaning.unclear is not case.expected_unclear:
        failures.append(
            f"unclear expected={case.expected_unclear} actual={meaning.unclear}"
        )

    return tuple(failures)


def percentile_95(values: list[float]) -> float:
    """Return the nearest-rank 95th percentile."""
    if not values:
        raise ValueError("Cannot calculate percentile of empty data.")

    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)

    return ordered[index]


def main() -> None:
    """Run the semantic benchmark against an Ollama interpreter."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
    )

    args = parser.parse_args()

    cases = load_cases(args.cases)
    scenario = build_scenario()
    state = build_initial_state(scenario)

    interpreter = OllamaConversationInterpreter(
        model=args.model,
        url=args.url,
        timeout_seconds=20.0,
    )

    passed = 0
    latencies: list[float] = []

    try:
        for case in cases:
            started = perf_counter()

            try:
                raw_meaning = interpreter.interpret(
                    scenario=scenario,
                    state=state,
                    agent_turn=case.utterance,
                )

                meaning = normalize_turn_meaning(
                    raw_meaning,
                    agent_turn=case.utterance,
                )

                grounded = ground_turn_meaning(
                    scenario=scenario,
                    meaning=meaning,
                )

                elapsed = perf_counter() - started
                latencies.append(elapsed)

                failures = compare_case(
                    case=case,
                    meaning=meaning,
                    grounded=grounded,
                )

            except (
                httpx.HTTPError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                elapsed = perf_counter() - started
                latencies.append(elapsed)

                print(
                    f"FAIL {case.case_id:<28} "
                    f"{elapsed:>6.3f}s  "
                    f"{type(error).__name__}: {error}"
                )
                continue

            if failures:
                print(f"FAIL {case.case_id:<28} {elapsed:>6.3f}s")

                for failure in failures:
                    print(f"     {failure}")

                print(f"     utterance={case.utterance!r}")
            else:
                passed += 1

                print(f"PASS {case.case_id:<28} {elapsed:>6.3f}s")

    finally:
        interpreter.close()

    total = len(cases)
    failed = total - passed
    accuracy = passed / total

    print()
    print("=" * 72)
    print("VOICEPROBE SEMANTIC BENCHMARK")
    print(f"Model:           {args.model}")
    print(f"Cases:           {total}")
    print(f"Passed:          {passed}")
    print(f"Failed:          {failed}")
    print(f"Exact accuracy:  {accuracy:.1%}")
    print(f"Mean latency:    {mean(latencies):.3f}s")
    print(f"P95 latency:     {percentile_95(latencies):.3f}s")


if __name__ == "__main__":
    main()
