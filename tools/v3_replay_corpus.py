#!/usr/bin/env python3
"""Replay the two-call v3 regression corpus through the deterministic fast path."""

from __future__ import annotations

from voiceprobe.v3.corpus import load_regression_cases
from voiceprobe.v3.fast_policy import RoutineSchedulingPolicy


def main() -> None:
    policy = RoutineSchedulingPolicy()
    cases = load_regression_cases()

    failures: list[str] = []
    fallback = 0
    critical_total = 0
    critical_passed = 0

    for case in cases:
        decision = policy.decide(case["agent_turn"])
        expected = case["expected_kind"]
        critical = bool(case["critical"])

        if critical:
            critical_total += 1

        kind_ok = decision.kind.value == expected
        text_ok = all(
            token.casefold() in decision.text.casefold()
            for token in case["expected_text_contains"]
        )

        if decision.kind.value == "fallback":
            fallback += 1

        if kind_ok and text_ok:
            if critical:
                critical_passed += 1
            continue

        failures.append(
            (
                f'{case["call_uuid"]}#{case["ordinal"]}: '
                f'expected={expected!r} got={decision.kind.value!r}; '
                f'expected_text_contains={case["expected_text_contains"]!r}; '
                f'actual_text={decision.text!r}; '
                f'agent_turn={case["agent_turn"]!r}'
            )
        )

    print("========== VOICEPROBE V3 CORPUS REPLAY ==========")
    print(f"cases:           {len(cases)}")
    print(f"critical:        {critical_passed}/{critical_total}")
    print(f"fallback turns:  {fallback}")
    print(f"failures:        {len(failures)}")

    if failures:
        print()
        print("FAILURES")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)

    print()
    print("TWO-CALL ROUTINE POLICY REPLAY: PASS")


if __name__ == "__main__":
    main()
