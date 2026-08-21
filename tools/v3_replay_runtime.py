#!/usr/bin/env python3
"""Replay both stored live-call transcript corpora through VoiceProbe v3 runtime."""

from __future__ import annotations

import asyncio
from collections import defaultdict

from voiceprobe.v3.corpus import load_regression_cases
from voiceprobe.v3.runtime import VoiceProbeV3Runtime


async def main() -> None:
    grouped = defaultdict(list)

    for case in load_regression_cases():
        grouped[case["call_uuid"]].append(case)

    total = 0
    passed = 0
    critical_total = 0
    critical_passed = 0
    aggregate_fallback = 0
    aggregate_latency_ms = 0.0

    print("========== VOICEPROBE V3 RUNTIME REPLAY ==========")

    for call_uuid, cases in grouped.items():
        runtime = VoiceProbeV3Runtime()

        print()
        print("CALL:", call_uuid)

        for case in sorted(cases, key=lambda item: item["ordinal"]):
            result = await runtime.process_turns(
                [case["agent_turn"]],
                ingress_reason="stored_live_call_replay",
            )

            expected = case["expected_kind"]
            kind_ok = result.decision.kind.value == expected
            text_ok = all(
                token.casefold() in result.decision.text.casefold()
                for token in case["expected_text_contains"]
            )

            total += 1

            if case["critical"]:
                critical_total += 1

            if kind_ok and text_ok:
                passed += 1

                if case["critical"]:
                    critical_passed += 1
            else:
                print(
                    "FAIL",
                    case["ordinal"],
                    "expected=",
                    expected,
                    "actual=",
                    result.decision.kind.value,
                    "text=",
                    repr(result.decision.text),
                )

        metrics = runtime.metrics
        aggregate_fallback += metrics.fallback_decisions
        aggregate_latency_ms += metrics.total_policy_latency_ms

        print(
            " decisions=",
            metrics.total_decisions,
            " fallback=",
            metrics.fallback_decisions,
            " avg_policy_ms=",
            f"{metrics.average_policy_latency_ms:.3f}",
            " max_policy_ms=",
            f"{metrics.max_policy_latency_ms:.3f}",
            sep="",
        )

    print()
    print("========== SUMMARY ==========")
    print(f"cases:             {passed}/{total}")
    print(f"critical:          {critical_passed}/{critical_total}")
    print(f"fallback decisions:{aggregate_fallback}")
    print(f"total policy ms:   {aggregate_latency_ms:.3f}")

    if passed != total:
        raise SystemExit(1)

    print()
    print("TWO-CALL V3 RUNTIME REPLAY: PASS")


if __name__ == "__main__":
    asyncio.run(main())
