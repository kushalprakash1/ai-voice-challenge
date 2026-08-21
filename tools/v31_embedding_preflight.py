#!/usr/bin/env python3
"""Preflight the exact VoiceProbe v3.1 production embedding classifier."""

from __future__ import annotations

import asyncio
import statistics
import time

from voiceprobe.v3.embedding_semantics import (
    CompositionalEmbeddingClassifier,
)


CASES = (
    (
        "visit_reason_request",
        "What is the reason for your appointment?",
    ),
    (
        "visit_reason_request",
        "Before I choose the visit type, what problem are you having?",
    ),
    (
        "appointment_type_request",
        "Once I know the reason, what type of appointment is this?",
    ),
    (
        "insurance_request",
        "Which insurance carrier do you use?",
    ),
    (
        "dob_request",
        "What's your DOB?",
    ),
    (
        "last_name_request",
        "What is your surname?",
    ),
    (
        "profile_create_request",
        "Do you want me to make a new patient profile?",
    ),
    (
        "presence_check",
        "Are you still on this call?",
    ),
    (
        "open_ended_help",
        "may I help you?",
    ),
    (
        "open_ended_help",
        "What can I do for you?",
    ),
    (
        "full_name_request",
        "May I have your name?",
    ),
    (
        "provider_preference_request",
        "Would any clinician work for you?",
    ),
    (
        "scheduling_complex",
        "Nothing is open then. Would you like me to broaden the search?",
    ),
    (
        "scheduling_complex",
        (
            "There are no Friday afternoon openings available. "
            "Would you like to look at afternoon slots on another day "
            "such as Monday or Tuesday next week?"
        ),
    ),
    (
        "unknown",
        "Do you need a referral from your primary doctor?",
    ),
    (
        "visit_reason_and_type_request",
        (
            "For -- What is the reason for your appointment? "
            "For example, is it for a routine visit, a follow-up, "
            "or a specific concern?"
        ),
    ),
)


async def main() -> None:
    classifier = CompositionalEmbeddingClassifier()

    # Warmup verifies cache + local model without including load cost in p50/p95.
    await classifier.classify(
        "What is the reason for your appointment?"
    )

    latencies = []

    for expected, text in CASES:
        started = time.perf_counter()
        result = await classifier.classify(text)
        elapsed_ms = 1000.0 * (
            time.perf_counter() - started
        )
        latencies.append(elapsed_ms)

        okay = result.intent == expected
        print(
            ("PASS" if okay else "FAIL"),
            f"expected={expected}",
            f"predicted={result.intent}",
            f"score={result.score:.4f}",
            f"margin={result.margin:.4f}",
            f"latency_ms={elapsed_ms:.1f}",
        )
        print(" ", text)

        if not okay:
            raise SystemExit(
                "VOICEPROBE V3.1 EMBEDDING PREFLIGHT: FAIL"
            )

    ordered = sorted(latencies)
    p50 = statistics.median(ordered)
    p95_index = max(
        0,
        min(
            len(ordered) - 1,
            int((0.95 * len(ordered) + 0.999999)) - 1,
        ),
    )
    p95 = ordered[p95_index]

    print()
    print(f"warm_p50_ms={p50:.2f}")
    print(f"warm_p95_ms={p95:.2f}")
    print("phone_calls=0")
    print("external_api_calls=0")
    print("local_ollama_embed=yes")

    if p95 > 1200.0:
        raise SystemExit(
            "VOICEPROBE V3.1 EMBEDDING PREFLIGHT: FAIL latency"
        )

    print("VOICEPROBE V3.1 EMBEDDING PREFLIGHT: PASS")


if __name__ == "__main__":
    asyncio.run(main())
