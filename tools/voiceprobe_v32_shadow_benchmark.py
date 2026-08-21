#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import statistics

from voiceprobe.v3.flow_state import (
    FlowSnapshot,
    FlowStage,
)
from voiceprobe.v3.models import PatientFacts
from voiceprobe.v32.ollama_backend import (
    OllamaBackend,
    OllamaConfig,
)
from voiceprobe.v32.reasoner import (
    ContextualReasoner,
)


CASES = (
    (
        "What is the reason you're changing "
        "your appointment?"
    ),
    "Why do you need to move the appointment?",
    (
        "Can I ask why the current appointment "
        "no longer works?"
    ),
    (
        "What's prompting you to change "
        "the scheduled visit?"
    ),
    (
        "May I document why you'd like "
        "to reschedule?"
    ),
    (
        "What changed that makes you need "
        "a different appointment?"
    ),
    (
        "Is there a particular reason "
        "you're moving your visit?"
    ),
    (
        "Could you tell me why you want "
        "to switch appointment times?"
    ),
)


def snapshot():
    return FlowSnapshot(
        communicated=frozenset(
            {
                FlowStage.PROFILE,
                FlowStage.IDENTITY,
                FlowStage.DOB,
                FlowStage.VISIT_REASON,
                FlowStage.APPOINTMENT_TYPE,
                FlowStage.DATE_TIME,
            }
        ),
        confirmed=frozenset(),
        current_stage=FlowStage.PROVIDER,
        complete=False,
        accepted_slot_text=None,
        booking_confirmation_text=None,
        allow_earlier_week_afternoons=False,
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="qwen3.5:4b",
    )
    args = parser.parse_args()

    reasoner = ContextualReasoner(
        backend=OllamaBackend(
            OllamaConfig(
                model=args.model
            )
        ),
        facts=PatientFacts(),
    )

    history = (
        (
            "PGAI: You already have an "
            "appointment Tuesday at 2:15 PM."
        ),
        (
            "PATIENT: I'd like to reschedule it. "
            "I'm looking for Friday afternoon."
        ),
    )

    latencies = []
    successes = 0

    print(
        "========== VOICEPROBE v3.2 "
        "SHADOW BENCHMARK =========="
    )
    print("model=", args.model)

    for i, turn in enumerate(CASES, 1):
        trace = await reasoner.reason(
            remote_turn=turn,
            snapshot=snapshot(),
            recent_dialogue=history,
        )

        passed = (
            trace.decision.kind.value
            not in {
                "fallback",
                "clarify",
            }
            and bool(
                trace.decision.text.strip()
            )
        )

        successes += int(passed)
        latencies.append(trace.total_ms)

        print()
        print(f"[{i}] PGAI: {turn}")
        print(
            "    meaning:",
            trace.rewrite.meaning,
        )
        print(
            "    subject:",
            trace.rewrite.subject,
        )
        print(
            "    action:",
            trace.proposal.action.value,
        )
        print(
            "    patient:",
            trace.decision.text,
        )
        print(
            "    reason:",
            trace.decision.reason,
        )
        print(
            "    latency_ms:",
            round(trace.total_ms, 1),
        )
        print(
            "    pass:",
            passed,
        )

    print()
    print("========== SUMMARY ==========")
    print(
        f"passed={successes}/{len(CASES)}"
    )
    print(
        "pass_rate=",
        round(
            successes / len(CASES),
            3,
        ),
    )
    print(
        "median_latency_ms=",
        round(
            statistics.median(
                latencies
            ),
            1,
        ),
    )
    print(
        "max_latency_ms=",
        round(max(latencies), 1),
    )


if __name__ == "__main__":
    asyncio.run(main())
