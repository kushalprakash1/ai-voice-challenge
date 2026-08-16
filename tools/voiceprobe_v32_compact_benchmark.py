#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import statistics

from voiceprobe.v3.flow_state import FlowSnapshot, FlowStage
from voiceprobe.v3.models import PatientFacts
from voiceprobe.v32.ollama_backend import OllamaBackend, OllamaConfig
from voiceprobe.v32.reasoner import ContextualReasoner


CASES = (
    "What is the reason you're changing your appointment?",
    "Why do you need to move the appointment?",
    "May I document why you'd like to reschedule?",
    "What's prompting you to switch to another appointment time?",
    "What changed with the original time?",
    "Can I note why Tuesday no longer works for you?",
    "Why would Friday work better?",
    "What's the reason for moving the visit?",
)


def snapshot():
    return FlowSnapshot(
        communicated=frozenset(),
        confirmed=frozenset(),
        current_stage=FlowStage.DATE_TIME,
        complete=False,
        accepted_slot_text=None,
        booking_confirmation_text=None,
        allow_earlier_week_afternoons=False,
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    backend = OllamaBackend(
        OllamaConfig(
            model=args.model,
            keep_alive="5m",
        )
    )

    reasoner = ContextualReasoner(
        backend=backend,
        facts=PatientFacts(),
    )

    history = (
        "PGAI: You already have an appointment Tuesday at 2:15 PM.",
        "PATIENT: I'd like to reschedule it for Friday afternoon.",
    )

    passed = 0
    times = []

    for index, turn in enumerate(CASES, 1):
        trace = await reasoner.reason(
            remote_turn=turn,
            snapshot=snapshot(),
            recent_dialogue=history,
        )

        p = trace.proposal
        response = trace.decision.text.casefold()

        ok = (
            p.action.value == "answer"
            and "friday" in response
            and "afternoon" in response
            and "shoulder" not in response
            and "pain" not in response
            and trace.decision.kind.value != "clarify"
        )

        passed += int(ok)
        times.append(trace.total_ms)

        print()
        print(f"[{index}] {turn}")
        print(" meaning=", p.meaning)
        print(" risk=", p.risk.value)
        print(" action=", p.action.value)
        print(" grounding=", p.grounding.value)
        print(" response=", trace.decision.text)
        print(" confidence=", p.confidence)

        if trace.validation_error:
            first_error_line = trace.validation_error.splitlines()[0]
            print(" validation_error=", first_error_line)

        print(" latency_ms=", round(trace.total_ms, 1))
        print(" PASS=", ok)

    print()
    print("========== SUMMARY ==========")
    print(f"model={args.model}")
    print(f"correct={passed}/{len(CASES)}")
    print(
        "median_ms=",
        round(statistics.median(times), 1),
    )
    print(
        "max_ms=",
        round(max(times), 1),
    )


if __name__ == "__main__":
    asyncio.run(main())
