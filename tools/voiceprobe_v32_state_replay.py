#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from voiceprobe.v3.flow_state import (
    FlowStage,
    SchedulingFlowTracker,
)
from voiceprobe.v3.models import PatientFacts

from voiceprobe.v32.ollama_backend import (
    OllamaBackend,
    OllamaConfig,
)
from voiceprobe.v32.semantic_adapter import (
    resolve_semantic_turn,
)
from voiceprobe.v32.semantic_parser import SemanticParser
from voiceprobe.v32.semantic_policy import (
    SemanticRoute,
    route_semantic_frame,
)


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    utterance: str
    expected_route: SemanticRoute
    expected_added: frozenset[FlowStage]
    expect_decision: bool
    history: tuple[str, ...]


SCHEDULE = (
    "PGAI: How may I help you today?",
    "PATIENT: I need to schedule an appointment for Friday afternoon.",
)

RESCHEDULE = (
    "PGAI: You already have an appointment Tuesday at 2:15 PM.",
    "PATIENT: I'd like to reschedule it for Friday afternoon.",
)


CASES = (
    Case(
        name="provider_preference",
        utterance=(
            "Of course. Is there a specific provider you'd like "
            "to see, or would you like to see whoever is available "
            "Friday afternoon?"
        ),
        expected_route=SemanticRoute.ANSWER_FACT,
        expected_added=frozenset({
            FlowStage.PROVIDER,
        }),
        expect_decision=True,
        history=SCHEDULE,
    ),
    Case(
        name="visit_reason",
        utterance="What is the reason for your appointment?",
        expected_route=SemanticRoute.ANSWER_FACT,
        expected_added=frozenset({
            FlowStage.VISIT_REASON,
        }),
        expect_decision=True,
        history=SCHEDULE,
    ),
    Case(
        name="insurance",
        utterance="Who is your insurance through?",
        expected_route=SemanticRoute.ANSWER_FACT,
        expected_added=frozenset({
            FlowStage.INSURANCE,
        }),
        expect_decision=True,
        history=SCHEDULE,
    ),
    Case(
        name="acknowledgement",
        utterance="No problem.",
        expected_route=SemanticRoute.WAIT,
        expected_added=frozenset(),
        expect_decision=True,
        history=SCHEDULE,
    ),
    Case(
        name="slot_intro",
        utterance="Here are some Friday afternoon",
        expected_route=SemanticRoute.HOLD,
        expected_added=frozenset(),
        expect_decision=True,
        history=SCHEDULE,
    ),
    Case(
        name="reschedule_reason",
        utterance=(
            "What is the reason for changing your appointment?"
        ),
        expected_route=SemanticRoute.ANSWER_RESCHEDULE_REASON,
        expected_added=frozenset(),
        expect_decision=True,
        history=RESCHEDULE,
    ),
    Case(
        name="booking_permission",
        utterance=(
            "Should I go ahead and book that appointment?"
        ),
        expected_route=SemanticRoute.TRANSACTION_GATE,
        expected_added=frozenset(),
        expect_decision=False,
        history=RESCHEDULE,
    ),
    Case(
        name="existing_appointment_status",
        utterance=(
            "You already have an appointment Tuesday at 2:15 PM."
        ),
        expected_route=SemanticRoute.WAIT,
        expected_added=frozenset(),
        expect_decision=True,
        history=(),
    ),
)


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="qwen3.5:4b",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:11434/api/chat",
    )

    args = parser.parse_args()

    semantic = SemanticParser(
        backend=OllamaBackend(
            OllamaConfig(
                model=args.model,
                endpoint=args.endpoint,
                timeout_seconds=30.0,
                keep_alive="15m",
                num_ctx=1024,
                temperature=0.0,
            )
        )
    )

    facts = PatientFacts()

    # Warm outside scoring.
    await semantic.parse(
        remote_turn="Okay, thank you.",
        recent_dialogue=(),
    )

    passed = 0

    print(
        "========== V3.2 STATE-MUTATION REPLAY =========="
    )

    for index, case in enumerate(CASES, 1):
        tracker = SchedulingFlowTracker()

        before = tracker.snapshot()

        trace = await semantic.parse(
            remote_turn=case.utterance,
            recent_dialogue=case.history,
        )

        routed = route_semantic_frame(
            trace.frame
        )

        resolution = resolve_semantic_turn(
            routed,
            facts=facts,
        )

        decision_exists = (
            resolution.decision is not None
        )

        if resolution.decision is not None:
            after = tracker.apply_decision(
                resolution.decision
            )
        else:
            after = tracker.snapshot()

        added = (
            after.communicated
            - before.communicated
        )

        transaction_safe = (
            after.accepted_slot_text
            == before.accepted_slot_text
            and after.booking_confirmation_text
            == before.booking_confirmation_text
            and after.complete
            == before.complete
            and after.confirmed
            == before.confirmed
        )

        ok = (
            trace.validation_error is None
            and routed.route
            is case.expected_route
            and decision_exists
            is case.expect_decision
            and added
            == case.expected_added
            and transaction_safe
        )

        passed += int(ok)

        print()
        print(f"[{index}] {case.name}")
        print(" PGAI:", case.utterance)
        print(
            " semantic=",
            {
                "speech_act": trace.frame.speech_act.value,
                "operation": trace.frame.operation.value,
                "focus": trace.frame.focus.value,
                "commitment": trace.frame.commitment.value,
            },
        )
        print(
            " route=",
            routed.route.value,
        )

        if resolution.decision is None:
            print(" decision=None")
        else:
            print(
                " decision_kind=",
                resolution.decision.kind.value,
            )
            print(
                " decision_reason=",
                resolution.decision.reason,
            )
            print(
                " decision_text=",
                resolution.decision.text,
            )

        print(
            " added_stages=",
            sorted(x.value for x in added),
        )
        print(
            " transaction_state_unchanged=",
            transaction_safe,
        )
        print(" PASS=", ok)

    print()
    print("========== SUMMARY ==========")
    print(f"correct={passed}/{len(CASES)}")
    print(f"failed={len(CASES) - passed}")

    if passed != len(CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
