#!/usr/bin/env python3

"""Replay historically difficult PGAI utterances through v3.2 semantics.

These cases come from prior VoiceProbe live-call artifacts and regression
evidence. No telephony is opened by this tool.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
from dataclasses import dataclass

from voiceprobe.v32.ollama_backend import (
    OllamaBackend,
    OllamaConfig,
)
from voiceprobe.v32.semantic_parser import SemanticParser
from voiceprobe.v32.semantic_policy import (
    SemanticRoute,
    route_semantic_frame,
)


@dataclass(frozen=True, slots=True)
class ReplayCase:
    name: str
    utterance: str
    expected_route: SemanticRoute
    expected_focus: str | None = None
    history: tuple[str, ...] = ()


SCHEDULING_HISTORY = (
    "PGAI: How may I help you today?",
    "PATIENT: I need to schedule an appointment for Friday afternoon.",
    "PGAI: Can you tell me the reason for your visit?",
    "PATIENT: I have right shoulder pain.",
)

RESCHEDULE_HISTORY = (
    "PGAI: You already have an appointment Tuesday at 2:15 PM.",
    "PATIENT: I'd like to reschedule it. I'm looking for Friday afternoon.",
)

PROVIDER_HISTORY = (
    *SCHEDULING_HISTORY,
    "PGAI: I'm checking Friday afternoon availability.",
)


CASES = (
    # ---------------------------------------------------------
    # Actual provider-preference language from failed live run.
    # ---------------------------------------------------------
    ReplayCase(
        name="historical_provider_preference_1",
        utterance=(
            "Thanks for sharing. Do you have a preferred provider, "
            "or are you open to any available provider for your "
            "Friday afternoon appointment?"
        ),
        expected_route=SemanticRoute.ANSWER_FACT,
        expected_focus="provider_preference",
        history=PROVIDER_HISTORY,
    ),
    ReplayCase(
        name="historical_provider_preference_2",
        utterance=(
            "Of course. Is there a specific provider you'd like to see, "
            "or would you like to see whoever is available Friday afternoon?"
        ),
        expected_route=SemanticRoute.ANSWER_FACT,
        expected_focus="provider_preference",
        history=PROVIDER_HISTORY,
    ),
    ReplayCase(
        name="historical_provider_preference_casual",
        utterance=(
            "you wanna see a certain provider, or is anyone available "
            "on Friday afternoon okay for your appointment?"
        ),
        expected_route=SemanticRoute.ANSWER_FACT,
        expected_focus="provider_preference",
        history=PROVIDER_HISTORY,
    ),

    # ---------------------------------------------------------
    # Actual acknowledgement that v3.1 unnecessarily answered.
    # ---------------------------------------------------------
    ReplayCase(
        name="historical_no_problem",
        utterance="No problem.",
        expected_route=SemanticRoute.WAIT,
        history=PROVIDER_HISTORY,
    ),

    # ---------------------------------------------------------
    # Actual incomplete live-ASR utterances.
    # ---------------------------------------------------------
    ReplayCase(
        name="historical_slot_intro_fragment",
        utterance="Here are some Friday afternoon",
        expected_route=SemanticRoute.HOLD,
        history=PROVIDER_HISTORY,
    ),
    ReplayCase(
        name="historical_for_example_fragment",
        utterance="For example,",
        expected_route=SemanticRoute.HOLD,
        history=SCHEDULING_HISTORY,
    ),

    # ---------------------------------------------------------
    # Visit-reason language that historically caused repeated
    # clarification loops.
    # ---------------------------------------------------------
    ReplayCase(
        name="contrast_visit_purpose",
        utterance="What is this visit for?",
        expected_route=SemanticRoute.ANSWER_FACT,
        expected_focus="complaint",
        history=SCHEDULING_HISTORY[:2],
    ),
    ReplayCase(
        name="contrast_what_brings_patient_in",
        utterance="What brings you in today?",
        expected_route=SemanticRoute.ANSWER_FACT,
        expected_focus="complaint",
        history=SCHEDULING_HISTORY[:2],
    ),
    ReplayCase(
        name="contrast_reschedule_why",
        utterance="What made you decide to move the existing visit?",
        expected_route=SemanticRoute.ANSWER_RESCHEDULE_REASON,
        history=RESCHEDULE_HISTORY,
    ),

    ReplayCase(
        name="historical_visit_reason_1",
        utterance="Can you tell me the reason for your visit?",
        expected_route=SemanticRoute.ANSWER_FACT,
        expected_focus="complaint",
        history=SCHEDULING_HISTORY[:2],
    ),
    ReplayCase(
        name="historical_visit_reason_2",
        utterance="What is the reason for your appointment?",
        expected_route=SemanticRoute.ANSWER_FACT,
        expected_focus="complaint",
        history=SCHEDULING_HISTORY[:2],
    ),
    ReplayCase(
        name="historical_visit_reason_with_example",
        utterance=(
            "What is the reason for your appointment? "
            "For example, is it for a routine visit, a follow-up, "
            "or a specific concern?"
        ),
        expected_route=SemanticRoute.ANSWER_FACT,
        expected_focus="complaint",
        history=SCHEDULING_HISTORY[:2],
    ),

    # ---------------------------------------------------------
    # Persistent appointment behavior.
    # ---------------------------------------------------------
    ReplayCase(
        name="existing_appointment_status",
        utterance=(
            "You already have an appointment Tuesday at 2:15 PM."
        ),
        expected_route=SemanticRoute.WAIT,
        history=(),
    ),

    # ---------------------------------------------------------
    # The language that triggered the newest architecture work.
    # ---------------------------------------------------------
    ReplayCase(
        name="reschedule_reason_failure",
        utterance=(
            "What is the reason for changing your appointment?"
        ),
        expected_route=SemanticRoute.ANSWER_RESCHEDULE_REASON,
        history=RESCHEDULE_HISTORY,
    ),
    ReplayCase(
        name="reschedule_reason_variant",
        utterance=(
            "Why do you need to move your existing appointment?"
        ),
        expected_route=SemanticRoute.ANSWER_RESCHEDULE_REASON,
        history=RESCHEDULE_HISTORY,
    ),

    # ---------------------------------------------------------
    # Transaction authority must remain outside the model.
    # ---------------------------------------------------------
    ReplayCase(
        name="booking_permission",
        utterance="Should I go ahead and book that appointment?",
        expected_route=SemanticRoute.TRANSACTION_GATE,
        history=RESCHEDULE_HISTORY,
    ),
)


async def main() -> None:
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

    backend = OllamaBackend(
        OllamaConfig(
            model=args.model,
            endpoint=args.endpoint,
            timeout_seconds=30.0,
            keep_alive="15m",
            num_ctx=1024,
            temperature=0.0,
        )
    )

    semantic = SemanticParser(
        backend=backend,
    )

    print("========== V3.2 HISTORICAL REPLAY ==========")
    print("model=", args.model)
    print("endpoint=", args.endpoint)

    # Explicit warmup outside scored cases.
    warm = await semantic.parse(
        remote_turn="Okay, thank you.",
        recent_dialogue=(),
    )

    print(
        "warmup_ms=",
        round(warm.latency_ms, 1),
    )

    correct = 0
    validation_failures = 0
    latencies: list[float] = []

    for index, case in enumerate(CASES, 1):
        trace = await semantic.parse(
            remote_turn=case.utterance,
            recent_dialogue=case.history,
        )

        routed = route_semantic_frame(
            trace.frame
        )

        route_ok = (
            routed.route
            is case.expected_route
        )

        focus_ok = True

        if case.expected_focus is not None:
            focus_ok = (
                routed.fact_focus.value
                == case.expected_focus
            )

        valid = (
            trace.validation_error is None
        )

        passed = (
            route_ok
            and focus_ok
            and valid
        )

        correct += int(passed)
        validation_failures += int(
            not valid
        )
        latencies.append(
            trace.latency_ms
        )

        frame = trace.frame

        print()
        print(
            f"[{index:02d}] {case.name}"
        )
        print(
            " PGAI:",
            case.utterance,
        )
        print(
            " frame:",
            {
                "speech_act": frame.speech_act.value,
                "operation": frame.operation.value,
                "focus": frame.focus.value,
                "commitment": frame.commitment.value,
                "certainty": frame.certainty.value,
            },
        )
        print(
            " route=",
            routed.route.value,
        )
        print(
            " expected=",
            case.expected_route.value,
        )

        if case.expected_focus is not None:
            print(
                " expected_focus=",
                case.expected_focus,
            )
            print(
                " routed_focus=",
                routed.fact_focus.value,
            )

        if trace.validation_error:
            print(
                " validation_error=",
                trace.validation_error.splitlines()[0],
            )

        print(
            " latency_ms=",
            round(trace.latency_ms, 1),
        )
        print(
            " PASS=",
            passed,
        )

    print()
    print("========== SUMMARY ==========")
    print(
        f"correct={correct}/{len(CASES)}"
    )
    print(
        f"validation_failures={validation_failures}"
    )
    print(
        "median_ms=",
        round(
            statistics.median(
                latencies
            ),
            1,
        ),
    )
    print(
        "max_ms=",
        round(
            max(latencies),
            1,
        ),
    )

    failed = (
        len(CASES) - correct
    )

    print(
        f"failed={failed}"
    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
