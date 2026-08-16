
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import statistics

from voiceprobe.v32.ollama_backend import (
    OllamaBackend,
    OllamaConfig,
)
from voiceprobe.v32.semantic_parser import SemanticParser
from voiceprobe.v32.semantic_policy import (
    SemanticRoute,
    route_semantic_frame,
)


CASES = (
    (
        "What is the reason you're changing your appointment?",
        SemanticRoute.ANSWER_RESCHEDULE_REASON,
        None,
    ),
    (
        "Why do you need to move the appointment?",
        SemanticRoute.ANSWER_RESCHEDULE_REASON,
        None,
    ),
    (
        "May I document why you'd like to reschedule?",
        SemanticRoute.ANSWER_RESCHEDULE_REASON,
        None,
    ),
    (
        "What's prompting you to switch to another appointment time?",
        SemanticRoute.ANSWER_RESCHEDULE_REASON,
        None,
    ),
    (
        "Who is your insurance through?",
        SemanticRoute.ANSWER_FACT,
        "insurance",
    ),
    (
        "Should I go ahead and book that appointment?",
        SemanticRoute.TRANSACTION_GATE,
        None,
    ),
    (
        "No problem.",
        SemanticRoute.WAIT,
        None,
    ),
    (
        "Would you like the first available provider?",
        SemanticRoute.ANSWER_FACT,
        "provider_preference",
    ),
    (
        "You already have an appointment Tuesday at 2:15 PM.",
        SemanticRoute.WAIT,
        None,
    ),
    (
        "Here are some Friday afternoon appointments.",
        SemanticRoute.HOLD,
        None,
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
        backend=backend
    )

    history = (
        "PGAI: You already have an appointment Tuesday at 2:15 PM.",
        "PATIENT: I'd like to reschedule it for Friday afternoon.",
    )

    # Deliberately warm with an unrelated category so the benchmark
    # does not measure cold model loading.
    warm = await semantic.parse(
        remote_turn="Okay, I understand.",
        recent_dialogue=history,
    )

    print(
        "warmup_ms=",
        round(warm.latency_ms, 1),
    )

    passed = 0
    latencies = []

    for index, (
        text,
        expected_route,
        expected_focus,
    ) in enumerate(CASES, 1):

        trace = await semantic.parse(
            remote_turn=text,
            recent_dialogue=history,
        )

        routed = route_semantic_frame(
            trace.frame
        )

        ok = routed.route is expected_route

        if expected_focus is not None:
            ok = (
                ok
                and routed.fact_focus.value
                == expected_focus
            )

        passed += int(ok)
        latencies.append(trace.latency_ms)

        f = trace.frame

        print()
        print(f"[{index}] {text}")
        print(" speech_act=", f.speech_act.value)
        print(" operation=", f.operation.value)
        print(" focus=", f.focus.value)
        print(" commitment=", f.commitment.value)
        print(" certainty=", f.certainty.value)
        print(" route=", routed.route.value)

        if trace.validation_error:
            print(
                " validation_error=",
                trace.validation_error.splitlines()[0],
            )

        print(
            " latency_ms=",
            round(trace.latency_ms, 1),
        )
        print(" PASS=", ok)

    print()
    print("========== SUMMARY ==========")
    print(f"correct={passed}/{len(CASES)}")
    print(
        "median_ms=",
        round(statistics.median(latencies), 1),
    )
    print(
        "max_ms=",
        round(max(latencies), 1),
    )


if __name__ == "__main__":
    asyncio.run(main())
