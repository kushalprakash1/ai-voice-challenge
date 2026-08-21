"""End-to-end shadow test: semantics -> world -> planner -> validator."""

from __future__ import annotations

import argparse
import json
import time

from voiceprobe.reasoning.planner import (
    QwenPatientPlanner,
)
from voiceprobe.reasoning.semantic_reasoner import (
    StructuredTurnReasoner,
)
from voiceprobe.reasoning.world_model import (
    build_world_model,
)
from voiceprobe.scenarios.catalog import (
    get_scenario,
)


CASES = (
    {
        "label": "REMOTE AGENT STILL WORKING",
        "history": (),
        "text": (
            "Let me check for available new patient consultation "
            "appointments for Friday afternoon. One moment."
        ),
        "expected": "wait",
    },
    {
        "label": "ALL OFFERED TIMES CONFLICT",
        "history": (),
        "text": (
            "On Friday, August 21st, there are three available "
            "times with Becker: 9 a.m., 9.45 a.m., and "
            "10.30 a.m. Would you like to book one of these "
            "or look at another time?"
        ),
        "expected": "request_alternative",
    },
    {
        "label": "MATCHING AFTERNOON OPTION EXISTS",
        "history": (),
        "text": (
            "On Friday I have 9 a.m., 2.30 p.m., "
            "and 4 p.m. available. Which would you like?"
        ),
        "expected": "select_option",
    },
    {
        "label": "SEARCH PERMISSION",
        "history": (),
        "text": (
            "Would you like me to check "
            "Friday afternoon appointments?"
        ),
        "expected": "grant_permission",
    },
    {
        "label": "INSURANCE FACT REQUEST",
        "history": (),
        "text": (
            "What insurance do you have?"
        ),
        "expected": "answer_fact",
    },
)


def dump(
    value: object,
) -> str:
    if hasattr(
        value,
        "model_dump",
    ):
        value = value.model_dump(
            mode="json",
        )

    return json.dumps(
        value,
        indent=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default="http://127.0.0.1:11434/api/chat",
    )

    parser.add_argument(
        "--model",
        default="qwen3:14b",
    )

    args = parser.parse_args()

    scenario = get_scenario(
        "autonomous-phone-diagnostic"
    )

    world = build_world_model(
        scenario
    )

    semantic = StructuredTurnReasoner(
        model=args.model,
        url=args.url,
    )

    planner = QwenPatientPlanner(
        model=args.model,
        url=args.url,
    )

    failures: list[
        tuple[str, str, str]
    ] = []

    try:
        print("=" * 80)
        print("PATIENT WORLD")
        print("=" * 80)
        print(
            dump(
                world
            )
        )

        for case in CASES:

            print()
            print("=" * 80)
            print(case["label"])
            print("=" * 80)

            print("REMOTE AGENT:")
            print(case["text"])
            print()

            semantic_started = (
                time.perf_counter()
            )

            frame = semantic.interpret(
                agent_turn=case["text"],
                recent_history=case["history"],
            )

            semantic_seconds = (
                time.perf_counter()
                - semantic_started
            )

            planner_started = (
                time.perf_counter()
            )

            plan, repaired_from = (
                planner.plan(
                    world=world,
                    turn=frame,
                )
            )

            planner_seconds = (
                time.perf_counter()
                - planner_started
            )

            print("TURN FRAME:")
            print(
                dump(
                    frame
                )
            )

            print()
            print("ACTION PLAN:")
            print(
                dump(
                    plan
                )
            )

            print()
            print(
                "SEMANTIC TIME:",
                f"{semantic_seconds:.3f}s",
            )

            print(
                "PLANNER TIME:",
                f"{planner_seconds:.3f}s",
            )

            if repaired_from:
                print()
                print(
                    "FIRST PLAN WAS REJECTED BY VALIDATOR:"
                )

                for violation in repaired_from:
                    print(
                        f"  {violation.code}: "
                        f"{violation.detail}"
                    )

            actual = (
                plan.action.value
            )

            expected = case[
                "expected"
            ]

            print()
            print(
                "EXPECTED:",
                expected,
            )

            print(
                "ACTUAL:  ",
                actual,
            )

            if actual != expected:
                failures.append(
                    (
                        case["label"],
                        expected,
                        actual,
                    )
                )

                print("RESULT:   FAIL")
            else:
                print("RESULT:   PASS")

    finally:
        semantic.close()
        planner.close()

    print()
    print("=" * 80)

    if failures:
        print("SHADOW PLANNER FAILURES")

        for (
            label,
            expected,
            actual,
        ) in failures:
            print(
                f"{label}: expected "
                f"{expected}, got {actual}"
            )

        raise SystemExit(1)

    print("ALL SHADOW PLANNER CASES PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
