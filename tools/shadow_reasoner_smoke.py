"""Exercise Reasoning Core v2 without affecting production calls."""

from __future__ import annotations

import argparse
import json

from voiceprobe.reasoning.semantic_reasoner import (
    StructuredTurnReasoner,
)

CASES = (
    {
        "label": "STATUS / WAIT",
        "history": (),
        "text": (
            "Let me check for available new patient consultation "
            "appointments for Friday afternoon. One moment."
        ),
    },
    {
        "label": "MULTI-SLOT OFFER",
        "history": (),
        "text": (
            "Sure. On Friday, August 21st, there are three "
            "available times with Becker. 9 a.m., 9.45 a.m., "
            "and 10.30 a.m. Would you like to book one of "
            "these or would you prefer to look at other days?"
        ),
    },
    {
        # Verify that source grounding rejects unsupported patient facts.
        # Friday must NOT appear here because latest speech and history
        # contain no day.
        "label": "CONTEXTLESS MULTI-SLOT CHOICE",
        "history": (),
        "text": (
            "Which time would you like to book? "
            "9 a.m., 9.45 a.m., or 10.30 a.m."
        ),
    },
    {
        # Here Friday MAY be inherited because the latest turn clearly
        # refers back to a remote-agent offer containing Friday.
        "label": "CONTEXTUAL MULTI-SLOT CHOICE",
        "history": (
            (
                "On Friday, August 21st, I have 9 a.m., "
                "9.45 a.m., and 10.30 a.m. available."
            ),
        ),
        "text": (
            "Which of those times would you like?"
        ),
    },
    {
        "label": "SEARCH PERMISSION",
        "history": (),
        "text": (
            "Would you like me to check "
            "Friday afternoon appointments?"
        ),
    },
    {
        "label": "FACT REQUEST",
        "history": (),
        "text": (
            "What insurance do you have?"
        ),
    },
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default=(
            "http://127.0.0.1:11434/api/chat"
        ),
    )

    parser.add_argument(
        "--model",
        default="qwen3:14b",
    )

    args = parser.parse_args()

    reasoner = StructuredTurnReasoner(
        model=args.model,
        url=args.url,
    )

    try:
        for case in CASES:
            print()
            print("=" * 80)
            print(case["label"])
            print("=" * 80)

            history = case["history"]
            text = case["text"]

            if history:
                print("RECENT REMOTE HISTORY:")

                for item in history:
                    print(f"  {item}")

                print()

            print("LATEST AGENT TURN:")
            print(text)
            print()

            frame = reasoner.interpret(
                agent_turn=text,
                recent_history=history,
            )

            print(
                json.dumps(
                    frame.model_dump(
                        mode="json",
                    ),
                    indent=2,
                )
            )
    finally:
        reasoner.close()


if __name__ == "__main__":
    main()
