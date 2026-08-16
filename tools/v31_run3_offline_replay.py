#!/usr/bin/env python3
"""Replay live run #3 through v3.1 without placing a phone call."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.runtime import VoiceProbeV3Runtime
from voiceprobe.v3.semantic_router import V31SemanticRouter

RUN3 = Path(
    "artifacts/runs/"
    "20260816T104724.902931Z-autonomous-phone-diagnostic/"
    "transcript.txt"
)

AGENT_RE = re.compile(r"^\[[^\]]+\]\s+AGENT:\s+(.*)$")


def load_agent_turns() -> list[str]:
    if not RUN3.is_file():
        raise SystemExit(f"Missing run-3 transcript: {RUN3}")

    turns = []
    for line in RUN3.read_text(encoding="utf-8").splitlines():
        match = AGENT_RE.match(line)
        if match:
            turns.append(match.group(1))
    return turns


async def main() -> None:
    # Match the production fallback semantic configuration.
    router = V31SemanticRouter(use_embeddings=True)
    runtime = VoiceProbeV3Runtime(fallback_resolver=router.resolve)

    previous_clarification = None
    saw_visit_reason_answer = False
    saw_compound_visit_answer = False
    saw_open_ended_objective = False
    saw_dob_correction = False

    for index, turn in enumerate(load_agent_turns(), start=1):
        result = await runtime.process_turns(
            [turn],
            ingress_reason="run3_v31_offline_replay",
        )

        print()
        print(f"[{index:02d}] REMOTE: {turn}")
        print(
            "     ",
            f"route={result.route.value}",
            f"kind={result.decision.kind.value}",
            f"reason={result.decision.reason}",
        )
        print("     ", f"speech={result.decision.text!r}")

        if result.decision.kind == DecisionKind.CLARIFY:
            assert result.decision.text != previous_clarification, (
                "Consecutive identical clarifications are forbidden."
            )
            previous_clarification = result.decision.text
        else:
            previous_clarification = None

        if turn.casefold().strip() == "may i help you?":
            assert result.decision.kind == DecisionKind.STATE_OBJECTIVE, (
                "Open-ended help prompt must state the scheduling objective."
            )
            assert result.decision.text == (
                "I need to schedule an appointment for Friday afternoon."
            )
            saw_open_ended_objective = True

        if (
            result.decision.kind == DecisionKind.ANSWER_COMPLAINT
            and result.decision.text == "I have right shoulder pain."
        ):
            saw_visit_reason_answer = True

        if (
            "routine visit" in turn.casefold()
            and "specific concern" in turn.casefold()
        ):
            assert result.decision.kind == DecisionKind.ANSWER_VISIT_DETAILS, (
                "Compound reason/type prompt must answer both requested facts."
            )
            assert result.decision.text == (
                "I have right shoulder pain. "
                "This is for a new patient consultation."
            )
            saw_compound_visit_answer = True

        if result.decision.kind in {
            DecisionKind.CORRECT_FACT,
            DecisionKind.CORRECT_AND_STATE_OBJECTIVE,
        } and "April 12, 1998" in result.decision.text:
            saw_dob_correction = True

    assert saw_open_ended_objective, (
        "Run-3 replay never recovered the open-ended scheduling objective."
    )
    assert saw_visit_reason_answer, "Run-3 replay never answered the visit reason."
    assert saw_compound_visit_answer, (
        "Run-3 replay never answered the compound reason/type prompt."
    )
    assert saw_dob_correction, "Run-3 replay never corrected the wrong DOB."

    print()
    print("=" * 72)
    print("RUN-3 V3.1 OFFLINE REPLAY: PASS")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
