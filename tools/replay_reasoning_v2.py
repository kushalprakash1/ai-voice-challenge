"""Replay historical VoiceProbe transcripts through Reasoning Core v2.

This tool never places a phone call and never changes production state.

It compares the historical patient response with the response the new
structured reasoning stack would produce for each recorded remote-agent turn.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from voiceprobe.reasoning.action_plan import (
    ActionPlan,
)
from voiceprobe.reasoning.action_verbalizer import (
    GenericActionVerbalizer,
)
from voiceprobe.reasoning.fact_grounding import (
    ground_fact_assertions,
)
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


_LINE_RE = re.compile(
    r"^\[(?P<seconds>[0-9.]+)s\] "
    r"(?P<speaker>AGENT|PATIENT): "
    r"(?P<text>.*)$"
)


def parse_transcript(
    path: Path,
) -> list[tuple[str, str]]:
    events: list[
        tuple[str, str]
    ] = []

    for raw in path.read_text(
        encoding="utf-8"
    ).splitlines():
        match = _LINE_RE.match(
            raw
        )

        if match is None:
            continue

        events.append(
            (
                match.group(
                    "speaker"
                ),
                match.group(
                    "text"
                ),
            )
        )

    return events


def agent_turns_with_old_response(
    events: list[tuple[str, str]],
) -> list[tuple[str, str | None]]:
    result: list[
        tuple[str, str | None]
    ] = []

    for index, (
        speaker,
        text,
    ) in enumerate(events):

        if speaker != "AGENT":
            continue

        old_patient: str | None = None

        if (
            index + 1
            < len(events)
            and events[
                index + 1
            ][0]
            == "PATIENT"
        ):
            old_patient = events[
                index + 1
            ][1]

        result.append(
            (
                text,
                old_patient,
            )
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help=(
            "Historical artifacts/runs/... directory."
        ),
    )

    parser.add_argument(
        "--scenario",
        default=(
            "autonomous-phone-diagnostic"
        ),
    )

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

    transcript = (
        args.run
        / "transcript.txt"
    )

    if not transcript.exists():
        raise SystemExit(
            f"Missing transcript: {transcript}"
        )

    scenario = get_scenario(
        args.scenario
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

    verbalizer = (
        GenericActionVerbalizer()
    )

    recent_agent_history: list[
        str
    ] = []

    recent_actions: list[
        ActionPlan
    ] = []

    rows = (
        agent_turns_with_old_response(
            parse_transcript(
                transcript
            )
        )
    )

    failures = 0

    try:
        for number, (
            agent_turn,
            old_patient,
        ) in enumerate(
            rows,
            start=1,
        ):
            print()
            print("=" * 80)
            print(
                f"TURN {number}"
            )
            print("=" * 80)

            print(
                "AGENT:"
            )
            print(
                agent_turn
            )

            print()
            print(
                "OLD PATIENT:"
            )
            print(
                old_patient
                if old_patient
                is not None
                else "<no response>"
            )

            try:
                frame = (
                    semantic.interpret(
                        agent_turn=agent_turn,
                        recent_history=(
                            recent_agent_history
                        ),
                    )
                )

                grounding = (
                    ground_fact_assertions(
                        world=world,
                        turn=frame,
                    )
                )

                plan, repaired_from = (
                    planner.plan(
                        world=world,
                        turn=frame,
                        recent_actions=(
                            recent_actions
                        ),
                    )
                )

                patient_text = (
                    verbalizer.verbalize(
                        world=world,
                        turn=frame,
                        plan=plan,
                        corrections=(
                            grounding.conflicts
                        ),
                    )
                )

                print()
                print(
                    "V2 SEMANTICS:"
                )
                print(
                    "  requested_action =",
                    frame.requested_action.value,
                )
                print(
                    "  options          =",
                    len(
                        frame.appointment_options
                    ),
                )
                print(
                    "  requested_facts  =",
                    [
                        item.value
                        for item
                        in frame.requested_facts
                    ],
                )

                print(
                    "  stated_facts     =",
                    [
                        {
                            "fact": item.fact.value,
                            "value": item.value,
                        }
                        for item
                        in frame.stated_facts
                    ],
                )

                print(
                    "  proposed_workflow=",
                    (
                        frame.proposed_workflow.model_dump(
                            mode="json",
                        )
                        if frame.proposed_workflow is not None
                        else None
                    ),
                )

                print(
                    "  booking_confirmed=",
                    frame.booking_confirmed,
                )

                print(
                    "  confirmed_slot   =",
                    (
                        frame.confirmed_appointment.model_dump(
                            mode="json",
                        )
                        if frame.confirmed_appointment is not None
                        else None
                    ),
                )

                print(
                    "  fact_conflicts   =",
                    [
                        {
                            "fact": item.fact.value,
                            "asserted": item.asserted_value,
                            "authoritative": item.authoritative_value,
                        }
                        for item
                        in grounding.conflicts
                    ],
                )

                print()
                print(
                    "V2 PLAN:"
                )
                print(
                    f"  action = "
                    f"{plan.action.value}"
                )

                print(
                    "  selected_option_index =",
                    plan.selected_option_index,
                )

                if repaired_from:
                    print(
                        "  repaired after validator rejection"
                    )

                print()
                print(
                    "V2 PATIENT:"
                )

                print(
                    patient_text
                    if patient_text
                    else "<silence>"
                )

                recent_actions.append(
                    plan
                )

            except Exception as error:
                failures += 1

                print()
                print(
                    "V2 ERROR:"
                )
                print(
                    f"{type(error).__name__}: "
                    f"{error}"
                )

            finally:
                recent_agent_history.append(
                    agent_turn
                )

    finally:
        semantic.close()
        planner.close()

    print()
    print("=" * 80)
    print(
        "REPLAY COMPLETE"
    )
    print(
        f"agent turns: {len(rows)}"
    )
    print(
        f"v2 errors:   {failures}"
    )
    print("=" * 80)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
