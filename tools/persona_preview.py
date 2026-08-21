"""Preview VoiceProbe personas without making a phone call."""

from __future__ import annotations

import argparse
import json

from voiceprobe.v3.personas import (
    PersonaRuntime,
    get_persona,
    list_personas,
    sequence_ids_for,
)


CALL6_STYLE_REMOTE_TURNS = (
    "Would you like me to create a new patient profile?",
    "How can I help you today?",
    "Is this for a new patient consultation?",
    "Do you have a provider preference?",
    "I don't have Friday afternoon. Would another weekday afternoon work?",
    (
        "Friday afternoon I have two fifteen PM, three PM, "
        "and three forty five PM. Which time works best for you?"
    ),
    "Okay. I have the option you mentioned selected.",
    "Would you like me to book that appointment?",
    "I have two fifteen PM selected. Is that correct?",
    "Anything else I can help you with?",
)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--persona",
        choices=tuple(
            persona.persona_id
            for persona in list_personas()
        ),
        required=True,
    )

    parser.add_argument("--sequence", default="")
    parser.add_argument("--seed", type=int, default=6)

    args = parser.parse_args()

    if args.sequence:
        available = sequence_ids_for(args.persona)

        if args.sequence not in available:
            parser.error(
                f"sequence {args.sequence!r} is not valid for "
                f"{args.persona!r}; choices: {available!r}"
            )

    runtime = PersonaRuntime(
        get_persona(args.persona),
        seed=args.seed,
        sequence_id=(args.sequence or None),
    )

    print(f"PERSONA: {runtime.definition.persona_id}")
    print(f"WORKFLOW: {runtime.definition.workflow}")
    print(f"REQUESTED SEQUENCE: {runtime.requested_sequence_id}")
    print(f"HYPOTHESIS: {runtime.definition.hypothesis}")
    print()

    for number, remote_turn in enumerate(
        CALL6_STYLE_REMOTE_TURNS,
        start=1,
    ):
        decision = runtime.consider(remote_turn)

        print(f"[{number}] PGAI: {remote_turn}")

        if decision.override_text is None:
            print(f"    PERSONA: no override ({decision.reason})")
        else:
            print(f"    PATIENT TEST MOVE: {decision.override_text}")
            print(f"    STATE EFFECT: {decision.state_effect.value}")

        print()

    print("EVIDENCE")
    print(json.dumps(runtime.evidence(), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
