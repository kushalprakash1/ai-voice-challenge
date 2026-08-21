"""Interactive terminal harness for the full VoiceProbe patient stack.

This module is a development tool. It lets a human type receptionist
utterances while VoiceProbe runs the same semantic interpretation,
grounding, patient reasoning, verbalization, and state transitions that
will later be driven by live ASR.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from time import perf_counter

import httpx

from voiceprobe.agents.brain import PatientBrain
from voiceprobe.conversation.session import (
    PatientSession,
    SessionTurnResult,
)
from voiceprobe.interpreters.ollama import (
    OllamaConversationInterpreter,
)
from voiceprobe.scenarios.models import (
    PatientFacts,
    PatientScenario,
)
from voiceprobe.verbalizers.deterministic import (
    DeterministicNaturalVerbalizer,
)

DEFAULT_MODEL = "qwen3:14b"
DEFAULT_URL = "http://127.0.0.1:11434/api/chat"


def build_terminal_scenario(
    *,
    name: str,
    complaint: str,
    duration: str,
    date_of_birth: str | None,
    insurance: str | None,
    preferred_day: str | None,
    preferred_time: str | None,
) -> PatientScenario:
    """Build the patient ground truth used by one terminal session."""
    return PatientScenario(
        scenario_id="terminal-patient",
        objective="Schedule an acceptable medical appointment.",
        facts=PatientFacts(
            name=name,
            complaint=complaint,
            duration=duration,
            date_of_birth=date_of_birth,
            insurance=insurance,
            preferred_day=preferred_day,
            preferred_time=preferred_time,
        ),
    )


def _positive_float(value: str) -> float:
    parsed = float(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")

    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete VoiceProbe simulated patient stack "
            "against manually typed receptionist turns."
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Ollama /api/chat endpoint. Default: {DEFAULT_URL}",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=20.0,
        help="Ollama HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show extracted meaning, grounding, and state after each turn.",
    )

    parser.add_argument(
        "--name",
        default="Alex Morgan",
    )
    parser.add_argument(
        "--complaint",
        default="right shoulder pain",
    )
    parser.add_argument(
        "--duration",
        default="five days",
    )
    parser.add_argument(
        "--date-of-birth",
        default="April 12, 1998",
    )
    parser.add_argument(
        "--insurance",
        default="Blue Cross",
    )
    parser.add_argument(
        "--preferred-day",
        default="Friday",
    )
    parser.add_argument(
        "--preferred-time",
        default="afternoon",
    )

    return parser


def _print_state(session: PatientSession) -> None:
    state = session.state
    progress = session.progress

    print()
    print("STATE")
    print(f"  messages:           {len(state.messages)}")
    print(f"  answered_facts:     {sorted(state.answered_facts)}")
    print(f"  corrections:        {len(state.corrections)}")
    print(f"  state_complete:     {state.objective_complete}")
    print(f"  preferred_day:      {progress.preferred_day_shared}")
    print(f"  preferred_time:     {progress.preferred_time_shared}")
    print(f"  offered_slot:       {progress.offered_day!r}, {progress.offered_time!r}")
    print(f"  offer_accepted:     {progress.offer_accepted}")
    print(f"  booking_confirmed:  {progress.booking_confirmed}")
    print(f"  objective_complete: {progress.objective_complete}")
    print()


def _print_debug(
    result: SessionTurnResult,
    *,
    elapsed_seconds: float,
) -> None:
    print()
    print(f"[TURN LATENCY: {elapsed_seconds:.3f}s]")
    print(
        "[DECISION: "
        f"{result.decision.kind.value}; "
        f"facts={list(result.decision.facts_to_communicate)}]"
    )

    if result.grounded.conflicts:
        print("[CONFLICTS]")
        for conflict in result.grounded.conflicts:
            print(
                "  "
                f"{conflict.fact}: "
                f"heard={conflict.heard_value!r} "
                f"truth={conflict.authoritative_value!r}"
            )

    print("[MEANING]")
    print(
        json.dumps(
            result.meaning.model_dump(mode="json"),
            indent=2,
        )
    )

    print(
        "[PROGRESS] "
        f"offer={result.progress.offered_day!r}/"
        f"{result.progress.offered_time!r}, "
        f"accepted={result.progress.offer_accepted}, "
        f"confirmed={result.progress.booking_confirmed}, "
        f"complete={result.progress.objective_complete}"
    )
    print()


def main(argv: Sequence[str] | None = None) -> int:
    """Run an interactive VoiceProbe patient conversation."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    scenario = build_terminal_scenario(
        name=args.name,
        complaint=args.complaint,
        duration=args.duration,
        date_of_birth=args.date_of_birth,
        insurance=args.insurance,
        preferred_day=args.preferred_day,
        preferred_time=args.preferred_time,
    )

    with httpx.Client(timeout=args.timeout) as client:
        interpreter = OllamaConversationInterpreter(
            model=args.model,
            url=args.url,
            client=client,
        )

        verbalizer = DeterministicNaturalVerbalizer(
            model=args.model,
            url=args.url,
            client=client,
        )

        session = PatientSession(
            scenario=scenario,
            interpreter=interpreter,
            verbalizer=verbalizer,
            brain=PatientBrain(),
        )

        print()
        print("=" * 72)
        print("VOICEPROBE INTERACTIVE PATIENT")
        print("=" * 72)
        print(f"Model:    {args.model}")
        print(f"Ollama:   {args.url}")
        print()
        print("Commands:")
        print("  /state   show conversation/objective state")
        print("  /help    show commands")
        print("  /quit    exit")
        print()
        print("Type what the medical scheduling agent/receptionist says.")
        print()

        while True:
            try:
                raw_turn = input("RECEPTIONIST> ")
            except (EOFError, KeyboardInterrupt):
                print()
                print("Ending VoiceProbe session.")
                break

            agent_turn = raw_turn.strip()

            if not agent_turn:
                continue

            command = agent_turn.casefold()

            if command in {"/quit", "/exit"}:
                print("Ending VoiceProbe session.")
                break

            if command == "/state":
                _print_state(session)
                continue

            if command == "/help":
                print("/state   show current state")
                print("/help    show commands")
                print("/quit    exit")
                continue

            started = perf_counter()

            try:
                result = session.handle_agent_turn(agent_turn)
            except (
                httpx.HTTPError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                print()
                print(f"TURN ERROR: {type(error).__name__}: {error}")
                print("The session was not partially committed.")
                print()
                continue

            elapsed_seconds = perf_counter() - started

            print(f"PATIENT> {result.patient_text}")

            if args.debug:
                _print_debug(
                    result,
                    elapsed_seconds=elapsed_seconds,
                )

            if result.progress.objective_complete:
                print()
                print("*** APPOINTMENT OBJECTIVE COMPLETE ***")
                print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
