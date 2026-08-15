"""Interactive terminal harness for the VoiceProbe patient agent.

This is the last text-only integration step before synthesized speech.
It exercises the same PatientAgent and planner stack that will later
receive turns from Moonshine during a live phone call.
"""

from __future__ import annotations

import time

from voiceprobe.agents.patient import PatientAgent
from voiceprobe.conversation.state import build_initial_state
from voiceprobe.planners.hybrid import HybridPatientPlanner
from voiceprobe.planners.ollama import OllamaActionSelector
from voiceprobe.scenarios.models import PatientFacts, PatientScenario


def build_demo_scenario() -> PatientScenario:
    """Create a synthetic scheduling scenario for local testing."""
    return PatientScenario(
        scenario_id="shoulder-friday",
        objective="Schedule an appointment for Friday afternoon.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            date_of_birth="January 12, 1998",
            insurance="Blue Cross",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
        test_targets=(
            "appointment scheduling",
            "fact consistency",
            "misunderstanding correction",
        ),
    )


def main() -> None:
    """Run an interactive simulated-patient conversation."""
    scenario = build_demo_scenario()
    selector = OllamaActionSelector()
    planner = HybridPatientPlanner(selector=selector)
    agent = PatientAgent(
        scenario=scenario,
        planner=planner,
    )
    state = build_initial_state(scenario)

    print()
    print("VoiceProbe Patient Agent")
    print(f"Scenario: {scenario.scenario_id}")
    print(f"Objective: {scenario.objective}")
    print()
    print("Type receptionist/agent messages below.")
    print("Type /quit to exit.")
    print()

    try:
        while True:
            try:
                agent_turn = input("Receptionist > ").strip()
            except EOFError:
                print()
                break

            if agent_turn == "/quit":
                break

            if not agent_turn:
                continue

            started_at = time.perf_counter()

            step = agent.respond(
                state,
                agent_turn,
            )

            elapsed_seconds = time.perf_counter() - started_at
            state = step.state

            print()
            print(f"Patient > {step.action.response}")
            print(
                f"[action={step.action.kind.value} "
                f"facts={list(step.action.facts_used)} "
                f"latency={elapsed_seconds:.3f}s]"
            )
            print()

            if state.objective_complete:
                print("Scenario objective marked complete.")
                break

    finally:
        selector.close()


if __name__ == "__main__":
    main()
