"""High-fidelity staged call driver for the v3.3 planner."""

from __future__ import annotations

import time
from dataclasses import dataclass

from voiceprobe.v33.planner import PlannerDecision, V33Planner

from .assertions import StageViolations, check_decision
from .mock_pgai import MockPGAI, MockState
from .simulated_clock import RealtimeBudget, SimulatedClock


@dataclass(frozen=True, slots=True)
class StageTurn:
    index: int
    pgai_text: str
    patient_text: str
    selected_kinds: tuple[str, ...]
    measured_planner_seconds: float
    modeled_tts_start_seconds: float
    estimated_audio_start_seconds: float
    repeat_risk: bool
    simulated_time_seconds: float


@dataclass(frozen=True, slots=True)
class StageResult:
    turns: tuple[StageTurn, ...]
    violations: tuple[str, ...]
    final_state: str
    completed: bool


class StageRunner:
    """Run the real async planner inside a deterministic mock-call clock."""

    def __init__(
        self,
        *,
        planner: V33Planner,
        environment: MockPGAI,
        budget: RealtimeBudget | None = None,
        tts_start_seconds: float = 1.425,
    ) -> None:
        if tts_start_seconds < 0:
            raise ValueError("tts_start_seconds must be non-negative")

        self.planner = planner
        self.environment = environment
        self.budget = budget or RealtimeBudget()
        self.tts_start_seconds = tts_start_seconds

    async def run(self, *, max_turns: int = 20) -> StageResult:
        clock = SimulatedClock()
        violations = StageViolations()
        recorded: list[StageTurn] = []
        remote = self.environment.opening()

        for index in range(1, max_turns + 1):
            if clock.now >= self.budget.max_call_duration_seconds:
                violations.add("simulated call exceeded max duration")
                break

            started = time.perf_counter()
            decision: PlannerDecision = await self.planner.decide(remote)
            planner_seconds = time.perf_counter() - started

            estimated_audio_start = planner_seconds + self.tts_start_seconds
            repeat_risk = (
                estimated_audio_start
                >= self.budget.remote_repeat_threshold_seconds
            )

            check_decision(
                decision,
                estimated_audio_start_seconds=estimated_audio_start,
                budget=self.budget,
                violations=violations,
            )

            if repeat_risk:
                violations.add(
                    "remote-repeat risk: first audio estimated at "
                    f"{estimated_audio_start:.3f}s >= "
                    f"{self.budget.remote_repeat_threshold_seconds:.3f}s "
                    "repeat threshold"
                )

            clock.advance(estimated_audio_start)

            recorded.append(
                StageTurn(
                    index=index,
                    pgai_text=remote,
                    patient_text=decision.spoken_text,
                    selected_kinds=tuple(
                        x.value for x in decision.selected.plan.kinds
                    ),
                    measured_planner_seconds=planner_seconds,
                    modeled_tts_start_seconds=self.tts_start_seconds,
                    estimated_audio_start_seconds=estimated_audio_start,
                    repeat_risk=repeat_risk,
                    simulated_time_seconds=clock.now,
                )
            )

            remote = self.environment.respond(decision.selected.plan)
            clock.advance(0.6)

            if self.environment.state is MockState.DONE:
                break

        return StageResult(
            turns=tuple(recorded),
            violations=tuple(violations.items),
            final_state=self.environment.state.value,
            completed=(
                self.environment.state is MockState.DONE
                and not violations.items
            ),
        )
