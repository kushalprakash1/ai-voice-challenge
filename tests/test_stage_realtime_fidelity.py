from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from voiceprobe.stage.mock_pgai import MockPGAI
from voiceprobe.stage.runner import StageRunner
from voiceprobe.stage.simulated_clock import RealtimeBudget
from voiceprobe.stage.trace_loader import load_historical_run, timing_profile
from voiceprobe.v33.actions import ActionKind, ActionMove, ActionPlan
from voiceprobe.v33.mind import AgentMind
from voiceprobe.v33.mission import adaptive_reschedule_mission
from voiceprobe.v33.planner import V33Planner
from voiceprobe.v33.world_model import ObservationKind, RemoteObservation


class SlowScriptedReasoner:
    async def propose(self, *, mind, remote_turn):
        await asyncio.sleep(0.025)
        observation = RemoteObservation(
            ObservationKind.OPEN_INTENT,
            remote_turn,
            True,
        )
        candidates = (
            ActionPlan(
                moves=(ActionMove(ActionKind.STATE_GOAL),),
                rationale="state current goal",
                utterance="I need an appointment.",
            ),
            ActionPlan(
                moves=(ActionMove(ActionKind.ASK_QUESTION),),
                rationale="steer",
                utterance="What appointments are available?",
            ),
        )
        return observation, candidates


def test_historical_profile_includes_policy_plus_audio_start(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()

    rows = [
        {
            "elapsed_seconds": 10.0,
            "event": "v3_runtime_decision",
            "details": {
                "decision_index": 1,
                "actionable_turn": "Question?",
                "route": "fallback",
                "decision_kind": "contextual_answer",
                "decision_reason": "test",
                "policy_latency_ms": 1500.0,
                "response_ready": True,
                "decision_text": "Answer.",
            },
        },
        {
            "elapsed_seconds": 12.4,
            "event": "v3_playback_started",
            "details": {"text": "Answer."},
        },
    ]

    (run / "events.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    loaded = load_historical_run(run)
    decision = loaded.decisions[0]

    assert decision.decision_to_audio_start_seconds == pytest.approx(2.4)
    assert decision.runtime_to_audio_start_seconds == pytest.approx(3.9)

    profile = timing_profile((loaded,))
    assert profile.runtime_to_audio_start_p95_seconds == pytest.approx(3.9)


@pytest.mark.asyncio
async def test_stage_runner_measures_real_planner_wall_time() -> None:
    planner = V33Planner(
        mind=AgentMind(adaptive_reschedule_mission()),
        reasoner=SlowScriptedReasoner(),
    )
    runner = StageRunner(
        planner=planner,
        environment=MockPGAI(seed=1),
        budget=RealtimeBudget(
            first_audio_deadline_seconds=10.0,
            remote_repeat_threshold_seconds=10.0,
        ),
        tts_start_seconds=0.4,
    )

    result = await runner.run(max_turns=1)
    turn = result.turns[0]

    assert turn.measured_planner_seconds >= 0.02
    assert turn.estimated_audio_start_seconds >= 0.42
    assert turn.modeled_tts_start_seconds == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_repeat_risk_uses_measured_planner_plus_tts() -> None:
    planner = V33Planner(
        mind=AgentMind(adaptive_reschedule_mission()),
        reasoner=SlowScriptedReasoner(),
    )
    runner = StageRunner(
        planner=planner,
        environment=MockPGAI(seed=1),
        budget=RealtimeBudget(
            first_audio_deadline_seconds=10.0,
            remote_repeat_threshold_seconds=0.03,
        ),
        tts_start_seconds=0.02,
    )

    result = await runner.run(max_turns=1)

    assert result.turns[0].repeat_risk is True
    assert any(
        "remote-repeat risk" in violation
        for violation in result.violations
    )
