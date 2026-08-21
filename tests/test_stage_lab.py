from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from voiceprobe.stage.assertions import StageViolations, check_decision
from voiceprobe.stage.manual_corpus import load_manual_call
from voiceprobe.stage.simulated_clock import RealtimeBudget
from voiceprobe.stage.trace_loader import load_historical_run, timing_profile
from voiceprobe.v33.actions import ActionKind, ActionMove, ActionPlan
from voiceprobe.v33.mind import AgentMind
from voiceprobe.v33.mission import adaptive_reschedule_mission
from voiceprobe.v33.planner import V33Planner
from voiceprobe.v33.world_model import ObservationKind, RemoteObservation


@dataclass
class ScriptedReasoner:
    observation: RemoteObservation
    candidates: tuple[ActionPlan, ...]

    async def propose(self, *, mind, remote_turn):
        return self.observation, self.candidates


def test_manual_corpus_redacts_human_identity() -> None:
    root = Path(__file__).parents[1]
    fixture = root / "artifacts" / "stage-corpus" / "manual" / "manual-001.json"
    call = load_manual_call(fixture)
    joined = "\n".join(turn.text for turn in call.turns)

    assert call.call_id == "manual-001"
    assert "<CALLER_NAME>" in joined
    assert "<DOB>" in joined
    assert "Kushal" not in joined
    assert "April 5th, 2001" not in joined


def test_trace_loader_measures_decision_to_audio_start(tmp_path: Path) -> None:
    run = tmp_path / "run-1"
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
                "policy_latency_ms": 1200.0,
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
    (run / "events.jsonl").write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    (run / "transcript.txt").write_text("AGENT: Question?\nPATIENT: Answer.\n", encoding="utf-8")

    loaded = load_historical_run(run)
    assert loaded.decisions[0].decision_to_audio_start_seconds == pytest.approx(2.4)
    profile = timing_profile((loaded,))
    assert profile.audio_start_p95_seconds == pytest.approx(2.4)


@pytest.mark.asyncio
async def test_stage_quality_gate_flags_slow_audio_start() -> None:
    observation = RemoteObservation(
        ObservationKind.OPEN_INTENT,
        "How may I help?",
        True,
    )
    candidate = ActionPlan(
        moves=(ActionMove(ActionKind.STATE_GOAL),),
        rationale="state goal",
        utterance="I need an appointment.",
    )
    planner = V33Planner(
        mind=AgentMind(adaptive_reschedule_mission()),
        reasoner=ScriptedReasoner(observation, (candidate, ActionPlan(
            moves=(ActionMove(ActionKind.ASK_QUESTION),),
            rationale="ask",
            utterance="What is available?",
        ))),
    )
    decision = await planner.decide(observation.raw_text)
    violations = StageViolations()
    check_decision(
        decision,
        estimated_audio_start_seconds=2.4,
        budget=RealtimeBudget(first_audio_deadline_seconds=2.0),
        violations=violations,
    )
    assert any("first audio estimated" in item for item in violations.items)


def test_turn_arbiter_merges_split_question_without_longer_grace() -> None:
    from voiceprobe.stage.faults import split_tail_question
    from voiceprobe.v33.turn_arbiter import ArbitrationKind, V33TurnArbiter

    text = "There are no preferred openings. Would you like another day or should I check mornings instead?"
    parts = split_tail_question(text)
    assert len(parts) == 2

    arbiter = V33TurnArbiter(continuation_grace_ms=900.0)
    first = arbiter.ingest(parts[0].text, at_ms=0.0)
    second = arbiter.ingest(parts[1].text, at_ms=parts[1].delay_ms)

    assert first.kind is ArbitrationKind.HOLD
    assert second.kind is ArbitrationKind.EMIT
    assert "should I check mornings" in second.text


def test_turn_arbiter_suppresses_repeat_while_response_pending() -> None:
    from voiceprobe.v33.turn_arbiter import ArbitrationKind, V33TurnArbiter

    arbiter = V33TurnArbiter()
    question = "Would you like another day or should I check Friday mornings?"
    first = arbiter.ingest(question, at_ms=0.0)
    repeated = arbiter.ingest(question, at_ms=3200.0, response_pending=True)

    assert first.kind is ArbitrationKind.EMIT
    assert repeated.kind is ArbitrationKind.SUPPRESS_DUPLICATE


def test_historical_regression_corpus_uses_invariants_not_expected_sentences() -> None:
    root = Path(__file__).parents[1]
    path = root / "artifacts" / "stage-corpus" / "historical" / "last-call-regressions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(payload["cases"]) >= 5
    for case in payload["cases"]:
        assert case["remote_turn"]
        assert case["requires_response"] is True
        assert "expected_response" not in case
        assert set(case["forbidden_actions"]) >= {"wait", "hold"}
