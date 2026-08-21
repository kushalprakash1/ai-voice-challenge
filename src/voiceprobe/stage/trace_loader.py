"""Load historical VoiceProbe call traces into reusable StageLab fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median


@dataclass(frozen=True, slots=True)
class HistoricalDecision:
    elapsed_seconds: float
    decision_index: int
    remote_text: str
    route: str
    decision_kind: str
    decision_reason: str
    policy_latency_ms: float
    response_ready: bool
    decision_to_audio_start_seconds: float | None
    runtime_to_audio_start_seconds: float | None


@dataclass(frozen=True, slots=True)
class HistoricalRun:
    run_id: str
    decisions: tuple[HistoricalDecision, ...]
    transcript_text: str


@dataclass(frozen=True, slots=True)
class TimingProfile:
    policy_p50_ms: float
    policy_p95_ms: float
    audio_start_p50_seconds: float
    audio_start_p95_seconds: float
    runtime_to_audio_start_p50_seconds: float
    runtime_to_audio_start_p95_seconds: float
    sample_count: int


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def load_historical_run(run_dir: str | Path) -> HistoricalRun:
    run_path = Path(run_dir)
    events_path = run_path / "events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(events_path)

    events: list[dict] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))

    decisions: list[HistoricalDecision] = []
    for i, row in enumerate(events):
        if row.get("event") != "v3_runtime_decision":
            continue

        details = row.get("details", {})
        elapsed = float(row.get("elapsed_seconds", 0.0))
        policy_ms = float(details.get("policy_latency_ms") or 0.0)
        expected_text = str(details.get("decision_text") or "")
        audio_start: float | None = None

        if details.get("response_ready"):
            for later in events[i + 1 :]:
                if later.get("event") == "v3_runtime_decision":
                    break
                if later.get("event") != "v3_playback_started":
                    continue

                later_text = str(later.get("details", {}).get("text") or "")
                if expected_text and later_text and later_text != expected_text:
                    continue

                audio_start = (
                    float(later.get("elapsed_seconds", 0.0))
                    - elapsed
                )
                break

        runtime_to_audio = (
            policy_ms / 1000.0 + audio_start
            if audio_start is not None
            else None
        )

        decisions.append(
            HistoricalDecision(
                elapsed_seconds=elapsed,
                decision_index=int(details.get("decision_index") or 0),
                remote_text=str(details.get("actionable_turn") or ""),
                route=str(details.get("route") or ""),
                decision_kind=str(details.get("decision_kind") or ""),
                decision_reason=str(details.get("decision_reason") or ""),
                policy_latency_ms=policy_ms,
                response_ready=bool(details.get("response_ready")),
                decision_to_audio_start_seconds=audio_start,
                runtime_to_audio_start_seconds=runtime_to_audio,
            )
        )

    transcript_path = run_path / "transcript.txt"
    transcript = (
        transcript_path.read_text(encoding="utf-8")
        if transcript_path.exists()
        else ""
    )

    return HistoricalRun(
        run_id=run_path.name,
        decisions=tuple(decisions),
        transcript_text=transcript,
    )


def timing_profile(runs: tuple[HistoricalRun, ...]) -> TimingProfile:
    policy = [d.policy_latency_ms for r in runs for d in r.decisions]
    starts = [
        d.decision_to_audio_start_seconds
        for r in runs
        for d in r.decisions
        if d.decision_to_audio_start_seconds is not None
    ]
    runtime_starts = [
        d.runtime_to_audio_start_seconds
        for r in runs
        for d in r.decisions
        if d.runtime_to_audio_start_seconds is not None
    ]

    return TimingProfile(
        policy_p50_ms=median(policy) if policy else 0.0,
        policy_p95_ms=_percentile(policy, 0.95),
        audio_start_p50_seconds=median(starts) if starts else 0.0,
        audio_start_p95_seconds=_percentile(starts, 0.95),
        runtime_to_audio_start_p50_seconds=(
            median(runtime_starts) if runtime_starts else 0.0
        ),
        runtime_to_audio_start_p95_seconds=_percentile(runtime_starts, 0.95),
        sample_count=len(starts),
    )
