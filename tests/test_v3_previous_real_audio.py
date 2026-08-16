import asyncio
import json
from pathlib import Path

from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.runtime import VoiceProbeV3Runtime
from voiceprobe.v3.turn_stabilizer import (
    TimedRemoteTurn,
    stabilize_timed_turns,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "v3_calls"
    / "flux_previous_eot_085.jsonl"
)


def load_fixture():
    return [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_timed(rows):
    return [
        TimedRemoteTurn(
            text=row["text"],
            gap_to_next_start_ms=row["gap_to_next_start_ms"],
            turn_index=row["turn_index"],
        )
        for row in rows
    ]


def test_600ms_grace_merges_only_observed_short_continuations() -> None:
    bursts = stabilize_timed_turns(
        make_timed(load_fixture()),
        continuation_grace_ms=600.0,
    )

    indices = [burst.turn_indices for burst in bursts]

    assert (9, 10) in indices
    assert (11, 12) in indices
    assert (3, 4) not in indices
    assert len(bursts) == 13


def test_spoken_morning_times_are_declined() -> None:
    runtime = VoiceProbeV3Runtime()
    row = load_fixture()[11]

    result = asyncio.run(runtime.process_turns([row["text"]]))

    assert result.decision.kind == DecisionKind.DECLINE_INCOMPATIBLE_OFFER
    assert "Friday afternoon" in result.decision.text


def test_spoken_august_twenty_eighth_selects_following_friday() -> None:
    runtime = VoiceProbeV3Runtime()
    row = load_fixture()[14]

    result = asyncio.run(runtime.process_turns([row["text"]]))

    assert result.decision.kind == DecisionKind.CHOOSE_SEARCH_BRANCH
    assert "August 28" in result.decision.text


def test_short_gap_continuation_does_not_reach_fallback() -> None:
    rows = load_fixture()[11:13]
    burst = stabilize_timed_turns(
        make_timed(rows),
        continuation_grace_ms=600.0,
    )[0]

    runtime = VoiceProbeV3Runtime()
    result = asyncio.run(runtime.process_turns(burst.texts))

    assert result.decision.kind == DecisionKind.DECLINE_INCOMPATIBLE_OFFER
    assert result.route.value == "deterministic"


def test_entire_previous_real_audio_fixture_has_zero_fallback_after_stabilization() -> None:
    async def scenario():
        rows = load_fixture()
        bursts = stabilize_timed_turns(
            make_timed(rows),
            continuation_grace_ms=600.0,
        )
        runtime = VoiceProbeV3Runtime()

        actual = []
        for burst in bursts:
            result = await runtime.process_turns(
                burst.texts,
                ingress_reason="frozen_previous_flux_085_stabilized",
            )
            actual.append(result.decision.kind)

        assert DecisionKind.FALLBACK not in actual
        assert actual.count(DecisionKind.HOLD) == 4
        assert DecisionKind.DECLINE_INCOMPATIBLE_OFFER in actual
        assert DecisionKind.CHOOSE_SEARCH_BRANCH in actual

    asyncio.run(scenario())
