"""Short-gap stabilization for consecutive Flux EndOfTurn segments.

Deepgram Flux can occasionally finalize a clause and then emit a continuation
hundreds of milliseconds later. VoiceProbe uses a small grace interval before
speaking so those segments can be treated as one conversational burst.

This module is transport-agnostic and is used by offline audio replay now. The
600 ms remains the generic/offline baseline. The production Pipecat ingress may deliberately configure a larger continuation window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DEFAULT_CONTINUATION_GRACE_MS = 600.0


@dataclass(frozen=True, slots=True)
class TimedRemoteTurn:
    text: str
    gap_to_next_start_ms: float | None = None
    turn_index: int | None = None


@dataclass(frozen=True, slots=True)
class StabilizedBurst:
    turns: tuple[TimedRemoteTurn, ...]

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(turn.text for turn in self.turns)

    @property
    def turn_indices(self) -> tuple[int | None, ...]:
        return tuple(turn.turn_index for turn in self.turns)


def stabilize_timed_turns(
    turns: Iterable[TimedRemoteTurn],
    *,
    continuation_grace_ms: float = DEFAULT_CONTINUATION_GRACE_MS,
) -> tuple[StabilizedBurst, ...]:
    """Group an EOT with the next EOT when new speech starts very quickly."""

    if continuation_grace_ms < 0:
        raise ValueError("continuation_grace_ms must be non-negative")

    source = tuple(turn for turn in turns if turn.text.strip())
    if not source:
        return ()

    bursts: list[StabilizedBurst] = []
    current: list[TimedRemoteTurn] = []

    for index, turn in enumerate(source):
        current.append(turn)

        has_next = index + 1 < len(source)
        gap = turn.gap_to_next_start_ms

        should_wait_for_continuation = (
            has_next
            and gap is not None
            and gap <= continuation_grace_ms
        )

        if not should_wait_for_continuation:
            bursts.append(StabilizedBurst(tuple(current)))
            current = []

    if current:
        bursts.append(StabilizedBurst(tuple(current)))

    return tuple(bursts)
