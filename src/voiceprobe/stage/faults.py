"""Deterministic perturbations that mimic common live voice transport faults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StagedSegment:
    text: str
    delay_ms: float


def split_tail_question(text: str) -> tuple[StagedSegment, ...]:
    """Split a compound question near its last alternative when possible."""
    lower = text.casefold()
    marker = " or should "
    index = lower.rfind(marker)
    if index <= 0:
        return (StagedSegment(text, 0.0),)

    prefix = text[:index].rstrip() + ","
    suffix = text[index + 1 :].lstrip()
    return (
        StagedSegment(prefix, 0.0),
        StagedSegment(suffix, 420.0),
    )


def repeat_after(text: str, *, delay_ms: float = 3200.0) -> tuple[StagedSegment, ...]:
    return (
        StagedSegment(text, 0.0),
        StagedSegment(text, delay_ms),
    )
