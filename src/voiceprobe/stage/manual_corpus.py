"""Load human reconnaissance calls stored as StageLab JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ManualTurn:
    speaker: str
    text: str


@dataclass(frozen=True, slots=True)
class ManualCall:
    call_id: str
    mission: str
    turns: tuple[ManualTurn, ...]
    notes: tuple[str, ...]


def load_manual_call(path: str | Path) -> ManualCall:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ManualCall(
        call_id=payload["call_id"],
        mission=payload["mission"],
        turns=tuple(ManualTurn(**turn) for turn in payload["turns"]),
        notes=tuple(payload.get("notes", [])),
    )
