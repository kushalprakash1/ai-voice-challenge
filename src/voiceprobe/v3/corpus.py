"""Helpers for the v3 live-call regression corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_corpus_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "fixtures"
        / "v3_calls"
        / "regression_cases.jsonl"
    )


def load_regression_cases(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    corpus_path = path or default_corpus_path()
    rows: list[dict[str, Any]] = []

    with corpus_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    return rows
