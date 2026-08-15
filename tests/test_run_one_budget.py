from decimal import Decimal
from pathlib import Path

import pytest

from voiceprobe.execution_state import BudgetStateError
from voiceprobe.run_one import (
    cumulative_assessment_commitment,
    enforce_cumulative_budget,
)


def write_budget(
    path: Path,
    *,
    reserved: str,
    actual: str | None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        (
            "{"
            '"execution_id": "test",'
            '"total_budget_usd": "5.00",'
            '"max_provider_rate_per_minute_usd": "0.10",'
            '"entries": ['
            "{"
            '"position": 1,'
            f'"reserved_usd": "{reserved}",'
            f'"actual_usd": ' + ("null" if actual is None else f'"{actual}"') + "}"
            "]"
            "}"
        ),
        encoding="utf-8",
    )


def test_cumulative_commitment_uses_reservation_when_actual_unknown(
    tmp_path: Path,
) -> None:
    write_budget(
        tmp_path / "run-a" / "budget.json",
        reserved="3.00",
        actual=None,
    )

    assert cumulative_assessment_commitment(tmp_path) == Decimal("3.00")


def test_cumulative_commitment_prefers_actual_provider_cost(
    tmp_path: Path,
) -> None:
    write_budget(
        tmp_path / "run-a" / "budget.json",
        reserved="0.30",
        actual="0.02",
    )

    assert cumulative_assessment_commitment(tmp_path) == Decimal("0.02")


def test_cumulative_commitment_sums_multiple_executions(
    tmp_path: Path,
) -> None:
    write_budget(
        tmp_path / "run-a" / "budget.json",
        reserved="3.00",
        actual=None,
    )
    write_budget(
        tmp_path / "run-b" / "budget.json",
        reserved="0.30",
        actual=None,
    )

    assert cumulative_assessment_commitment(tmp_path) == Decimal("3.30")


def test_cumulative_guard_allows_safe_next_call() -> None:
    enforce_cumulative_budget(
        prior_commitment_usd=Decimal("3.00"),
        next_call_reservation_usd=Decimal("0.30"),
    )


def test_cumulative_guard_blocks_hard_ceiling() -> None:
    with pytest.raises(
        BudgetStateError,
        match="cumulative",
    ):
        enforce_cumulative_budget(
            prior_commitment_usd=Decimal("19.70"),
            next_call_reservation_usd=Decimal("0.30"),
        )
