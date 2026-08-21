from decimal import Decimal

import pytest

from voiceprobe.execution_state import BudgetExceededError, BudgetLedger, BudgetPolicy


def test_one_call_reservation_must_fit_execution_budget() -> None:
    ledger = BudgetLedger(
        BudgetPolicy(
            total_budget_usd=Decimal("0.29"),
            max_provider_rate_per_minute_usd=Decimal("0.10"),
        )
    )

    with pytest.raises(BudgetExceededError):
        ledger.reserve_call(1, max_duration_seconds=180)


def test_one_call_reservation_uses_provider_rate_and_duration() -> None:
    ledger = BudgetLedger(
        BudgetPolicy(
            total_budget_usd=Decimal("1.00"),
            max_provider_rate_per_minute_usd=Decimal("0.10"),
        )
    )

    entry = ledger.reserve_call(1, max_duration_seconds=180)
    assert entry.reserved_usd == Decimal("0.30")
