from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from voiceprobe.execution import (
    LIVE_CONFIRMATION_TOKEN,
    CallLedgerError,
    CallStatus,
    authorize_live_execution,
    prepare_execution,
)
from voiceprobe.execution_state import (
    BudgetExceededError,
    BudgetLedger,
    BudgetPolicy,
    BudgetStateError,
    ExecutionStateError,
    PersistentBudgetLedger,
    PersistentCallLedger,
)
from voiceprobe.policy import CallPolicy
from voiceprobe.scenarios.catalog import (
    list_scenarios,
)
from voiceprobe.suite import build_suite_plan

ORIGINATING_NUMBER = "+12025550101"


def authorization(
    *,
    scenario_count: int = 3,
):
    policy = CallPolicy(
        originating_number=ORIGINATING_NUMBER,
        dry_run=False,
    )

    suite = build_suite_plan(
        policy,
        scenarios=(list_scenarios()[:scenario_count]),
    )

    manifest = prepare_execution(
        policy,
        suite,
        execution_id="persistent-test",
    )

    return authorize_live_execution(
        manifest,
        live_requested=True,
        confirmation_token=(LIVE_CONFIRMATION_TOKEN),
    )


def budget_policy(
    *,
    total: str = "5.00",
    rate: str = "0.25",
) -> BudgetPolicy:
    return BudgetPolicy(
        total_budget_usd=Decimal(total),
        max_provider_rate_per_minute_usd=(Decimal(rate)),
    )


def test_persistent_call_ledger_initializes_planned(
    tmp_path: Path,
) -> None:
    ledger = PersistentCallLedger.initialize(
        authorization(),
        path=tmp_path / "ledger.json",
    )

    assert [entry.status for entry in ledger.entries] == [
        CallStatus.PLANNED,
        CallStatus.PLANNED,
        CallStatus.PLANNED,
    ]


def test_started_transition_is_persisted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.json"

    ledger = PersistentCallLedger.initialize(
        authorization(),
        path=path,
    )

    ledger.start_call(
        1,
        provider_call_id="provider-1",
    )

    payload = json.loads(path.read_text())

    assert payload["entries"][0]["status"] == "started"
    assert payload["entries"][0]["provider_call_id"] == "provider-1"


def test_completed_transition_survives_reload(
    tmp_path: Path,
) -> None:
    auth = authorization()
    path = tmp_path / "ledger.json"

    ledger = PersistentCallLedger.initialize(
        auth,
        path=path,
    )

    ledger.start_call(
        1,
        provider_call_id="provider-1",
    )

    ledger.complete_call(
        1,
        duration_seconds=42.5,
        artifact_run_id="artifact-1",
    )

    loaded = PersistentCallLedger.load(
        auth,
        path=path,
    )

    entry = loaded.entries[0]

    assert entry.status is CallStatus.COMPLETED
    assert entry.duration_seconds == 42.5
    assert entry.artifact_run_id == "artifact-1"
    assert entry.provider_call_id == "provider-1"


def test_started_call_becomes_failed_after_crash_recovery(
    tmp_path: Path,
) -> None:
    auth = authorization()
    path = tmp_path / "ledger.json"

    ledger = PersistentCallLedger.initialize(
        auth,
        path=path,
    )

    ledger.start_call(
        1,
        provider_call_id="provider-1",
    )

    recovered = PersistentCallLedger.load(
        auth,
        path=path,
    )

    entry = recovered.entries[0]

    assert entry.status is CallStatus.FAILED
    assert entry.provider_call_id == "provider-1"
    assert entry.error is not None
    assert "Interrupted" in entry.error


def test_interrupted_call_cannot_be_retried_automatically(
    tmp_path: Path,
) -> None:
    auth = authorization()
    path = tmp_path / "ledger.json"

    ledger = PersistentCallLedger.initialize(
        auth,
        path=path,
    )

    ledger.start_call(1)

    recovered = PersistentCallLedger.load(
        auth,
        path=path,
    )

    with pytest.raises(
        CallLedgerError,
        match="not in planned state",
    ):
        recovered.start_call(1)


def test_recovery_allows_next_planned_call(
    tmp_path: Path,
) -> None:
    auth = authorization()
    path = tmp_path / "ledger.json"

    ledger = PersistentCallLedger.initialize(
        auth,
        path=path,
    )

    ledger.start_call(1)

    recovered = PersistentCallLedger.load(
        auth,
        path=path,
    )

    entry = recovered.start_call(2)

    assert entry.status is CallStatus.STARTED


def test_tampered_scenario_order_is_rejected(
    tmp_path: Path,
) -> None:
    auth = authorization()
    path = tmp_path / "ledger.json"

    PersistentCallLedger.initialize(
        auth,
        path=path,
    )

    payload = json.loads(path.read_text())

    payload["entries"][0]["scenario_id"] = "tampered-scenario"

    path.write_text(json.dumps(payload))

    with pytest.raises(
        ExecutionStateError,
        match="scenario IDs",
    ):
        PersistentCallLedger.load(
            auth,
            path=path,
        )


def test_budget_policy_accepts_explicit_per_run_ceiling() -> None:
    policy = budget_policy(total="20.00")
    assert policy.total_budget_usd == Decimal("20.00")


def test_budget_reservation_rounds_up_to_cent() -> None:
    ledger = BudgetLedger(
        budget_policy(
            total="5.00",
            rate="0.101",
        )
    )

    # 61 seconds at $0.101/min ~= $0.10268,
    # which must reserve $0.11 rather than round down.
    entry = ledger.reserve_call(
        1,
        max_duration_seconds=61,
    )

    assert entry.reserved_usd == Decimal("0.11")


def test_budget_blocks_call_before_limit_can_be_exceeded() -> None:
    ledger = BudgetLedger(
        budget_policy(
            total="0.50",
            rate="0.20",
        )
    )

    # One 180-second call reserves $0.60.
    with pytest.raises(
        BudgetExceededError,
        match="exceed",
    ):
        ledger.reserve_call(
            1,
            max_duration_seconds=180,
        )

    assert ledger.entries == ()


def test_duplicate_budget_reservation_is_rejected() -> None:
    ledger = BudgetLedger(budget_policy())

    ledger.reserve_call(
        1,
        max_duration_seconds=180,
    )

    with pytest.raises(
        BudgetStateError,
        match="already has",
    ):
        ledger.reserve_call(
            1,
            max_duration_seconds=180,
        )


def test_budget_reservation_survives_reload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "budget.json"
    policy = budget_policy()

    ledger = PersistentBudgetLedger.initialize(
        execution_id="persistent-test",
        policy=policy,
        path=path,
    )

    ledger.reserve_call(
        1,
        max_duration_seconds=180,
    )

    loaded = PersistentBudgetLedger.load(
        execution_id="persistent-test",
        policy=policy,
        path=path,
    )

    assert len(loaded.entries) == 1
    assert loaded.entries[0].reserved_usd == Decimal("0.75")
    assert loaded.committed_usd == Decimal("0.75")


def test_reconciliation_releases_unused_reservation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "budget.json"
    policy = budget_policy()

    ledger = PersistentBudgetLedger.initialize(
        execution_id="persistent-test",
        policy=policy,
        path=path,
    )

    ledger.reserve_call(
        1,
        max_duration_seconds=180,
    )

    ledger.reconcile_call(
        1,
        actual_usd=Decimal("0.20"),
    )

    assert ledger.committed_usd == Decimal("0.20")
    assert ledger.remaining_usd == Decimal("4.80")


def test_actual_cost_above_reservation_is_recorded() -> None:
    ledger = BudgetLedger(
        budget_policy(
            total="5.00",
            rate="0.10",
        )
    )

    ledger.reserve_call(
        1,
        max_duration_seconds=60,
    )

    entry = ledger.reconcile_call(
        1,
        actual_usd=Decimal("0.30"),
    )

    assert entry.reserved_usd == Decimal("0.10")
    assert entry.actual_usd == Decimal("0.30")
    assert ledger.committed_usd == Decimal("0.30")


def test_actual_over_budget_blocks_later_reservation() -> None:
    ledger = BudgetLedger(
        budget_policy(
            total="0.50",
            rate="0.10",
        )
    )

    ledger.reserve_call(
        1,
        max_duration_seconds=60,
    )

    ledger.reconcile_call(
        1,
        actual_usd=Decimal("0.51"),
    )

    assert ledger.over_budget is True

    with pytest.raises(
        BudgetExceededError,
    ):
        ledger.reserve_call(
            2,
            max_duration_seconds=60,
        )


def test_budget_policy_mismatch_is_rejected_on_reload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "budget.json"

    PersistentBudgetLedger.initialize(
        execution_id="persistent-test",
        policy=budget_policy(
            rate="0.25",
        ),
        path=path,
    )

    with pytest.raises(
        BudgetStateError,
        match="does not match",
    ):
        PersistentBudgetLedger.load(
            execution_id="persistent-test",
            policy=budget_policy(
                rate="0.30",
            ),
            path=path,
        )



def test_failed_call_evidence_round_trips(
    tmp_path: Path,
) -> None:
    auth = authorization()
    path = tmp_path / "failed-evidence.json"

    ledger = PersistentCallLedger.initialize(
        auth,
        path=path,
    )

    ledger.start_call(
        1,
        provider_call_id="provider-failed-1",
    )

    failed = ledger.fail_call(
        1,
        error="max duration termination",
        provider_call_id="provider-failed-1",
        artifact_run_id="artifact-failed-1",
        duration_seconds=182.5,
    )

    assert failed.status is CallStatus.FAILED
    assert failed.artifact_run_id == "artifact-failed-1"
    assert failed.duration_seconds == 182.5

    recovered = PersistentCallLedger.load(
        auth,
        path=path,
    )

    entry = recovered.entries[0]

    assert entry.status is CallStatus.FAILED
    assert entry.provider_call_id == "provider-failed-1"
    assert entry.artifact_run_id == "artifact-failed-1"
    assert entry.duration_seconds == 182.5
    assert entry.error == "max duration termination"


def test_failed_call_without_artifact_evidence_still_round_trips(
    tmp_path: Path,
) -> None:
    auth = authorization()
    path = tmp_path / "legacy-failed-entry.json"

    ledger = PersistentCallLedger.initialize(
        auth,
        path=path,
    )

    ledger.start_call(1)

    ledger.fail_call(
        1,
        error="early transport failure",
    )

    recovered = PersistentCallLedger.load(
        auth,
        path=path,
    )

    entry = recovered.entries[0]

    assert entry.status is CallStatus.FAILED
    assert entry.artifact_run_id is None
    assert entry.duration_seconds is None
    assert entry.error == "early transport failure"
