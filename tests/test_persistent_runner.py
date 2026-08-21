from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from voiceprobe.execution import (
    LIVE_CONFIRMATION_TOKEN,
    CallStatus,
    authorize_live_execution,
    prepare_execution,
)
from voiceprobe.execution_state import (
    BudgetPolicy,
    PersistentBudgetLedger,
    PersistentCallLedger,
)
from voiceprobe.policy import CallPolicy
from voiceprobe.runner import (
    AssessmentCallRequest,
    AssessmentCallResult,
    run_persistent_authorized_suite,
)
from voiceprobe.scenarios.catalog import list_scenarios
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
        scenarios=list_scenarios()[:scenario_count],
    )

    manifest = prepare_execution(
        policy,
        suite,
        execution_id="persistent-runner-test",
    )

    return authorize_live_execution(
        manifest,
        live_requested=True,
        confirmation_token=LIVE_CONFIRMATION_TOKEN,
    )


def ledgers(
    tmp_path: Path,
    auth,
    *,
    total: str = "5.00",
    rate: str = "0.10",
):
    call_ledger = PersistentCallLedger.initialize(
        auth,
        path=tmp_path / "ledger.json",
    )

    budget_ledger = PersistentBudgetLedger.initialize(
        execution_id=auth.manifest.execution_id,
        policy=BudgetPolicy(
            total_budget_usd=Decimal(total),
            max_provider_rate_per_minute_usd=Decimal(rate),
        ),
        path=tmp_path / "budget.json",
    )

    return call_ledger, budget_ledger


class CostedFakeAdapter:
    def __init__(self) -> None:
        self.positions: list[int] = []

    def execute_call(
        self,
        request: AssessmentCallRequest,
    ) -> AssessmentCallResult:
        self.positions.append(request.position)

        return AssessmentCallResult(
            provider_call_id=f"provider-{request.position}",
            artifact_run_id=f"artifact-{request.position}",
            duration_seconds=30.0,
            provider_cost_usd=Decimal("0.05"),
        )


class CrashOnSecondCall(BaseException):
    pass


class CrashAdapter:
    def __init__(self) -> None:
        self.positions: list[int] = []

    def execute_call(
        self,
        request: AssessmentCallRequest,
    ) -> AssessmentCallResult:
        self.positions.append(request.position)

        if request.position == 2:
            raise CrashOnSecondCall()

        return AssessmentCallResult(
            provider_call_id=f"provider-{request.position}",
            artifact_run_id=f"artifact-{request.position}",
            duration_seconds=30.0,
            provider_cost_usd=Decimal("0.05"),
        )


def test_persistent_runner_completes_and_reconciles(
    tmp_path: Path,
) -> None:
    auth = authorization()
    calls, budget = ledgers(
        tmp_path,
        auth,
    )

    adapter = CostedFakeAdapter()

    result = run_persistent_authorized_suite(
        auth,
        adapter,
        call_ledger=calls,
        budget_ledger=budget,
    )

    assert result.completed_count == 3
    assert adapter.positions == [1, 2, 3]
    assert budget.committed_usd == Decimal("0.15")


def test_budget_is_reserved_before_each_call(
    tmp_path: Path,
) -> None:
    auth = authorization(
        scenario_count=1,
    )
    calls, budget = ledgers(
        tmp_path,
        auth,
    )

    run_persistent_authorized_suite(
        auth,
        CostedFakeAdapter(),
        call_ledger=calls,
        budget_ledger=budget,
    )

    entry = budget.entries[0]

    assert entry.reserved_usd == Decimal("0.30")
    assert entry.actual_usd == Decimal("0.05")


def test_budget_exhaustion_stops_before_dial_attempt(
    tmp_path: Path,
) -> None:
    auth = authorization()
    calls, budget = ledgers(
        tmp_path,
        auth,
        total="0.20",
        rate="0.10",
    )

    adapter = CostedFakeAdapter()

    result = run_persistent_authorized_suite(
        auth,
        adapter,
        call_ledger=calls,
        budget_ledger=budget,
    )

    # A 180-second reservation at $0.10/min costs $0.30,
    # so not even the first adapter call may begin.
    assert adapter.positions == []
    assert result.completed_count == 0
    assert all(entry.status is CallStatus.PLANNED for entry in result.entries)


def test_process_crash_leaves_second_call_started(
    tmp_path: Path,
) -> None:
    auth = authorization()
    calls, budget = ledgers(
        tmp_path,
        auth,
    )

    adapter = CrashAdapter()

    try:
        run_persistent_authorized_suite(
            auth,
            adapter,
            call_ledger=calls,
            budget_ledger=budget,
        )
    except CrashOnSecondCall:
        pass
    else:
        raise AssertionError("Synthetic process crash was swallowed.")

    assert calls.entries[0].status is CallStatus.COMPLETED
    assert calls.entries[1].status is CallStatus.STARTED
    assert calls.entries[2].status is CallStatus.PLANNED


def test_restart_never_replays_completed_or_interrupted_calls(
    tmp_path: Path,
) -> None:
    auth = authorization()

    call_path = tmp_path / "ledger.json"
    budget_path = tmp_path / "budget.json"

    calls = PersistentCallLedger.initialize(
        auth,
        path=call_path,
    )

    policy = BudgetPolicy(
        total_budget_usd=Decimal("5.00"),
        max_provider_rate_per_minute_usd=Decimal("0.10"),
    )

    budget = PersistentBudgetLedger.initialize(
        execution_id=auth.manifest.execution_id,
        policy=policy,
        path=budget_path,
    )

    try:
        run_persistent_authorized_suite(
            auth,
            CrashAdapter(),
            call_ledger=calls,
            budget_ledger=budget,
        )
    except CrashOnSecondCall:
        pass

    recovered_calls = PersistentCallLedger.load(
        auth,
        path=call_path,
    )

    recovered_budget = PersistentBudgetLedger.load(
        execution_id=auth.manifest.execution_id,
        policy=policy,
        path=budget_path,
    )

    resumed_adapter = CostedFakeAdapter()

    result = run_persistent_authorized_suite(
        auth,
        resumed_adapter,
        call_ledger=recovered_calls,
        budget_ledger=recovered_budget,
    )

    assert resumed_adapter.positions == [3]

    assert [entry.status for entry in result.entries] == [
        CallStatus.COMPLETED,
        CallStatus.FAILED,
        CallStatus.COMPLETED,
    ]


def test_crashed_call_keeps_conservative_budget_reservation(
    tmp_path: Path,
) -> None:
    auth = authorization()
    calls, budget = ledgers(
        tmp_path,
        auth,
    )

    try:
        run_persistent_authorized_suite(
            auth,
            CrashAdapter(),
            call_ledger=calls,
            budget_ledger=budget,
        )
    except CrashOnSecondCall:
        pass

    # Call 1 reconciles to $0.05. Call 2 remains at its
    # conservative $0.30 reservation because final cost is unknown.
    assert budget.committed_usd == Decimal("0.35")
