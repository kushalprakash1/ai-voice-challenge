from decimal import Decimal
from pathlib import Path

from voiceprobe.execution import (
    AuthorizedExecution,
    CallStatus,
    ExecutionManifest,
    LIVE_CONFIRMATION_TOKEN,
)
from voiceprobe.execution_state import (
    BudgetPolicy,
    PersistentBudgetLedger,
    PersistentCallLedger,
)
from voiceprobe.runner import (
    AssessmentCallRequest,
    AssessmentCallResult,
    run_authorized_suite,
    run_persistent_authorized_suite,
)
from voiceprobe.safety import ALLOWED_TEST_NUMBER


def authorization() -> AuthorizedExecution:
    manifest = ExecutionManifest(
        execution_id="termination-classification-test",
        suite_id="termination-classification-suite",
        originating_number="+14155550100",
        destination=ALLOWED_TEST_NUMBER,
        scenario_ids=("autonomous-phone-diagnostic",),
        concurrency=1,
        max_call_duration_seconds=180,
        max_suite_calls=1,
        dry_run=False,
    )

    return AuthorizedExecution(
        manifest=manifest,
        confirmation_token=LIVE_CONFIRMATION_TOKEN,
    )


class PrematureRemoteAdapter:
    def execute_call(
        self,
        request: AssessmentCallRequest,
    ) -> AssessmentCallResult:
        return AssessmentCallResult(
            provider_call_id=f"asterisk-{request.position}",
            artifact_run_id=f"artifact-{request.position}",
            duration_seconds=42.5,
            provider_cost_usd=Decimal("0.05"),
            assessment_succeeded=False,
            failure_reason=(
                "premature_remote_termination: call ended before "
                "objective completion; booking_confirmed=False; "
                "offer_accepted=True; offered_day='Friday'; "
                "offered_time='2:30 PM'"
            ),
        )


def test_in_memory_runner_marks_semantic_call_failure() -> None:
    result = run_authorized_suite(
        authorization(),
        PrematureRemoteAdapter(),
    )

    assert result.completed_count == 0
    assert result.failed_count == 1

    entry = result.entries[0]

    assert entry.status is CallStatus.FAILED
    assert entry.provider_call_id == "asterisk-1"
    assert entry.artifact_run_id == "artifact-1"
    assert entry.duration_seconds == 42.5
    assert entry.error is not None
    assert "premature_remote_termination" in entry.error
    assert "artifact_run_id=artifact-1" in entry.error
    assert "duration_seconds=42.5" in entry.error


def test_persistent_runner_marks_failure_and_reconciles_cost(
    tmp_path: Path,
) -> None:
    auth = authorization()

    calls = PersistentCallLedger.initialize(
        auth,
        path=tmp_path / "calls.json",
    )

    budget = PersistentBudgetLedger.initialize(
        execution_id=auth.manifest.execution_id,
        policy=BudgetPolicy(
            total_budget_usd=Decimal("1.00"),
            max_provider_rate_per_minute_usd=Decimal("0.10"),
        ),
        path=tmp_path / "budget.json",
    )

    result = run_persistent_authorized_suite(
        auth,
        PrematureRemoteAdapter(),
        call_ledger=calls,
        budget_ledger=budget,
    )

    assert result.completed_count == 0
    assert result.failed_count == 1

    entry = result.entries[0]

    assert entry.status is CallStatus.FAILED
    assert entry.provider_call_id == "asterisk-1"
    assert entry.artifact_run_id == "artifact-1"
    assert entry.duration_seconds == 42.5
    assert entry.error is not None
    assert "premature_remote_termination" in entry.error
    assert "artifact_run_id=artifact-1" in entry.error

    assert len(budget.entries) == 1
    assert budget.entries[0].actual_usd == Decimal("0.05")
