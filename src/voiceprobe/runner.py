"""Sequential execution orchestration for authorized assessment suites.

The runner knows nothing about Telnyx, Asterisk, SIP, or any other provider.
A concrete call adapter must be injected. This keeps execution behavior fully
testable before a production telephony implementation exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, runtime_checkable

from voiceprobe.execution import (
    AuthorizedExecution,
    CallLedger,
    CallLedgerEntry,
    CallStatus,
)
from voiceprobe.execution_state import (
    BudgetExceededError,
    PersistentBudgetLedger,
    PersistentCallLedger,
)
from voiceprobe.safety import validate_destination


class CallExecutionError(RuntimeError):
    """Raised when an injected call adapter cannot complete one call."""


@dataclass(frozen=True, slots=True)
class AssessmentCallRequest:
    """Everything an adapter is permitted to use for one assessment call."""

    execution_id: str
    position: int
    scenario_id: str
    originating_number: str
    destination: str
    max_duration_seconds: int


@dataclass(frozen=True, slots=True)
class AssessmentCallResult:
    """Evidence returned after one complete call attempt.

    Transport completion is not automatically assessment success. A remote
    system can terminate a perfectly valid phone call before the scenario
    objective has been achieved.
    """

    provider_call_id: str
    artifact_run_id: str
    duration_seconds: float
    provider_cost_usd: Decimal | None = None
    assessment_succeeded: bool = True
    failure_reason: str | None = None


def _assessment_failure_error(
    *,
    result: AssessmentCallResult,
    artifact_run_id: str,
) -> str | None:
    """Return ledger-safe failure evidence for an unsuccessful assessment."""
    if type(result.assessment_succeeded) is not bool:
        raise TypeError("assessment_succeeded must be a boolean.")

    if result.assessment_succeeded:
        return None

    failure_reason = " ".join(
        (result.failure_reason or "").split()
    )

    if not failure_reason:
        raise CallExecutionError(
            "Unsuccessful assessment result requires a failure reason."
        )

    # Keep the evidence in the legacy error text for backwards-compatible
    # diagnostics while also storing it structurally on the failed ledger entry.
    return (
        f"{failure_reason} "
        f"[artifact_run_id={artifact_run_id}; "
        f"duration_seconds={result.duration_seconds!r}]"
    )


@dataclass(frozen=True, slots=True)
class SuiteRunResult:
    """Final deterministic state of one sequential suite execution."""

    execution_id: str
    entries: tuple[CallLedgerEntry, ...]

    @property
    def completed_count(self) -> int:
        return sum(entry.status is CallStatus.COMPLETED for entry in self.entries)

    @property
    def failed_count(self) -> int:
        return sum(entry.status is CallStatus.FAILED for entry in self.entries)


@runtime_checkable
class AssessmentCallAdapter(Protocol):
    """Injected boundary responsible for exactly one complete call."""

    def execute_call(
        self,
        request: AssessmentCallRequest,
    ) -> AssessmentCallResult:
        """Execute one call and return its provider/artifact evidence."""


def run_authorized_suite(
    authorization: AuthorizedExecution,
    adapter: AssessmentCallAdapter,
) -> SuiteRunResult:
    """Execute an authorized suite serially with zero automatic retries."""
    manifest = authorization.manifest

    validate_destination(manifest.destination)

    if manifest.concurrency != 1:
        raise CallExecutionError(
            "Suite runner requires concurrency exactly equal to one."
        )

    ledger = CallLedger(authorization)

    for position, scenario_id in enumerate(
        manifest.scenario_ids,
        start=1,
    ):
        request = AssessmentCallRequest(
            execution_id=manifest.execution_id,
            position=position,
            scenario_id=scenario_id,
            originating_number=manifest.originating_number,
            destination=manifest.destination,
            max_duration_seconds=(manifest.max_call_duration_seconds),
        )

        # Revalidate at the last possible point before handing the request
        # to an injected execution adapter.
        validate_destination(request.destination)

        ledger.start_call(position)

        try:
            result = adapter.execute_call(request)

            provider_call_id = result.provider_call_id.strip()

            artifact_run_id = result.artifact_run_id.strip()

            if not provider_call_id:
                raise CallExecutionError(
                    "Call adapter returned an empty provider call ID."
                )

            if not artifact_run_id:
                raise CallExecutionError(
                    "Call adapter returned an empty artifact run ID."
                )

            failure_error = _assessment_failure_error(
                result=result,
                artifact_run_id=artifact_run_id,
            )

            if failure_error is None:
                ledger.complete_call(
                    position,
                    duration_seconds=(result.duration_seconds),
                    artifact_run_id=artifact_run_id,
                    provider_call_id=provider_call_id,
                )
            else:
                ledger.fail_call(
                    position,
                    error=failure_error,
                    provider_call_id=provider_call_id,
                    artifact_run_id=artifact_run_id,
                    duration_seconds=(result.duration_seconds),
                )

        except (
            CallExecutionError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            # One attempt only. A failed scenario is recorded and execution
            # advances to the next planned scenario without retrying it.
            ledger.fail_call(
                position,
                error=(f"{type(error).__name__}: {error}"),
            )

    return SuiteRunResult(
        execution_id=manifest.execution_id,
        entries=ledger.entries,
    )


def run_persistent_authorized_suite(
    authorization: AuthorizedExecution,
    adapter: AssessmentCallAdapter,
    *,
    call_ledger: PersistentCallLedger,
    budget_ledger: PersistentBudgetLedger,
) -> SuiteRunResult:
    """Run or resume one suite using crash-safe call and budget state.

    Completed and failed calls are never automatically replayed. A budget
    reservation is persisted before a planned call may cross into started
    state. If the process dies after start, ledger recovery converts that
    abandoned call to failed on the next launch.
    """
    manifest = authorization.manifest

    validate_destination(manifest.destination)

    if manifest.concurrency != 1:
        raise CallExecutionError(
            "Persistent suite runner requires concurrency exactly one."
        )

    if call_ledger.execution_id != manifest.execution_id:
        raise CallExecutionError(
            "Call ledger execution ID does not match authorization."
        )

    if budget_ledger.execution_id != manifest.execution_id:
        raise CallExecutionError(
            "Budget ledger execution ID does not match authorization."
        )

    for position, scenario_id in enumerate(
        manifest.scenario_ids,
        start=1,
    ):
        current = call_ledger.entries[position - 1]

        # Resume semantics are intentionally conservative. Completed,
        # failed, and crash-recovered calls are never silently replayed.
        if current.status is not CallStatus.PLANNED:
            continue

        request = AssessmentCallRequest(
            execution_id=manifest.execution_id,
            position=position,
            scenario_id=scenario_id,
            originating_number=manifest.originating_number,
            destination=manifest.destination,
            max_duration_seconds=(manifest.max_call_duration_seconds),
        )

        validate_destination(request.destination)

        existing_budget_positions = {entry.position for entry in budget_ledger.entries}

        try:
            # Reuse an already-persisted reservation if a previous process
            # died after reservation but before the call entered started.
            if position not in existing_budget_positions:
                budget_ledger.reserve_call(
                    position,
                    max_duration_seconds=(request.max_duration_seconds),
                )

            call_ledger.start_call(position)

            result = adapter.execute_call(request)

            provider_call_id = result.provider_call_id.strip()
            artifact_run_id = result.artifact_run_id.strip()

            if not provider_call_id:
                raise CallExecutionError(
                    "Call adapter returned an empty provider call ID."
                )

            if not artifact_run_id:
                raise CallExecutionError(
                    "Call adapter returned an empty artifact run ID."
                )

            failure_error = _assessment_failure_error(
                result=result,
                artifact_run_id=artifact_run_id,
            )

            if failure_error is None:
                call_ledger.complete_call(
                    position,
                    duration_seconds=(result.duration_seconds),
                    artifact_run_id=artifact_run_id,
                    provider_call_id=provider_call_id,
                )
            else:
                call_ledger.fail_call(
                    position,
                    error=failure_error,
                    provider_call_id=provider_call_id,
                    artifact_run_id=artifact_run_id,
                    duration_seconds=(result.duration_seconds),
                )

            # A failed assessment can still have incurred a real provider
            # charge, so reconcile cost independently of semantic success.
            if result.provider_cost_usd is not None:
                budget_ledger.reconcile_call(
                    position,
                    actual_usd=(result.provider_cost_usd),
                )

        except BudgetExceededError:
            # Budget exhaustion stops the suite before another call starts.
            break

        except (
            CallExecutionError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            # Do not retry. Preserve the failure and continue to the next
            # still-planned scenario.
            current = call_ledger.entries[position - 1]

            if current.status is CallStatus.STARTED:
                call_ledger.fail_call(
                    position,
                    error=(f"{type(error).__name__}: {error}"),
                )

    return SuiteRunResult(
        execution_id=manifest.execution_id,
        entries=call_ledger.entries,
    )
