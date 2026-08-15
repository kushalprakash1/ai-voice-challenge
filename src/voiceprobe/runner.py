"""Sequential execution orchestration for authorized assessment suites.

The runner knows nothing about Telnyx, Asterisk, SIP, or any other provider.
A concrete call adapter must be injected. This keeps execution behavior fully
testable before a production telephony implementation exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from voiceprobe.execution import (
    AuthorizedExecution,
    CallLedger,
    CallLedgerEntry,
    CallStatus,
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
    """Evidence returned after one complete call attempt."""

    provider_call_id: str
    artifact_run_id: str
    duration_seconds: float


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

            ledger.complete_call(
                position,
                duration_seconds=(result.duration_seconds),
                artifact_run_id=artifact_run_id,
                provider_call_id=provider_call_id,
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
