"""Execution controls for VoiceProbe assessment suites.

This module defines the boundary between planning and real execution.
It does not contain a telephony-provider implementation and cannot dial.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Final
from uuid import uuid4

from voiceprobe.policy import CallPolicy
from voiceprobe.policy import MAX_CALL_DURATION_SECONDS
from voiceprobe.safety import validate_destination
from voiceprobe.suite import AssessmentSuitePlan

LIVE_CONFIRMATION_TOKEN: Final = "AUTHORIZE_ASSESSMENT_CALLS"
EXECUTION_CONCURRENCY: Final = 1

_EXECUTION_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


class ExecutionSafetyError(ValueError):
    """Raised when execution would violate a VoiceProbe safety boundary."""


class CallLedgerError(RuntimeError):
    """Raised when a call attempts an invalid state transition."""


class CallStatus(StrEnum):
    """Lifecycle states for one planned assessment call."""

    PLANNED = "planned"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ExecutionManifest:
    """Immutable description of exactly what an execution may attempt."""

    execution_id: str
    suite_id: str
    originating_number: str
    destination: str
    scenario_ids: tuple[str, ...]
    concurrency: int
    max_call_duration_seconds: int
    max_suite_calls: int
    dry_run: bool

    @property
    def call_count(self) -> int:
        """Return the number of calls authorized by this manifest."""
        return len(self.scenario_ids)


@dataclass(frozen=True, slots=True)
class AuthorizedExecution:
    """Manifest that crossed the explicit live-execution boundary."""

    manifest: ExecutionManifest
    confirmation_token: str


@dataclass(frozen=True, slots=True)
class CallLedgerEntry:
    """State of one scenario call inside an authorized execution."""

    position: int
    scenario_id: str
    status: CallStatus
    provider_call_id: str | None = None
    artifact_run_id: str | None = None
    duration_seconds: float | None = None
    error: str | None = None


def prepare_execution(
    policy: CallPolicy,
    suite_plan: AssessmentSuitePlan,
    *,
    execution_id: str | None = None,
) -> ExecutionManifest:
    """Freeze a validated suite into an immutable execution manifest."""
    destination = validate_destination(suite_plan.destination)

    if suite_plan.originating_number != policy.originating_number:
        raise ExecutionSafetyError(
            "Suite originating number does not match the active policy."
        )

    if suite_plan.concurrency != EXECUTION_CONCURRENCY:
        raise ExecutionSafetyError(
            "Assessment execution concurrency must be exactly one."
        )

    if suite_plan.call_count > policy.max_suite_calls:
        raise ExecutionSafetyError("Execution exceeds the active maximum suite size.")

    if suite_plan.max_call_duration_seconds != policy.max_call_duration_seconds:
        raise ExecutionSafetyError(
            "Suite call-duration limit does not match the active policy."
        )

    if not suite_plan.calls:
        raise ExecutionSafetyError("Execution must contain at least one scenario.")

    resolved_execution_id = (
        execution_id if execution_id is not None else f"assessment-{uuid4().hex[:12]}"
    )

    if not _EXECUTION_ID_PATTERN.fullmatch(resolved_execution_id):
        raise ExecutionSafetyError(
            "Execution ID must contain only lowercase letters, "
            "numbers, underscores, and hyphens."
        )

    return ExecutionManifest(
        execution_id=resolved_execution_id,
        suite_id=suite_plan.suite_id,
        originating_number=policy.originating_number,
        destination=destination,
        scenario_ids=tuple(call.scenario_id for call in suite_plan.calls),
        concurrency=EXECUTION_CONCURRENCY,
        max_call_duration_seconds=(policy.max_call_duration_seconds),
        max_suite_calls=policy.max_suite_calls,
        dry_run=policy.dry_run,
    )


def authorize_live_execution(
    manifest: ExecutionManifest,
    *,
    live_requested: bool,
    confirmation_token: str,
) -> AuthorizedExecution:
    """Cross the explicit boundary from planning into live authorization."""
    validate_destination(manifest.destination)

    if manifest.dry_run:
        raise ExecutionSafetyError(
            "Live execution is forbidden while dry_run is enabled."
        )

    if not live_requested:
        raise ExecutionSafetyError("Live execution requires an explicit live request.")

    if confirmation_token != LIVE_CONFIRMATION_TOKEN:
        raise ExecutionSafetyError("Live execution confirmation token is invalid.")

    if manifest.concurrency != EXECUTION_CONCURRENCY:
        raise ExecutionSafetyError("Live execution concurrency must be exactly one.")

    if not 1 <= manifest.call_count <= manifest.max_suite_calls:
        raise ExecutionSafetyError("Manifest call count violates the suite limit.")

    if not 1 <= manifest.max_call_duration_seconds <= MAX_CALL_DURATION_SECONDS:
        raise ExecutionSafetyError("Manifest call-duration limit is unsafe.")

    return AuthorizedExecution(
        manifest=manifest,
        confirmation_token=confirmation_token,
    )


def write_execution_manifest(
    manifest: ExecutionManifest,
    *,
    root: Path = Path("artifacts/executions"),
) -> Path:
    """Persist a prepared manifest locally without enabling execution."""
    execution_dir = root / manifest.execution_id
    execution_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    destination = execution_dir / "manifest.json"
    temporary = execution_dir / ".manifest.json.tmp"

    temporary.write_text(
        json.dumps(
            asdict(manifest),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    temporary.replace(destination)

    return destination


class CallLedger:
    """In-memory lifecycle ledger enforcing single-call execution."""

    def __init__(
        self,
        authorization: AuthorizedExecution,
    ) -> None:
        self._authorization = authorization
        self._entries = {
            position: CallLedgerEntry(
                position=position,
                scenario_id=scenario_id,
                status=CallStatus.PLANNED,
            )
            for position, scenario_id in enumerate(
                authorization.manifest.scenario_ids,
                start=1,
            )
        }

    @property
    def entries(self) -> tuple[CallLedgerEntry, ...]:
        """Return call states in deterministic suite order."""
        return tuple(self._entries[position] for position in sorted(self._entries))

    @property
    def active_call_count(self) -> int:
        """Return the number of calls currently marked started."""
        return sum(
            entry.status is CallStatus.STARTED for entry in self._entries.values()
        )

    def start_call(
        self,
        position: int,
        *,
        provider_call_id: str | None = None,
    ) -> CallLedgerEntry:
        """Transition one planned call to started."""
        entry = self._get(position)

        if entry.status is not CallStatus.PLANNED:
            raise CallLedgerError(f"Call {position} is not in planned state.")

        if self.active_call_count >= EXECUTION_CONCURRENCY:
            raise CallLedgerError("Another assessment call is already active.")

        updated = replace(
            entry,
            status=CallStatus.STARTED,
            provider_call_id=provider_call_id,
        )

        self._entries[position] = updated

        return updated

    def complete_call(
        self,
        position: int,
        *,
        duration_seconds: float,
        artifact_run_id: str,
        provider_call_id: str | None = None,
    ) -> CallLedgerEntry:
        """Transition one active call to completed."""
        entry = self._require_started(position)

        if isinstance(duration_seconds, bool) or not isinstance(
            duration_seconds,
            (int, float),
        ):
            raise CallLedgerError("Call duration must be numeric.")

        duration = float(duration_seconds)

        if not (
            0.0 <= duration <= self._authorization.manifest.max_call_duration_seconds
        ):
            raise CallLedgerError("Call duration exceeds the execution limit.")

        normalized_artifact = artifact_run_id.strip()

        if not normalized_artifact:
            raise CallLedgerError("Completed calls require an artifact run ID.")

        updated = replace(
            entry,
            status=CallStatus.COMPLETED,
            provider_call_id=(
                provider_call_id
                if provider_call_id is not None
                else entry.provider_call_id
            ),
            artifact_run_id=normalized_artifact,
            duration_seconds=duration,
        )

        self._entries[position] = updated

        return updated

    def fail_call(
        self,
        position: int,
        *,
        error: str,
        provider_call_id: str | None = None,
        artifact_run_id: str | None = None,
        duration_seconds: float | None = None,
    ) -> CallLedgerEntry:
        """Transition one active call to failed without automatic retry."""
        entry = self._require_started(position)
        normalized_error = " ".join(error.split())

        if not normalized_error:
            raise CallLedgerError("Failed calls require an error description.")

        normalized_artifact: str | None = None

        if artifact_run_id is not None:
            normalized_artifact = artifact_run_id.strip()

            if not normalized_artifact:
                raise CallLedgerError(
                    "Failed call artifact run ID cannot be empty."
                )

        duration: float | None = None

        if duration_seconds is not None:
            if isinstance(duration_seconds, bool) or not isinstance(
                duration_seconds,
                (int, float),
            ):
                raise CallLedgerError(
                    "Failed call duration must be numeric."
                )

            duration = float(duration_seconds)

            # A failed call may exceed the requested maximum slightly while
            # AudioSocket/recorder teardown completes, so unlike successful
            # calls we require only a non-negative measured duration.
            if duration < 0.0:
                raise CallLedgerError(
                    "Failed call duration cannot be negative."
                )

        updated = replace(
            entry,
            status=CallStatus.FAILED,
            provider_call_id=(
                provider_call_id
                if provider_call_id is not None
                else entry.provider_call_id
            ),
            artifact_run_id=normalized_artifact,
            duration_seconds=duration,
            error=normalized_error,
        )

        self._entries[position] = updated

        return updated

    def _get(
        self,
        position: int,
    ) -> CallLedgerEntry:
        try:
            return self._entries[position]
        except KeyError as error:
            raise CallLedgerError(f"Unknown call position {position}.") from error

    def _require_started(
        self,
        position: int,
    ) -> CallLedgerEntry:
        entry = self._get(position)

        if entry.status is not CallStatus.STARTED:
            raise CallLedgerError(f"Call {position} is not in started state.")

        return entry
