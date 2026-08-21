"""Crash-safe execution state and conservative budget accounting.

This module persists call lifecycle transitions and monetary reservations.
It does not place calls and has no dependency on a telephony provider.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from decimal import ROUND_UP, Decimal
from pathlib import Path
from typing import Final

from voiceprobe.execution import (
    AuthorizedExecution,
    CallLedger,
    CallLedgerEntry,
    CallLedgerError,
    CallStatus,
)
from voiceprobe.policy import MAX_CALL_DURATION_SECONDS

MONEY_RESERVATION_QUANTUM_USD: Final = Decimal("0.01")


class ExecutionStateError(RuntimeError):
    """Raised when persisted execution state is invalid or inconsistent."""


class BudgetStateError(ExecutionStateError):
    """Raised for an invalid budget-ledger transition."""


class BudgetExceededError(BudgetStateError):
    """Raised before a call when its reservation would exceed the budget."""


def _write_json_atomic(
    destination: Path,
    payload: dict[str, object],
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(f".{destination.name}.tmp")

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    temporary.replace(destination)


def _require_mapping(
    value: object,
    *,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ExecutionStateError(f"{name} must be a JSON object.")

    if not all(isinstance(key, str) for key in value):
        raise ExecutionStateError(f"{name} must use string keys.")

    return value


def _require_list(
    value: object,
    *,
    name: str,
) -> list[object]:
    if not isinstance(value, list):
        raise ExecutionStateError(f"{name} must be a JSON array.")

    return value


def _optional_text(
    value: object,
    *,
    name: str,
) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise ExecutionStateError(f"{name} must be a string or null.")

    return value


def _require_position(
    value: object,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ExecutionStateError("Ledger position must be a positive integer.")

    return value


def _require_duration(
    value: object,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ExecutionStateError("Persisted call duration must be numeric.")

    return float(value)


def _ledger_entry_payload(
    entry: CallLedgerEntry,
) -> dict[str, object]:
    return {
        "position": entry.position,
        "scenario_id": entry.scenario_id,
        "status": entry.status.value,
        "provider_call_id": (entry.provider_call_id),
        "artifact_run_id": (entry.artifact_run_id),
        "duration_seconds": (entry.duration_seconds),
        "error": entry.error,
    }


class PersistentCallLedger:
    """CallLedger wrapper that atomically persists every transition."""

    def __init__(
        self,
        *,
        authorization: AuthorizedExecution,
        path: Path,
        ledger: CallLedger,
    ) -> None:
        self._authorization = authorization
        self.path = path
        self._ledger = ledger

    @classmethod
    def initialize(
        cls,
        authorization: AuthorizedExecution,
        *,
        path: Path,
    ) -> PersistentCallLedger:
        """Create a new all-planned persistent ledger."""
        if path.exists():
            raise ExecutionStateError(f"Ledger already exists: {path}")

        instance = cls(
            authorization=authorization,
            path=path,
            ledger=CallLedger(authorization),
        )

        instance._persist()

        return instance

    @classmethod
    def load(
        cls,
        authorization: AuthorizedExecution,
        *,
        path: Path,
    ) -> PersistentCallLedger:
        """Load state and convert an abandoned started call into failed."""
        if not path.is_file():
            raise ExecutionStateError(f"Ledger does not exist: {path}")

        raw = json.loads(path.read_text())

        payload = _require_mapping(
            raw,
            name="ledger",
        )

        execution_id = payload.get("execution_id")

        if execution_id != authorization.manifest.execution_id:
            raise ExecutionStateError(
                "Persisted ledger execution ID does not match authorization."
            )

        raw_entries = _require_list(
            payload.get("entries"),
            name="ledger entries",
        )

        expected_scenarios = authorization.manifest.scenario_ids

        if len(raw_entries) != len(expected_scenarios):
            raise ExecutionStateError(
                "Persisted ledger call count does not match manifest."
            )

        ledger = CallLedger(authorization)

        recovered_interrupted_call = False

        for expected_position, (
            expected_scenario,
            raw_entry,
        ) in enumerate(
            zip(
                expected_scenarios,
                raw_entries,
                strict=True,
            ),
            start=1,
        ):
            entry = _require_mapping(
                raw_entry,
                name=(f"ledger entry {expected_position}"),
            )

            position = _require_position(entry.get("position"))

            scenario_id = entry.get("scenario_id")

            if position != expected_position or scenario_id != expected_scenario:
                raise ExecutionStateError(
                    "Persisted ledger order or scenario IDs "
                    "do not match the execution manifest."
                )

            status_value = entry.get("status")

            if not isinstance(
                status_value,
                str,
            ):
                raise ExecutionStateError("Persisted call status must be a string.")

            try:
                status = CallStatus(status_value)
            except ValueError as error:
                raise ExecutionStateError(
                    f"Unknown persisted call status {status_value!r}."
                ) from error

            provider_call_id = _optional_text(
                entry.get("provider_call_id"),
                name="provider_call_id",
            )

            try:
                if status is CallStatus.PLANNED:
                    continue

                ledger.start_call(
                    position,
                    provider_call_id=(provider_call_id),
                )

                if status is CallStatus.STARTED:
                    # We cannot know whether a crashed process completed
                    # the provider-side call. Never retry it automatically.
                    ledger.fail_call(
                        position,
                        error=(
                            "Interrupted before local completion; "
                            "recovered from persisted started state."
                        ),
                    )

                    recovered_interrupted_call = True
                    continue

                if status is CallStatus.COMPLETED:
                    artifact_run_id = _optional_text(
                        entry.get("artifact_run_id"),
                        name="artifact_run_id",
                    )

                    if artifact_run_id is None:
                        raise ExecutionStateError(
                            "Completed persisted call is missing artifact_run_id."
                        )

                    duration_seconds = _require_duration(entry.get("duration_seconds"))

                    ledger.complete_call(
                        position,
                        duration_seconds=(duration_seconds),
                        artifact_run_id=(artifact_run_id),
                        provider_call_id=(provider_call_id),
                    )

                    continue

                if status is CallStatus.FAILED:
                    error_text = _optional_text(
                        entry.get("error"),
                        name="error",
                    )

                    if error_text is None:
                        raise ExecutionStateError(
                            "Failed persisted call is missing an error."
                        )

                    artifact_run_id = _optional_text(
                        entry.get("artifact_run_id"),
                        name="artifact_run_id",
                    )

                    raw_duration = entry.get("duration_seconds")
                    duration_seconds = (
                        None
                        if raw_duration is None
                        else _require_duration(raw_duration)
                    )

                    ledger.fail_call(
                        position,
                        error=error_text,
                        provider_call_id=(provider_call_id),
                        artifact_run_id=(artifact_run_id),
                        duration_seconds=(duration_seconds),
                    )

            except CallLedgerError as error:
                raise ExecutionStateError(
                    "Persisted call ledger contains an invalid state transition."
                ) from error

        instance = cls(
            authorization=authorization,
            path=path,
            ledger=ledger,
        )

        if recovered_interrupted_call:
            instance._persist()

        return instance

    @property
    def execution_id(self) -> str:
        """Return the execution this ledger is permanently bound to."""
        return self._authorization.manifest.execution_id

    @property
    def entries(
        self,
    ) -> tuple[CallLedgerEntry, ...]:
        return self._ledger.entries

    @property
    def active_call_count(self) -> int:
        return self._ledger.active_call_count

    def start_call(
        self,
        position: int,
        *,
        provider_call_id: str | None = None,
    ) -> CallLedgerEntry:
        entry = self._ledger.start_call(
            position,
            provider_call_id=(provider_call_id),
        )

        self._persist()

        return entry

    def complete_call(
        self,
        position: int,
        *,
        duration_seconds: float,
        artifact_run_id: str,
        provider_call_id: str | None = None,
    ) -> CallLedgerEntry:
        entry = self._ledger.complete_call(
            position,
            duration_seconds=(duration_seconds),
            artifact_run_id=(artifact_run_id),
            provider_call_id=(provider_call_id),
        )

        self._persist()

        return entry

    def fail_call(
        self,
        position: int,
        *,
        error: str,
        provider_call_id: str | None = None,
        artifact_run_id: str | None = None,
        duration_seconds: float | None = None,
    ) -> CallLedgerEntry:
        entry = self._ledger.fail_call(
            position,
            error=error,
            provider_call_id=(provider_call_id),
            artifact_run_id=(artifact_run_id),
            duration_seconds=(duration_seconds),
        )

        self._persist()

        return entry

    def _persist(self) -> None:
        _write_json_atomic(
            self.path,
            {
                "execution_id": (self._authorization.manifest.execution_id),
                "entries": [
                    _ledger_entry_payload(entry) for entry in self._ledger.entries
                ],
            },
        )


def _validate_money(
    value: Decimal,
    *,
    name: str,
    allow_zero: bool,
) -> Decimal:
    if not isinstance(
        value,
        Decimal,
    ):
        raise BudgetStateError(f"{name} must be a Decimal.")

    if not value.is_finite():
        raise BudgetStateError(f"{name} must be finite.")

    minimum_ok = value >= Decimal(0) if allow_zero else value > Decimal(0)

    if not minimum_ok:
        comparison = "non-negative" if allow_zero else "greater than zero"

        raise BudgetStateError(f"{name} must be {comparison}.")

    return value


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """Conservative monetary boundary for one assessment execution."""

    total_budget_usd: Decimal
    max_provider_rate_per_minute_usd: Decimal

    def __post_init__(self) -> None:
        _validate_money(
            self.total_budget_usd,
            name="total_budget_usd",
            allow_zero=False,
        )

        _validate_money(
            self.max_provider_rate_per_minute_usd,
            name=("max_provider_rate_per_minute_usd"),
            allow_zero=False,
        )



@dataclass(frozen=True, slots=True)
class CallBudgetEntry:
    """Reserved and eventually reconciled cost for one call."""

    position: int
    reserved_usd: Decimal
    actual_usd: Decimal | None = None

    @property
    def committed_usd(self) -> Decimal:
        """Use actual cost when known, otherwise retain the reservation."""
        if self.actual_usd is not None:
            return self.actual_usd

        return self.reserved_usd


class BudgetLedger:
    """Conservative per-call reservation and reconciliation ledger."""

    def __init__(
        self,
        policy: BudgetPolicy,
        *,
        entries: tuple[
            CallBudgetEntry,
            ...,
        ] = (),
    ) -> None:
        self.policy = policy
        self._entries: dict[
            int,
            CallBudgetEntry,
        ] = {}

        for entry in entries:
            if entry.position in self._entries:
                raise BudgetStateError(
                    "Budget ledger contains duplicate call positions."
                )

            self._entries[entry.position] = entry

    @property
    def entries(
        self,
    ) -> tuple[
        CallBudgetEntry,
        ...,
    ]:
        return tuple(self._entries[position] for position in sorted(self._entries))

    @property
    def committed_usd(self) -> Decimal:
        return sum(
            (entry.committed_usd for entry in self._entries.values()),
            start=Decimal(0),
        )

    @property
    def remaining_usd(self) -> Decimal:
        return self.policy.total_budget_usd - self.committed_usd

    @property
    def over_budget(self) -> bool:
        return self.committed_usd > self.policy.total_budget_usd

    def worst_case_call_cost(
        self,
        max_duration_seconds: int,
    ) -> Decimal:
        """Reserve conservatively, rounded upward to the next cent."""
        if (
            isinstance(
                max_duration_seconds,
                bool,
            )
            or not isinstance(
                max_duration_seconds,
                int,
            )
            or not 1 <= max_duration_seconds <= MAX_CALL_DURATION_SECONDS
        ):
            raise BudgetStateError(
                f"Maximum call duration must be an integer between 1 and {MAX_CALL_DURATION_SECONDS} seconds."
            )

        seconds = Decimal(max_duration_seconds)

        raw_cost = self.policy.max_provider_rate_per_minute_usd * seconds / Decimal(60)

        return raw_cost.quantize(
            MONEY_RESERVATION_QUANTUM_USD,
            rounding=ROUND_UP,
        )

    def reserve_call(
        self,
        position: int,
        *,
        max_duration_seconds: int,
    ) -> CallBudgetEntry:
        """Reserve worst-case cost before an adapter may start a call."""
        if (
            isinstance(position, bool)
            or not isinstance(
                position,
                int,
            )
            or position < 1
        ):
            raise BudgetStateError("Budget call position must be a positive integer.")

        if position in self._entries:
            raise BudgetStateError(f"Call {position} already has a budget entry.")

        reservation = self.worst_case_call_cost(max_duration_seconds)

        projected = self.committed_usd + reservation

        if projected > self.policy.total_budget_usd:
            raise BudgetExceededError(
                "Starting this call could exceed the configured assessment budget."
            )

        entry = CallBudgetEntry(
            position=position,
            reserved_usd=reservation,
        )

        self._entries[position] = entry

        return entry

    def reconcile_call(
        self,
        position: int,
        *,
        actual_usd: Decimal,
    ) -> CallBudgetEntry:
        """Replace one reservation with confirmed provider cost."""
        actual = _validate_money(
            actual_usd,
            name="actual_usd",
            allow_zero=True,
        )

        try:
            entry = self._entries[position]
        except KeyError as error:
            raise BudgetStateError(
                f"Call {position} has no budget reservation."
            ) from error

        if entry.actual_usd is not None:
            raise BudgetStateError(f"Call {position} cost is already reconciled.")

        updated = replace(
            entry,
            actual_usd=actual,
        )

        self._entries[position] = updated

        # Actual provider cost is factual evidence and must be recorded even
        # if it exceeds the conservative reservation. In that case over_budget
        # becomes true and every later reservation is blocked.
        return updated


def _budget_entry_payload(
    entry: CallBudgetEntry,
) -> dict[str, object]:
    return {
        "position": entry.position,
        "reserved_usd": (
            format(
                entry.reserved_usd,
                "f",
            )
        ),
        "actual_usd": (
            None
            if entry.actual_usd is None
            else format(
                entry.actual_usd,
                "f",
            )
        ),
    }


class PersistentBudgetLedger:
    """BudgetLedger wrapper that atomically persists every mutation."""

    def __init__(
        self,
        *,
        execution_id: str,
        policy: BudgetPolicy,
        path: Path,
        ledger: BudgetLedger,
    ) -> None:
        self.execution_id = execution_id
        self.policy = policy
        self.path = path
        self._ledger = ledger

    @classmethod
    def initialize(
        cls,
        *,
        execution_id: str,
        policy: BudgetPolicy,
        path: Path,
    ) -> PersistentBudgetLedger:
        if path.exists():
            raise BudgetStateError(f"Budget ledger already exists: {path}")

        instance = cls(
            execution_id=execution_id,
            policy=policy,
            path=path,
            ledger=BudgetLedger(policy),
        )

        instance._persist()

        return instance

    @classmethod
    def load(
        cls,
        *,
        execution_id: str,
        policy: BudgetPolicy,
        path: Path,
    ) -> PersistentBudgetLedger:
        if not path.is_file():
            raise BudgetStateError(f"Budget ledger does not exist: {path}")

        raw = json.loads(path.read_text())

        payload = _require_mapping(
            raw,
            name="budget ledger",
        )

        if payload.get("execution_id") != execution_id:
            raise BudgetStateError("Budget execution ID does not match.")

        stored_budget = payload.get("total_budget_usd")

        stored_rate = payload.get("max_provider_rate_per_minute_usd")

        if stored_budget != format(
            policy.total_budget_usd,
            "f",
        ) or stored_rate != format(
            policy.max_provider_rate_per_minute_usd,
            "f",
        ):
            raise BudgetStateError(
                "Persisted budget policy does not match the active budget policy."
            )

        raw_entries = _require_list(
            payload.get("entries"),
            name="budget entries",
        )

        entries: list[CallBudgetEntry] = []

        for raw_entry in raw_entries:
            entry = _require_mapping(
                raw_entry,
                name="budget entry",
            )

            position = _require_position(entry.get("position"))

            reserved_text = entry.get("reserved_usd")

            actual_text = entry.get("actual_usd")

            if not isinstance(
                reserved_text,
                str,
            ):
                raise BudgetStateError("reserved_usd must be stored as a string.")

            try:
                reserved = Decimal(reserved_text)
            except Exception as error:
                raise BudgetStateError(
                    "reserved_usd is not a valid decimal."
                ) from error

            _validate_money(
                reserved,
                name="reserved_usd",
                allow_zero=False,
            )

            actual: Decimal | None = None

            if actual_text is not None:
                if not isinstance(
                    actual_text,
                    str,
                ):
                    raise BudgetStateError(
                        "actual_usd must be stored as a string or null."
                    )

                try:
                    actual = Decimal(actual_text)
                except Exception as error:
                    raise BudgetStateError(
                        "actual_usd is not a valid decimal."
                    ) from error

                _validate_money(
                    actual,
                    name="actual_usd",
                    allow_zero=True,
                )

            entries.append(
                CallBudgetEntry(
                    position=position,
                    reserved_usd=reserved,
                    actual_usd=actual,
                )
            )

        instance = cls(
            execution_id=execution_id,
            policy=policy,
            path=path,
            ledger=BudgetLedger(
                policy,
                entries=tuple(entries),
            ),
        )

        return instance

    @property
    def entries(
        self,
    ) -> tuple[
        CallBudgetEntry,
        ...,
    ]:
        return self._ledger.entries

    @property
    def committed_usd(
        self,
    ) -> Decimal:
        return self._ledger.committed_usd

    @property
    def remaining_usd(
        self,
    ) -> Decimal:
        return self._ledger.remaining_usd

    @property
    def over_budget(
        self,
    ) -> bool:
        return self._ledger.over_budget

    def reserve_call(
        self,
        position: int,
        *,
        max_duration_seconds: int,
    ) -> CallBudgetEntry:
        entry = self._ledger.reserve_call(
            position,
            max_duration_seconds=(max_duration_seconds),
        )

        self._persist()

        return entry

    def reconcile_call(
        self,
        position: int,
        *,
        actual_usd: Decimal,
    ) -> CallBudgetEntry:
        entry = self._ledger.reconcile_call(
            position,
            actual_usd=actual_usd,
        )

        self._persist()

        return entry

    def _persist(self) -> None:
        _write_json_atomic(
            self.path,
            {
                "execution_id": (self.execution_id),
                "total_budget_usd": format(
                    self.policy.total_budget_usd,
                    "f",
                ),
                "max_provider_rate_per_minute_usd": (
                    format(
                        self.policy.max_provider_rate_per_minute_usd,
                        "f",
                    )
                ),
                "committed_usd": format(
                    self._ledger.committed_usd,
                    "f",
                ),
                "remaining_usd": format(
                    self._ledger.remaining_usd,
                    "f",
                ),
                "over_budget": (self._ledger.over_budget),
                "entries": [
                    _budget_entry_payload(entry) for entry in self._ledger.entries
                ],
            },
        )
