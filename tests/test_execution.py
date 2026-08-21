from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from voiceprobe.execution import (
    LIVE_CONFIRMATION_TOKEN,
    CallLedger,
    CallLedgerError,
    CallStatus,
    ExecutionSafetyError,
    authorize_live_execution,
    prepare_execution,
    write_execution_manifest,
)
from voiceprobe.policy import CallPolicy
from voiceprobe.safety import ALLOWED_TEST_NUMBER
from voiceprobe.suite import build_suite_plan

ORIGINATING_NUMBER = "+12025550101"


def policy(
    *,
    dry_run: bool,
) -> CallPolicy:
    return CallPolicy(
        originating_number=ORIGINATING_NUMBER,
        dry_run=dry_run,
    )


def prepared_manifest(
    *,
    dry_run: bool,
):
    active_policy = policy(
        dry_run=dry_run,
    )

    return prepare_execution(
        active_policy,
        build_suite_plan(active_policy),
        execution_id="assessment-test",
    )


def authorized_execution():
    manifest = prepared_manifest(
        dry_run=False,
    )

    return authorize_live_execution(
        manifest,
        live_requested=True,
        confirmation_token=LIVE_CONFIRMATION_TOKEN,
    )


def test_prepare_execution_freezes_exact_suite() -> None:
    manifest = prepared_manifest(
        dry_run=True,
    )

    assert manifest.call_count == 16
    assert manifest.destination == ALLOWED_TEST_NUMBER
    assert manifest.originating_number == ORIGINATING_NUMBER
    assert manifest.concurrency == 1
    assert manifest.max_call_duration_seconds == 180


def test_prepare_rejects_invalid_execution_id() -> None:
    active_policy = policy(
        dry_run=True,
    )

    with pytest.raises(
        ExecutionSafetyError,
        match="Execution ID",
    ):
        prepare_execution(
            active_policy,
            build_suite_plan(active_policy),
            execution_id="INVALID ID",
        )


def test_prepare_rejects_changed_destination() -> None:
    active_policy = policy(
        dry_run=True,
    )

    suite = replace(
        build_suite_plan(active_policy),
        destination="+12025550109",
    )

    with pytest.raises(ValueError):
        prepare_execution(
            active_policy,
            suite,
        )


def test_prepare_rejects_non_single_concurrency() -> None:
    active_policy = policy(
        dry_run=True,
    )

    suite = replace(
        build_suite_plan(active_policy),
        concurrency=2,
    )

    with pytest.raises(
        ExecutionSafetyError,
        match="concurrency",
    ):
        prepare_execution(
            active_policy,
            suite,
        )


def test_dry_run_manifest_cannot_be_authorized() -> None:
    manifest = prepared_manifest(
        dry_run=True,
    )

    with pytest.raises(
        ExecutionSafetyError,
        match="dry_run",
    ):
        authorize_live_execution(
            manifest,
            live_requested=True,
            confirmation_token=LIVE_CONFIRMATION_TOKEN,
        )


def test_live_flag_is_required() -> None:
    manifest = prepared_manifest(
        dry_run=False,
    )

    with pytest.raises(
        ExecutionSafetyError,
        match="explicit live request",
    ):
        authorize_live_execution(
            manifest,
            live_requested=False,
            confirmation_token=LIVE_CONFIRMATION_TOKEN,
        )


def test_confirmation_token_is_required() -> None:
    manifest = prepared_manifest(
        dry_run=False,
    )

    with pytest.raises(
        ExecutionSafetyError,
        match="confirmation token",
    ):
        authorize_live_execution(
            manifest,
            live_requested=True,
            confirmation_token="wrong-token",
        )


def test_valid_manifest_can_cross_live_boundary() -> None:
    authorization = authorized_execution()

    assert authorization.manifest.call_count == 16
    assert authorization.confirmation_token == LIVE_CONFIRMATION_TOKEN


def test_execution_manifest_is_written_locally(
    tmp_path: Path,
) -> None:
    manifest = prepared_manifest(
        dry_run=True,
    )

    path = write_execution_manifest(
        manifest,
        root=tmp_path,
    )

    payload = json.loads(path.read_text())

    assert payload["execution_id"] == "assessment-test"
    assert payload["destination"] == ALLOWED_TEST_NUMBER
    assert payload["dry_run"] is True
    assert len(payload["scenario_ids"]) == 16


def test_ledger_starts_planned_call() -> None:
    ledger = CallLedger(authorized_execution())

    entry = ledger.start_call(
        1,
        provider_call_id="provider-1",
    )

    assert entry.status is CallStatus.STARTED
    assert ledger.active_call_count == 1


def test_ledger_prevents_concurrent_calls() -> None:
    ledger = CallLedger(authorized_execution())

    ledger.start_call(1)

    with pytest.raises(
        CallLedgerError,
        match="already active",
    ):
        ledger.start_call(2)


def test_completed_call_requires_started_state() -> None:
    ledger = CallLedger(authorized_execution())

    with pytest.raises(
        CallLedgerError,
        match="started state",
    ):
        ledger.complete_call(
            1,
            duration_seconds=30.0,
            artifact_run_id="run-1",
        )


def test_completed_call_enforces_duration_cap() -> None:
    ledger = CallLedger(authorized_execution())

    ledger.start_call(1)

    with pytest.raises(
        CallLedgerError,
        match="duration",
    ):
        ledger.complete_call(
            1,
            duration_seconds=181.0,
            artifact_run_id="run-1",
        )


def test_completed_call_records_artifact() -> None:
    ledger = CallLedger(authorized_execution())

    ledger.start_call(
        1,
        provider_call_id="provider-1",
    )

    entry = ledger.complete_call(
        1,
        duration_seconds=62.5,
        artifact_run_id="artifact-run-1",
    )

    assert entry.status is CallStatus.COMPLETED
    assert entry.duration_seconds == 62.5
    assert entry.artifact_run_id == "artifact-run-1"
    assert ledger.active_call_count == 0


def test_failed_call_does_not_automatically_retry() -> None:
    ledger = CallLedger(authorized_execution())

    ledger.start_call(1)

    failed = ledger.fail_call(
        1,
        error="provider rejected call",
    )

    assert failed.status is CallStatus.FAILED

    with pytest.raises(
        CallLedgerError,
        match="not in planned state",
    ):
        ledger.start_call(1)
