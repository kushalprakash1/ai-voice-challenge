from __future__ import annotations

from dataclasses import replace

from voiceprobe.execution import (
    LIVE_CONFIRMATION_TOKEN,
    CallStatus,
    authorize_live_execution,
    prepare_execution,
)
from voiceprobe.policy import CallPolicy
from voiceprobe.runner import (
    AssessmentCallRequest,
    AssessmentCallResult,
    CallExecutionError,
    run_authorized_suite,
)
from voiceprobe.safety import ALLOWED_TEST_NUMBER
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

    scenarios = list_scenarios()[:scenario_count]

    suite = build_suite_plan(
        policy,
        scenarios=scenarios,
    )

    manifest = prepare_execution(
        policy,
        suite,
        execution_id="runner-test",
    )

    return authorize_live_execution(
        manifest,
        live_requested=True,
        confirmation_token=(LIVE_CONFIRMATION_TOKEN),
    )


class FakeAdapter:
    def __init__(
        self,
        *,
        fail_positions: set[int] | None = None,
    ) -> None:
        self.fail_positions = set() if fail_positions is None else fail_positions
        self.requests: list[AssessmentCallRequest] = []
        self.attempts: dict[int, int] = {}
        self.active = 0
        self.max_active = 0

    def execute_call(
        self,
        request: AssessmentCallRequest,
    ) -> AssessmentCallResult:
        self.requests.append(request)

        self.attempts[request.position] = (
            self.attempts.get(
                request.position,
                0,
            )
            + 1
        )

        self.active += 1
        self.max_active = max(
            self.max_active,
            self.active,
        )

        try:
            if request.position in self.fail_positions:
                raise CallExecutionError("synthetic call failure")

            return AssessmentCallResult(
                provider_call_id=(f"provider-{request.position}"),
                artifact_run_id=(f"artifact-{request.position}"),
                duration_seconds=30.0,
            )

        finally:
            self.active -= 1


class MissingArtifactAdapter:
    def execute_call(
        self,
        request: AssessmentCallRequest,
    ) -> AssessmentCallResult:
        return AssessmentCallResult(
            provider_call_id=(f"provider-{request.position}"),
            artifact_run_id="",
            duration_seconds=20.0,
        )


class ExcessiveDurationAdapter:
    def execute_call(
        self,
        request: AssessmentCallRequest,
    ) -> AssessmentCallResult:
        return AssessmentCallResult(
            provider_call_id=(f"provider-{request.position}"),
            artifact_run_id=(f"artifact-{request.position}"),
            duration_seconds=(request.max_duration_seconds + 1),
        )


def test_runner_executes_scenarios_in_order() -> None:
    adapter = FakeAdapter()

    result = run_authorized_suite(
        authorization(),
        adapter,
    )

    assert [request.position for request in adapter.requests] == [1, 2, 3]

    assert result.completed_count == 3
    assert result.failed_count == 0


def test_runner_passes_exact_scenario_ids() -> None:
    adapter = FakeAdapter()
    auth = authorization()

    run_authorized_suite(
        auth,
        adapter,
    )

    assert (
        tuple(request.scenario_id for request in adapter.requests)
        == auth.manifest.scenario_ids
    )


def test_runner_passes_only_allowed_destination() -> None:
    adapter = FakeAdapter()

    run_authorized_suite(
        authorization(),
        adapter,
    )

    assert {request.destination for request in adapter.requests} == {
        ALLOWED_TEST_NUMBER,
    }


def test_runner_never_exceeds_concurrency_one() -> None:
    adapter = FakeAdapter()

    run_authorized_suite(
        authorization(),
        adapter,
    )

    assert adapter.max_active == 1


def test_failed_call_is_not_retried() -> None:
    adapter = FakeAdapter(
        fail_positions={2},
    )

    result = run_authorized_suite(
        authorization(),
        adapter,
    )

    assert adapter.attempts[2] == 1
    assert result.completed_count == 2
    assert result.failed_count == 1


def test_failure_does_not_prevent_next_planned_call() -> None:
    adapter = FakeAdapter(
        fail_positions={2},
    )

    run_authorized_suite(
        authorization(),
        adapter,
    )

    assert [request.position for request in adapter.requests] == [1, 2, 3]


def test_completed_calls_require_artifact_evidence() -> None:
    result = run_authorized_suite(
        authorization(
            scenario_count=1,
        ),
        MissingArtifactAdapter(),
    )

    assert result.completed_count == 0
    assert result.failed_count == 1
    assert result.entries[0].status is CallStatus.FAILED


def test_duration_cap_is_enforced_after_adapter_returns() -> None:
    result = run_authorized_suite(
        authorization(
            scenario_count=1,
        ),
        ExcessiveDurationAdapter(),
    )

    assert result.completed_count == 0
    assert result.failed_count == 1


def test_runner_rejects_tampered_concurrency() -> None:
    auth = authorization(
        scenario_count=1,
    )

    tampered_manifest = replace(
        auth.manifest,
        concurrency=2,
    )

    tampered_auth = replace(
        auth,
        manifest=tampered_manifest,
    )

    adapter = FakeAdapter()

    try:
        run_authorized_suite(
            tampered_auth,
            adapter,
        )
    except CallExecutionError as error:
        assert "concurrency" in str(error)
    else:
        raise AssertionError("Tampered concurrency was accepted.")

    assert adapter.requests == []
