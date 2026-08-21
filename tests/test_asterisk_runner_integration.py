from __future__ import annotations

from collections.abc import Callable, Iterator
from decimal import Decimal
from pathlib import Path
from uuid import UUID

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
    run_persistent_authorized_suite,
)
from voiceprobe.scenarios.catalog import list_scenarios
from voiceprobe.suite import build_suite_plan
from voiceprobe.telephony.ami import (
    AsteriskAMIConfig,
    OriginateResult,
)
from voiceprobe.telephony.asterisk_adapter import (
    AsteriskAssessmentCallAdapter,
    AsteriskMediaOutcome,
)

ORIGINATING_NUMBER = "+12025550101"

CALL_IDS = (
    UUID("11111111-2222-4333-8444-555555555555"),
    UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"),
    UUID("12345678-1234-4234-8234-123456789abc"),
)


def authorization(
    *,
    scenario_count: int,
):
    """Build the same real authorization object used by production runners."""
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
        execution_id="asterisk-runner-integration",
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
    policy = BudgetPolicy(
        total_budget_usd=Decimal(total),
        max_provider_rate_per_minute_usd=Decimal(rate),
    )

    calls = PersistentCallLedger.initialize(
        auth,
        path=tmp_path / "calls.json",
    )

    budget = PersistentBudgetLedger.initialize(
        execution_id=auth.manifest.execution_id,
        policy=policy,
        path=tmp_path / "budget.json",
    )

    return calls, budget, policy


class FakeAMIClient:
    """No-network AMI implementation used through the real adapter."""

    def __init__(
        self,
        *,
        attempt: int,
        events: list[tuple[str, int]],
    ) -> None:
        self._attempt = attempt
        self._events = events

    def connect(self) -> str:
        self._events.append(("ami_connect", self._attempt))
        return "Asterisk Call Manager/5.0"

    def login(
        self,
        *,
        events: str = "off",
    ) -> None:
        assert events == "call"

        self._events.append(("ami_login", self._attempt))

    def originate_audiosocket(
        self,
        destination: str,
        *,
        call_id: UUID | None = None,
        timeout_ms: int = 30_000,
    ) -> OriginateResult:
        assert destination == "+12025550100"
        assert call_id is not None
        assert timeout_ms == 30_000

        self._events.append(("ami_originate", self._attempt))

        return OriginateResult(
            action_id=f"action-{self._attempt}",
            audiosocket_call_id=call_id,
            asterisk_unique_id=(f"asterisk-{self._attempt}.001"),
            channel=(f"Local/{destination}@voiceprobe-test"),
            response="Success",
            reason=None,
        )

    def close(self) -> None:
        self._events.append(("ami_close", self._attempt))


def build_adapter(
    *,
    events: list[tuple[str, int]],
    call_ids: Iterator[UUID],
    media_executor: Callable[
        [
            AssessmentCallRequest,
            UUID,
            Callable[[], OriginateResult],
        ],
        AsteriskMediaOutcome,
    ],
) -> AsteriskAssessmentCallAdapter:
    client_count = 0

    def ami_factory(
        config: AsteriskAMIConfig,
    ) -> FakeAMIClient:
        nonlocal client_count

        # The fake still receives the same restricted-AMΙ config object.
        assert config.host == "127.0.0.1"

        client_count += 1

        return FakeAMIClient(
            attempt=client_count,
            events=events,
        )

    return AsteriskAssessmentCallAdapter(
        ami_config=AsteriskAMIConfig(
            username="voiceprobe",
            secret="synthetic-integration-secret",
        ),
        expected_originating_number=(ORIGINATING_NUMBER),
        ami_client_factory=ami_factory,
        call_id_factory=lambda: next(call_ids),
        media_executor=media_executor,
    )


def successful_media_executor(
    events: list[tuple[str, int]],
) -> Callable[
    [
        AssessmentCallRequest,
        UUID,
        Callable[[], OriginateResult],
    ],
    AsteriskMediaOutcome,
]:
    def execute(
        request: AssessmentCallRequest,
        call_id: UUID,
        originate: Callable[
            [],
            OriginateResult,
        ],
    ) -> AsteriskMediaOutcome:
        # Represents the point at which the real executor has established
        # its AudioSocket listener and is finally allowed to originate.
        events.append(("media_ready", request.position))

        originate_result = originate()

        events.append(("media_complete", request.position))

        return AsteriskMediaOutcome(
            call_id=call_id,
            artifact_run_id=(f"artifact-{request.position}"),
            duration_seconds=(40.0 + request.position),
            originate=originate_result,
        )

    return execute


def test_persistent_runner_records_real_asterisk_adapter_evidence(
    tmp_path: Path,
) -> None:
    auth = authorization(scenario_count=1)

    calls, budget, budget_policy = ledgers(
        tmp_path,
        auth,
    )

    events: list[tuple[str, int]] = []

    adapter = build_adapter(
        events=events,
        call_ids=iter(CALL_IDS),
        media_executor=successful_media_executor(events),
    )

    result = run_persistent_authorized_suite(
        auth,
        adapter,
        call_ledger=calls,
        budget_ledger=budget,
    )

    assert result.completed_count == 1
    assert result.failed_count == 0

    entry = result.entries[0]

    assert entry.status is CallStatus.COMPLETED
    assert entry.provider_call_id == ("asterisk-1.001")
    assert entry.artifact_run_id == ("artifact-1")
    assert entry.duration_seconds == 41.0

    # The Asterisk adapter does not invent provider billing data.
    # Until actual cost evidence exists, the conservative reservation remains.
    assert budget.entries[0].reserved_usd == (Decimal("0.30"))
    assert budget.entries[0].actual_usd is None
    assert budget.committed_usd == (Decimal("0.30"))

    assert events == [
        ("media_ready", 1),
        ("ami_connect", 1),
        ("ami_login", 1),
        ("ami_originate", 1),
        ("media_complete", 1),
        ("ami_close", 1),
    ]

    # Prove the evidence survives process restart instead of existing only
    # in the in-memory runner objects.
    recovered_calls = PersistentCallLedger.load(
        auth,
        path=tmp_path / "calls.json",
    )

    recovered_budget = PersistentBudgetLedger.load(
        execution_id=(auth.manifest.execution_id),
        policy=budget_policy,
        path=tmp_path / "budget.json",
    )

    recovered_entry = recovered_calls.entries[0]

    assert recovered_entry.status is CallStatus.COMPLETED
    assert recovered_entry.provider_call_id == ("asterisk-1.001")
    assert recovered_entry.artifact_run_id == ("artifact-1")
    assert recovered_entry.duration_seconds == (41.0)

    assert recovered_budget.committed_usd == (Decimal("0.30"))


def test_failed_asterisk_attempt_is_not_retried_and_next_scenario_runs(
    tmp_path: Path,
) -> None:
    auth = authorization(scenario_count=2)

    calls, budget, _ = ledgers(
        tmp_path,
        auth,
    )

    events: list[tuple[str, int]] = []

    def media_executor(
        request: AssessmentCallRequest,
        call_id: UUID,
        originate: Callable[
            [],
            OriginateResult,
        ],
    ) -> AsteriskMediaOutcome:
        events.append(("media_ready", request.position))

        originate_result = originate()

        if request.position == 1:
            events.append(("media_failed", request.position))

            raise RuntimeError("synthetic media failure")

        events.append(("media_complete", request.position))

        return AsteriskMediaOutcome(
            call_id=call_id,
            artifact_run_id=(f"artifact-{request.position}"),
            duration_seconds=42.0,
            originate=originate_result,
        )

    adapter = build_adapter(
        events=events,
        call_ids=iter(CALL_IDS),
        media_executor=media_executor,
    )

    result = run_persistent_authorized_suite(
        auth,
        adapter,
        call_ledger=calls,
        budget_ledger=budget,
    )

    assert result.failed_count == 1
    assert result.completed_count == 1

    assert [entry.status for entry in result.entries] == [
        CallStatus.FAILED,
        CallStatus.COMPLETED,
    ]

    assert result.entries[0].error == "RuntimeError: synthetic media failure"

    assert result.entries[1].provider_call_id == "asterisk-2.001"

    # There are two scenarios and exactly two originates. If call one had
    # silently retried, this list would contain a third attempt.
    originate_attempts = [
        attempt for event, attempt in events if event == "ami_originate"
    ]

    assert originate_attempts == [
        1,
        2,
    ]

    # Failed calls keep their conservative reservation because final
    # provider cost is unknown. The successful Asterisk result also carries
    # no invented billing value, so both remain reserved.
    assert budget.committed_usd == (Decimal("0.60"))


def test_budget_blocks_real_adapter_before_any_ami_or_media_side_effect(
    tmp_path: Path,
) -> None:
    auth = authorization(scenario_count=1)

    calls, budget, _ = ledgers(
        tmp_path,
        auth,
        total="0.20",
        rate="0.10",
    )

    events: list[tuple[str, int]] = []

    def forbidden_media_executor(
        request: AssessmentCallRequest,
        call_id: UUID,
        originate: Callable[
            [],
            OriginateResult,
        ],
    ) -> AsteriskMediaOutcome:
        del request, call_id, originate

        raise AssertionError("Adapter must not execute when budget reservation fails.")

    adapter = build_adapter(
        events=events,
        call_ids=iter(CALL_IDS),
        media_executor=forbidden_media_executor,
    )

    result = run_persistent_authorized_suite(
        auth,
        adapter,
        call_ledger=calls,
        budget_ledger=budget,
    )

    assert result.completed_count == 0
    assert result.failed_count == 0

    assert result.entries[0].status is (CallStatus.PLANNED)

    assert budget.entries == ()

    # Most important assertion: the production adapter boundary was never
    # crossed, so there was no fake AMI connect/originate or media activity.
    assert events == []
