from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

import pytest

from voiceprobe.runner import (
    AssessmentCallRequest,
    CallExecutionError,
)
from voiceprobe.safety import (
    ALLOWED_TEST_NUMBER,
    UnsafeDestinationError,
)
from voiceprobe.telephony.ami import (
    AsteriskAMIConfig,
    OriginateResult,
)
from voiceprobe.telephony.asterisk_adapter import (
    AsteriskAssessmentCallAdapter,
    AsteriskMediaOutcome,
)

CALLER = "+12025550101"
CALL_ID = UUID("11111111-2222-4333-8444-555555555555")


def make_request(
    *,
    destination: str = ALLOWED_TEST_NUMBER,
    originating_number: str = CALLER,
    max_duration_seconds: int = 180,
) -> AssessmentCallRequest:
    return AssessmentCallRequest(
        execution_id="execution-test",
        position=1,
        scenario_id="autonomous-phone-diagnostic",
        originating_number=originating_number,
        destination=destination,
        max_duration_seconds=max_duration_seconds,
    )


class FakeAMIClient:
    def __init__(
        self,
        events: list[str],
    ) -> None:
        self.events = events

    def connect(self) -> str:
        self.events.append("ami_connect")
        return "Asterisk Call Manager/5.0"

    def login(
        self,
        *,
        events: str = "off",
    ) -> None:
        self.events.append(f"ami_login:{events}")

    def originate_audiosocket(
        self,
        destination: str,
        *,
        call_id: UUID | None = None,
        timeout_ms: int = 30_000,
    ) -> OriginateResult:
        self.events.append("ami_originate")

        assert destination == ALLOWED_TEST_NUMBER
        assert call_id == CALL_ID
        assert timeout_ms == 30_000

        return OriginateResult(
            action_id="action-1",
            audiosocket_call_id=CALL_ID,
            asterisk_unique_id="asterisk-123.456",
            channel=(f"Local/{destination}@voiceprobe-test"),
            response="Success",
            reason="4",
        )

    def close(self) -> None:
        self.events.append("ami_close")


def build_adapter(
    events: list[str],
    media_executor: Callable[
        [
            AssessmentCallRequest,
            UUID,
            Callable[[], OriginateResult],
        ],
        AsteriskMediaOutcome,
    ],
) -> AsteriskAssessmentCallAdapter:
    fake_client = FakeAMIClient(events)

    return AsteriskAssessmentCallAdapter(
        ami_config=AsteriskAMIConfig(
            username="voiceprobe",
            secret="synthetic-test-secret",
        ),
        expected_originating_number=CALLER,
        ami_client_factory=(lambda config: fake_client),
        call_id_factory=(lambda: CALL_ID),
        media_executor=media_executor,
    )


def successful_media_executor(
    events: list[str],
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
        assert request.scenario_id == "autonomous-phone-diagnostic"
        assert call_id == CALL_ID

        # The production executor must have its listener ready before
        # invoking the one allowed originate callback.
        events.append("media_listening")

        originate_result = originate()

        events.append("media_complete")

        return AsteriskMediaOutcome(
            call_id=call_id,
            artifact_run_id="run-test-1",
            duration_seconds=84.25,
            originate=originate_result,
        )

    return execute


def test_successful_call_maps_asterisk_and_artifact_evidence() -> None:
    events: list[str] = []

    adapter = build_adapter(
        events,
        successful_media_executor(events),
    )

    result = adapter.execute_call(make_request())

    assert result.provider_call_id == ("asterisk-123.456")
    assert result.artifact_run_id == ("run-test-1")
    assert result.duration_seconds == (84.25)
    assert result.provider_cost_usd is None

    assert events == [
        "media_listening",
        "ami_connect",
        "ami_login:call",
        "ami_originate",
        "media_complete",
        "ami_close",
    ]


def test_unsafe_destination_is_rejected_before_any_side_effect() -> None:
    events: list[str] = []

    adapter = build_adapter(
        events,
        successful_media_executor(events),
    )

    with pytest.raises(UnsafeDestinationError):
        adapter.execute_call(make_request(destination="+12025550101"))

    assert events == []


def test_originating_number_mismatch_is_rejected_before_any_side_effect() -> None:
    events: list[str] = []

    adapter = build_adapter(
        events,
        successful_media_executor(events),
    )

    with pytest.raises(
        CallExecutionError,
        match="originating number",
    ):
        adapter.execute_call(make_request(originating_number=("+12025550102")))

    assert events == []


def test_invalid_max_duration_is_rejected_before_any_side_effect() -> None:
    events: list[str] = []

    adapter = build_adapter(
        events,
        successful_media_executor(events),
    )

    with pytest.raises(
        CallExecutionError,
        match="duration",
    ):
        adapter.execute_call(make_request(max_duration_seconds=601))

    assert events == []



def test_five_minute_duration_is_accepted_by_adapter() -> None:
    events: list[str] = []

    adapter = build_adapter(
        events,
        successful_media_executor(events),
    )

    result = adapter.execute_call(
        make_request(
            max_duration_seconds=300,
        )
    )

    assert result.duration_seconds == 84.25
    assert events.count("ami_originate") == 1

def test_media_call_id_mismatch_is_rejected() -> None:
    events: list[str] = []

    different_call_id = UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")

    def mismatched_media(
        request: AssessmentCallRequest,
        call_id: UUID,
        originate: Callable[
            [],
            OriginateResult,
        ],
    ) -> AsteriskMediaOutcome:
        del request

        events.append("media_listening")

        originate_result = originate()

        return AsteriskMediaOutcome(
            call_id=different_call_id,
            artifact_run_id="run-test-1",
            duration_seconds=2.0,
            originate=originate_result,
        )

    adapter = build_adapter(
        events,
        mismatched_media,
    )

    with pytest.raises(
        CallExecutionError,
        match="call ID",
    ):
        adapter.execute_call(make_request())

    assert events.count("ami_originate") == 1
