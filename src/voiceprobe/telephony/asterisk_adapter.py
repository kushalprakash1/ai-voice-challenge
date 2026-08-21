"""Production Asterisk execution adapter for authorized calls.

The generic suite runner owns authorization, sequencing, persistence, and
budget state. This module owns exactly one telephony attempt after the runner
has authorized it.

No destination normalization occurs here. The strict destination-number safety
boundary is revalidated immediately before any AMI or socket side effect.
"""

from __future__ import annotations

import os
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

import httpx
from kokoro import KPipeline

from voiceprobe.agents.brain import PatientBrain
from voiceprobe.artifacts.recorder import RunArtifactRecorder
from voiceprobe.autonomous_phone import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_PORT,
    DEFAULT_VOICE,
    build_pre_rendered_tts_cache,
    handle_call,
    terminate_audiosocket_connection,
)
from voiceprobe.conversation.session import PatientSession
from voiceprobe.interpreters.ollama import OllamaConversationInterpreter
from voiceprobe.reasoning.session_v2 import (
    ReasoningV2PatientSession,
    reasoning_v2_enabled_from_environment,
)
from voiceprobe.policy import CallPolicy
from voiceprobe.policy import MAX_CALL_DURATION_SECONDS
from voiceprobe.runner import (
    AssessmentCallRequest,
    AssessmentCallResult,
    CallExecutionError,
)
from voiceprobe.safety import validate_destination
from voiceprobe.scenarios.catalog import get_scenario
from voiceprobe.telephony.ami import (
    AMIClientError,
    AMIConnectionStateError,
    AsteriskAMIClient,
    AsteriskAMIConfig,
    AsteriskHangupResult,
    OriginateResult,
)
from voiceprobe.verbalizers.deterministic import DeterministicNaturalVerbalizer

VOICEPROBE_V3_LIVE_ENV = "VOICEPROBE_V3_LIVE"


def v3_live_enabled_from_environment() -> bool:
    """Return whether the explicit v3 live-media path is enabled."""

    raw = os.environ.get(VOICEPROBE_V3_LIVE_ENV, "").strip().casefold()

    if raw in {"", "0", "false", "no", "off"}:
        return False

    if raw in {"1", "true", "yes", "on"}:
        return True

    raise ValueError(
        f"{VOICEPROBE_V3_LIVE_ENV} must be one of "
        "0/1, false/true, no/yes, or off/on"
    )


DEFAULT_ACCEPT_TIMEOUT_SECONDS = 10.0
DEFAULT_ORIGINATE_TIMEOUT_MS = 30_000


class AsteriskTerminationStatus(StrEnum):
    """Why the media session stopped from VoiceProbe's perspective."""

    NORMAL_COMPLETION = "normal_completion"
    PREMATURE_REMOTE_TERMINATION = "premature_remote_termination"
    MAX_DURATION_TERMINATION = "max_duration_termination"


def _classify_termination(
    *,
    objective_complete: bool,
    max_duration_reached: bool,
) -> AsteriskTerminationStatus:
    """Classify termination without treating transport closure as success."""
    if objective_complete:
        return AsteriskTerminationStatus.NORMAL_COMPLETION

    if max_duration_reached:
        return AsteriskTerminationStatus.MAX_DURATION_TERMINATION

    # The autonomous call path only requests normal local termination after
    # PatientBrain returns END_CONVERSATION. The brain does not permit that
    # before objective completion. Therefore an incomplete, non-deadline
    # AudioSocket termination is not a successful local completion.
    return AsteriskTerminationStatus.PREMATURE_REMOTE_TERMINATION


def _termination_failure_reason(
    *,
    status: AsteriskTerminationStatus,
    booking_confirmed: bool,
    offer_accepted: bool,
    offered_day: str | None,
    offered_time: str | None,
) -> str | None:
    """Build durable evidence explaining an incomplete call objective."""
    if status is AsteriskTerminationStatus.NORMAL_COMPLETION:
        return None

    if status is AsteriskTerminationStatus.MAX_DURATION_TERMINATION:
        prefix = (
            "max_duration_termination: call reached VoiceProbe's maximum "
            "duration before the scheduling objective completed"
        )
    else:
        prefix = (
            "premature_remote_termination: call ended before the scheduling "
            "objective completed and VoiceProbe had not requested normal "
            "objective-complete termination"
        )

    return (
        f"{prefix}; "
        f"booking_confirmed={booking_confirmed}; "
        f"offer_accepted={offer_accepted}; "
        f"offered_day={offered_day!r}; "
        f"offered_time={offered_time!r}"
    )


@dataclass(frozen=True, slots=True)
class AsteriskMediaOutcome:
    """Evidence produced by one complete local AudioSocket media session."""

    call_id: UUID
    artifact_run_id: str
    duration_seconds: float
    originate: OriginateResult
    hangup: AsteriskHangupResult | None = None

    # Defaults preserve compatibility for injected media executors that
    # predate objective-aware termination classification.
    termination_status: AsteriskTerminationStatus = (
        AsteriskTerminationStatus.NORMAL_COMPLETION
    )
    objective_complete: bool = True
    booking_confirmed: bool = True
    offer_accepted: bool = True
    offered_day: str | None = None
    offered_time: str | None = None
    failure_reason: str | None = None


class _AMIClient(Protocol):
    """Small AMI surface required by the production adapter."""

    def connect(self) -> str:
        """Connect and validate the AMI banner."""
        ...

    def login(
        self,
        *,
        events: str = "off",
    ) -> None:
        """Authenticate the restricted local AMI user."""
        ...

    def originate_audiosocket(
        self,
        destination: str,
        *,
        call_id: UUID | None = None,
        timeout_ms: int = DEFAULT_ORIGINATE_TIMEOUT_MS,
    ) -> OriginateResult:
        """Originate exactly one AudioSocket call."""
        ...

    def wait_for_hangup(
        self,
        *,
        unique_id: str,
        channel: str,
        max_events: int = 2000,
    ) -> AsteriskHangupResult:
        """Wait for the correlated Hangup event."""
        ...

    def close(self) -> None:
        """Close the AMI transport."""
        ...


class _MediaExecutor(Protocol):
    """Injectable one-call media boundary used for deterministic testing."""

    def __call__(
        self,
        request: AssessmentCallRequest,
        call_id: UUID,
        originate: Callable[[], OriginateResult],
    ) -> AsteriskMediaOutcome:
        """Execute one listening media session around one originate."""
        ...


_AMIClientFactory = Callable[[AsteriskAMIConfig], _AMIClient]
_CallIDFactory = Callable[[], UUID]


class _MonitoredOriginate:
    """One-shot originate callback retaining AMI for Hangup observation."""

    def __init__(
        self,
        *,
        ami_config: AsteriskAMIConfig,
        ami_client_factory: _AMIClientFactory,
        destination: str,
        call_id: UUID,
    ) -> None:
        self._ami_config = ami_config
        self._ami_client_factory = ami_client_factory
        self._destination = destination
        self._call_id = call_id
        self._client: _AMIClient | None = None
        self._result: OriginateResult | None = None
        self._invoked = False

    def __call__(self) -> OriginateResult:
        """Originate once while retaining the authenticated AMI connection."""
        if self._invoked:
            raise CallExecutionError(
                "Assessment originate callback may be invoked only once."
            )

        self._invoked = True

        client = self._ami_client_factory(self._ami_config)
        self._client = client

        try:
            client.connect()
            client.login(events="call")

            result = client.originate_audiosocket(
                self._destination,
                call_id=self._call_id,
                timeout_ms=DEFAULT_ORIGINATE_TIMEOUT_MS,
            )
        except Exception:
            self.close()
            raise

        self._result = result

        return result

    def wait_for_hangup(
        self,
    ) -> AsteriskHangupResult:
        """Read the Hangup event for the originated Local channel."""
        client = self._client
        result = self._result

        if client is None or result is None:
            raise AMIConnectionStateError(
                "Cannot observe Hangup before a successful originate."
            )

        return client.wait_for_hangup(
            unique_id=result.asterisk_unique_id,
            channel=result.channel,
        )

    def close(self) -> None:
        """Close retained AMI without requesting a channel hangup."""
        client = self._client
        self._client = None

        if client is not None:
            client.close()


def _default_ami_client_factory(
    config: AsteriskAMIConfig,
) -> _AMIClient:
    return AsteriskAMIClient(config)


class AsteriskAssessmentCallAdapter:
    """Execute one already-authorized call through Asterisk.

    The adapter deliberately has no retry loop. One execute_call invocation
    maps to at most one AMI Originate operation.
    """

    def __init__(
        self,
        *,
        ami_config: AsteriskAMIConfig,
        expected_originating_number: str,
        artifact_root: Path | str = "artifacts/runs",
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        voice: str = DEFAULT_VOICE,
        accept_timeout_seconds: float = DEFAULT_ACCEPT_TIMEOUT_SECONDS,
        ami_client_factory: _AMIClientFactory = _default_ami_client_factory,
        call_id_factory: _CallIDFactory = uuid4,
        media_executor: _MediaExecutor | None = None,
    ) -> None:
        # Reuse CallPolicy's E.164/origin restrictions without weakening them.
        CallPolicy(
            originating_number=expected_originating_number,
            dry_run=False,
        )

        if host != DEFAULT_HOST:
            raise ValueError(
                "Production AudioSocket listener must remain on 127.0.0.1."
            )

        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65_535
        ):
            raise ValueError("AudioSocket port must be between 1 and 65535.")

        if (
            isinstance(accept_timeout_seconds, bool)
            or not isinstance(
                accept_timeout_seconds,
                (int, float),
            )
            or accept_timeout_seconds <= 0
        ):
            raise ValueError("AudioSocket accept timeout must be greater than zero.")

        self._ami_config = ami_config
        self._expected_originating_number = expected_originating_number
        self._artifact_root = Path(artifact_root)
        self._host = host
        self._port = port
        self._model = model
        self._ollama_url = ollama_url
        self._voice = voice
        self._accept_timeout_seconds = float(accept_timeout_seconds)
        self._ami_client_factory = ami_client_factory
        self._call_id_factory = call_id_factory

        self._pipeline: KPipeline | None = None
        self._tts_pcm_cache: dict[str, bytes] | None = None
        self._http_client: httpx.Client | None = None

        self._media_executor: _MediaExecutor

        if media_executor is not None:
            # Explicit test/injected media always wins over environment selection.
            self._media_executor = media_executor
        elif v3_live_enabled_from_environment():
            self._media_executor = self._execute_v3_media_call
        else:
            self._media_executor = self._execute_media_call

    def execute_call(
        self,
        request: AssessmentCallRequest,
    ) -> AssessmentCallResult:
        """Execute exactly one request after repeating every critical guard."""
        # Last adapter-level safety boundary before any resource that can dial.
        validate_destination(request.destination)

        if request.originating_number != self._expected_originating_number:
            raise CallExecutionError(
                "Assessment request originating number does not match "
                "the Asterisk adapter's configured caller identity."
            )

        if (
            isinstance(request.position, bool)
            or not isinstance(request.position, int)
            or request.position < 1
        ):
            raise CallExecutionError(
                "Assessment call position must be a positive integer."
            )

        if (
            isinstance(request.max_duration_seconds, bool)
            or not isinstance(
                request.max_duration_seconds,
                int,
            )
            or not 1 <= request.max_duration_seconds <= MAX_CALL_DURATION_SECONDS
        ):
            raise CallExecutionError(
                f"Assessment call duration must be between 1 and {MAX_CALL_DURATION_SECONDS} seconds."
            )

        # Resolve the scenario before any AMI side effect.
        get_scenario(request.scenario_id)

        call_id = self._call_id_factory()

        if not isinstance(call_id, UUID):
            raise TypeError("Asterisk adapter call_id_factory must return UUID.")

        originate = _MonitoredOriginate(
            ami_config=self._ami_config,
            ami_client_factory=self._ami_client_factory,
            destination=request.destination,
            call_id=call_id,
        )

        try:
            outcome = self._media_executor(
                request,
                call_id,
                originate,
            )
        finally:
            # Retain AMI throughout media execution so the correlated Hangup
            # event remains observable. Closing AMI itself does not request
            # a call hangup.
            originate.close()

        if outcome.call_id != call_id:
            raise CallExecutionError(
                "AudioSocket call ID did not match the authorized attempt."
            )

        if outcome.originate.audiosocket_call_id != call_id:
            raise CallExecutionError(
                "AMI originate result did not match the authorized call ID."
            )

        artifact_run_id = outcome.artifact_run_id.strip()

        if not artifact_run_id:
            raise CallExecutionError(
                "Asterisk media execution returned an empty artifact run ID."
            )

        provider_call_id = outcome.originate.asterisk_unique_id.strip()

        if not provider_call_id:
            raise CallExecutionError("Asterisk originate returned an empty Uniqueid.")

        if outcome.duration_seconds < 0:
            raise CallExecutionError(
                "Asterisk media execution returned a negative duration."
            )

        return AssessmentCallResult(
            # The runner calls this field provider_call_id. For the current
            # Asterisk transport, Asterisk Uniqueid is the strongest correlated
            # external call-control identifier available at this boundary.
            provider_call_id=provider_call_id,
            artifact_run_id=artifact_run_id,
            duration_seconds=outcome.duration_seconds,
            provider_cost_usd=None,
            assessment_succeeded=outcome.objective_complete,
            failure_reason=outcome.failure_reason,
        )

    def _execute_v3_media_call(
        self,
        request: AssessmentCallRequest,
        call_id: UUID,
        originate: Callable[[], OriginateResult],
    ) -> AsteriskMediaOutcome:
        """Execute the explicit v3 Pipecat/Flux live-media path."""

        deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()

        if not deepgram_api_key:
            # Fail before building the listener or invoking the originate callback.
            raise CallExecutionError(
                "VOICEPROBE_V3_LIVE=1 requires DEEPGRAM_API_KEY before dialing."
            )

        from voiceprobe.v3.asterisk_live import execute_v3_asterisk_media

        pipeline, _ = self._ensure_runtime()
        hangup_observer = (
            originate.wait_for_hangup
            if isinstance(originate, _MonitoredOriginate)
            else None
        )

        result = execute_v3_asterisk_media(
            request=request,
            call_id=call_id,
            originate=originate,
            pipeline=pipeline,
            voice=self._voice,
            tts_pcm_cache=self._tts_pcm_cache,
            deepgram_api_key=deepgram_api_key,
            artifact_root=self._artifact_root,
            host=self._host,
            port=self._port,
            accept_timeout_seconds=self._accept_timeout_seconds,
            hangup_observer=hangup_observer,
            ami_error_type=AMIClientError,
            classify_termination=_classify_termination,
            termination_failure_reason=_termination_failure_reason,
        )

        return AsteriskMediaOutcome(
            call_id=result.call_id,
            artifact_run_id=result.artifact_run_id,
            duration_seconds=result.duration_seconds,
            originate=result.originate,
            hangup=result.hangup,
            termination_status=result.termination_status,
            objective_complete=result.objective_complete,
            booking_confirmed=result.booking_confirmed,
            offer_accepted=result.offer_accepted,
            offered_day=result.offered_day,
            offered_time=result.offered_time,
            failure_reason=result.failure_reason,
        )

    def _execute_media_call(
        self,
        request: AssessmentCallRequest,
        call_id: UUID,
        originate: Callable[[], OriginateResult],
    ) -> AsteriskMediaOutcome:
        """Listen first, originate second, then own one AudioSocket session."""
        pipeline, http_client = self._ensure_runtime()

        scenario = get_scenario(request.scenario_id)

        if reasoning_v2_enabled_from_environment():
            session = ReasoningV2PatientSession(
                scenario=scenario,
                model=self._model,
                url=self._ollama_url,
                client=http_client,
            )
        else:
            # Preserve the existing production behavior exactly when
            # VOICEPROBE_REASONING_V2 is unset or zero.
            interpreter = OllamaConversationInterpreter(
                model=self._model,
                url=self._ollama_url,
                client=http_client,
            )

            verbalizer = DeterministicNaturalVerbalizer(
                model=self._model,
                url=self._ollama_url,
                client=http_client,
            )

            session = PatientSession(
                scenario=scenario,
                interpreter=interpreter,
                verbalizer=verbalizer,
                brain=PatientBrain(),
            )

        try:
            with socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM,
            ) as server:
                server.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_REUSEADDR,
                    1,
                )

                server.bind(
                    (
                        self._host,
                        self._port,
                    )
                )
                server.listen(1)
                server.settimeout(self._accept_timeout_seconds)

                # Critical ordering:
                # AudioSocket must already be listening before AMI Originate.
                originate_result = originate()

                try:
                    connection, address = server.accept()
                except TimeoutError as error:
                    raise CallExecutionError(
                        "Asterisk originated the call but did not connect "
                        "to the local AudioSocket listener in time."
                    ) from error

                with RunArtifactRecorder(
                    root=self._artifact_root,
                    scenario=scenario,
                ) as recorder:
                    recorder.record_event(
                        "suite_adapter_call_started",
                        execution_id=request.execution_id,
                        position=request.position,
                        call_id=str(call_id),
                        asterisk_unique_id=(originate_result.asterisk_unique_id),
                        address=address,
                    )

                    call_finished = threading.Event()
                    max_duration_reached = threading.Event()

                    def enforce_max_duration() -> None:
                        expired = not call_finished.wait(request.max_duration_seconds)

                        if not expired:
                            return

                        max_duration_reached.set()

                        recorder.record_event(
                            "max_call_duration_reached",
                            max_duration_seconds=(request.max_duration_seconds),
                        )

                        terminate_audiosocket_connection(connection)

                    watchdog = threading.Thread(
                        target=enforce_max_duration,
                        name=(f"voiceprobe-call-deadline-{request.position}"),
                        daemon=True,
                    )

                    watchdog.start()

                    try:
                        with connection:
                            observed_call_id = handle_call(
                                connection=connection,
                                session=session,
                                pipeline=pipeline,
                                voice=self._voice,
                                recorder=recorder,
                                tts_pcm_cache=self._tts_pcm_cache,
                            )
                    finally:
                        call_finished.set()
                        watchdog.join(timeout=1.0)

                    if observed_call_id is None:
                        raise CallExecutionError(
                            "AudioSocket session ended without a call UUID."
                        )

                    if observed_call_id != call_id:
                        raise CallExecutionError(
                            "AudioSocket UUID did not match the originated call."
                        )

                    hangup_result: AsteriskHangupResult | None = None

                    if isinstance(
                        originate,
                        _MonitoredOriginate,
                    ):
                        try:
                            hangup_result = originate.wait_for_hangup()
                        except AMIClientError as error:
                            recorder.record_event(
                                "asterisk_hangup_observer_error",
                                asterisk_unique_id=(
                                    originate_result.asterisk_unique_id
                                ),
                                channel=(originate_result.channel),
                                error_type=(type(error).__name__),
                                error_message=str(error),
                            )
                        else:
                            recorder.record_event(
                                "asterisk_hangup_observed",
                                asterisk_unique_id=(hangup_result.unique_id),
                                channel=hangup_result.channel,
                                linked_id=(hangup_result.linked_id),
                                cause=hangup_result.cause,
                                cause_text=(hangup_result.cause_text),
                                tech_cause=(hangup_result.tech_cause),
                            )
                    else:
                        recorder.record_event(
                            "asterisk_hangup_observer_unavailable",
                            asterisk_unique_id=(originate_result.asterisk_unique_id),
                            channel=originate_result.channel,
                        )

                    progress = session.progress

                    termination_status = _classify_termination(
                        objective_complete=progress.objective_complete,
                        max_duration_reached=max_duration_reached.is_set(),
                    )

                    failure_reason = _termination_failure_reason(
                        status=termination_status,
                        booking_confirmed=progress.booking_confirmed,
                        offer_accepted=progress.offer_accepted,
                        offered_day=progress.offered_day,
                        offered_time=progress.offered_time,
                    )

                    recorder.record_event(
                        "call_termination_classified",
                        termination_status=termination_status.value,
                        objective_complete=progress.objective_complete,
                        booking_confirmed=progress.booking_confirmed,
                        offer_accepted=progress.offer_accepted,
                        offered_day=progress.offered_day,
                        offered_time=progress.offered_time,
                        max_duration_reached=max_duration_reached.is_set(),
                        asterisk_hangup_observed=(hangup_result is not None),
                        asterisk_hangup_cause=(
                            hangup_result.cause
                            if hangup_result is not None
                            else None
                        ),
                        asterisk_hangup_cause_text=(
                            hangup_result.cause_text
                            if hangup_result is not None
                            else None
                        ),
                    )

                    duration_seconds = recorder.elapsed_seconds

                    artifact_status = (
                        "completed"
                        if progress.objective_complete
                        else termination_status.value
                    )

                    recorder.finalize(
                        status=artifact_status,
                        call_id=str(observed_call_id),
                        error=failure_reason,
                    )

                    return AsteriskMediaOutcome(
                        call_id=observed_call_id,
                        artifact_run_id=recorder.run_id,
                        duration_seconds=duration_seconds,
                        originate=originate_result,
                        hangup=hangup_result,
                        termination_status=termination_status,
                        objective_complete=progress.objective_complete,
                        booking_confirmed=progress.booking_confirmed,
                        offer_accepted=progress.offer_accepted,
                        offered_day=progress.offered_day,
                        offered_time=progress.offered_time,
                        failure_reason=failure_reason,
                    )
        finally:
            # Reasoning v2 owns and closes its semantic/planner components.
            # Legacy mode still owns the separate interpreter/verbalizer.
            if isinstance(
                session,
                ReasoningV2PatientSession,
            ):
                session.close()
            else:
                interpreter.close()
                verbalizer.close()

    def _ensure_runtime(
        self,
    ) -> tuple[
        KPipeline,
        httpx.Client,
    ]:
        """Lazily build expensive reusable runtime components before dialing."""
        if self._pipeline is None:
            pipeline = KPipeline(
                lang_code="a",
                repo_id="hexgrad/Kokoro-82M",
            )

            self._pipeline = pipeline

        if self._tts_pcm_cache is None:
            # Pre-render time-critical deterministic responses before the
            # listener can possibly originate a real call.
            self._tts_pcm_cache = build_pre_rendered_tts_cache(
                pipeline=self._pipeline,
                voice=self._voice,
            )

        if self._http_client is None:
            self._http_client = httpx.Client(
                timeout=20.0,
            )

        return (
            self._pipeline,
            self._http_client,
        )

    def close(self) -> None:
        """Release adapter-owned reusable network resources."""
        client = self._http_client
        self._http_client = None

        if client is not None:
            client.close()
