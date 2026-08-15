"""Production Asterisk execution adapter for authorized assessment calls.

The generic suite runner owns authorization, sequencing, persistence, and
budget state. This module owns exactly one telephony attempt after the runner
has authorized it.

No destination normalization occurs here. The strict assessment-number safety
boundary is revalidated immediately before any AMI or socket side effect.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass
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
    handle_call,
    synthesize,
    terminate_audiosocket_connection,
)
from voiceprobe.conversation.session import PatientSession
from voiceprobe.interpreters.ollama import OllamaConversationInterpreter
from voiceprobe.policy import CallPolicy
from voiceprobe.runner import (
    AssessmentCallRequest,
    AssessmentCallResult,
    CallExecutionError,
)
from voiceprobe.safety import validate_destination
from voiceprobe.scenarios.catalog import get_scenario
from voiceprobe.telephony.ami import (
    AsteriskAMIClient,
    AsteriskAMIConfig,
    OriginateResult,
)
from voiceprobe.verbalizers.ollama import OllamaNaturalVerbalizer

DEFAULT_ACCEPT_TIMEOUT_SECONDS = 10.0
DEFAULT_ORIGINATE_TIMEOUT_MS = 30_000


@dataclass(frozen=True, slots=True)
class AsteriskMediaOutcome:
    """Evidence produced by one complete local AudioSocket media session."""

    call_id: UUID
    artifact_run_id: str
    duration_seconds: float
    originate: OriginateResult


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


def _default_ami_client_factory(
    config: AsteriskAMIConfig,
) -> _AMIClient:
    return AsteriskAMIClient(config)


class AsteriskAssessmentCallAdapter:
    """Execute one already-authorized assessment call through Asterisk.

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
        self._http_client: httpx.Client | None = None

        self._media_executor: _MediaExecutor = (
            media_executor if media_executor is not None else self._execute_media_call
        )

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
            or not 1 <= request.max_duration_seconds <= 180
        ):
            raise CallExecutionError(
                "Assessment call duration must be between 1 and 180 seconds."
            )

        # Resolve the scenario before any AMI side effect.
        get_scenario(request.scenario_id)

        call_id = self._call_id_factory()

        if not isinstance(call_id, UUID):
            raise TypeError("Asterisk adapter call_id_factory must return UUID.")

        def originate() -> OriginateResult:
            client = self._ami_client_factory(self._ami_config)

            try:
                client.connect()
                client.login(events="call")

                return client.originate_audiosocket(
                    request.destination,
                    call_id=call_id,
                    timeout_ms=DEFAULT_ORIGINATE_TIMEOUT_MS,
                )
            finally:
                # Closing the restricted localhost AMI transport does not
                # terminate an already-originated Asterisk channel.
                client.close()

        outcome = self._media_executor(
            request,
            call_id,
            originate,
        )

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

        interpreter = OllamaConversationInterpreter(
            model=self._model,
            url=self._ollama_url,
            client=http_client,
        )

        verbalizer = OllamaNaturalVerbalizer(
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

                    def enforce_max_duration() -> None:
                        expired = not call_finished.wait(request.max_duration_seconds)

                        if not expired:
                            return

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

                    duration_seconds = recorder.elapsed_seconds

                    recorder.finalize(
                        status="completed",
                        call_id=str(observed_call_id),
                    )

                    return AsteriskMediaOutcome(
                        call_id=observed_call_id,
                        artifact_run_id=recorder.run_id,
                        duration_seconds=duration_seconds,
                        originate=originate_result,
                    )
        finally:
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

            # Warm the selected voice before the listener can possibly dial.
            synthesize(
                pipeline=pipeline,
                voice=self._voice,
                text="Hello.",
            )

            self._pipeline = pipeline

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
