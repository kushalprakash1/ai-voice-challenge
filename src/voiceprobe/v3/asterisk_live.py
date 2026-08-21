"""Live Asterisk/AudioSocket production boundary for VoiceProbe v3.

This module is deliberately separate from the legacy/v2 media executor.  It
joins the already-tested v3 pieces only when the Asterisk adapter explicitly
selects v3 live mode:

Asterisk AudioSocket -> native 8 kHz PCM -> Pipecat PipelineWorker -> Flux
-> VoiceProbeV3Runtime -> Kokoro -> synchronized AudioSocket output.

Dialing remains owned by the Asterisk adapter.  This module receives the
one-shot originate callback only after its localhost AudioSocket listener is
already listening.
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from voiceprobe.artifacts.recorder import RunArtifactRecorder
from voiceprobe.autonomous_phone import terminate_audiosocket_connection
from voiceprobe.media.live_asr import (
    TYPE_DTMF,
    TYPE_HANGUP,
    TYPE_PCM_8KHZ,
    TYPE_UUID,
)
from voiceprobe.runner import AssessmentCallRequest, CallExecutionError
from voiceprobe.scenarios.catalog import get_scenario
from voiceprobe.telephony.ami import AsteriskHangupResult, OriginateResult

from .live_monitor import LiveAudioMonitor

from .audiosocket_kokoro import (
    AudioSocketKokoroConfig,
    AudioSocketKokoroSpeechTask,
    AudioSocketV3MediaBoundary,
    KokoroTelephonyRenderer,
)
from .background import (
    BACKGROUND_SEED,
    background_mode_from_environment,
    background_snr_from_environment,
    validate_background_asset,
)
from .audiosocket_pipecat import build_audiosocket_flux_input_worker
from .accent import (
    ACCENT_MELO_INDIA,
    ACCENT_CHATTERBOX_KOREAN_HEAVY,
    DEFAULT_CACHE_ROOT,
    AccentCache,
    accent_cache_preflight,
    warm_melo_india_renderer,
    accent_mode_from_environment,
)
from .flow_controller import SchedulingFlowController
from .personas import (
    PersonaDecisionOverlay,
    PersonaRuntime,
    persona_runtime_from_environment,
)
from .flow_state import FlowSnapshot
from .production import PipecatRuntimeBridge, build_production_flux_service


DEFAULT_FLUX_CONNECT_TIMEOUT_SECONDS = 10.0
SOCKET_POLL_SECONDS = 0.10
RUNNER_SHUTDOWN_TIMEOUT_SECONDS = 10.0


class _FluxReadinessGate:
    """One-shot, level-triggered pre-dial readiness for the live pipeline."""

    def __init__(self) -> None:
        self.ready = threading.Event()
        self.pipeline_started = threading.Event()
        self.flux_connected = threading.Event()
        self._lock = threading.Lock()
        self._ready_observations = 0

    @property
    def ready_observations(self) -> int:
        with self._lock:
            return self._ready_observations

    def mark_pipeline_started(self) -> bool:
        self.pipeline_started.set()
        return self._refresh()

    def mark_flux_connected(self) -> bool:
        self.flux_connected.set()
        return self._refresh()

    def _refresh(self) -> bool:
        with self._lock:
            if (
                self.pipeline_started.is_set()
                and self.flux_connected.is_set()
                and not self.ready.is_set()
            ):
                self._ready_observations += 1
                self.ready.set()
                return True
        return False


async def _observe_flux_connected_state(service: Any, gate: _FluxReadinessGate) -> bool:
    """Observe Flux's server-confirmed state without an edge-triggered race."""

    established = getattr(service, "_connection_established_event", None)
    wait = getattr(established, "wait", None)
    if callable(wait):
        await wait()
        return gate.mark_flux_connected()
    return False


@dataclass(frozen=True, slots=True)
class V3LegacyProgressProjection:
    """Conservative projection from v3 flow evidence into adapter fields."""

    objective_complete: bool
    booking_confirmed: bool
    offer_accepted: bool
    offered_day: str | None
    offered_time: str | None
    accepted_slot_text: str | None
    booking_confirmation_text: str | None
    scenario_id: str = "autonomous-phone-diagnostic"
    scenario_metadata: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class V3AsteriskMediaResult:
    call_id: UUID
    artifact_run_id: str
    duration_seconds: float
    originate: OriginateResult
    hangup: AsteriskHangupResult | None
    termination_status: Any
    objective_complete: bool
    booking_confirmed: bool
    offer_accepted: bool
    offered_day: str | None
    offered_time: str | None
    failure_reason: str | None


def project_v3_flow_snapshot(
    snapshot: FlowSnapshot,
    *,
    scenario_id: str = "autonomous-phone-diagnostic",
    scenario_metadata: Mapping[str, object] | None = None,
) -> V3LegacyProgressProjection:
    """Map only evidence that v3 actually owns; never invent day/time fields."""

    adversarial = scenario_id in {
        "medication-refill-correction",
        "office-hours-location-insurance",
        "self-pay-location-switch",
        "doctor-specialist-directory",
    }
    objective_complete = (
        bool((scenario_metadata or {}).get("objective_complete"))
        if adversarial
        else bool(snapshot.complete)
    )

    return V3LegacyProgressProjection(
        objective_complete=objective_complete,
        # v3 completion itself means explicit remote booking confirmation.
        booking_confirmed=(objective_complete if not adversarial else False),
        # The tracker records this only after an accepted/concretely confirmed slot.
        offer_accepted=(snapshot.accepted_slot_text is not None if not adversarial else False),
        # v3 intentionally carries the exact accepted/confirmation text instead
        # of pretending it has the legacy split day/time representation.
        offered_day=None,
        offered_time=None,
        accepted_slot_text=snapshot.accepted_slot_text,
        booking_confirmation_text=snapshot.booking_confirmation_text,
        scenario_id=scenario_id,
        scenario_metadata=scenario_metadata,
    )


def scenario_termination_failure_reason(
    *, status: Any, projection: V3LegacyProgressProjection
) -> str | None:
    """Describe incomplete objectives using facts owned by that scenario."""
    if getattr(status, "value", status) == "normal_completion":
        return None
    if projection.scenario_id not in {
        "medication-refill-correction", "office-hours-location-insurance",
        "self-pay-location-switch",
        "doctor-specialist-directory",
    }:
        return None
    label = getattr(status, "value", str(status))
    metadata = dict(projection.scenario_metadata or {})
    if metadata.get("experiment_status") == "target_capability_blocked":
        return (
            "target_capability_blocked: the intended experiment could not be "
            "exercised after grounded productive recovery paths were exhausted; "
            f"blocking_target_statement={metadata.get('blocking_target_statement')!r}; "
            f"human_escalation_offered={metadata.get('human_escalation_offered')!r}; "
            f"human_escalation_requested={metadata.get('human_escalation_requested')!r}"
        )
    keys = (
        (
            "scenario_stage", "prerequisite_action", "medication_provided",
            "target_old_dose_observed", "dose_correction_spoken",
            "dose_acknowledged", "pharmacy_handled", "objective_complete",
        )
        if projection.scenario_id == "medication-refill-correction"
        else (
            "insurance_exercised", "self_pay_established", "locations_discovered",
            "location_a", "location_b", "location_switch_exercised",
            "hours_queries_completed", "contextual_hours_completed",
            "oracle_candidates", "objective_complete",
        )
    )
    detail = "; ".join(f"{key}={metadata.get(key)!r}" for key in keys)
    return (
        f"{label}: call ended before the {projection.scenario_id} objective "
        f"completed; scenario={projection.scenario_id}; {detail}"
    )


class _LocalMediaStop(Exception):
    """Internal control-flow signal used to poll async v3 state from recv()."""


class _RecordingPipecatRuntimeBridge(PipecatRuntimeBridge):
    """Persist every v3 runtime decision into the existing run artifacts.

    Remote text is a Deepgram Flux EndOfTurn transcript. Patient text is
    recorded only after the response has been successfully queued to the
    AudioSocket/Kokoro speech sink. The later ``v3_audio_sent`` event remains
    the authoritative proof that PCM was actually handed to AudioSocket.
    """

    def __init__(
        self,
        *,
        recorder: RunArtifactRecorder,
        tts_frame_factory: Callable[[str], Any] | None = None,
        persona_runtime: PersonaRuntime | None = None,
    ) -> None:
        self._recorder = recorder
        self._decision_index = 0
        self._persona_runtime = persona_runtime
        self._persona_event_cursor = 0
        self._persona_final_recorded = False
        self._last_response_fingerprint: tuple[str, str] | None = None
        self._repeated_response_count = 0

        flow_controller = None

        if persona_runtime is not None:
            flow_controller = SchedulingFlowController(
                decision_overlay=PersonaDecisionOverlay(
                    persona_runtime
                )
            )

        super().__init__(
            tts_frame_factory=tts_frame_factory,
            flow_controller=flow_controller,
        )

        if persona_runtime is not None:
            self._recorder.record_event(
                "persona_configured",
                **persona_runtime.configuration(),
            )

    def _flush_persona_events(self) -> None:
        if self._persona_runtime is None:
            return

        events = self._persona_runtime.events

        for event in events[self._persona_event_cursor:]:
            self._recorder.record_event(
                event.event_type,
                persona_id=event.persona_id,
                sequence_id=event.sequence_id,
                move_number=event.move_number,
                remote_turn=event.remote_turn,
                output_text=event.output_text,
                flow_stage=event.flow_stage,
                state_effect=event.state_effect,
                persona_turn_index=event.turn_index,
            )

        self._persona_event_cursor = len(events)

    def record_persona_final_evidence(self) -> None:
        if (
            self._persona_runtime is None
            or self._persona_final_recorded
        ):
            return

        self._flush_persona_events()

        self._recorder.record_event(
            "persona_final_evidence",
            **self._persona_runtime.evidence(),
        )

        self._persona_final_recorded = True

    async def _on_runtime_decision(self, result) -> None:
        self._decision_index += 1
        decision_index = self._decision_index

        self._flush_persona_events()

        for ordinal, turn in enumerate(result.source_turns, start=1):
            self._recorder.record_transcript_turn(
                speaker="agent",
                text=turn,
                source="deepgram_flux_eot",
                v3_decision_index=decision_index,
                source_turn_ordinal=ordinal,
                ingress_reason=result.ingress_reason,
            )

        scenario_metadata = self.scenario_metadata
        route_owner = (
            "prerequisite"
            if result.decision.reason.startswith("prerequisite:")
            else (
                "medication"
                if result.decision.reason.startswith("medication_refill:")
                else (
                    "self_pay"
                    if result.decision.reason.startswith("self_pay_location:")
                    else result.route.value
                )
            )
        )
        self._recorder.record_event(
            "v3_runtime_decision",
            decision_index=decision_index,
            source_turns=list(result.source_turns),
            actionable_turn=result.actionable_turn,
            decision_kind=result.decision.kind.value,
            decision_text=result.decision.text,
            decision_reason=result.decision.reason,
            route=result.route.value,
            ingress_reason=result.ingress_reason,
            policy_latency_ms=round(result.policy_latency_ms, 6),
            requires_response=result.requires_response,
            response_ready=result.response_ready,
            before_stage=result.before.current_stage.value,
            after_stage=result.after.current_stage.value,
            objective_complete=result.after.complete,
            accepted_slot_text=result.after.accepted_slot_text,
            booking_confirmation_text=result.after.booking_confirmation_text,
            route_owner=route_owner,
            scenario_state=scenario_metadata,
        )

        if result.response_ready:
            fingerprint = (result.decision.reason, result.decision.text)
            if fingerprint == self._last_response_fingerprint:
                self._repeated_response_count += 1
            else:
                self._last_response_fingerprint = fingerprint
                self._repeated_response_count = 1
            if self._repeated_response_count >= 3:
                self._recorder.record_event(
                    "v3_suspicious_response_repetition",
                    repeated_count=self._repeated_response_count,
                    decision_reason=result.decision.reason,
                    decision_text=result.decision.text,
                    source_turns=list(result.source_turns),
                    scenario=self._selected_scenario or None,
                    assessment_result=(
                        "FAIL" if self._selected_scenario == "doctor-specialist-directory" else "observe"
                    ),
                )

        try:
            await super()._on_runtime_decision(result)
        except BaseException as error:
            self._recorder.record_event(
                "v3_runtime_decision_delivery_error",
                decision_index=decision_index,
                decision_kind=result.decision.kind.value,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            raise

        if result.response_ready:
            self._recorder.record_transcript_turn(
                speaker="patient",
                text=result.decision.text,
                source="voiceprobe_v3",
                delivery_status="queued_for_tts",
                v3_decision_index=decision_index,
                decision_kind=result.decision.kind.value,
                decision_reason=result.decision.reason,
                route=result.route.value,
            )

        self._recorder.record_turn_metrics(
            {
                "policy_latency_ms": round(result.policy_latency_ms, 6),
                "ingress_reason": result.ingress_reason,
                "route": result.route.value,
                "decision_kind": result.decision.kind.value,
                "decision_reason": result.decision.reason,
                "requires_response": result.requires_response,
                "response_ready": result.response_ready,
                "source_turn_count": len(result.source_turns),
                "objective_complete": result.after.complete,
                "current_stage": result.after.current_stage.value,
                "accepted_slot_text": result.after.accepted_slot_text,
                "booking_confirmation_text": (
                    result.after.booking_confirmation_text
                ),
            }
        )


class _AsyncV3Runtime:
    """Own one Pipecat WorkerRunner and expose thread-safe PCM submission."""

    def __init__(
        self,
        *,
        api_key: str,
        speech_task: AudioSocketKokoroSpeechTask,
        recorder: RunArtifactRecorder,
        persona_runtime: PersonaRuntime | None = None,
    ) -> None:
        self._api_key = api_key
        self._speech_task = speech_task
        self._recorder = recorder
        self._persona_runtime = persona_runtime

        self._readiness = _FluxReadinessGate()
        self.connected = self._readiness.ready
        self.pipeline_started = self._readiness.pipeline_started
        self.disconnected = threading.Event()
        self.objective_complete = threading.Event()
        self.scenario_terminal = threading.Event()
        self.error_event = threading.Event()
        self.stopped = threading.Event()
        self._stop_requested = threading.Event()

        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_async: asyncio.Event | None = None
        self._bundle: Any | None = None
        self._bridge: PipecatRuntimeBridge | None = None
        self._latest_snapshot: FlowSnapshot | None = None
        self._latest_scenario_metadata: dict[str, object] = {}
        self._error: BaseException | None = None

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def start(
        self,
        *,
        timeout: float = DEFAULT_FLUX_CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        if self._thread is not None:
            raise RuntimeError("VoiceProbe v3 WorkerRunner has already been started")

        self._thread = threading.Thread(
            target=self._thread_main,
            name="voiceprobe-v3-worker-runner",
            daemon=True,
        )
        self._thread.start()

        deadline = time.monotonic() + timeout

        while not self.connected.is_set():
            if self.error_event.is_set():
                error = self.error
                raise CallExecutionError(
                    "VoiceProbe v3 WorkerRunner/Flux failed before becoming ready: "
                    f"{type(error).__name__ if error is not None else 'unknown'}: "
                    f"{error if error is not None else 'unknown error'}"
                ) from error

            if self.stopped.is_set():
                raise CallExecutionError(
                    "VoiceProbe v3 WorkerRunner stopped before Flux connected."
                )

            if time.monotonic() >= deadline:
                self.request_stop()
                raise CallExecutionError(
                    "VoiceProbe v3 timed out waiting for Deepgram Flux to connect."
                )

            time.sleep(0.01)

    def submit_pcm(self, pcm16: bytes) -> None:
        if self.error_event.is_set():
            error = self.error
            raise RuntimeError("VoiceProbe v3 media runtime is in an error state") from error

        if not self.connected.is_set():
            raise RuntimeError("VoiceProbe v3 PCM arrived before Flux was connected")

        with self._lock:
            bundle = self._bundle

        if bundle is None:
            raise RuntimeError("VoiceProbe v3 PCM feeder is unavailable")

        future = bundle.feeder.submit_pcm(pcm16)

        def capture_submission_error(done) -> None:
            try:
                done.result()
            except BaseException as error:  # includes event-loop cancellation failures
                if self._stop_requested.is_set():
                    return
                self._set_error(error)
                self.request_stop()

        future.add_done_callback(capture_submission_error)

    def snapshot(self) -> FlowSnapshot:
        with self._lock:
            snapshot = self._latest_snapshot

        if snapshot is None:
            raise RuntimeError("VoiceProbe v3 flow snapshot is not available yet")

        return snapshot

    def scenario_metadata(self) -> dict[str, object]:
        with self._lock:
            return dict(self._latest_scenario_metadata)

    def flush_and_snapshot(self, *, timeout: float = 5.0) -> FlowSnapshot:
        with self._lock:
            loop = self._loop
            bridge = self._bridge

        if loop is None or bridge is None or loop.is_closed():
            return self.snapshot()

        async def flush() -> FlowSnapshot:
            await bridge.runtime.ingress.flush_stabilized_pending()
            snapshot = bridge.runtime.flow_controller.tracker.snapshot()
            self._store_snapshot(snapshot)
            with self._lock:
                self._latest_scenario_metadata = dict(bridge.scenario_metadata)
            return snapshot

        future = asyncio.run_coroutine_threadsafe(flush(), loop)
        return future.result(timeout=timeout)

    def wait_for_speech_idle(self, *, timeout: float = 30.0) -> None:
        with self._lock:
            loop = self._loop

        if loop is None or loop.is_closed():
            return

        future = asyncio.run_coroutine_threadsafe(
            self._speech_task.wait_for_idle(),
            loop,
        )
        future.result(timeout=timeout)

    def record_persona_final_evidence(self) -> None:
        with self._lock:
            bridge = self._bridge

        if isinstance(
            bridge,
            _RecordingPipecatRuntimeBridge,
        ):
            bridge.record_persona_final_evidence()

    def request_stop(self) -> None:
        self._stop_requested.set()

        with self._lock:
            loop = self._loop
            stop_async = self._stop_async

        if loop is not None and stop_async is not None and not loop.is_closed():
            loop.call_soon_threadsafe(stop_async.set)

    def stop(self) -> None:
        self.request_stop()
        thread = self._thread

        if thread is None:
            return

        thread.join(timeout=RUNNER_SHUTDOWN_TIMEOUT_SECONDS + 2.0)

        if thread.is_alive():
            raise RuntimeError("VoiceProbe v3 WorkerRunner thread did not stop cleanly")

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._async_main())
        except BaseException as error:
            # Any exception escaping _async_main is infrastructure failure;
            # normal requested shutdown returns without raising.
            if not isinstance(error, asyncio.CancelledError):
                self._set_error(error)
        finally:
            self.stopped.set()

    async def _async_main(self) -> None:
        from pipecat.workers.runner import WorkerRunner

        loop = asyncio.get_running_loop()
        stop_async = asyncio.Event()

        with self._lock:
            self._loop = loop
            self._stop_async = stop_async

        flux = build_production_flux_service(api_key=self._api_key)

        @flux.service.event_handler("on_connected")
        async def on_connected(service) -> None:
            del service
            if self._readiness.mark_flux_connected():
                self._recorder.record_event("v3_flux_connected")

        @flux.service.event_handler("on_disconnected")
        async def on_disconnected(service) -> None:
            del service
            self.disconnected.set()
            self._recorder.record_event("v3_flux_disconnected")

        bridge = _RecordingPipecatRuntimeBridge(
            recorder=self._recorder,
            persona_runtime=self._persona_runtime,
        )
        def on_startframe() -> None:
            self._recorder.record_event("v3_pipeline_startframe_established")
            if self._readiness.mark_pipeline_started():
                self._recorder.record_event("v3_flux_connected")

        bundle = build_audiosocket_flux_input_worker(
            stt_service=flux.service,
            bridge=bridge,
            speech_sink=self._speech_task,
            loop=loop,
            on_startframe=on_startframe,
        )

        # Install the level-triggered observer before WorkerRunner can start the
        # service. Pipecat sets this Event immediately after the server-ready log
        # and before scheduling on_connected handlers.
        async def observe_flux_state() -> None:
            if await _observe_flux_connected_state(flux.service, self._readiness):
                self._recorder.record_event("v3_flux_connected")

        flux_state_task = asyncio.create_task(
            observe_flux_state(),
            name="voiceprobe-v3-flux-readiness-state",
        )
        self._speech_task.set_on_playback_finished(bridge.on_tts_stopped)
        self._store_snapshot(bridge.runtime.flow_controller.tracker.snapshot())

        with self._lock:
            self._bundle = bundle
            self._bridge = bridge

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(bundle.worker)
        runner_task = asyncio.create_task(
            runner.run(),
            name="voiceprobe-v3-live-worker-runner",
        )
        monitor_task = asyncio.create_task(
            self._monitor_runtime(stop_async),
            name="voiceprobe-v3-live-monitor",
        )
        stop_task = asyncio.create_task(
            stop_async.wait(),
            name="voiceprobe-v3-live-stop",
        )

        try:
            done, _ = await asyncio.wait(
                {runner_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if runner_task in done and not self._stop_requested.is_set():
                await runner_task
                raise RuntimeError("Pipecat WorkerRunner stopped unexpectedly")
        finally:
            self._stop_requested.set()
            stop_async.set()
            stop_task.cancel()
            monitor_task.cancel()
            flux_state_task.cancel()

            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            try:
                await flux_state_task
            except asyncio.CancelledError:
                pass

            if not runner_task.done():
                await bundle.worker.cancel()

            try:
                await asyncio.wait_for(
                    runner_task,
                    timeout=RUNNER_SHUTDOWN_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                runner_task.cancel()
                try:
                    await runner_task
                except asyncio.CancelledError:
                    pass

            with self._lock:
                self._bundle = None
                self._bridge = None
                self._stop_async = None
                self._loop = None

    async def _monitor_runtime(self, stop_async: asyncio.Event) -> None:
        while not stop_async.is_set():
            with self._lock:
                bridge = self._bridge

            if bridge is not None:
                snapshot = bridge.runtime.flow_controller.tracker.snapshot()
                self._store_snapshot(snapshot)
                with self._lock:
                    self._latest_scenario_metadata = dict(bridge.scenario_metadata)

                if bridge.objective_complete:
                    self.objective_complete.set()
                if bridge.scenario_terminal:
                    self.scenario_terminal.set()

            speech_error = self._speech_task.last_error

            if speech_error is not None and not self._stop_requested.is_set():
                self._set_error(speech_error)
                self.request_stop()
                return

            if (
                self.connected.is_set()
                and self.disconnected.is_set()
                and not self._stop_requested.is_set()
            ):
                error = RuntimeError("Deepgram Flux disconnected during the live call")
                self._set_error(error)
                self.request_stop()
                return

            await asyncio.sleep(0.02)

    def _store_snapshot(self, snapshot: FlowSnapshot) -> None:
        with self._lock:
            self._latest_snapshot = snapshot

    def _set_error(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
        self.error_event.set()


def _recv_exact_polling(
    connection: socket.socket,
    size: int,
    *,
    should_stop: Callable[[], bool],
) -> bytes | None:
    """Receive exactly ``size`` bytes while polling local completion state."""

    data = bytearray()

    while len(data) < size:
        if should_stop():
            raise _LocalMediaStop

        try:
            chunk = connection.recv(size - len(data))
        except socket.timeout:
            continue
        except OSError:
            if should_stop():
                raise _LocalMediaStop
            raise

        if not chunk:
            return None

        data.extend(chunk)

    return bytes(data)


def _recv_message_polling(
    connection: socket.socket,
    *,
    should_stop: Callable[[], bool],
) -> tuple[int, bytes] | None:
    header = _recv_exact_polling(connection, 3, should_stop=should_stop)

    if header is None:
        return None

    payload_length = int.from_bytes(header[1:3], "big")
    payload = _recv_exact_polling(
        connection,
        payload_length,
        should_stop=should_stop,
    )

    if payload is None:
        return None

    return header[0], payload


def _terminate_with_media_lock(
    connection: socket.socket,
    *,
    idle_stop: threading.Event,
    send_lock: threading.Lock,
) -> bool:
    idle_stop.set()

    with send_lock:
        return terminate_audiosocket_connection(connection)


def _record_hangup_observation(
    *,
    recorder: RunArtifactRecorder,
    originate_result: OriginateResult,
    hangup_observer: Callable[[], AsteriskHangupResult] | None,
    ami_error_type: type[BaseException],
) -> AsteriskHangupResult | None:
    if hangup_observer is None:
        recorder.record_event(
            "asterisk_hangup_observer_unavailable",
            asterisk_unique_id=originate_result.asterisk_unique_id,
            channel=originate_result.channel,
        )
        return None

    try:
        result = hangup_observer()
    except ami_error_type as error:
        recorder.record_event(
            "asterisk_hangup_observer_error",
            asterisk_unique_id=originate_result.asterisk_unique_id,
            channel=originate_result.channel,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        return None

    recorder.record_event(
        "asterisk_hangup_observed",
        asterisk_unique_id=result.unique_id,
        channel=result.channel,
        linked_id=result.linked_id,
        cause=result.cause,
        cause_text=result.cause_text,
        tech_cause=result.tech_cause,
    )
    return result


def execute_v3_asterisk_media(
    *,
    request: AssessmentCallRequest,
    call_id: UUID,
    originate: Callable[[], OriginateResult],
    pipeline: Any,
    voice: str,
    tts_pcm_cache: Mapping[str, bytes] | None,
    deepgram_api_key: str,
    artifact_root: Path | str,
    host: str,
    port: int,
    accept_timeout_seconds: float,
    hangup_observer: Callable[[], AsteriskHangupResult] | None,
    ami_error_type: type[BaseException],
    classify_termination: Callable[..., Any],
    termination_failure_reason: Callable[..., str | None],
) -> V3AsteriskMediaResult:
    """Run exactly one v3 live-media call after all adapter safety checks."""

    api_key = deepgram_api_key.strip()

    if not api_key:
        raise CallExecutionError(
            "VOICEPROBE_V3_LIVE=1 requires DEEPGRAM_API_KEY before dialing."
        )

    scenario = get_scenario(request.scenario_id)

    accent_mode = accent_mode_from_environment()
    background_mode = background_mode_from_environment()
    background_snr_db = background_snr_from_environment()
    background_asset = validate_background_asset()
    if background_mode != "none" and not background_asset["valid"]:
        raise CallExecutionError("Background asset validation failed before telephony")
    accent_cache: AccentCache | None = None
    if accent_mode in {ACCENT_MELO_INDIA, ACCENT_CHATTERBOX_KOREAN_HEAVY}:
        accent_cache = AccentCache(
            os.environ.get("VOICEPROBE_ACCENT_CACHE_ROOT", str(DEFAULT_CACHE_ROOT)),
            mode=accent_mode,
            scenario=request.scenario_id,
        )
        preflight = accent_cache_preflight(
            scenario=request.scenario_id,
            mode=accent_mode,
            cache=accent_cache,
        )
        if preflight["missing"]:
            raise CallExecutionError(
                "Strict accent cache preflight failed before telephony: "
                f"scenario={request.scenario_id}; missing={len(preflight['missing'])}"
            )
        if accent_mode == ACCENT_MELO_INDIA:
            # Spawn the one EN_INDIA miss renderer without gating cache-hit
            # call startup on model loading. A miss waits for this same worker.
            warm_melo_india_renderer()

    # Validate persona configuration before opening the live-call boundary.
    # Empty configuration preserves normal VoiceProbe behavior.
    persona_runtime = persona_runtime_from_environment()

    with (
        LiveAudioMonitor.from_environment() as live_monitor,
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server,
    ):
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        server.settimeout(accept_timeout_seconds)

        # Preserve the adapter's critical safety ordering: listener first.
        originate_result = originate()

        try:
            connection, address = server.accept()
        except TimeoutError as error:
            raise CallExecutionError(
                "Asterisk originated the call but did not connect "
                "to the local AudioSocket listener in time."
            ) from error

        with RunArtifactRecorder(root=artifact_root, scenario=scenario) as recorder:
            recorder.record_event(
                "suite_adapter_call_started",
                execution_id=request.execution_id,
                position=request.position,
                call_id=str(call_id),
                asterisk_unique_id=originate_result.asterisk_unique_id,
                address=address,
                reasoning_mode="v3_live",
                accent_mode=accent_mode,
                tts_backend=(("melo_accent_cache" if accent_mode == ACCENT_MELO_INDIA else "chatterbox_multilingual_cache") if accent_cache else "kokoro"),
                accent_speaker=(("EN_INDIA" if accent_mode == ACCENT_MELO_INDIA else "synthetic_local_korean_reference_v1") if accent_cache else None),
                background_mode=background_mode,
                background_snr_db=background_snr_db if background_mode != "none" else None,
                background_seed=BACKGROUND_SEED if background_mode != "none" else None,
                max_duration_seconds=request.max_duration_seconds,
            )

            call_finished = threading.Event()
            max_duration_reached = threading.Event()
            idle_stop = threading.Event()
            send_lock = threading.Lock()
            boundary: AudioSocketV3MediaBoundary | None = None
            live_runtime: _AsyncV3Runtime | None = None
            observed_call_id: UUID | None = None
            completion_requested = False

            def enforce_max_duration() -> None:
                expired = not call_finished.wait(request.max_duration_seconds)

                if not expired:
                    return

                max_duration_reached.set()
                recorder.record_event(
                    "max_call_duration_reached",
                    max_duration_seconds=request.max_duration_seconds,
                )
                _terminate_with_media_lock(
                    connection,
                    idle_stop=idle_stop,
                    send_lock=send_lock,
                )

            watchdog = threading.Thread(
                target=enforce_max_duration,
                name=f"voiceprobe-v3-call-deadline-{request.position}",
                daemon=True,
            )
            watchdog.start()

            try:
                with connection:
                    connection.settimeout(SOCKET_POLL_SECONDS)

                    def should_stop_receiving() -> bool:
                        return (
                            max_duration_reached.is_set()
                            or (
                                live_runtime is not None
                                and (
                                    live_runtime.objective_complete.is_set()
                                    or live_runtime.scenario_terminal.is_set()
                                    or live_runtime.error_event.is_set()
                                )
                            )
                        )

                    while True:
                        if max_duration_reached.is_set():
                            break

                        if live_runtime is not None and live_runtime.error_event.is_set():
                            error = live_runtime.error
                            raise CallExecutionError(
                                "VoiceProbe v3 live media runtime failed: "
                                f"{type(error).__name__ if error is not None else 'unknown'}: "
                                f"{error if error is not None else 'unknown error'}"
                            ) from error

                        if (
                            live_runtime is not None
                            and (
                                live_runtime.objective_complete.is_set()
                                or live_runtime.scenario_terminal.is_set()
                            )
                        ):
                            snapshot = live_runtime.flush_and_snapshot()

                            if not (
                                live_runtime.objective_complete.is_set()
                                or live_runtime.scenario_terminal.is_set()
                            ):
                                live_runtime.objective_complete.clear()
                                live_runtime.scenario_terminal.clear()
                                continue

                            live_runtime.wait_for_speech_idle()
                            recorder.record_event(
                                (
                                    "v3_objective_complete"
                                    if live_runtime.objective_complete.is_set()
                                    else "v3_scenario_terminal"
                                ),
                                accepted_slot_text=snapshot.accepted_slot_text,
                                booking_confirmation_text=(
                                    snapshot.booking_confirmation_text
                                ),
                            )
                            completion_requested = True
                            _terminate_with_media_lock(
                                connection,
                                idle_stop=idle_stop,
                                send_lock=send_lock,
                            )
                            break

                        try:
                            message = _recv_message_polling(
                                connection,
                                should_stop=should_stop_receiving,
                            )
                        except _LocalMediaStop:
                            continue
                        except OSError:
                            if max_duration_reached.is_set() or completion_requested:
                                break
                            raise

                        if message is None:
                            recorder.record_event("audiosocket_disconnected")
                            break

                        message_type, payload = message

                        if message_type == TYPE_HANGUP:
                            recorder.record_event("hangup_received")

                            if live_runtime is not None:
                                # Drain a just-arrived final Flux EOT.  A booking
                                # confirmation normally produces no patient speech.
                                live_runtime.flush_and_snapshot()

                            break

                        if message_type == TYPE_UUID:
                            if len(payload) != 16:
                                raise CallExecutionError(
                                    "AudioSocket UUID frame did not contain 16 bytes."
                                )

                            received = UUID(bytes=payload)

                            if received != call_id:
                                raise CallExecutionError(
                                    "AudioSocket UUID did not match the originated call."
                                )

                            if observed_call_id is not None:
                                if received != observed_call_id:
                                    raise CallExecutionError(
                                        "AudioSocket supplied conflicting call UUIDs."
                                    )
                                continue

                            observed_call_id = received
                            recorder.record_event(
                                "call_uuid_received",
                                call_id=str(received),
                            )

                            renderer = KokoroTelephonyRenderer(
                                pipeline=pipeline,
                                config=AudioSocketKokoroConfig(voice=voice),
                                pcm_cache=tts_pcm_cache,
                                accent_cache=accent_cache,
                            )
                            speech_task = AudioSocketKokoroSpeechTask(
                                connection=connection,
                                renderer=renderer,
                                send_lock=send_lock,
                                recorder=recorder,
                                config=AudioSocketKokoroConfig(voice=voice),
                                audio_observer=live_monitor.observe_outbound,
                            )
                            def observe_inbound(pcm: bytes) -> None:
                                live_monitor.observe_inbound(pcm)

                            boundary = AudioSocketV3MediaBoundary(
                                connection=connection,
                                speech_task=speech_task,
                                send_lock=send_lock,
                                recorder=recorder,
                                audio_observer=observe_inbound,
                            )

                            # Contract: UUID -> idle media -> WorkerRunner/Flux.
                            boundary.start_idle_silence(stop=idle_stop)
                            recorder.record_event("idle_silence_media_started")

                            live_runtime = _AsyncV3Runtime(
                                api_key=api_key,
                                speech_task=speech_task,
                                recorder=recorder,
                                persona_runtime=persona_runtime,
                            )
                            live_runtime.start()
                            recorder.record_event("v3_live_media_started")
                            continue

                        if message_type == TYPE_DTMF:
                            recorder.record_event(
                                "dtmf_received",
                                digit=payload.decode("ascii", errors="replace"),
                            )
                            continue

                        if message_type != TYPE_PCM_8KHZ:
                            continue

                        if boundary is None or live_runtime is None:
                            # Preserve raw evidence even if Asterisk violates the
                            # expected UUID-before-media ordering.
                            recorder.record_inbound_pcm(payload)
                            recorder.record_event(
                                "v3_pcm_before_uuid_dropped",
                                pcm_bytes=len(payload),
                            )
                            continue

                        boundary.forward_inbound_pcm(
                            payload,
                            submit_pcm=live_runtime.submit_pcm,
                        )
            finally:
                call_finished.set()
                idle_stop.set()
                watchdog.join(timeout=1.0)

                if live_runtime is not None:
                    try:
                        # Final deterministic drain before the event loop closes.
                        live_runtime.flush_and_snapshot()
                    except BaseException as error:
                        recorder.record_event(
                            "v3_final_flush_error",
                            error_type=type(error).__name__,
                            error_message=str(error),
                        )

                    try:
                        live_runtime.record_persona_final_evidence()
                    except BaseException as error:
                        recorder.record_event(
                            "persona_final_evidence_error",
                            error_type=type(error).__name__,
                            error_message=str(error),
                        )

                    try:
                        live_runtime.stop()
                    except BaseException as error:
                        recorder.record_event(
                            "v3_worker_shutdown_error",
                            error_type=type(error).__name__,
                            error_message=str(error),
                        )
                        raise

                if boundary is not None:
                    boundary.join_idle_silence(timeout=1.0)

            if observed_call_id is None:
                raise CallExecutionError(
                    "AudioSocket session ended without a call UUID."
                )

            if live_runtime is None:
                raise CallExecutionError(
                    "VoiceProbe v3 runtime never started after the AudioSocket UUID."
                )

            final_snapshot = live_runtime.snapshot()
            scenario_metadata = live_runtime.scenario_metadata()
            projection = project_v3_flow_snapshot(
                final_snapshot,
                scenario_id=request.scenario_id,
                scenario_metadata=scenario_metadata,
            )
            hangup_result = _record_hangup_observation(
                recorder=recorder,
                originate_result=originate_result,
                hangup_observer=hangup_observer,
                ami_error_type=ami_error_type,
            )

            termination_status = classify_termination(
                objective_complete=projection.objective_complete,
                max_duration_reached=max_duration_reached.is_set(),
            )
            failure_reason = termination_failure_reason(
                status=termination_status,
                booking_confirmed=projection.booking_confirmed,
                offer_accepted=projection.offer_accepted,
                offered_day=projection.offered_day,
                offered_time=projection.offered_time,
            )
            scenario_reason = scenario_termination_failure_reason(
                status=termination_status, projection=projection
            )
            if scenario_reason is not None:
                failure_reason = scenario_reason

            recorder.record_event(
                "v3_flow_snapshot",
                communicated=sorted(
                    stage.value for stage in final_snapshot.communicated
                ),
                confirmed=sorted(
                    stage.value for stage in final_snapshot.confirmed
                ),
                current_stage=final_snapshot.current_stage.value,
                complete=final_snapshot.complete,
                accepted_slot_text=final_snapshot.accepted_slot_text,
                booking_confirmation_text=(
                    final_snapshot.booking_confirmation_text
                ),
            )
            recorder.record_event(
                "call_termination_classified",
                termination_status=termination_status.value,
                objective_complete=projection.objective_complete,
                booking_confirmed=projection.booking_confirmed,
                offer_accepted=projection.offer_accepted,
                offered_day=projection.offered_day,
                offered_time=projection.offered_time,
                accepted_slot_text=projection.accepted_slot_text,
                booking_confirmation_text=projection.booking_confirmation_text,
                max_duration_reached=max_duration_reached.is_set(),
                asterisk_hangup_observed=(hangup_result is not None),
                asterisk_hangup_cause=(
                    hangup_result.cause if hangup_result is not None else None
                ),
                asterisk_hangup_cause_text=(
                    hangup_result.cause_text if hangup_result is not None else None
                ),
                scenario=request.scenario_id,
                scenario_state=scenario_metadata,
            )

            duration_seconds = recorder.elapsed_seconds
            artifact_status = (
                "completed"
                if projection.objective_complete
                else (
                    "target_capability_blocked"
                    if scenario_metadata.get("experiment_status")
                    == "target_capability_blocked"
                    else termination_status.value
                )
            )
            recorder.finalize(
                status=artifact_status,
                call_id=str(observed_call_id),
                error=failure_reason,
            )

            return V3AsteriskMediaResult(
                call_id=observed_call_id,
                artifact_run_id=recorder.run_id,
                duration_seconds=duration_seconds,
                originate=originate_result,
                hangup=hangup_result,
                termination_status=termination_status,
                objective_complete=projection.objective_complete,
                booking_confirmed=projection.booking_confirmed,
                offer_accepted=projection.offer_accepted,
                offered_day=projection.offered_day,
                offered_time=projection.offered_time,
                failure_reason=failure_reason,
            )
