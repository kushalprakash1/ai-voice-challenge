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

from .audiosocket_kokoro import (
    AudioSocketKokoroConfig,
    AudioSocketKokoroSpeechTask,
    AudioSocketV3MediaBoundary,
    KokoroTelephonyRenderer,
)
from .audiosocket_pipecat import build_audiosocket_flux_input_worker
from .flow_state import FlowSnapshot
from .production import PipecatRuntimeBridge, build_production_flux_service


DEFAULT_FLUX_CONNECT_TIMEOUT_SECONDS = 10.0
SOCKET_POLL_SECONDS = 0.10
RUNNER_SHUTDOWN_TIMEOUT_SECONDS = 10.0


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


def project_v3_flow_snapshot(snapshot: FlowSnapshot) -> V3LegacyProgressProjection:
    """Map only evidence that v3 actually owns; never invent day/time fields."""

    objective_complete = bool(snapshot.complete)

    return V3LegacyProgressProjection(
        objective_complete=objective_complete,
        # v3 completion itself means explicit remote booking confirmation.
        booking_confirmed=objective_complete,
        # The tracker records this only after an accepted/concretely confirmed slot.
        offer_accepted=snapshot.accepted_slot_text is not None,
        # v3 intentionally carries the exact accepted/confirmation text instead
        # of pretending it has the legacy split day/time representation.
        offered_day=None,
        offered_time=None,
        accepted_slot_text=snapshot.accepted_slot_text,
        booking_confirmation_text=snapshot.booking_confirmation_text,
    )


class _LocalMediaStop(Exception):
    """Internal control-flow signal used to poll async v3 state from recv()."""


class _AsyncV3Runtime:
    """Own one Pipecat WorkerRunner and expose thread-safe PCM submission."""

    def __init__(
        self,
        *,
        api_key: str,
        speech_task: AudioSocketKokoroSpeechTask,
        recorder: RunArtifactRecorder,
    ) -> None:
        self._api_key = api_key
        self._speech_task = speech_task
        self._recorder = recorder

        self.connected = threading.Event()
        self.disconnected = threading.Event()
        self.objective_complete = threading.Event()
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
            self.connected.set()
            self._recorder.record_event("v3_flux_connected")

        @flux.service.event_handler("on_disconnected")
        async def on_disconnected(service) -> None:
            del service
            self.disconnected.set()
            self._recorder.record_event("v3_flux_disconnected")

        bridge = PipecatRuntimeBridge()
        bundle = build_audiosocket_flux_input_worker(
            stt_service=flux.service,
            bridge=bridge,
            speech_sink=self._speech_task,
            loop=loop,
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

            try:
                await monitor_task
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

                if snapshot.complete:
                    self.objective_complete.set()

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

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
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
                            and live_runtime.objective_complete.is_set()
                        ):
                            snapshot = live_runtime.flush_and_snapshot()

                            if not snapshot.complete:
                                live_runtime.objective_complete.clear()
                                continue

                            live_runtime.wait_for_speech_idle()
                            recorder.record_event(
                                "v3_objective_complete",
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
                            )
                            speech_task = AudioSocketKokoroSpeechTask(
                                connection=connection,
                                renderer=renderer,
                                send_lock=send_lock,
                                recorder=recorder,
                                config=AudioSocketKokoroConfig(voice=voice),
                            )
                            boundary = AudioSocketV3MediaBoundary(
                                connection=connection,
                                speech_task=speech_task,
                                send_lock=send_lock,
                                recorder=recorder,
                            )

                            # Contract: UUID -> idle media -> WorkerRunner/Flux.
                            boundary.start_idle_silence(stop=idle_stop)
                            recorder.record_event("idle_silence_media_started")

                            live_runtime = _AsyncV3Runtime(
                                api_key=api_key,
                                speech_task=speech_task,
                                recorder=recorder,
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
            projection = project_v3_flow_snapshot(final_snapshot)
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
            )

            duration_seconds = recorder.elapsed_seconds
            artifact_status = (
                "completed"
                if projection.objective_complete
                else termination_status.value
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
