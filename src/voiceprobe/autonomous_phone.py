"""Autonomous half-duplex VoiceProbe phone diagnostic.

Inbound PSTN audio:
AudioSocket -> Moonshine -> TurnAssembler -> PatientSession

Outbound patient audio:
PatientSession -> Kokoro -> 8 kHz PCM16 -> AudioSocket

This diagnostic intentionally ignores inbound audio while the simulated
patient is reasoning or speaking. Full-duplex/barge-in comes later.
"""

from __future__ import annotations

import argparse
import queue
import socket
import threading
import time
import uuid
from collections.abc import Sequence
from time import perf_counter

import httpx
import numpy as np
from kokoro import KPipeline

from voiceprobe.agents.brain import (
    CommunicationKind,
    PatientBrain,
)
from voiceprobe.artifacts.recorder import RunArtifactRecorder
from voiceprobe.conversation.session import PatientSession
from voiceprobe.conversation.turns import CompletedTurn
from voiceprobe.interpreters.ollama import OllamaConversationInterpreter
from voiceprobe.media.live_asr import (
    AUDIO_SAMPLE_RATE_HZ,
    TYPE_DTMF,
    TYPE_HANGUP,
    TYPE_PCM_8KHZ,
    TYPE_UUID,
    build_transcriber,
    pcm16_to_float32,
    recv_exact,
)
from voiceprobe.scenarios.catalog import (
    DEFAULT_SCENARIO_ID,
    get_scenario,
    scenario_ids,
)
from voiceprobe.scenarios.models import PatientScenario
from voiceprobe.tts.telephony import (
    FRAME_BYTES,
    FRAME_DURATION_SECONDS,
    build_audiosocket_packet,
    build_audiosocket_terminate_packet,
    float_audio_to_pcm16,
    iter_pcm_frames,
    normalize_text_for_tts,
    resample_to_telephony,
)
from voiceprobe.verbalizers.ollama import OllamaNaturalVerbalizer

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9019

DEFAULT_MODEL = "qwen3:14b"
DEFAULT_OLLAMA_URL = "http://10.0.0.219:11434/api/chat"

DEFAULT_VOICE = "af_heart"

ECHO_GUARD_SECONDS = 0.35


# High-frequency responses produced verbatim by the deterministic
# verbalizer. These are rendered to telephony-ready PCM before dialing
# so time-critical replies do not wait for Kokoro inference.
PRE_RENDERED_TTS_TEXTS = (
    "No, I need an appointment.",
    "I need to schedule an appointment for Friday afternoon.",
    "Yes, please.",
    "Yes, that works.",
    "Could you repeat that?",
    "Okay, thank you. Bye.",
    'Alex.',
    'Morgan.',
    'April 12, 1998.',
    'Blue Cross.',
    'Friday afternoon.',
    'I need a new patient consultation.',
    "I'm a new patient.",
    "No, I haven't visited before.",
    "I'm a new patient. No, I haven't visited before.",
    "I don't have a preference. Any available provider is fine.",
    'No, I need Friday afternoon.',
)


def build_scenario(
    scenario_id: str = DEFAULT_SCENARIO_ID,
) -> PatientScenario:
    """Resolve immutable patient truth for one autonomous call."""
    return get_scenario(scenario_id)


def synthesize(
    *,
    pipeline: KPipeline,
    voice: str,
    text: str,
) -> np.ndarray:
    """Generate one complete Kokoro utterance."""
    pieces: list[np.ndarray] = []

    for _, _, audio in pipeline(
        text,
        voice=voice,
        speed=1.0,
    ):
        pieces.append(
            np.asarray(
                audio,
                dtype=np.float32,
            )
        )

    if not pieces:
        raise RuntimeError("Kokoro generated no audio.")

    return np.concatenate(pieces)


def build_pre_rendered_tts_cache(
    *,
    pipeline: KPipeline,
    voice: str,
    texts: tuple[str, ...] = PRE_RENDERED_TTS_TEXTS,
) -> dict[str, bytes]:
    """Render fixed deterministic responses to 8 kHz PCM16 before dialing.

    Cache keys use the exact normalized text that process_turns sends to
    Kokoro. A missing entry is always safe because the live path falls back
    to ordinary synthesis.
    """
    cache: dict[str, bytes] = {}

    for patient_text in texts:
        tts_text = normalize_text_for_tts(patient_text)

        if tts_text in cache:
            continue

        audio_24k = synthesize(
            pipeline=pipeline,
            voice=voice,
            text=tts_text,
        )

        audio_8k = resample_to_telephony(audio_24k)
        cache[tts_text] = float_audio_to_pcm16(audio_8k)

    return cache


def send_audio(
    connection: socket.socket,
    pcm16: bytes,
    *,
    recorder: RunArtifactRecorder | None = None,
) -> None:
    """Send one patient response at telephony playback cadence."""
    next_deadline = time.monotonic()

    for frame in iter_pcm_frames(pcm16):
        connection.sendall(build_audiosocket_packet(frame))

        # Capture only audio successfully handed to AudioSocket.
        # A failed send must not appear as if it reached the call.
        if recorder is not None:
            recorder.record_outbound_pcm(frame)

        next_deadline += FRAME_DURATION_SECONDS

        delay = next_deadline - time.monotonic()

        if delay > 0:
            time.sleep(delay)


# CONTINUOUS AUDIOSOCKET IDLE SILENCE
#
# Asterisk's rtp_keepalive sends sparse comfort-noise packets when the
# application provides no outbound AudioSocket PCM. Real calls repeatedly
# terminated about 15-16 seconds after VoiceProbe's final spoken PCM frame.
#
# Keep the media leg continuously active by supplying correctly framed,
# real-time-paced PCM16 silence while VoiceProbe is listening. Speech and
# silence share one lock so silence can never be interleaved into TTS audio.
def send_audio_synchronized(
    connection: socket.socket,
    pcm16: bytes,
    *,
    send_lock: threading.Lock | None = None,
    recorder: RunArtifactRecorder | None = None,
) -> None:
    """Send patient speech without interleaving idle-silence frames."""

    if send_lock is None:
        send_audio(
            connection,
            pcm16,
            recorder=recorder,
        )
        return

    with send_lock:
        send_audio(
            connection,
            pcm16,
            recorder=recorder,
        )


def send_idle_silence(
    connection: socket.socket,
    *,
    stop: threading.Event,
    send_lock: threading.Lock,
) -> None:
    """Continuously send 20 ms PCM16 silence until the call stops.

    The sender is paced against monotonic time and shares the exact same
    output lock as patient speech. It therefore maintains ordinary outbound
    AudioSocket media while listening without mixing silence into TTS.
    """

    silence_frame = bytes(FRAME_BYTES)
    silence_packet = build_audiosocket_packet(
        silence_frame
    )

    next_deadline = time.monotonic()

    while not stop.is_set():

        with send_lock:

            # The call may have ended while this thread was waiting for
            # patient speech to release the output lock.
            if stop.is_set():
                return

            try:
                connection.sendall(
                    silence_packet
                )
            except OSError:
                return

        next_deadline += FRAME_DURATION_SECONDS

        delay = (
            next_deadline
            - time.monotonic()
        )

        if delay > 0:
            if stop.wait(delay):
                return
        else:
            # Never "catch up" by flooding stale silence frames if the
            # scheduler temporarily stalls.
            next_deadline = time.monotonic()


def terminate_audiosocket_connection(
    connection: socket.socket,
) -> bool:
    """Request AudioSocket termination and unblock the receive loop.

    Returns whether the explicit protocol termination frame was sent.
    Socket shutdown is attempted regardless, because the remote side may
    already have disconnected by the time the local patient finishes.
    """
    packet_sent = False

    try:
        connection.sendall(build_audiosocket_terminate_packet())
        packet_sent = True
    except OSError:
        pass

    try:
        connection.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass

    return packet_sent


def process_turns(
    *,
    turns: queue.Queue[CompletedTurn | None],
    connection: socket.socket,
    session: PatientSession,
    pipeline: KPipeline,
    voice: str,
    busy: threading.Event,
    stop: threading.Event,
    recorder: RunArtifactRecorder,
    audiosocket_send_lock: threading.Lock | None = None,
    tts_pcm_cache: dict[str, bytes] | None = None,
) -> None:
    """Process complete ASR turns sequentially."""
    while True:
        turn = turns.get()

        try:
            if turn is None:
                return

            if stop.is_set():
                return

            print()
            print("=" * 72)
            print(f"ASR TURN: {turn.text}")
            print("=" * 72)

            reasoning_started = perf_counter()

            result = session.handle_agent_turn(turn.text)

            reasoning_seconds = perf_counter() - reasoning_started

            print(f"PATIENT TEXT: {result.patient_text}")
            print(f"DECISION:     {result.decision.kind.value}")

            if result.decision.kind is CommunicationKind.WAIT:
                response_prep_seconds = max(
                    0.0,
                    time.monotonic() - turn.completed_at,
                )
                endpoint_and_queue_seconds = max(
                    0.0,
                    response_prep_seconds - reasoning_seconds,
                )

                recorder.record_turn_metrics(
                    {
                        "agent_turn": turn.text,
                        "decision": result.decision.kind.value,
                        "interpreter_seconds": (result.timings.interpreter_seconds),
                        "decision_seconds": (result.timings.decision_seconds),
                        "verbalizer_seconds": 0.0,
                        "state_update_seconds": (result.timings.state_update_seconds),
                        "reasoning_seconds": reasoning_seconds,
                        "tts_seconds": 0.0,
                        "endpoint_queue_seconds": (endpoint_and_queue_seconds),
                        "response_prep_seconds": response_prep_seconds,
                        "speech_seconds": 0.0,
                        "objective_complete": (result.progress.objective_complete),
                    }
                )

                recorder.record_event(
                    "patient_wait",
                    agent_turn=turn.text,
                    decision=result.decision.kind.value,
                    meaning=result.meaning,
                    response_prep_seconds=response_prep_seconds,
                    objective_complete=(result.progress.objective_complete),
                )

                print(
                    "[WAIT] No patient response required; continuing to listen.",
                    flush=True,
                )

                # Do not run the verbalizer, Kokoro, resampling, AudioSocket
                # playback, patient transcript recording, or echo guard.
                continue

            recorder.record_event(
                "patient_response_generated",
                agent_turn=turn.text,
                patient_text=result.patient_text,
                decision=result.decision.kind.value,
                facts_to_communicate=(result.decision.facts_to_communicate),
                probe=(
                    result.decision.probe.value
                    if result.decision.probe is not None
                    else None
                ),
                meaning=result.meaning,
                objective_complete=(result.progress.objective_complete),
            )

            if result.progress.objective_complete:
                recorder.record_event(
                    "objective_complete",
                    decision=result.decision.kind.value,
                )
                print()
                print("*** APPOINTMENT OBJECTIVE COMPLETE ***")

            if stop.is_set():
                recorder.record_event(
                    "tts_skipped",
                    reason="call_ended",
                    patient_text=result.patient_text,
                )

                print(
                    "[CALL ENDED] Skipping TTS for completed turn.",
                    flush=True,
                )
                return

            tts_started = perf_counter()

            tts_text = normalize_text_for_tts(result.patient_text)

            if tts_text != result.patient_text:
                recorder.record_event(
                    "tts_text_normalized",
                    original_text=result.patient_text,
                    tts_text=tts_text,
                )

                print(
                    f"TTS TEXT:     {tts_text}",
                    flush=True,
                )

            cached_pcm16 = (
                tts_pcm_cache.get(tts_text)
                if tts_pcm_cache is not None
                else None
            )

            if cached_pcm16 is not None:
                pcm16 = cached_pcm16
                tts_seconds = 0.0
                audio_seconds = len(pcm16) / (8_000 * 2)

                recorder.record_event(
                    "tts_cache_hit",
                    tts_text=tts_text,
                    pcm16_bytes=len(pcm16),
                )
            else:
                audio_24k = synthesize(
                    pipeline=pipeline,
                    voice=voice,
                    text=tts_text,
                )

                tts_seconds = perf_counter() - tts_started

                audio_8k = resample_to_telephony(audio_24k)

                pcm16 = float_audio_to_pcm16(audio_8k)

                audio_seconds = len(audio_8k) / 8_000

                recorder.record_event(
                    "tts_cache_miss",
                    tts_text=tts_text,
                )

            response_prep_seconds = time.monotonic() - turn.completed_at

            print()
            endpoint_and_queue_seconds = max(
                0.0,
                response_prep_seconds - reasoning_seconds - tts_seconds,
            )

            print(f"Interpreter:    {result.timings.interpreter_seconds:.3f}s")
            print(f"Brain/ground:   {result.timings.decision_seconds:.3f}s")
            print(f"Verbalizer:     {result.timings.verbalizer_seconds:.3f}s")
            print(f"State update:   {result.timings.state_update_seconds:.3f}s")
            print(f"Reasoning:      {reasoning_seconds:.3f}s")
            print(f"TTS:            {tts_seconds:.3f}s")
            print(f"Endpoint/queue: {endpoint_and_queue_seconds:.3f}s")
            print(f"Response prep:  {response_prep_seconds:.3f}s")
            print(f"Speech length:  {audio_seconds:.3f}s")

            recorder.record_turn_metrics(
                {
                    "agent_turn": turn.text,
                    "decision": result.decision.kind.value,
                    "interpreter_seconds": (result.timings.interpreter_seconds),
                    "decision_seconds": (result.timings.decision_seconds),
                    "verbalizer_seconds": (result.timings.verbalizer_seconds),
                    "state_update_seconds": (result.timings.state_update_seconds),
                    "reasoning_seconds": reasoning_seconds,
                    "tts_seconds": tts_seconds,
                    "endpoint_queue_seconds": (endpoint_and_queue_seconds),
                    "response_prep_seconds": (response_prep_seconds),
                    "speech_seconds": audio_seconds,
                    "objective_complete": (result.progress.objective_complete),
                }
            )

            recorder.record_event(
                "response_prepared",
                decision=result.decision.kind.value,
                response_prep_seconds=(response_prep_seconds),
                speech_seconds=audio_seconds,
            )

            if stop.is_set():
                recorder.record_event(
                    "playback_skipped",
                    reason="call_ended",
                    patient_text=result.patient_text,
                )

                print(
                    "[CALL ENDED] Skipping audio playback.",
                    flush=True,
                )
                return

            recorder.record_event(
                "playback_started",
                patient_text=result.patient_text,
                tts_text=tts_text,
            )

            print("Speaking...")

            send_audio_synchronized(
                connection,
                pcm16,
                send_lock=audiosocket_send_lock,
                recorder=recorder,
            )

            recorder.record_transcript_turn(
                speaker="patient",
                text=result.patient_text,
                decision=result.decision.kind.value,
                tts_text=tts_text,
                objective_complete=(result.progress.objective_complete),
            )

            recorder.record_event(
                "playback_finished",
                patient_text=result.patient_text,
                speech_seconds=audio_seconds,
            )

            print("Speech complete.")

            if result.decision.kind is CommunicationKind.END_CONVERSATION:
                recorder.record_event(
                    "local_hangup_requested",
                    decision=result.decision.kind.value,
                    patient_text=result.patient_text,
                )

                # Prevent any new ASR turn from entering the worker while
                # the AudioSocket receive loop is being unblocked.
                stop.set()

                packet_sent = terminate_audiosocket_connection(connection)

                recorder.record_event(
                    "local_hangup_signaled",
                    termination_packet_sent=packet_sent,
                )

                print(
                    "VoiceProbe ended the call.",
                    flush=True,
                )

                return

            # Keep inbound ASR muted briefly after ordinary playback so PSTN
            # echo does not immediately become the next receptionist turn.
            time.sleep(ECHO_GUARD_SECONDS)

        except (
            httpx.HTTPError,
            BrokenPipeError,
            ConnectionError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            recorder.record_event(
                "turn_error",
                error_type=type(error).__name__,
                error_message=str(error),
                agent_turn=(turn.text if turn is not None else None),
            )

            print()
            print(f"TURN ERROR: {type(error).__name__}: {error}")

        finally:
            if turn is not None:
                busy.clear()

            turns.task_done()


def handle_call(
    *,
    connection: socket.socket,
    session: PatientSession,
    pipeline: KPipeline,
    voice: str,
    recorder: RunArtifactRecorder,
    tts_pcm_cache: dict[str, bytes] | None = None,
) -> uuid.UUID | None:
    """Run one autonomous half-duplex AudioSocket call."""
    turn_queue: queue.Queue[CompletedTurn | None] = queue.Queue()

    busy = threading.Event()
    stop = threading.Event()

    # Speech and idle-silence media must never write to AudioSocket
    # concurrently.
    audiosocket_send_lock = threading.Lock()

    # Start only after Asterisk has supplied the AudioSocket UUID.
    idle_silence_thread: threading.Thread | None = None

    def prefetch_candidate(
        candidate_text: str,
    ) -> None:
        if stop.is_set() or busy.is_set():
            return

        started = session.prefetch_agent_turn(candidate_text)

        if started:
            recorder.record_event(
                "prefetch_started",
                candidate_text=candidate_text,
            )

            print(
                f"[PREFETCH START] {candidate_text!r}",
                flush=True,
            )

    def invalidate_candidate() -> None:
        session.invalidate_prefetch()

    def enqueue_turn(
        turn: CompletedTurn,
    ) -> None:
        # Claim the half-duplex response window immediately. This closes
        # the race between Moonshine completing a turn and the worker
        # beginning inference.
        if stop.is_set():
            return

        if busy.is_set():
            recorder.record_event(
                "agent_turn_dropped",
                reason="half_duplex_busy",
                text=turn.text,
            )

            print(f"[HALF-DUPLEX] Dropping overlapping turn: {turn.text!r}")
            return

        recorder.record_transcript_turn(
            speaker="agent",
            text=turn.text,
            asr_lines=turn.lines,
            asr_started_at_monotonic=turn.started_at,
            asr_completed_at_monotonic=turn.completed_at,
        )

        recorder.record_event(
            "agent_turn_completed",
            text=turn.text,
            lines=turn.lines,
        )

        busy.set()
        turn_queue.put_nowait(turn)

    transcriber, listener = build_transcriber(
        on_turn=enqueue_turn,
        on_candidate=prefetch_candidate,
        on_speech_activity=invalidate_candidate,
    )

    worker = threading.Thread(
        target=process_turns,
        kwargs={
            "turns": turn_queue,
            "connection": connection,
            "session": session,
            "pipeline": pipeline,
            "voice": voice,
            "busy": busy,
            "stop": stop,
            "recorder": recorder,
            "audiosocket_send_lock": audiosocket_send_lock,
            "tts_pcm_cache": tts_pcm_cache,
        },
        daemon=True,
    )

    call_id: uuid.UUID | None = None

    transcriber.start()
    worker.start()

    recorder.record_event(
        "call_processing_started",
    )

    try:
        while True:
            header = recv_exact(
                connection,
                3,
            )

            if header is None:
                if stop.is_set():
                    recorder.record_event(
                        "audiosocket_local_disconnect_confirmed",
                    )
                    print("AudioSocket closed after local hangup.")
                else:
                    recorder.record_event(
                        "audiosocket_disconnected",
                    )
                    print("AudioSocket disconnected.")

                break

            message_type = header[0]
            payload_length = int.from_bytes(
                header[1:3],
                "big",
            )

            payload = recv_exact(
                connection,
                payload_length,
            )

            if payload is None:
                if stop.is_set():
                    recorder.record_event(
                        "audiosocket_local_disconnect_confirmed",
                    )
                    print("AudioSocket closed after local hangup.")
                else:
                    recorder.record_event(
                        "audiosocket_payload_disconnected",
                    )
                    print("AudioSocket disconnected during payload.")

                break

            if message_type == TYPE_HANGUP:
                recorder.record_event(
                    "hangup_received",
                )
                print("Call ended.")
                break

            if message_type == TYPE_UUID:
                if len(payload) == 16:
                    call_id = uuid.UUID(bytes=payload)

                    recorder.record_event(
                        "call_uuid_received",
                        call_id=str(call_id),
                    )

                    print(f"Call UUID: {call_id}")

                    if idle_silence_thread is None:
                        idle_silence_thread = threading.Thread(
                            target=send_idle_silence,
                            kwargs={
                                "connection": connection,
                                "stop": stop,
                                "send_lock": audiosocket_send_lock,
                            },
                            name="voiceprobe-idle-silence",
                            daemon=True,
                        )

                        idle_silence_thread.start()

                        recorder.record_event(
                            "idle_silence_media_started",
                            frame_bytes=FRAME_BYTES,
                            frame_duration_seconds=(
                                FRAME_DURATION_SECONDS
                            ),
                        )

                continue

            if message_type == TYPE_DTMF:
                digit = payload.decode(
                    "ascii",
                    errors="replace",
                )
                recorder.record_event(
                    "dtmf_received",
                    digit=digit,
                )

                print(f"DTMF: {digit}")
                continue

            if message_type != TYPE_PCM_8KHZ:
                continue

            # Preserve everything actually received from AudioSocket,
            # including speech that half-duplex logic intentionally
            # withholds from Moonshine.
            recorder.record_inbound_pcm(payload)

            # First diagnostic is intentionally half-duplex. Audio
            # received while VoiceProbe is reasoning/speaking is consumed
            # from AudioSocket but never sent to Moonshine.
            if busy.is_set():
                continue

            audio = pcm16_to_float32(payload)

            transcriber.add_audio(
                audio,
                AUDIO_SAMPLE_RATE_HZ,
            )

    finally:
        stop.set()

        if idle_silence_thread is not None:
            idle_silence_thread.join(
                timeout=1.0
            )

        session.invalidate_prefetch()

        transcriber.stop()
        listener.close()

        turn_queue.put(None)

        worker.join(
            timeout=5.0,
        )

        if worker.is_alive():
            recorder.record_event(
                "worker_drain_wait",
                initial_timeout_seconds=5.0,
            )

            # The Ollama client itself is timeout-bounded. Finish draining
            # the worker before closing/finalizing call artifacts so it
            # cannot write into already-closed recorder files.
            worker.join()

        transcriber.close()

        recorder.record_event(
            "call_processing_finished",
            call_id=(str(call_id) if call_id is not None else None),
        )

        print(f"Autonomous call complete: call_id={call_id}")

    return call_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        choices=scenario_ids(),
        default=DEFAULT_SCENARIO_ID,
        help=(f"Immutable patient scenario to run. Default: {DEFAULT_SCENARIO_ID}"),
    )

    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_OLLAMA_URL,
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = _build_parser().parse_args(argv)

    print("Loading Moonshine...")
    # build_transcriber() is intentionally called inside handle_call so
    # its listener can be bound to that call's turn queue.

    print("Loading Kokoro...")

    kokoro_started = perf_counter()

    pipeline = KPipeline(
        lang_code="a",
        repo_id="hexgrad/Kokoro-82M",
    )

    print(f"Kokoro loaded in {perf_counter() - kokoro_started:.3f}s")

    print(f"Warming voice {args.voice}...")

    warm_started = perf_counter()

    synthesize(
        pipeline=pipeline,
        voice=args.voice,
        text="Hello.",
    )

    print(f"Kokoro warm-up complete in {perf_counter() - warm_started:.3f}s")

    scenario = build_scenario(args.scenario)

    print(f"Scenario: {scenario.scenario_id} | {scenario.objective}")

    with httpx.Client(
        timeout=20.0,
    ) as client:
        interpreter = OllamaConversationInterpreter(
            model=args.model,
            url=args.url,
            client=client,
        )

        verbalizer = OllamaNaturalVerbalizer(
            model=args.model,
            url=args.url,
            client=client,
        )

        session = PatientSession(
            scenario=scenario,
            interpreter=interpreter,
            verbalizer=verbalizer,
            brain=PatientBrain(),
        )

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
                    args.host,
                    args.port,
                )
            )
            server.listen(1)

            print()
            print(f"VoiceProbe autonomous phone listening on {args.host}:{args.port}")
            print("Waiting for one Asterisk call...")

            connection, address = server.accept()

            with RunArtifactRecorder(
                root="artifacts/runs",
                scenario=scenario,
            ) as recorder:
                print(f"Run artifacts: {recorder.run_dir}")

                recorder.record_event(
                    "asterisk_connected",
                    address=address,
                )

                with connection:
                    print(f"Asterisk connected from {address}")

                    call_id = handle_call(
                        connection=connection,
                        session=session,
                        pipeline=pipeline,
                        voice=args.voice,
                        recorder=recorder,
                    )

                recorder.finalize(
                    status="completed",
                    call_id=(str(call_id) if call_id is not None else None),
                )

                print(f"Run artifacts finalized: {recorder.run_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
