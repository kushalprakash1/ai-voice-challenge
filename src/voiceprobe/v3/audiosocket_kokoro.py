"""AudioSocket/Kokoro media boundary for VoiceProbe v3.

This module preserves the working telephony media contracts while keeping
Pipecat/Flux reasoning separate:

- Kokoro renders speech at its native rate.
- Existing telephony helpers convert speech to 8 kHz little-endian PCM16.
- Patient speech and continuous idle silence share one AudioSocket send lock.
- Input PCM is always recorded, but Flux forwarding is muted only during
  actual patient playback plus the existing echo guard.
- TTS synthesis/playback runs off the asyncio event loop so Flux can continue
  receiving remote audio during response preparation.

Legacy AudioSocket/Asterisk code is not modified by this module.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol


DEFAULT_VOICE = "af_heart"
TELEPHONY_SAMPLE_RATE = 8_000
ECHO_GUARD_SECONDS = 0.35


class RecorderLike(Protocol):
    def record_event(self, event_name: str, **fields: object) -> None: ...

    def record_inbound_pcm(self, payload: bytes) -> None: ...


SynthesizeFn = Callable[..., Any]
NormalizeFn = Callable[[str], str]
ResampleFn = Callable[[Any], Any]
EncodeFn = Callable[[Any], bytes]
SendAudioFn = Callable[..., None]
IdleSilenceFn = Callable[..., None]
PlaybackFinishedFn = Callable[[], Any]
SubmitPCMFn = Callable[[bytes], Any]


@dataclass(frozen=True, slots=True)
class AudioSocketKokoroConfig:
    voice: str = DEFAULT_VOICE
    telephony_sample_rate: int = TELEPHONY_SAMPLE_RATE
    echo_guard_seconds: float = ECHO_GUARD_SECONDS

    def validate(self) -> None:
        if not self.voice.strip():
            raise ValueError("Kokoro voice must be non-empty")
        if self.telephony_sample_rate != TELEPHONY_SAMPLE_RATE:
            raise ValueError("VoiceProbe AudioSocket media must remain native 8 kHz")
        if self.echo_guard_seconds < 0:
            raise ValueError("echo_guard_seconds must be non-negative")


class KokoroTelephonyRenderer:
    """Render one deterministic response to 8 kHz PCM16.

    The default dependency path reuses the existing, already-tested VoiceProbe
    Kokoro and telephony conversion functions. Tests can inject pure fakes so
    importing this module never requires Kokoro or Moonshine.
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        config: AudioSocketKokoroConfig = AudioSocketKokoroConfig(),
        pcm_cache: Mapping[str, bytes] | None = None,
        synthesize_fn: SynthesizeFn | None = None,
        normalize_fn: NormalizeFn | None = None,
        resample_fn: ResampleFn | None = None,
        encode_fn: EncodeFn | None = None,
    ) -> None:
        config.validate()

        self._pipeline = pipeline
        self._config = config
        self._pcm_cache = dict(pcm_cache or {})

        self._synthesize_fn = synthesize_fn
        self._normalize_fn = normalize_fn
        self._resample_fn = resample_fn
        self._encode_fn = encode_fn

    @property
    def config(self) -> AudioSocketKokoroConfig:
        return self._config

    def render(self, text: str) -> bytes:
        stripped = text.strip()

        if not stripped:
            raise ValueError("Cannot render empty patient speech")

        synthesize_fn, normalize_fn, resample_fn, encode_fn = (
            self._resolved_functions()
        )

        tts_text = normalize_fn(stripped)
        cached = self._pcm_cache.get(tts_text)

        if cached is not None:
            return bytes(cached)

        audio_24k = synthesize_fn(
            pipeline=self._pipeline,
            voice=self._config.voice,
            text=tts_text,
        )
        audio_8k = resample_fn(audio_24k)
        pcm16 = encode_fn(audio_8k)

        if not pcm16:
            raise ValueError("Kokoro telephony rendering produced empty PCM")

        return pcm16

    def _resolved_functions(
        self,
    ) -> tuple[SynthesizeFn, NormalizeFn, ResampleFn, EncodeFn]:
        synthesize_fn = self._synthesize_fn

        if synthesize_fn is None:
            from voiceprobe.autonomous_phone import synthesize

            synthesize_fn = synthesize

        normalize_fn = self._normalize_fn
        resample_fn = self._resample_fn
        encode_fn = self._encode_fn

        if (
            normalize_fn is None
            or resample_fn is None
            or encode_fn is None
        ):
            from voiceprobe.tts.telephony import (
                float_audio_to_pcm16,
                normalize_text_for_tts,
                resample_to_telephony,
            )

            normalize_fn = normalize_fn or normalize_text_for_tts
            resample_fn = resample_fn or resample_to_telephony
            encode_fn = encode_fn or float_audio_to_pcm16

        return (
            synthesize_fn,
            normalize_fn,
            resample_fn,
            encode_fn,
        )


class AudioSocketKokoroSpeechTask:
    """Task-like output target accepted by PipecatRuntimeBridge.

    `queue_frame()` deliberately returns as soon as playback has been scheduled.
    Rendering and paced AudioSocket transmission happen in background threads,
    keeping the asyncio loop available to Flux.
    """

    def __init__(
        self,
        *,
        connection: Any,
        renderer: KokoroTelephonyRenderer,
        send_lock: threading.Lock,
        recorder: RecorderLike | None = None,
        config: AudioSocketKokoroConfig = AudioSocketKokoroConfig(),
        send_audio_fn: SendAudioFn | None = None,
        on_playback_finished: PlaybackFinishedFn | None = None,
    ) -> None:
        config.validate()

        self._connection = connection
        self._renderer = renderer
        self._send_lock = send_lock
        self._recorder = recorder
        self._config = config
        self._send_audio_fn = send_audio_fn
        self._on_playback_finished = on_playback_finished

        self._playback_active = threading.Event()
        self._playback_task: asyncio.Task[None] | None = None
        self._last_error: BaseException | None = None
        self._queued_count = 0

    @property
    def playback_active(self) -> threading.Event:
        return self._playback_active

    @property
    def queued_count(self) -> int:
        return self._queued_count

    @property
    def last_error(self) -> BaseException | None:
        return self._last_error

    @property
    def busy(self) -> bool:
        task = self._playback_task
        return task is not None and not task.done()

    def set_on_playback_finished(
        self,
        callback: PlaybackFinishedFn,
    ) -> None:
        self._on_playback_finished = callback

    async def queue_frames(self, frames: list[Any]) -> None:
        if len(frames) != 1:
            raise ValueError(
                "AudioSocket Kokoro speech sink accepts exactly one speech frame"
            )

        await self.queue_frame(frames[0])

    async def queue_frame(self, frame: Any) -> None:
        text = getattr(frame, "text", None)

        if not isinstance(text, str) or not text.strip():
            raise TypeError(
                "AudioSocketKokoroSpeechTask expects a frame with non-empty text"
            )

        if self.busy:
            raise RuntimeError(
                "A second speech frame was queued before prior playback finished"
            )

        self._last_error = None
        self._playback_active.set()
        self._queued_count += 1
        self._playback_task = asyncio.create_task(
            self._render_and_play(text.strip())
        )

    async def wait_for_idle(self) -> None:
        while True:
            task = self._playback_task

            if task is None:
                break

            await task

            if self._last_error is not None:
                raise self._last_error

            if self._playback_task is task:
                self._playback_task = None

        if self._last_error is not None:
            raise self._last_error

    async def _render_and_play(self, text: str) -> None:
        success = False

        try:
            self._record_event(
                "v3_playback_preparing",
                text=text,
            )

            pcm16 = await asyncio.to_thread(
                self._renderer.render,
                text,
            )

            self._record_event(
                "v3_playback_started",
                text=text,
                pcm_bytes=len(pcm16),
            )

            await asyncio.to_thread(
                self._send_audio,
                pcm16,
            )

            self._record_event(
                "v3_audio_sent",
                text=text,
                pcm_bytes=len(pcm16),
            )

            if self._config.echo_guard_seconds:
                await asyncio.sleep(
                    self._config.echo_guard_seconds
                )

            success = True
        except BaseException as error:
            self._last_error = error
            self._record_event(
                "v3_playback_error",
                text=text,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        finally:
            self._playback_active.clear()

            current_task = asyncio.current_task()

            if self._playback_task is current_task:
                self._playback_task = None

        if not success:
            return

        self._record_event(
            "v3_playback_finished",
            text=text,
        )

        callback = self._on_playback_finished

        if callback is not None:
            maybe = callback()

            if inspect.isawaitable(maybe):
                await maybe

    def _send_audio(self, pcm16: bytes) -> None:
        send_audio_fn = self._send_audio_fn

        if send_audio_fn is None:
            from voiceprobe.autonomous_phone import (
                send_audio_synchronized,
            )

            send_audio_fn = send_audio_synchronized

        send_audio_fn(
            self._connection,
            pcm16,
            send_lock=self._send_lock,
            recorder=self._recorder,
        )

    def _record_event(
        self,
        event_name: str,
        **fields: object,
    ) -> None:
        if self._recorder is not None:
            self._recorder.record_event(
                event_name,
                **fields,
            )


class AudioSocketV3MediaBoundary:
    """Own the shared AudioSocket media lock and input mute boundary."""

    def __init__(
        self,
        *,
        connection: Any,
        speech_task: AudioSocketKokoroSpeechTask,
        send_lock: threading.Lock,
        recorder: RecorderLike | None = None,
        idle_silence_fn: IdleSilenceFn | None = None,
    ) -> None:
        self._connection = connection
        self._speech_task = speech_task
        self._send_lock = send_lock
        self._recorder = recorder
        self._idle_silence_fn = idle_silence_fn
        self._idle_thread: threading.Thread | None = None

    @property
    def speech_task(self) -> AudioSocketKokoroSpeechTask:
        return self._speech_task

    def start_idle_silence(
        self,
        *,
        stop: threading.Event,
    ) -> threading.Thread:
        if self._idle_thread is not None:
            raise RuntimeError("Idle AudioSocket media has already been started")

        idle_silence_fn = self._idle_silence_fn

        if idle_silence_fn is None:
            from voiceprobe.autonomous_phone import (
                send_idle_silence,
            )

            idle_silence_fn = send_idle_silence

        thread = threading.Thread(
            target=idle_silence_fn,
            kwargs={
                "connection": self._connection,
                "stop": stop,
                "send_lock": self._send_lock,
            },
            name="voiceprobe-v3-idle-silence",
            daemon=True,
        )
        thread.start()
        self._idle_thread = thread
        return thread

    def forward_inbound_pcm(
        self,
        payload: bytes,
        *,
        submit_pcm: SubmitPCMFn,
    ) -> bool:
        """Record all inbound PCM and forward only outside playback/echo guard."""

        if not payload:
            return False

        if self._recorder is not None:
            self._recorder.record_inbound_pcm(payload)

        if self._speech_task.playback_active.is_set():
            return False

        result = submit_pcm(payload)

        if inspect.isawaitable(result):
            raise TypeError(
                "submit_pcm must be thread-safe synchronous handoff; "
                "schedule coroutine work inside the callback"
            )

        return True

    def join_idle_silence(
        self,
        *,
        timeout: float = 1.0,
    ) -> None:
        thread = self._idle_thread

        if thread is not None:
            thread.join(timeout=timeout)
