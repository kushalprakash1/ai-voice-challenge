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
from voiceprobe.v3.accent import ACCENT_MELO_INDIA, AccentCache
from voiceprobe.v3.background import (
    background_mode_from_environment,
    background_snr_from_environment,
    mix_background,
)
DEFAULT_VOICE = "af_heart"
TELEPHONY_SAMPLE_RATE = 8_000
ECHO_GUARD_SECONDS = 0.35


class RecorderLike(Protocol):
    def record_event(self, event_name: str, **fields: object) -> None: ...

    def record_inbound_pcm(self, payload: bytes) -> None: ...


SynthesizeFn = Callable[..., Any]
MeloSynthesizeFn = Callable[[str], bytes]
NormalizeFn = Callable[[str], str]
ResampleFn = Callable[[Any], Any]
EncodeFn = Callable[[Any], bytes]
SendAudioFn = Callable[..., None]
IdleSilenceFn = Callable[..., None]
PlaybackFinishedFn = Callable[[], Any]
SubmitPCMFn = Callable[[bytes], Any]


async def _run_in_owned_thread(
    operation: Callable[..., Any],
    *args: Any,
) -> Any:
    """Run one blocking media operation without the shared asyncio executor."""

    completed = threading.Event()
    outcome: list[tuple[bool, Any]] = []

    def run() -> None:
        try:
            outcome.append((True, operation(*args)))
        except BaseException as error:
            outcome.append((False, error))
        finally:
            completed.set()

    worker = threading.Thread(
        target=run,
        name="voiceprobe-v3-media-operation",
    )
    worker.start()

    try:
        while not completed.is_set():
            await asyncio.sleep(0.001)
    finally:
        worker.join()

    succeeded, value = outcome[0]
    if not succeeded:
        raise value
    return value


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
        accent_cache: AccentCache | None = None,
        melo_synthesize_fn: MeloSynthesizeFn | None = None,
    ) -> None:
        config.validate()

        self._pipeline = pipeline
        self._config = config
        self._pcm_cache = dict(pcm_cache or {})

        self._synthesize_fn = synthesize_fn
        self._normalize_fn = normalize_fn
        self._resample_fn = resample_fn
        self._encode_fn = encode_fn
        self._accent_cache = accent_cache
        self._melo_synthesize_fn = melo_synthesize_fn

    @property
    def config(self) -> AudioSocketKokoroConfig:
        return self._config

    def render(self, text: str) -> bytes:
        pcm, _ = self.render_with_metadata(text)
        return pcm

    def render_with_metadata(self, text: str) -> tuple[bytes, dict[str, object]]:
        stripped = text.strip()

        if not stripped:
            raise ValueError("Cannot render empty patient speech")

        if self._accent_cache is not None:
            lookup = self._accent_cache.lookup(stripped)
            korean = self._accent_cache.mode == "chatterbox_korean_heavy"
            same_voice_miss = not lookup.hit and self._accent_cache.mode == ACCENT_MELO_INDIA
            metadata: dict[str, object] = {
                "accent_mode": self._accent_cache.mode,
                "tts_backend": (("chatterbox_multilingual_cache" if korean else "melo_accent_cache") if lookup.hit else ("MeloTTS" if same_voice_miss else "kokoro_fallback")),
                "accent_renderer": "Chatterbox Multilingual/synthetic Korean reference" if korean else "MeloTTS/EN_INDIA",
                "accent_speaker": "synthetic_local_korean_reference_v1" if korean else "EN_INDIA",
                "accent_cache_hit": lookup.hit,
                "accent_cache_key": lookup.cache_key,
                "accent_fallback_used": not lookup.hit and not same_voice_miss,
                "accent_same_voice_miss_rendered": same_voice_miss,
                "accent_cache_lookup_ms": round(lookup.lookup_ms, 6),
                "accent_audio_load_ms": round(lookup.audio_load_ms, 6),
            }
            if lookup.hit and lookup.pcm is not None:
                return self._apply_background(lookup.pcm, metadata)
            metadata["accent_cache_invalid_reason"] = lookup.invalid_reason
            pcm = self._render_melo_india(stripped) if same_voice_miss else self._render_kokoro(stripped)
            return self._apply_background(pcm, metadata)

        return self._apply_background(self._render_kokoro(stripped), {
            "accent_mode": "none",
            "tts_backend": "kokoro",
            "accent_cache_hit": False,
            "accent_cache_key": None,
            "accent_fallback_used": False,
        })

    @staticmethod
    def _apply_background(pcm: bytes, metadata: dict[str, object]) -> tuple[bytes, dict[str, object]]:
        result = mix_background(
            pcm,
            mode=background_mode_from_environment(),
            target_snr_db=background_snr_from_environment(),
        )
        metadata.update(result.metadata)
        return result.pcm, metadata

    def _render_kokoro(self, stripped: str) -> bytes:

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

    def _render_melo_india(self, stripped: str) -> bytes:
        synthesize_fn = self._melo_synthesize_fn
        if synthesize_fn is None:
            from voiceprobe.v3.accent import render_melo_india_pcm

            synthesize_fn = render_melo_india_pcm
        pcm16 = synthesize_fn(stripped)
        if not pcm16:
            raise ValueError("MeloTTS EN_INDIA rendering produced empty PCM")
        return bytes(pcm16)

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
        audio_observer: Callable[[bytes], None] | None = None,
    ) -> None:
        config.validate()

        self._connection = connection
        self._renderer = renderer
        self._send_lock = send_lock
        self._recorder = recorder
        self._config = config
        self._send_audio_fn = send_audio_fn
        self._on_playback_finished = on_playback_finished
        self._audio_observer = audio_observer

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

    async def wait_for_idle(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        loop = asyncio.get_running_loop()
        deadline = (
            None
            if timeout_seconds is None
            else loop.time() + timeout_seconds
        )

        while True:
            task = self._playback_task

            if task is None:
                break

            while not task.done():
                if deadline is not None and loop.time() >= deadline:
                    raise TimeoutError(
                        "AudioSocket playback did not become idle before timeout"
                    )
                await asyncio.sleep(0.001)

            # Retrieve the task result only after observing completion. Direct
            # suspension on this self-clearing background task can strand an
            # awaiter on Python 3.12 even though task.done() is already true.
            task.result()

            if self._last_error is not None:
                raise self._last_error

            if self._playback_task is task:
                self._playback_task = None

        if self._last_error is not None:
            raise self._last_error

    async def _render_and_play(self, text: str) -> None:
        success = False
        render_metadata: dict[str, object] = {}

        try:
            self._record_event(
                "v3_playback_preparing",
                text=text,
            )

            render_with_metadata = getattr(self._renderer, "render_with_metadata", None)
            if render_with_metadata is None:
                pcm16 = await _run_in_owned_thread(self._renderer.render, text)
            else:
                pcm16, render_metadata = await _run_in_owned_thread(render_with_metadata, text)

            self._record_event("v3_playback_prepared", text=text, pcm_bytes=len(pcm16), **render_metadata)
            if render_metadata.get("accent_cache_hit") is False and render_metadata.get("accent_mode") != "none":
                self._record_event(
                    "v3_accent_cache_miss",
                    scenario=self._accent_cache_scenario(),
                    accent=render_metadata.get("accent_mode"),
                    exact_response_text=text,
                    cache_key=render_metadata.get("accent_cache_key"),
                    fallback_used=bool(render_metadata.get("accent_fallback_used")),
                    same_voice_miss_rendered=bool(render_metadata.get("accent_same_voice_miss_rendered")),
                )

            await _run_in_owned_thread(self._send_audio, pcm16)
            self._record_event(
                "v3_audio_sent", text=text, pcm_bytes=len(pcm16), **render_metadata
            )
            if self._config.echo_guard_seconds:
                await asyncio.sleep(self._config.echo_guard_seconds)
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
            **render_metadata,
        )

        callback = self._on_playback_finished
        if callback is not None:
            maybe = callback()

            if inspect.isawaitable(maybe):
                await maybe

    def _accent_cache_scenario(self) -> str | None:
        cache = getattr(self._renderer, "_accent_cache", None)
        return getattr(cache, "scenario", None)

    def _send_audio(self, pcm16: bytes) -> None:
        send_audio_fn = self._send_audio_fn

        if send_audio_fn is None:
            from voiceprobe.autonomous_phone import (
                send_audio_synchronized,
            )

            send_audio_fn = send_audio_synchronized

        observer = self._audio_observer
        if observer is not None:
            try:
                observer(pcm16)
            except Exception:
                pass

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
        audio_observer: Callable[[bytes], None] | None = None,
    ) -> None:
        self._connection = connection
        self._speech_task = speech_task
        self._send_lock = send_lock
        self._recorder = recorder
        self._idle_silence_fn = idle_silence_fn
        self._audio_observer = audio_observer
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

        observer = self._audio_observer

        if observer is not None:
            try:
                # Monitor every raw inbound frame, including genuine remote
                # overlap while Alex is speaking. Never let an observer error
                # alter Flux forwarding or the echo guard.
                observer(payload)
            except Exception:
                pass

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
