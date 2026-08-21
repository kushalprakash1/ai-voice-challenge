import asyncio
import threading
import time
from dataclasses import dataclass

import pytest

from voiceprobe.v3.audiosocket_kokoro import (
    AudioSocketKokoroConfig,
    AudioSocketKokoroSpeechTask,
    AudioSocketV3MediaBoundary,
    KokoroTelephonyRenderer,
)


@dataclass
class FakeFrame:
    text: str


class FakeRecorder:
    def __init__(self) -> None:
        self.events = []
        self.inbound = []

    def record_event(self, event_name, **fields):
        self.events.append((event_name, fields))

    def record_inbound_pcm(self, payload):
        self.inbound.append(payload)


def make_renderer(*, cache=None):
    calls = []

    def synthesize_fn(*, pipeline, voice, text):
        calls.append(("synthesize", pipeline, voice, text))
        return [1.0, 2.0]

    def normalize_fn(text):
        calls.append(("normalize", text))
        return text.replace("9:45 a.m.", "9:45 AM")

    def resample_fn(audio):
        calls.append(("resample", tuple(audio)))
        return [3.0]

    def encode_fn(audio):
        calls.append(("encode", tuple(audio)))
        return b"pcm"

    renderer = KokoroTelephonyRenderer(
        pipeline="kokoro",
        pcm_cache=cache,
        synthesize_fn=synthesize_fn,
        normalize_fn=normalize_fn,
        resample_fn=resample_fn,
        encode_fn=encode_fn,
    )
    return renderer, calls


def test_renderer_preserves_existing_kokoro_telephony_contract() -> None:
    renderer, calls = make_renderer()

    pcm = renderer.render("Meet at 9:45 a.m.")

    assert pcm == b"pcm"
    assert calls == [
        ("normalize", "Meet at 9:45 a.m."),
        ("synthesize", "kokoro", "af_heart", "Meet at 9:45 AM"),
        ("resample", (1.0, 2.0)),
        ("encode", (3.0,)),
    ]


def test_renderer_uses_pre_rendered_pcm_without_kokoro() -> None:
    renderer, calls = make_renderer(
        cache={"First available is fine.": b"cached"},
    )

    pcm = renderer.render("First available is fine.")

    assert pcm == b"cached"
    assert calls == [("normalize", "First available is fine.")]


def test_config_freezes_8khz_and_existing_echo_guard() -> None:
    config = AudioSocketKokoroConfig()

    assert config.telephony_sample_rate == 8000
    assert config.echo_guard_seconds == 0.35

    with pytest.raises(ValueError):
        AudioSocketKokoroConfig(
            telephony_sample_rate=16000,
        ).validate()


def test_queue_frame_returns_before_background_playback_finishes() -> None:
    async def scenario():
        renderer, _ = make_renderer()
        gate = threading.Event()
        sent = []

        def slow_send(
            connection,
            pcm16,
            *,
            send_lock,
            recorder,
        ):
            del connection, send_lock, recorder
            gate.wait(timeout=2)
            sent.append(pcm16)

        speech = AudioSocketKokoroSpeechTask(
            connection=object(),
            renderer=renderer,
            send_lock=threading.Lock(),
            config=AudioSocketKokoroConfig(
                echo_guard_seconds=0,
            ),
            send_audio_fn=slow_send,
        )

        started = time.perf_counter()
        await speech.queue_frame(FakeFrame("Hello"))
        elapsed = time.perf_counter() - started

        assert elapsed < 0.2
        assert speech.playback_active.is_set()
        assert sent == []

        gate.set()
        await speech.wait_for_idle(timeout_seconds=2)

        assert sent == [b"pcm"]
        assert not speech.playback_active.is_set()

    asyncio.run(scenario())


def test_playback_finish_callback_runs_after_echo_guard() -> None:
    async def scenario():
        renderer, _ = make_renderer()
        sequence = []

        def send_audio(
            connection,
            pcm16,
            *,
            send_lock,
            recorder,
        ):
            del connection, pcm16, send_lock, recorder
            sequence.append("sent")

        async def finished():
            sequence.append("finished")

        speech = AudioSocketKokoroSpeechTask(
            connection=object(),
            renderer=renderer,
            send_lock=threading.Lock(),
            config=AudioSocketKokoroConfig(
                echo_guard_seconds=0.01,
            ),
            send_audio_fn=send_audio,
            on_playback_finished=finished,
        )

        await speech.queue_frame(FakeFrame("Hello"))
        await speech.wait_for_idle(timeout_seconds=2)

        assert sequence == ["sent", "finished"]

    asyncio.run(scenario())


def test_failed_playback_does_not_release_runtime_callback() -> None:
    async def scenario():
        renderer, _ = make_renderer()
        callbacks = []

        def broken_send(*args, **kwargs):
            del args, kwargs
            raise BrokenPipeError("synthetic disconnect")

        speech = AudioSocketKokoroSpeechTask(
            connection=object(),
            renderer=renderer,
            send_lock=threading.Lock(),
            config=AudioSocketKokoroConfig(
                echo_guard_seconds=0,
            ),
            send_audio_fn=broken_send,
            on_playback_finished=lambda: callbacks.append("finished"),
        )

        await speech.queue_frame(FakeFrame("Hello"))

        with pytest.raises(BrokenPipeError):
            await speech.wait_for_idle(timeout_seconds=2)

        assert callbacks == []
        assert not speech.playback_active.is_set()

    asyncio.run(scenario())


def test_media_boundary_records_everything_but_mutes_flux_during_playback() -> None:
    async def scenario():
        renderer, _ = make_renderer()
        gate = threading.Event()
        recorder = FakeRecorder()
        forwarded = []

        def slow_send(
            connection,
            pcm16,
            *,
            send_lock,
            recorder,
        ):
            del connection, pcm16, send_lock, recorder
            gate.wait(timeout=2)

        lock = threading.Lock()
        speech = AudioSocketKokoroSpeechTask(
            connection=object(),
            renderer=renderer,
            send_lock=lock,
            recorder=recorder,
            config=AudioSocketKokoroConfig(
                echo_guard_seconds=0,
            ),
            send_audio_fn=slow_send,
        )
        boundary = AudioSocketV3MediaBoundary(
            connection=object(),
            speech_task=speech,
            send_lock=lock,
            recorder=recorder,
            idle_silence_fn=lambda **kwargs: None,
        )

        assert boundary.forward_inbound_pcm(
            b"before",
            submit_pcm=forwarded.append,
        )

        await speech.queue_frame(FakeFrame("Hello"))

        assert boundary.forward_inbound_pcm(
            b"during",
            submit_pcm=forwarded.append,
        ) is False

        gate.set()
        await speech.wait_for_idle(timeout_seconds=2)

        assert boundary.forward_inbound_pcm(
            b"after",
            submit_pcm=forwarded.append,
        )

        assert recorder.inbound == [
            b"before",
            b"during",
            b"after",
        ]
        assert forwarded == [
            b"before",
            b"after",
        ]

    asyncio.run(scenario())


def test_idle_silence_uses_same_shared_send_lock() -> None:
    seen = []
    stop = threading.Event()
    lock = threading.Lock()

    renderer, _ = make_renderer()
    speech = AudioSocketKokoroSpeechTask(
        connection=object(),
        renderer=renderer,
        send_lock=lock,
        config=AudioSocketKokoroConfig(
            echo_guard_seconds=0,
        ),
        send_audio_fn=lambda *args, **kwargs: None,
    )

    def idle_silence_fn(
        connection,
        *,
        stop,
        send_lock,
    ):
        del connection
        seen.append(send_lock is lock)
        stop.set()

    boundary = AudioSocketV3MediaBoundary(
        connection=object(),
        speech_task=speech,
        send_lock=lock,
        idle_silence_fn=idle_silence_fn,
    )

    thread = boundary.start_idle_silence(stop=stop)
    thread.join(timeout=1)

    assert seen == [True]
def test_playback_completion_can_chain_buffered_runtime_response() -> None:
    from dataclasses import replace

    from voiceprobe.v3.production import (
        DEFAULT_PRODUCTION_FLUX_CONFIG,
        PipecatRuntimeBridge,
    )

    class FakeFrame:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeFlux:
        def __init__(self) -> None:
            self.handlers = {}

        def event_handler(self, name):
            def decorator(func):
                self.handlers[name] = func
                return func

            return decorator

    async def scenario():
        sent = []

        def send_audio(
            connection,
            pcm16,
            *,
            send_lock,
            recorder,
        ):
            del connection, send_lock, recorder
            sent.append(pcm16)

        renderer = KokoroTelephonyRenderer(
            pipeline=object(),
            synthesize_fn=lambda **kwargs: [0.5],
            normalize_fn=lambda text: text,
            resample_fn=lambda audio: audio,
            encode_fn=lambda audio: b"pcm",
        )

        speech = AudioSocketKokoroSpeechTask(
            connection=object(),
            renderer=renderer,
            send_lock=threading.Lock(),
            config=AudioSocketKokoroConfig(
                echo_guard_seconds=0,
            ),
            send_audio_fn=send_audio,
        )

        config = replace(
            DEFAULT_PRODUCTION_FLUX_CONFIG,
            continuation_grace_ms=0,
        )
        bridge = PipecatRuntimeBridge(
            config=config,
            tts_frame_factory=FakeFrame,
        )
        bridge.bind_frame_sink(speech)
        speech.set_on_playback_finished(
            bridge.on_tts_stopped,
        )

        flux = FakeFlux()
        bridge.attach_flux(flux)

        first = await bridge.runtime.process_turns(
            ["What is the reason for your visit?"]
        )
        assert first.response_ready
        assert speech.queued_count == 1

        await flux.handlers["on_end_of_turn"](
            flux,
            (
                "We have Friday afternoon openings with two providers. "
                "Would you prefer either provider or is first available okay?"
            ),
        )

        assert bridge.runtime.ingress.burst_buffer.pending_turns
        assert speech.queued_count == 1

        await speech.wait_for_idle(timeout_seconds=2)

        assert sent == [b"pcm", b"pcm"]
        assert speech.queued_count == 2
        assert speech.last_error is None
        assert speech.busy is False
        assert bridge.runtime.ingress.burst_buffer.response_busy is False

    asyncio.run(scenario())
