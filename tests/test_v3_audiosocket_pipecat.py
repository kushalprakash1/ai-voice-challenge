import asyncio
import threading
from dataclasses import replace

import pytest

from voiceprobe.v3.audiosocket_kokoro import (
    AudioSocketKokoroConfig,
    AudioSocketKokoroSpeechTask,
    KokoroTelephonyRenderer,
)
from voiceprobe.v3.audiosocket_pipecat import PipecatPCMFeeder
from voiceprobe.v3.production import (
    DEFAULT_PRODUCTION_FLUX_CONFIG,
    PipecatRuntimeBridge,
)


class FakeFrame:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeAudioFrame:
    def __init__(
        self,
        audio: bytes,
        sample_rate: int,
        num_channels: int,
    ) -> None:
        self.audio = audio
        self.sample_rate = sample_rate
        self.num_channels = num_channels


class FakeWorker:
    def __init__(self) -> None:
        self.frames = []

    async def queue_frames(self, frames) -> None:
        self.frames.extend(frames)


def fake_renderer() -> KokoroTelephonyRenderer:
    return KokoroTelephonyRenderer(
        pipeline=object(),
        synthesize_fn=lambda **kwargs: [0.5],
        normalize_fn=lambda text: text,
        resample_fn=lambda audio: audio,
        encode_fn=lambda audio: b"pcm",
    )


def test_pcm_feeder_builds_native_8khz_mono_frame() -> None:
    async def scenario():
        worker = FakeWorker()
        feeder = PipecatPCMFeeder(
            worker=worker,
            loop=asyncio.get_running_loop(),
            frame_factory=FakeAudioFrame,
        )

        await feeder.queue_pcm(bytes(320))

        assert len(worker.frames) == 1
        frame = worker.frames[0]
        assert len(frame.audio) == 320
        assert frame.sample_rate == 8000
        assert frame.num_channels == 1

    asyncio.run(scenario())


def test_pcm_feeder_rejects_partial_pcm16_sample() -> None:
    async def scenario():
        feeder = PipecatPCMFeeder(
            worker=FakeWorker(),
            loop=asyncio.get_running_loop(),
            frame_factory=FakeAudioFrame,
        )

        with pytest.raises(ValueError):
            feeder.make_frame(b"\x00")

    asyncio.run(scenario())


def test_submit_pcm_is_thread_safe_handoff() -> None:
    async def scenario():
        worker = FakeWorker()
        feeder = PipecatPCMFeeder(
            worker=worker,
            loop=asyncio.get_running_loop(),
            frame_factory=FakeAudioFrame,
        )

        submitted = []
        completed = threading.Event()

        def submit() -> None:
            try:
                submitted.append(feeder.submit_pcm(bytes(320)))
            finally:
                completed.set()

        thread = threading.Thread(target=submit)
        thread.start()
        async with asyncio.timeout(1):
            while not completed.is_set():
                await asyncio.sleep(0.001)
        thread.join(timeout=1)
        assert not thread.is_alive()

        future = submitted[0]
        await asyncio.wrap_future(future)

        assert len(worker.frames) == 1
        assert len(worker.frames[0].audio) == 320

    asyncio.run(scenario())


def test_runtime_bridge_can_bind_generic_frame_sink() -> None:
    bridge = PipecatRuntimeBridge(
        tts_frame_factory=FakeFrame,
    )
    sink = FakeWorker()

    bridge.bind_frame_sink(sink)

    assert bridge.frame_sink_bound
    assert bridge.worker_bound


def test_runtime_decision_reaches_kokoro_speech_sink_and_releases_busy() -> None:
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

        speech = AudioSocketKokoroSpeechTask(
            connection=object(),
            renderer=fake_renderer(),
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

        result = await bridge.runtime.process_turns(
            ["What is the reason for your visit?"]
        )

        assert result.response_ready
        assert speech.queued_count == 1

        await speech.wait_for_idle(timeout_seconds=2)

        assert sent == [b"pcm"]
        assert bridge.runtime.ingress.burst_buffer.response_busy is False

    asyncio.run(scenario())


def test_speech_task_queue_frames_accepts_single_runtime_frame() -> None:
    async def scenario():
        speech = AudioSocketKokoroSpeechTask(
            connection=object(),
            renderer=fake_renderer(),
            send_lock=threading.Lock(),
            config=AudioSocketKokoroConfig(
                echo_guard_seconds=0,
            ),
            send_audio_fn=lambda *args, **kwargs: None,
        )

        await speech.queue_frames(
            [FakeFrame("First available is fine.")]
        )
        await speech.wait_for_idle(timeout_seconds=2)

        assert speech.queued_count == 1

    asyncio.run(scenario())


def test_speech_task_queue_frames_rejects_batch_speech() -> None:
    async def scenario():
        speech = AudioSocketKokoroSpeechTask(
            connection=object(),
            renderer=fake_renderer(),
            send_lock=threading.Lock(),
            config=AudioSocketKokoroConfig(
                echo_guard_seconds=0,
            ),
            send_audio_fn=lambda *args, **kwargs: None,
        )

        with pytest.raises(ValueError):
            await speech.queue_frames(
                [FakeFrame("one"), FakeFrame("two")]
            )

    asyncio.run(scenario())
