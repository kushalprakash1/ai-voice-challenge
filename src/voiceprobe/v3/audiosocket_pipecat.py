"""AudioSocket <-> Pipecat bridge for VoiceProbe v3.

This module connects the already-proven pieces without owning dialing:

AudioSocket PCM16 -> InputAudioRawFrame -> PipelineWorker -> Deepgram Flux
Deepgram Flux events -> VoiceProbeV3Runtime
Runtime response -> TTSSpeakFrame -> AudioSocketKokoroSpeechTask -> AudioSocket

The synchronous AudioSocket receive loop can submit PCM into an asyncio-owned
PipelineWorker through `PipecatPCMFeeder.submit_pcm()`. That handoff is
thread-safe and does not block the socket receive loop on Deepgram processing.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass
from typing import Any, Callable

from .production import (
    DEFAULT_PRODUCTION_FLUX_CONFIG,
    PipecatRuntimeBridge,
    ProductionFluxConfig,
)


FrameFactory = Callable[[bytes, int, int], Any]


@dataclass(frozen=True, slots=True)
class AudioSocketPipecatBundle:
    worker: Any
    feeder: "PipecatPCMFeeder"
    bridge: PipecatRuntimeBridge
    stt_service: Any


class PipecatPCMFeeder:
    """Thread-safe 8 kHz mono PCM injection into a Pipecat PipelineWorker."""

    def __init__(
        self,
        *,
        worker: Any,
        loop: asyncio.AbstractEventLoop,
        sample_rate: int = 8_000,
        num_channels: int = 1,
        frame_factory: FrameFactory | None = None,
    ) -> None:
        if sample_rate != 8_000:
            raise ValueError("VoiceProbe AudioSocket PCM must remain at 8 kHz")
        if num_channels != 1:
            raise ValueError("VoiceProbe AudioSocket PCM must remain mono")
        if not hasattr(worker, "queue_frames"):
            raise TypeError("Pipecat worker must provide queue_frames(frames)")
        if loop.is_closed():
            raise ValueError("Pipecat event loop is already closed")

        self._worker = worker
        self._loop = loop
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._frame_factory = frame_factory

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return self._num_channels

    def make_frame(self, pcm16: bytes) -> Any:
        payload = bytes(pcm16)

        if not payload:
            raise ValueError("AudioSocket PCM payload cannot be empty")
        if len(payload) % 2:
            raise ValueError("PCM16 payload must contain complete 16-bit samples")

        factory = self._frame_factory

        if factory is not None:
            return factory(
                payload,
                self._sample_rate,
                self._num_channels,
            )

        from pipecat.frames.frames import InputAudioRawFrame

        return InputAudioRawFrame(
            audio=payload,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
        )

    async def queue_pcm(self, pcm16: bytes) -> None:
        frame = self.make_frame(pcm16)
        await self._worker.queue_frames([frame])

    def submit_pcm(
        self,
        pcm16: bytes,
    ) -> concurrent.futures.Future[None]:
        """Schedule PCM from the blocking AudioSocket thread."""

        if self._loop.is_closed():
            raise RuntimeError("Pipecat event loop is closed")

        return asyncio.run_coroutine_threadsafe(
            self.queue_pcm(pcm16),
            self._loop,
        )


def build_audiosocket_flux_input_worker(
    *,
    stt_service: Any,
    bridge: PipecatRuntimeBridge,
    speech_sink: Any,
    loop: asyncio.AbstractEventLoop,
    config: ProductionFluxConfig = DEFAULT_PRODUCTION_FLUX_CONFIG,
    enable_metrics: bool = True,
    enable_usage_metrics: bool = True,
    on_startframe: Callable[[], None] | None = None,
) -> AudioSocketPipecatBundle:
    """Build the Pipecat worker used only for AudioSocket -> Flux input."""

    config.validate()

    if not hasattr(speech_sink, "queue_frames"):
        raise TypeError("speech_sink must provide queue_frames(frames)")

    from pipecat.frames.frames import StartFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class VoiceProbeStartFrameProcessor(FrameProcessor):
        async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)
            if isinstance(frame, StartFrame) and on_startframe is not None:
                on_startframe()
            await self.push_frame(frame, direction)

    # Establish VoiceProbe's pipeline lifecycle before Flux receives StartFrame
    # and begins connecting/emitting metrics.
    pipeline = Pipeline([VoiceProbeStartFrameProcessor(), stt_service])
    worker = PipelineWorker(
        pipeline,
        # This is a server-side AudioSocket pipeline, not an RTVI client.
        # Excluding the unused default processor prevents runner metrics from
        # reaching RTVI before its lifecycle StartFrame.
        enable_rtvi=False,
        params=PipelineParams(
            audio_in_sample_rate=config.sample_rate,
            audio_out_sample_rate=config.sample_rate,
            enable_metrics=enable_metrics,
            enable_usage_metrics=enable_usage_metrics,
        ),
    )

    bridge.bind_frame_sink(speech_sink)
    bridge.attach_flux(stt_service)

    feeder = PipecatPCMFeeder(
        worker=worker,
        loop=loop,
        sample_rate=config.sample_rate,
        num_channels=1,
    )

    return AudioSocketPipecatBundle(
        worker=worker,
        feeder=feeder,
        bridge=bridge,
        stt_service=stt_service,
    )
