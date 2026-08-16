"""Production-facing Pipecat/Flux assembly for VoiceProbe v3.

This module keeps routine reasoning deterministic while providing the narrow
adapter needed by a Pipecat task:

Flux STT events -> VoiceProbeV3Runtime -> TTSSpeakFrame -> TTS lifecycle

Pipecat imports are intentionally lazy. The repository's normal test
environment does not need Pipecat installed; the separate voiceprobe-v3
environment verifies the real integration.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from .runtime import RuntimeDecision, VoiceProbeV3Runtime
from .turn_stabilizer import DEFAULT_CONTINUATION_GRACE_MS


DEFAULT_KEYTERMS = (
    "Pivot Point",
    "Alex Morgan",
    "Blue Cross",
    "new patient consultation",
)


@dataclass(frozen=True, slots=True)
class ProductionFluxConfig:
    model: str = "flux-general-en"
    sample_rate: int = 8000
    eot_threshold: float = 0.85
    eot_timeout_ms: int = 5000
    eager_eot_threshold: float | None = None
    continuation_grace_ms: float = DEFAULT_CONTINUATION_GRACE_MS
    keyterms: tuple[str, ...] = DEFAULT_KEYTERMS
    flux_encoding: str = "linear16"

    def validate(self) -> None:
        if self.model != "flux-general-en":
            raise ValueError("VoiceProbe v3 production model must be flux-general-en")
        if self.sample_rate != 8000:
            raise ValueError("VoiceProbe v3 production input must remain native 8 kHz")
        if self.flux_encoding != "linear16":
            raise ValueError("Deepgram Flux production encoding must be linear16")
        if not 0.0 < self.eot_threshold <= 1.0:
            raise ValueError("eot_threshold must be in (0, 1]")
        if self.eot_timeout_ms <= 0:
            raise ValueError("eot_timeout_ms must be positive")
        if self.eager_eot_threshold is not None:
            raise ValueError("EagerEndOfTurn remains disabled for the current production gate")
        if self.continuation_grace_ms < 0:
            raise ValueError("continuation_grace_ms must be non-negative")
        if not self.keyterms:
            raise ValueError("At least one production keyterm is required")


DEFAULT_PRODUCTION_FLUX_CONFIG = ProductionFluxConfig()


@dataclass(frozen=True, slots=True)
class ProductionFluxBundle:
    service: Any
    config: ProductionFluxConfig


@dataclass(frozen=True, slots=True)
class ProductionPipelineBundle:
    pipeline: Any
    task: Any
    lifecycle_processor: Any


def build_production_flux_service(
    *,
    api_key: str,
    config: ProductionFluxConfig = DEFAULT_PRODUCTION_FLUX_CONFIG,
) -> ProductionFluxBundle:
    """Instantiate the exact Deepgram Flux service used by VoiceProbe v3.

    Instantiation does not start the Pipecat pipeline or open the websocket.
    """

    if not api_key.strip():
        raise ValueError("Deepgram api_key is required")

    config.validate()

    from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService

    service = DeepgramFluxSTTService(
        api_key=api_key,
        sample_rate=config.sample_rate,
        flux_encoding=config.flux_encoding,
        settings=DeepgramFluxSTTService.Settings(
            model=config.model,
            eager_eot_threshold=config.eager_eot_threshold,
            eot_threshold=config.eot_threshold,
            eot_timeout_ms=config.eot_timeout_ms,
            keyterm=list(config.keyterms),
        ),
    )

    return ProductionFluxBundle(
        service=service,
        config=config,
    )


class PipecatRuntimeBridge:
    """Connect VoiceProbe runtime decisions to a Pipecat PipelineTask."""

    def __init__(
        self,
        *,
        config: ProductionFluxConfig = DEFAULT_PRODUCTION_FLUX_CONFIG,
        tts_frame_factory: Callable[[str], Any] | None = None,
    ) -> None:
        config.validate()

        self._config = config
        self._task: Any | None = None
        self._queued_speech_count = 0
        self._tts_frame_factory = tts_frame_factory or _default_tts_frame_factory

        self._runtime = VoiceProbeV3Runtime(
            on_decision=self._on_runtime_decision,
            continuation_grace_ms=config.continuation_grace_ms,
        )

    @property
    def config(self) -> ProductionFluxConfig:
        return self._config

    @property
    def runtime(self) -> VoiceProbeV3Runtime:
        return self._runtime

    @property
    def queued_speech_count(self) -> int:
        return self._queued_speech_count

    @property
    def task_bound(self) -> bool:
        return self._task is not None

    def bind_task(self, task: Any) -> None:
        if not hasattr(task, "queue_frame"):
            raise TypeError("Pipecat task must provide queue_frame(frame)")
        if self._task is not None and self._task is not task:
            raise RuntimeError("PipecatRuntimeBridge is already bound to another task")
        self._task = task

    def attach_flux(self, stt_service: Any) -> None:
        self._runtime.attach_flux(stt_service)

    async def _on_runtime_decision(
        self,
        result: RuntimeDecision,
    ) -> None:
        if not result.response_ready:
            return

        if self._task is None:
            raise RuntimeError(
                "A response became ready before the Pipecat task was bound"
            )

        # Busy begins before TTS synthesis is queued, so any remote speech
        # arriving during synthesis/playback enters the existing burst buffer.
        self._runtime.mark_response_started()

        frame = self._tts_frame_factory(result.decision.text)
        maybe = self._task.queue_frame(frame)

        if inspect.isawaitable(maybe):
            await maybe

        self._queued_speech_count += 1

    async def on_tts_stopped(self) -> None:
        """Release response-busy state only after Pipecat reports TTS stopped."""

        await self._runtime.mark_response_finished()

    def clear_pending(self) -> tuple[str, ...]:
        return self._runtime.ingress.clear_pending()


def _default_tts_frame_factory(text: str) -> Any:
    from pipecat.frames.frames import TTSSpeakFrame

    return TTSSpeakFrame(
        text=text,
        append_to_context=False,
    )


def build_tts_lifecycle_processor(
    bridge: PipecatRuntimeBridge,
) -> Any:
    """Create a Pipecat processor that releases busy state on TTSStoppedFrame."""

    from pipecat.frames.frames import TTSStoppedFrame
    from pipecat.processors.frame_processor import (
        FrameDirection,
        FrameProcessor,
    )

    class VoiceProbeTTSLifecycleProcessor(FrameProcessor):
        def __init__(self) -> None:
            super().__init__(name="VoiceProbeTTSLifecycleProcessor")

        async def process_frame(
            self,
            frame: Any,
            direction: FrameDirection,
        ) -> None:
            await super().process_frame(frame, direction)

            if isinstance(frame, TTSStoppedFrame):
                await bridge.on_tts_stopped()

            await self.push_frame(frame, direction)

    return VoiceProbeTTSLifecycleProcessor()


def build_production_pipeline_task(
    *,
    transport: Any,
    stt_service: Any,
    tts_service: Any,
    bridge: PipecatRuntimeBridge,
    enable_metrics: bool = True,
    enable_usage_metrics: bool = True,
) -> ProductionPipelineBundle:
    """Build the minimal deterministic Pipecat pipeline for VoiceProbe v3."""

    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineParams, PipelineTask

    lifecycle = build_tts_lifecycle_processor(bridge)

    pipeline = Pipeline(
        [
            transport.input(),
            stt_service,
            tts_service,
            lifecycle,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=bridge.config.sample_rate,
            audio_out_sample_rate=bridge.config.sample_rate,
            enable_metrics=enable_metrics,
            enable_usage_metrics=enable_usage_metrics,
        ),
    )

    bridge.bind_task(task)
    bridge.attach_flux(stt_service)

    return ProductionPipelineBundle(
        pipeline=pipeline,
        task=task,
        lifecycle_processor=lifecycle,
    )
