#!/usr/bin/env python3
"""Verify Pipecat 1.7 audio injection contracts without network I/O."""

from __future__ import annotations

import inspect

from pipecat.frames.frames import InputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker

from voiceprobe.v3.production import build_production_flux_service


def main() -> None:
    bundle = build_production_flux_service(
        api_key="offline-placeholder",
    )

    frame = InputAudioRawFrame(
        audio=bytes(320),
        sample_rate=8000,
        num_channels=1,
    )

    pipeline = Pipeline([bundle.service])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    print("InputAudioRawFrame instantiated: PASS")
    print(" audio_bytes:", len(frame.audio))
    print(" sample_rate:", frame.sample_rate)
    print(" num_channels:", frame.num_channels)
    print("Pipeline([DeepgramFluxSTTService]) instantiated: PASS")
    print("PipelineWorker instantiated: PASS")
    print(
        "PipelineWorker.queue_frames is coroutine:",
        inspect.iscoroutinefunction(worker.queue_frames),
    )
    print("No PipelineRunner started; no Deepgram websocket was opened.")


if __name__ == "__main__":
    main()
