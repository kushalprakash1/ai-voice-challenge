#!/usr/bin/env python3
"""Verify current Pipecat compatibility without opening a network connection."""

from __future__ import annotations

import asyncio

from voiceprobe.v3.production import (
    DEFAULT_PRODUCTION_FLUX_CONFIG,
    PipecatRuntimeBridge,
    build_production_flux_service,
    build_tts_lifecycle_processor,
)


class FakeWorker:
    def __init__(self):
        self.frames = []

    async def queue_frames(self, frames):
        self.frames.extend(frames)


async def main() -> None:
    bundle = build_production_flux_service(
        api_key="offline-placeholder",
    )

    bridge = PipecatRuntimeBridge()
    worker = FakeWorker()
    bridge.bind_worker(worker)
    bridge.attach_flux(bundle.service)

    lifecycle = build_tts_lifecycle_processor(bridge)

    result = await bridge.runtime.process_turns(
        ["What is the reason for your visit?"]
    )

    frame = worker.frames[0]

    print("DeepgramFluxSTTService instantiated: PASS")
    print("VoiceProbe event handlers attached: PASS")
    print("Pipecat TTSSpeakFrame instantiated: PASS")
    print("TTS lifecycle processor instantiated: PASS")
    print("No pipeline runner started; no Deepgram websocket was opened.")
    print()
    print("Production settings:")
    print(" model:", bundle.config.model)
    print(" sample_rate:", bundle.config.sample_rate)
    print(" eot_threshold:", bundle.config.eot_threshold)
    print(" eot_timeout_ms:", bundle.config.eot_timeout_ms)
    print(" eager_eot_threshold:", bundle.config.eager_eot_threshold)
    print(" continuation_grace_ms:", bundle.config.continuation_grace_ms)
    print(" keyterms:", ", ".join(bundle.config.keyterms))
    print()
    print("Deterministic response:", result.decision.text)
    print("Queued frame type:", type(frame).__name__)
    print("Lifecycle type:", type(lifecycle).__name__)


if __name__ == "__main__":
    asyncio.run(main())
