#!/usr/bin/env python3
"""Offline simulation of the production decision-to-TTS bridge."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from voiceprobe.v3.production import (
    DEFAULT_PRODUCTION_FLUX_CONFIG,
    PipecatRuntimeBridge,
)


class FakeFrame:
    def __init__(self, text: str):
        self.text = text


class FakeTask:
    def __init__(self):
        self.frames = []

    async def queue_frame(self, frame):
        self.frames.append(frame)
        print("TTS QUEUED:", frame.text)


class FakeFlux:
    def __init__(self):
        self.handlers = {}

    def event_handler(self, name):
        def decorator(func):
            self.handlers[name] = func
            return func
        return decorator


async def main() -> None:
    # Keep production semantics but use a long test grace so the simulation
    # manually controls when the stabilized burst is released.
    config = replace(
        DEFAULT_PRODUCTION_FLUX_CONFIG,
        continuation_grace_ms=60_000,
    )

    bridge = PipecatRuntimeBridge(
        config=config,
        tts_frame_factory=FakeFrame,
    )
    task = FakeTask()
    stt = FakeFlux()

    bridge.bind_task(task)
    bridge.attach_flux(stt)

    first = (
        "We have openings for new patient consultation on Friday, "
        "August twenty first. The available times are nine AM, "
        "nine forty five AM, and ten thirty AM. "
        "Would any of these work for your Friday afternoon?"
    )
    continuation = (
        "preference, or would you like to look at later dates or times?"
    )

    print("REMOTE EOT:", first)
    await stt.handlers["on_end_of_turn"](stt, first)
    print("REMOTE START within grace: preference")
    await stt.handlers["on_start_of_turn"](stt, "preference")
    print("REMOTE EOT:", continuation)
    await stt.handlers["on_end_of_turn"](stt, continuation)

    print("Queued before stabilization flush:", len(task.frames))
    await bridge.runtime.ingress.flush_stabilized_pending()

    assert len(task.frames) == 1
    assert task.frames[0].text.startswith("Those times don't work")

    print("Simulated TTS stopped.")
    await bridge.on_tts_stopped()

    final = (
        "There are no Friday afternoon openings on August twenty first. "
        "Would you like to look at afternoon times on the following Friday, "
        "August twenty eighth, or check other days next week?"
    )

    print("REMOTE EOT:", final)
    await stt.handlers["on_end_of_turn"](stt, final)
    await bridge.runtime.ingress.flush_stabilized_pending()

    assert len(task.frames) == 2
    assert "August 28th" in task.frames[1].text

    print()
    print("PRODUCTION DECISION -> TTS BRIDGE SIMULATION: PASS")


if __name__ == "__main__":
    asyncio.run(main())
