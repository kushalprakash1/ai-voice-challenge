#!/usr/bin/env python3
"""Offline compatibility check against the installed Pipecat/Flux version.

This imports and instantiates DeepgramFluxSTTService but does not start a
pipeline, open a websocket, or contact Deepgram.
"""

from __future__ import annotations

import asyncio

from pipecat.services.deepgram.flux.stt import (
    DeepgramFluxSTTService,
)

from voiceprobe.v3.ingress import (
    FluxIngressController,
)


async def main() -> None:
    emitted = []

    stt = DeepgramFluxSTTService(
        api_key="offline-placeholder",
        settings=DeepgramFluxSTTService.Settings(
            model="flux-general-en",
            eot_threshold=0.8,
            eot_timeout_ms=2500,
            keyterm=[
                "Pivot Point",
                "Alex Morgan",
                "Blue Cross",
                "new patient consultation",
            ],
        ),
    )

    controller = FluxIngressController(
        on_decision=emitted.append,
    )
    controller.attach(stt)

    print("DeepgramFluxSTTService instantiated: PASS")
    print("VoiceProbe event handlers attached: PASS")
    print("No network connection was opened.")
    print()
    print("Registered event API:")
    print("- on_start_of_turn")
    print("- on_turn_resumed")
    print("- on_end_of_turn")


if __name__ == "__main__":
    asyncio.run(main())
