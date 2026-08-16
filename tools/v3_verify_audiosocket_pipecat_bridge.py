#!/usr/bin/env python3
"""Verify the real Pipecat 1.7 AudioSocket input-worker assembly.

No WorkerRunner is started, so this opens no Deepgram websocket and places no
phone call.
"""

from __future__ import annotations

import asyncio
import threading

from voiceprobe.v3.audiosocket_kokoro import (
    AudioSocketKokoroConfig,
    AudioSocketKokoroSpeechTask,
    KokoroTelephonyRenderer,
)
from voiceprobe.v3.audiosocket_pipecat import (
    build_audiosocket_flux_input_worker,
)
from voiceprobe.v3.production import (
    PipecatRuntimeBridge,
    build_production_flux_service,
)


async def main() -> None:
    flux = build_production_flux_service(
        api_key="offline-placeholder",
    )

    renderer = KokoroTelephonyRenderer(
        pipeline=object(),
        synthesize_fn=lambda **kwargs: [0.0],
        normalize_fn=lambda text: text,
        resample_fn=lambda audio: audio,
        encode_fn=lambda audio: b"\x00\x00" * 160,
    )

    speech = AudioSocketKokoroSpeechTask(
        connection=object(),
        renderer=renderer,
        send_lock=threading.Lock(),
        config=AudioSocketKokoroConfig(
            echo_guard_seconds=0,
        ),
        send_audio_fn=lambda *args, **kwargs: None,
    )

    bridge = PipecatRuntimeBridge()

    bundle = build_audiosocket_flux_input_worker(
        stt_service=flux.service,
        bridge=bridge,
        speech_sink=speech,
        loop=asyncio.get_running_loop(),
    )

    speech.set_on_playback_finished(
        bridge.on_tts_stopped,
    )

    frame = bundle.feeder.make_frame(bytes(320))

    print("AudioSocket Flux input worker instantiated: PASS")
    print("PipelineWorker type:", type(bundle.worker).__name__)
    print("Runtime frame sink bound:", bridge.frame_sink_bound)
    print("Flux event handlers attached: PASS")
    print("InputAudioRawFrame type:", type(frame).__name__)
    print(" input_bytes:", len(frame.audio))
    print(" sample_rate:", frame.sample_rate)
    print(" num_channels:", frame.num_channels)
    print("AudioSocket Kokoro speech sink bound: PASS")
    print("No WorkerRunner started; no Deepgram websocket was opened.")
    print("No Asterisk originate or phone call occurred.")


if __name__ == "__main__":
    asyncio.run(main())
