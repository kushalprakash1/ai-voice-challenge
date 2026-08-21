# AudioSocket / Pipecat bridge

The v3 production path separates the inbound Pipecat worker from the
already-proven AudioSocket/Kokoro output path.

Inbound:

```text
blocking AudioSocket receive loop
    -> AudioSocketV3MediaBoundary.forward_inbound_pcm()
    -> PipecatPCMFeeder.submit_pcm()
    -> asyncio.run_coroutine_threadsafe(...)
    -> PipelineWorker.queue_frames([InputAudioRawFrame])
    -> DeepgramFluxSTTService
    -> Flux turn events
    -> VoiceProbeV3Runtime
```

Outbound:

```text
VoiceProbeV3Runtime
    -> PipecatRuntimeBridge
    -> TTSSpeakFrame
    -> AudioSocketKokoroSpeechTask.queue_frames()
    -> background Kokoro render
    -> existing 24 kHz -> 8 kHz conversion
    -> existing synchronized AudioSocket sender
    -> 350 ms echo guard
    -> runtime response-busy release
```

The output sink is deliberately not a Pipecat network transport. That avoids
rewriting a stable AudioSocket sender merely to satisfy a transport abstraction.

This phase still does not modify AMI Originate, socket ownership, the Asterisk
call adapter, or place a live call.
