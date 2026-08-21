# Production Pipecat / Deepgram Flux assembly

VoiceProbe v3 now has a single production-facing assembly module.

## Frozen production STT settings

- model: `flux-general-en`
- native input sample rate: `8000`
- encoding: `linear16`
- EndOfTurn threshold: `0.85`
- EndOfTurn timeout: `5000 ms`
- EagerEndOfTurn: disabled
- VoiceProbe production continuation grace: `3000 ms`
- keyterms: Pivot Point, Alex Morgan, Blue Cross, new patient consultation

The settings are represented by `ProductionFluxConfig` and validated before a
Deepgram service can be created.

## Deterministic output path

`PipecatRuntimeBridge` is the only object that turns a VoiceProbe runtime
decision into speech.

A response is queued only when `RuntimeDecision.response_ready` is true. WAIT,
HOLD, and unresolved FALLBACK decisions never produce `TTSSpeakFrame`.

Immediately before a speech frame is queued, the bridge marks VoiceProbe's
response path busy. That preserves remote speech arriving during TTS
synthesis/playback. Busy state is released only when the pipeline emits
`TTSStoppedFrame`; any preserved remote burst is then evaluated and may queue
the next speech response.

## Minimal pipeline

`build_production_pipeline_worker()` assembles:

```text
transport.input()
    -> DeepgramFluxSTTService
    -> TTS service
    -> VoiceProbeTTSLifecycleProcessor
    -> transport.output()
```

There is no LLM on the routine hot path.

The function accepts a Pipecat-compatible transport and TTS processor so the
existing telephony and Kokoro migration can be handled separately instead of
changing transport, STT, reasoning, and TTS simultaneously.
