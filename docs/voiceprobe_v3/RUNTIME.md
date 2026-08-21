# VoiceProbe v3 offline runtime

The runtime is the first single object that joins:

```text
Flux turn events
    -> burst ingress
    -> conversation coalescing
    -> deterministic fast policy
    -> structured scheduling state
    -> optional focused fallback
    -> decision metrics
```

## Why the runtime is transport-agnostic

The same `VoiceProbeV3Runtime` can be driven by:

1. real Pipecat/Deepgram Flux `on_end_of_turn` events;
2. stored transcript turns from failed calls;
3. future recorded-audio replay after audio is passed through Flux.

This keeps live telephony out of the debugging loop.

## Fallback boundary

Routine turns never invoke an LLM.

Only `DecisionKind.FALLBACK` may call an injected `fallback_resolver`. The
resolver receives:

- the latest actionable remote turn;
- an immutable flow snapshot.

It returns a structured `PolicyDecision`. It cannot directly mutate patient
facts or flow state.

## Metrics

Every emitted decision records deterministic-policy latency separately from
future STT/TTS/network timing.

Run the stored transcript runtime replay:

```bash
PYTHONPATH=src \
.venv/bin/python \
  tools/v3_replay_runtime.py
```

This is still text replay. The next gate is recorded-audio replay.
