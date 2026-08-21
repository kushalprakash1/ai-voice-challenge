# Architecture

VoiceProbe is a focused calling and evaluation system. Its central design
choice is to separate interpretation from ownership of patient and transaction
state.

## Call path

```text
Asterisk AMI originate
  -> AudioSocket PCM transport
  -> Deepgram Flux STT
  -> FluxIngressCoordinator / RemoteSpeechBurstBuffer
  -> ConversationBurstCoalescer
  -> VoiceProbeV3Runtime / SchedulingFlowController
  -> deterministic policy or semantic fallback
  -> grounded FlowSnapshot
  -> Kokoro or scenario accent renderer
  -> 8 kHz PCM AudioSocket playback
```

`AsteriskAssessmentCallAdapter` owns the one-call telephony lifecycle. It opens
the AudioSocket listener before AMI originate, validates the call UUID, starts
the Pipecat/Flux worker, and projects the final scenario state into a durable
result. `AudioSocketV3MediaBoundary` records all inbound PCM while suppressing
ASR forwarding during local playback and its echo guard.

Flux emits speech events into `FluxIngressCoordinator`. The coordinator gives
short continuations time to arrive and passes stabilized bursts to
`RemoteSpeechBurstBuffer`. `ConversationBurstCoalescer` chooses the actionable
part of a burst instead of replaying acknowledgements and fragments as a FIFO.
`PipecatRuntimeBridge` then routes the result through the scenario runtime and
queues speech only when a response is ready.

## Model and deterministic ownership

The semantic model interprets what the remote agent communicated: its speech
act, requested fact, proposed operation, offered options, and contextual
references. It does not own the patient profile.

Python-owned state decides:

- which patient facts may be spoken;
- scheduling constraints and relaxation policy;
- which target-offered slot is selected;
- whether an offer was actually accepted;
- whether the remote side confirmed a booking; and
- whether the scenario objective is complete.

This boundary uses language models where paraphrase and context make fixed
phrases brittle, while preventing a plausible model completion from becoming a
transaction. Appointment choices must come from observed offers, and booking
completion requires explicit remote evidence.

## Why not one end-to-end realtime model

A single realtime model could reduce integration code and provide naturally
unified speech, turn-taking, and reasoning. The modular path was useful here
because each boundary can be replayed and inspected independently. ASR
thresholds, semantic models, deterministic policy, and TTS can change without
moving transaction ownership. It also permits local Qwen experiments, exact
state assertions, cached rendering, and evidence capture around telephony
events.

The cost is more coordination code. Turn boundaries, playback state, semantic
latency, and scenario state must agree, and behavior outside the modeled
workflows falls back to clarification rather than unconstrained conversation.

## Turn-taking

Flux EndOfTurn events are treated as candidates rather than unquestionable
sentence boundaries. PGAI sometimes continued after a pause that looked final
to ASR. Immediate response could therefore overlap the continuation or answer
only the first clause. Production uses a continuation grace period, invalidates
an authorized response when remote speech resumes, and re-coalesces the whole
burst before deciding again. Speech received while VoiceProbe is rendering or
playing remains buffered until the response path is free.

This adds latency to the first response, but makes fragmented remote turns
reproducible and keeps stale responses from reaching the wire.

## Voice rendering

`KokoroTelephonyRenderer` normalizes text, renders it, resamples it to 8 kHz
PCM16, and optionally mixes configured background audio. Scenario accent caches
are an optimization: a cache hit loads prepared PCM, while a supported cache
miss uses the same scenario renderer and speaker. Cache availability must not
change voice identity. Strict preflight can require all expected phrases before
telephony when a renderer cannot safely synthesize them during a call.

## Evidence and artifacts

`RunArtifactRecorder` stores manifests, event streams, transcripts, metrics,
and aligned inbound/outbound audio under `artifacts/`. These files make timing,
turn ownership, policy decisions, and termination causes inspectable. Raw media
and provider metadata remain local and are gitignored. Public fixtures contain
only synthetic terminal examples and structured regression cases.

## Tradeoffs and limitations

- The scenario policies are intentionally bounded; this is not a general
  patient assistant.
- Asterisk, Flux, local model, and renderer setup remains environment-specific.
- Continuation grace improves turn ownership at the cost of response latency.
- Semantic fallback can abstain or time out, so deterministic clarification is
  still necessary.
- The repository records evaluation evidence but is not production operations
  infrastructure.
