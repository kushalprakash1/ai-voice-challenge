# VoiceProbe v3 live observability

## Why this exists

The 2026-08-16 v3 live call produced valid mixed, inbound, and outbound audio
and multiple deterministic responses, but `transcript.txt` was empty and
`metrics.json` reported `turn_count: 0`.

The cause was not the artifact recorder. `RunArtifactRecorder` already had
working `record_transcript_turn()` and `record_turn_metrics()` APIs. The v3
Pipecat/Flux live boundary simply never called them.

## Live recording contract

The Asterisk v3 live path now uses a recording subclass of
`PipecatRuntimeBridge`.

For every finalized runtime decision it records:

- each preserved Deepgram Flux EndOfTurn source turn as speaker `agent`;
- a `v3_runtime_decision` event containing the structured decision, reason,
  route, ingress reason, flow stage, latency, accepted slot, and booking state;
- one turn-metrics entry so `metrics.json` has a real v3 `turn_count`;
- the patient response text after it is successfully queued to the TTS speech
  sink.

Patient transcript metadata uses `delivery_status=queued_for_tts`. This is
intentional: transcript text proves the exact text accepted by the speech
queue, while the existing `v3_audio_sent` event remains the authoritative
evidence that synthesized PCM was actually handed to AudioSocket.

WAIT/HOLD decisions are recorded even when no patient speech is produced.

## Scope

This changes only v3 live observability. It does not modify:

- deterministic scheduling policy;
- Asterisk originate behavior;
- AMI authorization;
- AudioSocket framing;
- Deepgram Flux configuration;
- Kokoro synthesis;
- destination validation;
- live-call retry policy;
- legacy/v2 behavior.
