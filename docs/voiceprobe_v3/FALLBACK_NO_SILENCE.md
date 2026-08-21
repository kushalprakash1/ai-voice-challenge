# VoiceProbe v3 production fallback: no-silence invariant

## Problem

The low-level `VoiceProbeV3Runtime` intentionally permits an unresolved
`DecisionKind.FALLBACK` when no fallback resolver is installed. Such a result
has `requires_response=True` but `response_ready=False`.

That behavior is useful for offline layering tests, but it is unsafe as the
default production telephony behavior: an unfamiliar but valid scheduler
utterance can otherwise make VoiceProbe remain completely silent until the
remote side hangs up.

## Production invariant

`PipecatRuntimeBridge` now always has a fallback resolver by default.

When the deterministic fast policy returns FALLBACK, production resolves it to:

    DecisionKind.CLARIFY
    "Could you please repeat that question?"

The runtime route remains `fallback`, so metrics and observability still show
that the fast policy did not understand the turn.

`CLARIFY` is intentionally a response-producing decision with no corresponding
flow-state mutation. It therefore does not fabricate patient facts, relax
scheduling constraints, accept a slot, confirm a booking, or mark scheduling
progress.

The production bridge rejects `fallback_resolver=None` so the live default
cannot silently regress.

## Low-level runtime contract

`VoiceProbeV3Runtime()` without a resolver is unchanged. Its existing test
continues to prove that an unresolved FALLBACK remains explicit and
`response_ready=False`. This preserves the runtime as a composable primitive
for future model-backed fallback layers.

## Scope

This hardening does not change Asterisk, AudioSocket, Deepgram Flux settings,
Kokoro synthesis, authorization, budgets, retry behavior, deterministic
scheduling rules, accepted-slot logic, or booking-confirmation semantics.
