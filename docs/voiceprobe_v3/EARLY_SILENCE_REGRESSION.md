# VoiceProbe v3 early-silence regression

## Evidence

A historical live run ended after
about 66 seconds with no accepted slot and no booking confirmation.

The new observability path showed the exact failure sequence:

1. Flux emitted the profile/open-intent sentence ending in a comma:
   `... date of birth is July fourth ... How may I help you today,`
2. The fast policy returned `HOLD / obvious_incomplete_asr_fragment`.
3. Flux later emitted `can I help you today?`.
4. That wording was not recognized by the open-ended-intent vocabulary, so
   the policy returned `FALLBACK / novel_or_ambiguous_turn`.
5. No fallback resolver was configured, therefore `response_ready=false` and
   VoiceProbe stayed silent.
6. `Are you still there?` also fell through to FALLBACK, after which the
   remote scheduler ended the call.

## Fix

The deterministic policy now:

- recognizes wrong-DOB + open-intent content before the generic trailing-comma
  fragment gate;
- recognizes `can I help you today?` as an open-ended scheduling prompt;
- responds to `Are you still there?` by confirming presence and restating the
  scheduling objective.

The separate split-profile coalescing behavior is intentionally unchanged.
The remote scheduler accepted `Alex Morgan.` and created the
profile, so it was not the termination defect.

## Scope

No telephony, Asterisk, AudioSocket, Deepgram configuration, Kokoro,
authorization, budget, retry, legacy/v2, or booking-completion semantics are
changed.
