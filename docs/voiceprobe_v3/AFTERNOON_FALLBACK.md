# VoiceProbe v3 afternoon fallback regression

## Evidence

A recorded live Asterisk/Flux run successfully carried
two-way audio but failed the scheduling objective.

The artifact showed two late generic patient replies (`Yes, please.`), followed
by roughly fifty seconds with no further outbound patient speech. The final
flow snapshot had no accepted slot and no booking confirmation.

The remote scheduling branch was manually transcribed from the mixed call audio
as variants of:

- `afternoon options on a different day or check with a different provider?`
- `check afternoon options earlier in the week?`

The live recorder did not yet persist Flux input turns into `transcript.txt`, so
these remote prompt strings are manual call-audio evidence rather than a
machine-generated transcript.

## Deterministic fallback contract

VoiceProbe remains Friday-afternoon-first.

Only after the remote scheduler explicitly offers the alternate-day afternoon
fallback does v3 relax the *day* constraint while preserving the *afternoon*
constraint:

1. Friday afternoon is attempted first.
2. If offered alternate day vs alternate provider, explicitly choose afternoon
   options earlier in the week instead of replying with a bare yes.
3. Monday-Thursday PM slots may then be accepted.
4. Morning slots remain incompatible.
5. Weekend PM slots are not treated as "earlier in the week."
6. A concrete compatible slot is recorded only after VoiceProbe accepts it.
7. Flow completion still requires explicit remote booked/scheduled/confirmed
   evidence.

This fallback does not modify telephony, Asterisk, AudioSocket, Deepgram Flux,
Kokoro, call authorization, or retry behavior.
