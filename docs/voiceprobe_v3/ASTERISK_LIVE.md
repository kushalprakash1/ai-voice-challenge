# VoiceProbe v3 live Asterisk integration

The v3 live path is opt-in.  The existing legacy/v2 Asterisk media executor is
left unchanged and remains the default.

Enable the new path with:

```bash
export VOICEPROBE_V3_LIVE=1
export DEEPGRAM_API_KEY="..."
```

When enabled, one authorized call follows this order:

```text
localhost AudioSocket listener
-> AMI Originate
-> AudioSocket UUID validation
-> continuous idle silence
-> Pipecat WorkerRunner + Deepgram Flux
-> native 8 kHz AudioSocket PCM to Pipecat
-> VoiceProbeV3Runtime
-> deterministic TTSSpeakFrame
-> Kokoro 24 kHz render -> existing 8 kHz telephony conversion
-> synchronized AudioSocket playback + 350 ms echo guard
-> FlowSnapshot.complete
-> protocol AudioSocket termination
-> existing Asterisk termination classifier
```

## Completion contract

`FlowSnapshot.complete` is the authoritative v3 success signal.  The tracker
sets it only after `FlowStage.CONFIRMATION` is explicitly confirmed.  The
adapter projection therefore maps:

- `objective_complete = snapshot.complete`
- `booking_confirmed = snapshot.complete`
- `offer_accepted = snapshot.accepted_slot_text is not None`
- legacy `offered_day` / `offered_time` remain `None`

The exact v3 evidence remains available in the artifact event as
`accepted_slot_text` and `booking_confirmation_text`; the live bridge does not
invent split legacy fields from free-form confirmation text.

## Safety and compatibility

- The listener is bound and listening before the one-shot AMI Originate.
- The new path is selected only by `VOICEPROBE_V3_LIVE=1`.
- An explicitly injected media executor still takes precedence for tests.
- Missing `DEEPGRAM_API_KEY` fails before the listener or Originate side effect.
- The legacy `_execute_media_call` implementation is not edited.
- Enabling the code path alone does not connect to Deepgram, Asterisk, or a phone.
