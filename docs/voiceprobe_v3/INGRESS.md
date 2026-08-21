# VoiceProbe v3 Flux ingress

This phase connects the v3 dialogue policy to Pipecat's Deepgram Flux turn
events without touching the production v2 caller.

## Contract

`FluxIngressController` attaches to:

- `on_start_of_turn`
- `on_turn_resumed`
- `on_end_of_turn`

Only `on_end_of_turn` is actionable.

When VoiceProbe is idle, a Flux-confirmed end of turn is evaluated immediately.

When VoiceProbe is preparing or playing a response, subsequent remote turns are
not put into a FIFO reasoning queue. They are accumulated into one
`RemoteSpeechBurstBuffer`. Once the response completes, the whole burst is
passed once through `ConversationBurstCoalescer`.

Example:

```text
VoiceProbe busy
    remote: "Thanks, Alex."
    remote: "Let me check appointments."
    remote: "Thanks for confirming your DOB."
    remote: "What is the reason for your visit?"

VoiceProbe ready
    -> coalesce all four
    -> latest actionable request:
       "What is the reason for your visit?"
    -> fast policy:
       "I have right shoulder pain."
```

The raw turns remain available in `FluxIngressResult.source_turns` for evidence,
debugging, and replay.

## Important boundary

This module does not yet create a live Deepgram connection, synthesize audio,
or replace the existing Asterisk adapter. It is the tested coordination layer
that the transport will call.

The compatibility tool can instantiate the installed Pipecat Flux service and
attach the handlers without connecting to Deepgram:

```bash
PYTHONPATH=src \
~/.venvs/voiceprobe-v3/bin/python \
  tools/v3_verify_pipecat_ingress.py
```
