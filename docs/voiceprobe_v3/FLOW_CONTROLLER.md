# VoiceProbe v3 structured flow controller

This phase introduces explicit scheduling progress without forcing the remote
agent through a rigid script.

## Two dimensions of state

For each workflow stage, VoiceProbe distinguishes:

- **communicated**: VoiceProbe has actually supplied or selected that piece of
  information;
- **confirmed**: the remote scheduling agent has explicitly acknowledged or
  confirmed it.

This prevents optimistic state from becoming fabricated truth.

For example, saying `First available is fine` communicates the provider
preference. It does not prove the remote system accepted or applied it.

Likewise, accepting a slot does not complete the objective. Completion requires
an explicit remote booking confirmation containing a concrete time.

## Stages

```text
PROFILE
IDENTITY
DOB
VISIT_REASON
APPOINTMENT_TYPE
INSURANCE
DATE_TIME
PROVIDER
SLOT
CONFIRMATION
COMPLETE
```

The remote agent may skip, combine, repeat, or revisit them. The tracker updates
only the stages actually evidenced by the conversation.

## Fallback LLM nodes

`flow_nodes.py` creates Pipecat `NodeConfig`-compatible dictionaries for the
currently missing stage. They intentionally keep the fallback model focused on
one narrow task and repeat the immutable patient facts.

The deterministic fast policy remains the first choice. These nodes are for
future novel/ambiguous-turn fallback, not for routine questions.

Verify the installed Pipecat type with:

```bash
PYTHONPATH=src \
~/.venvs/voiceprobe-v3/bin/python \
  tools/v3_verify_flow_nodes.py
```
