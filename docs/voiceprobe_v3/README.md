# VoiceProbe v3

VoiceProbe v3 is a replay-driven rebuild of the live conversational path.

## Why this branch exists

The v2 reasoning core can answer clean isolated prompts, but the two most recent
live calls show three separate production failures:

1. elementary fact/workflow questions unnecessarily enter a multi-second LLM
   path;
2. buffering every finalized remote utterance creates stale FIFO work when the
   remote agent continues speaking;
3. text-only smoke tests do not reproduce the timing and segmentation failures
   present in real calls.

V3 therefore uses a layered design:

```text
telephony audio
    -> conversational STT / turn detection
    -> conversation-burst coalescer
    -> deterministic routine scheduling policy
    -> structured flow state
    -> LLM fallback only for genuinely novel language
    -> streaming TTS
```

## Frozen evidence

`tests/fixtures/v3_calls/raw/` contains sanitized terminal examples. Structured
regression cases preserve routing behavior without publishing raw calls.

`tests/fixtures/v3_calls/regression_cases.jsonl` contains the first set of
turn-level expected outcomes. It includes both successful historical behavior
and known failures such as:

- answering a first/last-name request with the scheduling objective;
- answering "What is the reason for your visit?" with the date/time objective;
- failing to use "first available" when explicitly offered;
- treating `Would any...` as an actionable complete turn;
- dropping the following-Friday search question.

These fixtures are not model training data in the gradient-descent sense.
They are executable behavioral regression data.

## Phase 1: local deterministic baseline

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_v3_corpus.py \
  tests/test_v3_fast_policy.py \
  tests/test_v3_coalescer.py \
  -q

PYTHONPATH=src .venv/bin/python tools/v3_replay_corpus.py
```

Routine turns should be answered without Qwen.

## Phase 2: Pipecat + Deepgram Flux

Do not integrate this into the live caller until Phase 1 is green.

Current Pipecat releases include Flows under `pipecat.flows`, and the Deepgram
extra provides `DeepgramFluxSTTService`. The intended install command is:

```bash
.venv/bin/pip install "pipecat-ai[deepgram]"
```

Flux should own end-of-turn detection. The v3 coalescer sits after final turn
events and collapses acknowledgement/status bursts before policy.

Initial Flux configuration target:

- model: `flux-general-en`
- eot_threshold: 0.8
- eager EOT: disabled initially
- keyterms: Pivot Point, Alex Morgan, Blue Cross, new patient consultation
- inbound telephony audio may remain 8 kHz because Flux supports 8 kHz raw
  audio; 16 kHz remains recommended when we control resampling.

## Phase 3: flow state

Build scheduling states around the task, not around conversational wording:

```text
PROFILE
NAME
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

The remote agent is allowed to skip, combine, repeat, or revisit states.
Authoritative patient facts remain immutable.

## Phase 4: audio replay gate

No live call is allowed merely because text tests pass.

A release build must replay the stored inbound audio from previous calls
through:

```text
audio -> STT -> turn detection -> coalescer -> policy/flow -> decision
```

and meet latency plus behavioral assertions before another paid call.
