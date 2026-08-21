# Iteration from real calls

The frozen batch contains ten calls. The numbers below are presentation order:
Call 10 is the earliest and roughest example, while Call 1 is the latest. They
were not originally executed with these labels. Calls 10–7 are development
runs; Calls 6–1 are the final behavioral cases.

## Development calls

### Call 10

Early responses treated individual ASR endings as complete conversational
turns. When the remote agent continued after a short pause, VoiceProbe could
answer the fragment or speak over the continuation. The first corrective step
was to preserve completed fragments and replay the exact ingress sequence.

### Call 9

Burst buffering prevented simple FIFO replay, but acknowledgements and examples
could still displace the actionable question. Coalescing was changed to assign
one owner to the burst and prefer the direct request over illustrative tails.

### Call 8

Unfamiliar wording exposed gaps in deterministic matching and could leave the
caller silent. A semantic fallback was added for ambiguous turns, with safe
clarification on abstention or timeout. Patient facts and transaction state
remained outside the model.

### Call 7

Concrete slot wording exposed a grounding problem between parsing, state, and
spoken acceptance. Slot extraction was made the shared source for both the
state transition and response text. The failed turn was then replayed before
another live test.

## Gold behavioral calls

### Call 6 — farthest-date scheduling

The caller asked for the furthest bookable date, navigated an existing
appointment, relaxed day and time constraints, and selected a slot from the
latest date offered. The case exercises objective following, horizon handling,
grounded slot selection, and booking confirmation.

### Call 5 — doctor directory

The caller registered a Korean name, asked the system to repeat and spell it,
selected a doctor from the offered directory, and asked doctor-specific
location and hours questions. The call tests name robustness, grounded provider
selection, and context across related questions.

### Call 4 — office information

The caller established self-pay status, requested available locations, selected
one, switched to another, and asked location-specific hours. The case tests
whether inventory, address, hours, and active-location context remain
consistent.

### Call 3 — medication workflow

The caller created a demo profile, requested a refill, and tried to add the
medication needed to continue. The case tests whether the demo workflow offers
a usable path instead of ending at missing chart data.

### Call 2 — escalation handoff

After the refill could not proceed, the caller explicitly accepted an offer to
speak with staff. The case follows the handoff through transfer messaging and
observes whether a useful destination actually receives the caller.

### Call 1 — booking completion

The caller accepted a concrete appointment offered by the remote system. The
case distinguishes spoken acceptance from a completed booking: VoiceProbe does
not mark success until the remote side confirms the transaction.

The development loop was consistent across these cases: listen to the call,
identify which side owned the failure, add a focused replay or state assertion,
then rerun the relevant path. Remote-agent findings are kept separate from
VoiceProbe implementation defects in [BUGS.md](../BUGS.md).
