# Real-audio findings: latest failed call at Flux EOT 0.85

The stored inbound track for call `96f769c8-50b6-43cd-8a0a-39295b7791c3`
was replayed through Deepgram Flux at native 8 kHz, mono, linear16 with
`eot_threshold=0.85`.

The replay exposed four deterministic-policy gaps that transcript-only tests
had not covered. First, Flux can combine the recording disclaimer and the first
actionable profile request into one EndOfTurn, so actionable requests must
outrank a boilerplate prefix. Second, a short acknowledgement such as
`Great, Alex.` is non-actionable and should be WAIT rather than FALLBACK.
Third, `routine checkup` is appointment-type language. Fourth, a Flux turn
ending on a function word such as `for` is syntactically incomplete and should
HOLD rather than enter fallback.

The replay also rendered the authoritative DOB as
`April twelfth nineteen ninety eight`; flow-state confirmation now accepts that
spoken-number form.

This historical recording is open-loop evidence. It is appropriate for testing
ASR boundaries, routing, response correctness, and state evidence, but not for
proving that a counterfactual v3 live conversation would follow the same later
path.

The exact 13 Flux EndOfTurn transcripts are frozen in
`tests/fixtures/v3_calls/flux_latest_eot_085.jsonl`.

## Real WorkerRunner latest-call provider wording

The real Pipecat WorkerRunner replay surfaced a second provider-choice wording:
the remote agent named two doctors and asked whether the patient had a
preference or whether it should offer the first available. This is semantically
the same provider-preference primitive as the previously observed
"first available okay?" wording.

The fast policy now recognizes provider-choice turns structurally when they
mention first available, identify a provider/doctor/physician option, and ask
for a preference/choice. No provider names are encoded.

