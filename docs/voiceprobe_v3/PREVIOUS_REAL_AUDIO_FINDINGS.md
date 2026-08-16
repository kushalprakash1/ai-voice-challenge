# Previous-call real-audio findings at Flux EOT 0.85

The older failed call exposed two issues not present in the latest-call replay.

First, Deepgram rendered the offered morning slots as spoken words:
`nine AM`, `nine forty five AM`, and `ten thirty AM`. The deterministic policy
now recognizes those as the same incompatible morning slots as their numeric
forms and preserves the hard Friday-afternoon preference.

Second, Deepgram rendered `August 28th` as `August twenty eighth`. The
following-Friday branch now recognizes both numeric and spoken forms.

The audio also showed a genuine short-gap continuation. Flux ended the morning
offer and then began `preference, or would you like to look at later dates or
times?` about 505 ms later. Treating those two EOTs independently creates a
spurious fallback. VoiceProbe therefore introduces a 600 ms continuation grace
for Flux EOT stabilization. This is deliberately below the observed 681.8 ms
gap between the separate reason-for-visit question and its later example, so
that pair remains separate.

The exact historical EOT sequence and measured gaps are frozen in:

`tests/fixtures/v3_calls/flux_previous_eot_085.jsonl`
