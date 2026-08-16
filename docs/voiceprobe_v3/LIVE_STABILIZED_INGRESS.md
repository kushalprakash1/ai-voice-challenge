# Live Flux continuation stabilization

The 600 ms continuation rule proven by the historical audio replays is now part
of the live `FluxIngressController`.

## Behavior

A Flux `on_end_of_turn` no longer causes an immediate patient response. The
controller holds the finalized transcript for the configured continuation grace
period.

If no new remote speech begins during that interval, the turn is released into
the existing burst/coalescing policy and VoiceProbe may respond.

If `on_start_of_turn` or `on_turn_resumed` occurs before the grace period
expires, the release timer is cancelled while the transcript is retained. The
next `on_end_of_turn` is appended to the same conversational burst. After the
remote speaker finally remains quiet for the full grace period, the combined
burst is routed once.

This directly handles the real-audio failure where Flux produced:

```text
"We have openings ... Would any of these work for your Friday afternoon?"
489 ms later:
"preference, or would you like to look at later dates or times?"
```

The two segments now become one deterministic decision rather than a correct
decision followed by a spurious fallback.

## Latency tradeoff

The live path intentionally pays up to 600 ms of turn-stabilization latency.
This is a correctness-first setting derived from the recorded calls. It remains
far below the multi-second reasoning delays of the previous architecture.

The grace is configurable and unit tests can set it to zero when testing
unrelated ingress behavior.
