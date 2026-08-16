# VoiceProbe v3.1 Calibrated Embedding Semantics

The production semantic fallback remains subordinate to the deterministic v3
fast policy.

For a FALLBACK turn, production now uses a local
`qwen3-embedding:0.6b` perception layer when its validated prototype cache is
available:

1. segment the remote utterance into semantic clauses;
2. embed all clauses in one local Ollama `/api/embed` request;
3. compare each clause to closed-domain prototype vectors and class centroids;
4. apply calibrated high-specificity speech-act overrides;
5. select at most one atomic fact request per clause;
6. compose complaint + appointment-type only from independent clause evidence;
7. return a typed semantic intent;
8. map the intent to an authoritative deterministic `PolicyDecision`.

The embedding layer never generates patient speech, patient facts, slot
acceptance, or booking state.

If the cache is missing, stale, invalid, Ollama is unavailable, the local model
times out, or the response shape is invalid, the router falls safely back to the
existing statistical v3.1 classifier.

The generative qwen3 intent classifier remains disabled from production intent
control.

## Cache

Prepare the prototype cache offline:

```bash
PYTHONPATH=src ~/.venvs/voiceprobe-v3/bin/python \
  tools/v31_prepare_embedding_cache.py
```

The cache is written outside the repository under:

`~/.cache/voiceprobe/`

and is fingerprinted against the exact model name and semantic prototype corpus.

## Preflight

Immediately before a live assessment call, run:

```bash
PYTHONPATH=src ~/.venvs/voiceprobe-v3/bin/python \
  tools/v31_embedding_preflight.py
```

The preflight requires exact semantic classifications for the critical
held-out/regression cases and a warm p95 below 1200 ms. It places no phone call.
