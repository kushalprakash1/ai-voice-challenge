# VoiceProbe v3.1 semantic router

The old production fallback converted every unresolved deterministic turn into
the same clarification. That removed silence but allowed normal paraphrases to
create repeat loops.

v3.1 keeps the deterministic fast policy first, then routes only explicit
FALLBACK turns through:

1. a sparse n-gram cosine prototype scorer;
2. local qwen3:1.7b structured classification when uncertain;
3. qwen3:4b escalation for low-confidence/unknown primary results;
4. deterministic mapping from semantic intent to PolicyDecision.

The model cannot emit arbitrary patient speech, patient facts, slot acceptance,
constraint relaxation, or booking completion.

Mathematical fast semantic score:

    score_k = 0.88 * max_cosine(turn, intent_prototypes_k)
              + 0.12 * flow_stage_prior_k

Prototype acceptance requires:

    top_score >= 0.82
    top_score - second_score >= 0.08

Repeated unresolved utterances rotate clarification strategies, making
consecutive identical clarification speech impossible.

Offline gates before another live call:

    PYTHONPATH=src ~/.venvs/voiceprobe-v3/bin/python -m pytest -q       tests/test_v31_semantic_router.py       tests/test_v31_production_semantic.py       tests/test_v3_production.py       tests/test_v3_runtime.py       tests/test_v3_flow_controller.py

    PYTHONPATH=src ~/.venvs/voiceprobe-v3/bin/python       tools/v31_semantic_benchmark.py --full

    PYTHONPATH=src ~/.venvs/voiceprobe-v3/bin/python       tools/v31_run3_offline_replay.py

Do not place another live call until all three gates pass.
