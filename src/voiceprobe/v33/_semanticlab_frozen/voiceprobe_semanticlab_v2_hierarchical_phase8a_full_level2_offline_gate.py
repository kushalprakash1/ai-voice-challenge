#!/usr/bin/env python3
"""Offline full-Level-2 evaluation with hierarchical Phase 8A substituted only.

Candidate under test
--------------------
artifacts/candidates/semanticlab_v2_phase8a_hierarchical_multiview.pt

Everything else remains the current V13.2 Level-2 stack:
- Phase 6B unchanged
- Phase 7C/7D/7I/7J unchanged
- Phase 8B1 unchanged
- Phase 8C unchanged
- transaction normalizer unchanged
- V2 -> V13.2 deterministic architecture unchanged

This harness replaces Phase 8A predictions at BOTH places they are consumed:
1. base whole-turn + clause dense inference
2. V2 clause-level dense repair

It does NOT overwrite A8A3 and does NOT modify production/runtime files.

Evaluation
----------
A. Established 1,146 DEVELOPMENT cases
   Must retain the exact V13.2 contract:
   - historical 133: 133/133
   - exposed 108: 107/108 with ONLY h2_asr_027 transaction_signal conflict
   - all other established groups: perfect

B. Exposed final diagnostic 120
   Report:
   - raw formal 120 (traceability only)
   - ontology-coherent 112:
       excludes 7 cases whose expected act/topic pair is impossible under
       the frozen 20-pair ontology
       excludes f2a_001 as explicit spec-review case

The exposed 120 is NOT used to train or select the Phase 8A candidate.
This script only evaluates the already-selected candidate.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = Path(".").resolve()

V132_FILE = HERE / "voiceprobe_semanticlab_v2_post_fresh_architecture_v13_2_final.py"
HIER_FILE = HERE / "voiceprobe_semanticlab_v2_phase8a_hierarchical_multiview_architecture.py"

ADV120_FILE = HERE / "semanticlab_v2_v6_adversarial_generality_cases.jsonl"
FINAL132_FILE = HERE / "semanticlab_v2_final_unseen_holdout_v2_20260817.jsonl"
COHERENT142_FILE = HERE / "semanticlab_v2_v9_coherent_fresh_adversarial_142.jsonl"
COHERENT127_FILE = HERE / "semanticlab_v2_v11_coherent_fresh_adversarial_127.jsonl"
V11FRESH128_FILE = HERE / "semanticlab_v2_v11_fresh_adversarial_generality_128_v2.jsonl"
V12FRESH128_FILE = HERE / "semanticlab_v2_v12_fresh_adversarial_generality_128_v2.jsonl"
V13FRESH128_FILE = HERE / "semanticlab_v2_v13_fresh_adversarial_generality_128_v2.jsonl"
EXPOSED120_FILE = HERE / "semanticlab_v2_level2_final_unseen_holdout_120_v2_20260817.jsonl"

HIER_ARTIFACT = ROOT / "artifacts/candidates/semanticlab_v2_phase8a_hierarchical_multiview.pt"

DECLARED_EXPOSED_CONFLICT = "h2_asr_027"
SPEC_REVIEW_IDS = {"f2a_001"}

for p in (
    V132_FILE,
    HIER_FILE,
    ADV120_FILE,
    FINAL132_FILE,
    COHERENT142_FILE,
    COHERENT127_FILE,
    V11FRESH128_FILE,
    V12FRESH128_FILE,
    V13FRESH128_FILE,
    EXPOSED120_FILE,
    HIER_ARTIFACT,
):
    if not p.is_file():
        raise SystemExit(f"Missing required file: {p}")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v132 = load_mod("l2_hier_full_v132", V132_FILE)
hier = load_mod("l2_hier_full_model", HIER_FILE)

v13 = v132.v13
v12 = v132.v12
v11 = v132.v11
v10 = v132.v10
v8 = v132.v8
v6 = v132.v6
v5 = v132.v5
v2 = v132.v2
base = v132.base

from voiceprobe.v33.semantic_corpus import load_semanticlab_cases
from voiceprobe.v33.semantic_frame_eval import evaluate_frame


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_ck(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


CAND = load_ck(HIER_ARTIFACT)

VALID = tuple(tuple(x) for x in CAND["valid_pairs"])
VALID_SET = set(VALID)
ACTS = tuple(hier.p8a.SPEECH_ACTS)
TOPICS = tuple(hier.p8a.TOPICS)

if CAND.get("candidate_status") != "PROMISING":
    raise SystemExit(
        f"Candidate artifact status is not PROMISING: {CAND.get('candidate_status')!r}"
    )

if tuple(CAND.get("speech_acts", ACTS)) != ACTS:
    raise SystemExit("Candidate speech-act ontology mismatch.")
if tuple(CAND.get("topics", TOPICS)) != TOPICS:
    raise SystemExit("Candidate topic ontology mismatch.")


class HierarchicalPhase8APredictor:
    """Read-only Phase 8A predictor backed by frozen A8A3 + hierarchical heads."""

    def __init__(self):
        self.teacher = hier.p8a.DenseModel()
        a8a3 = load_ck(base.A8A3)
        self.teacher.load_state_dict(a8a3["state_dict"])
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        if tuple(tuple(x) for x in a8a3["valid_pairs"]) != VALID:
            raise RuntimeError("Candidate/frozen A8A3 legal-pair ontology mismatch.")

        self.model = hier.HierarchicalFusion(
            hidden_size=int(self.teacher.encoder.config.dim),
            act_count=len(ACTS),
            topic_count=len(TOPICS),
            pair_count=len(VALID),
        )
        self.model.load_state_dict(CAND["hierarchical_state_dict"])
        self.model.eval()

        self.factor_scale = float(CAND["factor_scale"])
        self.interaction_scale = float(CAND["interaction_scale"])
        self.tok = base.tokenizer_for(hier.p8a.MODEL_NAME)

    def _parts(self, item):
        return tuple(item.context), str(item.utterance)

    def _encode_texts(self, texts):
        rows = []
        with torch.no_grad():
            for start in range(0, len(texts), 64):
                z = self.tok(
                    texts[start:start+64],
                    padding=True,
                    truncation=True,
                    max_length=hier.p8a.MAX_LENGTH,
                    return_tensors="pt",
                )
                h = self.teacher.encoder(
                    input_ids=z["input_ids"],
                    attention_mask=z["attention_mask"],
                ).last_hidden_state[:, 0, :]
                rows.append(h.cpu())
        return torch.cat(rows, dim=0)

    def predict(self, items):
        if not items:
            return [], [], []

        combined_texts = []
        turn_texts = []
        context_texts = []

        for item in items:
            context, turn = self._parts(item)
            ctx = " || ".join(context) if context else "<none>"
            combined_texts.append(hier.p8a.serialize(context, turn))
            turn_texts.append("Latest clinic utterance: " + turn)
            context_texts.append("Recent clinic context: " + ctx)

        combined_h = self._encode_texts(combined_texts)
        turn_h = self._encode_texts(turn_texts)
        context_h = self._encode_texts(context_texts)

        with torch.no_grad():
            base_act = self.teacher.act_head(combined_h)
            base_topic = self.teacher.topic_head(combined_h)
            base_pair = hier.p8a.pair_logits(base_act, base_topic, VALID)

            ar, tr, pr = self.model(
                combined_h,
                turn_h,
                context_h,
                base_act,
                base_topic,
                base_pair,
            )

            final_act = base_act + self.factor_scale * ar
            final_topic = base_topic + self.factor_scale * tr
            final_pair = (
                hier.p8a.pair_logits(final_act, final_topic, VALID)
                + self.interaction_scale * pr
            )

            pair_ids = final_pair.argmax(dim=-1).tolist()
            act_ids = final_act.argmax(dim=-1).tolist()
            topic_ids = final_topic.argmax(dim=-1).tolist()

        pairs = [VALID[i] for i in pair_ids]
        acts = [ACTS[i] for i in act_ids]
        topics = [TOPICS[i] for i in topic_ids]
        return pairs, acts, topics

    def close(self):
        del self.tok, self.model, self.teacher
        gc.collect()


PREDICTOR = None


def candidate_run_phase8a_dense(runtime, flat_clauses, _p8a, _ck8a):
    """Drop-in replacement for base.run_phase8a_dense."""
    whole_pairs, independent_acts, independent_topics = PREDICTOR.predict(runtime)

    clause_pairs = []
    if flat_clauses:
        clause_pairs, _, _ = PREDICTOR.predict(flat_clauses)

    return (
        list(whole_pairs),
        list(independent_acts),
        list(independent_topics),
        list(clause_pairs),
        VALID,
    )


def candidate_repair_dense_pairs(runtime, result, checkpoints, diag):
    """V2 dense repair with hierarchical Phase 8A used for clause inference too."""
    p8c = v2.load_mod("l2_hier_candidate_p8c", base.P8C)

    repaired = list(result.dense_pairs)

    for i, turn in enumerate(runtime):
        pair = repaired[i]
        text = turn.utterance

        if (
            pair == ("offer", "availability")
            and v2.NEGATIVE_AVAILABILITY_RE.search(text)
            and "?" not in text
            and not v2.OFFER_ACTION_RE.search(text)
        ):
            pair = ("statement", "availability")
            diag.hit(i, "dense_negative_availability_statement")

        if (
            pair == ("question", "patient_fact")
            and v2.PATIENT_FACT_REQUEST_RE.search(text)
        ):
            pair = ("request", "patient_fact")
            diag.hit(i, "dense_patient_fact_indirect_request")

        if (
            result.ambiguity_details[i] == "option_reference"
            and v2.VAGUE_SELECTION_RE.search(text)
        ):
            if v2.latest_context_is_explicit_offer(turn):
                pair = ("confirmation", "availability")
                diag.hit(i, "dense_vague_selection_after_explicit_offer")
            else:
                pair = ("statement", "other")
                diag.hit(i, "dense_vague_selection_after_nonoffer_list")

        clauses = p8c.split_clauses(text)
        if len(clauses) > 1 and pair in {
            ("statement", "appointment_state"),
            ("statement", "profile"),
        }:
            items = [
                base.RuntimeTurn(
                    context=turn.context,
                    utterance=clause,
                )
                for clause in clauses
            ]
            clause_pairs, _, _ = PREDICTOR.predict(items)
            if ("offer", "availability") in clause_pairs:
                pair = ("offer", "availability")
                diag.hit(i, "dense_background_record_plus_availability_offer")

        repaired[i] = pair

    return repaired


def assemble_v132(runtime, checkpoints):
    """Run current V13.2 composition using whichever Phase8A hooks are active."""
    result = base.assemble_level2(runtime, checkpoints)

    (
        v2_frames,
        v2_schedules,
        _dense,
        _facts,
        _refs,
        _diag2,
        v2_errors,
    ) = v2.construct_candidate_frames(
        runtime,
        result,
        checkpoints,
    )

    (
        frames,
        _schedules,
        diag,
        v132_errors,
    ) = v132.construct_v13_2_frames(
        runtime,
        result,
        checkpoints,
        v2_frames,
        v2_schedules,
    )

    return result, frames, diag, v2_errors, v132_errors


def fmap(cases, failures):
    return {
        c.case_id: tuple(x.field for x in fs)
        for c, fs in zip(cases, failures)
        if fs
    }


def metrics(label, cases, failures):
    exact = sum(not fs for fs in failures)
    print()
    print(f"========== {label} ==========")
    print("cases=", len(cases))
    print("exact=", exact, "/", len(cases), "accuracy=", round(exact / len(cases), 4))
    for field in base.FIELDS:
        passed = sum(field not in {x.field for x in fs} for fs in failures)
        print(
            field,
            f"pass={passed}",
            f"fail={len(cases)-passed}",
            f"accuracy={passed/len(cases):.4f}",
        )
    return exact


def subset(cases, frames, keep_indices):
    return (
        [cases[i] for i in keep_indices],
        [frames[i] for i in keep_indices],
    )


def main():
    global PREDICTOR

    print("========== HIERARCHICAL PHASE 8A — OFFLINE FULL LEVEL 2 EVALUATION ==========")
    print("telephony=DISABLED")
    print("training=NO")
    print("phase8a_candidate_already_selected=YES")
    print("phase8b_retrained=NO")
    print("phase7c_retrained=NO")
    print("phase6b_retrained=NO")
    print("runtime_wiring=NO")
    print("production_artifact_overwrite=NO")
    print("v0_17_modified=NO")
    print("gold_runtime_inputs=NO")
    print("case_id_runtime_inputs=NO")
    print("category_runtime_inputs=NO")
    print("tags_runtime_inputs=NO")
    print("exposed120_used_for_candidate_selection=NO")
    print("level2_frozen=NO")

    source_before = base.source_snapshot()
    a8a3_sha_before = sha256_file(base.A8A3)
    candidate_sha_before = sha256_file(HIER_ARTIFACT)

    expected_a8a3_sha = CAND.get("a8a3_sha256")
    if expected_a8a3_sha and expected_a8a3_sha != a8a3_sha_before:
        raise SystemExit(
            "Candidate was built against a different A8A3 artifact:\n"
            f" candidate_expected={expected_a8a3_sha}\n"
            f" current_a8a3={a8a3_sha_before}"
        )

    checkpoints = base.validate_environment()

    if tuple(tuple(x) for x in checkpoints["8a"]["valid_pairs"]) != VALID:
        raise SystemExit("Environment Phase8A ontology differs from candidate.")

    print("candidate_artifact=", HIER_ARTIFACT)
    print("candidate_sha256=", candidate_sha_before)
    print("candidate_factor_scale=", CAND["factor_scale"])
    print("candidate_interaction_scale=", CAND["interaction_scale"])
    print("candidate_best_epoch=", CAND["best_epoch"])
    print("frozen_valid_pair_count=", len(VALID))

    # ------------------------------------------------------------------
    # Baseline V13.2 on exposed diagnostic BEFORE any monkey patching.
    # This is for direct whole-frame transition attribution only.
    # ------------------------------------------------------------------
    exposed120 = list(load_semanticlab_cases(EXPOSED120_FILE))
    assert len(exposed120) == 120

    exposed_runtime = [
        base.RuntimeTurn(
            context=tuple(c.context),
            utterance=str(c.utterance),
        )
        for c in exposed120
    ]

    print()
    print("========== BASELINE V13.2 EXPOSED-120 INFERENCE ==========")
    baseline_result, baseline_frames, _, baseline_v2err, baseline_v132err = (
        assemble_v132(exposed_runtime, checkpoints)
    )
    print("baseline_exposed_inference_complete=YES")

    # Determine benchmark coherence from the frozen ontology itself.
    illegal_pair_ids = []
    coherent_indices = []

    for i, c in enumerate(exposed120):
        gp = hier.p8a.gold_pair(c)
        if gp not in VALID_SET:
            illegal_pair_ids.append(c.case_id)
            continue
        if c.case_id in SPEC_REVIEW_IDS:
            continue
        coherent_indices.append(i)

    print("exposed120_illegal_pair_case_ids=", illegal_pair_ids)
    print("exposed120_spec_review_case_ids=", sorted(SPEC_REVIEW_IDS))
    print("exposed120_ontology_coherent_cases=", len(coherent_indices))

    # ------------------------------------------------------------------
    # Activate hierarchical Phase8A ONLY for candidate inference.
    # ------------------------------------------------------------------
    print()
    print("========== ACTIVATING OFFLINE PHASE 8A SUBSTITUTE ==========")
    PREDICTOR = HierarchicalPhase8APredictor()

    original_run_phase8a_dense = base.run_phase8a_dense
    original_repair_dense_pairs = v2.repair_dense_pairs

    base.run_phase8a_dense = candidate_run_phase8a_dense
    v2.repair_dense_pairs = candidate_repair_dense_pairs

    print("base_phase8a_hook=HIERARCHICAL")
    print("v2_clause_phase8a_hook=HIERARCHICAL")
    print("runtime_source_files_modified=NO")

    # ------------------------------------------------------------------
    # Established 1,146 + exposed 120 in ONE candidate assembly.
    # ------------------------------------------------------------------
    historical = list(load_semanticlab_cases())
    exposed108 = list(load_semanticlab_cases(v5.EXPOSED_FILE))
    adv120 = list(load_semanticlab_cases(ADV120_FILE))
    final132 = list(load_semanticlab_cases(FINAL132_FILE))
    coherent142 = list(load_semanticlab_cases(COHERENT142_FILE))
    coherent127 = list(load_semanticlab_cases(COHERENT127_FILE))
    v11fresh128 = list(load_semanticlab_cases(V11FRESH128_FILE))
    v12fresh128 = list(load_semanticlab_cases(V12FRESH128_FILE))
    v13fresh128 = list(load_semanticlab_cases(V13FRESH128_FILE))

    assert len(historical) == 133
    assert len(exposed108) == 108
    assert len(adv120) == 120
    assert len(final132) == 132
    assert len(coherent142) == 142
    assert len(coherent127) == 127
    assert len(v11fresh128) == 128
    assert len(v12fresh128) == 128
    assert len(v13fresh128) == 128

    groups = [
        ("HISTORICAL 133 — HIER PHASE8A", historical),
        ("EXPOSED 108 — HIER PHASE8A", exposed108),
        ("ADVERSARIAL 120 — HIER PHASE8A", adv120),
        ("EXPOSED FINAL 132 — HIER PHASE8A", final132),
        ("COHERENT FRESH 142 — HIER PHASE8A", coherent142),
        ("COHERENT V10 FRESH 127 — HIER PHASE8A", coherent127),
        ("EXPOSED V11 FRESH 128 — HIER PHASE8A", v11fresh128),
        ("EXPOSED V12 FRESH 128 — HIER PHASE8A", v12fresh128),
        ("EXPOSED V13 FRESH 128 — HIER PHASE8A", v13fresh128),
        ("EXPOSED LEVEL2 FINAL 120 — HIER PHASE8A", exposed120),
    ]

    all_cases = [c for _, cases in groups for c in cases]
    runtime = [
        base.RuntimeTurn(
            context=tuple(c.context),
            utterance=str(c.utterance),
        )
        for c in all_cases
    ]

    print()
    print("candidate_combined_inference_cases=", len(runtime))
    print("candidate_gold_scoring_begins_after_all_predictions=YES")

    candidate_result, candidate_frames, candidate_diag, cand_v2err, cand_v132err = (
        assemble_v132(runtime, checkpoints)
    )

    print("candidate_full_level2_inference_complete=YES")

    # Restore hooks immediately after candidate inference.
    base.run_phase8a_dense = original_run_phase8a_dense
    v2.repair_dense_pairs = original_repair_dense_pairs
    PREDICTOR.close()
    PREDICTOR = None

    print("offline_phase8a_hooks_restored=YES")

    # ------------------------------------------------------------------
    # Slice candidate frames by group and score.
    # ------------------------------------------------------------------
    chunks = []
    pos = 0
    for _, cases in groups:
        chunks.append(candidate_frames[pos:pos+len(cases)])
        pos += len(cases)

    failure_sets = []
    exacts = []

    for (label, cases), frames in zip(groups[:9], chunks[:9]):
        failures = [
            evaluate_frame(c, f)
            for c, f in zip(cases, frames)
        ]
        failure_sets.append(failures)
        exacts.append(metrics(label, cases, failures))

    (
        hfail,
        efail,
        afail,
        ffail,
        c142fail,
        c127fail,
        v11fail,
        v12fail,
        v13fail,
    ) = failure_sets

    hmap = fmap(historical, hfail)
    emap = fmap(exposed108, efail)
    enon = {
        cid: fields
        for cid, fields in emap.items()
        if cid != DECLARED_EXPOSED_CONFLICT
    }
    conflict_fields = emap.get(DECLARED_EXPOSED_CONFLICT, ())
    conflict_ok = conflict_fields == ("transaction_signal",)

    amap = fmap(adv120, afail)
    finalmap = fmap(final132, ffail)
    c142map = fmap(coherent142, c142fail)
    c127map = fmap(coherent127, c127fail)
    v11map = fmap(v11fresh128, v11fail)
    v12map = fmap(v12fresh128, v12fail)
    v13map = fmap(v13fresh128, v13fail)

    established_cases = all_cases[:1146]
    established_frames = candidate_frames[:1146]

    search_violations = []
    for c, fr in zip(established_cases, established_frames):
        if (
            fr.transaction_operation.value == "search"
            and fr.transaction_signal.value != "none"
        ):
            search_violations.append(
                (c.case_id, fr.transaction_signal.value, c.utterance)
            )

    safety_fields = {
        "record_claims",
        "transaction_operation",
        "transaction_signal",
    }
    v13_safety = [
        c.case_id
        for c, fs in zip(v13fresh128, v13fail)
        if "safety" in c.tags
        and ({x.field for x in fs} & safety_fields)
    ]

    ambiguity_fields = {
        "ambiguity.kind",
        "ambiguity.candidates",
    }
    v13_ambiguity = [
        c.case_id
        for c, fs in zip(v13fresh128, v13fail)
        if "ambiguity" in c.tags
        and ({x.field for x in fs} & ambiguity_fields)
    ]

    print()
    print("========== ESTABLISHED 1146 FAILURE MAP ==========")
    print("historical_failure_fields=", hmap)
    print("exposed_108_failure_fields=", emap)
    print("exposed_108_nonconflict_failure_fields=", enon)
    print("declared_exposed_conflict_fields=", conflict_fields)
    print(
        "declared_exposed_conflict_shape_ok=",
        "YES" if conflict_ok else "NO",
    )
    print("adversarial_120_failure_fields=", amap)
    print("final_132_failure_fields=", finalmap)
    print("coherent_142_failure_fields=", c142map)
    print("coherent_127_failure_fields=", c127map)
    print("v11_fresh_128_failure_fields=", v11map)
    print("v12_fresh_128_failure_fields=", v12map)
    print("v13_fresh_128_failure_fields=", v13map)
    print("search_signal_violations=", search_violations)
    print("v13_fresh_safety_failure_case_ids=", v13_safety)
    print("v13_fresh_ambiguity_failure_case_ids=", v13_ambiguity)

    # ------------------------------------------------------------------
    # Exposed 120: raw and ontology-coherent 112.
    # ------------------------------------------------------------------
    candidate_exposed_frames = chunks[-1]

    baseline_raw_fail = [
        evaluate_frame(c, f)
        for c, f in zip(exposed120, baseline_frames)
    ]
    candidate_raw_fail = [
        evaluate_frame(c, f)
        for c, f in zip(exposed120, candidate_exposed_frames)
    ]

    baseline_raw_exact = metrics(
        "BASELINE V13.2 — EXPOSED 120 RAW TRACEABILITY",
        exposed120,
        baseline_raw_fail,
    )
    candidate_raw_exact = metrics(
        "HIER PHASE8A — EXPOSED 120 RAW TRACEABILITY",
        exposed120,
        candidate_raw_fail,
    )

    coherent_cases = [exposed120[i] for i in coherent_indices]
    baseline_coherent_frames = [baseline_frames[i] for i in coherent_indices]
    candidate_coherent_frames = [
        candidate_exposed_frames[i]
        for i in coherent_indices
    ]

    baseline_coherent_fail = [
        evaluate_frame(c, f)
        for c, f in zip(coherent_cases, baseline_coherent_frames)
    ]
    candidate_coherent_fail = [
        evaluate_frame(c, f)
        for c, f in zip(coherent_cases, candidate_coherent_frames)
    ]

    baseline_coherent_exact = metrics(
        "BASELINE V13.2 — EXPOSED ONTOLOGY-COHERENT 112",
        coherent_cases,
        baseline_coherent_fail,
    )
    candidate_coherent_exact = metrics(
        "HIER PHASE8A — EXPOSED ONTOLOGY-COHERENT 112",
        coherent_cases,
        candidate_coherent_fail,
    )

    improved = []
    regressed = []
    still_failing = []

    for c, bfs, cfs in zip(
        coherent_cases,
        baseline_coherent_fail,
        candidate_coherent_fail,
    ):
        if bfs and not cfs:
            improved.append(c.case_id)
        if not bfs and cfs:
            regressed.append(c.case_id)
        if cfs:
            still_failing.append(
                (c.case_id, tuple(x.field for x in cfs))
            )

    print()
    print("========== EXPOSED 112 WHOLE-FRAME TRANSITIONS ==========")
    print("baseline_exact=", baseline_coherent_exact, "/ 112")
    print("candidate_exact=", candidate_coherent_exact, "/ 112")
    print("net_exact_gain=", candidate_coherent_exact - baseline_coherent_exact)
    print("improved_case_ids=", improved)
    print("regressed_case_ids=", regressed)
    print("remaining_failure_fields=", still_failing)

    # ------------------------------------------------------------------
    # Diagnostics / integrity.
    # ------------------------------------------------------------------
    counts = Counter(
        rule
        for rules in candidate_diag.rules.values()
        for rule in rules
    )

    print()
    print("========== V13.2 RULE HIT COUNTS WITH HIER PHASE8A ==========")
    for rule, count in sorted(counts.items()):
        print(rule, "hits=", count)

    source_after = base.source_snapshot()
    a8a3_sha_after = sha256_file(base.A8A3)
    candidate_sha_after = sha256_file(HIER_ARTIFACT)

    baseline_v2_errors = sum(len(v) for v in baseline_v2err.values())
    baseline_v132_errors = sum(len(v) for v in baseline_v132err.values())
    candidate_v2_errors = sum(len(v) for v in cand_v2err.values())
    candidate_v132_errors = sum(len(v) for v in cand_v132err.values())

    established_pass = all((
        exacts[0] == 133,
        exacts[1] == 107,
        exacts[2] == 120,
        exacts[3] == 132,
        exacts[4] == 142,
        exacts[5] == 127,
        exacts[6] == 128,
        exacts[7] == 128,
        exacts[8] == 128,
        not hmap,
        not enon,
        conflict_ok,
        not amap,
        not finalmap,
        not c142map,
        not c127map,
        not v11map,
        not v12map,
        not v13map,
        not search_violations,
        not v13_safety,
        not v13_ambiguity,
        candidate_v2_errors == 0,
        candidate_v132_errors == 0,
    ))

    exposed_transition_pass = all((
        not regressed,
        candidate_coherent_exact >= baseline_coherent_exact,
    ))

    integrity_pass = all((
        source_before == source_after,
        a8a3_sha_before == a8a3_sha_after,
        candidate_sha_before == candidate_sha_after,
    ))

    promising = all((
        established_pass,
        exposed_transition_pass,
        integrity_pass,
    ))

    print()
    print("========== HIERARCHICAL PHASE 8A FULL LEVEL 2 DECISION ==========")
    print("established_1146_contract=", "PASS" if established_pass else "FAIL")
    print(
        "exposed_112_no_whole_frame_regressions=",
        "PASS" if not regressed else "FAIL",
    )
    print(
        "exposed_112_exact_non_decreasing=",
        "PASS"
        if candidate_coherent_exact >= baseline_coherent_exact
        else "FAIL",
    )
    print(
        "baseline_exposed_120_raw_exact=",
        baseline_raw_exact,
        "/ 120",
    )
    print(
        "candidate_exposed_120_raw_exact=",
        candidate_raw_exact,
        "/ 120",
    )
    print(
        "baseline_exposed_112_coherent_exact=",
        baseline_coherent_exact,
        "/ 112",
    )
    print(
        "candidate_exposed_112_coherent_exact=",
        candidate_coherent_exact,
        "/ 112",
    )
    print("baseline_v2_constructor_errors=", baseline_v2_errors)
    print("baseline_v13_2_constructor_errors=", baseline_v132_errors)
    print("candidate_v2_constructor_errors=", candidate_v2_errors)
    print("candidate_v13_2_constructor_errors=", candidate_v132_errors)
    print(
        "source_tree_python_unchanged=",
        "YES" if source_before == source_after else "NO",
    )
    print(
        "a8a3_artifact_unchanged=",
        "YES" if a8a3_sha_before == a8a3_sha_after else "NO",
    )
    print(
        "hierarchical_candidate_unchanged=",
        "YES"
        if candidate_sha_before == candidate_sha_after
        else "NO",
    )
    print("phase8b_retrained=NO")
    print("phase7c_retrained=NO")
    print("phase6b_retrained=NO")
    print("runtime_wiring_modified=NO")
    print("level2_frozen=NO")

    if promising:
        print("HIERARCHICAL_PHASE8A_FULL_LEVEL2_OFFLINE_GATE=PASS")
        print(
            "NEXT_ACTION=REFRESH_PHASE8B_FROM_FROZEN_A8A3_ENCODER_"
            "WHILE_USING_HIERARCHICAL_PHASE8A_FOR_LEVEL2_COMPOSITION"
        )
        return 0

    print("HIERARCHICAL_PHASE8A_FULL_LEVEL2_OFFLINE_GATE=FAIL")
    print(
        "NEXT_ACTION=CLASSIFY_FULL_LEVEL2_REGRESSIONS_BEFORE_ANY_"
        "PHASE8B_RETRAINING"
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
