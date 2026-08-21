#!/usr/bin/env python3
"""Reference multi-option boundary localizer V2 (read-only).

Purpose
-------
Localize ONLY the failed V1 rule:
    suppress:multi_option_without_unique_resolution

No candidate is promoted. No model is trained. No frozen ambiguity/OOS/reference
checkpoint is modified.

For every multi-option case, inference-only features are computed before gold:
  - context role: active_offer / passive_inventory / neutral_alternatives
  - frozen Phase8A speech act + topic
  - frozen Phase7D typed reference kind
  - frozen Phase8D normalized transaction operation
  - selection surface class
  - candidate count / alternative structure

Gold is consulted only afterward to show which semantic combinations correspond
to true reference vs non-reference.

No case ID or label participates in feature extraction.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

PROOF_BASENAME = "voiceprobe_semanticlab_v2_phase7c_reference_composition_proof_v1.py"
EXPECTED_PROOF_SHA256 = "bd5e0eaeece9fe2043f5f504e1f2832d4c6b7e996d4acb9e5fedfe138a1aee1e"

ACTIVE_OFFER_RE = re.compile(
    r"\b(?:can|could|may)\s+(?:offer|do|make|schedule)\b"
    r"|\bwould\b.{0,100}\bwork\b"
    r"|\b(?:available|availability|openings?|slots?)\b",
    re.I,
)
PASSIVE_LIST_RE = re.compile(
    r"\blisted\b|\bboth\s+listed\b|\bare\s+both\s+listed\b",
    re.I,
)
I_HAVE_RE = re.compile(r"^\s*i\s+have\b", re.I)

EXPLICIT_CHOICE_RE = re.compile(
    r"\b(?:book|schedule|take|choose|pick|use|prefer)\b"
    r"|\bgo\s+with\b"
    r"|\blet'?s\s+do\b"
    r"|\bother\s+one\b",
    re.I,
)
COMPARATIVE_SELECTION_RE = re.compile(
    r"\b(?:first|second|1st|2nd|former|latter|earliest|latest|earlier|later|sooner)\b",
    re.I,
)
TENTATIVE_EVAL_RE = re.compile(
    r"\b(?:seems?|appears?|maybe|possibly|reasonable|acceptable)\b",
    re.I,
)
ACCEPTANCE_RE = re.compile(
    r"\b(?:works?|fine|good|better|okay|ok|sounds?\s+(?:good|better|fine))\b",
    re.I,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_named(basename: str) -> Path:
    candidates = [
        Path("/mnt/c/Users/llehs/Downloads") / basename,
        Path.cwd() / basename,
        Path(__file__).resolve().parent / basename,
    ]
    checked = []
    for p in candidates:
        rp = p.expanduser().resolve()
        checked.append(str(rp))
        if rp.is_file():
            return rp
    raise SystemExit(
        "Could not locate "
        + basename
        + ". Checked: "
        + ", ".join(checked)
    )


def context_role(context) -> str:
    text = " || ".join(str(x) for x in context)
    if ACTIVE_OFFER_RE.search(text):
        return "active_offer"
    if PASSIVE_LIST_RE.search(text):
        return "passive_inventory"
    if I_HAVE_RE.search(text):
        return "passive_inventory"
    return "neutral_alternatives"


def selection_surface(turn: str) -> str:
    turn = str(turn)
    if COMPARATIVE_SELECTION_RE.search(turn):
        return "comparative_or_ordinal"
    if EXPLICIT_CHOICE_RE.search(turn):
        return "explicit_choice_action"
    if TENTATIVE_EVAL_RE.search(turn):
        return "tentative_evaluation"
    if ACCEPTANCE_RE.search(turn):
        return "acceptance"
    return "generic_deictic"


def runtime_items(rows, proof):
    return [
        SimpleNamespace(
            context=proof.context_of(r),
            utterance=proof.turn_text(r),
        )
        for r in rows
    ]


def main() -> int:
    print("========== REFERENCE MULTI-OPTION BOUNDARY LOCALIZER V2 ==========")
    print("telephony=DISABLED")
    print("training=NO")
    print("candidate_promoted=NO")
    print("ambiguity_v7_modified=NO")
    print("oos_modified=NO")
    print("runtime_wiring_modified=NO")
    print("gold_visible_to_feature_extraction=NO")

    proof_path = resolve_named(PROOF_BASENAME)
    proof_hash = sha256_file(proof_path)
    print("reference_proof_v1_source=", proof_path)
    print("reference_proof_v1_sha256=", proof_hash)
    if proof_hash != EXPECTED_PROOF_SHA256:
        raise RuntimeError(
            f"Reference proof V1 drift expected={EXPECTED_PROOF_SHA256} actual={proof_hash}"
        )

    proof = load_mod("reference_boundary_proof_v1", proof_path)

    # Reuse proof V1's exact frozen dependency chain.
    v7_path = proof.resolve_named(proof.V7_BASENAME)
    if sha256_file(v7_path) != proof.EXPECTED_V7_SHA256:
        raise RuntimeError("V7 source drift")
    v7 = load_mod("reference_boundary_v7", v7_path)

    sep_path = v7.resolve_named(None, v7.SEP_BASENAME)
    sep = load_mod("reference_boundary_sep", sep_path)
    loc_path = sep.resolve_named(None, sep.LOCALIZER_BASENAME)
    loc = load_mod("reference_boundary_loc", loc_path)
    app_path = loc.resolve_named(None, loc.APP_BASENAME)
    app = load_mod("reference_boundary_app", app_path)
    comp_path = app.resolve_named(None, app.COMP_BASENAME)
    comp = load_mod("reference_boundary_comp", comp_path)

    v52_path = comp.resolve_named(None, comp.V52_BASENAME)
    feas_path = comp.resolve_named(None, comp.FEAS_BASENAME)
    v52 = load_mod("reference_boundary_v52", v52_path)
    feas = load_mod("reference_boundary_feas", feas_path)
    base = v52.base

    source_before = base.source_snapshot()
    watched = {
        "p7c": base.P7C,
        "a7c": base.A7C,
        "p7d": base.P7D,
        "a7d": base.A7D,
        "p7h": base.P7H,
        "a7i": base.A7I,
        "p7jr": base.P7JR,
        "p8a": base.P8A,
        "a8a3": base.A8A3,
        "p8dn": base.P8DN,
    }
    watched_before = {k: sha256_file(v) for k, v in watched.items()}

    random.seed(v52.SEED)
    torch.manual_seed(v52.SEED)
    gate_ck, gate_model, gate_tok = v52.load_current_model()
    thresholds = {
        f: float(gate_ck["thresholds"][f])
        for f in v52.FIELDS
    }

    original_train, original_val = v52.build_synthetic()
    original_train, original_val, _blocked, _blocked_files = (
        feas.filter_original_synthetic(v52, original_train, original_val)
    )

    # Preserve established capture order.
    v52.capture_features(
        gate_model,
        gate_tok,
        v52.runtime_for_examples(original_train),
        thresholds,
    )
    _, _, _, oval_base, _, _ = v52.capture_features(
        gate_model,
        gate_tok,
        v52.runtime_for_examples(original_val),
        thresholds,
    )

    groups, exposed = v52.load_groups()
    established = [c for _, cases in groups for c in cases]
    historical = list(v52.load_semanticlab_cases())
    targeted = list(feas.build_targeted_validation())
    probes = list(feas.build_metamorphic_probes())

    def capture(rows, are_cases):
        runtime = (
            v52.runtime_for_cases(rows)
            if are_cases
            else v52.runtime_for_examples(rows)
        )
        return v52.capture_features(
            gate_model, gate_tok, runtime, thresholds
        )

    _, _, _, hist_base, _, _ = capture(historical, True)
    _, _, _, est_base, _, _ = capture(established, True)
    _, _, _, exp_base, _, _ = capture(exposed, True)
    _, _, _, tgt_base, _, _ = capture(targeted, False)
    _, _, _, prb_base, _, _ = capture(probes, False)

    p7d = base.load_mod("reference_boundary_p7d", base.P7D)
    p7jr = base.load_mod("reference_boundary_p7jr", base.P7JR)
    p7h = base.load_mod("reference_boundary_p7h", base.P7H)
    p8a = base.load_mod("reference_boundary_p8a", base.P8A)
    p8dn = base.load_mod("reference_boundary_p8dn", base.P8DN)

    app.install_extended_candidate_coverage(p7jr, p7h)

    ck7d = base.load_checkpoint(base.A7D)
    ref_model = p7d.RefKindModel()
    ref_model.load_state_dict(ck7d["state_dict"])
    ref_model.eval()
    ref_tok = base.tokenizer_for(
        ck7d.get(
            "model_name",
            getattr(p7d, "MODEL_NAME", "distilbert/distilbert-base-uncased"),
        )
    )

    ck8a = base.load_checkpoint(base.A8A3)
    dense_model = p8a.DenseModel()
    dense_model.load_state_dict(ck8a["state_dict"])
    dense_model.eval()
    dense_tok = base.tokenizer_for(
        ck8a.get(
            "model_name",
            getattr(p8a, "MODEL_NAME", "distilbert/distilbert-base-uncased"),
        )
    )
    valid_pairs = tuple(tuple(x) for x in ck8a["valid_pairs"])
    print("FROZEN_SPECIALIST_LOAD=PASS")

    datasets = [
        ("original_val", original_val, False, oval_base),
        ("historical133", historical, True, hist_base),
        ("established1146", established, True, est_base),
        ("exposed120", exposed, True, exp_base),
        ("targeted", targeted, False, tgt_base),
        ("metamorphic", probes, False, prb_base),
    ]

    # ---------------- INFERENCE-ONLY FEATURE EXTRACTION ----------------
    records_by_dataset = {}
    for name, rows, are_cases, base_pred in datasets:
        ref_kinds, _ref_probs = proof.predict_ref_kinds(
            rows, p7d, ref_model, ref_tok
        )

        evidence = [
            proof.structural_reference_evidence(
                row,
                kind,
                v7,
                p7jr,
                p7h,
                loc,
            )
            for row, kind in zip(rows, ref_kinds)
        ]

        pairs, _independent, acts, topics = p8a.predict(
            dense_model,
            dense_tok,
            runtime_items(rows, proof),
            valid_pairs,
        )

        baseline = [int(x) for x in base_pred[:, 0].tolist()]
        candidate, reasons = proof.compose_reference_candidate(
            baseline, evidence
        )

        recs = []
        for i, row in enumerate(rows):
            if reasons[i] != "suppress:multi_option_without_unique_resolution":
                continue

            option = evidence[i]["option"]
            recs.append({
                "index": i,
                "baseline": baseline[i],
                "candidate": int(candidate[i]),
                "phase7d": str(ref_kinds[i]),
                "phase8a_pair": tuple(str(x) for x in pairs[i]),
                "phase8a_act": str(acts[i]),
                "phase8a_topic": str(topics[i]),
                "transaction_operation": str(
                    p8dn.normalize_operation(proof.turn_text(row))
                ),
                "context_role": context_role(proof.context_of(row)),
                "selection_surface": selection_surface(proof.turn_text(row)),
                "candidate_count": int(option.get("candidate_count", 0)),
                "alternative_structure": bool(option.get("alternative_structure", False)),
                "option_reason": str(option.get("reason", "")),
                "turn": proof.turn_text(row),
                "context": proof.context_of(row),
            })
        records_by_dataset[name] = recs

    print("BOUNDARY_FEATURE_EXTRACTION_COMPLETE=YES")
    print("gold_consulted_before_boundary_features=NO")

    # -------------------------- GOLD BOUNDARY --------------------------
    print("\n========== GOLD SCORING BEGINS ONLY NOW ==========")

    all_records = []
    for name, rows, are_cases, _base_pred in datasets:
        if are_cases:
            gold = [
                int(x)
                for x in v52.gold_case_tensor(rows)[:, 0].long().tolist()
            ]
        else:
            gold = [
                int(x)
                for x in v52.gold_example_tensor(rows)[:, 0].long().tolist()
            ]

        for rec in records_by_dataset[name]:
            rec = dict(rec)
            rec["dataset"] = name
            rec["gold"] = int(gold[rec["index"]])
            rec["id"] = proof.row_id(rows[rec["index"]], rec["index"])
            all_records.append(rec)

    print("multioption_suppression_case_count=", len(all_records))
    print(
        "multioption_gold_counts=",
        dict(Counter(int(r["gold"]) for r in all_records)),
    )

    grouping = Counter(
        (
            int(r["gold"]),
            int(r["baseline"]),
            r["context_role"],
            r["phase8a_act"],
            r["phase8a_topic"],
            r["transaction_operation"],
            r["selection_surface"],
            r["phase7d"],
        )
        for r in all_records
    )
    print("MULTIOPTION_BOUNDARY_GROUP_COUNTS=")
    for key, count in sorted(
        grouping.items(),
        key=lambda kv: (-kv[1], str(kv[0])),
    ):
        print("  GROUP", {
            "count": count,
            "gold": key[0],
            "baseline": key[1],
            "context_role": key[2],
            "act": key[3],
            "topic": key[4],
            "transaction_operation": key[5],
            "selection_surface": key[6],
            "phase7d": key[7],
        })

    # Summaries that expose whether context-role / semantic-act factorization
    # separates the boundary.
    print(
        "GOLD_BY_CONTEXT_ROLE=",
        {
            role: dict(Counter(r["gold"] for r in all_records if r["context_role"] == role))
            for role in sorted({r["context_role"] for r in all_records})
        },
    )
    print(
        "GOLD_BY_SELECTION_SURFACE=",
        {
            surface: dict(Counter(r["gold"] for r in all_records if r["selection_surface"] == surface))
            for surface in sorted({r["selection_surface"] for r in all_records})
        },
    )
    print(
        "GOLD_BY_PHASE8A_ACT=",
        {
            act: dict(Counter(r["gold"] for r in all_records if r["phase8a_act"] == act))
            for act in sorted({r["phase8a_act"] for r in all_records})
        },
    )
    print(
        "GOLD_BY_TRANSACTION_OPERATION=",
        {
            op: dict(
                Counter(
                    r["gold"]
                    for r in all_records
                    if r["transaction_operation"] == op
                )
            )
            for op in sorted({r["transaction_operation"] for r in all_records})
        },
    )

    # Print every baseline-right regression candidate first, because those define
    # what V2 must preserve; then a bounded sample of true suppressions.
    would_regress = [
        r for r in all_records
        if int(r["baseline"]) == int(r["gold"]) == 1
    ]
    true_suppressions = [
        r for r in all_records
        if int(r["baseline"]) == 1 and int(r["gold"]) == 0
    ]

    print("V1_MULTIOPTION_WOULD_REGRESS_count=", len(would_regress))
    for r in would_regress:
        print("V1_MULTIOPTION_WOULD_REGRESS", r)

    print("V1_MULTIOPTION_TRUE_SUPPRESSION_count=", len(true_suppressions))
    for r in true_suppressions[:40]:
        print("V1_MULTIOPTION_TRUE_SUPPRESSION", r)

    # Pure diagnostic separability counts; these are NOT candidate policies.
    feature_keys = (
        "context_role",
        "phase8a_act",
        "phase8a_topic",
        "transaction_operation",
        "selection_surface",
        "phase7d",
    )
    collisions = []
    bucket = defaultdict(set)
    for r in all_records:
        key = tuple(r[k] for k in feature_keys)
        bucket[key].add(int(r["gold"]))
    for key, labels in bucket.items():
        if len(labels) > 1:
            collisions.append((key, tuple(sorted(labels))))

    print("SEMANTIC_FEATURE_COLLISION_COUNT=", len(collisions))
    for key, labels in collisions:
        print("SEMANTIC_FEATURE_COLLISION", {
            "features": dict(zip(feature_keys, key)),
            "gold_labels": labels,
        })

    if not collisions:
        verdict = "MULTIOPTION_BOUNDARY_V2_SEPARABLE_WITH_EXISTING_FROZEN_SEMANTICS"
        next_action = (
            "BUILD_REFERENCE_COMPOSITION_V2_USING_THE_SEPARATED_CONTEXT_ROLE_"
            "SPEECH_ACT_AND_SELECTION_BOUNDARY__DO_NOT_CHANGE_OTHER_V1_RULES"
        )
    else:
        verdict = "MULTIOPTION_BOUNDARY_V2_STILL_HAS_SEMANTIC_COLLISIONS"
        next_action = (
            "DO_NOT_PATCH__ADD_ONLY_ONE_MORE_EXISTING_STRUCTURAL_FEATURE_"
            "SUCH_AS_CONTEXT_OFFER_ROLE_OR_SELECTION_RESOLUTION_STATE_TO_THE_"
            "PRINTED_COLLISION_BUCKETS"
        )

    print("\n========== POSTFLIGHT INTEGRITY ==========")
    source_after = base.source_snapshot()
    watched_after = {k: sha256_file(v) for k, v in watched.items()}
    print(
        "source_tree_python_unchanged=",
        "YES" if source_before == source_after else "NO",
    )
    print(
        "watched_source_checkpoint_hashes_unchanged=",
        "YES" if watched_before == watched_after else "NO",
    )
    print("candidate_artifact_written=NO")
    print("runtime_wiring_modified=NO")
    print("training_performed=NO")
    print("ambiguity_v7_modified=NO")
    print("oos_modified=NO")

    print("\n========== AUTHORITATIVE MULTIOPTION BOUNDARY VERDICT ==========")
    print("MULTIOPTION_BOUNDARY_VERDICT=", verdict)
    print("NEXT_ACTION=", next_action)
    print("reference_multioption_boundary_localizer_v2_completed=YES")

    del dense_tok, dense_model
    del ref_tok, ref_model
    del gate_tok, gate_model
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
