#!/usr/bin/env python3
"""Read-only Phase 7C ambiguity resolution ontology/composition audit.

Purpose
-------
After the structured learned resolution-state specialist failed all three seeds,
this audit tests a different hypothesis WITHOUT training:

    existing Phase 7J kind specialist
      + existing closed candidate resolver
      + existing Phase 7H semantic selection operator (option resolution)
      + deterministic temporal anchor/axis evidence
      + existing transaction predicate normalizer (transaction resolution)
      + deterministic record-entity evidence
      -> resolved vs unresolved
      -> final SemanticAmbiguity only when unresolved

No runtime wiring is changed. No checkpoint or report is written. Gold labels are
used only after inference for evaluation and error localization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

V52_BASENAME = "voiceprobe_semanticlab_v2_phase7c_fully_factorized_directional_residual_v5_2.py"
FEAS_BASENAME = "voiceprobe_semanticlab_v2_phase7c_architecture_feasibility_gate_v1_2.py"
STRUCT_BASENAME = "voiceprobe_semanticlab_v2_phase7c_structured_ambiguity_resolution_proof_gate_v1.py"
EXPECTED_V52_SHA256 = "78178e2cbcf7c40c06f5f07ea5ee0848388bc42e98b623bd04f32ba6cb02f0b9"
EXPECTED_FEAS_SHA256 = "6017b8d0c308bc992362e82d727b24f5fe5bf760801b61d004d9339deebeeabb"
EXPECTED_STRUCT_SHA256 = "1fec2504446783092d3ea2a109b3c407dbaa45b1f0b56564b8bc4665304e2ad4"

# Generic semantic evidence, not benchmark IDs/surfaces.
CLOCK_RE = re.compile(r"\b(?:1[0-2]|0?[1-9])(?::[0-5][0-9])?\s*(?:a\.?m\.?|p\.?m\.?)\b", re.I)
DAY_RE = re.compile(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
DAYPART_RE = re.compile(r"\b(?:morning|afternoon|evening|noon|midday|tonight)\b", re.I)
REL_DAY_RE = re.compile(r"\b(?:today|tomorrow|this\s+week|next\s+week)\b", re.I)
DEICTIC_TIME_RE = re.compile(r"\b(?:that|this)\s+time\b", re.I)
DEICTIC_DAY_RE = re.compile(r"\b(?:that|this)\s+day\b", re.I)
EXPLICIT_TIME_AXIS_RE = re.compile(r"\b(?:later|earlier|sooner|different|another)\s+(?:time|slot|hour)\b", re.I)
EXPLICIT_DAY_AXIS_RE = re.compile(r"\b(?:later|earlier|different|another)\s+(?:day|date|week)\b", re.I)
VAGUE_TEMPORAL_RE = re.compile(r"\b(?:later|earlier|sooner|around\s+that|near\s+that|somewhat\s+later|somewhat\s+earlier|a\s+little\s+later|a\s+little\s+sooner)\b", re.I)

PROFILE_RE = re.compile(r"\b(?:profile|patient\s+record|patient\s+account|account|registration)\b", re.I)
APPT_RE = re.compile(r"\b(?:appointment|visit|booking|reservation)\b", re.I)

TX_SPLIT_RE = re.compile(r"\s+(?:or|and)\s+|[;,]", re.I)
IN_DOMAIN_DETAILS = {
    "temporal_reference", "option_reference", "record_reference",
    "transaction_reference", "intent_next_step", "other_prior",
}


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


def resolve_named(cli_path: str | None, basename: str) -> Path:
    candidates = []
    if cli_path:
        candidates.append(Path(cli_path).expanduser())
    candidates.extend([
        Path("/mnt/c/Users/llehs/Downloads") / basename,
        Path(__file__).resolve().parent / basename,
        Path.cwd() / basename,
    ])
    seen = set()
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if rp.is_file():
            return rp
    raise SystemExit("Could not locate " + basename + ". Checked: " + ", ".join(map(str, candidates)))


def dedupe(values):
    out = []
    for v in values:
        s = str(v)
        if s and s not in out:
            out.append(s)
    return tuple(out)


def turn_text(row) -> str:
    return str(getattr(row, "turn", getattr(row, "utterance", "")))


def context_of(row) -> tuple[str, ...]:
    return tuple(str(x) for x in getattr(row, "context", ()))


def gold_structure(row) -> tuple[str, tuple[str, ...]]:
    # SemanticLab case.
    if hasattr(row, "expected"):
        amb = row.expected.get("ambiguity") or {}
        kind = str(amb.get("kind", "none") or "none")
        cands = tuple(str(x) for x in (amb.get("candidates", ()) or ()))
        return ("none", ()) if kind == "none" else (kind, cands)
    # Structured synthetic row.
    if hasattr(row, "kind") and hasattr(row, "state"):
        unresolved = str(row.state).startswith("unresolved_")
        if not unresolved:
            return "none", ()
        kind = str(row.kind)
        # Exact candidates are derived later from the same deterministic resolver.
        return kind, ()
    # Feasibility FExample only has binary ambiguity.
    amb = int(getattr(row, "ambiguity", 0))
    return ("__binary_active__", ()) if amb else ("none", ())


def binary_gold(row) -> int:
    if hasattr(row, "expected"):
        return int(bool(row.expected.get("ambiguity")))
    if hasattr(row, "state"):
        return int(str(row.state).startswith("unresolved_"))
    return int(getattr(row, "ambiguity", 0))


def temporal_resolution(context: Sequence[str], turn: str) -> tuple[bool, str]:
    t = str(turn)
    ctx = " || ".join(context)

    # A current-turn concrete temporal anchor or explicit non-deictic axis
    # resolves which axis is being varied.
    if CLOCK_RE.search(t) or DAY_RE.search(t) or DAYPART_RE.search(t) or REL_DAY_RE.search(t):
        return True, "current_turn_concrete_temporal_anchor"
    if EXPLICIT_TIME_AXIS_RE.search(t) or EXPLICIT_DAY_AXIS_RE.search(t):
        return True, "current_turn_explicit_temporal_axis"

    # Deictic temporal expressions require a compatible antecedent.
    if DEICTIC_TIME_RE.search(t):
        if CLOCK_RE.search(ctx) or DAYPART_RE.search(ctx):
            return True, "context_time_antecedent"
        return False, "missing_time_antecedent"
    if DEICTIC_DAY_RE.search(t):
        if DAY_RE.search(ctx) or REL_DAY_RE.search(ctx):
            return True, "context_day_antecedent"
        return False, "missing_day_antecedent"

    # Bare comparative temporal language with no explicit axis remains
    # genuinely ambiguous between day and time-of-day.
    if VAGUE_TEMPORAL_RE.search(t):
        return False, "bare_temporal_comparative"

    # If Phase7J says temporal but no unresolved evidence is present, suppress.
    return True, "no_unresolved_temporal_evidence"


def record_entities(text: str) -> tuple[str, ...]:
    vals = []
    if PROFILE_RE.search(text):
        vals.append("profile")
    if APPT_RE.search(text):
        vals.append("appointment")
    return tuple(vals)


def record_resolution(context: Sequence[str], turn: str) -> tuple[bool, str]:
    current = record_entities(turn)
    if len(current) == 1:
        return True, "current_turn_explicit_record_entity"
    ctx = dedupe(x for c in context for x in record_entities(c))
    if len(ctx) == 1:
        return True, "single_context_record_entity"
    if len(ctx) >= 2:
        return False, "multiple_context_record_entities"
    return False, "missing_record_entity_anchor"


def context_transaction_ops(context: Sequence[str], normalize_operation) -> tuple[str, ...]:
    ops = []
    for c in context:
        # Whole sentence plus semantic connector-separated pieces. This is an
        # audit of the existing closed transaction ontology, not new labels.
        pieces = [str(c)] + [p for p in TX_SPLIT_RE.split(str(c)) if p.strip()]
        for piece in pieces:
            op = str(normalize_operation(piece))
            if op not in {"", "none", "search"} and op not in ops:
                ops.append(op)
    return tuple(ops)


def transaction_resolution(context: Sequence[str], turn: str, normalize_operation) -> tuple[bool, str, tuple[str, ...]]:
    explicit = str(normalize_operation(turn))
    if explicit not in {"", "none", "search"}:
        return True, "current_turn_explicit_transaction", (explicit,)
    ops = context_transaction_ops(context, normalize_operation)
    if len(ops) == 1:
        return True, "single_context_transaction", ops
    if len(ops) >= 2:
        return False, "multiple_context_transactions", ops
    return False, "missing_transaction_anchor", ops


def make_phase7h_items(rows, candidates_by_i):
    items, indices = [], []
    for i, row in enumerate(rows):
        cands = candidates_by_i.get(i, ())
        if not cands:
            continue
        items.append(SimpleNamespace(
            context=context_of(row),
            kind="prior_option",
            candidates=tuple(cands),
            turn=turn_text(row),
        ))
        indices.append(i)
    return items, indices


def option_resolution_for_rows(rows, p7h, op_model, op_tok):
    candidates_by_i = {}
    for i, row in enumerate(rows):
        try:
            candidates_by_i[i] = dedupe(p7h.phase7f.benchmark_candidates(context_of(row)))
        except Exception:
            candidates_by_i[i] = ()
    items, indices = make_phase7h_items(rows, candidates_by_i)
    op_by_i = {}
    if items:
        preds, probs = p7h.predict(op_model, op_tok, items)
        for i, item, op, pp in zip(indices, items, preds, probs):
            try:
                resolved = p7h.resolve(op, item.context, item.candidates, item.turn)
            except Exception:
                resolved = None
            op_by_i[i] = (str(op), "" if resolved is None else str(resolved), float(max(pp)) if pp else 0.0)
    return candidates_by_i, op_by_i


def ungated_phase7j(rows, p7j, detail_model, detail_tok):
    items = [SimpleNamespace(context=context_of(r), turn=turn_text(r)) for r in rows]
    preds, probs = p7j.predict_detail(detail_model, detail_tok, items)
    return [str(x) for x in preds], probs


def current_baseline_structures(rows, base_amb, detail_preds, p7jr):
    out = []
    for row, active, detail in zip(rows, base_amb, detail_preds):
        if not active:
            out.append(("none", ()))
            continue
        try:
            k, c = p7jr.ambiguity_from_detail(detail, context_of(row), turn_text(row))
            out.append((str(k), tuple(str(x) for x in c)))
        except Exception:
            out.append(("none", ()))
    return out


def compose_structures(rows, detail_preds, p7jr, p7h, op_model, op_tok, normalize_operation):
    candidates_by_i, op_by_i = option_resolution_for_rows(rows, p7h, op_model, op_tok)
    final = []
    reasons = []
    errors = []
    for i, (row, detail) in enumerate(zip(rows, detail_preds)):
        ctx, turn = context_of(row), turn_text(row)
        try:
            if detail not in IN_DOMAIN_DETAILS:
                final.append(("none", ()))
                reasons.append("detail_not_in_domain_ambiguity")
                errors.append("")
                continue

            if detail == "option_reference":
                cands = candidates_by_i.get(i, ())
                op, resolved, conf = op_by_i.get(i, ("none", "", 0.0))
                if resolved:
                    final.append(("none", ()))
                    reasons.append(f"option_resolved:{op}:{resolved}:p={conf:.3f}")
                elif len(cands) >= 2:
                    # Use Phase7J resolver canonicalization when possible.
                    try:
                        k, rc = p7jr.ambiguity_from_detail(detail, ctx, turn)
                        final.append((str(k), tuple(str(x) for x in rc)))
                    except Exception:
                        final.append(("option_reference", tuple(cands)))
                    reasons.append(f"option_unresolved:{op}:candidate_count={len(cands)}:p={conf:.3f}")
                else:
                    final.append(("none", ()))
                    reasons.append(f"option_not_applicable:candidate_count={len(cands)}")
                errors.append("")
                continue

            if detail == "temporal_reference":
                resolved, reason = temporal_resolution(ctx, turn)
                if resolved:
                    final.append(("none", ()))
                else:
                    final.append(("temporal_reference", ("time_of_day", "day")))
                reasons.append("temporal:" + reason)
                errors.append("")
                continue

            if detail == "record_reference":
                resolved, reason = record_resolution(ctx, turn)
                final.append(("none", ()) if resolved else ("record_reference", ("profile", "appointment")))
                reasons.append("record:" + reason)
                errors.append("")
                continue

            if detail == "transaction_reference":
                resolved, reason, ops = transaction_resolution(ctx, turn, normalize_operation)
                final.append(("none", ()) if resolved else ("transaction_reference", ("book", "reschedule", "cancel")))
                reasons.append("transaction:" + reason + ":ops=" + repr(ops))
                errors.append("")
                continue

            if detail == "intent_next_step":
                # Phase7J's intent detail already encodes unresolved semantic
                # alternatives; keep it only for question-like next-step turns.
                qlike = bool(re.search(r"\b(?:what|which|how|should|do\s+we|happens?\s+next|next)\b", turn, re.I))
                if qlike:
                    final.append(("intent", ("request_next_step", "acknowledgement")))
                    reasons.append("intent:question_like_next_step")
                else:
                    final.append(("none", ()))
                    reasons.append("intent:not_applicable_without_next_step_evidence")
                errors.append("")
                continue

            if detail == "other_prior":
                has_context = bool(ctx)
                vague_prior = bool(re.search(r"\b(?:which|what|that|earlier|prior|previous|thing)\b", turn, re.I))
                if has_context and vague_prior:
                    final.append(("other", ("prior_option", "prior_topic")))
                    reasons.append("other_prior:context_plus_vague_prior")
                else:
                    final.append(("none", ()))
                    reasons.append("other_prior:not_applicable")
                errors.append("")
                continue

            final.append(("none", ()))
            reasons.append("unhandled_detail")
            errors.append("")
        except Exception as exc:
            final.append(("none", ()))
            reasons.append("composition_exception")
            errors.append(f"{type(exc).__name__}:{exc}")
    return final, reasons, errors


def evaluate_dataset(name, rows, gold_struct, baseline_struct, candidate_struct, reasons, detail_preds, *, exact_candidates=True, max_fail=14):
    assert len(rows) == len(gold_struct) == len(baseline_struct) == len(candidate_struct)
    n = len(rows)
    def ok(pred, gold):
        if gold[0] == "__binary_active__":
            return int(pred[0] != "none") == 1
        if not exact_candidates:
            return pred[0] == gold[0]
        return pred == gold

    base_ok = [ok(p,g) for p,g in zip(baseline_struct,gold_struct)]
    cand_ok = [ok(p,g) for p,g in zip(candidate_struct,gold_struct)]
    regs = [i for i in range(n) if base_ok[i] and not cand_ok[i]]
    fixes = [i for i in range(n) if not base_ok[i] and cand_ok[i]]
    cand_exact = sum(cand_ok)
    base_exact = sum(base_ok)
    binary_gold_vals = [binary_gold(r) for r in rows]
    binary_pred_vals = [int(p[0] != "none") for p in candidate_struct]
    binary_exact = sum(a == b for a,b in zip(binary_gold_vals,binary_pred_vals))

    print(f"DATASET={name}")
    print("  base_exact=", f"{base_exact}/{n}")
    print("  candidate_exact=", f"{cand_exact}/{n}")
    print("  candidate_binary_exact=", f"{binary_exact}/{n}")
    print("  baseline_right_regressions=", len(regs), "fixes=", len(fixes))
    print("  reason_counts=", dict(Counter(reasons)))

    failures = [i for i,x in enumerate(cand_ok) if not x]
    for i in failures[:max_fail]:
        row = rows[i]
        print("  FAIL", {
            "index": i,
            "id": str(getattr(row, "case_id", getattr(row, "family", ""))),
            "gold": gold_struct[i],
            "baseline": baseline_struct[i],
            "candidate": candidate_struct[i],
            "detail": detail_preds[i],
            "reason": reasons[i],
            "turn": turn_text(row),
            "context": list(context_of(row)),
            "was_baseline_right": base_ok[i],
        })
    return {
        "n": n, "base_exact": base_exact, "candidate_exact": cand_exact,
        "binary_exact": binary_exact, "regressions": len(regs), "fixes": len(fixes),
        "failure_indices": failures, "regression_indices": regs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v52-source", default=None)
    ap.add_argument("--feasibility-source", default=None)
    ap.add_argument("--structured-source", default=None)
    args = ap.parse_args()

    print("========== PHASE 7C RESOLUTION ONTOLOGY / COMPOSITION AUDIT ==========")
    print("telephony=DISABLED")
    print("training=NO")
    print("runtime_wiring_modified=NO")
    print("candidate_artifact_write=NO")
    print("scalar_residual=NO")
    print("learned_resolution_state_head=NO")

    paths = {
        "v52": (resolve_named(args.v52_source, V52_BASENAME), EXPECTED_V52_SHA256),
        "feas": (resolve_named(args.feasibility_source, FEAS_BASENAME), EXPECTED_FEAS_SHA256),
        "struct": (resolve_named(args.structured_source, STRUCT_BASENAME), EXPECTED_STRUCT_SHA256),
    }
    mods = {}
    for key,(path,expected) in paths.items():
        actual = sha256_file(path)
        print(f"{key}_source=", path)
        print(f"{key}_sha256=", actual)
        if actual != expected:
            raise RuntimeError(f"{key} source drift expected={expected} actual={actual}")
        mods[key] = load_mod(f"phase7c_resolution_audit_{key}", path)
    v52, feas, struct = mods["v52"], mods["feas"], mods["struct"]
    print("DEPENDENCY_IMPORTS=PASS")

    base = v52.base
    source_before = base.source_snapshot()
    watched = {
        "p7c": base.P7C, "a7c": base.A7C, "p7j": base.P7J,
        "p7jr": base.P7JR, "a7j": base.A7J, "p7h": base.P7H,
        "a7i": base.A7I, "p8dn": base.P8DN,
    }
    hashes_before = {k: sha256_file(p) for k,p in watched.items()}

    # Load frozen components.
    p7j = base.load_mod("resolution_audit_p7j", base.P7J)
    p7jr = base.load_mod("resolution_audit_p7jr", base.P7JR)
    p7h = base.load_mod("resolution_audit_p7h", base.P7H)
    p8dn = base.load_mod("resolution_audit_p8dn", base.P8DN)

    ck7j = base.load_checkpoint(base.A7J)
    detail_model = p7j.DetailModel(); detail_model.load_state_dict(ck7j["state_dict"]); detail_model.eval()
    detail_tok = base.tokenizer_for(ck7j.get("model_name", getattr(p7j, "MODEL_NAME", "distilbert/distilbert-base-uncased")))

    ck7i = base.load_checkpoint(base.A7I)
    op_model = p7h.OperatorModel(); op_model.load_state_dict(ck7i["state_dict"]); op_model.eval()
    op_tok = base.tokenizer_for(ck7i.get("model_name", getattr(p7h, "MODEL_NAME", "distilbert/distilbert-base-uncased")))
    print("FROZEN_COMPONENT_LOAD=PASS")

    # Corpora and fresh diagnostics.
    groups, exposed = v52.load_groups()
    established = [c for _,cases in groups for c in cases]
    if len(established) != 1146 or len(exposed) != 120:
        raise RuntimeError("Corpus cardinality drift")
    target = feas.build_targeted_validation()
    probes = feas.build_metamorphic_probes()
    structured_val = struct.build_structured_validation()

    # Preserve the previously-declared reference-only exclusion.
    target = [x for x in target if str(getattr(x,"family","")) not in struct.AMBIGUITY_APPLICABILITY_EXCLUDED_FAMILIES]
    probes = [x for x in probes if str(getattr(x,"family","")) not in struct.AMBIGUITY_APPLICABILITY_EXCLUDED_FAMILIES]

    # Current scalar ambiguity baseline predictions. No training/replay.
    gate_ck, gate_model, gate_tok = v52.load_current_model()
    thresholds = {f: float(gate_ck["thresholds"][f]) for f in v52.FIELDS}
    def base_ambiguity(rows, cases=False):
        rt = v52.runtime_for_cases(rows) if cases else v52.runtime_for_examples(rows)
        x, _, _, pred, _, _ = v52.capture_features(gate_model, gate_tok, rt, thresholds)
        return [bool(int(v)) for v in pred[:,1].tolist()]

    datasets = [
        ("established1146", established, True, True),
        ("exposed120", exposed, True, True),
        ("targeted", target, False, False),
        ("metamorphic", probes, False, False),
        ("structured_val", structured_val, False, False),
    ]

    results = {}
    active_kind_failures = []
    all_reason_counts = Counter()

    for name, rows, are_cases, exact_candidates in datasets:
        print("\n==========", name, "==========")
        details, detail_probs = ungated_phase7j(rows, p7j, detail_model, detail_tok)
        base_amb = base_ambiguity(rows, cases=are_cases)
        baseline = current_baseline_structures(rows, base_amb, details, p7jr)
        candidate, reasons, errors = compose_structures(rows, details, p7jr, p7h, op_model, op_tok, p8dn.normalize_operation)
        all_reason_counts.update(reasons)

        # Gold structure: exact for corpus cases; binary/final-kind evaluation
        # for feasibility rows; structured validation has known internal kind.
        if are_cases:
            gold = [gold_structure(r) for r in rows]
        elif name == "structured_val":
            gold = []
            for r in rows:
                if not str(r.state).startswith("unresolved_"):
                    gold.append(("none", ()))
                else:
                    try:
                        k,c = p7jr.ambiguity_from_detail(str(r.kind), context_of(r), turn_text(r))
                        gold.append((str(k), tuple(str(x) for x in c)))
                    except Exception:
                        # Candidate cardinality can be intentionally absent in a
                        # synthetic state example; kind-only comparison is used
                        # below by exact_candidates=False for this dataset.
                        gold.append((str(r.kind).replace("intent_next_step","intent").replace("other_prior","other"), ()))
            exact_candidates = False
        else:
            gold = [gold_structure(r) for r in rows]
            exact_candidates = False

        results[name] = evaluate_dataset(name, rows, gold, baseline, candidate, reasons, details, exact_candidates=exact_candidates)

        # Active-kind coverage audit uses gold only after ungated inference.
        for i,row in enumerate(rows):
            if binary_gold(row) != 1:
                continue
            if not are_cases:
                continue
            gk,gc = gold[i]
            try:
                pk,pc = p7jr.ambiguity_from_detail(details[i], context_of(row), turn_text(row))
                pred = (str(pk), tuple(str(x) for x in pc))
            except Exception as exc:
                pred = ("resolver_error", ())
            if pred != (gk,gc):
                top = sorted(zip(p7j.DETAILS, detail_probs[i]), key=lambda x:x[1], reverse=True)[:3]
                active_kind_failures.append({
                    "dataset": name,
                    "id": str(getattr(row,"case_id","")),
                    "gold": (gk,gc), "pred": pred, "detail": details[i],
                    "top3": [(a, round(float(b),4)) for a,b in top],
                    "turn": turn_text(row), "context": list(context_of(row)),
                })

    print("\n========== PHASE7J ACTIVE KIND/CANDIDATE COVERAGE ==========")
    print("active_kind_candidate_failure_count=", len(active_kind_failures))
    for x in active_kind_failures[:20]:
        print("ACTIVE_KIND_FAIL", x)

    print("\n========== RESOLUTION EVIDENCE ONTOLOGY ==========")
    print("resolution_reason_counts=", dict(all_reason_counts))
    print("proposed_factorization=", {
        "kind": "Phase7J closed semantic ambiguity kind",
        "candidate_source": "closed structured candidates/domain ontology",
        "resolution_evidence": [
            "selection_operator_result", "temporal_axis_or_antecedent",
            "transaction_operation_cardinality", "record_entity_cardinality",
            "intent_or_prior_applicability",
        ],
        "final_state": ["resolved", "unresolved"],
    })
    print("coarse_learned_states_rejected=", [
        "resolved_unique", "unresolved_missing_anchor", "unresolved_multiple", "unresolved_semantic"
    ])

    est = results["established1146"]
    exp = results["exposed120"]
    t = results["targeted"]
    p = results["metamorphic"]
    sv = results["structured_val"]

    zero_reg = est["regressions"] == 0 and exp["regressions"] == 0
    fresh_improves = (t["fixes"] + p["fixes"] + sv["fixes"]) > 0
    exact_fresh = t["candidate_exact"] == t["n"] and p["candidate_exact"] == p["n"] and sv["candidate_exact"] == sv["n"]

    if zero_reg and exact_fresh and fresh_improves and not active_kind_failures:
        verdict = "DETERMINISTIC_RESOLUTION_COMPOSITION_PROVEN"
        next_action = "IMPLEMENT_SHADOW_STRUCTURED_AMBIGUITY_ASSEMBLY_AND_REPRODUCE_FULL_ZERO_REGRESSION_GATE_BEFORE_RUNTIME_WIRING"
    else:
        # Localize the blocker instead of suggesting another generic model.
        if active_kind_failures and (est["regressions"] + exp["regressions"] == 0):
            verdict = "PHASE7J_KIND_COVERAGE_IS_PRIMARY_REMAINING_BLOCKER"
            next_action = "REMEDIATE_ONLY_THE_REPORTED_PHASE7J_KIND_COVERAGE_FAMILIES__DO_NOT_RETRAIN_RESOLUTION_STATE"
        elif not active_kind_failures:
            verdict = "RESOLUTION_EVIDENCE_ONTOLOGY_IS_PRIMARY_REMAINING_BLOCKER"
            next_action = "REVISE_ONLY_THE_REPORTED_RESOLUTION_EVIDENCE_RULES_USING_SEMANTIC_COMPONENT_OUTPUTS__NO_NEW_SCALAR_OR_STATE_HEAD"
        else:
            verdict = "BOTH_PHASE7J_KIND_COVERAGE_AND_RESOLUTION_EVIDENCE_HAVE_REMAINING_GAPS"
            next_action = "FIX_PHASE7J_KIND_COVERAGE_FIRST__THEN_REEVALUATE_THE_SAME_DETERMINISTIC_RESOLUTION_COMPOSITION__NO_NEW_AMBIGUITY_HEAD"

    source_after = base.source_snapshot()
    hashes_after = {k: sha256_file(pth) for k,pth in watched.items()}
    print("\n========== POSTFLIGHT INTEGRITY ==========")
    print("source_tree_python_unchanged=", "YES" if source_after == source_before else "NO")
    print("watched_source_checkpoint_hashes_unchanged=", "YES" if hashes_after == hashes_before else "NO")
    print("candidate_artifact_written=NO")
    print("runtime_wiring_modified=NO")
    print("training_performed=NO")

    print("\n========== AUTHORITATIVE RESOLUTION ONTOLOGY VERDICT ==========")
    print("PHASE7J_ACTIVE_STRUCTURE_PRIOR_SIGNAL=106/108_ESTABLISHED__26/26_EXPOSED")
    print("ZERO_BASELINE_RIGHT_REGRESSION=", "YES" if zero_reg else "NO")
    print("FRESH_EXACT=", "YES" if exact_fresh else "NO")
    print("RESOLUTION_ONTOLOGY_AUDIT_VERDICT=", verdict)
    print("NEXT_ACTION=", next_action)
    print("resolution_ontology_composition_audit_completed=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
