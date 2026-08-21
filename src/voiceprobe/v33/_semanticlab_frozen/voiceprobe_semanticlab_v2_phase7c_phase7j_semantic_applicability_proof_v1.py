#!/usr/bin/env python3
"""Read-only Phase 7J kind/candidate semantic-applicability proof.

Purpose
-------
Follow the authoritative resolution-ontology audit without creating another
ambiguity model or retraining any checkpoint.

This proof makes two generic, in-memory diagnostic corrections only:

1) Candidate coverage:
   extend the existing structured appointment candidate parser to understand
   the ordinary temporal tokens ``noon`` and ``midday`` in addition to clock
   times/dayparts.

2) Kind applicability:
   use already-existing deterministic semantic evidence to reject a Phase 7J
   detail when that detail is structurally inapplicable, and redirect only to
   a semantically supported existing Phase 7J kind:

   * option_reference with <2 legal option candidates
       -> temporal_reference only when temporal ambiguity is actually unresolved
       -> other_prior only when context + vague prior-object evidence exists
   * temporal_reference with no unresolved temporal evidence
       -> transaction_reference only when context contains multiple closed
          transaction operations and the current turn is a vague action request

No benchmark case ID or gold label participates in inference. Gold is used only
post-inference for evaluation. No source file, runtime wiring, checkpoint, or
artifact is written.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

COMP_BASENAME = "voiceprobe_semanticlab_v2_phase7c_resolution_ontology_composition_audit_v1.py"
EXPECTED_COMP_SHA256 = "76ab60e37bc3741d5e2afda4151d0c797f17978cc899785a2d30c7dbbce5be95"

# Generic semantic patterns. These are not benchmark phrases or IDs.
TEMPORAL_AXIS_ALTERNATIVE_RE = re.compile(
    r"(?:\b(?:earlier|later|sooner)\b.{0,80}\b(?:or|versus|vs\.?|instead\s+of)\b.{0,80}"
    r"\b(?:different|another|earlier|later)\s+(?:day|date|week)\b)"
    r"|(?:\b(?:different|another|earlier|later)\s+(?:day|date|week)\b.{0,80}"
    r"\b(?:or|versus|vs\.?|instead\s+of)\b.{0,80}\b(?:earlier|later|sooner)\b)",
    re.I | re.S,
)
NON_TEMPORAL_PRIOR_OBJECT_RE = re.compile(
    r"\b(?:earlier|later|previous|prior|other)\s+(?:thing|topic|subject|option|choice)\b",
    re.I,
)
VAGUE_PRIOR_OBJECT_RE = re.compile(
    r"\b(?:which|what|that|this|other|earlier|previous|prior)\b.*\b(?:thing|topic|subject)\b"
    r"|\b(?:other|previous|prior)\s+(?:one|thing|topic|subject)\b",
    re.I | re.S,
)
VAGUE_TRANSACTION_ACTION_RE = re.compile(
    r"\b(?:proceed|go\s+ahead|continue|do\s+(?:it|that)|take\s+care\s+of\s+(?:it|that)|"
    r"perform\s+(?:it|that|the\s+action)|carry\s+(?:it|that)\s+out)\b",
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


def install_extended_candidate_coverage(p7jr, p7h):
    """Patch imported modules in memory only; never mutate source files."""
    original_time = str(p7jr.TIME)
    if "noon" not in original_time.casefold():
        p7jr.TIME = rf"(?:{original_time}|noon|midday)"

    original_canonical_piece = p7jr.canonical_piece

    def canonical_piece_extended(piece):
        value = original_canonical_piece(piece)
        if value is None:
            return None
        # Preserve normal clock canonicalization; normalize only the new lexical
        # time atoms so exact candidate values remain human-readable.
        value = re.sub(r"\bNOON\b", "noon", str(value))
        value = re.sub(r"\bMIDDAY\b", "midday", str(value))
        return value

    p7jr.canonical_piece = canonical_piece_extended

    original_benchmark_candidates = p7h.phase7f.benchmark_candidates

    def benchmark_candidates_extended(context, *args, **kwargs):
        try:
            values = tuple(str(x) for x in original_benchmark_candidates(context, *args, **kwargs))
        except TypeError:
            values = tuple(str(x) for x in original_benchmark_candidates(context))
        dedup = []
        for v in values:
            if v and v not in dedup:
                dedup.append(v)
        if len(dedup) >= 2:
            return tuple(dedup)
        fallback = tuple(str(x) for x in p7jr.structured_context_candidates(context))
        if len(fallback) >= 2:
            return fallback
        return tuple(dedup)

    p7h.phase7f.benchmark_candidates = benchmark_candidates_extended


def temporal_resolution_v2(comp, context, turn):
    """Refine the prior deterministic temporal evidence ontology."""
    text = str(turn)
    # Explicitly contrasting a temporal comparative with a different day/date
    # leaves the intended axis unresolved even though each clause contains a
    # recognizable temporal cue.
    if TEMPORAL_AXIS_ALTERNATIVE_RE.search(text):
        return False, "explicit_temporal_axis_alternative"

    # 'earlier thing/topic' is a discourse object, not temporal axis evidence.
    if NON_TEMPORAL_PRIOR_OBJECT_RE.search(text):
        concrete_temporal = bool(
            comp.CLOCK_RE.search(text)
            or comp.DAY_RE.search(text)
            or comp.DAYPART_RE.search(text)
            or comp.REL_DAY_RE.search(text)
            or comp.DEICTIC_TIME_RE.search(text)
            or comp.DEICTIC_DAY_RE.search(text)
            or comp.EXPLICIT_TIME_AXIS_RE.search(text)
            or comp.EXPLICIT_DAY_AXIS_RE.search(text)
        )
        if not concrete_temporal:
            return True, "non_temporal_prior_object"

    return comp._original_temporal_resolution(context, turn)


def other_prior_applicable(context, turn) -> bool:
    return bool(tuple(context)) and bool(VAGUE_PRIOR_OBJECT_RE.search(str(turn)))


def corrected_phase7j_details(rows, raw_details, probs, p7j, p7jr, comp, normalize_operation):
    """Apply only generic structural-applicability corrections."""
    corrected = []
    reasons = []
    details_to_i = {str(v): i for i, v in enumerate(p7j.DETAILS)}

    for row, raw_detail, pp in zip(rows, raw_details, probs):
        detail = str(raw_detail)
        context = comp.context_of(row)
        turn = comp.turn_text(row)
        reason = "argmax_kept"

        # Option ambiguity requires a legal multi-candidate set. If it does not
        # exist, only redirect when another existing kind has positive semantic
        # applicability evidence.
        if detail == "option_reference":
            try:
                option_candidates = tuple(p7jr.structured_context_candidates(context))
            except Exception:
                option_candidates = ()

            if len(option_candidates) < 2:
                temporal_resolved, temporal_reason = comp.temporal_resolution(context, turn)
                if not temporal_resolved:
                    detail = "temporal_reference"
                    reason = "option_inapplicable_to_temporal:" + temporal_reason
                elif other_prior_applicable(context, turn):
                    detail = "other_prior"
                    reason = "option_inapplicable_to_other_prior"

        # A temporal detail with no unresolved temporal evidence is allowed to
        # move to transaction_reference only when closed transaction semantics
        # independently show multiple possible operations and the turn is a
        # vague action request.
        elif detail == "temporal_reference":
            temporal_resolved, temporal_reason = comp.temporal_resolution(context, turn)
            if temporal_resolved and VAGUE_TRANSACTION_ACTION_RE.search(turn):
                tx_resolved, tx_reason, tx_ops = comp.transaction_resolution(
                    context, turn, normalize_operation
                )
                if (not tx_resolved) and len(tx_ops) >= 2:
                    detail = "transaction_reference"
                    reason = "temporal_inapplicable_to_transaction:" + tx_reason

        corrected.append(detail)
        reasons.append(reason)

    return corrected, reasons


def exact_gold_for_dataset(name, rows, are_cases, p7jr, comp):
    if are_cases:
        return [comp.gold_structure(r) for r in rows], True
    if name == "structured_val":
        gold = []
        for r in rows:
            if not str(r.state).startswith("unresolved_"):
                gold.append(("none", ()))
            else:
                try:
                    k, c = p7jr.ambiguity_from_detail(str(r.kind), comp.context_of(r), comp.turn_text(r))
                    gold.append((str(k), tuple(str(x) for x in c)))
                except Exception:
                    kind = str(r.kind).replace("intent_next_step", "intent").replace("other_prior", "other")
                    gold.append((kind, ()))
        return gold, False
    return [comp.gold_structure(r) for r in rows], False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--composition-audit-source", default=None)
    args = ap.parse_args()

    print("========== PHASE 7J SEMANTIC APPLICABILITY PROOF ==========")
    print("telephony=DISABLED")
    print("training=NO")
    print("repo_modified=NO")
    print("runtime_wiring_modified=NO")
    print("new_ambiguity_head=NO")
    print("benchmark_ids_used_for_inference=NO")

    comp_path = resolve_named(args.composition_audit_source, COMP_BASENAME)
    comp_hash = sha256_file(comp_path)
    print("composition_audit_source=", comp_path)
    print("composition_audit_sha256=", comp_hash)
    if comp_hash != EXPECTED_COMP_SHA256:
        raise RuntimeError(
            f"Composition audit source drift expected={EXPECTED_COMP_SHA256} actual={comp_hash}"
        )
    comp = load_mod("phase7c_phase7j_applicability_comp", comp_path)

    # Import the exact dependency chain already validated by the composition audit.
    v52_path = comp.resolve_named(None, comp.V52_BASENAME)
    feas_path = comp.resolve_named(None, comp.FEAS_BASENAME)
    struct_path = comp.resolve_named(None, comp.STRUCT_BASENAME)
    for label, path, expected in (
        ("v52", v52_path, comp.EXPECTED_V52_SHA256),
        ("feas", feas_path, comp.EXPECTED_FEAS_SHA256),
        ("struct", struct_path, comp.EXPECTED_STRUCT_SHA256),
    ):
        actual = sha256_file(path)
        print(f"{label}_source=", path)
        print(f"{label}_sha256=", actual)
        if actual != expected:
            raise RuntimeError(f"{label} source drift expected={expected} actual={actual}")

    v52 = load_mod("phase7c_phase7j_applicability_v52", v52_path)
    feas = load_mod("phase7c_phase7j_applicability_feas", feas_path)
    struct = load_mod("phase7c_phase7j_applicability_struct", struct_path)
    base = v52.base

    source_before = base.source_snapshot()
    watched = {
        "p7c": base.P7C, "a7c": base.A7C, "p7j": base.P7J,
        "p7jr": base.P7JR, "a7j": base.A7J, "p7h": base.P7H,
        "a7i": base.A7I, "p8dn": base.P8DN,
    }
    hashes_before = {k: sha256_file(p) for k, p in watched.items()}

    p7j = base.load_mod("phase7j_applicability_p7j", base.P7J)
    p7jr = base.load_mod("phase7j_applicability_p7jr", base.P7JR)
    p7h = base.load_mod("phase7j_applicability_p7h", base.P7H)
    p8dn = base.load_mod("phase7j_applicability_p8dn", base.P8DN)

    install_extended_candidate_coverage(p7jr, p7h)
    print("candidate_coverage_extension=NOON_MIDDAY_IN_MEMORY_ONLY")

    # Preserve the old temporal function so v2 cannot recurse after monkeypatch.
    comp._original_temporal_resolution = comp.temporal_resolution
    comp.temporal_resolution = lambda context, turn: temporal_resolution_v2(comp, context, turn)

    # Runtime smoke of the exact generic boundary conditions before evaluating any gold.
    smoke = {
        "noon_candidates": tuple(p7jr.structured_context_candidates(("I have 8 AM and noon.",))),
        "axis_alternative": comp.temporal_resolution((), "Can we move it earlier, or use a different day?"),
        "non_temporal_earlier_thing": comp.temporal_resolution(("We discussed two topics.",), "Which earlier thing did you mean?"),
        "missing_time_antecedent": comp.temporal_resolution(("We are checking Saturday availability.",), "Anything else near that time?"),
        "concrete_time_antecedent": comp.temporal_resolution(("The 1 PM opening is taken.",), "Anything else around that time?"),
    }
    print("SEMANTIC_BOUNDARY_SMOKE=", smoke)
    if smoke["noon_candidates"] != ("8 AM", "noon"):
        raise RuntimeError("Noon candidate coverage smoke failed")
    if smoke["axis_alternative"][0] is not False:
        raise RuntimeError("Temporal axis-alternative smoke failed")
    if smoke["non_temporal_earlier_thing"][0] is not True:
        raise RuntimeError("Non-temporal prior-object smoke failed")
    if smoke["missing_time_antecedent"][0] is not False:
        raise RuntimeError("Missing time antecedent smoke failed")
    if smoke["concrete_time_antecedent"][0] is not True:
        raise RuntimeError("Concrete time antecedent smoke failed")
    print("PRE_GOLD_RUNTIME_SMOKE=PASS")

    ck7j = base.load_checkpoint(base.A7J)
    detail_model = p7j.DetailModel()
    detail_model.load_state_dict(ck7j["state_dict"])
    detail_model.eval()
    detail_tok = base.tokenizer_for(
        ck7j.get("model_name", getattr(p7j, "MODEL_NAME", "distilbert/distilbert-base-uncased"))
    )

    ck7i = base.load_checkpoint(base.A7I)
    op_model = p7h.OperatorModel()
    op_model.load_state_dict(ck7i["state_dict"])
    op_model.eval()
    op_tok = base.tokenizer_for(
        ck7i.get("model_name", getattr(p7h, "MODEL_NAME", "distilbert/distilbert-base-uncased"))
    )

    groups, exposed = v52.load_groups()
    established = [c for _, cases in groups for c in cases]
    if len(established) != 1146 or len(exposed) != 120:
        raise RuntimeError("Corpus cardinality drift")

    target = [
        x for x in feas.build_targeted_validation()
        if str(getattr(x, "family", "")) not in struct.AMBIGUITY_APPLICABILITY_EXCLUDED_FAMILIES
    ]
    probes = [
        x for x in feas.build_metamorphic_probes()
        if str(getattr(x, "family", "")) not in struct.AMBIGUITY_APPLICABILITY_EXCLUDED_FAMILIES
    ]
    structured_val = struct.build_structured_validation()

    gate_ck, gate_model, gate_tok = v52.load_current_model()
    thresholds = {f: float(gate_ck["thresholds"][f]) for f in v52.FIELDS}

    def base_ambiguity(rows, cases=False):
        rt = v52.runtime_for_cases(rows) if cases else v52.runtime_for_examples(rows)
        _x, _y, _m, pred, _raw, _pairs = v52.capture_features(
            gate_model, gate_tok, rt, thresholds
        )
        return [bool(int(v)) for v in pred[:, 1].tolist()]

    datasets = [
        ("established1146", established, True),
        ("exposed120", exposed, True),
        ("targeted", target, False),
        ("metamorphic", probes, False),
        ("structured_val", structured_val, False),
    ]

    results = {}
    active_kind_failures = []
    correction_counts = Counter()

    for name, rows, are_cases in datasets:
        print("\n==========", name, "==========")
        raw_details, detail_probs = comp.ungated_phase7j(rows, p7j, detail_model, detail_tok)
        details, correction_reasons = corrected_phase7j_details(
            rows, raw_details, detail_probs, p7j, p7jr, comp, p8dn.normalize_operation
        )
        correction_counts.update(correction_reasons)

        base_amb = base_ambiguity(rows, cases=are_cases)
        baseline = comp.current_baseline_structures(rows, base_amb, raw_details, p7jr)
        candidate, reasons, errors = comp.compose_structures(
            rows, details, p7jr, p7h, op_model, op_tok, p8dn.normalize_operation
        )

        gold, exact_candidates = exact_gold_for_dataset(name, rows, are_cases, p7jr, comp)
        results[name] = comp.evaluate_dataset(
            name, rows, gold, baseline, candidate, reasons, details,
            exact_candidates=exact_candidates,
        )

        changed = [
            i for i, (a, b) in enumerate(zip(raw_details, details)) if a != b
        ]
        print("  kind_corrections=", len(changed))
        for i in changed[:12]:
            print("  KIND_CORRECTION", {
                "id": str(getattr(rows[i], "case_id", getattr(rows[i], "family", ""))),
                "raw": raw_details[i],
                "corrected": details[i],
                "reason": correction_reasons[i],
                "turn": comp.turn_text(rows[i]),
                "context": list(comp.context_of(rows[i])),
            })

        # Active kind/candidate coverage is evaluated only after all inference.
        if are_cases:
            for i, row in enumerate(rows):
                if comp.binary_gold(row) != 1:
                    continue
                gk, gc = gold[i]
                try:
                    pk, pc = p7jr.ambiguity_from_detail(
                        details[i], comp.context_of(row), comp.turn_text(row)
                    )
                    pred = (str(pk), tuple(str(x) for x in pc))
                except Exception as exc:
                    pred = ("resolver_error", ())
                if pred != (gk, gc):
                    top = sorted(
                        zip(p7j.DETAILS, detail_probs[i]),
                        key=lambda x: x[1], reverse=True,
                    )[:3]
                    active_kind_failures.append({
                        "dataset": name,
                        "id": str(getattr(row, "case_id", "")),
                        "gold": (gk, gc),
                        "pred": pred,
                        "raw_detail": raw_details[i],
                        "corrected_detail": details[i],
                        "correction_reason": correction_reasons[i],
                        "top3": [(a, round(float(b), 4)) for a, b in top],
                        "turn": comp.turn_text(row),
                        "context": list(comp.context_of(row)),
                    })

    print("\n========== PHASE7J KIND/CANDIDATE COVERAGE AFTER APPLICABILITY ==========")
    print("active_kind_candidate_failure_count=", len(active_kind_failures))
    for item in active_kind_failures[:20]:
        print("ACTIVE_KIND_FAIL", item)
    print("correction_reason_counts=", dict(correction_counts))

    est = results["established1146"]
    exp = results["exposed120"]
    target_r = results["targeted"]
    probe_r = results["metamorphic"]
    struct_r = results["structured_val"]

    zero_reg = est["regressions"] == 0 and exp["regressions"] == 0
    kind_coverage = len(active_kind_failures) == 0
    fresh_exact = (
        target_r["candidate_exact"] == target_r["n"]
        and probe_r["candidate_exact"] == probe_r["n"]
        and struct_r["candidate_exact"] == struct_r["n"]
    )

    if kind_coverage and zero_reg and fresh_exact:
        verdict = "PHASE7J_KIND_AND_DETERMINISTIC_RESOLUTION_COMPOSITION_PROVEN"
        next_action = (
            "BUILD_READ_ONLY_SHADOW_FULL_SEMANTICFRAME_ASSEMBLY_WITH_THIS_ORDERING__"
            "REPRODUCE_FULL_FIELD_ZERO_REGRESSION_BEFORE_ANY_RUNTIME_WIRING"
        )
    elif kind_coverage and zero_reg:
        verdict = "PHASE7J_KIND_COVERAGE_FIXED__RESOLUTION_EVIDENCE_REMAINS"
        next_action = (
            "KEEP_THIS_KIND_COVERAGE_FIXED_IN_PROOF_ONLY__REVISE_ONLY_THE_REPORTED_"
            "DETERMINISTIC_RESOLUTION_EVIDENCE_FAILURES__NO_NEW_HEAD_AND_NO_RETRAINING"
        )
    elif kind_coverage:
        verdict = "PHASE7J_KIND_COVERAGE_FIXED_BUT_APPLICABILITY_CAUSES_STABILITY_REGRESSIONS"
        next_action = (
            "DO_NOT_PROMOTE__LOCALIZE_ONLY_THE_BASELINE_RIGHT_REGRESSIONS_FROM_THIS_"
            "APPLICABILITY_ARBITER_BEFORE_ANY_OTHER_CHANGE"
        )
    else:
        verdict = "PHASE7J_KIND_COVERAGE_STILL_NOT_PROVEN"
        next_action = (
            "DO_NOT_ADD_ANOTHER_AMBIGUITY_HEAD__ONLY_IF_THE_REMAINING_FAILURE_IS_TRUE_"
            "CLASSIFIER_CONFUSION_CONSIDER_BOUNDED_SAME_PHASE7J_KIND_REMEDIATION_WITH_"
            "FRESH_CONTRASTS_AND_ZERO_REGRESSION_SELECTION"
        )

    source_after = base.source_snapshot()
    hashes_after = {k: sha256_file(p) for k, p in watched.items()}
    print("\n========== POSTFLIGHT INTEGRITY ==========")
    print("source_tree_python_unchanged=", "YES" if source_after == source_before else "NO")
    print("watched_source_checkpoint_hashes_unchanged=", "YES" if hashes_after == hashes_before else "NO")
    print("candidate_artifact_written=NO")
    print("runtime_wiring_modified=NO")
    print("training_performed=NO")

    print("\n========== AUTHORITATIVE PHASE7J APPLICABILITY VERDICT ==========")
    print("ACTIVE_KIND_CANDIDATE_COVERAGE_PASS=", "YES" if kind_coverage else "NO")
    print("ZERO_BASELINE_RIGHT_REGRESSION=", "YES" if zero_reg else "NO")
    print("FRESH_EXACT=", "YES" if fresh_exact else "NO")
    print("PHASE7J_APPLICABILITY_PROOF_VERDICT=", verdict)
    print("NEXT_ACTION=", next_action)
    print("phase7j_semantic_applicability_proof_completed=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
