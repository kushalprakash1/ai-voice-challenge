#!/usr/bin/env python3
"""Read-only causal separator for Phase 7J applicability regressions.

Purpose
-------
The prior regression localizer proved that baseline-right regressions have more
than one cause.  This script follows its authoritative NEXT_ACTION without
changing inference: separate non-component regressions from candidate-grouping
and identify the exact architectural boundary responsible for each regression.

It distinguishes:
* frozen OOS/detail passthrough being incorrectly suppressed by the in-domain
  ambiguity composition audit;
* missing-positive-evidence applicability errors for record/transaction/intent;
* temporal applicability/evidence gaps;
* one composite appointment split into pseudo-options;
* Phase 7H vs Phase 7J candidate-source mismatch on genuine option lists;
* Phase 7H operator prediction misses when another existing operator resolves;
* true Phase 7H operator-ontology gaps when no existing operator resolves.

No source file, checkpoint, runtime wiring, or artifact is written.  Gold labels
are consulted only after all model/deterministic inference for error diagnosis.
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

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

LOCALIZER_BASENAME = "voiceprobe_semanticlab_v2_phase7c_applicability_regression_localizer_v1.py"
EXPECTED_LOCALIZER_SHA256 = "ec9ceaba2a74bcecf31509b71e48a4d2e447f979945672dd29a2b753c248c455"

OOS_DETAILS = {"oos_generic", "oos_injection", "oos_unclear"}
EXPLICIT_AXIS_PLURAL_RE = re.compile(
    r"\b(?:later|earlier|sooner|different|another)\s+"
    r"(?:days?|dates?|weeks?|times?|slots?|hours?|appointments?)\b",
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
    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path).expanduser())
    candidates.extend([
        Path("/mnt/c/Users/llehs/Downloads") / basename,
        Path(__file__).resolve().parent / basename,
        Path.cwd() / basename,
    ])
    seen: set[Path] = set()
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if rp.is_file():
            return rp
    raise SystemExit(
        "Could not locate " + basename + ". Checked: " + ", ".join(map(str, candidates))
    )


def normalized_candidates(values, p7h) -> tuple[str, ...]:
    return tuple(p7h.norm(v) for v in values)


def all_operator_resolutions(p7h, context, candidates, turn) -> dict[str, str]:
    out: dict[str, str] = {}
    for op in tuple(p7h.OPERATORS):
        try:
            value = p7h.resolve(op, context, candidates, turn)
        except Exception:
            value = None
        if value is not None and str(value).strip():
            out[str(op)] = str(value)
    return out


def option_cause(
    *,
    row,
    provenance,
    actual_candidates,
    structured_candidates,
    predicted_op,
    predicted_resolved,
    p7h,
    comp,
) -> tuple[str, dict]:
    ctx = comp.context_of(row)
    turn = comp.turn_text(row)
    visible = bool(provenance["visible_alternative_anywhere"])

    actual = tuple(str(x) for x in actual_candidates)
    structured = tuple(str(x) for x in structured_candidates)
    actual_norm = normalized_candidates(actual, p7h)
    structured_norm = normalized_candidates(structured, p7h)

    actual_all = all_operator_resolutions(p7h, ctx, actual, turn)
    structured_all = all_operator_resolutions(p7h, ctx, structured, turn)
    try:
        pred_on_structured = p7h.resolve(predicted_op, ctx, structured, turn)
    except Exception:
        pred_on_structured = None

    diag = {
        "visible_alternative": visible,
        "phase7h_candidates": actual,
        "phase7j_structured_candidates": structured,
        "candidate_sets_same_ordered": actual_norm == structured_norm,
        "predicted_operator": str(predicted_op),
        "predicted_resolved_phase7h": str(predicted_resolved or ""),
        "predicted_operator_on_phase7j_candidates": "" if pred_on_structured is None else str(pred_on_structured),
        "resolving_ops_phase7h": actual_all,
        "resolving_ops_phase7j": structured_all,
    }

    # One sentence yielding multiple atoms with no visible alternative is the
    # already-proven composite-fact split class.
    if provenance["multi_candidate_without_visible_alternative"]:
        return "OPTION_COMPONENT_SPLIT_WITHOUT_VISIBLE_ALTERNATIVE", diag

    # Phase7H can aggregate candidates across context even when Phase7J's
    # structured resolver does not see a legal explicit list.  Keep that
    # separate from component splitting.
    if not visible and len(actual) >= 2:
        return "OPTION_CANDIDATE_AGGREGATION_WITHOUT_VISIBLE_ALTERNATIVE", diag

    if visible:
        # Both modules are looking at a real alternative structure, but may not
        # agree on the legal candidate units (e.g. 4 atoms vs 2 composites).
        if actual_norm != structured_norm:
            return "OPTION_PHASE7H_PHASE7J_CANDIDATE_SOURCE_MISMATCH", diag

        # Same legal candidates.  If another already-existing operator resolves
        # but the predicted one did not, the ontology is sufficient and this is
        # specifically an operator prediction miss.
        if not predicted_resolved and actual_all:
            if str(predicted_op) not in actual_all:
                return "OPTION_OPERATOR_PREDICTION_MISS", diag

        # No existing semantic operator can produce a unique candidate from the
        # agreed candidate set: this is a genuine operator ontology gap.
        if not predicted_resolved and not actual_all:
            return "OPTION_OPERATOR_ONTOLOGY_GAP", diag

        return "OPTION_RESOLUTION_OTHER", diag

    return "OPTION_OTHER_NO_VISIBLE_ALTERNATIVE", diag


def non_option_cause(*, row, detail: str, reason: str, comp, normalize_operation) -> tuple[str, dict]:
    ctx = comp.context_of(row)
    turn = comp.turn_text(row)
    diag: dict = {}

    if detail in OOS_DETAILS:
        return "OOS_PASSTHROUGH_SUPPRESSED_BY_IN_DOMAIN_COMPOSITION", {
            "detail": detail,
            "reason": reason,
        }

    if detail == "record_reference":
        resolved, record_reason = comp.record_resolution(ctx, turn)
        diag.update({
            "record_resolution": (resolved, record_reason),
            "turn_record_entities": comp.record_entities(turn),
            "context_record_entities": comp.dedupe(x for c in ctx for x in comp.record_entities(c)),
        })
        if record_reason == "missing_record_entity_anchor":
            return "RECORD_MISSING_POSITIVE_EVIDENCE_TREATED_AS_UNRESOLVED", diag
        if record_reason == "multiple_context_record_entities":
            return "RECORD_MULTIPLE_ENTITY_RESOLUTION", diag
        return "RECORD_OTHER_APPLICABILITY", diag

    if detail == "temporal_reference":
        resolved, temporal_reason = comp.temporal_resolution(ctx, turn)
        diag.update({
            "temporal_resolution": (resolved, temporal_reason),
            "explicit_axis_plural_match": bool(EXPLICIT_AXIS_PLURAL_RE.search(turn)),
            "has_context": bool(ctx),
        })
        if EXPLICIT_AXIS_PLURAL_RE.search(turn):
            return "TEMPORAL_EXPLICIT_AXIS_LEXICON_OR_MORPHOLOGY_GAP", diag
        if temporal_reason == "bare_temporal_comparative" and not ctx:
            return "TEMPORAL_COMPARATIVE_WITHOUT_ANTECEDENT_OVERACTIVATION", diag
        if temporal_reason in {"missing_time_antecedent", "missing_day_antecedent"} and not ctx:
            return "TEMPORAL_DEICTIC_WITHOUT_CONTEXT_OVERACTIVATION", diag
        return "TEMPORAL_OTHER_APPLICABILITY", diag

    if detail == "transaction_reference":
        resolved, tx_reason, ops = comp.transaction_resolution(ctx, turn, normalize_operation)
        diag.update({"transaction_resolution": (resolved, tx_reason, ops)})
        if tx_reason == "missing_transaction_anchor":
            return "TRANSACTION_MISSING_POSITIVE_EVIDENCE_TREATED_AS_UNRESOLVED", diag
        return "TRANSACTION_OTHER_APPLICABILITY", diag

    if detail == "intent_next_step":
        return "INTENT_NEXT_STEP_APPLICABILITY_TOO_BROAD", {"reason": reason}

    if detail == "other_prior":
        return "OTHER_PRIOR_CONTEXT_REQUIREMENT_TOO_STRICT", {"reason": reason, "has_context": bool(ctx)}

    return "NON_OPTION_OTHER", {"detail": detail, "reason": reason}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--localizer-source", default=None)
    args = ap.parse_args()

    print("========== PHASE 7J REGRESSION CAUSAL SEPARATOR ==========")
    print("telephony=DISABLED")
    print("training=NO")
    print("repo_modified=NO")
    print("runtime_wiring_modified=NO")
    print("inference_changes=NO")
    print("gold_used_for_inference=NO")
    print("scope=BASELINE_RIGHT_REGRESSIONS_ONLY")

    loc_path = resolve_named(args.localizer_source, LOCALIZER_BASENAME)
    loc_hash = sha256_file(loc_path)
    print("localizer_source=", loc_path)
    print("localizer_source_sha256=", loc_hash)
    if loc_hash != EXPECTED_LOCALIZER_SHA256:
        raise RuntimeError(
            f"Localizer source drift expected={EXPECTED_LOCALIZER_SHA256} actual={loc_hash}"
        )
    loc = load_mod("phase7c_regression_causal_separator_localizer", loc_path)

    app_path = loc.resolve_named(None, loc.APP_BASENAME)
    app_hash = sha256_file(app_path)
    if app_hash != loc.EXPECTED_APP_SHA256:
        raise RuntimeError(
            f"Applicability source drift expected={loc.EXPECTED_APP_SHA256} actual={app_hash}"
        )
    app = load_mod("phase7c_regression_causal_separator_app", app_path)

    comp_path = app.resolve_named(None, app.COMP_BASENAME)
    comp_hash = sha256_file(comp_path)
    if comp_hash != app.EXPECTED_COMP_SHA256:
        raise RuntimeError(
            f"Composition source drift expected={app.EXPECTED_COMP_SHA256} actual={comp_hash}"
        )
    comp = load_mod("phase7c_regression_causal_separator_comp", comp_path)

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

    v52 = load_mod("phase7c_regression_causal_separator_v52", v52_path)
    feas = load_mod("phase7c_regression_causal_separator_feas", feas_path)
    struct = load_mod("phase7c_regression_causal_separator_struct", struct_path)
    base = v52.base

    source_before = base.source_snapshot()
    watched = {
        "p7c": base.P7C,
        "a7c": base.A7C,
        "p7j": base.P7J,
        "p7jr": base.P7JR,
        "a7j": base.A7J,
        "p7h": base.P7H,
        "a7i": base.A7I,
        "p8dn": base.P8DN,
    }
    hashes_before = {k: sha256_file(p) for k, p in watched.items()}

    p7j = base.load_mod("phase7c_regression_causal_separator_p7j", base.P7J)
    p7jr = base.load_mod("phase7c_regression_causal_separator_p7jr", base.P7JR)
    p7h = base.load_mod("phase7c_regression_causal_separator_p7h", base.P7H)
    p8dn = base.load_mod("phase7c_regression_causal_separator_p8dn", base.P8DN)

    original_benchmark_candidates = p7h.phase7f.benchmark_candidates
    app.install_extended_candidate_coverage(p7jr, p7h)
    comp._original_temporal_resolution = comp.temporal_resolution
    comp.temporal_resolution = lambda context, turn: app.temporal_resolution_v2(comp, context, turn)

    # Pure helper smoke before any benchmark evaluation.
    mock_prov = {
        "visible_alternative_anywhere": True,
        "multi_candidate_without_visible_alternative": False,
    }
    mock_row = SimpleNamespace(
        context=("I can offer Monday or Friday.",),
        turn="Use the second option.",
    )
    mock_cause, mock_diag = option_cause(
        row=mock_row,
        provenance=mock_prov,
        actual_candidates=("Monday", "Friday"),
        structured_candidates=("Monday", "Friday"),
        predicted_op="none",
        predicted_resolved="",
        p7h=SimpleNamespace(
            OPERATORS=("none", "ordinal_second"),
            norm=lambda x: str(x).casefold(),
            resolve=lambda op, context, candidates, turn: (
                candidates[1] if op == "ordinal_second" and len(candidates) >= 2 else None
            ),
        ),
        comp=SimpleNamespace(
            context_of=lambda r: tuple(r.context),
            turn_text=lambda r: str(r.turn),
        ),
    )
    if mock_cause != "OPTION_OPERATOR_PREDICTION_MISS":
        raise RuntimeError(f"Causal helper smoke failed: {mock_cause} {mock_diag}")
    print("PRE_GOLD_CAUSAL_HELPER_SMOKE=PASS")

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

    cause_counts = Counter()
    dataset_counts = Counter()
    detail_counts = Counter()
    correction_regressions = Counter()
    examples_by_cause: dict[str, list[dict]] = defaultdict(list)
    total = 0

    for name, rows, are_cases in datasets:
        print("\n========== SEPARATE", name, "==========")
        raw_details, detail_probs = comp.ungated_phase7j(rows, p7j, detail_model, detail_tok)
        details, correction_reasons = app.corrected_phase7j_details(
            rows,
            raw_details,
            detail_probs,
            p7j,
            p7jr,
            comp,
            p8dn.normalize_operation,
        )

        base_amb = base_ambiguity(rows, cases=are_cases)
        baseline = comp.current_baseline_structures(rows, base_amb, raw_details, p7jr)
        candidate, reasons, _errors = comp.compose_structures(
            rows,
            details,
            p7jr,
            p7h,
            op_model,
            op_tok,
            p8dn.normalize_operation,
        )
        gold, _ = app.exact_gold_for_dataset(name, rows, are_cases, p7jr, comp)

        # Capture Phase7H candidate/operator state once per dataset, not once per
        # failure.  This avoids a costly repeated model call in diagnostics.
        candidates_by_i, op_by_i = comp.option_resolution_for_rows(rows, p7h, op_model, op_tok)

        here = 0
        for i, row in enumerate(rows):
            if baseline[i] != gold[i] or candidate[i] == gold[i]:
                continue

            here += 1
            total += 1
            dataset_counts[name] += 1
            detail = str(details[i])
            detail_counts[detail] += 1
            if str(raw_details[i]) != detail:
                correction_regressions[(str(raw_details[i]), detail, str(correction_reasons[i]))] += 1

            if detail == "option_reference":
                provenance = loc.candidate_provenance(
                    comp.context_of(row), p7jr, original_benchmark_candidates
                )
                actual = tuple(candidates_by_i.get(i, ()))
                try:
                    structured = tuple(str(x) for x in p7jr.structured_context_candidates(comp.context_of(row)))
                except Exception:
                    structured = ()
                op, resolved, conf = op_by_i.get(i, ("none", "", 0.0))
                cause, diag = option_cause(
                    row=row,
                    provenance=provenance,
                    actual_candidates=actual,
                    structured_candidates=structured,
                    predicted_op=str(op),
                    predicted_resolved=str(resolved),
                    p7h=p7h,
                    comp=comp,
                )
                diag["operator_confidence"] = round(float(conf), 6)
            else:
                cause, diag = non_option_cause(
                    row=row,
                    detail=detail,
                    reason=str(reasons[i]),
                    comp=comp,
                    normalize_operation=p8dn.normalize_operation,
                )

            cause_counts[cause] += 1
            if len(examples_by_cause[cause]) < 6:
                examples_by_cause[cause].append({
                    "dataset": name,
                    "id": str(getattr(row, "case_id", getattr(row, "family", ""))),
                    "gold": gold[i],
                    "baseline": baseline[i],
                    "candidate": candidate[i],
                    "raw_detail": str(raw_details[i]),
                    "corrected_detail": detail,
                    "correction_reason": str(correction_reasons[i]),
                    "composition_reason": str(reasons[i]),
                    "turn": comp.turn_text(row),
                    "context": list(comp.context_of(row)),
                    "diagnostic": diag,
                })

        print("baseline_right_regressions_separated=", here)

    print("\n========== CAUSAL SEPARATION SUMMARY ==========")
    print("total_baseline_right_regressions=", total)
    print("dataset_regression_counts=", dict(dataset_counts))
    print("detail_counts=", dict(detail_counts))
    print("cause_counts=", dict(cause_counts))
    print("applicability_kind_reroute_regression_counts=", {str(k): v for k, v in correction_regressions.items()})

    for cause, count in cause_counts.most_common():
        print("\nCAUSE=", cause, "count=", count)
        for item in examples_by_cause[cause]:
            print(" EXAMPLE", item)

    accounted = sum(cause_counts.values()) == total
    print("\nall_regressions_accounted_for=", "YES" if accounted else "NO")
    if not accounted:
        verdict = "SEPARATOR_INTERNAL_ACCOUNTING_FAILURE"
        next_action = "DO_NOT_CHANGE_INFERENCE__FIX_DIAGNOSTIC_ACCOUNTING"
    else:
        verdict = "REGRESSIONS_CAUSALLY_SEPARATED"
        # This is intentionally diagnostic: prioritize the semantic contract
        # layer only after counts prove which causes dominate.  Do not mutate
        # runtime in this script.
        next_action = (
            "USE_CAUSE_COUNTS_TO_BUILD_ONE_BOUNDED_COMPOSITION_V2_PROOF__"
            "PRESERVE_FROZEN_OOS__REQUIRE_POSITIVE_APPLICABILITY_EVIDENCE_PER_KIND__"
            "UNIFY_PHASE7H_PHASE7J_OPTION_CANDIDATE_UNITS__"
            "THEN_REEVALUATE_ZERO_REGRESSION_WITHOUT_NEW_AMBIGUITY_TRAINING"
        )

    source_after = base.source_snapshot()
    hashes_after = {k: sha256_file(p) for k, p in watched.items()}
    print("\n========== POSTFLIGHT INTEGRITY ==========")
    print("source_tree_python_unchanged=", "YES" if source_after == source_before else "NO")
    print(
        "watched_source_checkpoint_hashes_unchanged=",
        "YES" if hashes_after == hashes_before else "NO",
    )
    print("candidate_artifact_written=NO")
    print("runtime_wiring_modified=NO")
    print("training_performed=NO")

    print("\n========== AUTHORITATIVE CAUSAL SEPARATOR VERDICT ==========")
    print("CAUSAL_SEPARATOR_VERDICT=", verdict)
    print("NEXT_ACTION=", next_action)
    print("regression_causal_separator_completed=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
