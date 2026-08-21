#!/usr/bin/env python3
"""Read-only localizer for Phase 7J applicability baseline-right regressions.

Purpose
-------
Follow the authoritative applicability proof verdict exactly: do not promote,
train, tune, or add another ambiguity rule. Reproduce the proof and inspect only
rows that were correct under the frozen baseline but become wrong after the
semantic-applicability composition.

Primary diagnostic hypothesis
-----------------------------
Some ``option_reference`` regressions may not be applicability mistakes at all.
The structured context candidate parser can fall back to scanning provider/day/
time atoms from one sentence. That can turn one composite appointment fact into
multiple pseudo-options (for example provider + daypart, or clock time + day).
This script measures that hypothesis without changing inference.

No source file, checkpoint, runtime wiring, or artifact is written.
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

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

APP_BASENAME = "voiceprobe_semanticlab_v2_phase7c_phase7j_semantic_applicability_proof_v1.py"
EXPECTED_APP_SHA256 = "e062c20a2bea869842df68044831f5c41d239395bd40404af90d2c34f537d7ad"

# This is deliberately structural rather than benchmark-specific.  The point is
# only to tell whether a context sentence visibly presents alternatives.
ALT_SEPARATOR_RE = re.compile(
    r"(?:\b(?:or|versus|vs\.?)\b|\b(?:either|one\s+of)\b|\b(?:options?|choices?)\b.{0,30}\b(?:are|include)\b)",
    re.I,
)
AND_SEPARATOR_RE = re.compile(r"\band\b", re.I)
BOTH_LISTED_RE = re.compile(r"\b(?:both|listed|options?|choices?|offer|available)\b", re.I)


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


def visible_alternative_evidence(text: str) -> tuple[bool, str]:
    """Return whether one context sentence visibly presents alternatives.

    ``or`` is strong evidence. ``and`` is counted only with list/offer language,
    because ordinary conjunction also joins components inside one fact.
    """
    t = str(text)
    if ALT_SEPARATOR_RE.search(t):
        return True, "explicit_or_or_option_list"
    if AND_SEPARATOR_RE.search(t) and BOTH_LISTED_RE.search(t):
        return True, "and_with_list_or_offer_evidence"
    return False, "no_visible_alternative_separator"


def candidate_component_types(candidate: str, p7jr) -> tuple[str, ...]:
    """Classify candidate atoms with the resolver's own closed regex ontology."""
    c = str(candidate).strip()
    kinds: list[str] = []
    try:
        if re.search(rf"^{p7jr.PROVIDER}$", c, re.I):
            kinds.append("provider")
    except Exception:
        pass
    try:
        if re.search(rf"^{p7jr.DAY}$", c, re.I):
            kinds.append("day")
    except Exception:
        pass
    try:
        if re.search(rf"^{p7jr.TIME}$", c, re.I):
            kinds.append("time")
    except Exception:
        pass
    try:
        if re.search(rf"^{p7jr.DAYPART}$", c, re.I):
            kinds.append("daypart")
    except Exception:
        pass
    return tuple(kinds) or ("composite_or_other",)


def candidate_provenance(context, p7jr, original_benchmark_candidates) -> dict:
    ctx = tuple(str(x) for x in context)

    try:
        original = tuple(str(x) for x in original_benchmark_candidates(ctx))
    except TypeError:
        original = tuple(str(x) for x in original_benchmark_candidates(ctx, None))
    except Exception as exc:
        original = (f"<ERROR:{type(exc).__name__}:{exc}>",)

    try:
        structured = tuple(str(x) for x in p7jr.structured_context_candidates(ctx))
    except Exception as exc:
        structured = (f"<ERROR:{type(exc).__name__}:{exc}>",)

    per_turn = []
    any_alt = False
    alt_reasons = []
    for turn in ctx:
        has_alt, why = visible_alternative_evidence(turn)
        any_alt = any_alt or has_alt
        alt_reasons.append(why)
        per_turn.append({"text": turn, "visible_alternatives": has_alt, "reason": why})

    component_types = [candidate_component_types(c, p7jr) for c in structured]
    multi_without_alt = len(structured) >= 2 and not any_alt

    return {
        "original_benchmark_candidates": original,
        "structured_candidates": structured,
        "structured_candidate_types": component_types,
        "context_turns": per_turn,
        "visible_alternative_anywhere": any_alt,
        "alternative_reasons": tuple(alt_reasons),
        "multi_candidate_without_visible_alternative": multi_without_alt,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--applicability-source", default=None)
    args = ap.parse_args()

    print("========== PHASE 7J APPLICABILITY REGRESSION LOCALIZER ==========")
    print("telephony=DISABLED")
    print("training=NO")
    print("repo_modified=NO")
    print("runtime_wiring_modified=NO")
    print("inference_corrections_added=NO")
    print("scope=BASELINE_RIGHT_REGRESSIONS_ONLY")

    app_path = resolve_named(args.applicability_source, APP_BASENAME)
    app_hash = sha256_file(app_path)
    print("applicability_source=", app_path)
    print("applicability_source_sha256=", app_hash)
    if app_hash != EXPECTED_APP_SHA256:
        raise RuntimeError(
            f"Applicability source drift expected={EXPECTED_APP_SHA256} actual={app_hash}"
        )

    app = load_mod("phase7c_applicability_regression_localizer_app", app_path)

    comp_path = app.resolve_named(None, app.COMP_BASENAME)
    comp_hash = sha256_file(comp_path)
    print("composition_source=", comp_path)
    print("composition_source_sha256=", comp_hash)
    if comp_hash != app.EXPECTED_COMP_SHA256:
        raise RuntimeError(
            f"Composition source drift expected={app.EXPECTED_COMP_SHA256} actual={comp_hash}"
        )
    comp = load_mod("phase7c_applicability_regression_localizer_comp", comp_path)

    # Resolve the same exact dependency chain as the applicability proof.
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

    v52 = load_mod("phase7c_applicability_regression_localizer_v52", v52_path)
    feas = load_mod("phase7c_applicability_regression_localizer_feas", feas_path)
    struct = load_mod("phase7c_applicability_regression_localizer_struct", struct_path)
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

    p7j = base.load_mod("phase7c_applicability_regression_localizer_p7j", base.P7J)
    p7jr = base.load_mod("phase7c_applicability_regression_localizer_p7jr", base.P7JR)
    p7h = base.load_mod("phase7c_applicability_regression_localizer_p7h", base.P7H)
    p8dn = base.load_mod("phase7c_applicability_regression_localizer_p8dn", base.P8DN)

    # Preserve the pre-proof benchmark parser for provenance, then install the
    # exact in-memory coverage extension used by the applicability proof.
    original_benchmark_candidates = p7h.phase7f.benchmark_candidates
    app.install_extended_candidate_coverage(p7jr, p7h)
    comp._original_temporal_resolution = comp.temporal_resolution
    comp.temporal_resolution = lambda context, turn: app.temporal_resolution_v2(comp, context, turn)

    # Pre-gold structural smoke: these are generic invented strings and verify
    # only the localizer's provenance logic.
    smoke_composite_provider = candidate_provenance(
        ("Dr. Example has a Tuesday afternoon opening.",),
        p7jr,
        original_benchmark_candidates,
    )
    smoke_true_options = candidate_provenance(
        ("I can offer Tuesday morning or Friday afternoon.",),
        p7jr,
        original_benchmark_candidates,
    )
    print("PROVENANCE_SMOKE_COMPOSITE=", smoke_composite_provider)
    print("PROVENANCE_SMOKE_OPTIONS=", smoke_true_options)
    if smoke_composite_provider["visible_alternative_anywhere"]:
        raise RuntimeError("Composite provenance smoke incorrectly detected alternatives")
    if not smoke_true_options["visible_alternative_anywhere"]:
        raise RuntimeError("Alternative provenance smoke failed")
    print("PRE_GOLD_PROVENANCE_SMOKE=PASS")

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

    all_regressions = []
    regression_causes = Counter()
    dataset_counts = Counter()
    raw_to_corrected = Counter()

    for name, rows, are_cases in datasets:
        print("\n========== LOCALIZE", name, "==========")
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
        gold, _exact_candidates = app.exact_gold_for_dataset(name, rows, are_cases, p7jr, comp)

        regressions_here = 0
        for i, row in enumerate(rows):
            baseline_right = baseline[i] == gold[i]
            candidate_right = candidate[i] == gold[i]
            if not baseline_right or candidate_right:
                continue

            regressions_here += 1
            dataset_counts[name] += 1
            raw_to_corrected[(str(raw_details[i]), str(details[i]))] += 1
            provenance = candidate_provenance(
                comp.context_of(row), p7jr, original_benchmark_candidates
            )

            if (
                str(details[i]) == "option_reference"
                and provenance["multi_candidate_without_visible_alternative"]
            ):
                cause = "OPTION_COMPONENT_SPLIT_WITHOUT_VISIBLE_ALTERNATIVE"
            elif str(raw_details[i]) != str(details[i]):
                cause = "APPLICABILITY_KIND_REROUTE"
            elif str(details[i]) == "option_reference":
                cause = "OPTION_RESOLUTION_WITH_REAL_ALTERNATIVE_STRUCTURE"
            else:
                cause = "NON_OPTION_COMPOSITION"
            regression_causes[cause] += 1

            item = {
                "dataset": name,
                "index": i,
                "id": str(getattr(row, "case_id", getattr(row, "family", ""))),
                "gold": gold[i],
                "baseline": baseline[i],
                "candidate": candidate[i],
                "raw_detail": str(raw_details[i]),
                "corrected_detail": str(details[i]),
                "correction_reason": str(correction_reasons[i]),
                "composition_reason": str(reasons[i]),
                "localized_cause": cause,
                "turn": comp.turn_text(row),
                "context": list(comp.context_of(row)),
                "candidate_provenance": provenance,
            }
            all_regressions.append(item)
            print("BASELINE_RIGHT_REGRESSION", item)

        print("baseline_right_regressions_localized=", regressions_here)

    component_split_count = regression_causes[
        "OPTION_COMPONENT_SPLIT_WITHOUT_VISIBLE_ALTERNATIVE"
    ]
    total = len(all_regressions)
    all_component_split = total > 0 and component_split_count == total

    print("\n========== REGRESSION LOCALIZATION SUMMARY ==========")
    print("total_baseline_right_regressions=", total)
    print("dataset_regression_counts=", dict(dataset_counts))
    print("cause_counts=", dict(regression_causes))
    print("raw_to_corrected_counts=", {str(k): v for k, v in raw_to_corrected.items()})
    print("all_regressions_explained_by_component_split=", "YES" if all_component_split else "NO")

    if total == 0:
        verdict = "NO_REPRODUCED_REGRESSIONS__SOURCE_OR_ENVIRONMENT_DRIFT_SUSPECTED"
        next_action = "DO_NOT_CHANGE_INFERENCE__VERIFY_REPLAY_INPUTS"
    elif all_component_split:
        verdict = "REGRESSIONS_LOCALIZED_TO_COMPOSITE_FACT_SPLIT_INTO_PSEUDO_OPTIONS"
        next_action = (
            "NEXT_PROOF_SHOULD_CHANGE_ONLY_STRUCTURED_CANDIDATE_GROUPING__"
            "REQUIRE_TRUE_OPTION_BOUNDARIES_OR_COMPOSE_PROVIDER_DAY_TIME_ATOMS_INTO_ONE_SLOT__"
            "KEEP_PHASE7J_APPLICABILITY_CORRECTIONS_UNCHANGED"
        )
    else:
        verdict = "REGRESSIONS_HAVE_MORE_THAN_ONE_CAUSE"
        next_action = (
            "DO_NOT_PATCH_YET__SEPARATE_THE_REPORTED_NON_COMPONENT_SPLIT_REGRESSIONS_"
            "FROM_CANDIDATE_GROUPING_BEFORE_ANY_CHANGE"
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

    print("\n========== AUTHORITATIVE REGRESSION LOCALIZATION VERDICT ==========")
    print("REGRESSION_LOCALIZATION_VERDICT=", verdict)
    print("NEXT_ACTION=", next_action)
    print("applicability_regression_localizer_completed=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
