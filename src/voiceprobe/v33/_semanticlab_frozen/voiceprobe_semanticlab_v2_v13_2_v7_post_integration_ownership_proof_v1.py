#!/usr/bin/env python3
"""Read-only V13.2 -> Ambiguity V7 post-integration ownership proof V1.

Purpose
-------
Localize the exact ambiguity regressions printed by Runtime Wiring Offline Gate V1.

This diagnostic does NOT:
- modify V13.2
- modify frozen Ambiguity V7
- modify Reference V2
- modify OOS
- modify runtime/planner/telephony
- train or update any model
- use case IDs, categories, tags, or gold labels during inference

It asks one narrow architectural question:

    When V13.2 already has an active option ambiguity, is frozen V7 suppressing
    that ambiguity because the learned Phase7H operator claims a concrete
    resolution even though deterministic strong-resolution evidence and the
    utterance surface do not support that operator?

All regression discovery is inference-derived. Gold is consulted only after
the complete V13.2 + V7 outputs have been constructed.
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
from typing import Any

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

RUNTIME_GATE_BASENAME = (
    "voiceprobe_semanticlab_v2_runtime_wiring_offline_gate_v1.py"
)
EXPECTED_RUNTIME_GATE_SHA256 = (
    "b5504d914ad4b653d4d6a25e7dbf3991b5c126bc1005569b99ff57780179b941"
)

# Generic semantic surface evidence only. These are NOT benchmark strings.
ORDINAL_FIRST_RE = re.compile(
    r"\b(?:first|1st|former)\b",
    re.IGNORECASE,
)
ORDINAL_SECOND_RE = re.compile(
    r"\b(?:second|2nd|latter)\b",
    re.IGNORECASE,
)
TEMPORAL_EARLIER_RE = re.compile(
    r"\b(?:earlier|sooner)\b",
    re.IGNORECASE,
)
TEMPORAL_LATER_RE = re.compile(
    r"\b(?:later)\b",
    re.IGNORECASE,
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
    candidates = (
        Path("/mnt/c/Users/llehs/Downloads") / basename,
        Path.cwd() / basename,
        Path(__file__).resolve().parent / basename,
    )
    checked = []
    for p in candidates:
        rp = p.expanduser().resolve()
        checked.append(str(rp))
        if rp.is_file():
            return rp
    raise SystemExit(
        "Missing required file "
        + basename
        + ". Checked: "
        + ", ".join(checked)
    )


def enum_value(value: Any) -> str:
    raw = getattr(value, "value", None)
    return str(raw if raw is not None else value)


def ambiguity_tuple(frame) -> tuple[str, tuple[str, ...]]:
    return (
        enum_value(frame.ambiguity.kind),
        tuple(str(x) for x in frame.ambiguity.candidates),
    )


def expected_ambiguity(case) -> tuple[str, tuple[str, ...]]:
    """Post-inference only helper for readable gold reporting."""
    expected = getattr(case, "expected", {})
    if not isinstance(expected, dict):
        try:
            expected = dict(expected)
        except Exception:
            expected = {}
    raw = expected.get("ambiguity") or {}
    if not isinstance(raw, dict):
        raw = {}
    return (
        str(raw.get("kind", "none")),
        tuple(str(x) for x in raw.get("candidates", ()) or ()),
    )


def explicit_support_for_operator(
    turn: str,
    op: str,
    candidates: tuple[str, ...],
) -> tuple[bool, str]:
    """Diagnostic only: does the utterance explicitly support the learned op?"""
    text = str(turn)
    folded = " ".join(text.casefold().split())
    op = str(op)

    if op == "ordinal_first":
        return bool(ORDINAL_FIRST_RE.search(text)), "explicit_first_ordinal"
    if op == "ordinal_second":
        return bool(ORDINAL_SECOND_RE.search(text)), "explicit_second_ordinal"
    if op == "temporal_earlier":
        return bool(TEMPORAL_EARLIER_RE.search(text)), "explicit_earlier"
    if op == "temporal_later":
        return bool(TEMPORAL_LATER_RE.search(text)), "explicit_later"
    if op == "literal":
        for candidate in candidates:
            c = " ".join(str(candidate).casefold().split())
            if c and c in folded:
                return True, "literal_candidate_in_turn"
        return False, "literal_candidate_absent"
    if op == "accept_single":
        return len(candidates) == 1, "single_candidate_cardinality"
    if op == "latest_provider":
        # Strong provider resolution is owned by the existing deterministic
        # resolver; this diagnostic does not invent a lexical substitute.
        return False, "no_independent_surface_claim"
    if op in {"none", ""}:
        return False, "no_operator"
    return False, "unrecognized_operator_surface"


def print_rows(title: str, rows: list[dict[str, Any]], limit: int = 30) -> None:
    print(title + "_count=", len(rows))
    for row in rows[:limit]:
        print(title, row)


def main() -> int:
    print("========== V13.2 -> V7 POST-INTEGRATION OWNERSHIP PROOF V1 ==========")
    print("telephony=DISABLED")
    print("runtime_wiring=NO")
    print("runtime_source_write=NO")
    print("training=NO")
    print("gradient_updates=NO")
    print("reference_v2_modified=NO")
    print("oos_modified=NO")
    print("v13_2_modified=NO")
    print("ambiguity_v7_modified=NO")
    print("case_id_runtime_inputs=NO")
    print("category_runtime_inputs=NO")
    print("tags_runtime_inputs=NO")
    print("gold_runtime_inputs=NO")

    gate_path = resolve_named(RUNTIME_GATE_BASENAME)
    gate_sha = sha256_file(gate_path)
    print("runtime_gate_source=", gate_path)
    print("runtime_gate_sha256=", gate_sha)
    if gate_sha != EXPECTED_RUNTIME_GATE_SHA256:
        raise RuntimeError(
            "Runtime gate drift: "
            f"expected={EXPECTED_RUNTIME_GATE_SHA256} actual={gate_sha}"
        )
    rt = load_mod("v13v7_localizer_runtime_gate", gate_path)

    combined_path = rt.resolve_named(rt.COMBINED_SHADOW_BASENAME)
    combined_sha = sha256_file(combined_path)
    if combined_sha != rt.EXPECTED_COMBINED_SHADOW_SHA256:
        raise RuntimeError("Combined shadow source drift")
    combined = load_mod("v13v7_localizer_combined", combined_path)

    v7_path = combined.resolve_named(combined.V7_BASENAME)
    if sha256_file(v7_path) != combined.EXPECTED_V7_SHA256:
        raise RuntimeError("V7 source drift")
    v7 = load_mod("v13v7_localizer_v7", v7_path)

    sep_path = v7.resolve_named(None, v7.SEP_BASENAME)
    if sha256_file(sep_path) != v7.EXPECTED_SEP_SHA256:
        raise RuntimeError("V7 separator source drift")
    sep = load_mod("v13v7_localizer_sep", sep_path)

    loc_path = sep.resolve_named(None, sep.LOCALIZER_BASENAME)
    if sha256_file(loc_path) != sep.EXPECTED_LOCALIZER_SHA256:
        raise RuntimeError("V7 localizer dependency drift")
    loc = load_mod("v13v7_localizer_loc", loc_path)

    app_path = loc.resolve_named(None, loc.APP_BASENAME)
    if sha256_file(app_path) != loc.EXPECTED_APP_SHA256:
        raise RuntimeError("V7 applicability source drift")
    app = load_mod("v13v7_localizer_app", app_path)

    comp_path = app.resolve_named(None, app.COMP_BASENAME)
    if sha256_file(comp_path) != app.EXPECTED_COMP_SHA256:
        raise RuntimeError("V7 composition source drift")
    comp = load_mod("v13v7_localizer_comp", comp_path)

    v52_path = comp.resolve_named(None, comp.V52_BASENAME)
    feas_path = comp.resolve_named(None, comp.FEAS_BASENAME)
    if sha256_file(v52_path) != comp.EXPECTED_V52_SHA256:
        raise RuntimeError("V5.2 source drift")
    if sha256_file(feas_path) != comp.EXPECTED_FEAS_SHA256:
        raise RuntimeError("Feasibility/OOS source drift")
    v52 = load_mod("v13v7_localizer_v52", v52_path)
    feas = load_mod("v13v7_localizer_feas", feas_path)
    base = v52.base

    full_path = combined.resolve_named(combined.FULL_BASENAME)
    if sha256_file(full_path) != combined.EXPECTED_FULL_SHA256:
        raise RuntimeError("Full evaluator source drift")
    full = load_mod("v13v7_localizer_full", full_path)

    v13_2_path = rt.resolve_named(rt.V13_2_BASENAME)
    if sha256_file(v13_2_path) != rt.EXPECTED_V13_2_SHA256:
        raise RuntimeError("V13.2 source drift")
    v13_2 = load_mod("v13v7_localizer_v13_2", v13_2_path)

    boundary_path = combined.resolve_named(combined.REF_BOUNDARY_BASENAME)
    if sha256_file(boundary_path) != combined.EXPECTED_REF_BOUNDARY_SHA256:
        raise RuntimeError("Reference boundary source drift")
    boundary = load_mod("v13v7_localizer_boundary", boundary_path)

    source_before = base.source_snapshot()
    full_source_before = full.source_snapshot()
    watched = {
        "p7c": base.P7C,
        "a7c": base.A7C,
        "p7d": base.P7D,
        "a7d": base.A7D,
        "p7j": base.P7J,
        "p7jr": base.P7JR,
        "a7j": base.A7J,
        "p7h": base.P7H,
        "a7i": base.A7I,
        "p8a": base.P8A,
        "a8a3": base.A8A3,
        "p8dn": base.P8DN,
    }
    watched_before = {k: sha256_file(v) for k, v in watched.items()}

    # ------------------------------------------------------------------
    # Exact frozen Phase7C + OOS replay reconstruction.
    #
    # This ordering intentionally mirrors Runtime Wiring Offline Gate V1.
    # V1 incorrectly passed all-False OOS values into V7, which manufactured
    # out-of-scope ambiguity regressions. V1.1 restores the authoritative
    # epoch-52/scale-.9 frozen OOS state before localizing V7.
    # ------------------------------------------------------------------
    print("\n========== EXACT FROZEN OOS RECONSTRUCTION ==========")
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

    orig_x, _, orig_margin, _orig_base, _, _ = feas.capture(
        v52,
        gate_model,
        gate_tok,
        original_train,
        thresholds,
    )
    orig_y = v52.gold_example_tensor(original_train)

    # Replay-sensitive ordering from the proven runtime gate.
    feas.capture(
        v52,
        gate_model,
        gate_tok,
        original_val,
        thresholds,
    )
    historical = list(v52.load_semanticlab_cases())
    v52.capture_features(
        gate_model,
        gate_tok,
        v52.runtime_for_cases(historical),
        thresholds,
    )

    replay_head = v52.DirectionalFactorizedResidual(orig_x.shape[1])
    frozen_oos_head = feas.reconstruct_frozen_oos(
        v52,
        replay_head.oos,
        orig_x,
        orig_y,
        orig_margin,
        original_train,
    )

    # ------------------------------------------------------------------
    # Established cases. Case IDs are loaded but never used to route
    # inference. All 1,146 cases receive exactly the same pipeline.
    # ------------------------------------------------------------------
    groups, _exposed = v52.load_groups()
    cases = [
        case
        for _label, group_cases in groups
        for case in group_cases
    ]
    if len(cases) != 1146:
        raise RuntimeError(
            f"Established corpus cardinality drift: {len(cases)}"
        )

    est_x, _, est_margin, _est_base_pred, _, _ = v52.capture_features(
        gate_model,
        gate_tok,
        v52.runtime_for_cases(cases),
        thresholds,
    )
    est_oos, _ = feas.frozen_oos_pred(
        frozen_oos_head,
        est_x,
        est_margin,
    )
    print("frozen_oos_inference_complete=YES")
    print("gold_consulted=NO")

    runtime = [
        full.RuntimeTurn(
            context=tuple(case.context),
            utterance=str(case.utterance),
        )
        for case in cases
    ]

    print("\n========== V13.2 BASELINE INFERENCE ==========")
    checkpoints = full.validate_environment()
    raw = full.assemble_level2(runtime, checkpoints)

    (
        v2_frames,
        v2_schedules,
        _dense2,
        _facts2,
        _refs2,
        _diag2,
        v2_constructor_errors,
    ) = v13_2.v2.construct_candidate_frames(
        runtime,
        raw,
        checkpoints,
    )
    (
        v13_frames,
        _v13_schedules,
        _v13_diag,
        v13_constructor_errors,
    ) = v13_2.construct_v13_2_frames(
        runtime,
        raw,
        checkpoints,
        v2_frames,
        v2_schedules,
    )

    print("established_cases=", len(cases))
    print("V2_CONSTRUCTOR_ERROR_count=", len(v2_constructor_errors))
    print("V13_2_CONSTRUCTOR_ERROR_count=", len(v13_constructor_errors))
    print("gold_consulted=NO")

    # ------------------------------------------------------------------
    # Exact frozen V7 inference over V13.2 baseline ambiguity.
    # Frozen OOS is passed exactly as in the authoritative runtime gate so
    # generic/injection ownership cannot contaminate this ambiguity localizer.
    # ------------------------------------------------------------------
    print("\n========== FROZEN V7 OPTION-RESOLUTION INFERENCE ==========")
    p7j = base.load_mod("v13v7_localizer_p7j", base.P7J)
    p7jr = base.load_mod("v13v7_localizer_p7jr", base.P7JR)
    p7h = base.load_mod("v13v7_localizer_p7h", base.P7H)
    p8dn = base.load_mod("v13v7_localizer_p8dn", base.P8DN)

    app.install_extended_candidate_coverage(p7jr, p7h)
    comp._original_temporal_resolution = comp.temporal_resolution
    comp.temporal_resolution = (
        lambda context, turn:
        app.temporal_resolution_v2(comp, context, turn)
    )

    ck7j = base.load_checkpoint(base.A7J)
    detail_model = p7j.DetailModel()
    detail_model.load_state_dict(ck7j["state_dict"])
    detail_model.eval()
    detail_tok = base.tokenizer_for(
        ck7j.get(
            "model_name",
            getattr(
                p7j,
                "MODEL_NAME",
                "distilbert/distilbert-base-uncased",
            ),
        )
    )

    ck7i = base.load_checkpoint(base.A7I)
    op_model = p7h.OperatorModel()
    op_model.load_state_dict(ck7i["state_dict"])
    op_model.eval()
    op_tok = base.tokenizer_for(
        ck7i.get(
            "model_name",
            getattr(
                p7h,
                "MODEL_NAME",
                "distilbert/distilbert-base-uncased",
            ),
        )
    )

    raw_details, detail_probs = comp.ungated_phase7j(
        cases,
        p7j,
        detail_model,
        detail_tok,
    )
    corrected_details, correction_reasons = (
        app.corrected_phase7j_details(
            cases,
            raw_details,
            detail_probs,
            p7j,
            p7jr,
            comp,
            p8dn.normalize_operation,
        )
    )

    final_details = []
    arbitration_reasons = []
    for row, detail in zip(cases, corrected_details):
        corrected, why = v7.final_capability_kind_arbitration(
            row,
            detail,
            p7jr,
            comp,
            p8dn.normalize_operation,
        )
        final_details.append(str(corrected))
        arbitration_reasons.append(str(why))

    ref_kinds = [str(x) for x in raw.reference_kinds]
    dense_pairs = [
        tuple(str(x) for x in pair)
        for pair in raw.dense_pairs
    ]
    v13_ambiguity = [ambiguity_tuple(frame) for frame in v13_frames]

    strict, strict_reasons, candidate_units, op_by_i = v7.compose_v3(
        cases,
        final_details,
        ref_kinds,
        dense_pairs,
        [bool(int(x)) for x in est_oos.tolist()],
        v13_ambiguity,
        p7jr,
        p7h,
        op_model,
        op_tok,
        comp,
        app,
        sep,
        loc,
        p8dn.normalize_operation,
    )
    safe, safe_reasons = v7.apply_stability_wrapper(
        v13_ambiguity,
        strict,
        strict_reasons,
    )

    # --------------------------------------------------------------
    # BOUNDED POST-V7 OWNERSHIP CANDIDATE
    #
    # V13.2 owns an already-active option ambiguity unless V7 has a
    # justified positive resolution proof.  This candidate does not modify
    # V7 itself; it is a post-composition ownership rule tested in shadow.
    # --------------------------------------------------------------
    ownership_candidate = []
    ownership_reasons = []
    ownership_restores = []

    for i, (
        case,
        before,
        after,
        strict_reason,
    ) in enumerate(
        zip(
            cases,
            v13_ambiguity,
            safe,
            safe_reasons,
        )
    ):
        current = after
        reason = "preserve:v7_output"

        if (
            before[0] == "option_reference"
            and after[0] == "none"
            and str(strict_reason).startswith("option:phase7h_resolved:")
        ):
            candidates = tuple()
            # compose_v3 exposes the same structured candidate units used by
            # Phase7H.  Reuse them; never re-extract from benchmark labels.
            if i in candidate_units:
                candidates = tuple(str(x) for x in candidate_units[i])

            op, resolved, conf = op_by_i.get(i, ("none", "", 0.0))
            strong_op, strong_resolved = v7.strong_option_resolution(
                str(case.utterance),
                tuple(case.context),
                candidates,
                p7h,
                p7jr,
            )
            explicit_support, support_reason = explicit_support_for_operator(
                str(case.utterance),
                str(op),
                candidates,
            )

            if not strong_resolved and not explicit_support:
                current = before
                reason = (
                    "restore:v13_2_active_option_ambiguity:"
                    "unsupported_learned_phase7h_resolution:"
                    f"op={op}:p={float(conf):.3f}:"
                    f"surface={support_reason}"
                )
                ownership_restores.append({
                    "index": i,
                    "before": before,
                    "v7_after": after,
                    "restored": current,
                    "phase7h_op": str(op),
                    "phase7h_resolved": str(resolved),
                    "phase7h_confidence": round(float(conf), 6),
                    "deterministic_strong_op": str(strong_op),
                    "deterministic_strong_resolved": str(strong_resolved),
                    "explicit_surface_support": bool(explicit_support),
                    "surface_support_reason": str(support_reason),
                    "candidates": candidates,
                    "turn": str(case.utterance),
                    "context": tuple(str(x) for x in case.context),
                })

        ownership_candidate.append(current)
        ownership_reasons.append(reason)

    print("v7_inference_complete=YES")
    print("frozen_oos_active_count=", int(est_oos.sum().item()))
    print("gold_consulted=NO")

    # ------------------------------------------------------------------
    # Discover the integration surface WITHOUT gold.
    # ------------------------------------------------------------------
    active_to_none = []
    all_active_changes = []
    learned_resolution_without_surface = []
    learned_resolution_with_surface = []
    strong_resolution_suppressions = []

    for i, (
        case,
        before,
        strict_value,
        safe_value,
        detail,
        strict_reason,
        safe_reason,
    ) in enumerate(
        zip(
            cases,
            v13_ambiguity,
            strict,
            safe,
            final_details,
            strict_reasons,
            safe_reasons,
        )
    ):
        if before[0] == "none":
            continue

        if safe_value != before:
            all_active_changes.append(i)

        if not (
            before[0] == "option_reference"
            and safe_value[0] == "none"
        ):
            continue

        candidates = tuple(candidate_units.get(i, ()))
        op, resolved, conf = op_by_i.get(i, ("none", "", 0.0))
        strong_op, strong_resolved = v7.strong_option_resolution(
            str(case.utterance),
            tuple(case.context),
            candidates,
            p7h,
            p7jr,
        )
        explicit_support, support_reason = explicit_support_for_operator(
            str(case.utterance),
            str(op),
            candidates,
        )

        record = {
            "index": i,
            "v13_2_before": before,
            "v7_strict": strict_value,
            "v7_safe": safe_value,
            "detail": str(detail),
            "strict_reason": str(strict_reason),
            "safe_reason": str(safe_reason),
            "phase7h_op": str(op),
            "phase7h_resolved": str(resolved),
            "phase7h_confidence": round(float(conf), 6),
            "deterministic_strong_op": str(strong_op),
            "deterministic_strong_resolved": str(strong_resolved),
            "explicit_surface_support": bool(explicit_support),
            "surface_support_reason": str(support_reason),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "context_role": boundary.context_role(tuple(case.context)),
            "selection_surface": boundary.selection_surface(
                str(case.utterance)
            ),
            "phase7d": ref_kinds[i],
            "dense_pair": dense_pairs[i],
            "turn": str(case.utterance),
            "context": tuple(str(x) for x in case.context),
        }
        active_to_none.append(record)

        if str(strict_reason).startswith(
            "option:strong_resolved:"
        ):
            strong_resolution_suppressions.append(record)
        elif str(strict_reason).startswith(
            "option:phase7h_resolved:"
        ):
            if explicit_support:
                learned_resolution_with_surface.append(record)
            else:
                learned_resolution_without_surface.append(record)

    print("\n========== PRE-GOLD INTEGRATION LOCALIZATION ==========")
    print("V13_2_ACTIVE_AMBIGUITY_count=", sum(
        value[0] != "none" for value in v13_ambiguity
    ))
    print("V13_2_ACTIVE_AMBIGUITY_CHANGED_BY_V7_count=", len(all_active_changes))
    print_rows("V13_2_OPTION_ACTIVE_TO_V7_NONE", active_to_none)
    print_rows(
        "PHASE7H_SUPPRESSION_WITHOUT_EXPLICIT_SURFACE_SUPPORT",
        learned_resolution_without_surface,
    )
    print_rows(
        "PHASE7H_SUPPRESSION_WITH_EXPLICIT_SURFACE_SUPPORT",
        learned_resolution_with_surface,
    )
    print_rows(
        "DETERMINISTIC_STRONG_RESOLUTION_SUPPRESSION",
        strong_resolution_suppressions,
    )
    print_rows(
        "POST_V7_OWNERSHIP_RESTORE",
        ownership_restores,
    )

    ownership_changes_from_v13 = [
        i
        for i, (before, after)
        in enumerate(zip(v13_ambiguity, ownership_candidate))
        if before != after
    ]
    ownership_changes_from_v7 = [
        i
        for i, (before, after)
        in enumerate(zip(safe, ownership_candidate))
        if before != after
    ]
    print(
        "OWNERSHIP_CANDIDATE_CHANGE_FROM_V13_2_count=",
        len(ownership_changes_from_v13),
    )
    print(
        "OWNERSHIP_CANDIDATE_CHANGE_FROM_V7_count=",
        len(ownership_changes_from_v7),
    )

    by_signature = Counter(
        (
            row["context_role"],
            row["selection_surface"],
            row["phase7h_op"],
            row["explicit_surface_support"],
            bool(row["deterministic_strong_resolved"]),
        )
        for row in active_to_none
    )
    print("ACTIVE_TO_NONE_CAUSAL_SIGNATURE_COUNTS=")
    for key, count in sorted(
        by_signature.items(),
        key=lambda kv: (-kv[1], repr(kv[0])),
    ):
        print("  ", count, {
            "context_role": key[0],
            "selection_surface": key[1],
            "phase7h_op": key[2],
            "explicit_surface_support": key[3],
            "deterministic_strong_resolution": key[4],
        })

    # ------------------------------------------------------------------
    # GOLD only now. Report whether the runtime regressions are precisely this
    # inference-derived suppression class. No candidate repair is applied.
    # ------------------------------------------------------------------
    print("\n========== GOLD SCORING BEGINS ONLY NOW ==========")
    print("case_id_visible_to_inference=NO")
    print("expected_gold_visible_to_inference=NO")

    v13_failures = [
        combined.fields_failed(full, case, frame)
        for case, frame in zip(cases, v13_frames)
    ]

    v7_shadow_frames = []
    ownership_shadow_frames = []
    for frame, v7_amb, owned_amb in zip(
        v13_frames,
        safe,
        ownership_candidate,
    ):
        v7_shadow_frames.append(
            combined.clone_with_shadow_fields(
                full,
                frame,
                enum_value(frame.reference),
                v7_amb[0],
                v7_amb[1],
            )
        )
        ownership_shadow_frames.append(
            combined.clone_with_shadow_fields(
                full,
                frame,
                enum_value(frame.reference),
                owned_amb[0],
                owned_amb[1],
            )
        )

    v7_failures = [
        combined.fields_failed(full, case, frame)
        for case, frame in zip(cases, v7_shadow_frames)
    ]
    ownership_failures = [
        combined.fields_failed(full, case, frame)
        for case, frame in zip(cases, ownership_shadow_frames)
    ]

    def exact_count(failure_rows):
        return sum(not row for row in failure_rows)

    v13_exact = exact_count(v13_failures)
    v7_exact = exact_count(v7_failures)
    ownership_exact = exact_count(ownership_failures)

    v13_amb_exact = sum(
        not (fs & combined.AMB_FIELDS)
        for fs in v13_failures
    )
    v7_amb_exact = sum(
        not (fs & combined.AMB_FIELDS)
        for fs in v7_failures
    )
    ownership_amb_exact = sum(
        not (fs & combined.AMB_FIELDS)
        for fs in ownership_failures
    )

    v7_regressions = []
    ownership_regressions = []
    ownership_fixes_vs_v7 = []
    ownership_non_ambiguity_drift = []

    for i, (base_fs, v7_fs, own_fs) in enumerate(
        zip(v13_failures, v7_failures, ownership_failures)
    ):
        if not base_fs and v7_fs:
            v7_regressions.append({
                "index": i,
                "case_id": str(cases[i].case_id),
                "new_failures": tuple(sorted(v7_fs - base_fs)),
            })

        if not base_fs and own_fs:
            ownership_regressions.append({
                "index": i,
                "case_id": str(cases[i].case_id),
                "new_failures": tuple(sorted(own_fs - base_fs)),
            })

        if v7_fs and not own_fs:
            ownership_fixes_vs_v7.append({
                "index": i,
                "case_id": str(cases[i].case_id),
                "fixed_fields": tuple(sorted(v7_fs)),
                "ownership_reason": ownership_reasons[i],
            })

        base_non_amb = base_fs - combined.AMB_FIELDS
        own_non_amb = own_fs - combined.AMB_FIELDS
        if base_non_amb != own_non_amb:
            ownership_non_ambiguity_drift.append({
                "index": i,
                "case_id": str(cases[i].case_id),
                "v13_2_non_ambiguity": tuple(sorted(base_non_amb)),
                "ownership_non_ambiguity": tuple(sorted(own_non_amb)),
            })

    ownership_failure_map = {
        str(case.case_id): tuple(sorted(fs))
        for case, fs in zip(cases, ownership_failures)
        if fs
    }
    declared_conflict_only = (
        ownership_failure_map
        == {"h2_asr_027": ("transaction_signal",)}
    )

    print("v13_2_fullframe_exact=", f"{v13_exact}/{len(cases)}")
    print("raw_v7_fullframe_exact=", f"{v7_exact}/{len(cases)}")
    print(
        "ownership_candidate_fullframe_exact=",
        f"{ownership_exact}/{len(cases)}",
    )
    print("v13_2_ambiguity_exact=", f"{v13_amb_exact}/{len(cases)}")
    print("raw_v7_ambiguity_exact=", f"{v7_amb_exact}/{len(cases)}")
    print(
        "ownership_candidate_ambiguity_exact=",
        f"{ownership_amb_exact}/{len(cases)}",
    )
    print_rows("RAW_V7_FULLFRAME_REGRESSION", v7_regressions)
    print_rows(
        "OWNERSHIP_FULLFRAME_REGRESSION",
        ownership_regressions,
    )
    print_rows(
        "OWNERSHIP_FIX_VS_RAW_V7",
        ownership_fixes_vs_v7,
    )
    print_rows(
        "OWNERSHIP_NON_AMBIGUITY_SCORE_DRIFT",
        ownership_non_ambiguity_drift,
    )
    print("ownership_failure_map=", ownership_failure_map)
    print(
        "declared_exposed_conflict_only=",
        "YES" if declared_conflict_only else "NO",
    )

    # The proof contract is stronger than "fix the four":
    #   * zero baseline-right regression over all 1,146
    #   * ambiguity exact over all 1,146
    #   * only the pre-declared exposed transaction-signal conflict remains
    #   * no non-ambiguity score drift
    #   * the candidate changes exactly the unsupported learned suppression
    #     surface discovered pre-gold.
    unsupported_indices = {
        int(row["index"])
        for row in learned_resolution_without_surface
    }
    restore_indices = {
        int(row["index"])
        for row in ownership_restores
    }
    proof_surface_exact = (
        restore_indices == unsupported_indices
        and len(restore_indices) == 4
    )
    zero_regression = not ownership_regressions
    ambiguity_exact = ownership_amb_exact == len(cases)
    fullframe_expected = ownership_exact == (len(cases) - 1)
    non_ambiguity_stable = not ownership_non_ambiguity_drift
    candidate_matches_v13_ambiguity = all(
        owned == baseline
        for owned, baseline in zip(
            ownership_candidate,
            v13_ambiguity,
        )
    )

    print(
        "OWNERSHIP_SURFACE_EQUALS_UNSUPPORTED_PHASE7H_CLASS=",
        "YES" if proof_surface_exact else "NO",
    )
    print(
        "OWNERSHIP_ZERO_BASELINE_RIGHT_REGRESSION=",
        "YES" if zero_regression else "NO",
    )
    print(
        "OWNERSHIP_AMBIGUITY_1146_EXACT=",
        "YES" if ambiguity_exact else "NO",
    )
    print(
        "OWNERSHIP_FULLFRAME_ONLY_DECLARED_CONFLICT=",
        "YES"
        if fullframe_expected and declared_conflict_only
        else "NO",
    )
    print(
        "OWNERSHIP_NON_AMBIGUITY_STABLE=",
        "YES" if non_ambiguity_stable else "NO",
    )
    print(
        "OWNERSHIP_MATCHES_V13_2_AMBIGUITY_BASELINE=",
        "YES" if candidate_matches_v13_ambiguity else "NO",
    )

    # ------------------------------------------------------------------
    # Postflight.
    # ------------------------------------------------------------------
    print("\n========== POSTFLIGHT INTEGRITY ==========")
    source_after = base.source_snapshot()
    full_source_after = full.source_snapshot()
    watched_after = {k: sha256_file(v) for k, v in watched.items()}

    source_unchanged = source_before == source_after
    full_source_unchanged = full_source_before == full_source_after
    watched_unchanged = watched_before == watched_after

    print(
        "source_tree_python_unchanged=",
        "YES" if source_unchanged else "NO",
    )
    print(
        "full_evaluator_source_snapshot_unchanged=",
        "YES" if full_source_unchanged else "NO",
    )
    print(
        "watched_source_checkpoint_hashes_unchanged=",
        "YES" if watched_unchanged else "NO",
    )
    print("candidate_artifact_written=NO")
    print("runtime_wiring_modified=NO")
    print("training_performed=NO")
    print("ambiguity_v7_modified=NO")
    print("reference_v2_modified=NO")
    print("oos_modified=NO")
    print("telephony_modified=NO")

    print("\n========== AUTHORITATIVE POST-V7 OWNERSHIP PROOF VERDICT ==========")

    proof_pass = (
        proof_surface_exact
        and zero_regression
        and ambiguity_exact
        and fullframe_expected
        and declared_conflict_only
        and non_ambiguity_stable
        and candidate_matches_v13_ambiguity
        and source_unchanged
        and full_source_unchanged
        and watched_unchanged
    )

    if proof_pass:
        primary = "NONE"
        verdict = "V13_2_V7_POST_INTEGRATION_OWNERSHIP_PROOF_PASS"
        next_action = (
            "FREEZE_POST_V7_OWNERSHIP_RULE__RETURN_TO_RUNTIME_WIRING_GATE_WITH_"
            "EXACT_FROZEN_V13_2_V7_REFERENCE_V2_OOS_STACK_AND_THIS_OWNERSHIP_"
            "RULE__TELEPHONY_DISABLED__THEN_ADDRESS_PERSISTENT_CACHE_LATENCY"
        )
    else:
        if not proof_surface_exact:
            primary = "OWNERSHIP_SURFACE_NOT_EXACT"
        elif not zero_regression:
            primary = "BASELINE_RIGHT_REGRESSION"
        elif not ambiguity_exact:
            primary = "AMBIGUITY_NOT_EXACT"
        elif not non_ambiguity_stable:
            primary = "NON_AMBIGUITY_SCORE_DRIFT"
        elif not declared_conflict_only:
            primary = "UNEXPECTED_FULLFRAME_FAILURE"
        else:
            primary = "PROOF_INTEGRITY_OR_BASELINE_PARITY"
        verdict = "V13_2_V7_POST_INTEGRATION_OWNERSHIP_PROOF_BLOCKED"
        next_action = (
            "DO_NOT_MODIFY_RUNTIME_OR_V7__LOCALIZE_ONLY_THE_PRINTED_"
            + primary
            + "_FAILURES"
        )

    print("PRIMARY_BLOCKER=", primary)
    print("V13_2_V7_POST_INTEGRATION_OWNERSHIP_VERDICT=", verdict)
    print("NEXT_ACTION=", next_action)
    print("v13_2_v7_post_integration_ownership_proof_v1_completed=YES")

    del detail_model, detail_tok, op_model, op_tok
    del gate_model, gate_tok
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
