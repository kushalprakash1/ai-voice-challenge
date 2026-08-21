#!/usr/bin/env python3
"""Read-only FULL SemanticFrame shadow integration for frozen Ambiguity V7 + Reference V2.

Purpose
-------
Integrate the two already-proven and independently reproduced semantic
compositions into the existing FULL OFFLINE Level-2 SemanticFrame assembler in
SHADOW ONLY:

  * Composition V7 may change only SemanticFrame.ambiguity.
  * Reference Composition V2 may change only SemanticFrame.reference.
  * Every other native SemanticFrame field remains frozen.

The harness:
  1. hash-locks the exact frozen V7 ambiguity proof and Reference V2 proof;
  2. reconstructs the exact frozen OOS epoch-52/scale-.9 head at the same
     replay-sensitive initialization point;
  3. runs the existing full Level-2 assembler unchanged for the baseline;
  4. reproduces the V7 ambiguity shadow using the actual full-pipeline Phase7D
     and Phase8A outputs;
  5. reproduces Reference V2 binary composition using the same frozen
     Phase7D/antecedent/selection/transaction evidence;
  6. composes a native typed reference conservatively: preserved baselines stay
     preserved, proven suppressions become none, and any proven activation must
     have a justified typed reference or the shadow blocks;
  7. creates native shadow frames by replacing ONLY reference + ambiguity;
  8. verifies all other frame fields and scores are identical;
  9. only after all inference completes, scores baseline vs combined shadow.

No case ID, category, tag, expected frame, or evaluation result participates in
shadow inference. No runtime/source/checkpoint is modified.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import os
import random
import sys
from collections import Counter
from dataclasses import replace, is_dataclass
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

V7_BASENAME = "voiceprobe_semanticlab_v2_phase7c_composition_v7_unclear_ownership_proof.py"
EXPECTED_V7_SHA256 = "1313d5f81bade95b667cf32391681ec31b15d357d8cef3f7473b8edf1eb88eb1"

REF_V2_BASENAME = "voiceprobe_semanticlab_v2_phase7c_reference_composition_proof_v2.py"
EXPECTED_REF_V2_SHA256 = "d5afaad3398542d8737f2762b4c93d72aa336db03734d2b1c950bcd679c90c88"

REF_BOUNDARY_BASENAME = "voiceprobe_semanticlab_v2_phase7c_reference_multioption_boundary_localizer_v2.py"
EXPECTED_REF_BOUNDARY_SHA256 = "6d62f40178c79fb02b861641ea961c57b20ae3e2b8eeb2675a300796b4236cc9"

FULL_BASENAME = "voiceprobe_semanticlab_v2_full_semanticframe_eval.py"
EXPECTED_FULL_SHA256 = "220a1ab5a12469c3925f7ab4a864abc348e543536fb1a1b369039511b86c21b3"

AMB_FIELDS = {"ambiguity.kind", "ambiguity.candidates"}
REF_FIELDS = {"reference"}
TARGET_FIELDS = AMB_FIELDS | REF_FIELDS


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
        "Could not locate required file "
        + basename
        + ". Checked: "
        + ", ".join(checked)
    )


def enum_value(x: Any) -> str:
    value = getattr(x, "value", None)
    return str(value if value is not None else x)


def ambiguity_tuple(frame) -> tuple[str, tuple[str, ...]]:
    amb = frame.ambiguity
    return (
        enum_value(amb.kind),
        tuple(str(x) for x in amb.candidates),
    )


def frozen_field_signature(frame) -> tuple[Any, ...]:
    """Exact structural signature for every native frame field except targets."""
    return (
        str(frame.raw_text),
        enum_value(frame.speech_act),
        enum_value(frame.topic),
        enum_value(frame.requested_fact),
        tuple(enum_value(x) for x in frame.failed_constraints),
        tuple(enum_value(x) for x in frame.proposed_changes),
        tuple(enum_value(x) for x in frame.retained_constraints),
        tuple(str(x) for x in frame.offered_options),
        "" if frame.selected_option is None else str(frame.selected_option),
        tuple(enum_value(x) for x in frame.record_claims),
        enum_value(frame.transaction_operation),
        enum_value(frame.transaction_signal),
    )


def reference_value(frame) -> str:
    return enum_value(frame.reference)


def compose_typed_reference(
    baseline_reference: str,
    scalar_baseline_active: bool,
    candidate_active: bool,
    predicted_kind: str,
    evidence: dict[str, Any],
    shadow_ambiguity: tuple[str, tuple[str, ...]],
) -> tuple[str, str, str]:
    """Map Reference V2 binary composition into the native typed reference.

    Preserved binary decisions preserve the native baseline exactly.
    Suppression is exact -> none.
    Activation requires a typed semantic justification; unresolved activation is
    returned as a composition failure rather than guessed.
    """
    baseline_reference = str(baseline_reference)
    predicted_kind = str(predicted_kind)

    if bool(candidate_active) == bool(scalar_baseline_active):
        return baseline_reference, "preserve:native_reference_baseline", ""

    if not candidate_active:
        return "none", "suppress:reference_v2_binary", ""

    if shadow_ambiguity[0] != "none":
        return "ambiguous", "activate:reference_v2_under_active_ambiguity", ""

    option = evidence.get("option", {})
    if bool(option.get("positive")) and str(option.get("resolved", "")):
        return "prior_option", "activate:unique_option_resolution", ""

    if (
        bool(evidence.get("positive"))
        and predicted_kind
        in {"prior_option", "prior_day", "prior_time", "prior_provider"}
    ):
        return predicted_kind, "activate:typed_antecedent_agreement", ""

    structural = tuple(str(x) for x in evidence.get("structural_kinds", ()))
    if len(structural) == 1 and structural[0] in {
        "prior_option", "prior_day", "prior_time", "prior_provider"
    }:
        return structural[0], "activate:unique_structural_reference_kind", ""

    if predicted_kind in {
        "prior_option", "prior_day", "prior_time", "prior_provider"
    }:
        return predicted_kind, "activate:phase7d_typed_kind", ""

    return (
        "none",
        "block:reference_activation_without_typed_kind",
        "reference_v2_activation_without_typed_kind",
    )


def clone_with_shadow_fields(
    full,
    frame,
    reference: str,
    ambiguity_kind: str,
    ambiguity_candidates: tuple[str, ...],
):
    new_ambiguity = full.SemanticAmbiguity(
        kind=full.AmbiguityKind(str(ambiguity_kind)),
        candidates=tuple(str(x) for x in ambiguity_candidates),
        detail="",
    )
    new_reference = full.ReferenceKind(str(reference))

    if is_dataclass(frame):
        try:
            return replace(
                frame,
                reference=new_reference,
                ambiguity=new_ambiguity,
            )
        except Exception:
            pass

    return full.SemanticFrame(
        raw_text=frame.raw_text,
        speech_act=frame.speech_act,
        topic=frame.topic,
        requested_fact=frame.requested_fact,
        failed_constraints=tuple(frame.failed_constraints),
        proposed_changes=tuple(frame.proposed_changes),
        retained_constraints=tuple(frame.retained_constraints),
        offered_options=tuple(frame.offered_options),
        selected_option=frame.selected_option,
        record_claims=tuple(frame.record_claims),
        transaction_operation=frame.transaction_operation,
        transaction_signal=frame.transaction_signal,
        reference=new_reference,
        ambiguity=new_ambiguity,
    )


def fields_failed(full, case, frame) -> set[str]:
    return {str(f.field) for f in full.evaluate_frame(case, frame)}


def print_first(title: str, rows: list[dict[str, Any]], limit: int = 12):
    print(title + "_count=", len(rows))
    for row in rows[:limit]:
        print(title, row)


def main() -> int:
    print("========== FROZEN V7 + REFERENCE V2 FULL SEMANTICFRAME SHADOW ==========")
    print("telephony=DISABLED")
    print("training=NO")
    print("gradient_updates=NO")
    print("runtime_wiring_modified=NO")
    print("full_evaluator_modified=NO")
    print("reasoner_modified=NO")
    print("candidate_artifact_written=NO")
    print("gold_runtime_inputs=NO")
    print("shadow_changes_only=SemanticFrame.ambiguity|SemanticFrame.reference")

    v7_path = resolve_named(V7_BASENAME)
    ref_v2_path = resolve_named(REF_V2_BASENAME)
    ref_boundary_path = resolve_named(REF_BOUNDARY_BASENAME)
    full_path = resolve_named(FULL_BASENAME)

    v7_hash = sha256_file(v7_path)
    ref_v2_hash = sha256_file(ref_v2_path)
    ref_boundary_hash = sha256_file(ref_boundary_path)
    full_hash = sha256_file(full_path)

    print("v7_source=", v7_path)
    print("v7_sha256=", v7_hash)
    print("reference_v2_source=", ref_v2_path)
    print("reference_v2_sha256=", ref_v2_hash)
    print("reference_boundary_source=", ref_boundary_path)
    print("reference_boundary_sha256=", ref_boundary_hash)
    print("full_evaluator_source=", full_path)
    print("full_evaluator_sha256=", full_hash)

    if v7_hash != EXPECTED_V7_SHA256:
        raise RuntimeError(
            f"V7 source drift expected={EXPECTED_V7_SHA256} actual={v7_hash}"
        )
    if ref_v2_hash != EXPECTED_REF_V2_SHA256:
        raise RuntimeError(
            f"Reference V2 source drift expected={EXPECTED_REF_V2_SHA256} actual={ref_v2_hash}"
        )
    if ref_boundary_hash != EXPECTED_REF_BOUNDARY_SHA256:
        raise RuntimeError(
            "Reference boundary source drift "
            f"expected={EXPECTED_REF_BOUNDARY_SHA256} actual={ref_boundary_hash}"
        )
    if full_hash != EXPECTED_FULL_SHA256:
        raise RuntimeError(
            f"Full evaluator source drift expected={EXPECTED_FULL_SHA256} actual={full_hash}"
        )

    v7 = load_mod("phase7c_v7_shadow_policy", v7_path)

    sep_path = v7.resolve_named(None, v7.SEP_BASENAME)
    sep_hash = sha256_file(sep_path)
    if sep_hash != v7.EXPECTED_SEP_SHA256:
        raise RuntimeError(
            f"Separator source drift expected={v7.EXPECTED_SEP_SHA256} actual={sep_hash}"
        )
    sep = load_mod("phase7c_v7_shadow_separator", sep_path)

    loc_path = sep.resolve_named(None, sep.LOCALIZER_BASENAME)
    if sha256_file(loc_path) != sep.EXPECTED_LOCALIZER_SHA256:
        raise RuntimeError("Localizer source drift")
    loc = load_mod("phase7c_v7_shadow_localizer", loc_path)

    app_path = loc.resolve_named(None, loc.APP_BASENAME)
    if sha256_file(app_path) != loc.EXPECTED_APP_SHA256:
        raise RuntimeError("Applicability source drift")
    app = load_mod("phase7c_v7_shadow_applicability", app_path)

    comp_path = app.resolve_named(None, app.COMP_BASENAME)
    if sha256_file(comp_path) != app.EXPECTED_COMP_SHA256:
        raise RuntimeError("Composition source drift")
    comp = load_mod("phase7c_v7_shadow_composition", comp_path)

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
            raise RuntimeError(
                f"{label} source drift expected={expected} actual={actual}"
            )

    v52 = load_mod("phase7c_v7_shadow_v52", v52_path)
    feas = load_mod("phase7c_v7_shadow_feas", feas_path)
    _struct = load_mod("phase7c_v7_shadow_struct", struct_path)
    base = v52.base

    source_before = base.source_snapshot()
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
    watched_before = {k: sha256_file(p) for k, p in watched.items()}

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
        v52, gate_model, gate_tok, original_train, thresholds
    )
    orig_y = v52.gold_example_tensor(original_train)

    # Preserve validation capture ordering used by the authoritative replay.
    feas.capture(
        v52, gate_model, gate_tok, original_val, thresholds
    )

    cases = list(v52.load_semanticlab_cases())
    if len(cases) != 133:
        raise RuntimeError(
            f"Historical/full-pipeline cardinality drift: {len(cases)}"
        )

    v52.capture_features(
        gate_model,
        gate_tok,
        v52.runtime_for_cases(cases),
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
    print("oos_replay_head_initialized_at_authoritative_v52_point=YES")

    hist_x, _, hist_margin, hist_base_pred, _, _ = v52.capture_features(
        gate_model,
        gate_tok,
        v52.runtime_for_cases(cases),
        thresholds,
    )
    hist_oos, _ = feas.frozen_oos_pred(
        frozen_oos_head,
        hist_x,
        hist_margin,
    )

    print("\n========== EXISTING FULL PIPELINE BASELINE ==========")
    full = load_mod("phase7c_v7_shadow_full_eval", full_path)
    checkpoints = full.validate_environment()
    full_source_before = full.source_snapshot()

    runtime = [
        full.RuntimeTurn(
            context=tuple(case.context),
            utterance=str(case.utterance),
        )
        for case in cases
    ]
    if set(full.RuntimeTurn.__dataclass_fields__) != {"context", "utterance"}:
        raise RuntimeError("Full evaluator runtime input boundary drift")

    baseline_result = full.assemble_level2(runtime, checkpoints)
    if len(baseline_result.frames) != 133:
        raise RuntimeError("Full evaluator did not return 133 frames")
    print("full_pipeline_inference_complete=YES")
    print("full_pipeline_gold_consulted=NO")

    print("\n========== FULL PIPELINE REFERENCE BASELINE CONTRACT ==========")
    full_baseline_ambiguity = [
        ambiguity_tuple(frame)
        for frame in baseline_result.frames
    ]
    full_baseline_reference = [
        reference_value(frame)
        for frame in baseline_result.frames
    ]
    scalar_reference_active = [
        bool(int(x))
        for x in hist_base_pred[:, 0].tolist()
    ]

    reference_gate_contract_mismatches = []
    reference_assembly_contract_mismatches = []
    for i, frame in enumerate(baseline_result.frames):
        gate = baseline_result.gate_labels[i]
        gate_active = bool(int(gate.get("reference", 0)))
        if gate_active != scalar_reference_active[i]:
            reference_gate_contract_mismatches.append({
                "index": i,
                "scalar_reference_active": scalar_reference_active[i],
                "full_gate_reference_active": gate_active,
                "turn": str(cases[i].utterance),
                "context": tuple(cases[i].context),
            })

        predicted_kind = str(baseline_result.reference_kinds[i])
        violations: list[str] = []
        expected_reference = full.assemble_reference(
            gate,
            predicted_kind,
            full_baseline_ambiguity[i][0],
            violations,
        )
        actual_reference = full_baseline_reference[i]
        if str(expected_reference) != actual_reference:
            reference_assembly_contract_mismatches.append({
                "index": i,
                "expected_reference": str(expected_reference),
                "actual_reference": actual_reference,
                "predicted_kind": predicted_kind,
                "gate_reference_active": gate_active,
                "ambiguity": full_baseline_ambiguity[i],
                "violations": tuple(violations),
                "turn": str(cases[i].utterance),
                "context": tuple(cases[i].context),
            })

    print_first(
        "REFERENCE_GATE_CONTRACT_MISMATCH",
        reference_gate_contract_mismatches,
    )
    print_first(
        "REFERENCE_ASSEMBLY_CONTRACT_MISMATCH",
        reference_assembly_contract_mismatches,
    )

    # Import frozen Reference V2 helpers only after the replay-sensitive OOS
    # reconstruction and unchanged full-pipeline baseline are complete.
    ref_v2 = load_mod("phase7c_ref_v2_shadow_policy", ref_v2_path)
    ref_boundary = load_mod(
        "phase7c_ref_v2_shadow_boundary",
        ref_boundary_path,
    )

    print("\n========== V7 SHADOW AMBIGUITY ASSEMBLY ==========")
    p7j = base.load_mod("phase7c_v7_shadow_p7j", base.P7J)
    p7jr_baseline = base.load_mod(
        "phase7c_v7_shadow_p7jr_baseline", base.P7JR
    )
    p7jr = base.load_mod(
        "phase7c_v7_shadow_p7jr_candidate", base.P7JR
    )
    p7h = base.load_mod("phase7c_v7_shadow_p7h", base.P7H)
    p8dn = base.load_mod("phase7c_v7_shadow_p8dn", base.P8DN)

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
        cases, p7j, detail_model, detail_tok
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
    for row, detail in zip(cases, corrected_details):
        corrected, _why = v7.final_capability_kind_arbitration(
            row,
            detail,
            p7jr,
            comp,
            p8dn.normalize_operation,
        )
        final_details.append(str(corrected))

    ref_kinds = [
        str(x)
        for x in baseline_result.reference_kinds
    ]
    dense_pairs = [
        tuple(str(x) for x in pair)
        for pair in baseline_result.dense_pairs
    ]

    scalar_ambiguity_active = [
        bool(int(x))
        for x in hist_base_pred[:, 1].tolist()
    ]
    isolated_baseline = comp.current_baseline_structures(
        cases,
        scalar_ambiguity_active,
        raw_details,
        p7jr_baseline,
    )
    full_baseline = [
        ambiguity_tuple(frame)
        for frame in baseline_result.frames
    ]

    baseline_contract_mismatches = []
    for i, (iso, native) in enumerate(
        zip(isolated_baseline, full_baseline)
    ):
        if iso != native:
            baseline_contract_mismatches.append({
                "index": i,
                "isolated": iso,
                "full": native,
                "turn": str(cases[i].utterance),
                "context": tuple(cases[i].context),
            })

    print_first(
        "BASELINE_CONTRACT_MISMATCH",
        baseline_contract_mismatches,
    )

    strict_shadow, strict_reasons, _candidate_units, _op_state = (
        v7.compose_v3(
            cases,
            final_details,
            ref_kinds,
            dense_pairs,
            [bool(int(x)) for x in hist_oos.tolist()],
            full_baseline,
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
    )
    safe_shadow, safe_reasons = v7.apply_stability_wrapper(
        full_baseline,
        strict_shadow,
        strict_reasons,
    )

    print("\n========== REFERENCE V2 SHADOW ASSEMBLY ==========")
    reference_evidence = [
        ref_v2.augment_reference_evidence_v2(
            row,
            ref_v2.structural_reference_evidence(
                row,
                predicted_kind,
                v7,
                p7jr,
                p7h,
                loc,
            ),
            ref_boundary,
            p8dn,
        )
        for row, predicted_kind in zip(cases, ref_kinds)
    ]

    reference_binary_candidate, reference_binary_reasons = (
        ref_v2.compose_reference_candidate(
            [int(x) for x in scalar_reference_active],
            reference_evidence,
        )
    )

    reference_binary_changes = [
        i
        for i, (before, after) in enumerate(
            zip(scalar_reference_active, reference_binary_candidate)
        )
        if bool(before) != bool(after)
    ]
    print(
        "reference_v2_binary_change_count=",
        len(reference_binary_changes),
    )

    shadow_reference_values = []
    shadow_reference_reasons = []
    reference_composition_failures = []

    for i, (
        baseline_ref,
        scalar_active,
        candidate_active,
        predicted_kind,
        evidence,
        ambiguity_structure,
    ) in enumerate(
        zip(
            full_baseline_reference,
            scalar_reference_active,
            reference_binary_candidate,
            ref_kinds,
            reference_evidence,
            safe_shadow,
        )
    ):
        value, reason, failure = compose_typed_reference(
            baseline_ref,
            bool(scalar_active),
            bool(candidate_active),
            str(predicted_kind),
            evidence,
            ambiguity_structure,
        )
        shadow_reference_values.append(value)
        shadow_reference_reasons.append(reason)
        if failure:
            reference_composition_failures.append({
                "index": i,
                "failure": failure,
                "baseline_reference": baseline_ref,
                "scalar_reference_active": bool(scalar_active),
                "candidate_reference_active": bool(candidate_active),
                "predicted_kind": str(predicted_kind),
                "structural_kinds": tuple(
                    evidence.get("structural_kinds", ())
                ),
                "option": evidence.get("option", {}),
                "shadow_ambiguity": ambiguity_structure,
                "turn": str(cases[i].utterance),
                "context": tuple(cases[i].context),
            })

    print_first(
        "REFERENCE_COMPOSITION_FAILURE",
        reference_composition_failures,
    )

    shadow_frames = []
    constructor_failures = []
    for i, (frame, structure, reference) in enumerate(
        zip(
            baseline_result.frames,
            safe_shadow,
            shadow_reference_values,
        )
    ):
        try:
            shadow = clone_with_shadow_fields(
                full,
                frame,
                reference,
                structure[0],
                structure[1],
            )
        except Exception as exc:
            constructor_failures.append({
                "index": i,
                "type": type(exc).__name__,
                "error": str(exc),
                "reference": reference,
                "ambiguity": structure,
                "turn": str(cases[i].utterance),
            })
            shadow = frame
        shadow_frames.append(shadow)

    print_first(
        "SHADOW_CONSTRUCTOR_FAILURE",
        constructor_failures,
    )

    frozen_field_changes = []
    reference_field_changes = []
    ambiguity_field_changes = []
    for i, (before, after) in enumerate(
        zip(baseline_result.frames, shadow_frames)
    ):
        if frozen_field_signature(before) != frozen_field_signature(after):
            frozen_field_changes.append({
                "index": i,
                "turn": str(cases[i].utterance),
            })
        if reference_value(before) != reference_value(after):
            reference_field_changes.append({
                "index": i,
                "before": reference_value(before),
                "after": reference_value(after),
                "binary_reason": reference_binary_reasons[i],
                "typed_reason": shadow_reference_reasons[i],
                "turn": str(cases[i].utterance),
                "context": tuple(cases[i].context),
            })
        if ambiguity_tuple(before) != ambiguity_tuple(after):
            ambiguity_field_changes.append({
                "index": i,
                "before": ambiguity_tuple(before),
                "after": ambiguity_tuple(after),
                "reason": safe_reasons[i],
                "turn": str(cases[i].utterance),
                "context": tuple(cases[i].context),
            })

    print_first(
        "FROZEN_NON_TARGET_FIELD_CHANGE",
        frozen_field_changes,
    )
    print_first(
        "REFERENCE_FIELD_CHANGE",
        reference_field_changes,
    )
    print_first(
        "AMBIGUITY_FIELD_CHANGE",
        ambiguity_field_changes,
    )

    baseline_coherence = []
    shadow_coherence = []

    for i, frame in enumerate(baseline_result.frames):
        ref = reference_value(frame)
        amb = ambiguity_tuple(frame)
        if ref == "ambiguous" and amb[0] == "none":
            baseline_coherence.append(
                (i, "reference_ambiguous_with_none_ambiguity")
            )
        if (
            amb[0] == "option_reference"
            and frame.selected_option not in (None, "")
        ):
            baseline_coherence.append(
                (i, "selected_option_present_while_option_ambiguous")
            )

    for i, frame in enumerate(shadow_frames):
        ref = reference_value(frame)
        amb = ambiguity_tuple(frame)
        if ref == "ambiguous" and amb[0] == "none":
            shadow_coherence.append(
                (i, "reference_ambiguous_with_none_ambiguity")
            )
        if (
            amb[0] == "option_reference"
            and frame.selected_option not in (None, "")
        ):
            shadow_coherence.append(
                (i, "selected_option_present_while_option_ambiguous")
            )

    baseline_coherence_set = set(baseline_coherence)
    introduced_coherence = [
        {
            "index": i,
            "issue": issue,
            "turn": str(cases[i].utterance),
            "context": tuple(cases[i].context),
            "baseline_reference": full_baseline_reference[i],
            "shadow_reference": reference_value(shadow_frames[i]),
            "baseline_ambiguity": full_baseline[i],
            "shadow_ambiguity": ambiguity_tuple(shadow_frames[i]),
            "selected_option": (
                ""
                if shadow_frames[i].selected_option is None
                else str(shadow_frames[i].selected_option)
            ),
        }
        for i, issue in shadow_coherence
        if (i, issue) not in baseline_coherence_set
    ]
    print_first(
        "INTRODUCED_CROSS_FIELD_COHERENCE",
        introduced_coherence,
    )

    unexpected_reference_field_changes = [
        row
        for row in reference_field_changes
        if row["index"] not in set(reference_binary_changes)
    ]
    print_first(
        "UNEXPECTED_REFERENCE_FIELD_CHANGE",
        unexpected_reference_field_changes,
    )

    print("\n========== GOLD SCORING BEGINS ONLY NOW ==========")
    print("case_id_visible_to_shadow_inference=NO")
    print("expected_gold_visible_to_shadow_inference=NO")

    hist_gold = v52.gold_case_tensor(cases).long()
    oos_exact = int(
        (hist_oos == hist_gold[:, 2]).sum()
    )
    print(
        "frozen_oos_historical_exact=",
        f"{oos_exact}/{len(cases)}",
    )

    baseline_failures = [
        fields_failed(full, case, frame)
        for case, frame in zip(
            cases,
            baseline_result.frames,
        )
    ]
    shadow_failures = [
        fields_failed(full, case, frame)
        for case, frame in zip(
            cases,
            shadow_frames,
        )
    ]

    baseline_exact = sum(not fields for fields in baseline_failures)
    shadow_exact = sum(not fields for fields in shadow_failures)

    ambiguity_baseline_exact = sum(
        not (fields & AMB_FIELDS)
        for fields in baseline_failures
    )
    ambiguity_shadow_exact = sum(
        not (fields & AMB_FIELDS)
        for fields in shadow_failures
    )
    reference_baseline_exact = sum(
        not (fields & REF_FIELDS)
        for fields in baseline_failures
    )
    reference_shadow_exact = sum(
        not (fields & REF_FIELDS)
        for fields in shadow_failures
    )

    regressions = []
    fixes = []
    frozen_score_drift = []

    for i, (base_fields, shadow_fields) in enumerate(
        zip(baseline_failures, shadow_failures)
    ):
        base_ok = not base_fields
        shadow_ok = not shadow_fields

        if base_ok and not shadow_ok:
            regressions.append({
                "index": i,
                "case_id": str(cases[i].case_id),
                "new_failed_fields": sorted(
                    shadow_fields - base_fields
                ),
                "reference_binary_reason": reference_binary_reasons[i],
                "reference_typed_reason": shadow_reference_reasons[i],
                "ambiguity_reason": safe_reasons[i],
                "baseline_reference": full_baseline_reference[i],
                "shadow_reference": reference_value(shadow_frames[i]),
                "baseline_ambiguity": full_baseline[i],
                "shadow_ambiguity": ambiguity_tuple(shadow_frames[i]),
                "turn": str(cases[i].utterance),
                "context": tuple(cases[i].context),
            })

        if not base_ok and shadow_ok:
            fixes.append({
                "index": i,
                "case_id": str(cases[i].case_id),
                "reference_binary_reason": reference_binary_reasons[i],
                "reference_typed_reason": shadow_reference_reasons[i],
                "ambiguity_reason": safe_reasons[i],
                "baseline_reference": full_baseline_reference[i],
                "shadow_reference": reference_value(shadow_frames[i]),
                "baseline_ambiguity": full_baseline[i],
                "shadow_ambiguity": ambiguity_tuple(shadow_frames[i]),
            })

        base_frozen = base_fields - TARGET_FIELDS
        shadow_frozen = shadow_fields - TARGET_FIELDS
        if base_frozen != shadow_frozen:
            frozen_score_drift.append({
                "index": i,
                "case_id": str(cases[i].case_id),
                "baseline_frozen_failures": sorted(base_frozen),
                "shadow_frozen_failures": sorted(shadow_frozen),
            })

    print(
        "baseline_fullframe_exact=",
        f"{baseline_exact}/{len(cases)}",
    )
    print(
        "shadow_fullframe_exact=",
        f"{shadow_exact}/{len(cases)}",
    )
    print(
        "baseline_reference_exact=",
        f"{reference_baseline_exact}/{len(cases)}",
    )
    print(
        "shadow_reference_exact=",
        f"{reference_shadow_exact}/{len(cases)}",
    )
    print(
        "baseline_ambiguity_exact=",
        f"{ambiguity_baseline_exact}/{len(cases)}",
    )
    print(
        "shadow_ambiguity_exact=",
        f"{ambiguity_shadow_exact}/{len(cases)}",
    )
    print_first("FULLFRAME_REGRESSION", regressions)
    print_first("FULLFRAME_FIX", fixes)
    print_first(
        "FROZEN_NON_TARGET_SCORE_DRIFT",
        frozen_score_drift,
    )

    print(
        "shadow_reference_changed_count=",
        len(reference_field_changes),
    )
    print(
        "shadow_ambiguity_changed_count=",
        len(ambiguity_field_changes),
    )

    print("\n========== POSTFLIGHT INTEGRITY ==========")
    source_after = base.source_snapshot()
    watched_after = {
        k: sha256_file(p)
        for k, p in watched.items()
    }
    full_source_after = full.source_snapshot()

    source_unchanged = source_before == source_after
    watched_unchanged = watched_before == watched_after
    full_source_unchanged = full_source_before == full_source_after

    print(
        "source_tree_python_unchanged=",
        "YES" if source_unchanged else "NO",
    )
    print(
        "watched_source_checkpoint_hashes_unchanged=",
        "YES" if watched_unchanged else "NO",
    )
    print(
        "full_evaluator_source_snapshot_unchanged=",
        "YES" if full_source_unchanged else "NO",
    )
    print("candidate_artifact_written=NO")
    print("runtime_wiring_modified=NO")
    print("training_performed=NO")
    print("ambiguity_runtime_modified=NO")
    print("reference_runtime_modified=NO")

    ambiguity_baseline_contract_pass = not baseline_contract_mismatches
    reference_baseline_contract_pass = (
        not reference_gate_contract_mismatches
        and not reference_assembly_contract_mismatches
    )
    reference_composition_pass = not reference_composition_failures
    constructor_pass = not constructor_failures
    frozen_identity_pass = (
        not frozen_field_changes
        and not frozen_score_drift
    )
    reference_change_scope_pass = not unexpected_reference_field_changes
    zero_regression = not regressions
    no_new_coherence = not introduced_coherence
    oos_pass = oos_exact == len(cases)
    score_non_degrading = shadow_exact >= baseline_exact
    reference_non_degrading = (
        reference_shadow_exact >= reference_baseline_exact
    )
    ambiguity_non_degrading = (
        ambiguity_shadow_exact >= ambiguity_baseline_exact
    )
    reference_candidate_exercised = bool(reference_binary_changes)
    integrity_pass = (
        source_unchanged
        and watched_unchanged
        and full_source_unchanged
    )

    all_pass = (
        ambiguity_baseline_contract_pass
        and reference_baseline_contract_pass
        and reference_composition_pass
        and constructor_pass
        and frozen_identity_pass
        and reference_change_scope_pass
        and zero_regression
        and no_new_coherence
        and oos_pass
        and score_non_degrading
        and reference_non_degrading
        and ambiguity_non_degrading
        and reference_candidate_exercised
        and integrity_pass
    )

    if all_pass:
        verdict = "V7_REFERENCE_V2_FULL_SEMANTICFRAME_SHADOW_PASS"
        primary_blocker = "NONE"
        next_action = (
            "FREEZE_COMBINED_AMBIGUITY_V7_REFERENCE_V2_SEMANTICFRAME_SHADOW__"
            "NEXT_BUILD_BOUNDED_RUNTIME_WIRING_CANDIDATE_WITH_TELEPHONY_STILL_"
            "DISABLED_THEN_RUN_OFFLINE_INTEGRATION_AND_TIMING_GATES_BEFORE_ANY_"
            "LIVE_CALL"
        )
    elif not ambiguity_baseline_contract_pass:
        verdict = "V7_REFERENCE_V2_FULL_SEMANTICFRAME_SHADOW_BLOCKED"
        primary_blocker = "AMBIGUITY_FULL_PIPELINE_BASELINE_CONTRACT_MISMATCH"
        next_action = (
            "DO_NOT_WIRE_RUNTIME__LOCALIZE_ONLY_THE_PRINTED_AMBIGUITY_BASELINE_"
            "CONTRACT_MISMATCHES"
        )
    elif not reference_baseline_contract_pass:
        verdict = "V7_REFERENCE_V2_FULL_SEMANTICFRAME_SHADOW_BLOCKED"
        primary_blocker = "REFERENCE_FULL_PIPELINE_BASELINE_CONTRACT_MISMATCH"
        next_action = (
            "DO_NOT_WIRE_RUNTIME__LOCALIZE_ONLY_THE_PRINTED_REFERENCE_GATE_OR_"
            "ASSEMBLY_CONTRACT_MISMATCHES"
        )
    elif not reference_composition_pass:
        verdict = "V7_REFERENCE_V2_FULL_SEMANTICFRAME_SHADOW_BLOCKED"
        primary_blocker = "REFERENCE_TYPED_COMPOSITION_UNRESOLVED"
        next_action = (
            "DO_NOT_GUESS_REFERENCE_KIND__LOCALIZE_ONLY_THE_PRINTED_REFERENCE_"
            "COMPOSITION_FAILURES_WITH_REFERENCE_V2_AND_V7_FROZEN"
        )
    elif not no_new_coherence:
        verdict = "V7_REFERENCE_V2_FULL_SEMANTICFRAME_SHADOW_BLOCKED"
        primary_blocker = "REFERENCE_AMBIGUITY_SELECTION_CROSS_FIELD_COHERENCE"
        next_action = (
            "DO_NOT_CHANGE_FROZEN_V7_OR_REFERENCE_V2__LOCALIZE_ONLY_THE_PRINTED_"
            "INTRODUCED_CROSS_FIELD_COHERENCE"
        )
    elif not zero_regression:
        verdict = "V7_REFERENCE_V2_FULL_SEMANTICFRAME_SHADOW_BLOCKED"
        primary_blocker = "FULL_FRAME_BASELINE_RIGHT_REGRESSION"
        next_action = (
            "DO_NOT_WIRE_RUNTIME__LOCALIZE_ONLY_THE_PRINTED_FULLFRAME_"
            "REGRESSIONS_WITH_BOTH_COMPOSITIONS_FROZEN"
        )
    else:
        verdict = "V7_REFERENCE_V2_FULL_SEMANTICFRAME_SHADOW_BLOCKED"
        primary_blocker = "COMBINED_SHADOW_INTEGRATION_INVARIANT_FAILURE"
        next_action = (
            "DO_NOT_WIRE_RUNTIME__LOCALIZE_ONLY_THE_PRINTED_FAILED_INVARIANT_"
            "WITHOUT_RETRAINING_OR_POLICY_TUNING"
        )

    print(
        "\n========== AUTHORITATIVE V7 + REFERENCE V2 FULL SEMANTICFRAME SHADOW VERDICT =========="
    )
    print(
        "AMBIGUITY_BASELINE_CONTRACT_PARITY=",
        "YES" if ambiguity_baseline_contract_pass else "NO",
    )
    print(
        "REFERENCE_BASELINE_CONTRACT_PARITY=",
        "YES" if reference_baseline_contract_pass else "NO",
    )
    print(
        "REFERENCE_TYPED_COMPOSITION_COMPLETE=",
        "YES" if reference_composition_pass else "NO",
    )
    print(
        "FROZEN_NON_TARGET_FIELDS_IDENTICAL=",
        "YES" if frozen_identity_pass else "NO",
    )
    print(
        "REFERENCE_CHANGE_SCOPE_VALID=",
        "YES" if reference_change_scope_pass else "NO",
    )
    print(
        "ZERO_BASELINE_RIGHT_REGRESSION=",
        "YES" if zero_regression else "NO",
    )
    print(
        "NO_NEW_CROSS_FIELD_COHERENCE_VIOLATION=",
        "YES" if no_new_coherence else "NO",
    )
    print(
        "OOS_REMAINS_FROZEN=",
        "YES" if oos_pass else "NO",
    )
    print(
        "FULLFRAME_SCORE_NON_DEGRADING=",
        "YES" if score_non_degrading else "NO",
    )
    print(
        "REFERENCE_SCORE_NON_DEGRADING=",
        "YES" if reference_non_degrading else "NO",
    )
    print(
        "AMBIGUITY_SCORE_NON_DEGRADING=",
        "YES" if ambiguity_non_degrading else "NO",
    )
    print(
        "REFERENCE_CANDIDATE_EXERCISED=",
        "YES" if reference_candidate_exercised else "NO",
    )
    print("PRIMARY_BLOCKER=", primary_blocker)
    print("V7_REFERENCE_V2_FULL_SEMANTICFRAME_SHADOW_VERDICT=", verdict)
    print("NEXT_ACTION=", next_action)
    print("v7_reference_v2_full_semanticframe_shadow_completed=YES")

    del detail_tok, detail_model, op_tok, op_model
    del gate_tok, gate_model
    gc.collect()

    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
