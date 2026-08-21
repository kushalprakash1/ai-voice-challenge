#!/usr/bin/env python3
"""VoiceProbe v3.3 Level-2 runtime-wiring OFFLINE gate V2.

This is the first runtime-integration gate after freezing:
  * V13.2 full-frame normalization stack
  * Ambiguity Composition V7 stability wrapper
  * Reference Composition V2
  * frozen OOS residual

It does NOT modify runtime source or telephony.

The gate:
  1. reconstructs the established 1,146-case Level-2 stack;
  2. applies frozen V13.2 BEFORE Ambiguity V7 / Reference V2;
  3. verifies the combined stack does not regress any established field;
  4. exercises the actual SemanticFrame -> RemoteObservation boundary;
  5. routes unresolved ambiguity or frozen OOS conservatively to the existing
     CLARIFICATION_REQUEST planner path instead of coercing an actionable frame;
  6. feeds every observation through the current StrategicActionGenerator;
  7. runs focused offline v3.3 planner tests;
  8. measures uncached single-turn Level-2/V13.2 latency so the next engineering
     action is determined by evidence rather than guesswork.

No case ID/category/tag/gold is visible to semantic inference or runtime routing.
Gold is consulted only after the complete combined frame stack and runtime
observations have already been constructed.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

COMBINED_SHADOW_BASENAME = (
    "voiceprobe_semanticlab_v2_phase7c_v7_reference_v2_full_semanticframe_shadow.py"
)
EXPECTED_COMBINED_SHADOW_SHA256 = (
    "e168a5a473bbe0cb8f411d6fc957dc21959ca2838a8c3337c4b86b5f1a72f192"
)

OWNERSHIP_PROOF_BASENAME = (
    "voiceprobe_semanticlab_v2_v13_2_v7_post_integration_ownership_proof_v1.py"
)
EXPECTED_OWNERSHIP_PROOF_SHA256 = (
    "362e0f2f84762e3e4cb53ccf359d2fa53f7510b52022161063332c409d980b92"
)

V13_2_BASENAME = "voiceprobe_semanticlab_v2_post_fresh_architecture_v13_2_final.py"
EXPECTED_V13_2_SHA256 = (
    "387fe102b7d12b5c6b091605a7489f0517b12e6533e8719742cd35a7c8cff4ac"
)

DECLARED_EXPOSED_CONFLICT = "h2_asr_027"
PROVISIONAL_SINGLE_TURN_SEMANTIC_BUDGET_MS = 1000.0

REQUIRED_OFFLINE_TESTS = (
    "tests/test_v33_planner.py",
    "tests/test_v33_semantic_planner.py",
)
OPTIONAL_OFFLINE_TESTS = (
    "tests/test_stage_lab.py",
    "tests/test_stage_realtime_fidelity.py",
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
        "Missing required file "
        + basename
        + ". Checked: "
        + ", ".join(checked)
    )


def print_first(title: str, rows: list[dict[str, Any]], limit: int = 15) -> None:
    print(title + "_count=", len(rows))
    for row in rows[:limit]:
        print(title, row)


def context_from_mind(mind, limit: int = 6) -> tuple[str, ...]:
    """Build semantic context from prior remote-agent turns only."""
    rows = []
    for speaker, text in getattr(mind.world, "history", ()):
        if str(speaker).strip().upper() == "PGAI" and str(text).strip():
            rows.append(str(text).strip())
    return tuple(rows[-limit:])


def safe_clarification_observation(world_model, frame):
    """Non-action-bearing bridge for unresolved ambiguity/OOS."""
    return world_model.RemoteObservation(
        kind=world_model.ObservationKind.CLARIFICATION_REQUEST,
        raw_text=str(frame.raw_text),
        requires_response=True,
    )


def observation_action_payload(observation) -> dict[str, Any]:
    fields = {
        "requested_fact": "",
        "offered_options": (),
        "unavailable_constraints": (),
        "remote_claims": (),
        "selected_option": "",
        "transaction_operation": "",
        "search_constraints": (),
    }
    return {
        key: getattr(observation, key, default)
        for key, default in fields.items()
    }


def is_empty_safe_payload(payload: dict[str, Any]) -> bool:
    for key, value in payload.items():
        if key == "transaction_operation":
            if str(value) not in {"", "none"}:
                return False
        elif isinstance(value, tuple):
            if value:
                return False
        elif value not in {"", None}:
            return False
    return True


def run_focused_pytests(repo_root: Path) -> tuple[bool, str]:
    missing = [
        path
        for path in REQUIRED_OFFLINE_TESTS
        if not (repo_root / path).is_file()
    ]
    if missing:
        return False, "missing_required_tests=" + repr(missing)

    selected = list(REQUIRED_OFFLINE_TESTS)
    selected.extend(
        path
        for path in OPTIONAL_OFFLINE_TESTS
        if (repo_root / path).is_file()
    )

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *selected,
    ]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(repo_root / "src")

    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout or ""
    if completed.returncode == 0:
        tail = "\n".join(output.splitlines()[-12:])
        return True, tail

    return False, output


def main() -> int:
    print("========== LEVEL-2 RUNTIME WIRING OFFLINE GATE V2 ==========")
    print("telephony=DISABLED")
    print("live_call=NO")
    print("runtime_source_write=NO")
    print("reasoner_source_write=NO")
    print("training=NO")
    print("gradient_updates=NO")
    print("case_id_runtime_inputs=NO")
    print("category_runtime_inputs=NO")
    print("tags_runtime_inputs=NO")
    print("gold_runtime_inputs=NO")
    print(
        "stack=RAW_LEVEL2__V13_2__AMBIGUITY_V7__POST_V7_OWNERSHIP__REFERENCE_V2__"
        "SAFE_RUNTIME_ADAPTER__CURRENT_STRATEGIC_ACTION_GENERATOR"
    )

    combined_path = resolve_named(COMBINED_SHADOW_BASENAME)
    combined_hash = sha256_file(combined_path)
    print("combined_shadow_source=", combined_path)
    print("combined_shadow_sha256=", combined_hash)
    if combined_hash != EXPECTED_COMBINED_SHADOW_SHA256:
        raise RuntimeError(
            "Combined shadow drift: "
            f"expected={EXPECTED_COMBINED_SHADOW_SHA256} actual={combined_hash}"
        )

    combined = load_mod("runtime_gate_combined_shadow", combined_path)

    ownership_path = resolve_named(OWNERSHIP_PROOF_BASENAME)
    ownership_hash = sha256_file(ownership_path)
    print("ownership_proof_source=", ownership_path)
    print("ownership_proof_sha256=", ownership_hash)
    if ownership_hash != EXPECTED_OWNERSHIP_PROOF_SHA256:
        raise RuntimeError(
            "Ownership proof drift: "
            f"expected={EXPECTED_OWNERSHIP_PROOF_SHA256} actual={ownership_hash}"
        )
    ownership = load_mod("runtime_gate_ownership_proof", ownership_path)

    # Resolve/hash-lock all exact frozen sources from the proven combined shadow.
    v7_path = combined.resolve_named(combined.V7_BASENAME)
    ref_v2_path = combined.resolve_named(combined.REF_V2_BASENAME)
    ref_boundary_path = combined.resolve_named(combined.REF_BOUNDARY_BASENAME)
    full_path = combined.resolve_named(combined.FULL_BASENAME)

    for label, path, expected in (
        ("v7", v7_path, combined.EXPECTED_V7_SHA256),
        ("reference_v2", ref_v2_path, combined.EXPECTED_REF_V2_SHA256),
        (
            "reference_boundary",
            ref_boundary_path,
            combined.EXPECTED_REF_BOUNDARY_SHA256,
        ),
        ("full_evaluator", full_path, combined.EXPECTED_FULL_SHA256),
    ):
        actual = sha256_file(path)
        print(label + "_source=", path)
        print(label + "_sha256=", actual)
        if actual != expected:
            raise RuntimeError(
                f"{label} source drift expected={expected} actual={actual}"
            )

    v13_2_path = resolve_named(V13_2_BASENAME)
    v13_2_hash = sha256_file(v13_2_path)
    print("v13_2_source=", v13_2_path)
    print("v13_2_sha256=", v13_2_hash)
    if v13_2_hash != EXPECTED_V13_2_SHA256:
        raise RuntimeError(
            f"V13.2 source drift expected={EXPECTED_V13_2_SHA256} "
            f"actual={v13_2_hash}"
        )

    # Load V7 dependency chain.
    v7 = load_mod("runtime_gate_v7", v7_path)
    sep_path = v7.resolve_named(None, v7.SEP_BASENAME)
    if sha256_file(sep_path) != v7.EXPECTED_SEP_SHA256:
        raise RuntimeError("V7 separator source drift")
    sep = load_mod("runtime_gate_sep", sep_path)

    loc_path = sep.resolve_named(None, sep.LOCALIZER_BASENAME)
    if sha256_file(loc_path) != sep.EXPECTED_LOCALIZER_SHA256:
        raise RuntimeError("V7 localizer source drift")
    loc = load_mod("runtime_gate_loc", loc_path)

    app_path = loc.resolve_named(None, loc.APP_BASENAME)
    if sha256_file(app_path) != loc.EXPECTED_APP_SHA256:
        raise RuntimeError("V7 applicability source drift")
    app = load_mod("runtime_gate_app", app_path)

    comp_path = app.resolve_named(None, app.COMP_BASENAME)
    if sha256_file(comp_path) != app.EXPECTED_COMP_SHA256:
        raise RuntimeError("V7 composition source drift")
    comp = load_mod("runtime_gate_comp", comp_path)

    v52_path = comp.resolve_named(None, comp.V52_BASENAME)
    feas_path = comp.resolve_named(None, comp.FEAS_BASENAME)
    struct_path = comp.resolve_named(None, comp.STRUCT_BASENAME)
    for label, path, expected in (
        ("v52", v52_path, comp.EXPECTED_V52_SHA256),
        ("feas", feas_path, comp.EXPECTED_FEAS_SHA256),
        ("struct", struct_path, comp.EXPECTED_STRUCT_SHA256),
    ):
        actual = sha256_file(path)
        print(label + "_source=", path)
        print(label + "_sha256=", actual)
        if actual != expected:
            raise RuntimeError(
                f"{label} source drift expected={expected} actual={actual}"
            )

    v52 = load_mod("runtime_gate_v52", v52_path)
    feas = load_mod("runtime_gate_feas", feas_path)
    _struct = load_mod("runtime_gate_struct", struct_path)
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

    # ------------------------------------------------------------------
    # Frozen Phase7C baseline + frozen OOS reconstruction.
    # ------------------------------------------------------------------
    print("\n========== FROZEN GATE + OOS RECONSTRUCTION ==========")
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

    # Preserve replay ordering before residual-head construction.
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

    groups, _exposed120 = v52.load_groups()
    cases = [
        case
        for _label, group_cases in groups
        for case in group_cases
    ]
    if len(cases) != 1146:
        raise RuntimeError(
            f"Established corpus cardinality drift: {len(cases)}"
        )

    est_x, _, est_margin, est_base_pred, _, _ = v52.capture_features(
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
    print("established_cases=", len(cases))
    print("frozen_oos_inference_complete=YES")
    print("gold_consulted=NO")

    # ------------------------------------------------------------------
    # Raw full Level-2 -> frozen V13.2.
    # ------------------------------------------------------------------
    print("\n========== RAW LEVEL-2 + FROZEN V13.2 ==========")
    full = load_mod("runtime_gate_full_eval", full_path)
    checkpoints = full.validate_environment()
    full_source_before = full.source_snapshot()

    runtime = [
        full.RuntimeTurn(
            context=tuple(case.context),
            utterance=str(case.utterance),
        )
        for case in cases
    ]

    raw_started = time.perf_counter()
    raw_result = full.assemble_level2(runtime, checkpoints)
    raw_batch_ms = (time.perf_counter() - raw_started) * 1000.0
    if len(raw_result.frames) != len(cases):
        raise RuntimeError("Raw Level-2 cardinality mismatch")

    v13_2 = load_mod("runtime_gate_v13_2", v13_2_path)
    companion_chain = [
        getattr(v13_2, "V13_1_FILE", None),
        getattr(v13_2.v13, "V12_FILE", None),
    ]
    print("v13_2_companion_hashes=")
    for p in companion_chain:
        if p is not None and Path(p).is_file():
            print(" ", Path(p), sha256_file(Path(p)))

    v13_started = time.perf_counter()
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
        raw_result,
        checkpoints,
    )
    (
        v13_frames,
        v13_schedules,
        _v13_diag,
        v13_constructor_errors,
    ) = v13_2.construct_v13_2_frames(
        runtime,
        raw_result,
        checkpoints,
        v2_frames,
        v2_schedules,
    )
    v13_batch_ms = (time.perf_counter() - v13_started) * 1000.0

    print("raw_level2_batch1146_ms=", round(raw_batch_ms, 3))
    print("v13_2_wrapper_batch1146_ms=", round(v13_batch_ms, 3))
    print("V2_CONSTRUCTOR_ERROR_count=", len(v2_constructor_errors))
    print("V13_2_CONSTRUCTOR_ERROR_count=", len(v13_constructor_errors))
    if len(v13_frames) != len(cases):
        raise RuntimeError("V13.2 cardinality mismatch")

    # ------------------------------------------------------------------
    # Frozen V7 ambiguity on TOP of frozen V13.2.
    # ------------------------------------------------------------------
    print("\n========== FROZEN V7 + REFERENCE V2 OVER V13.2 ==========")
    ref_v2 = load_mod("runtime_gate_ref_v2", ref_v2_path)
    ref_boundary = load_mod(
        "runtime_gate_ref_boundary",
        ref_boundary_path,
    )

    p7j = base.load_mod("runtime_gate_p7j", base.P7J)
    p7jr = base.load_mod("runtime_gate_p7jr", base.P7JR)
    p7h = base.load_mod("runtime_gate_p7h", base.P7H)
    p8dn = base.load_mod("runtime_gate_p8dn", base.P8DN)

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

    target_started = time.perf_counter()
    raw_details, detail_probs = comp.ungated_phase7j(
        cases,
        p7j,
        detail_model,
        detail_tok,
    )
    corrected_details, _correction_reasons = (
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

    ref_kinds = [str(x) for x in raw_result.reference_kinds]
    dense_pairs = [
        tuple(str(x) for x in pair)
        for pair in raw_result.dense_pairs
    ]

    v13_ambiguity = [
        combined.ambiguity_tuple(frame)
        for frame in v13_frames
    ]
    strict_shadow, strict_reasons, candidate_units, op_state = (
        v7.compose_v3(
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
    )
    safe_shadow, safe_reasons = v7.apply_stability_wrapper(
        v13_ambiguity,
        strict_shadow,
        strict_reasons,
    )

    # --------------------------------------------------------------
    # Frozen post-V7 ownership rule.
    #
    # Proven on all established 1,146 cases:
    # if V13.2 already has an active option ambiguity, V7 may erase it
    # through learned Phase7H resolution only when deterministic strong
    # resolution OR explicit operator surface evidence supports that claim.
    # --------------------------------------------------------------
    owned_shadow = []
    owned_reasons = []
    ownership_restores = []

    for i, (case, before, after, v7_reason) in enumerate(
        zip(cases, v13_ambiguity, safe_shadow, safe_reasons)
    ):
        current = after
        reason = "preserve:v7_output"

        if (
            before[0] == "option_reference"
            and after[0] == "none"
            and str(v7_reason).startswith("option:phase7h_resolved:")
        ):
            candidates = tuple(
                str(x) for x in candidate_units.get(i, ())
            )
            op, resolved, conf = op_state.get(
                i, ("none", "", 0.0)
            )

            strong_op, strong_resolved = v7.strong_option_resolution(
                str(case.utterance),
                tuple(case.context),
                candidates,
                p7h,
                p7jr,
            )
            explicit_support, support_reason = (
                ownership.explicit_support_for_operator(
                    str(case.utterance),
                    str(op),
                    candidates,
                )
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
                    "owned_after": current,
                    "phase7h_op": str(op),
                    "phase7h_resolved": str(resolved),
                    "phase7h_confidence": round(float(conf), 6),
                    "deterministic_strong_op": str(strong_op),
                    "deterministic_strong_resolved": str(strong_resolved),
                    "explicit_surface_support": bool(explicit_support),
                    "surface_support_reason": str(support_reason),
                })

        owned_shadow.append(current)
        owned_reasons.append(reason)

    print("POST_V7_OWNERSHIP_RESTORE_count=", len(ownership_restores))
    for row in ownership_restores[:15]:
        print("POST_V7_OWNERSHIP_RESTORE", row)

    scalar_reference_active = [
        bool(int(x))
        for x in est_base_pred[:, 0].tolist()
    ]
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
    ref_candidate, ref_binary_reasons = (
        ref_v2.compose_reference_candidate(
            [int(x) for x in scalar_reference_active],
            reference_evidence,
        )
    )

    v13_reference = [
        combined.reference_value(frame)
        for frame in v13_frames
    ]
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
            v13_reference,
            scalar_reference_active,
            ref_candidate,
            ref_kinds,
            reference_evidence,
            owned_shadow,
        )
    ):
        value, reason, failure = combined.compose_typed_reference(
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
                "predicted_kind": str(predicted_kind),
                "baseline_reference": baseline_ref,
                "candidate_active": bool(candidate_active),
                "ambiguity": ambiguity_structure,
            })

    shadow_frames = []
    shadow_constructor_failures = []
    for i, (frame, amb, ref) in enumerate(
        zip(v13_frames, owned_shadow, shadow_reference_values)
    ):
        try:
            shadow_frames.append(
                combined.clone_with_shadow_fields(
                    full,
                    frame,
                    ref,
                    amb[0],
                    amb[1],
                )
            )
        except Exception as exc:
            shadow_constructor_failures.append({
                "index": i,
                "type": type(exc).__name__,
                "error": str(exc),
            })
            shadow_frames.append(frame)

    target_batch_ms = (time.perf_counter() - target_started) * 1000.0

    print(
        "v7_reference_v2_batch1146_ms=",
        round(target_batch_ms, 3),
    )
    print_first(
        "REFERENCE_COMPOSITION_FAILURE",
        reference_composition_failures,
    )
    print_first(
        "SHADOW_CONSTRUCTOR_FAILURE",
        shadow_constructor_failures,
    )

    non_target_changes = []
    reference_changes = []
    ambiguity_changes = []
    for i, (before, after) in enumerate(zip(v13_frames, shadow_frames)):
        if (
            combined.frozen_field_signature(before)
            != combined.frozen_field_signature(after)
        ):
            non_target_changes.append({"index": i})
        if combined.reference_value(before) != combined.reference_value(after):
            reference_changes.append({
                "index": i,
                "before": combined.reference_value(before),
                "after": combined.reference_value(after),
                "binary_reason": ref_binary_reasons[i],
                "typed_reason": shadow_reference_reasons[i],
            })
        if combined.ambiguity_tuple(before) != combined.ambiguity_tuple(after):
            ambiguity_changes.append({
                "index": i,
                "before": combined.ambiguity_tuple(before),
                "after": combined.ambiguity_tuple(after),
                "v7_reason": safe_reasons[i],
                "ownership_reason": owned_reasons[i],
            })

    print_first("FROZEN_NON_TARGET_FIELD_CHANGE", non_target_changes)
    print_first("REFERENCE_FIELD_CHANGE", reference_changes)
    print_first("AMBIGUITY_FIELD_CHANGE", ambiguity_changes)

    # Cross-field structural safety, pre-gold.
    introduced_coherence = []
    for i, frame in enumerate(shadow_frames):
        ref = combined.reference_value(frame)
        amb = combined.ambiguity_tuple(frame)
        if ref == "ambiguous" and amb[0] == "none":
            introduced_coherence.append({
                "index": i,
                "issue": "reference_ambiguous_with_none_ambiguity",
            })
        if (
            amb[0] == "option_reference"
            and frame.selected_option not in (None, "")
        ):
            introduced_coherence.append({
                "index": i,
                "issue": "selected_option_present_while_option_ambiguous",
            })
    print_first(
        "CROSS_FIELD_COHERENCE_VIOLATION",
        introduced_coherence,
    )

    # ------------------------------------------------------------------
    # Runtime adapter -> current planner/generator. Still no gold.
    # ------------------------------------------------------------------
    print("\n========== RUNTIME ADAPTER + CURRENT PLANNER ==========")
    from voiceprobe.v33 import world_model
    from voiceprobe.v33.action_generator import StrategicActionGenerator
    from voiceprobe.v33.actions import ActionKind
    from voiceprobe.v33.mind import AgentMind
    from voiceprobe.v33.mission import adaptive_reschedule_mission

    # Context extraction smoke: patient text must never become semantic input
    # context for the remote clinic interpreter.
    context_mind = AgentMind(adaptive_reschedule_mission())
    context_mind.world.history.extend([
        ("PGAI", "Remote turn one."),
        ("PATIENT", "Patient reply that must be excluded."),
        ("PGAI", "Remote turn two."),
    ])
    context_smoke = context_from_mind(context_mind)
    context_bridge_ok = (
        context_smoke
        == ("Remote turn one.", "Remote turn two.")
    )
    print(
        "RUNTIME_CONTEXT_REMOTE_ONLY_SMOKE=",
        "PASS" if context_bridge_ok else "FAIL",
        context_smoke,
    )

    generator = StrategicActionGenerator()
    adapter_failures = []
    native_unresolved_contract_failures = []
    safe_payload_failures = []
    safe_action_leaks = []
    planner_generation_failures = []
    safe_route_count = 0
    ambiguity_safe_route_count = 0
    oos_safe_route_count = 0
    native_resolved_count = 0
    plan_kind_counts = Counter()

    adapter_started = time.perf_counter()

    for i, (frame, oos_value) in enumerate(
        zip(shadow_frames, est_oos.tolist())
    ):
        amb_active = combined.ambiguity_tuple(frame)[0] != "none"
        oos_active = bool(int(oos_value))

        if amb_active:
            try:
                frame.to_remote_observation()
            except Exception as exc:
                if type(exc).__name__ != "UnresolvedSemanticFrameError":
                    native_unresolved_contract_failures.append({
                        "index": i,
                        "type": type(exc).__name__,
                        "error": str(exc),
                    })
            else:
                native_unresolved_contract_failures.append({
                    "index": i,
                    "type": "NO_EXCEPTION",
                    "error": "Native adapter accepted unresolved ambiguity.",
                })

        try:
            if amb_active or oos_active:
                observation = safe_clarification_observation(
                    world_model,
                    frame,
                )
                safe_route_count += 1
                if amb_active:
                    ambiguity_safe_route_count += 1
                if oos_active:
                    oos_safe_route_count += 1

                payload = observation_action_payload(observation)
                if not is_empty_safe_payload(payload):
                    safe_payload_failures.append({
                        "index": i,
                        "payload": payload,
                    })
            else:
                observation = frame.to_remote_observation()
                native_resolved_count += 1
        except Exception as exc:
            adapter_failures.append({
                "index": i,
                "type": type(exc).__name__,
                "error": str(exc),
                "ambiguity": combined.ambiguity_tuple(frame),
                "oos": oos_active,
            })
            continue

        # Deliberately use a fresh but transaction-sensitive mind. If a safe
        # clarification route can accidentally authorize/select even when world
        # state already contains a selection, this catches it.
        mind = AgentMind(adaptive_reschedule_mission())
        mind.world.selected_option = "preexisting-sensitive-option"
        mind.world.selection_verified = True

        try:
            plans = generator.generate(
                mind=mind,
                observation=observation,
            )
        except Exception as exc:
            planner_generation_failures.append({
                "index": i,
                "type": type(exc).__name__,
                "error": str(exc),
                "observation_kind": str(observation.kind),
            })
            continue

        if not plans:
            planner_generation_failures.append({
                "index": i,
                "type": "EMPTY_PLAN_SET",
                "error": "StrategicActionGenerator returned no plans.",
                "observation_kind": str(observation.kind),
            })
            continue

        for plan in plans:
            for kind in plan.kinds:
                plan_kind_counts[str(getattr(kind, "value", kind))] += 1

        if amb_active or oos_active:
            safe_allowed = {
                ActionKind.STATE_GOAL,
                ActionKind.ASK_QUESTION,
            }
            leaked = sorted({
                str(getattr(kind, "value", kind))
                for plan in plans
                for kind in plan.kinds
                if kind not in safe_allowed
            })
            if leaked:
                safe_action_leaks.append({
                    "index": i,
                    "leaked_action_kinds": tuple(leaked),
                    "observation_kind": str(
                        getattr(observation.kind, "value", observation.kind)
                    ),
                })

    adapter_plan_ms = (time.perf_counter() - adapter_started) * 1000.0

    print("native_resolved_adapter_count=", native_resolved_count)
    print("safe_clarification_route_count=", safe_route_count)
    print(
        "ambiguity_safe_route_count=",
        ambiguity_safe_route_count,
    )
    print("oos_safe_route_count=", oos_safe_route_count)
    print(
        "runtime_adapter_planner_batch1146_ms=",
        round(adapter_plan_ms, 3),
    )
    print("planner_action_kind_counts=", dict(plan_kind_counts))
    print_first("RUNTIME_ADAPTER_FAILURE", adapter_failures)
    print_first(
        "NATIVE_UNRESOLVED_CONTRACT_FAILURE",
        native_unresolved_contract_failures,
    )
    print_first("SAFE_PAYLOAD_FAILURE", safe_payload_failures)
    print_first("SAFE_ACTION_LEAK", safe_action_leaks)
    print_first(
        "PLANNER_GENERATION_FAILURE",
        planner_generation_failures,
    )

    # ------------------------------------------------------------------
    # Uncached single-turn lower-bound timing.
    # This intentionally measures the current script architecture, which loads
    # model objects inside assemble_level2(). If this exceeds the provisional
    # budget, the next step is a persistent cached runtime, not a live call.
    # ------------------------------------------------------------------
    print("\n========== SINGLE-TURN UN-CACHED TIMING ==========")
    representative = {}

    for i, (frame, oos_value) in enumerate(
        zip(shadow_frames, est_oos.tolist())
    ):
        amb_active = combined.ambiguity_tuple(frame)[0] != "none"
        oos_active = bool(int(oos_value))
        ref = combined.reference_value(frame)

        if (
            "plain" not in representative
            and not amb_active
            and not oos_active
            and ref == "none"
        ):
            representative["plain"] = i

        if (
            "reference" not in representative
            and not amb_active
            and not oos_active
            and ref not in {"none", "ambiguous"}
        ):
            representative["reference"] = i

        if "ambiguity" not in representative and amb_active:
            representative["ambiguity"] = i

        if len(representative) == 3:
            break

    timing_rows = []
    for role in ("plain", "reference", "ambiguity"):
        idx = representative.get(role)
        if idx is None:
            continue

        one_runtime = [runtime[idx]]
        started = time.perf_counter()
        one_raw = full.assemble_level2(one_runtime, checkpoints)
        (
            one_v2_frames,
            one_v2_schedules,
            _d,
            _f,
            _r,
            _diag,
            one_v2_errors,
        ) = v13_2.v2.construct_candidate_frames(
            one_runtime,
            one_raw,
            checkpoints,
        )
        (
            one_v13_frames,
            _one_sched,
            _one_diag13,
            one_v13_errors,
        ) = v13_2.construct_v13_2_frames(
            one_runtime,
            one_raw,
            checkpoints,
            one_v2_frames,
            one_v2_schedules,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timing_rows.append({
            "role": role,
            "index": idx,
            "ms": elapsed_ms,
            "v2_errors": len(one_v2_errors),
            "v13_errors": len(one_v13_errors),
            "frame_count": len(one_v13_frames),
        })
        print(
            "UNCACHED_SINGLE_TURN",
            {
                "role": role,
                "index": idx,
                "ms": round(elapsed_ms, 3),
                "v2_errors": len(one_v2_errors),
                "v13_errors": len(one_v13_errors),
            },
        )

    timing_values = [row["ms"] for row in timing_rows]
    timing_median_ms = (
        statistics.median(timing_values)
        if timing_values
        else float("inf")
    )
    timing_max_ms = max(timing_values) if timing_values else float("inf")
    print(
        "PROVISIONAL_SINGLE_TURN_SEMANTIC_BUDGET_MS=",
        PROVISIONAL_SINGLE_TURN_SEMANTIC_BUDGET_MS,
    )
    print(
        "uncached_single_turn_median_ms=",
        round(timing_median_ms, 3),
    )
    print(
        "uncached_single_turn_max_ms=",
        round(timing_max_ms, 3),
    )

    if timing_median_ms <= PROVISIONAL_SINGLE_TURN_SEMANTIC_BUDGET_MS:
        latency_class = "WITHIN_PROVISIONAL_BUDGET"
    elif timing_median_ms <= 2000.0:
        latency_class = "OVER_1S_UNDER_2S__CACHE_RECOMMENDED"
    else:
        latency_class = "OVER_2S__PERSISTENT_CACHE_REQUIRED"
    print("UNCACHED_SINGLE_TURN_LATENCY_CLASS=", latency_class)

    # ------------------------------------------------------------------
    # Focused current planner tests. No telephony tests.
    # ------------------------------------------------------------------
    print("\n========== FOCUSED OFFLINE PLANNER TESTS ==========")
    repo_root = Path.cwd().resolve()
    tests_ok, test_output = run_focused_pytests(repo_root)
    print("focused_offline_tests=", "PASS" if tests_ok else "FAIL")
    print("FOCUSED_TEST_OUTPUT_BEGIN")
    print(test_output)
    print("FOCUSED_TEST_OUTPUT_END")

    # ------------------------------------------------------------------
    # GOLD SCORING ONLY NOW.
    # ------------------------------------------------------------------
    print("\n========== GOLD SCORING BEGINS ONLY NOW ==========")
    print("case_id_visible_to_runtime_inference=NO")
    print("expected_gold_visible_to_runtime_inference=NO")

    est_gold = v52.gold_case_tensor(cases).long()
    oos_exact = int((est_oos == est_gold[:, 2]).sum())
    print(
        "frozen_oos_established_exact=",
        f"{oos_exact}/{len(cases)}",
    )

    v13_failures = [
        combined.fields_failed(full, case, frame)
        for case, frame in zip(cases, v13_frames)
    ]
    shadow_failures = [
        combined.fields_failed(full, case, frame)
        for case, frame in zip(cases, shadow_frames)
    ]

    v13_exact = sum(not fs for fs in v13_failures)
    shadow_exact = sum(not fs for fs in shadow_failures)

    v13_reference_exact = sum(
        "reference" not in fs for fs in v13_failures
    )
    shadow_reference_exact = sum(
        "reference" not in fs for fs in shadow_failures
    )
    v13_ambiguity_exact = sum(
        not (fs & combined.AMB_FIELDS)
        for fs in v13_failures
    )
    shadow_ambiguity_exact = sum(
        not (fs & combined.AMB_FIELDS)
        for fs in shadow_failures
    )

    regressions = []
    fixes = []
    non_target_score_drift = []
    for i, (base_fs, shadow_fs) in enumerate(
        zip(v13_failures, shadow_failures)
    ):
        if not base_fs and shadow_fs:
            regressions.append({
                "index": i,
                "case_id": str(cases[i].case_id),
                "new_failures": sorted(shadow_fs - base_fs),
            })
        if base_fs and not shadow_fs:
            fixes.append({
                "index": i,
                "case_id": str(cases[i].case_id),
                "fixed_fields": sorted(base_fs),
            })

        base_non_target = base_fs - combined.TARGET_FIELDS
        shadow_non_target = shadow_fs - combined.TARGET_FIELDS
        if base_non_target != shadow_non_target:
            non_target_score_drift.append({
                "index": i,
                "case_id": str(cases[i].case_id),
                "v13_non_target": sorted(base_non_target),
                "shadow_non_target": sorted(shadow_non_target),
            })

    shadow_failure_map = {
        str(case.case_id): tuple(sorted(fs))
        for case, fs in zip(cases, shadow_failures)
        if fs
    }
    declared_conflict_ok = (
        shadow_failure_map
        == {DECLARED_EXPOSED_CONFLICT: ("transaction_signal",)}
    )

    print("v13_2_fullframe_exact=", f"{v13_exact}/{len(cases)}")
    print(
        "combined_shadow_fullframe_exact=",
        f"{shadow_exact}/{len(cases)}",
    )
    print(
        "v13_2_reference_exact=",
        f"{v13_reference_exact}/{len(cases)}",
    )
    print(
        "combined_shadow_reference_exact=",
        f"{shadow_reference_exact}/{len(cases)}",
    )
    print(
        "v13_2_ambiguity_exact=",
        f"{v13_ambiguity_exact}/{len(cases)}",
    )
    print(
        "combined_shadow_ambiguity_exact=",
        f"{shadow_ambiguity_exact}/{len(cases)}",
    )
    print_first("FULLFRAME_REGRESSION", regressions)
    print_first("FULLFRAME_FIX", fixes)
    print_first(
        "FROZEN_NON_TARGET_SCORE_DRIFT",
        non_target_score_drift,
    )
    print("shadow_failure_map=", shadow_failure_map)
    print(
        "declared_exposed_conflict_only=",
        "YES" if declared_conflict_ok else "NO",
    )

    # ------------------------------------------------------------------
    # Postflight integrity.
    # ------------------------------------------------------------------
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
    print("reasoner_modified=NO")
    print("telephony_modified=NO")
    print("training_performed=NO")

    # ------------------------------------------------------------------
    # Verdict.
    # ------------------------------------------------------------------
    constructors_ok = (
        not v2_constructor_errors
        and not v13_constructor_errors
        and not shadow_constructor_failures
        and not reference_composition_failures
    )
    semantic_scope_ok = (
        not non_target_changes
        and not introduced_coherence
        and not regressions
        and not non_target_score_drift
        and declared_conflict_ok
        and shadow_reference_exact == len(cases)
        and shadow_ambiguity_exact == len(cases)
        and oos_exact == len(cases)
    )
    adapter_ok = (
        context_bridge_ok
        and not adapter_failures
        and not native_unresolved_contract_failures
        and not safe_payload_failures
        and not safe_action_leaks
        and not planner_generation_failures
    )
    integrity_ok = (
        source_unchanged
        and watched_unchanged
        and full_source_unchanged
    )

    semantic_runtime_pass = (
        constructors_ok
        and semantic_scope_ok
        and adapter_ok
        and tests_ok
        and integrity_ok
    )

    cache_required = (
        timing_median_ms
        > PROVISIONAL_SINGLE_TURN_SEMANTIC_BUDGET_MS
    )

    print("\n========== AUTHORITATIVE RUNTIME WIRING OFFLINE VERDICT ==========")
    print(
        "V13_2_STACK_PRESERVED=",
        "YES" if semantic_scope_ok else "NO",
    )
    print(
        "AMBIGUITY_V7_OWNERSHIP_REFERENCE_V2_RUNTIME_SAFE=",
        "YES" if adapter_ok else "NO",
    )
    print(
        "FOCUSED_PLANNER_TESTS_PASS=",
        "YES" if tests_ok else "NO",
    )
    print(
        "ZERO_ESTABLISHED_FULLFRAME_REGRESSION=",
        "YES" if not regressions else "NO",
    )
    print(
        "REFERENCE_ESTABLISHED_EXACT=",
        "YES" if shadow_reference_exact == len(cases) else "NO",
    )
    print(
        "AMBIGUITY_ESTABLISHED_EXACT=",
        "YES" if shadow_ambiguity_exact == len(cases) else "NO",
    )
    print(
        "OOS_ESTABLISHED_EXACT=",
        "YES" if oos_exact == len(cases) else "NO",
    )
    print(
        "UNCACHED_SINGLE_TURN_CACHE_REQUIRED=",
        "YES" if cache_required else "NO",
    )

    if not semantic_runtime_pass:
        if not constructors_ok:
            blocker = "SEMANTIC_STACK_CONSTRUCTION"
        elif not semantic_scope_ok:
            blocker = "SEMANTIC_STACK_PARITY_OR_COHERENCE"
        elif not adapter_ok:
            blocker = "RUNTIME_ADAPTER_OR_PLANNER_SAFETY"
        elif not tests_ok:
            blocker = "FOCUSED_CURRENT_PLANNER_TEST_FAILURE"
        else:
            blocker = "POSTFLIGHT_INTEGRITY"

        verdict = "RUNTIME_WIRING_OFFLINE_GATE_BLOCKED"
        next_action = (
            "DO_NOT_MODIFY_RUNTIME_OR_TELEPHONY__LOCALIZE_ONLY_THE_PRINTED_"
            + blocker
            + "_FAILURES"
        )
    elif cache_required:
        blocker = "UNCACHED_SINGLE_TURN_LATENCY"
        verdict = (
            "RUNTIME_WIRING_SEMANTIC_PASS__"
            "PERSISTENT_CACHE_REQUIRED_BEFORE_RUNTIME_PATCH"
        )
        next_action = (
            "BUILD_PERSISTENT_CACHED_LEVEL2_RUNTIME_CANDIDATE_WITH_THE_EXACT_"
            "FROZEN_V13_2_V7_REFERENCE_V2_STACK__TELEPHONY_DISABLED__THEN_RUN_"
            "OFFLINE_TRANSCRIPT_REPLAY_AND_TIMING_PARITY"
        )
    else:
        blocker = "NONE"
        verdict = "RUNTIME_WIRING_OFFLINE_GATE_PASS"
        next_action = (
            "BUILD_FEATURE_FLAGGED_RUNTIME_INTEGRATION_PATCH_WITH_TELEPHONY_"
            "DISABLED__RUN_OFFLINE_TRANSCRIPT_REPLAY_BEFORE_ANY_LIVE_CALL"
        )

    print("PRIMARY_BLOCKER=", blocker)
    print("RUNTIME_WIRING_OFFLINE_VERDICT=", verdict)
    print("NEXT_ACTION=", next_action)
    print("runtime_wiring_offline_gate_v2_completed=YES")

    del detail_tok, detail_model
    del op_tok, op_model
    del gate_tok, gate_model
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
