#!/usr/bin/env python3
"""Read-only presence/requested-fact precedence proof V1.

Candidate structural rule
-------------------------
If the already-frozen dense semantic pair is:
    speech_act = presence_check
    topic      = presence
then a simultaneously predicted requested_fact is incompatible with the
dominant dialogue act and is cleared before V8+ normalization.

This is NOT phrase matching and does not add an open-intent lexical detector.
It preserves the frozen learned pair and suppresses only the contradictory
independent requested-fact head.

Proof requirements:
1. established1146 zero baseline-right regression;
2. no new constructor error;
3. exact target runtime probes construct successfully;
4. presence probe reaches current StrategicActionGenerator with
   RESUME_WORKFLOW / STATE_GOAL available;
5. no source/runtime/telephony writes or training.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

RUNTIME_GATE_BASENAME = (
    "voiceprobe_semanticlab_v2_runtime_wiring_offline_gate_v2.py"
)
EXPECTED_RUNTIME_GATE_SHA256 = (
    "c19cbd0bac21a53cc3955524e91a1d2d3cc62fdf86f5702ec3325287315cb102"
)

PROBES = (
    "Hello, how can I help you today?",
    "How can I help you today?",
    "Are you still there?",
    "How may I help you today?",
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
    for p in candidates:
        rp = p.expanduser().resolve()
        if rp.is_file():
            return rp
    raise SystemExit(
        "Missing required file "
        + basename
        + ". Checked: "
        + ", ".join(str(x) for x in candidates)
    )


def ev(value: Any) -> str:
    raw = getattr(value, "value", None)
    return str(raw if raw is not None else value)


def frame_signature(frame) -> tuple:
    return (
        ev(frame.speech_act),
        ev(frame.topic),
        str(frame.requested_fact),
        tuple(ev(x) for x in frame.failed_constraints),
        tuple(ev(x) for x in frame.proposed_changes),
        tuple(ev(x) for x in frame.retained_constraints),
        tuple(str(x) for x in frame.offered_options),
        str(frame.selected_option),
        tuple(ev(x) for x in frame.record_claims),
        ev(frame.transaction_operation),
        ev(frame.transaction_signal),
        ev(frame.reference),
        ev(frame.ambiguity.kind),
        tuple(str(x) for x in frame.ambiguity.candidates),
    )


def apply_presence_requested_fact_precedence(frames):
    out = []
    hits = []
    for i, frame in enumerate(frames):
        if (
            ev(frame.speech_act) == "presence_check"
            and ev(frame.topic) == "presence"
            and bool(str(frame.requested_fact))
        ):
            hits.append({
                "index": i,
                "requested_fact_before": str(frame.requested_fact),
            })
            frame = dataclasses.replace(frame, requested_fact="")
        out.append(frame)
    return out, hits


def main() -> int:
    print("========== PRESENCE / REQUESTED-FACT PRECEDENCE PROOF V1 ==========")
    print("telephony=DISABLED")
    print("runtime_source_modified=NO")
    print("semantic_source_modified=NO")
    print("training=NO")
    print("lexical_runtime_patch=NO")
    print("gold_visible_before_inference=NO")

    gate_path = resolve_named(RUNTIME_GATE_BASENAME)
    actual = sha256_file(gate_path)
    print("runtime_gate_sha256=", actual)
    if actual != EXPECTED_RUNTIME_GATE_SHA256:
        raise RuntimeError("Runtime Gate V2 source drift")

    rt = load_mod("presence_proof_rt", gate_path)
    combined = load_mod(
        "presence_proof_combined",
        rt.resolve_named(rt.COMBINED_SHADOW_BASENAME),
    )
    full = load_mod(
        "presence_proof_full",
        combined.resolve_named(combined.FULL_BASENAME),
    )
    v132 = load_mod(
        "presence_proof_v132",
        rt.resolve_named(rt.V13_2_BASENAME),
    )

    # established1146 loader
    v7 = load_mod(
        "presence_proof_v7",
        combined.resolve_named(combined.V7_BASENAME),
    )
    sep = load_mod(
        "presence_proof_sep",
        v7.resolve_named(None, v7.SEP_BASENAME),
    )
    loc = load_mod(
        "presence_proof_loc",
        sep.resolve_named(None, sep.LOCALIZER_BASENAME),
    )
    app = load_mod(
        "presence_proof_app",
        loc.resolve_named(None, loc.APP_BASENAME),
    )
    comp = load_mod(
        "presence_proof_comp",
        app.resolve_named(None, app.COMP_BASENAME),
    )
    v52 = load_mod(
        "presence_proof_v52",
        comp.resolve_named(None, comp.V52_BASENAME),
    )

    source_before = full.source_snapshot()
    checkpoints = full.validate_environment()

    groups, _exposed = v52.load_groups()
    cases = [case for _group, rows in groups for case in rows]
    if len(cases) != 1146:
        raise RuntimeError(f"Established corpus drift: {len(cases)}")

    runtime = [
        full.RuntimeTurn(
            context=tuple(case.context),
            utterance=str(case.utterance),
        )
        for case in cases
    ]

    # ------------------------------------------------------------------
    # Full established inference first. No gold yet.
    # ------------------------------------------------------------------
    print("\n========== ESTABLISHED1146 INFERENCE ==========")
    raw = full.assemble_level2(runtime, checkpoints)

    (
        base_v2_frames,
        base_v2_schedules,
        _dense,
        _facts,
        _refs,
        _diag2,
        base_v2_errors,
    ) = v132.v2.construct_candidate_frames(
        runtime,
        raw,
        checkpoints,
    )
    if base_v2_errors:
        raise RuntimeError(
            f"Unexpected established V2 constructor errors: {base_v2_errors}"
        )

    (
        baseline_frames,
        _baseline_sched,
        _baseline_diag,
        baseline_errors,
    ) = v132.construct_v13_2_frames(
        runtime,
        raw,
        checkpoints,
        base_v2_frames,
        base_v2_schedules,
    )

    candidate_v2_frames, precedence_hits = (
        apply_presence_requested_fact_precedence(base_v2_frames)
    )

    (
        candidate_frames,
        _candidate_sched,
        _candidate_diag,
        candidate_errors,
    ) = v132.construct_v13_2_frames(
        runtime,
        raw,
        checkpoints,
        candidate_v2_frames,
        base_v2_schedules,
    )

    print("precedence_hit_count=", len(precedence_hits))
    for row in precedence_hits[:20]:
        print("PRECEDENCE_HIT", row)
    print("baseline_constructor_error_count=", len(baseline_errors))
    print("candidate_constructor_error_count=", len(candidate_errors))
    print("gold_consulted=NO")

    baseline_sig = [frame_signature(f) for f in baseline_frames]
    candidate_sig = [frame_signature(f) for f in candidate_frames]
    changed = [
        i for i, (a, b) in enumerate(zip(baseline_sig, candidate_sig))
        if a != b
    ]
    print("ESTABLISHED_FRAME_CHANGE_count=", len(changed))
    for i in changed[:20]:
        print("ESTABLISHED_FRAME_CHANGE", {
            "index": i,
            "before": baseline_sig[i],
            "after": candidate_sig[i],
        })

    # ------------------------------------------------------------------
    # Runtime probes before any gold scoring.
    # ------------------------------------------------------------------
    print("\n========== RUNTIME PROBES ==========")
    from voiceprobe.v33.action_generator import StrategicActionGenerator
    from voiceprobe.v33.mind import AgentMind
    from voiceprobe.v33.mission import adaptive_reschedule_mission

    generator = StrategicActionGenerator()
    probe_results = []

    for text in PROBES:
        one_runtime = [
            full.RuntimeTurn(context=(), utterance=text)
        ]
        one_raw = full.assemble_level2(one_runtime, checkpoints)

        (
            one_v2,
            one_sched,
            _d,
            _f,
            _r,
            _diag,
            one_v2_errors,
        ) = v132.v2.construct_candidate_frames(
            one_runtime,
            one_raw,
            checkpoints,
        )
        if one_v2_errors:
            raise RuntimeError(
                f"Probe V2 constructor error: {text!r} {one_v2_errors}"
            )

        repaired_v2, hits = apply_presence_requested_fact_precedence(one_v2)

        (
            one_final,
            _s,
            _diag132,
            one_errors,
        ) = v132.construct_v13_2_frames(
            one_runtime,
            one_raw,
            checkpoints,
            repaired_v2,
            one_sched,
        )

        if one_errors:
            frame = one_final[0] if one_final else None
            probe_results.append({
                "turn": text,
                "status": "constructor_error",
                "hits": hits,
                "errors": dict(one_errors),
                "frame": frame_signature(frame) if frame else None,
            })
            print("RUNTIME_PROBE", probe_results[-1])
            continue

        frame = one_final[0]

        try:
            observation = frame.to_remote_observation()
            plans = generator.generate(
                mind=AgentMind(adaptive_reschedule_mission()),
                observation=observation,
            )
            plan_kinds = sorted({
                ev(kind)
                for plan in plans
                for kind in plan.kinds
            })
            result = {
                "turn": text,
                "status": "ok",
                "hits": hits,
                "frame": frame_signature(frame),
                "observation_kind": ev(observation.kind),
                "plan_kinds": tuple(plan_kinds),
            }
        except Exception as exc:
            result = {
                "turn": text,
                "status": "adapter_error",
                "hits": hits,
                "frame": frame_signature(frame),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        probe_results.append(result)
        print("RUNTIME_PROBE", result)

    # Current transcript blocker is the first exact probe.
    target = probe_results[0]
    target_runtime_fixed = (
        target.get("status") == "ok"
        and target.get("observation_kind") == "presence_check"
        and (
            "state_goal" in target.get("plan_kinds", ())
            or "resume_workflow" in target.get("plan_kinds", ())
        )
    )

    # ------------------------------------------------------------------
    # Gold scoring begins only after all candidate inference.
    # ------------------------------------------------------------------
    print("\n========== GOLD SCORING BEGINS ONLY NOW ==========")
    print("case_id_visible_to_inference=NO")
    print("expected_gold_visible_to_inference=NO")

    baseline_failures = [
        combined.fields_failed(full, case, frame)
        for case, frame in zip(cases, baseline_frames)
    ]
    candidate_failures = [
        combined.fields_failed(full, case, frame)
        for case, frame in zip(cases, candidate_frames)
    ]

    regressions = []
    fixes = []
    for i, (before, after) in enumerate(
        zip(baseline_failures, candidate_failures)
    ):
        new_failures = after - before
        fixed = before - after
        if new_failures:
            regressions.append({
                "index": i,
                "case_id": str(cases[i].case_id),
                "new_failures": tuple(sorted(new_failures)),
            })
        if fixed:
            fixes.append({
                "index": i,
                "case_id": str(cases[i].case_id),
                "fixed_fields": tuple(sorted(fixed)),
            })

    baseline_exact = sum(not x for x in baseline_failures)
    candidate_exact = sum(not x for x in candidate_failures)

    print(
        "baseline_v13_2_fullframe_exact=",
        f"{baseline_exact}/{len(cases)}",
    )
    print(
        "candidate_v13_2_fullframe_exact=",
        f"{candidate_exact}/{len(cases)}",
    )
    print("FULLFRAME_REGRESSION_count=", len(regressions))
    for row in regressions[:20]:
        print("FULLFRAME_REGRESSION", row)
    print("FULLFRAME_FIX_count=", len(fixes))
    for row in fixes[:20]:
        print("FULLFRAME_FIX", row)

    # Separate diagnostic: the ontology still cannot represent OPEN_INTENT
    # directly in frozen Phase8A. Do not hide that limitation.
    p8a = load_mod("presence_proof_p8a", full.P8A)
    open_intent_in_frozen_8a = "open_intent" in set(p8a.TOPICS)
    print(
        "FROZEN_PHASE8A_HAS_OPEN_INTENT_TOPIC=",
        "YES" if open_intent_in_frozen_8a else "NO",
    )

    source_after = full.source_snapshot()
    source_unchanged = source_before == source_after

    print("\n========== POSTFLIGHT ==========")
    print(
        "source_tree_python_unchanged=",
        "YES" if source_unchanged else "NO",
    )
    print("runtime_source_modified=NO")
    print("telephony_modified=NO")
    print("training_performed=NO")
    print("candidate_artifact_written=NO")

    zero_regression = not regressions
    no_candidate_errors = not candidate_errors

    print("\n========== AUTHORITATIVE PRECEDENCE PROOF VERDICT ==========")
    print(
        "ESTABLISHED1146_ZERO_REGRESSION=",
        "YES" if zero_regression else "NO",
    )
    print(
        "CANDIDATE_CONSTRUCTOR_CLEAN=",
        "YES" if no_candidate_errors else "NO",
    )
    print(
        "TARGET_RUNTIME_TURN_FIXED=",
        "YES" if target_runtime_fixed else "NO",
    )
    print(
        "SOURCE_INTEGRITY_PASS=",
        "YES" if source_unchanged else "NO",
    )

    if (
        zero_regression
        and no_candidate_errors
        and target_runtime_fixed
        and source_unchanged
    ):
        blocker = "NONE_FOR_CURRENT_TRANSCRIPT"
        verdict = "PRESENCE_REQUESTED_FACT_PRECEDENCE_PROVEN"
        next_action = (
            "FREEZE_STRUCTURAL_PRECEDENCE_RULE__REBUILD_FULLSTACK_RUNTIME_REPLAY_"
            "WITH_RULE_INSERTED_BEFORE_V8__TELEPHONY_DISABLED__REQUIRE_5_TURN_"
            "TRANSCRIPT_PASS_AND_MEDIAN_UNDER_2S__KEEP_OPEN_INTENT_ONTOLOGY_GAP_"
            "EXPLICIT_FOR_NON_PRESENCE_PARAPHRASES"
        )
    else:
        if regressions:
            blocker = "ESTABLISHED_REGRESSION"
        elif candidate_errors:
            blocker = "CANDIDATE_CONSTRUCTOR_ERROR"
        elif not target_runtime_fixed:
            blocker = "TARGET_RUNTIME_NOT_FIXED"
        else:
            blocker = "SOURCE_INTEGRITY"
        verdict = "PRESENCE_REQUESTED_FACT_PRECEDENCE_BLOCKED"
        next_action = (
            "DO_NOT_PATCH_RUNTIME__LOCALIZE_ONLY_PRINTED_" + blocker
        )

    print("PRIMARY_BLOCKER=", blocker)
    print("PRESENCE_REQUESTED_FACT_PRECEDENCE_VERDICT=", verdict)
    print("NEXT_ACTION=", next_action)
    print("presence_requested_fact_precedence_proof_v1_completed=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
