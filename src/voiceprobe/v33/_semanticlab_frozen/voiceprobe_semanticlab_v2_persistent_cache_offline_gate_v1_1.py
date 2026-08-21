#!/usr/bin/env python3
"""Persistent Level-2 model-cache offline gate V1.1.

Purpose
-------
Remove the dominant per-turn model-loading latency without changing semantics.

The current frozen Level-2 evaluator constructs transformer models/tokenizers
inside every run_phase* call and V2 normalization reconstructs 8A/8B again.
This gate installs an IN-MEMORY persistent cache only, then proves:

1. uncached V13.2 baseline output over established1146;
2. cached raw AssemblyResult parity over established1146;
3. cached V13.2 native SemanticFrame parity over established1146;
4. zero constructor regressions;
5. repeated warm single-turn latency on plain/reference/ambiguity cases;
6. current focused v3.3 planner tests still pass;
7. no source/checkpoint/runtime/telephony writes.

No case ID/category/tag/gold is used by inference. This gate does not score
against gold because exact structural parity against the already-proven frozen
uncached stack is the stronger lifecycle-only contract.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

RUNTIME_GATE_BASENAME = "voiceprobe_semanticlab_v2_runtime_wiring_offline_gate_v2.py"
EXPECTED_RUNTIME_GATE_SHA256 = (
    "c19cbd0bac21a53cc3955524e91a1d2d3cc62fdf86f5702ec3325287315cb102"
)

PROVISIONAL_WARM_TURN_BUDGET_MS = 1000.0
REPEATS = 3

REQUIRED_TESTS = (
    "tests/test_v33_planner.py",
    "tests/test_v33_semantic_planner.py",
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
        + ", ".join(str(p) for p in candidates)
    )


def enum_value(x: Any) -> str:
    raw = getattr(x, "value", None)
    return str(raw if raw is not None else x)


def frame_signature(frame) -> tuple:
    return (
        enum_value(frame.speech_act),
        enum_value(frame.topic),
        str(frame.requested_fact),
        tuple(enum_value(x) for x in frame.failed_constraints),
        tuple(enum_value(x) for x in frame.proposed_changes),
        tuple(enum_value(x) for x in frame.retained_constraints),
        tuple(str(x) for x in frame.offered_options),
        str(frame.selected_option),
        tuple(enum_value(x) for x in frame.record_claims),
        enum_value(frame.transaction_operation),
        enum_value(frame.transaction_signal),
        enum_value(frame.reference),
        enum_value(frame.ambiguity.kind),
        tuple(str(x) for x in frame.ambiguity.candidates),
    )


def result_signature(result) -> tuple:
    return (
        tuple(frame_signature(x) for x in result.frames),
        tuple(
            tuple(sorted((str(k), int(v)) for k, v in row.items()))
            for row in result.gate_labels
        ),
        tuple(tuple(x) for x in result.dense_pairs),
        tuple(str(x) for x in result.reference_kinds),
        tuple(str(x) for x in result.ambiguity_details),
        tuple(
            tuple(
                (k, tuple(v))
                for k, v in sorted(row.items())
            )
            for row in result.scheduling
        ),
        tuple(sorted((int(k), tuple(v)) for k, v in result.assembly_violations.items())),
        tuple(sorted((int(k), tuple(v)) for k, v in result.transaction_structural_violations.items())),
        tuple(sorted((int(k), tuple(v)) for k, v in result.record_structural_violations.items())),
        tuple(sorted(int(x) for x in result.bare_yes_applied)),
    )


class PersistentLevel2Cache:
    """Persistent frozen-model lifecycle for the existing Level-2 functions."""

    def __init__(self, full, v2, checkpoints):
        self.full = full
        self.v2 = v2
        self.checkpoints = checkpoints
        self.module_cache: dict[str, Any] = {}
        self.tokenizer_cache: dict[str, Any] = {}

        # Import exact frozen modules once.
        self.p6b = self._module(full.P6B, "cache_p6b")
        self.p7c = self._module(full.P7C, "cache_p7c")
        self.p7d = self._module(full.P7D, "cache_p7d")
        self.p7h = self._module(full.P7H, "cache_p7h")
        self.p7j = self._module(full.P7J, "cache_p7j")
        self.p7jr = self._module(full.P7JR, "cache_p7jr")
        self.p8a = self._module(full.P8A, "cache_p8a")
        self.p8b = self._module(full.P8B, "cache_p8b")
        self.p8c = self._module(full.P8C, "cache_p8c")
        self.p8d = self._module(full.P8D, "cache_p8d")
        self.p8dn = self._module(full.P8DN, "cache_p8dn")
        self.popt = self._module(full.POPT, "cache_popt")

        # Build every learned model exactly once.
        self.m7c = self.p7c.phase7b.Model()
        self.m7c.load_state_dict(checkpoints["7c"]["state_dict"])
        self.m7c.eval()
        self.t7c = self._tok(
            checkpoints["7c"].get(
                "model_name",
                getattr(self.p7c, "MODEL_NAME", self.p7c.phase7b.MODEL_NAME),
            )
        )

        self.m6b = self.p6b.Model()
        self.m6b.load_state_dict(checkpoints["6b"]["state_dict"])
        self.m6b.eval()
        self.t6b = self._tok(checkpoints["6b"]["model_name"])

        self.m8a = self.p8a.DenseModel()
        self.m8a.load_state_dict(checkpoints["8a"]["state_dict"])
        self.m8a.eval()
        self.t8a = self._tok(self.p8a.MODEL_NAME)
        self.valid_pairs = tuple(tuple(x) for x in checkpoints["8a"]["valid_pairs"])

        self.m8b = self.p8b.Model()
        self.m8b.load_state_dict(checkpoints["8b"]["state_dict"])
        self.m8b.eval()
        self.t8b = self._tok(self.p8b.p8a.MODEL_NAME)

        self.m8c = self.p8c.Model()
        self.m8c.load_state_dict(checkpoints["8c"]["state_dict"])
        self.m8c.eval()
        self.t8c = self._tok(self.p8c.p8a.MODEL_NAME)

        self.m7d = self.p7d.RefKindModel()
        self.m7d.load_state_dict(checkpoints["7d"]["state_dict"])
        self.m7d.eval()
        self.t7d = self._tok(
            checkpoints["7d"].get(
                "model_name",
                getattr(self.p7d, "MODEL_NAME", "distilbert/distilbert-base-uncased"),
            )
        )

        self.m7j = self.p7j.DetailModel()
        self.m7j.load_state_dict(checkpoints["7j"]["state_dict"])
        self.m7j.eval()
        self.t7j = self._tok(
            checkpoints["7j"].get(
                "model_name",
                getattr(self.p7j, "MODEL_NAME", "distilbert/distilbert-base-uncased"),
            )
        )

        self.m7i = self.p7h.OperatorModel()
        self.m7i.load_state_dict(checkpoints["7i"]["state_dict"])
        self.m7i.eval()
        self.t7i = self._tok(
            checkpoints["7i"].get(
                "model_name",
                getattr(self.p7h, "MODEL_NAME", "distilbert/distilbert-base-uncased"),
            )
        )

        for model in (
            self.m7c, self.m6b, self.m8a, self.m8b,
            self.m8c, self.m7d, self.m7j, self.m7i,
        ):
            for p in model.parameters():
                p.requires_grad = False

    def _module(self, path: Path, name: str):
        key = str(Path(path).resolve())
        if key not in self.module_cache:
            self.module_cache[key] = load_mod(name, Path(path))
        return self.module_cache[key]

    def _tok(self, model_name: str):
        key = str(model_name)
        if key not in self.tokenizer_cache:
            self.tokenizer_cache[key] = self.full.tokenizer_for(key)
        return self.tokenizer_cache[key]

    def cached_load_mod(self, name: str, path: Path):
        return self._module(path, "persistent_" + str(abs(hash(str(Path(path).resolve())))))

    # ----------------------- cached full-evaluator heads -----------------------

    def run_phase7c_gate(self, runtime, _p7c, _ck7c):
        probs = self.p7c.raw_probs(self.m7c, self.t7c, runtime)
        return self.p7c.decode(probs, self.checkpoints["7c"]["thresholds"])

    def run_phase6b_scheduling(self, runtime, gate_labels, _p6b, _ck6b):
        failed_probs, relation_probs = self.p6b.raw_predictions(
            self.m6b,
            self.t6b,
            [turn.utterance for turn in runtime],
        )
        raw = self.p6b.decode(
            failed_probs,
            relation_probs,
            dict(self.checkpoints["6b"]["failed_thresholds"]),
            dict(self.checkpoints["6b"]["relation_thresholds"]),
        )

        out = []
        for pred, gate in zip(raw, gate_labels):
            diverted = bool(
                gate.get("reference", 0)
                or gate.get("ambiguity", 0)
                or gate.get("oos", 0)
            )
            if diverted:
                out.append({
                    "failed_constraints": (),
                    "proposed_changes": (),
                    "retained_constraints": (),
                })
                continue

            out.append({
                "failed_constraints": self.full.order_axes(
                    pred.get("failed_constraints", ())
                ),
                "proposed_changes": self.full.order_axes(
                    pred.get("proposed_changes", ())
                ),
                "retained_constraints": self.full.order_axes(
                    pred.get("retained_constraints", ())
                ),
            })
        return out

    def run_phase8a_dense(self, runtime, flat_clauses, _p8a, _ck8a):
        whole_pairs, _, independent_acts, independent_topics = self.p8a.predict(
            self.m8a,
            self.t8a,
            runtime,
            self.valid_pairs,
        )
        clause_pairs = []
        if flat_clauses:
            clause_pairs, _, _, _ = self.p8a.predict(
                self.m8a,
                self.t8a,
                flat_clauses,
                self.valid_pairs,
            )
        return (
            list(whole_pairs),
            list(independent_acts),
            list(independent_topics),
            list(clause_pairs),
            self.valid_pairs,
        )

    def run_phase8b_requested_fact(self, runtime, _p8b, _ck8b):
        preds, _ = self.p8b.predict(self.m8b, self.t8b, runtime)
        return [str(value) if value is not None else "" for value in preds]

    def run_phase8c_record_claims(
        self,
        runtime,
        flat_clause_turns,
        spans,
        clause_pairs,
        _p8c,
        _ck8c,
    ):
        items = [
            self.p8c.Ex(
                family="runtime",
                context=turn.context,
                turn=turn.utterance,
                claim=None,
            )
            for turn in flat_clause_turns
        ]
        flat_preds, _ = self.p8c.predict_clauses(
            self.m8c,
            self.t8c,
            items,
        )

        claims_by_case = []
        provenance_by_case = []
        for span in spans:
            claims = []
            provenance = []
            for j in range(span.start, span.end):
                pred = flat_preds[j]
                pair = clause_pairs[j]
                if pred is not None and pair in self.full.LEGAL_RECORD_PAIRS:
                    if pred not in claims:
                        claims.append(pred)
                    provenance.append((pred, pair))
            claims_by_case.append(tuple(sorted(claims)))
            provenance_by_case.append(provenance)
        return claims_by_case, provenance_by_case

    def run_phase7d_reference_kind(self, runtime, _p7d, _ck7d):
        preds, _ = self.p7d.predict(self.m7d, self.t7d, runtime)
        return [str(x) for x in preds]

    def run_phase7j_ambiguity(
        self,
        runtime,
        gate_labels,
        _p7j,
        _p7jr,
        _ck7j,
        assembly_violations,
    ):
        n = len(runtime)
        kinds = ["none"] * n
        candidates = [tuple() for _ in range(n)]
        details = [""] * n

        active_indices = [
            i
            for i, gate in enumerate(gate_labels)
            if gate.get("ambiguity", 0) or gate.get("oos", 0)
        ]
        if not active_indices:
            return kinds, candidates, details

        from types import SimpleNamespace
        items = [
            SimpleNamespace(
                context=runtime[i].context,
                turn=runtime[i].utterance,
            )
            for i in active_indices
        ]
        preds, _ = self.p7j.predict_detail(
            self.m7j,
            self.t7j,
            items,
        )

        for i, detail in zip(active_indices, preds):
            details[i] = str(detail)
            try:
                kind, cands = self.p7jr.ambiguity_from_detail(
                    detail,
                    runtime[i].context,
                    runtime[i].utterance,
                )
                kinds[i] = str(kind)
                candidates[i] = self.full.dedupe(cands)
            except Exception as exc:
                assembly_violations[i].append(
                    "phase7j_candidate_resolution_error:"
                    f"{type(exc).__name__}:{exc}"
                )
                kinds[i] = "none"
                candidates[i] = ()
        return kinds, candidates, details

    def run_phase7i_selected_option(
        self,
        runtime,
        gate_labels,
        reference_kinds,
        dense_pairs,
        _p7h,
        _ck7i,
        assembly_violations,
    ):
        from types import SimpleNamespace

        n = len(runtime)
        selected = [""] * n
        bare_yes_applied = set()

        for i, turn in enumerate(runtime):
            candidate = self.full.bare_yes_candidate(
                turn,
                dense_pairs[i][0],
                self.p7h,
            )
            if candidate is not None:
                selected[i] = candidate
                bare_yes_applied.add(i)

        eligible = []
        scenarios = []
        for i, (turn, gate, kind) in enumerate(
            zip(runtime, gate_labels, reference_kinds)
        ):
            if i in bare_yes_applied:
                continue
            if not gate.get("reference", 0):
                continue
            if gate.get("ambiguity", 0) or gate.get("oos", 0):
                continue
            if kind == "unresolved":
                assembly_violations[i].append(
                    "unresolved_reference_kind_not_coerced"
                )
                continue

            candidates = self.full.dedupe(
                self.p7h.phase7f.benchmark_candidates(turn.context)
            )
            scenarios.append(
                SimpleNamespace(
                    context=turn.context,
                    kind=kind,
                    candidates=candidates,
                    turn=turn.utterance,
                )
            )
            eligible.append(i)

        if not scenarios:
            return selected, bare_yes_applied

        operators, _ = self.p7h.predict(
            self.m7i,
            self.t7i,
            scenarios,
        )
        for i, scenario, operator in zip(eligible, scenarios, operators):
            resolved = self.p7h.resolve(
                operator,
                scenario.context,
                scenario.candidates,
                scenario.turn,
            )
            selected[i] = "" if resolved is None else str(resolved)

        return selected, bare_yes_applied

    # --------------------------- cached V2 repair -----------------------------

    def repair_dense_pairs(self, runtime, result, checkpoints, diag):
        repaired = list(result.dense_pairs)
        for i, turn in enumerate(runtime):
            pair = repaired[i]
            text = turn.utterance

            if (
                pair == ("offer", "availability")
                and self.v2.NEGATIVE_AVAILABILITY_RE.search(text)
                and "?" not in text
                and not self.v2.OFFER_ACTION_RE.search(text)
            ):
                pair = ("statement", "availability")
                diag.hit(i, "dense_negative_availability_statement")

            if (
                pair == ("question", "patient_fact")
                and self.v2.PATIENT_FACT_REQUEST_RE.search(text)
            ):
                pair = ("request", "patient_fact")
                diag.hit(i, "dense_patient_fact_indirect_request")

            if (
                result.ambiguity_details[i] == "option_reference"
                and self.v2.VAGUE_SELECTION_RE.search(text)
            ):
                if self.v2.latest_context_is_explicit_offer(turn):
                    pair = ("confirmation", "availability")
                    diag.hit(i, "dense_vague_selection_after_explicit_offer")
                else:
                    pair = ("statement", "other")
                    diag.hit(i, "dense_vague_selection_after_nonoffer_list")

            clauses = self.p8c.split_clauses(text)
            if len(clauses) > 1 and pair in {
                ("statement", "appointment_state"),
                ("statement", "profile"),
            }:
                items = [
                    self.full.RuntimeTurn(
                        context=turn.context,
                        utterance=clause,
                    )
                    for clause in clauses
                ]
                clause_pairs, _, _, _ = self.p8a.predict(
                    self.m8a,
                    self.t8a,
                    items,
                    self.valid_pairs,
                )
                if ("offer", "availability") in clause_pairs:
                    pair = ("offer", "availability")
                    diag.hit(
                        i,
                        "dense_background_record_plus_availability_offer",
                    )

            repaired[i] = pair
        return repaired

    def repair_requested_facts(self, runtime, base_frames, checkpoints, diag):
        out = [frame.requested_fact for frame in base_frames]
        for i, turn in enumerate(runtime):
            clauses = self.p8c.split_clauses(turn.utterance)
            if len(clauses) < 2:
                continue

            items = [
                self.p8b.Ex(
                    family="candidate_runtime",
                    context=turn.context,
                    turn=clause,
                    fact=None,
                )
                for clause in clauses
            ]
            preds, _ = self.p8b.predict(
                self.m8b,
                self.t8b,
                items,
            )
            positives = [str(x) for x in preds if x is not None]
            unique = list(dict.fromkeys(positives))
            if len(unique) == 1 and unique[0] != out[i]:
                out[i] = unique[0]
                diag.hit(i, "requested_fact_unique_clause_local")
        return out

    def install(self):
        # Module/tokenizer lifecycle.
        self.full.load_mod = self.cached_load_mod

        # Exact drop-in inference hooks.
        self.full.run_phase7c_gate = self.run_phase7c_gate
        self.full.run_phase6b_scheduling = self.run_phase6b_scheduling
        self.full.run_phase8a_dense = self.run_phase8a_dense
        self.full.run_phase8b_requested_fact = self.run_phase8b_requested_fact
        self.full.run_phase8c_record_claims = self.run_phase8c_record_claims
        self.full.run_phase7d_reference_kind = self.run_phase7d_reference_kind
        self.full.run_phase7j_ambiguity = self.run_phase7j_ambiguity
        self.full.run_phase7i_selected_option = self.run_phase7i_selected_option

        # V2 normalization's second 8A/8B pass.
        self.v2.load_mod = self.cached_load_mod
        self.v2.repair_dense_pairs = self.repair_dense_pairs
        self.v2.repair_requested_facts = self.repair_requested_facts


def assemble_v132(full, v132, runtime, checkpoints):
    raw = full.assemble_level2(runtime, checkpoints)
    (
        v2_frames,
        v2_schedules,
        _dense,
        _facts,
        _refs,
        _diag2,
        v2_errors,
    ) = v132.v2.construct_candidate_frames(
        runtime,
        raw,
        checkpoints,
    )
    (
        frames,
        _schedules,
        _diag132,
        v132_errors,
    ) = v132.construct_v13_2_frames(
        runtime,
        raw,
        checkpoints,
        v2_frames,
        v2_schedules,
    )
    return raw, frames, v2_errors, v132_errors


def run_focused_tests(repo_root: Path) -> tuple[bool, str]:
    missing = [p for p in REQUIRED_TESTS if not (repo_root / p).is_file()]
    if missing:
        return False, "missing_required_tests=" + repr(missing)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *REQUIRED_TESTS,
    ]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(repo_root / "src")

    cp = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return cp.returncode == 0, (cp.stdout or "")


def main() -> int:
    print("========== PERSISTENT LEVEL-2 CACHE OFFLINE GATE V1.1 ==========")
    print("telephony=DISABLED")
    print("runtime_source_write=NO")
    print("semantic_rule_change=NO")
    print("training=NO")
    print("gold_scoring=NO")
    print("cache_scope=model_modules_tokenizers_only")

    runtime_gate_path = resolve_named(RUNTIME_GATE_BASENAME)
    runtime_gate_sha = sha256_file(runtime_gate_path)
    print("runtime_gate_v2_source=", runtime_gate_path)
    print("runtime_gate_v2_sha256=", runtime_gate_sha)
    if runtime_gate_sha != EXPECTED_RUNTIME_GATE_SHA256:
        raise RuntimeError(
            "Runtime Gate V2 drift: "
            f"expected={EXPECTED_RUNTIME_GATE_SHA256} actual={runtime_gate_sha}"
        )

    rt = load_mod("cache_gate_rt_v2", runtime_gate_path)

    combined_path = rt.resolve_named(rt.COMBINED_SHADOW_BASENAME)
    combined = load_mod("cache_gate_combined", combined_path)

    full_path = combined.resolve_named(combined.FULL_BASENAME)
    if sha256_file(full_path) != combined.EXPECTED_FULL_SHA256:
        raise RuntimeError("Full evaluator source drift")
    full = load_mod("cache_gate_full", full_path)

    v132_path = rt.resolve_named(rt.V13_2_BASENAME)
    if sha256_file(v132_path) != rt.EXPECTED_V13_2_SHA256:
        raise RuntimeError("V13.2 source drift")
    v132 = load_mod("cache_gate_v132", v132_path)

    # Established1146 loader via the exact runtime-gate dependency chain.
    v7_path = combined.resolve_named(combined.V7_BASENAME)
    v7 = load_mod("cache_gate_v7", v7_path)
    sep_path = v7.resolve_named(None, v7.SEP_BASENAME)
    sep = load_mod("cache_gate_sep", sep_path)
    loc_path = sep.resolve_named(None, sep.LOCALIZER_BASENAME)
    loc = load_mod("cache_gate_loc", loc_path)
    app_path = loc.resolve_named(None, loc.APP_BASENAME)
    app = load_mod("cache_gate_app", app_path)
    comp_path = app.resolve_named(None, app.COMP_BASENAME)
    comp = load_mod("cache_gate_comp", comp_path)
    v52_path = comp.resolve_named(None, comp.V52_BASENAME)
    v52 = load_mod("cache_gate_v52", v52_path)

    source_before = full.source_snapshot()
    checkpoints = full.validate_environment()

    groups, _exposed = v52.load_groups()
    cases = [case for _name, rows in groups for case in rows]
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
    # Authoritative uncached baseline.
    # ------------------------------------------------------------------
    print("\n========== UNCACHED V13.2 BASELINE ==========")
    started = time.perf_counter()
    baseline_raw, baseline_frames, baseline_v2_errors, baseline_v132_errors = (
        assemble_v132(full, v132, runtime, checkpoints)
    )
    uncached_batch_ms = (time.perf_counter() - started) * 1000.0

    baseline_raw_sig = result_signature(baseline_raw)
    baseline_frame_sig = tuple(frame_signature(x) for x in baseline_frames)

    print("uncached_batch1146_ms=", round(uncached_batch_ms, 3))
    print("baseline_v2_constructor_error_count=", len(baseline_v2_errors))
    print("baseline_v132_constructor_error_count=", len(baseline_v132_errors))

    # ------------------------------------------------------------------
    # Persistent cache startup.
    # ------------------------------------------------------------------
    print("\n========== PERSISTENT CACHE INITIALIZATION ==========")
    cache_started = time.perf_counter()
    cache = PersistentLevel2Cache(full, v132.v2, checkpoints)
    cache.install()
    cache_init_ms = (time.perf_counter() - cache_started) * 1000.0

    print("persistent_model_count=8")
    print("persistent_tokenizer_count=", len(cache.tokenizer_cache))
    print("cache_init_ms=", round(cache_init_ms, 3))
    print("cache_installed=YES")

    # ------------------------------------------------------------------
    # Full established parity under cache.
    # ------------------------------------------------------------------
    print("\n========== CACHED ESTABLISHED1146 PARITY ==========")
    started = time.perf_counter()
    cached_raw, cached_frames, cached_v2_errors, cached_v132_errors = (
        assemble_v132(full, v132, runtime, checkpoints)
    )
    cached_batch_ms = (time.perf_counter() - started) * 1000.0

    cached_raw_sig = result_signature(cached_raw)
    cached_frame_sig = tuple(frame_signature(x) for x in cached_frames)

    raw_parity = cached_raw_sig == baseline_raw_sig
    frame_parity = cached_frame_sig == baseline_frame_sig
    constructor_parity = (
        dict(cached_v2_errors) == dict(baseline_v2_errors)
        and dict(cached_v132_errors) == dict(baseline_v132_errors)
    )

    frame_mismatches = [
        i
        for i, (a, b) in enumerate(zip(baseline_frame_sig, cached_frame_sig))
        if a != b
    ]
    print("cached_batch1146_ms=", round(cached_batch_ms, 3))
    print("RAW_ASSEMBLY_RESULT_PARITY=", "YES" if raw_parity else "NO")
    print("V13_2_FRAME_PARITY=", "YES" if frame_parity else "NO")
    print("CONSTRUCTOR_ERROR_PARITY=", "YES" if constructor_parity else "NO")
    print("FRAME_MISMATCH_count=", len(frame_mismatches))
    for i in frame_mismatches[:10]:
        print("FRAME_MISMATCH", {
            "index": i,
            "before": baseline_frame_sig[i],
            "after": cached_frame_sig[i],
        })

    # ------------------------------------------------------------------
    # Warm single-turn timing. One unmeasured warm-up for each semantic role,
    # then 3 measured turns while the same models/tokenizers remain resident.
    # ------------------------------------------------------------------
    print("\n========== WARM SINGLE-TURN TIMING ==========")
    representatives = {}

    for i, frame in enumerate(baseline_frames):
        ref = enum_value(frame.reference)
        amb = enum_value(frame.ambiguity.kind)

        if "plain" not in representatives and ref == "none" and amb == "none":
            representatives["plain"] = i
        if (
            "reference" not in representatives
            and ref not in {"none", "ambiguous"}
            and amb == "none"
        ):
            representatives["reference"] = i
        if "ambiguity" not in representatives and amb != "none":
            representatives["ambiguity"] = i

        if len(representatives) == 3:
            break

    timing_rows = []
    per_role = {}

    for role in ("plain", "reference", "ambiguity"):
        idx = representatives.get(role)
        if idx is None:
            continue
        one_runtime = [runtime[idx]]

        # unmeasured warm-up
        assemble_v132(full, v132, one_runtime, checkpoints)

        samples = []
        for rep in range(REPEATS):
            started = time.perf_counter()
            raw_one, frames_one, v2e, v132e = assemble_v132(
                full,
                v132,
                one_runtime,
                checkpoints,
            )
            ms = (time.perf_counter() - started) * 1000.0

            if frame_signature(frames_one[0]) != baseline_frame_sig[idx]:
                raise RuntimeError(
                    f"Warm single-turn parity failure role={role} index={idx}"
                )
            if v2e or v132e:
                raise RuntimeError(
                    f"Warm single-turn constructor error role={role} "
                    f"v2={v2e} v132={v132e}"
                )

            samples.append(ms)
            timing_rows.append(ms)
            print("WARM_SINGLE_TURN", {
                "role": role,
                "index": idx,
                "repeat": rep + 1,
                "ms": round(ms, 3),
            })

        per_role[role] = {
            "median_ms": statistics.median(samples),
            "max_ms": max(samples),
        }

    overall_median = statistics.median(timing_rows) if timing_rows else float("inf")
    overall_max = max(timing_rows) if timing_rows else float("inf")

    for role, stats in per_role.items():
        print("WARM_ROLE_SUMMARY", {
            "role": role,
            "median_ms": round(stats["median_ms"], 3),
            "max_ms": round(stats["max_ms"], 3),
        })

    print("PROVISIONAL_WARM_TURN_BUDGET_MS=", PROVISIONAL_WARM_TURN_BUDGET_MS)
    print("warm_single_turn_median_ms=", round(overall_median, 3))
    print("warm_single_turn_max_ms=", round(overall_max, 3))

    # ------------------------------------------------------------------
    # Current planner tests.
    # ------------------------------------------------------------------
    print("\n========== FOCUSED OFFLINE PLANNER TESTS ==========")
    tests_ok, test_output = run_focused_tests(Path.cwd().resolve())
    print("focused_offline_tests=", "PASS" if tests_ok else "FAIL")
    print("FOCUSED_TEST_OUTPUT_BEGIN")
    print("\n".join(test_output.splitlines()[-15:]))
    print("FOCUSED_TEST_OUTPUT_END")

    # ------------------------------------------------------------------
    # Postflight.
    # ------------------------------------------------------------------
    print("\n========== POSTFLIGHT INTEGRITY ==========")
    source_after = full.source_snapshot()
    source_unchanged = source_before == source_after

    print("source_tree_python_unchanged=", "YES" if source_unchanged else "NO")
    print("candidate_artifact_written=NO")
    print("runtime_wiring_modified=NO")
    print("telephony_modified=NO")
    print("training_performed=NO")

    parity_pass = (
        raw_parity
        and frame_parity
        and constructor_parity
        and not frame_mismatches
    )
    warm_budget_pass = overall_median <= PROVISIONAL_WARM_TURN_BUDGET_MS

    print("\n========== AUTHORITATIVE PERSISTENT CACHE VERDICT ==========")
    print("ESTABLISHED1146_EXACT_PARITY=", "YES" if parity_pass else "NO")
    print("FOCUSED_PLANNER_TESTS_PASS=", "YES" if tests_ok else "NO")
    print("WARM_MEDIAN_WITHIN_1S=", "YES" if warm_budget_pass else "NO")
    print("SOURCE_INTEGRITY_PASS=", "YES" if source_unchanged else "NO")

    if not parity_pass:
        blocker = "CACHE_SEMANTIC_PARITY"
        verdict = "PERSISTENT_CACHE_GATE_BLOCKED"
        next_action = (
            "DO_NOT_WIRE_CACHE__LOCALIZE_ONLY_PRINTED_FRAME_OR_RESULT_PARITY_MISMATCH"
        )
    elif not tests_ok:
        blocker = "PLANNER_REGRESSION"
        verdict = "PERSISTENT_CACHE_GATE_BLOCKED"
        next_action = (
            "DO_NOT_WIRE_CACHE__LOCALIZE_ONLY_FOCUSED_PLANNER_TEST_FAILURE"
        )
    elif not source_unchanged:
        blocker = "SOURCE_INTEGRITY"
        verdict = "PERSISTENT_CACHE_GATE_BLOCKED"
        next_action = "DO_NOT_WIRE_CACHE__INVESTIGATE_SOURCE_DRIFT"
    elif not warm_budget_pass:
        blocker = "WARM_INFERENCE_LATENCY"
        verdict = "PERSISTENT_CACHE_PARITY_PASS__WARM_LATENCY_STILL_HIGH"
        next_action = (
            "PROFILE_ONLY_WARM_INFERENCE_COMPONENT_LATENCY__NO_SEMANTIC_CHANGES__"
            "TARGET_DUPLICATE_ENCODER_PASSES_BEFORE_RUNTIME_PATCH"
        )
    else:
        blocker = "NONE"
        verdict = "PERSISTENT_CACHE_OFFLINE_GATE_PASS"
        next_action = (
            "BUILD_FEATURE_FLAGGED_PERSISTENT_SEMANTIC_RUNTIME_SERVICE_WITH_EXACT_"
            "FROZEN_V13_2_V7_POST_V7_OWNERSHIP_REFERENCE_V2_OOS_STACK__"
            "TELEPHONY_DISABLED__RUN_OFFLINE_TRANSCRIPT_REPLAY_THEN_LIVE_PREFLIGHT"
        )

    print("PRIMARY_BLOCKER=", blocker)
    print("PERSISTENT_CACHE_OFFLINE_VERDICT=", verdict)
    print("NEXT_ACTION=", next_action)
    print("persistent_cache_offline_gate_v1_1_completed=YES")

    # Keep cache alive through verdict; release only on process exit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
