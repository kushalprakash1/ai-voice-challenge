#!/usr/bin/env python3
"""Phase 7C deterministic reference composition proof V2 (read-only).

Purpose
-------
After Composition V7 ambiguity/OOS is frozen, prove the V2 deterministic
reference composition using the collision-free multi-option boundary from
the read-only V2 localizer, without changing any model, threshold, source
file, checkpoint, or runtime wiring.

This audit separates:
  1. binary Phase 7C reference-gate error;
  2. Phase 7D typed-reference error;
  3. deterministic antecedent/applicability error;
  4. option-selection evidence error.

No gold label, case ID, family, category, or expected frame is visible to
inference. Gold is consulted only after every prediction/evidence record has
been computed.

Provider pronouns are treated gender-invariantly (he/she/they etc.) while the
actual provider antecedent identity remains in context. No pronoun is rewritten
into another pronoun.
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
from typing import Any

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

V7_BASENAME = "voiceprobe_semanticlab_v2_phase7c_composition_v7_unclear_ownership_proof.py"
EXPECTED_V7_SHA256 = "1313d5f81bade95b667cf32391681ec31b15d357d8cef3f7473b8edf1eb88eb1"

BOUNDARY_BASENAME = "voiceprobe_semanticlab_v2_phase7c_reference_multioption_boundary_localizer_v2.py"
EXPECTED_BOUNDARY_SHA256 = "6d62f40178c79fb02b861641ea961c57b20ae3e2b8eeb2675a300796b4236cc9"

PROVIDER_PRONOUN_RE = re.compile(
    r"\b(?:he|she|they|him|her|them|his|hers|their|theirs)\b",
    re.I,
)
PROVIDER_DEICTIC_RE = re.compile(
    r"\b(?:that|same|this)\s+(?:doctor|provider|clinician|physician)\b",
    re.I,
)
TIME_DEICTIC_RE = re.compile(
    r"\b(?:that|same)\s+(?:time|hour)\b"
    r"|\b(?:around|near|close\s+to)\s+that\s+time\b",
    re.I,
)
DAY_DEICTIC_RE = re.compile(
    r"\b(?:that|same)\s+(?:day|date)\b"
    r"|\bon\s+that\s+day\b",
    re.I,
)
OPTION_DEICTIC_RE = re.compile(
    r"\b(?:that|this|same)\s+(?:one|option|choice|slot|appointment)\b"
    r"|\b(?:that|this)\s+works\b"
    r"|\b(?:take|use|choose|pick|go\s+with)\s+(?:that|this|it)\b",
    re.I,
)
ORDINAL_OR_COMPARATIVE_RE = re.compile(
    r"\b(?:first|second|1st|2nd|former|latter|earlier|later|earliest|latest|sooner)\b",
    re.I,
)
META_CLARIFICATION_RE = re.compile(
    r"\b(?:repeat|say\s+that\s+again|say\s+it\s+again|what\s+did\s+you\s+say|"
    r"can\s+you\s+hear\s+me|still\s+hear\s+me)\b",
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
        "Could not locate required file "
        + basename
        + ". Checked: "
        + ", ".join(checked)
    )


def context_of(row) -> tuple[str, ...]:
    return tuple(str(x) for x in getattr(row, "context", ()))


def turn_text(row) -> str:
    if hasattr(row, "utterance"):
        return str(row.utterance)
    return str(getattr(row, "turn", ""))


def row_id(row, index: int) -> str:
    # Post-inference display only.
    for attr in ("case_id", "id", "family"):
        value = getattr(row, attr, None)
        if value:
            return str(value)
    return f"row_{index}"


def regex_values(pattern, text: str) -> tuple[str, ...]:
    if not pattern:
        return ()
    try:
        vals = []
        for m in re.finditer(pattern, str(text), re.I):
            v = str(m.group(0)).strip()
            if v and v.casefold() not in {x.casefold() for x in vals}:
                vals.append(v)
        return tuple(vals)
    except Exception:
        return ()


def context_values(pattern, context) -> tuple[str, ...]:
    out = []
    for c in context:
        for v in regex_values(pattern, c):
            if v.casefold() not in {x.casefold() for x in out}:
                out.append(v)
    return tuple(out)


def provider_anchors(context, p7jr) -> tuple[str, ...]:
    return context_values(getattr(p7jr, "PROVIDER", None), context)


def day_anchors(context, p7jr) -> tuple[str, ...]:
    return context_values(getattr(p7jr, "DAY", None), context)


def time_anchors(context, p7jr) -> tuple[str, ...]:
    vals = []
    for attr in ("TIME", "DAYPART"):
        for v in context_values(getattr(p7jr, attr, None), context):
            if v.casefold() not in {x.casefold() for x in vals}:
                vals.append(v)
    return tuple(vals)


def current_values(turn: str, p7jr) -> dict[str, tuple[str, ...]]:
    return {
        "provider": regex_values(getattr(p7jr, "PROVIDER", None), turn),
        "day": regex_values(getattr(p7jr, "DAY", None), turn),
        "time": tuple(
            list(regex_values(getattr(p7jr, "TIME", None), turn))
            + list(regex_values(getattr(p7jr, "DAYPART", None), turn))
        ),
    }



def safe_option_evidence(row, v7, p7jr, p7h, loc):
    context = context_of(row)
    turn = turn_text(row)
    candidates = tuple(v7.structured_candidates(context, p7jr))
    alt_ok, alt_reason, _diag = v7.positive_option_structure(
        context, candidates, p7jr, loc
    )

    strong_kind, strong_resolved = v7.strong_option_resolution(
        turn, context, candidates, p7h, p7jr
    )

    if strong_resolved:
        return {
            "positive": True,
            "kind": "prior_option",
            "reason": f"resolved_option:{strong_kind}",
            "candidates": candidates,
            "resolved": str(strong_resolved),
            "alternative_structure": bool(alt_ok),
            "candidate_count": len(candidates),
        }

    # One prior offer may be referred to deictically even though it is not a
    # multi-alternative set. This remains evidence only when the turn is clearly
    # deictic and the resolver exposes exactly one candidate.
    if len(candidates) == 1 and OPTION_DEICTIC_RE.search(turn):
        return {
            "positive": True,
            "kind": "prior_option",
            "reason": "single_candidate_deictic",
            "candidates": candidates,
            "resolved": str(candidates[0]),
            "alternative_structure": False,
            "candidate_count": len(candidates),
        }

    # A vague deictic over multiple real alternatives is ambiguity, not a
    # resolved reference.
    if len(candidates) >= 2 and alt_ok and OPTION_DEICTIC_RE.search(turn):
        return {
            "positive": False,
            "kind": "prior_option",
            "reason": "multiple_options_vague_deictic",
            "candidates": candidates,
            "resolved": "",
            "alternative_structure": True,
            "candidate_count": len(candidates),
        }

    return {
        "positive": False,
        "kind": "prior_option",
        "reason": (
            f"no_resolved_option:{strong_kind}:"
            f"count={len(candidates)}:alt={bool(alt_ok)}:{alt_reason}"
        ),
        "candidates": candidates,
        "resolved": "",
        "alternative_structure": bool(alt_ok),
        "candidate_count": len(candidates),
    }


def structural_reference_evidence(row, predicted_kind, v7, p7jr, p7h, loc):
    """Inference-only evidence record. No labels or IDs."""
    context = context_of(row)
    turn = turn_text(row)

    providers = provider_anchors(context, p7jr)
    days = day_anchors(context, p7jr)
    times = time_anchors(context, p7jr)
    current = current_values(turn, p7jr)

    surface_provider = bool(
        PROVIDER_PRONOUN_RE.search(turn)
        or PROVIDER_DEICTIC_RE.search(turn)
    )
    surface_time = bool(TIME_DEICTIC_RE.search(turn))
    surface_day = bool(DAY_DEICTIC_RE.search(turn))

    provider_positive = (
        surface_provider
        and bool(providers)
        and not bool(current["provider"])
    )
    time_positive = (
        surface_time
        and bool(times)
        and not bool(current["time"])
    )
    day_positive = (
        surface_day
        and bool(days)
        and not bool(current["day"])
    )

    option = safe_option_evidence(row, v7, p7jr, p7h, loc)

    structural_kinds = []
    reasons = []
    if provider_positive:
        structural_kinds.append("prior_provider")
        reasons.append("provider_pronoun_with_provider_antecedent")
    if time_positive:
        structural_kinds.append("prior_time")
        reasons.append("time_deictic_with_time_antecedent")
    if day_positive:
        structural_kinds.append("prior_day")
        reasons.append("day_deictic_with_day_antecedent")
    if option["positive"]:
        structural_kinds.append("prior_option")
        reasons.append(option["reason"])

    # Strong negative evidence is semantic, not label-driven.
    negative_reasons = []
    if META_CLARIFICATION_RE.search(turn):
        negative_reasons.append("metaclarification")
    if surface_provider and not providers:
        negative_reasons.append("provider_pronoun_missing_provider_antecedent")
    if surface_time and not times:
        negative_reasons.append("time_deictic_missing_time_antecedent")
    if surface_day and not days:
        negative_reasons.append("day_deictic_missing_day_antecedent")
    if current["provider"] and not context:
        negative_reasons.append("explicit_provider_without_prior_context")
    if current["time"] and not context and not OPTION_DEICTIC_RE.search(turn):
        negative_reasons.append("explicit_time_without_prior_context")
    if current["day"] and not context and not OPTION_DEICTIC_RE.search(turn):
        negative_reasons.append("explicit_day_without_prior_context")
    if option["reason"] == "multiple_options_vague_deictic":
        negative_reasons.append("vague_multi_option_is_ambiguity_not_reference")

    # Generic option/reference boundary: if context presents 2+ genuine
    # alternatives and the existing deterministic selection operator cannot
    # uniquely resolve one, that turn cannot safely be promoted to a resolved
    # reference. This is structural and independent of benchmark wording.
    if (
        int(option.get("candidate_count", len(option.get("candidates", ())))) >= 2
        and bool(option.get("alternative_structure", False))
        and not str(option.get("resolved", ""))
    ):
        negative_reasons.append("multi_option_without_unique_resolution")

    predicted_kind = str(predicted_kind)
    kind_agrees = predicted_kind in structural_kinds
    any_positive = bool(structural_kinds)
    unique_positive = len(set(structural_kinds)) == 1

    if kind_agrees:
        diagnosis = "typed_kind_and_structure_agree"
    elif any_positive and predicted_kind == "unresolved":
        diagnosis = "phase7d_unresolved_despite_structural_reference"
    elif any_positive:
        diagnosis = "phase7d_kind_disagrees_with_structural_reference"
    elif predicted_kind != "unresolved":
        diagnosis = "phase7d_typed_reference_without_structural_support"
    else:
        diagnosis = "no_reference_evidence"

    return {
        "predicted_kind": predicted_kind,
        "structural_kinds": tuple(structural_kinds),
        "kind_agrees": bool(kind_agrees),
        "unique_positive": bool(unique_positive),
        "positive": bool(kind_agrees and unique_positive),
        "strong_negative": bool(negative_reasons),
        "positive_reasons": tuple(reasons),
        "negative_reasons": tuple(negative_reasons),
        "diagnosis": diagnosis,
        "provider_anchors": providers,
        "day_anchors": days,
        "time_anchors": times,
        "current_values": current,
        "option": option,
    }


def predict_ref_kinds(rows, p7d, model, tok):
    runtime = [
        SimpleNamespace(
            context=context_of(row),
            utterance=turn_text(row),
        )
        for row in rows
    ]
    preds, probs = p7d.predict(model, tok, runtime)
    return [str(x) for x in preds], probs




def augment_reference_evidence_v2(row, evidence, boundary, p8dn):
    """Attach only the already-proven V2 boundary features.

    These are inference-only semantic features:
      - active_offer / passive_inventory / neutral_alternatives
      - selection surface class
      - normalized transaction operation
    """
    out = dict(evidence)
    out["multioption_context_role"] = str(
        boundary.context_role(context_of(row))
    )
    out["multioption_selection_surface"] = str(
        boundary.selection_surface(turn_text(row))
    )
    out["transaction_operation"] = str(
        p8dn.normalize_operation(turn_text(row))
    )
    return out


def unresolved_multioption(evidence) -> bool:
    option = evidence.get("option", {})
    return bool(
        int(option.get("candidate_count", len(option.get("candidates", ())))) >= 2
        and bool(option.get("alternative_structure", False))
        and not str(option.get("resolved", ""))
    )


def v2_multioption_suppression_reason(evidence) -> str:
    """Collision-free V2 option/reference boundary.

    Proven by the V2 localizer:
      * active offer/proposal + tentative evaluation => ambiguity-only;
      * active offer/proposal + commitment/acceptance/comparative => preserve
        reference baseline;
      * passive inventory/list + no transaction => ambiguity-only;
      * passive inventory/list + real transaction operation => preserve
        reference baseline;
      * neutral alternative contexts are preserved rather than guessed.

    This function does not activate reference. Positive activation remains owned
    by the existing typed-antecedent / unique-selection evidence.
    """
    if not unresolved_multioption(evidence):
        return ""

    role = str(evidence.get("multioption_context_role", ""))
    surface = str(evidence.get("multioption_selection_surface", ""))
    operation = str(evidence.get("transaction_operation", "none"))

    if role == "active_offer":
        if surface == "tentative_evaluation":
            return "suppress:v2_active_offer_tentative_evaluation"
        return ""

    if role == "passive_inventory":
        if operation != "none":
            return ""
        return "suppress:v2_passive_inventory_without_transaction"

    return ""



def positive_activation_reason(evidence) -> str:
    """High-precision positive reference evidence.

    1. A uniquely resolved option is sufficient even when Phase7D abstains.
    2. Provider/day/time references require typed-kind + structural agreement.
    """
    option = evidence["option"]
    if bool(option.get("positive")) and str(option.get("resolved", "")):
        return "activate:unique_option_resolution:" + str(option.get("reason"))

    if (
        bool(evidence.get("positive"))
        and str(evidence.get("predicted_kind"))
        in {"prior_provider", "prior_day", "prior_time"}
    ):
        return (
            "activate:typed_antecedent_agreement:"
            + str(evidence.get("predicted_kind"))
        )

    return ""


def negative_suppression_reason(evidence) -> str:
    """V2 high-precision suppression evidence.

    The broad V1 multi-option suppression is retired. All other V1 negative
    evidence remains unchanged.
    """
    v2_multi = v2_multioption_suppression_reason(evidence)
    if v2_multi:
        return v2_multi

    reasons = tuple(str(x) for x in evidence.get("negative_reasons", ()))
    priority = (
        "provider_pronoun_missing_provider_antecedent",
        "time_deictic_missing_time_antecedent",
        "day_deictic_missing_day_antecedent",
        "explicit_provider_without_prior_context",
        "explicit_time_without_prior_context",
        "explicit_day_without_prior_context",
        "metaclarification",
    )
    for item in priority:
        if item in reasons:
            return "suppress:" + item
    return ""



def compose_reference_candidate(baseline, evidence):
    """Conservative monotonic reference composition."""
    out = []
    reasons = []

    for base, ev in zip(baseline, evidence):
        base = int(base)
        pos = positive_activation_reason(ev)
        neg = negative_suppression_reason(ev)

        if pos and neg:
            out.append(base)
            reasons.append("preserve:conflicting_positive_negative_evidence")
        elif pos:
            out.append(1)
            reasons.append(pos)
        elif neg:
            out.append(0)
            reasons.append(neg)
        else:
            out.append(base)
            reasons.append("preserve:no_decisive_structural_evidence")

    return out, reasons


def evaluate_candidate(name, rows, baseline, candidate, gold, evidence, reasons):
    n = len(rows)
    regressions = []
    fixes = []
    failures = []

    for i, (base, cand, target) in enumerate(zip(baseline, candidate, gold)):
        base = int(base)
        cand = int(cand)
        target = int(target)

        rec = {
            "index": i,
            "id": row_id(rows[i], i),
            "gold": target,
            "baseline": base,
            "candidate": cand,
            "reason": reasons[i],
            "phase7d": evidence[i]["predicted_kind"],
            "structural_kinds": evidence[i]["structural_kinds"],
            "positive_reasons": evidence[i]["positive_reasons"],
            "negative_reasons": evidence[i]["negative_reasons"],
            "context_role": evidence[i].get("multioption_context_role"),
            "selection_surface": evidence[i].get("multioption_selection_surface"),
            "transaction_operation": evidence[i].get("transaction_operation"),
            "turn": turn_text(rows[i]),
            "context": context_of(rows[i]),
        }

        if base == target and cand != target:
            regressions.append(rec)
        if base != target and cand == target:
            fixes.append(rec)
        if cand != target:
            failures.append(rec)

    base_exact = sum(int(b) == int(g) for b, g in zip(baseline, gold))
    cand_exact = sum(int(c) == int(g) for c, g in zip(candidate, gold))

    print(f"\n========== {name} REFERENCE COMPOSITION ==========")
    print("base_exact=", f"{base_exact}/{n}")
    print("candidate_exact=", f"{cand_exact}/{n}")
    print("baseline_right_regressions=", len(regressions))
    print("fixes=", len(fixes))
    print("candidate_failure_count=", len(failures))
    print("reason_counts=", dict(Counter(reasons)))

    for label, records in (
        ("REFERENCE_REGRESSION", regressions),
        ("REFERENCE_FIX", fixes),
        ("REFERENCE_REMAINING_FAIL", failures),
    ):
        for rec in records[:20]:
            print(label, rec)

    return {
        "n": n,
        "base_exact": base_exact,
        "candidate_exact": cand_exact,
        "regressions": len(regressions),
        "fixes": len(fixes),
        "failures": len(failures),
    }



def evaluate_binary(name, rows, baseline, gold, evidence):
    n = len(rows)
    fp = []
    fn = []
    correct_pos = []
    correct_neg = []

    for i in range(n):
        b = int(baseline[i])
        g = int(gold[i])
        rec = {
            "index": i,
            "id": row_id(rows[i], i),
            "gold": g,
            "baseline": b,
            "turn": turn_text(rows[i]),
            "context": context_of(rows[i]),
            "evidence": evidence[i],
        }
        if g == 1 and b == 0:
            fn.append(rec)
        elif g == 0 and b == 1:
            fp.append(rec)
        elif g == 1:
            correct_pos.append(rec)
        else:
            correct_neg.append(rec)

    exact = n - len(fp) - len(fn)
    print(f"\n========== {name} ==========")
    print("baseline_exact=", f"{exact}/{n}")
    print("false_positive_count=", len(fp))
    print("false_negative_count=", len(fn))
    print("gold_active_count=", sum(int(x) for x in gold))
    print("baseline_active_count=", sum(int(x) for x in baseline))

    print("FN_DIAGNOSIS_COUNTS=", dict(Counter(x["evidence"]["diagnosis"] for x in fn)))
    print("FP_DIAGNOSIS_COUNTS=", dict(Counter(x["evidence"]["diagnosis"] for x in fp)))
    print(
        "FN_STRUCTURAL_POSITIVE=",
        sum(1 for x in fn if x["evidence"]["positive"]),
        "/",
        len(fn),
    )
    print(
        "FP_STRONG_NEGATIVE=",
        sum(1 for x in fp if x["evidence"]["strong_negative"]),
        "/",
        len(fp),
    )

    for label, items in (("REFERENCE_FN", fn), ("REFERENCE_FP", fp)):
        for rec in items[:20]:
            e = rec["evidence"]
            print(label, {
                "index": rec["index"],
                "id": rec["id"],
                "gold": rec["gold"],
                "baseline": rec["baseline"],
                "phase7d": e["predicted_kind"],
                "structural_kinds": e["structural_kinds"],
                "kind_agrees": e["kind_agrees"],
                "strong_negative": e["strong_negative"],
                "positive_reasons": e["positive_reasons"],
                "negative_reasons": e["negative_reasons"],
                "diagnosis": e["diagnosis"],
                "option": e["option"],
                "turn": rec["turn"],
                "context": rec["context"],
            })

    return {
        "name": name,
        "n": n,
        "exact": exact,
        "fp": fp,
        "fn": fn,
        "fn_structural_positive": sum(
            1 for x in fn if x["evidence"]["positive"]
        ),
        "fp_strong_negative": sum(
            1 for x in fp if x["evidence"]["strong_negative"]
        ),
    }


def main() -> int:
    print("========== PHASE 7C REFERENCE COMPOSITION PROOF V2 ==========")
    print("telephony=DISABLED")
    print("training=NO")
    print("gradient_updates=NO")
    print("runtime_wiring_modified=NO")
    print("ambiguity_v7_modified=NO")
    print("oos_modified=NO")
    print("reference_candidate_applied=SHADOW_ONLY")
    print("gold_visible_to_inference=NO")

    v7_path = resolve_named(V7_BASENAME)
    v7_hash = sha256_file(v7_path)
    print("v7_source=", v7_path)
    print("v7_sha256=", v7_hash)
    if v7_hash != EXPECTED_V7_SHA256:
        raise RuntimeError(
            f"V7 source drift expected={EXPECTED_V7_SHA256} actual={v7_hash}"
        )
    v7 = load_mod("ref_audit_v7", v7_path)

    boundary_path = resolve_named(BOUNDARY_BASENAME)
    boundary_hash = sha256_file(boundary_path)
    print("boundary_v2_source=", boundary_path)
    print("boundary_v2_sha256=", boundary_hash)
    if boundary_hash != EXPECTED_BOUNDARY_SHA256:
        raise RuntimeError(
            f"Boundary V2 source drift expected={EXPECTED_BOUNDARY_SHA256} actual={boundary_hash}"
        )
    boundary = load_mod("ref_comp_v2_boundary", boundary_path)

    # Load exact V7 dependency chain by its own hash locks.
    sep_path = v7.resolve_named(None, v7.SEP_BASENAME)
    if sha256_file(sep_path) != v7.EXPECTED_SEP_SHA256:
        raise RuntimeError("Separator source drift")
    sep = load_mod("ref_audit_sep", sep_path)

    loc_path = sep.resolve_named(None, sep.LOCALIZER_BASENAME)
    if sha256_file(loc_path) != sep.EXPECTED_LOCALIZER_SHA256:
        raise RuntimeError("Localizer source drift")
    loc = load_mod("ref_audit_loc", loc_path)

    app_path = loc.resolve_named(None, loc.APP_BASENAME)
    if sha256_file(app_path) != loc.EXPECTED_APP_SHA256:
        raise RuntimeError("Applicability source drift")
    app = load_mod("ref_audit_app", app_path)

    comp_path = app.resolve_named(None, app.COMP_BASENAME)
    if sha256_file(comp_path) != app.EXPECTED_COMP_SHA256:
        raise RuntimeError("Composition source drift")
    comp = load_mod("ref_audit_comp", comp_path)

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

    v52 = load_mod("ref_audit_v52", v52_path)
    feas = load_mod("ref_audit_feas", feas_path)
    struct = load_mod("ref_audit_struct", struct_path)
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

    # Reconstruct Phase7C baseline features. No residual/reference candidate is
    # trained or applied.
    random.seed(v52.SEED)
    torch.manual_seed(v52.SEED)
    gate_ck, gate_model, gate_tok = v52.load_current_model()
    thresholds = {
        f: float(gate_ck["thresholds"][f])
        for f in v52.FIELDS
    }

    original_train, original_val = v52.build_synthetic()
    original_train, original_val, _blocked, blocked_files = (
        feas.filter_original_synthetic(v52, original_train, original_val)
    )

    # Preserve the same early capture order as prior audits.
    v52.capture_features(
        gate_model,
        gate_tok,
        v52.runtime_for_examples(original_train),
        thresholds,
    )
    oval_x, _, _oval_margin, oval_base, _, _ = v52.capture_features(
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

    if (
        len(historical) != 133
        or len(established) != 1146
        or len(exposed) != 120
    ):
        raise RuntimeError(
            "Corpus cardinality drift: "
            + repr((len(historical), len(established), len(exposed)))
        )

    def capture(rows, are_cases):
        runtime = (
            v52.runtime_for_cases(rows)
            if are_cases
            else v52.runtime_for_examples(rows)
        )
        return v52.capture_features(
            gate_model, gate_tok, runtime, thresholds
        )

    hist_x, _, _hist_margin, hist_base, _, _ = capture(historical, True)
    est_x, _, _est_margin, est_base, _, _ = capture(established, True)
    exp_x, _, _exp_margin, exp_base, _, _ = capture(exposed, True)
    tgt_x, _, _tgt_margin, tgt_base, _, _ = capture(targeted, False)
    prb_x, _, _prb_margin, prb_base, _, _ = capture(probes, False)

    # Existing typed-reference and option-selection specialists.
    p7d = base.load_mod("ref_audit_p7d", base.P7D)
    p7jr = base.load_mod("ref_audit_p7jr", base.P7JR)
    p7h = base.load_mod("ref_audit_p7h", base.P7H)
    p8dn = base.load_mod("ref_comp_v2_p8dn", base.P8DN)

    # V7's already-proven in-memory candidate coverage is frozen and reused
    # read-only for option evidence.
    app.install_extended_candidate_coverage(p7jr, p7h)

    ck7d = base.load_checkpoint(base.A7D)
    ref_model = p7d.RefKindModel()
    ref_model.load_state_dict(ck7d["state_dict"])
    ref_model.eval()
    ref_tok = base.tokenizer_for(
        ck7d.get(
            "model_name",
            getattr(
                p7d,
                "MODEL_NAME",
                "distilbert/distilbert-base-uncased",
            ),
        )
    )

    datasets = [
        ("original_val", original_val, False, oval_base),
        ("historical133", historical, True, hist_base),
        ("established1146", established, True, est_base),
        ("exposed120", exposed, True, exp_base),
        ("targeted", targeted, False, tgt_base),
        ("metamorphic", probes, False, prb_base),
    ]

    # -------------------- INFERENCE / EVIDENCE ONLY --------------------
    inference = {}
    for name, rows, are_cases, base_pred in datasets:
        kinds, probs = predict_ref_kinds(rows, p7d, ref_model, ref_tok)
        evidence = [
            augment_reference_evidence_v2(
                row,
                structural_reference_evidence(
                    row,
                    kind,
                    v7,
                    p7jr,
                    p7h,
                    loc,
                ),
                boundary,
                p8dn,
            )
            for row, kind in zip(rows, kinds)
        ]
        inference[name] = {
            "rows": rows,
            "are_cases": are_cases,
            "baseline": [int(x) for x in base_pred[:, 0].tolist()],
            "kinds": kinds,
            "probs": probs,
            "evidence": evidence,
        }

    print("REFERENCE_INFERENCE_COMPLETE=YES")
    print("gold_consulted_before_inference=NO")

    # Pre-gold semantic boundary smoke.
    smoke_rows = [
        SimpleNamespace(
            context=("Dr. Calder has a Thursday afternoon opening.",),
            turn="Does she have any other times nearby?",
        ),
        SimpleNamespace(
            context=("Dr. Calder has a Thursday afternoon opening.",),
            turn="Does he have any other times nearby?",
        ),
        SimpleNamespace(
            context=("Dr. Calder has a Thursday afternoon opening.",),
            turn="Do they have any other times nearby?",
        ),
        SimpleNamespace(
            context=("The 2:30 PM slot is unavailable.",),
            turn="Can you check another opening close to that time?",
        ),
        SimpleNamespace(
            context=("Tuesday has no remaining openings.",),
            turn="What other availability is there that day?",
        ),
        SimpleNamespace(
            context=("I can offer Monday morning or Thursday evening.",),
            turn="Could the first option work?",
        ),
    ]
    smoke_kinds, _ = predict_ref_kinds(smoke_rows, p7d, ref_model, ref_tok)
    smoke_evidence = [
        augment_reference_evidence_v2(
            row,
            structural_reference_evidence(
                row, kind, v7, p7jr, p7h, loc
            ),
            boundary,
            p8dn,
        )
        for row, kind in zip(smoke_rows, smoke_kinds)
    ]
    print("GENDER_INVARIANT_PROVIDER_SMOKE_PHASE7D=", smoke_kinds[:3])
    print(
        "GENDER_INVARIANT_PROVIDER_SMOKE_STRUCTURE=",
        [x["structural_kinds"] for x in smoke_evidence[:3]],
    )
    print(
        "REFERENCE_BOUNDARY_SMOKE=",
        [
            {
                "phase7d": k,
                "structural": e["structural_kinds"],
                "positive": e["positive"],
                "diagnosis": e["diagnosis"],
            }
            for k, e in zip(smoke_kinds, smoke_evidence)
        ],
    )

    # --------------------------- GOLD BOUNDARY -------------------------
    print("\n========== GOLD SCORING BEGINS ONLY NOW ==========")

    results = {}
    for name, data in inference.items():
        rows = data["rows"]
        if data["are_cases"]:
            gold = [
                int(x)
                for x in v52.gold_case_tensor(rows)[:, 0].long().tolist()
            ]
        else:
            gold = [
                int(x)
                for x in v52.gold_example_tensor(rows)[:, 0].long().tolist()
            ]

        candidate, candidate_reasons = compose_reference_candidate(
            data["baseline"],
            data["evidence"],
        )
        results[name] = evaluate_candidate(
            name,
            rows,
            data["baseline"],
            candidate,
            gold,
            data["evidence"],
            candidate_reasons,
        )

    print("\n========== REFERENCE COMPOSITION SUMMARY ==========")
    for name, metrics in results.items():
        print(name, metrics)

    stability_names = (
        "historical133",
        "established1146",
        "exposed120",
        "targeted",
        "metamorphic",
    )
    zero_regression = all(
        results[name]["regressions"] == 0
        for name in stability_names
    )
    fresh_exact = all(
        results[name]["candidate_exact"] == results[name]["n"]
        for name in ("targeted", "metamorphic")
    )
    established_exposed_non_degrading = (
        results["established1146"]["candidate_exact"]
        >= results["established1146"]["base_exact"]
        and results["exposed120"]["candidate_exact"]
        >= results["exposed120"]["base_exact"]
    )
    historical_exact = (
        results["historical133"]["candidate_exact"]
        == results["historical133"]["n"]
    )
    original_val_exact = (
        results["original_val"]["candidate_exact"]
        == results["original_val"]["n"]
    )

    if (
        zero_regression
        and fresh_exact
        and established_exposed_non_degrading
        and historical_exact
        and original_val_exact
    ):
        verdict = "REFERENCE_COMPOSITION_V2_PROVEN"
        next_action = (
            "RUN_EXACTLY_ONE_INDEPENDENT_COLD_REPRODUCTION_OF_THIS_REFERENCE_"
            "COMPOSITION__KEEP_COMPOSITION_V7_AMBIGUITY_OOS_AND_ALL_OTHER_"
            "SEMANTICFRAME_FIELDS_FROZEN"
        )
        primary_blocker = "NONE"
    elif not zero_regression:
        verdict = "REFERENCE_COMPOSITION_V2_STABILITY_BLOCKED"
        next_action = (
            "DO_NOT_PROMOTE__LOCALIZE_ONLY_THE_PRINTED_BASELINE_RIGHT_"
            "REFERENCE_REGRESSIONS_WITHOUT_CHANGING_AMBIGUITY_OR_OOS"
        )
        primary_blocker = "BASELINE_RIGHT_REFERENCE_REGRESSION"
    elif not fresh_exact:
        verdict = "REFERENCE_COMPOSITION_V2_CAPABILITY_UNDERCOVERAGE"
        next_action = (
            "DO_NOT_TRAIN__LOCALIZE_ONLY_THE_PRINTED_FRESH_REFERENCE_FAILURES_"
            "AND_EXTEND_THE_EXISTING_TYPED_ANTECEDENT_OR_SELECTION_COMPOSITION"
        )
        primary_blocker = "FRESH_REFERENCE_UNDERCOVERAGE"
    else:
        verdict = "REFERENCE_COMPOSITION_V2_NOT_PROVEN"
        next_action = (
            "DO_NOT_PROMOTE__LOCALIZE_ONLY_THE_PRINTED_REFERENCE_FAILURE_"
            "SURFACE_WITH_ALL_FROZEN_FIELDS_UNCHANGED"
        )
        primary_blocker = "REFERENCE_PROOF_INVARIANT"

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

    print("\n========== AUTHORITATIVE REFERENCE COMPOSITION VERDICT ==========")
    print(
        "ZERO_BASELINE_RIGHT_REFERENCE_REGRESSION=",
        "YES" if zero_regression else "NO",
    )
    print(
        "FRESH_REFERENCE_EXACT=",
        "YES" if fresh_exact else "NO",
    )
    print(
        "HISTORICAL_REFERENCE_EXACT=",
        "YES" if historical_exact else "NO",
    )
    print(
        "ORIGINAL_VAL_REFERENCE_EXACT=",
        "YES" if original_val_exact else "NO",
    )
    print(
        "ESTABLISHED_EXPOSED_REFERENCE_NON_DEGRADING=",
        "YES" if established_exposed_non_degrading else "NO",
    )
    print("PRIMARY_BLOCKER=", primary_blocker)
    print("REFERENCE_COMPOSITION_VERDICT=", verdict)
    print("NEXT_ACTION=", next_action)
    print("reference_composition_proof_v2_completed=YES")

    del ref_tok, ref_model
    del gate_tok, gate_model
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
