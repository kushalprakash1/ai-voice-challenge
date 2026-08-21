#!/usr/bin/env python3
"""Phase 8A explicit semantic composition probe V2 with confidence-gated Rule A.

No more speech-act-head retraining.

Architecture under test:
- frozen A8A3 legal-pair expert
- selected parent hierarchical Phase8A expert
- deterministic semantic arbitration

Rule A — transaction search offer arbitration, confidence gated
---------------------------------------------------------------
Candidate shape:
    hierarchical pair == statement / transaction
    A8A3 pair         == offer / transaction
    transaction_operation == search
    transaction_signal == none

Override to OFFER only if BOTH:
    A8A3 P(offer/transaction) >= 0.75
    support_delta > 0

where:
    support_delta =
      [A8A3 P(offer) - A8A3 P(statement)]
      -
      [HIER P(statement) - HIER P(offer)]

The thresholds are conservative and not midpoint-fit to the two observed cases.

Rule B — actionable availability offer arbitration
--------------------------------------------------
Same semantic Rule B that previously fixed:
    v8c_001
    v11c_007
    v11c_008
with zero regressions.

No lexical routing.
No case-id routing.
No gold routing.
Only speech_act may change.

Evaluation:
- established 1,146 development contract
- exposed raw120 traceability
- exposed ontology-coherent112
- every override printed
- constructor / integrity checks

No runtime/source/model writes.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent

FULL_GATE = HERE / "voiceprobe_semanticlab_v2_hierarchical_phase8a_full_level2_offline_gate.py"

if not FULL_GATE.is_file():
    raise SystemExit(f"Missing required file: {FULL_GATE}")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


g = load_mod("phase8a_semantic_composition_v2_base", FULL_GATE)

base = g.base
v2 = g.v2
hier = g.hier

from voiceprobe.v33.semantic_corpus import load_semanticlab_cases
from voiceprobe.v33.semantic_frame_eval import evaluate_frame


VALID = tuple(tuple(x) for x in g.VALID)
PAIR_TO_I = {p: i for i, p in enumerate(VALID)}
OFFER_TX_I = PAIR_TO_I[("offer", "transaction")]
STATEMENT_TX_I = PAIR_TO_I[("statement", "transaction")]

DECLARED_EXPOSED_CONFLICT = g.DECLARED_EXPOSED_CONFLICT
SPEC_REVIEW_IDS = set(g.SPEC_REVIEW_IDS)

RULE_A_A8A3_OFFER_MIN = 0.75
RULE_A_SUPPORT_DELTA_MIN = 0.0


def enum_value(x):
    return getattr(x, "value", x)


def nonempty_value(x):
    if x is None:
        return False
    v = enum_value(x)
    if isinstance(v, str):
        return v not in {"", "none"}
    try:
        return len(v) > 0
    except Exception:
        return bool(v)


def ambiguity_is_none(frame):
    try:
        return enum_value(frame.ambiguity.kind) == "none"
    except Exception:
        return True


def reference_is_none(frame):
    try:
        return enum_value(frame.reference) == "none"
    except Exception:
        return True


def selected_option_is_none(frame):
    try:
        return not nonempty_value(frame.selected_option)
    except Exception:
        return True


def convert_speech_act(frame, act_value):
    current = frame.speech_act
    try:
        replacement = type(current)(act_value)
    except Exception:
        replacement = act_value

    if dataclasses.is_dataclass(frame):
        return dataclasses.replace(frame, speech_act=replacement)

    clone = copy.copy(frame)
    clone.speech_act = replacement
    return clone


def frame_features(frame):
    return {
        "speech_act": enum_value(frame.speech_act),
        "topic": enum_value(frame.topic),
        "transaction_operation": enum_value(frame.transaction_operation),
        "transaction_signal": enum_value(frame.transaction_signal),
        "failed_constraints_nonempty": nonempty_value(frame.failed_constraints),
        "proposed_changes_nonempty": nonempty_value(frame.proposed_changes),
        "retained_constraints_nonempty": nonempty_value(frame.retained_constraints),
        "offered_options_nonempty": nonempty_value(frame.offered_options),
        "selected_option_nonempty": nonempty_value(frame.selected_option),
        "record_claims_nonempty": nonempty_value(frame.record_claims),
        "reference": enum_value(frame.reference),
        "ambiguity_kind": enum_value(frame.ambiguity.kind),
    }


class PairConfidencePredictor:
    """Read-only pair probabilities for frozen A8A3 and parent hierarchical expert."""

    def __init__(self):
        a8a3 = g.load_ck(base.A8A3)

        self.teacher = hier.p8a.DenseModel()
        self.teacher.load_state_dict(a8a3["state_dict"])
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.model = hier.HierarchicalFusion(
            hidden_size=int(self.teacher.encoder.config.dim),
            act_count=len(g.ACTS),
            topic_count=len(g.TOPICS),
            pair_count=len(VALID),
        )
        self.model.load_state_dict(g.CAND["hierarchical_state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.factor_scale = float(g.CAND["factor_scale"])
        self.interaction_scale = float(g.CAND["interaction_scale"])
        self.tok = base.tokenizer_for(hier.p8a.MODEL_NAME)

    def _encode(self, texts):
        rows = []
        with torch.no_grad():
            for start in range(0, len(texts), 64):
                z = self.tok(
                    texts[start:start+64],
                    padding=True,
                    truncation=True,
                    max_length=hier.p8a.MAX_LENGTH,
                    return_tensors="pt",
                )
                h = self.teacher.encoder(
                    input_ids=z["input_ids"],
                    attention_mask=z["attention_mask"],
                ).last_hidden_state[:, 0, :]
                rows.append(h.cpu())
        return torch.cat(rows, dim=0)

    def probabilities(self, runtime):
        combined_texts = []
        turn_texts = []
        context_texts = []

        for item in runtime:
            context = tuple(item.context)
            turn = str(item.utterance)
            ctx = " || ".join(context) if context else "<none>"

            combined_texts.append(hier.p8a.serialize(context, turn))
            turn_texts.append("Latest clinic utterance: " + turn)
            context_texts.append("Recent clinic context: " + ctx)

        combined_h = self._encode(combined_texts)
        turn_h = self._encode(turn_texts)
        context_h = self._encode(context_texts)

        with torch.no_grad():
            base_act = self.teacher.act_head(combined_h)
            base_topic = self.teacher.topic_head(combined_h)
            base_pair_logits = hier.p8a.pair_logits(
                base_act,
                base_topic,
                VALID,
            )

            ar, tr, pr = self.model(
                combined_h,
                turn_h,
                context_h,
                base_act,
                base_topic,
                base_pair_logits,
            )

            final_act = base_act + self.factor_scale * ar
            final_topic = base_topic + self.factor_scale * tr
            hier_pair_logits = (
                hier.p8a.pair_logits(final_act, final_topic, VALID)
                + self.interaction_scale * pr
            )

            base_probs = torch.softmax(base_pair_logits, dim=-1).cpu()
            hier_probs = torch.softmax(hier_pair_logits, dim=-1).cpu()

        return base_probs, hier_probs

    def close(self):
        del self.tok, self.model, self.teacher


def rule_a_confidence(base_probs_row, hier_probs_row):
    a3_offer = float(base_probs_row[OFFER_TX_I])
    a3_stmt = float(base_probs_row[STATEMENT_TX_I])
    h_offer = float(hier_probs_row[OFFER_TX_I])
    h_stmt = float(hier_probs_row[STATEMENT_TX_I])

    support_delta = (
        (a3_offer - a3_stmt)
        - (h_stmt - h_offer)
    )

    passed = (
        a3_offer >= RULE_A_A8A3_OFFER_MIN
        and support_delta > RULE_A_SUPPORT_DELTA_MIN
    )

    return {
        "a8a3_offer_p": a3_offer,
        "a8a3_statement_p": a3_stmt,
        "hier_offer_p": h_offer,
        "hier_statement_p": h_stmt,
        "support_delta": support_delta,
        "passed": passed,
    }


def arbitrate_speech_act(a8a3_pair, hier_pair, frame, base_probs_row, hier_probs_row):
    """Return (new_act_or_None, rule_or_None, audit_metadata)."""
    a3 = tuple(a8a3_pair)
    hp = tuple(hier_pair)

    txop = enum_value(frame.transaction_operation)
    txsig = enum_value(frame.transaction_signal)

    # Rule A: semantic shape + conservative model-confidence gate.
    if (
        hp == ("statement", "transaction")
        and a3 == ("offer", "transaction")
        and txop == "search"
        and txsig == "none"
    ):
        conf = rule_a_confidence(base_probs_row, hier_probs_row)
        if conf["passed"]:
            return "offer", "confidence_gated_search_operation_offer_arbitration", conf
        return None, None, {
            **conf,
            "blocked_rule": "confidence_gated_search_operation_offer_arbitration",
        }

    actionable_schedule = (
        nonempty_value(frame.proposed_changes)
        or nonempty_value(frame.offered_options)
    )

    # Rule B unchanged from prior probe.
    if (
        hp[1] == "availability"
        and hp[0] in {"statement", "question"}
        and a3 == ("offer", "availability")
        and actionable_schedule
        and selected_option_is_none(frame)
        and reference_is_none(frame)
        and ambiguity_is_none(frame)
    ):
        return "offer", "semantic_actionable_availability_offer_arbitration", {}

    return None, None, {}


def failure_fields(case, frame):
    return tuple(x.field for x in evaluate_frame(case, frame))


def load_groups():
    historical = list(load_semanticlab_cases())
    exposed108 = list(load_semanticlab_cases(g.v5.EXPOSED_FILE))
    adv120 = list(load_semanticlab_cases(g.ADV120_FILE))
    final132 = list(load_semanticlab_cases(g.FINAL132_FILE))
    coherent142 = list(load_semanticlab_cases(g.COHERENT142_FILE))
    coherent127 = list(load_semanticlab_cases(g.COHERENT127_FILE))
    v11fresh128 = list(load_semanticlab_cases(g.V11FRESH128_FILE))
    v12fresh128 = list(load_semanticlab_cases(g.V12FRESH128_FILE))
    v13fresh128 = list(load_semanticlab_cases(g.V13FRESH128_FILE))
    exposed120 = list(load_semanticlab_cases(g.EXPOSED120_FILE))

    return [
        ("HISTORICAL 133", historical),
        ("EXPOSED 108", exposed108),
        ("ADVERSARIAL 120", adv120),
        ("EXPOSED FINAL 132", final132),
        ("COHERENT FRESH 142", coherent142),
        ("COHERENT V10 FRESH 127", coherent127),
        ("EXPOSED V11 FRESH 128", v11fresh128),
        ("EXPOSED V12 FRESH 128", v12fresh128),
        ("EXPOSED V13 FRESH 128", v13fresh128),
        ("EXPOSED LEVEL2 FINAL 120", exposed120),
    ]


def main():
    print("========== PHASE 8A EXPLICIT SEMANTIC COMPOSITION V2 ==========")
    print("training=NO")
    print("speech_act_head_retraining=STOPPED")
    print("parent_hierarchical_candidate=READ_ONLY")
    print("a8a3=READ_ONLY")
    print("arbitration_uses_utterance_text=NO")
    print("arbitration_uses_case_id=NO")
    print("arbitration_uses_gold=NO")
    print("rule_a_a8a3_offer_min=", RULE_A_A8A3_OFFER_MIN)
    print("rule_a_support_delta_min=", RULE_A_SUPPORT_DELTA_MIN)
    print("runtime_wiring=NO")
    print("telephony=DISABLED")

    source_before = base.source_snapshot()
    a8a3_sha_before = g.sha256_file(base.A8A3)
    candidate_sha_before = g.sha256_file(g.HIER_ARTIFACT)

    checkpoints = base.validate_environment()

    groups = load_groups()
    all_cases = [c for _, cases in groups for c in cases]

    assert sum(len(cases) for _, cases in groups[:9]) == 1146
    assert len(groups[-1][1]) == 120

    runtime = [
        base.RuntimeTurn(
            context=tuple(c.context),
            utterance=str(c.utterance),
        )
        for c in all_cases
    ]

    print("combined_cases=", len(all_cases))
    print("established_cases=", 1146)
    print("exposed_raw_cases=", 120)

    # ---------------------------------------------------------------
    # Frozen A8A3 Level2 baseline.
    # ---------------------------------------------------------------
    print()
    print("========== FROZEN A8A3 BASELINE INFERENCE ==========")

    (
        baseline_result,
        baseline_frames,
        _baseline_diag,
        baseline_v2_errors,
        baseline_v132_errors,
    ) = g.assemble_v132(runtime, checkpoints)

    baseline_pairs = list(baseline_result.dense_pairs)
    print("a8a3_baseline_inference_complete=YES")

    # ---------------------------------------------------------------
    # Parent hierarchical full Level2.
    # ---------------------------------------------------------------
    print()
    print("========== PARENT HIERARCHICAL INFERENCE ==========")

    g.PREDICTOR = g.HierarchicalPhase8APredictor()
    original_run_phase8a_dense = base.run_phase8a_dense
    original_repair_dense_pairs = v2.repair_dense_pairs

    try:
        base.run_phase8a_dense = g.candidate_run_phase8a_dense
        v2.repair_dense_pairs = g.candidate_repair_dense_pairs

        (
            hier_result,
            hier_frames,
            _hier_diag,
            hier_v2_errors,
            hier_v132_errors,
        ) = g.assemble_v132(runtime, checkpoints)
    finally:
        base.run_phase8a_dense = original_run_phase8a_dense
        v2.repair_dense_pairs = original_repair_dense_pairs
        if g.PREDICTOR is not None:
            g.PREDICTOR.close()
        g.PREDICTOR = None

    hier_pairs = list(hier_result.dense_pairs)
    print("hierarchical_inference_complete=YES")
    print("offline_phase8a_hooks_restored=YES")

    # ---------------------------------------------------------------
    # Pair-confidence inference.
    # ---------------------------------------------------------------
    print()
    print("========== PAIR CONFIDENCE INFERENCE ==========")

    conf_pred = PairConfidencePredictor()
    base_probs, hier_probs = conf_pred.probabilities(runtime)
    conf_pred.close()

    print("pair_confidence_inference_complete=YES")

    # ---------------------------------------------------------------
    # Arbitration.
    # ---------------------------------------------------------------
    print()
    print("========== APPLYING CONFIDENCE-GATED SEMANTIC ARBITRATION ==========")

    composed_frames = []
    override_rows = []
    blocked_rule_a_rows = []
    rule_counts = Counter()

    for i, (case, a3_pair, hp, frame) in enumerate(
        zip(all_cases, baseline_pairs, hier_pairs, hier_frames)
    ):
        new_act, rule, meta = arbitrate_speech_act(
            a3_pair,
            hp,
            frame,
            base_probs[i],
            hier_probs[i],
        )

        if meta.get("blocked_rule"):
            blocked_rule_a_rows.append(
                {
                    "case_id": case.case_id,
                    "rule": meta["blocked_rule"],
                    "a8a3_pair": tuple(a3_pair),
                    "hier_pair": tuple(hp),
                    "features": frame_features(frame),
                    "confidence": {
                        k: v for k, v in meta.items()
                        if k != "blocked_rule"
                    },
                    "turn": case.utterance,
                }
            )

        if new_act is None or enum_value(frame.speech_act) == new_act:
            composed_frames.append(frame)
            continue

        before_fields = failure_fields(case, frame)
        new_frame = convert_speech_act(frame, new_act)
        after_fields = failure_fields(case, new_frame)

        rule_counts[rule] += 1

        override_rows.append(
            {
                "case_id": case.case_id,
                "rule": rule,
                "a8a3_pair": tuple(a3_pair),
                "hier_pair": tuple(hp),
                "before_act": enum_value(frame.speech_act),
                "after_act": new_act,
                "before_failures": before_fields,
                "after_failures": after_fields,
                "features": frame_features(frame),
                "confidence": meta,
                "turn": case.utterance,
            }
        )
        composed_frames.append(new_frame)

    print("override_count=", len(override_rows))
    print("blocked_rule_a_count=", len(blocked_rule_a_rows))
    print("rule_counts=", dict(rule_counts))

    for row in blocked_rule_a_rows:
        print(
            "BLOCKED_RULE_A",
            "case_id=", row["case_id"],
            "a8a3_pair=", row["a8a3_pair"],
            "hier_pair=", row["hier_pair"],
            "confidence=", row["confidence"],
            "features=", row["features"],
            "turn=", repr(row["turn"]),
        )

    for row in override_rows:
        print(
            "OVERRIDE",
            "case_id=", row["case_id"],
            "rule=", row["rule"],
            "a8a3_pair=", row["a8a3_pair"],
            "hier_pair=", row["hier_pair"],
            "before_act=", row["before_act"],
            "after_act=", row["after_act"],
            "before_failures=", row["before_failures"],
            "after_failures=", row["after_failures"],
            "confidence=", row["confidence"],
            "features=", row["features"],
            "turn=", repr(row["turn"]),
        )

    # ---------------------------------------------------------------
    # Established 1,146 contract.
    # ---------------------------------------------------------------
    print()
    print("========== ESTABLISHED 1146 CONTRACT ==========")

    pos = 0
    established_failure_maps = {}
    established_exacts = []

    for label, cases in groups[:9]:
        frames = composed_frames[pos:pos+len(cases)]
        pos += len(cases)

        failures = [
            evaluate_frame(c, f)
            for c, f in zip(cases, frames)
        ]

        fmap = {
            c.case_id: tuple(x.field for x in fs)
            for c, fs in zip(cases, failures)
            if fs
        }

        exact = sum(not fs for fs in failures)
        established_exacts.append(exact)
        established_failure_maps[label] = fmap

        print(
            label,
            "exact=",
            f"{exact}/{len(cases)}",
            "failure_fields=",
            fmap,
        )

    exposed_108_map = established_failure_maps["EXPOSED 108"]
    exposed_nonconflict = {
        cid: fields
        for cid, fields in exposed_108_map.items()
        if cid != DECLARED_EXPOSED_CONFLICT
    }
    conflict_fields = exposed_108_map.get(
        DECLARED_EXPOSED_CONFLICT,
        (),
    )

    established_pass = all(
        (
            established_exacts[0] == 133,
            established_exacts[1] == 107,
            established_exacts[2] == 120,
            established_exacts[3] == 132,
            established_exacts[4] == 142,
            established_exacts[5] == 127,
            established_exacts[6] == 128,
            established_exacts[7] == 128,
            established_exacts[8] == 128,
            not established_failure_maps["HISTORICAL 133"],
            not exposed_nonconflict,
            conflict_fields == ("transaction_signal",),
            not established_failure_maps["ADVERSARIAL 120"],
            not established_failure_maps["EXPOSED FINAL 132"],
            not established_failure_maps["COHERENT FRESH 142"],
            not established_failure_maps["COHERENT V10 FRESH 127"],
            not established_failure_maps["EXPOSED V11 FRESH 128"],
            not established_failure_maps["EXPOSED V12 FRESH 128"],
            not established_failure_maps["EXPOSED V13 FRESH 128"],
        )
    )

    # ---------------------------------------------------------------
    # Exposed raw120 / coherent112.
    # ---------------------------------------------------------------
    exposed120 = groups[-1][1]
    exposed_frames = composed_frames[1146:]

    raw_fail = [
        evaluate_frame(c, f)
        for c, f in zip(exposed120, exposed_frames)
    ]
    raw_exact = sum(not fs for fs in raw_fail)

    coherent_indices = []
    illegal_ids = []
    valid_set = set(VALID)

    for i, c in enumerate(exposed120):
        gp = hier.p8a.gold_pair(c)
        if gp not in valid_set:
            illegal_ids.append(c.case_id)
            continue
        if c.case_id in SPEC_REVIEW_IDS:
            continue
        coherent_indices.append(i)

    coherent_cases = [exposed120[i] for i in coherent_indices]
    coherent_frames = [exposed_frames[i] for i in coherent_indices]

    coherent_fail = [
        evaluate_frame(c, f)
        for c, f in zip(coherent_cases, coherent_frames)
    ]
    coherent_exact = sum(not fs for fs in coherent_fail)

    parent_coherent_frames = [
        hier_frames[1146+i]
        for i in coherent_indices
    ]
    parent_coherent_fail = [
        evaluate_frame(c, f)
        for c, f in zip(coherent_cases, parent_coherent_frames)
    ]
    parent_coherent_exact = sum(
        not fs for fs in parent_coherent_fail
    )

    coherent_regressions = []
    coherent_fixes = []

    for c, pfs, cfs in zip(
        coherent_cases,
        parent_coherent_fail,
        coherent_fail,
    ):
        if not pfs and cfs:
            coherent_regressions.append(
                (c.case_id, tuple(x.field for x in cfs))
            )
        elif pfs and not cfs:
            coherent_fixes.append(c.case_id)

    print()
    print("========== EXPOSED DIAGNOSTIC ==========")
    print("illegal_pair_case_ids=", illegal_ids)
    print("spec_review_case_ids=", sorted(SPEC_REVIEW_IDS))
    print("raw120_exact=", raw_exact, "/120")
    print(
        "parent_hierarchical_coherent112_exact=",
        parent_coherent_exact,
        "/112",
    )
    print(
        "composed_coherent112_exact=",
        coherent_exact,
        "/112",
    )
    print("coherent112_regressions=", coherent_regressions)
    print("coherent112_fixes=", coherent_fixes)

    # ---------------------------------------------------------------
    # Override quality / integrity.
    # ---------------------------------------------------------------
    override_regressions = [
        row["case_id"]
        for row in override_rows
        if not row["before_failures"]
        and row["after_failures"]
    ]
    override_fixes = [
        row["case_id"]
        for row in override_rows
        if row["before_failures"]
        and not row["after_failures"]
    ]
    override_neutral = [
        row["case_id"]
        for row in override_rows
        if bool(row["before_failures"]) == bool(row["after_failures"])
    ]

    print()
    print("========== OVERRIDE QUALITY ==========")
    print("override_fixes=", override_fixes)
    print("override_regressions=", override_regressions)
    print("override_neutral=", override_neutral)

    source_after = base.source_snapshot()
    a8a3_sha_after = g.sha256_file(base.A8A3)
    candidate_sha_after = g.sha256_file(g.HIER_ARTIFACT)

    baseline_constructor_errors = (
        sum(len(v) for v in baseline_v2_errors.values())
        + sum(len(v) for v in baseline_v132_errors.values())
    )
    hier_constructor_errors = (
        sum(len(v) for v in hier_v2_errors.values())
        + sum(len(v) for v in hier_v132_errors.values())
    )

    integrity_pass = all(
        (
            source_before == source_after,
            a8a3_sha_before == a8a3_sha_after,
            candidate_sha_before == candidate_sha_after,
        )
    )

    composition_pass = all(
        (
            established_pass,
            not coherent_regressions,
            not override_regressions,
            coherent_exact >= parent_coherent_exact,
            baseline_constructor_errors == 0,
            hier_constructor_errors == 0,
            integrity_pass,
        )
    )

    print()
    print("========== CONFIDENCE-GATED COMPOSITION DECISION ==========")
    print(
        "established_1146_contract=",
        "PASS" if established_pass else "FAIL",
    )
    print(
        "exposed_112_no_regressions=",
        "PASS" if not coherent_regressions else "FAIL",
    )
    print("override_regressions=", override_regressions)
    print("baseline_constructor_errors=", baseline_constructor_errors)
    print("hierarchical_constructor_errors=", hier_constructor_errors)
    print(
        "source_tree_python_unchanged=",
        "YES" if source_before == source_after else "NO",
    )
    print(
        "a8a3_artifact_unchanged=",
        "YES" if a8a3_sha_before == a8a3_sha_after else "NO",
    )
    print(
        "hierarchical_candidate_unchanged=",
        "YES" if candidate_sha_before == candidate_sha_after else "NO",
    )
    print("runtime_wiring_modified=NO")
    print("phase8b_retrained=NO")
    print("level2_frozen=NO")

    if composition_pass:
        print("PHASE8A_CONFIDENCE_GATED_SEMANTIC_COMPOSITION=PASS")
        print(
            "NEXT_ACTION=FREEZE_PHASE8A_ARCHITECTURE_AS_"
            "HIERARCHICAL_PLUS_CONFIDENCE_GATED_SEMANTIC_ARBITRATION_"
            "THEN_BEGIN_PHASE8B"
        )
        return 0

    print("PHASE8A_CONFIDENCE_GATED_SEMANTIC_COMPOSITION=FAIL")
    print(
        "NEXT_ACTION=INSPECT_ONLY_REMAINING_ARBITRATION_REGRESSIONS_"
        "BEFORE_ANY_PHASE8B_WORK"
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
