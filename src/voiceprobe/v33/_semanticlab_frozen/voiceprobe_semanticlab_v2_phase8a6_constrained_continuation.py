#!/usr/bin/env python3
"""SemanticLab v2 Phase 8A6 constrained continuation from stable interpolation.

Starts from the A8A3/A8A5 alpha=0.225 interpolation that already demonstrated:
- zero native A8A3 regressions
- zero prior-corpus pair regressions
- exposed legal improvement from 0.8125 -> 0.8571

A8A6 makes a SMALL supervised continuation targeted at the remaining semantic
DISTINCTIONS, not the exact exposed sentences.

Hard rules:
- frozen 20-pair ontology
- only encoder block 5 + heads trainable
- strong A8A3 legacy replay/distillation
- L2 anchor to the stable alpha=0.225 initialization
- model selection uses native133 + synthetic validation ONLY
- an epoch is ineligible if it loses ANY native case A8A3 got right
- exposed120 and prior corpora are evaluated only AFTER checkpoint selection
- current A8A3/A8A5 artifacts never overwritten
- no runtime/planner/telephony changes
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

ROOT = Path(".").resolve()
HERE = Path(__file__).resolve().parent

P8A = ROOT / "voiceprobe_semanticlab_v2_phase8a_speech_act_topic.py"
A8A3 = ROOT / "artifacts/semanticlab_v2_phase8a3_speech_act_topic.pt"
A8A5 = ROOT / "artifacts/candidates/semanticlab_v2_phase8a5_stability_preserving.pt"
EXPOSED120 = HERE / "semanticlab_v2_level2_final_unseen_holdout_120_v2_20260817.jsonl"
OUT = ROOT / "artifacts/candidates/semanticlab_v2_phase8a6_constrained.pt"

for p in (P8A, A8A3, A8A5, EXPOSED120):
    if not p.is_file():
        raise SystemExit(f"Missing required file: {p}")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p8a = load_mod("phase8a6_base", P8A)
from voiceprobe.v33.semantic_corpus import load_semanticlab_cases


SEED = 8606
random.seed(SEED)
torch.manual_seed(SEED)

ALPHA = 0.225
EPOCHS = 6
BATCH_SIZE = 24
TEMPERATURE = 2.0

ck3 = torch.load(A8A3, map_location="cpu")
ck5 = torch.load(A8A5, map_location="cpu")

VALID = tuple(tuple(x) for x in ck3["valid_pairs"])
if VALID != tuple(tuple(x) for x in ck5["valid_pairs"]):
    raise RuntimeError("A8A5 ontology differs from A8A3")
VALID_SET = set(VALID)
PAIR_TO_I = {pair: i for i, pair in enumerate(VALID)}
SPEC_REVIEW = {"f2a_001"}


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_state():
    out = {}
    for k, a in ck3["state_dict"].items():
        b = ck5["state_dict"][k]
        if a.is_floating_point():
            out[k] = (1.0 - ALPHA) * a + ALPHA * b
        else:
            out[k] = a.clone()
    return out


def ser_key(context, turn):
    return p8a.norm(p8a.serialize(tuple(context), str(turn)))


def add(out, family, contexts, turns, act, topic):
    if (act, topic) not in VALID_SET:
        raise ValueError(f"Illegal pair {(act, topic)}")
    p8a.add(out, family, contexts, turns, act, topic)


def build_hard():
    o = []

    # ================================================================
    # A. IDENTICAL RESPONSE, DIFFERENT CONTEXT SEMANTICS.
    # Explicit offer => confirmation availability.
    # Mere listing/enumeration => statement other.
    # ================================================================
    offer_contexts = [
        ("I can offer Monday morning or Friday evening.",),
        ("I can offer 8 AM or 1 PM.",),
        ("I can offer Dr. Rivera Tuesday or Dr. Shah Thursday.",),
        ("Would you like Wednesday afternoon or Saturday morning?",),
    ]
    listing_contexts = [
        ("The schedule lists Monday morning and Friday evening.",),
        ("The chart shows 8 AM and 1 PM.",),
        ("The record lists Dr. Rivera Tuesday and Dr. Shah Thursday.",),
        ("Two entries are shown: Wednesday afternoon and Saturday morning.",),
    ]
    same_responses = [
        "That option works.",
        "That choice is fine.",
        "That one sounds good.",
        "That selection works for me.",
    ]

    add(
        o, "a6_offer_context_confirmation",
        offer_contexts, same_responses,
        "confirmation", "availability",
    )
    add(
        o, "a6_listing_context_statement_other",
        listing_contexts, same_responses,
        "statement", "other",
    )

    add(
        o, "a6_offer_context_explicit_selection",
        offer_contexts,
        [
            "I'll take the first choice.",
            "I'll choose the second option.",
            "I'll go with whichever one is earlier.",
            "Use the later offered choice.",
        ],
        "confirmation", "availability",
    )

    # ================================================================
    # B. VAGUE TEMPORAL SHIFT => question other.
    # Concrete named search => offer availability.
    # ================================================================
    add(
        o, "a6_vague_temporal_question_other",
        [()],
        [
            "Could that appointment be shifted somewhat earlier?",
            "Can the visit be moved a little later somehow?",
            "Could we push the appointment earlier somehow?",
            "Can we move that somewhat later?",
            "Would it be possible to shift it earlier?",
            "Could that be pushed back a little?",
        ],
        "question", "other",
    )
    add(
        o, "a6_concrete_temporal_offer_availability",
        [()],
        [
            "Could I check an earlier time this evening?",
            "May I look for a later appointment this morning?",
            "Can I try Thursday instead?",
            "Could I check something later this afternoon?",
            "May I try another provider instead?",
        ],
        "offer", "availability",
    )

    # ================================================================
    # C. Passive completed transaction.
    # ================================================================
    add(
        o, "a6_completed_transaction_confirmation",
        [()],
        [
            "The visit was moved to Monday.",
            "The appointment was rescheduled to Friday.",
            "Your booking was moved to Thursday morning.",
            "The cancellation was completed.",
            "The new appointment has been confirmed.",
        ],
        "confirmation", "transaction",
    )

    # ================================================================
    # D. Statement transaction vs offer transaction.
    # "I can book/move..." is statement transaction in frozen ontology.
    # ================================================================
    add(
        o, "a6_statement_transaction_capability",
        [()],
        [
            "I can book Dr. Rivera on Tuesday.",
            "I can move the appointment to Saturday morning.",
            "I can cancel the old booking if needed.",
            "I can place you into the Friday opening.",
            "I can search Dr. Shah's Thursday schedule.",
        ],
        "statement", "transaction",
    )

    # ================================================================
    # E. Explicit "for you" check => question transaction.
    # ================================================================
    add(
        o, "a6_question_transaction_for_you",
        [()],
        [
            "Can I check Friday at eleven for you?",
            "May I check Tuesday afternoon for you?",
            "Should I search Dr. Rivera's schedule for you?",
            "Could I check another opening for you?",
            "Would you like me to check Thursday morning?",
        ],
        "question", "transaction",
    )

    # ================================================================
    # F. Grounded option availability questions.
    # ================================================================
    add(
        o, "a6_question_availability_option",
        offer_contexts,
        [
            "Could the first option still work?",
            "Would the second choice still be available?",
            "Can the later option still work?",
            "Could the earlier choice still be open?",
        ],
        "question", "availability",
    )

    # ================================================================
    # G. Genuine OOS and vague transaction-reference questions.
    # ================================================================
    add(
        o, "a6_question_other_oos",
        [()],
        [
            "Can you recommend a movie for this evening?",
            "What television show should I watch tonight?",
            "Can you suggest a podcast for the weekend?",
            "Who won the baseball game yesterday?",
        ],
        "question", "other",
    )
    add(
        o, "a6_question_other_unspecified_transaction",
        [()],
        [
            "Should I go ahead with that unspecified change?",
            "Do you want me to proceed with that?",
            "Should I continue with whatever change we discussed?",
            "Would you like me to act on that step?",
        ],
        "question", "other",
    )

    # ================================================================
    # H. Legacy vulnerable controls with fresh surfaces.
    # ================================================================
    add(
        o, "a6_legacy_offer_availability_controls",
        [()],
        [
            "How about late-day appointments?",
            "Could next month work instead?",
            "Nothing is open then; can I check a later time?",
            "Dr. Rivera is booked; can I check another clinician?",
            "Friday is full, but I can check Tuesday if you'd like.",
            "Dr. Rivera is booked, but I can try Dr. Shah.",
            "I have Friday at around four PM.",
        ],
        "offer", "availability",
    )
    add(
        o, "a6_legacy_request_patient_fact_controls",
        [()],
        [
            "Can I get your full name please?",
            "Could you give me your last name please?",
            "May I have your complete name?",
        ],
        "request", "patient_fact",
    )
    add(
        o, "a6_legacy_question_transaction_controls",
        [()],
        [
            "Okay, can I go ahead and reschedule that appointment?",
            "Can I cancel that booking now?",
            "Should I book that opening for you?",
        ],
        "question", "transaction",
    )

    # ================================================================
    # Validation-only paired contrasts. Distinct surfaces.
    # ================================================================
    val_offer_ctx = [
        ("I can offer Tuesday morning or Sunday evening.",),
        ("I can offer 10 AM or 3 PM.",),
    ]
    val_list_ctx = [
        ("The schedule lists Tuesday morning and Sunday evening.",),
        ("The system shows 10 AM and 3 PM.",),
    ]

    add(
        o, "val_a6_offer_context",
        val_offer_ctx,
        ["That option sounds good.", "I'll use the second choice."],
        "confirmation", "availability",
    )
    add(
        o, "val_a6_listing_context",
        val_list_ctx,
        ["That option sounds good.", "That choice seems fine."],
        "statement", "other",
    )
    add(
        o, "val_a6_vague_temporal",
        [()],
        [
            "Could the visit be shifted a little earlier somehow?",
            "Can that appointment be pushed somewhat later?",
        ],
        "question", "other",
    )
    add(
        o, "val_a6_offer_availability",
        [()],
        [
            "Could I check something earlier tomorrow afternoon?",
            "May I try another doctor instead?",
        ],
        "offer", "availability",
    )
    add(
        o, "val_a6_completed_transaction",
        [()],
        [
            "The appointment was moved to Wednesday.",
        ],
        "confirmation", "transaction",
    )
    add(
        o, "val_a6_statement_transaction",
        [()],
        [
            "I can book Dr. Shah on Friday.",
        ],
        "statement", "transaction",
    )
    add(
        o, "val_a6_question_transaction",
        [()],
        [
            "Could I check Monday at two for you?",
        ],
        "question", "transaction",
    )
    add(
        o, "val_a6_question_availability",
        val_offer_ctx,
        [
            "Would the later option still work?",
        ],
        "question", "availability",
    )
    add(
        o, "val_a6_question_other_oos",
        [()],
        [
            "Can you suggest a film for tonight?",
        ],
        "question", "other",
    )
    add(
        o, "val_a6_question_other_reference",
        [()],
        [
            "Should I proceed with that unspecified change?",
        ],
        "question", "other",
    )

    dedup = []
    seen = set()
    for ex in o:
        k = (ser_key(ex.context, ex.turn), ex.speech_act, ex.topic)
        if k in seen:
            continue
        seen.add(k)
        dedup.append(ex)
    return dedup


VAL_FAMILIES = {
    "val_a6_offer_context",
    "val_a6_listing_context",
    "val_a6_vague_temporal",
    "val_a6_offer_availability",
    "val_a6_completed_transaction",
    "val_a6_statement_transaction",
    "val_a6_question_transaction",
    "val_a6_question_availability",
    "val_a6_question_other_oos",
    "val_a6_question_other_reference",
}


def blocked_surfaces():
    blocked = set()
    for c in load_semanticlab_cases():
        blocked.add(ser_key(c.context, c.utterance))

    names = []
    for p in sorted(HERE.glob("semanticlab_v2_*.jsonl")):
        try:
            rows = [
                json.loads(line)
                for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except Exception:
            continue
        names.append(p.name)
        for row in rows:
            blocked.add(ser_key(row.get("context", ()), row.get("utterance", "")))
    return blocked, names


def legal_subset(cases):
    return [c for c in cases if p8a.gold_pair(c) in VALID_SET]


def load_prior_groups():
    names = (
        "semanticlab_v2_fresh_holdout_20260817.jsonl",
        "semanticlab_v2_v6_adversarial_generality_cases.jsonl",
        "semanticlab_v2_final_unseen_holdout_v2_20260817.jsonl",
        "semanticlab_v2_v9_coherent_fresh_adversarial_142.jsonl",
        "semanticlab_v2_v11_coherent_fresh_adversarial_127.jsonl",
        "semanticlab_v2_v11_fresh_adversarial_generality_128_v2.jsonl",
        "semanticlab_v2_v12_fresh_adversarial_generality_128_v2.jsonl",
        "semanticlab_v2_v13_fresh_adversarial_generality_128_v2.jsonl",
    )
    out = []
    for name in names:
        p = HERE / name
        if p.is_file():
            out.append((name, legal_subset(list(load_semanticlab_cases(p)))))
    return out


def evalm(model, tok, items):
    return p8a.eval_examples(model, tok, items, VALID)


def correct_ids(items, preds):
    return {
        item.case_id
        for item, pred in zip(items, preds)
        if tuple(pred) == p8a.gold_pair(item)
    }


def kl(student_logits, teacher_logits):
    t = TEMPERATURE
    return nn.functional.kl_div(
        nn.functional.log_softmax(student_logits / t, dim=-1),
        nn.functional.softmax(teacher_logits / t, dim=-1),
        reduction="batchmean",
    ) * t * t


def supervised(model, batch, act_weights, topic_weights):
    a, t = model(batch["input_ids"], batch["attention_mask"])
    la = nn.functional.cross_entropy(
        a, batch["act"], weight=act_weights, label_smoothing=0.003
    )
    lt = nn.functional.cross_entropy(
        t, batch["topic"], weight=topic_weights, label_smoothing=0.003
    )
    lp = nn.functional.cross_entropy(
        p8a.pair_logits(a, t, VALID), batch["pair"]
    )
    return a, t, la + lt + 1.15 * lp


def distill(student, teacher, batch):
    sa, st = student(batch["input_ids"], batch["attention_mask"])
    with torch.no_grad():
        ta, tt = teacher(batch["input_ids"], batch["attention_mask"])
    sp = p8a.pair_logits(sa, st, VALID)
    tp = p8a.pair_logits(ta, tt, VALID)
    return 0.4 * kl(sa, ta) + 0.4 * kl(st, tt) + 1.0 * kl(sp, tp)


def main():
    print("========== PHASE 8A6 CONSTRAINED CONTINUATION ==========")
    print("telephony=DISABLED")
    print("runtime_wiring_modified=NO")
    print("current_a8a3_overwritten=NO")
    print("current_a8a5_overwritten=NO")
    print("initialization=A8A3_A8A5_INTERPOLATION_ALPHA_0.225")
    print("trainable_encoder_layers=5")
    print("native_zero_regression_epoch_constraint=YES")
    print("exposed120_used_for_gradients=NO")
    print("exposed120_used_for_model_selection=NO")
    print("prior_corpora_used_for_model_selection=NO")

    sha3_before = sha256_file(A8A3)
    sha5_before = sha256_file(A8A5)

    stable = stable_state()

    tok = AutoTokenizer.from_pretrained(p8a.MODEL_NAME)

    teacher = p8a.DenseModel()
    teacher.load_state_dict(ck3["state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = p8a.DenseModel()
    student.load_state_dict(stable)

    for li, layer in enumerate(student.encoder.transformer.layer):
        for p in layer.parameters():
            p.requires_grad = (li == 5)

    # Store trainable initialization for explicit drift penalty.
    anchor = {
        name: p.detach().clone()
        for name, p in student.named_parameters()
        if p.requires_grad
    }

    blocked, blocked_files = blocked_surfaces()

    original = p8a.build_examples()
    legacy_train = [
        ex for ex in original
        if ex.family not in p8a.VALIDATION_FAMILIES
        and ser_key(ex.context, ex.turn) not in blocked
    ]
    legacy_val = [
        ex for ex in original
        if ex.family in p8a.VALIDATION_FAMILIES
        and ser_key(ex.context, ex.turn) not in blocked
    ]

    authored = build_hard()
    hard_train = [
        ex for ex in authored
        if ex.family not in VAL_FAMILIES
        and ser_key(ex.context, ex.turn) not in blocked
    ]
    hard_val = [
        ex for ex in authored
        if ex.family in VAL_FAMILIES
        and ser_key(ex.context, ex.turn) not in blocked
    ]

    print("blocked_jsonl_files=", len(blocked_files))
    print("legacy_train_examples=", len(legacy_train))
    print("legacy_val_examples=", len(legacy_val))
    print("hard_train_examples=", len(hard_train))
    print("hard_val_examples=", len(hard_val))
    print(
        "post_filter_exact_benchmark_overlap=",
        sum(
            ser_key(ex.context, ex.turn) in blocked
            for ex in legacy_train + legacy_val + hard_train + hard_val
        ),
    )

    all_for_weights = legacy_train + hard_train
    act_weights, _ = p8a.class_weights(all_for_weights, "act")
    topic_weights, _ = p8a.class_weights(all_for_weights, "topic")

    # Strong replay weighting around historically vulnerable semantic pairs.
    replay_weights = []
    for ex in legacy_train:
        pair = (ex.speech_act, ex.topic)
        if pair == ("offer", "availability"):
            w = 5.0
        elif pair == ("confirmation", "availability"):
            w = 4.0
        elif pair == ("request", "patient_fact"):
            w = 4.0
        elif pair == ("question", "transaction"):
            w = 4.0
        elif pair == ("statement", "other"):
            w = 3.0
        else:
            w = 1.0
        replay_weights.append(w)

    sampler = WeightedRandomSampler(
        replay_weights,
        num_samples=max(480, len(legacy_train)),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    legacy_loader = DataLoader(
        p8a.DenseDataset(legacy_train, tok, PAIR_TO_I),
        batch_size=BATCH_SIZE,
        sampler=sampler,
    )
    hard_loader = DataLoader(
        p8a.DenseDataset(hard_train, tok, PAIR_TO_I),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in student.encoder.parameters() if p.requires_grad],
                "lr": 1.25e-6,
                "weight_decay": 0.01,
            },
            {
                "params": list(student.act_head.parameters())
                + list(student.topic_head.parameters()),
                "lr": 3.5e-5,
                "weight_decay": 1e-4,
            },
        ]
    )

    native = list(load_semanticlab_cases())
    baseline_native = evalm(teacher, tok, native)
    baseline_native_correct = correct_ids(native, baseline_native["pair_preds"])

    stable_model = p8a.DenseModel()
    stable_model.load_state_dict(stable)
    stable_model.eval()
    stable_native = evalm(stable_model, tok, native)

    print()
    print("========== BASELINES ==========")
    print("a8a3_native_pair=", round(baseline_native["pair_accuracy"], 4))
    print("alpha0225_native_pair=", round(stable_native["pair_accuracy"], 4))

    best_state = None
    best_epoch = None
    best_score = None
    best_native_regressions = None

    print()
    print("========== A8A6 TRAINING ==========")
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        student.train()
        replay_iter = itertools.cycle(legacy_loader)
        hard_iter = itertools.cycle(hard_loader)
        steps = max(len(legacy_loader), len(hard_loader))
        running = 0.0

        for _ in range(steps):
            rb = next(replay_iter)
            hb = next(hard_iter)
            optimizer.zero_grad()

            _, _, hard_loss = supervised(student, hb, act_weights, topic_weights)
            _, _, replay_loss = supervised(student, rb, act_weights, topic_weights)
            kd = distill(student, teacher, rb)

            drift = torch.zeros((), dtype=torch.float32)
            for name, p in student.named_parameters():
                if p.requires_grad:
                    drift = drift + (p - anchor[name]).pow(2).mean()

            loss = hard_loss + 0.95 * replay_loss + 1.65 * kd + 0.12 * drift
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 0.6)
            optimizer.step()
            running += float(loss.item())

        legacy_m = evalm(student, tok, legacy_val)
        hard_m = evalm(student, tok, hard_val)
        native_m = evalm(student, tok, native)
        native_correct = correct_ids(native, native_m["pair_preds"])
        native_regressions = sorted(baseline_native_correct - native_correct)

        eligible = (
            not native_regressions
            and native_m["pair_accuracy"] + 1e-12 >= baseline_native["pair_accuracy"]
            and native_m["act_accuracy"] + 1e-12 >= baseline_native["act_accuracy"]
            and native_m["topic_accuracy"] + 1e-12 >= baseline_native["topic_accuracy"]
            and legacy_m["pair_accuracy"] >= 0.90
        )

        print(
            f"epoch={epoch:02d}",
            f"loss={running/steps:.4f}",
            f"native_pair={native_m['pair_accuracy']:.4f}",
            f"native_regressions={len(native_regressions)}",
            f"legacy_val={legacy_m['pair_accuracy']:.4f}",
            f"hard_val={hard_m['pair_accuracy']:.4f}",
            f"eligible={'YES' if eligible else 'NO'}",
        )

        if eligible:
            score = (
                hard_m["pair_accuracy"],
                legacy_m["pair_accuracy"],
                native_m["pair_accuracy"],
                hard_m["act_accuracy"],
                hard_m["topic_accuracy"],
            )
            if best_score is None or score > best_score:
                best_score = score
                best_epoch = epoch
                best_native_regressions = native_regressions
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in student.state_dict().items()
                }

    print("training_wall_s=", round(time.perf_counter() - started, 3))

    if best_state is None:
        print()
        print("PHASE8A6_CONSTRAINED_CANDIDATE=NO_ELIGIBLE_EPOCH")
        print("candidate_artifact_written=NO")
        print("NEXT_ACTION=STOP_PHASE8A_CONTINUATION_AND_RECONSIDER_SPECIALIST_ARCHITECTURE")
        return 3

    print("best_epoch=", best_epoch)

    candidate = p8a.DenseModel()
    candidate.load_state_dict(best_state)
    candidate.eval()

    cand_native = evalm(candidate, tok, native)
    cand_native_correct = correct_ids(native, cand_native["pair_preds"])
    native_regressions = sorted(baseline_native_correct - cand_native_correct)

    print()
    print("========== SELECTED A8A6 — SELECTION DATA ==========")
    print("native_pair=", round(cand_native["pair_accuracy"], 4))
    print("native_act=", round(cand_native["act_accuracy"], 4))
    print("native_topic=", round(cand_native["topic_accuracy"], 4))
    print("native_regressions=", native_regressions)
    print("legacy_val_pair=", round(best_score[1], 4))
    print("hard_val_pair=", round(best_score[0], 4))

    # --------------------------------------------------------------
    # POST-selection diagnostics only.
    # --------------------------------------------------------------
    exposed_all = list(load_semanticlab_cases(EXPOSED120))
    exposed = [
        c for c in legal_subset(exposed_all)
        if c.case_id not in SPEC_REVIEW
    ]
    base_exp = evalm(teacher, tok, exposed)
    cand_exp = evalm(candidate, tok, exposed)

    base_exp_correct = correct_ids(exposed, base_exp["pair_preds"])
    cand_exp_correct = correct_ids(exposed, cand_exp["pair_preds"])
    exp_regressions = sorted(base_exp_correct - cand_exp_correct)
    exp_fixes = sorted(cand_exp_correct - base_exp_correct)

    print()
    print("========== POST-SELECTION EXPOSED DIAGNOSTIC ==========")
    print("cases=", len(exposed))
    print("a8a3_pair=", round(base_exp["pair_accuracy"], 4))
    print("a8a6_pair=", round(cand_exp["pair_accuracy"], 4))
    print("absolute_gain=", round(cand_exp["pair_accuracy"] - base_exp["pair_accuracy"], 4))
    print("regressions_vs_a8a3=", exp_regressions)
    print("fixes_vs_a8a3=", exp_fixes)

    residuals = []
    for c, pred in zip(exposed, cand_exp["pair_preds"]):
        if tuple(pred) != p8a.gold_pair(c):
            residuals.append(c.case_id)
            print(
                "RESIDUAL", c.case_id,
                "gold=", p8a.gold_pair(c),
                "pred=", tuple(pred),
                "turn=", repr(c.utterance),
                "context=", list(c.context),
            )

    prior_regressions = []
    print()
    print("========== POST-SELECTION PRIOR CORPORA ==========")
    for name, cases in load_prior_groups():
        bm = evalm(teacher, tok, cases)
        cm = evalm(candidate, tok, cases)
        print(
            name,
            "baseline=", round(bm["pair_accuracy"], 4),
            "candidate=", round(cm["pair_accuracy"], 4),
            "cases=", len(cases),
        )
        if cm["pair_accuracy"] + 1e-12 < bm["pair_accuracy"]:
            prior_regressions.append(
                (name, bm["pair_accuracy"], cm["pair_accuracy"])
            )

    promising = all(
        (
            not native_regressions,
            not prior_regressions,
            cand_exp["pair_accuracy"] >= 0.90,
            cand_exp["pair_accuracy"] >= base_exp["pair_accuracy"] + 0.07,
            best_score[0] >= 0.90,
            best_score[1] >= 0.90,
        )
    )

    if promising:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "model_name": p8a.MODEL_NAME,
                "speech_acts": tuple(p8a.SPEECH_ACTS),
                "topics": tuple(p8a.TOPICS),
                "valid_pairs": VALID,
                "seed": SEED,
                "initialization": "a8a3_a8a5_alpha_0.225",
                "source_a8a3_sha256": sha3_before,
                "source_a8a5_sha256": sha5_before,
                "best_epoch": best_epoch,
                "native133_pair_accuracy": cand_native["pair_accuracy"],
                "legacy_validation_pair_accuracy": best_score[1],
                "hard_validation_pair_accuracy": best_score[0],
                "exposed_legal_no_spec_pair_accuracy": cand_exp["pair_accuracy"],
                "candidate_status": "PROMISING",
            },
            OUT,
        )
        candidate_sha = sha256_file(OUT)
    else:
        if OUT.exists():
            OUT.unlink()
        candidate_sha = ""

    print()
    print("========== PHASE 8A6 DECISION ==========")
    print("native_regressions=", native_regressions)
    print("prior_pair_regressions=", prior_regressions)
    print("remaining_exposed_failure_ids=", residuals)
    print("a8a3_unchanged=", "YES" if sha256_file(A8A3) == sha3_before else "NO")
    print("a8a5_unchanged=", "YES" if sha256_file(A8A5) == sha5_before else "NO")
    print("runtime_wiring_modified=NO")
    print("phase8b_retrained=NO")
    print("phase7c_retrained=NO")
    print("phase6b_retrained=NO")

    if promising:
        print("candidate_artifact=", OUT)
        print("candidate_sha256=", candidate_sha)
        print("PHASE8A6_CONSTRAINED_CANDIDATE=PROMISING")
        print("NEXT_ACTION=OFFLINE_WIRE_A8A6_INTO_FULL_LEVEL2_EVALUATOR_BEFORE_PHASE8B")
        return 0

    print("candidate_artifact_written=NO")
    print("PHASE8A6_CONSTRAINED_CANDIDATE=NOT_GOOD_ENOUGH")
    print("NEXT_ACTION=STOP_PHASE8A_ITERATION_AND_RECONSIDER_SPECIALIST_ARCHITECTURE")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
