#!/usr/bin/env python3
"""SemanticLab v2 Phase 8A5 stability-preserving natural-generalization continuation.

Goal
----
Keep A8A3's strong legacy behavior while retaining the real natural-language
gains demonstrated by rejected A8A4.

Method
------
- initialize student from frozen A8A3
- keep the exact A8A3 legal-pair ontology
- freeze encoder blocks 0-4; train ONLY block 5 + act/topic heads
- lower continuation learning rates substantially vs A8A4
- use two streams:
    1) legacy replay, with extra sampling weight on pairs A8A4 forgot
    2) new natural-language hard contrasts derived from FAILURE TYPES only
- add teacher/student distillation on legacy replay using frozen A8A3 logits:
    * speech-act KL
    * topic KL
    * legal-pair KL
- exposed 120 is diagnostic only:
    * never used for gradients
    * never used for model selection
    * evaluated only after the candidate is chosen
- no current artifact overwrite
- no runtime/planner/telephony changes
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
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
EXPOSED120 = HERE / "semanticlab_v2_level2_final_unseen_holdout_120_v2_20260817.jsonl"
OUT = ROOT / "artifacts/candidates/semanticlab_v2_phase8a5_stability_preserving.pt"

for p in (P8A, A8A3, EXPOSED120):
    if not p.is_file():
        raise SystemExit(f"Missing required file: {p}")


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p8a = load_mod("phase8a5_base", P8A)

from voiceprobe.v33.semantic_corpus import load_semanticlab_cases


SEED = 8505
random.seed(SEED)
torch.manual_seed(SEED)

EPOCHS = 8
BATCH_SIZE = 16
TEMPERATURE = 2.0

CURRENT = torch.load(A8A3, map_location="cpu")
VALID_PAIRS = tuple(tuple(x) for x in CURRENT["valid_pairs"])
VALID_SET = set(VALID_PAIRS)
PAIR_TO_I = {pair: i for i, pair in enumerate(VALID_PAIRS)}

SPEC_REVIEW_IDS = {"f2a_001"}

# A8A4 audit: these pairs disproportionately regressed.
LEGACY_PAIR_SAMPLE_WEIGHT = {
    ("offer", "availability"): 4.0,
    ("request", "patient_fact"): 3.0,
    ("question", "transaction"): 3.0,
    ("confirmation", "availability"): 3.0,
    ("statement", "other"): 2.5,
}
DEFAULT_REPLAY_WEIGHT = 1.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ser_key(context, turn):
    return p8a.norm(p8a.serialize(tuple(context), str(turn)))


def add(out, family, contexts, turns, act, topic):
    if (act, topic) not in VALID_SET:
        raise ValueError(f"Illegal pair {(act, topic)}")
    p8a.add(out, family, contexts, turns, act, topic)


def build_a8a5_hard_examples():
    out = []

    # ------------------------------------------------------------------
    # 1) Explicit prior OFFER => confirmation availability.
    # A8A4 forgot these while learning non-offer option ambiguity.
    # ------------------------------------------------------------------
    add(
        out,
        "a5_confirmation_after_explicit_offer",
        [
            ("I can offer Monday morning or Saturday evening.",),
            ("I can offer 9 AM or 2 PM.",),
            ("I can offer Dr. Rivera Tuesday or Dr. Shah Friday.",),
            ("Would you like Wednesday morning or Sunday afternoon?",),
        ],
        [
            "I'll take the first one.",
            "I'll choose the second choice.",
            "That option sounds good to me.",
            "The earlier choice works for me.",
            "I'll go with whichever one is later.",
            "Yes, use that option.",
        ],
        "confirmation",
        "availability",
    )

    # ------------------------------------------------------------------
    # 2) NON-offer enumeration => statement other, not confirmation.
    # This is the exact residual TYPE A8A4 still missed.
    # ------------------------------------------------------------------
    add(
        out,
        "a5_statement_other_after_nonoffer_enumeration",
        [
            ("The schedule lists Monday morning and Saturday evening.",),
            ("The chart shows 9 AM and 2 PM.",),
            ("The notes contain Dr. Rivera Tuesday and Dr. Shah Friday.",),
            ("Two entries are shown: Wednesday and Sunday.",),
        ],
        [
            "That option seems fine.",
            "That choice sounds okay.",
            "That one would work.",
            "I'm fine with that option.",
            "That selection seems acceptable.",
        ],
        "statement",
        "other",
    )

    # ------------------------------------------------------------------
    # 3) offer availability: polite "can/may I try/check" is a proposal,
    # not a question or transaction, unless it explicitly says "for you".
    # ------------------------------------------------------------------
    add(
        out,
        "a5_offer_availability_try_provider",
        [()],
        [
            "May I try Dr. Rivera instead?",
            "Could I try another clinician instead?",
            "Can I try Dr. Shah instead?",
            "Could I try a different provider?",
            "May I try someone else?",
        ],
        "offer",
        "availability",
    )
    add(
        out,
        "a5_offer_availability_fallback",
        [()],
        [
            "Nothing is open then; can I check later appointments?",
            "The afternoon is booked; could I check the morning?",
            "Dr. Rivera is full; can I check another provider?",
            "Friday is unavailable; could I look at Monday instead?",
            "That time is taken; may I try something later?",
        ],
        "offer",
        "availability",
    )
    add(
        out,
        "a5_offer_availability_relative",
        [()],
        [
            "Could I check something a bit earlier this afternoon?",
            "Can I look a little later this morning?",
            "May I check an earlier part of the evening?",
            "Could I try a later calendar date instead?",
        ],
        "offer",
        "availability",
    )

    # ------------------------------------------------------------------
    # 4) explicit transaction permission/search, to protect question×transaction.
    # ------------------------------------------------------------------
    add(
        out,
        "a5_question_transaction_permission",
        [()],
        [
            "Can I go ahead and move the appointment now?",
            "Would you like me to reschedule the current visit?",
            "Should I cancel the existing appointment for you?",
            "May I book that opening for you?",
            "Can I check Tuesday at noon for you?",
            "Should I search Dr. Rivera's Thursday schedule for you?",
        ],
        "question",
        "transaction",
    )

    # ------------------------------------------------------------------
    # 5) request patient_fact: protect polite request forms.
    # ------------------------------------------------------------------
    add(
        out,
        "a5_request_patient_fact_polite",
        [()],
        [
            "Can I get your complete name, please?",
            "Could you give me your date of birth, please?",
            "May I have your insurance information?",
            "Please give me your first and last name.",
            "Can I get your surname, please?",
        ],
        "request",
        "patient_fact",
    )

    # ------------------------------------------------------------------
    # 6) natural question×other contrasts learned well by A8A4.
    # ------------------------------------------------------------------
    add(
        out,
        "a5_question_other_vague_change",
        [()],
        [
            "Do you want me to proceed with that unspecified step?",
            "Should I go ahead with whatever change we discussed?",
            "Could we move it somewhat later?",
            "Can that be shifted earlier somehow?",
            "Would changing it to something later be possible?",
        ],
        "question",
        "other",
    )
    add(
        out,
        "a5_question_other_oos",
        [()],
        [
            "Can you recommend a television show for tonight?",
            "What film should I watch this evening?",
            "Who won the basketball game yesterday?",
            "Can you suggest a podcast for the weekend?",
        ],
        "question",
        "other",
    )

    # ------------------------------------------------------------------
    # 7) grounded option questions: question availability.
    # ------------------------------------------------------------------
    add(
        out,
        "a5_question_availability_grounded_option",
        [
            ("I can offer Monday morning or Saturday evening.",),
            ("I can offer 9 AM or 2 PM.",),
            ("I can offer Dr. Rivera Tuesday or Dr. Shah Friday.",),
        ],
        [
            "Could the first choice still work?",
            "Would the second option still be open?",
            "Can the earlier choice still be available?",
            "Could the later option work?",
        ],
        "question",
        "availability",
    )

    # ------------------------------------------------------------------
    # 8) transaction statement/completion natural variants learned by A8A4.
    # ------------------------------------------------------------------
    add(
        out,
        "a5_statement_transaction_search",
        [()],
        [
            "I'm checking whether Dr. Rivera has Tuesday availability.",
            "I am checking the schedule for Friday at noon.",
            "I'm searching the calendar for another opening.",
        ],
        "statement",
        "transaction",
    )
    add(
        out,
        "a5_statement_transaction_action",
        [()],
        [
            "I can book Dr. Rivera on Tuesday.",
            "I can move the visit to Saturday morning.",
            "I can cancel the current booking if needed.",
        ],
        "statement",
        "transaction",
    )
    add(
        out,
        "a5_confirmation_transaction_completed",
        [()],
        [
            "The visit has been moved to Monday.",
            "Your appointment was rescheduled to Thursday morning.",
            "The booking has now been confirmed for Saturday.",
        ],
        "confirmation",
        "transaction",
    )

    # ------------------------------------------------------------------
    # Validation-only: independent wording.
    # ------------------------------------------------------------------
    add(
        out,
        "val_a5_confirmation_offer_context",
        [
            ("I can offer Tuesday morning or Friday evening.",),
        ],
        [
            "I'll go with the earlier selection.",
            "That offered choice sounds good.",
        ],
        "confirmation",
        "availability",
    )
    add(
        out,
        "val_a5_statement_nonoffer_context",
        [
            ("The system lists Tuesday morning and Friday evening.",),
        ],
        [
            "That listed option seems fine.",
            "That listed choice would work.",
        ],
        "statement",
        "other",
    )
    add(
        out,
        "val_a5_offer_availability",
        [()],
        [
            "Could I try another physician instead?",
            "The evening is full; may I look at the morning?",
        ],
        "offer",
        "availability",
    )
    add(
        out,
        "val_a5_request_patient_fact",
        [()],
        [
            "Could you give me your insurance details, please?",
        ],
        "request",
        "patient_fact",
    )
    add(
        out,
        "val_a5_question_transaction",
        [()],
        [
            "May I check Dr. Shah's Friday schedule for you?",
        ],
        "question",
        "transaction",
    )
    add(
        out,
        "val_a5_question_other",
        [()],
        [
            "Should I carry out that unspecified change?",
            "Can you suggest a movie for this evening?",
        ],
        "question",
        "other",
    )
    add(
        out,
        "val_a5_question_availability",
        [
            ("I can offer Wednesday morning or Sunday afternoon.",),
        ],
        [
            "Would the later option still be open?",
        ],
        "question",
        "availability",
    )
    add(
        out,
        "val_a5_statement_transaction",
        [()],
        [
            "I am checking whether Wednesday afternoon is available.",
        ],
        "statement",
        "transaction",
    )
    add(
        out,
        "val_a5_confirmation_transaction",
        [()],
        [
            "The appointment has been moved to Friday.",
        ],
        "confirmation",
        "transaction",
    )

    # Dedup exact serialized+pair.
    dedup = []
    seen = set()
    for ex in out:
        key = (ser_key(ex.context, ex.turn), ex.speech_act, ex.topic)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(ex)
    return dedup


VAL_A5 = {
    "val_a5_confirmation_offer_context",
    "val_a5_statement_nonoffer_context",
    "val_a5_offer_availability",
    "val_a5_request_patient_fact",
    "val_a5_question_transaction",
    "val_a5_question_other",
    "val_a5_question_availability",
    "val_a5_statement_transaction",
    "val_a5_confirmation_transaction",
}


def blocked_surfaces():
    blocked = set()
    native = list(load_semanticlab_cases())
    for c in native:
        blocked.add(ser_key(c.context, c.utterance))

    jsonls = []
    for p in sorted(HERE.glob("semanticlab_v2_*.jsonl")):
        try:
            rows = [
                json.loads(line)
                for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except Exception:
            continue
        jsonls.append(p.name)
        for row in rows:
            blocked.add(ser_key(row.get("context", ()), row.get("utterance", "")))
    return blocked, jsonls


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


def supervised_loss(model, batch, act_weights, topic_weights):
    act_logits, topic_logits = model(batch["input_ids"], batch["attention_mask"])
    act_loss = nn.functional.cross_entropy(
        act_logits,
        batch["act"],
        weight=act_weights,
        label_smoothing=0.005,
    )
    topic_loss = nn.functional.cross_entropy(
        topic_logits,
        batch["topic"],
        weight=topic_weights,
        label_smoothing=0.005,
    )
    pair_loss = nn.functional.cross_entropy(
        p8a.pair_logits(act_logits, topic_logits, VALID_PAIRS),
        batch["pair"],
    )
    return act_logits, topic_logits, act_loss + topic_loss + pair_loss


def kl_teacher_student(student_logits, teacher_logits, temperature):
    t = temperature
    return nn.functional.kl_div(
        nn.functional.log_softmax(student_logits / t, dim=-1),
        nn.functional.softmax(teacher_logits / t, dim=-1),
        reduction="batchmean",
    ) * (t * t)


def replay_distillation(student, teacher, batch):
    s_act, s_topic = student(batch["input_ids"], batch["attention_mask"])
    with torch.no_grad():
        t_act, t_topic = teacher(batch["input_ids"], batch["attention_mask"])

    s_pair = p8a.pair_logits(s_act, s_topic, VALID_PAIRS)
    t_pair = p8a.pair_logits(t_act, t_topic, VALID_PAIRS)

    return (
        0.35 * kl_teacher_student(s_act, t_act, TEMPERATURE)
        + 0.35 * kl_teacher_student(s_topic, t_topic, TEMPERATURE)
        + 0.80 * kl_teacher_student(s_pair, t_pair, TEMPERATURE)
    )


def metric_tuple(m):
    return (
        m["pair_accuracy"],
        m["act_accuracy"],
        m["topic_accuracy"],
    )


def main():
    print("========== PHASE 8A5 STABILITY-PRESERVING NATURAL GENERALIZATION ==========")
    print("telephony=DISABLED")
    print("runtime_wiring_modified=NO")
    print("production_source_modified=NO")
    print("current_phase8a3_overwritten=NO")
    print("initialization=CURRENT_A8A3")
    print("teacher=CURRENT_A8A3_FROZEN")
    print("student_trainable_encoder_layers=5")
    print("legacy_distillation=YES")
    print("legacy_weighted_replay=YES")
    print("exposed120_gradient_updates=NO")
    print("exposed120_model_selection=NO")

    a8a3_sha_before = sha256_file(A8A3)
    source_sha_before = sha256_file(P8A)

    blocked, jsonl_names = blocked_surfaces()
    print("blocked_jsonl_files=", len(jsonl_names))
    print("blocked_serialized_surfaces=", len(blocked))

    old_examples = p8a.build_examples()
    old_train = [
        ex for ex in old_examples
        if ex.family not in p8a.VALIDATION_FAMILIES
        and ser_key(ex.context, ex.turn) not in blocked
    ]
    old_val = [
        ex for ex in old_examples
        if ex.family in p8a.VALIDATION_FAMILIES
        and ser_key(ex.context, ex.turn) not in blocked
    ]

    hard = build_a8a5_hard_examples()
    hard_train = [
        ex for ex in hard
        if ex.family not in VAL_A5
        and ser_key(ex.context, ex.turn) not in blocked
    ]
    hard_val = [
        ex for ex in hard
        if ex.family in VAL_A5
        and ser_key(ex.context, ex.turn) not in blocked
    ]

    # Dedup each stream.
    def dedup(items):
        out = []
        seen = set()
        for ex in items:
            key = (ser_key(ex.context, ex.turn), ex.speech_act, ex.topic)
            if key in seen:
                continue
            seen.add(key)
            out.append(ex)
        return out

    old_train = dedup(old_train)
    old_val = dedup(old_val)
    hard_train = dedup(hard_train)
    hard_val = dedup(hard_val)

    print("legacy_train_examples=", len(old_train))
    print("legacy_validation_examples=", len(old_val))
    print("new_hard_train_examples=", len(hard_train))
    print("new_hard_validation_examples=", len(hard_val))
    print(
        "post_filter_exact_benchmark_overlap=",
        sum(
            ser_key(ex.context, ex.turn) in blocked
            for ex in old_train + old_val + hard_train + hard_val
        ),
    )

    tok = AutoTokenizer.from_pretrained(p8a.MODEL_NAME)

    teacher = p8a.DenseModel()
    teacher.load_state_dict(CURRENT["state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = p8a.DenseModel()
    student.load_state_dict(CURRENT["state_dict"])

    # Freeze blocks 0-4. Only final encoder block adapts.
    for layer_i, layer in enumerate(student.encoder.transformer.layer):
        trainable = layer_i == 5
        for p in layer.parameters():
            p.requires_grad = trainable

    # Weights from the union improve class balance without enormous oversampling.
    combined_for_weights = old_train + hard_train
    act_weights, act_counts = p8a.class_weights(combined_for_weights, "act")
    topic_weights, topic_counts = p8a.class_weights(combined_for_weights, "topic")

    print("train_act_counts=", dict(act_counts))
    print("train_topic_counts=", dict(topic_counts))

    replay_sample_weights = [
        LEGACY_PAIR_SAMPLE_WEIGHT.get((ex.speech_act, ex.topic), DEFAULT_REPLAY_WEIGHT)
        for ex in old_train
    ]
    replay_sampler = WeightedRandomSampler(
        weights=replay_sample_weights,
        num_samples=max(len(old_train), 640),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    replay_loader = DataLoader(
        p8a.DenseDataset(old_train, tok, PAIR_TO_I),
        batch_size=BATCH_SIZE,
        sampler=replay_sampler,
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
                "lr": 2.5e-6,
                "weight_decay": 0.01,
            },
            {
                "params": list(student.act_head.parameters())
                + list(student.topic_head.parameters()),
                "lr": 7.5e-5,
                "weight_decay": 1e-4,
            },
        ]
    )

    baseline_native_cases = list(load_semanticlab_cases())
    exposed_all = list(load_semanticlab_cases(EXPOSED120))
    exposed_legal = legal_subset(exposed_all)
    exposed_legal_no_spec = [
        c for c in exposed_legal if c.case_id not in SPEC_REVIEW_IDS
    ]
    prior_groups = load_prior_groups()

    baseline_native = p8a.eval_examples(teacher, tok, baseline_native_cases, VALID_PAIRS)
    baseline_exp = p8a.eval_examples(teacher, tok, exposed_legal_no_spec, VALID_PAIRS)
    baseline_prior = {
        name: p8a.eval_examples(teacher, tok, cases, VALID_PAIRS)
        for name, cases in prior_groups
    }

    print()
    print("========== BASELINE A8A3 ==========")
    print(
        "native133_pair=", round(baseline_native["pair_accuracy"], 4),
        "act=", round(baseline_native["act_accuracy"], 4),
        "topic=", round(baseline_native["topic_accuracy"], 4),
    )
    print(
        "exposed_legal_no_spec_pair=", round(baseline_exp["pair_accuracy"], 4),
        "cases=", len(exposed_legal_no_spec),
    )

    best_state = None
    best_epoch = None
    best_score = None
    stale = 0

    print()
    print("========== A8A5 CONTINUATION TRAINING ==========")
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        student.train()
        running = 0.0
        replay_iter = itertools.cycle(replay_loader)
        hard_iter = itertools.cycle(hard_loader)
        steps = max(len(replay_loader), len(hard_loader))

        for _ in range(steps):
            replay_batch = next(replay_iter)
            hard_batch = next(hard_iter)

            optimizer.zero_grad()

            # New natural-language supervision.
            _, _, hard_sup = supervised_loss(
                student,
                hard_batch,
                act_weights,
                topic_weights,
            )

            # Legacy supervision + A8A3 teacher anchor.
            _, _, replay_sup = supervised_loss(
                student,
                replay_batch,
                act_weights,
                topic_weights,
            )
            distill = replay_distillation(student, teacher, replay_batch)

            loss = hard_sup + 0.85 * replay_sup + 1.15 * distill
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 0.8)
            optimizer.step()
            running += float(loss.item())

        old_m = p8a.eval_examples(student, tok, old_val, VALID_PAIRS)
        new_m = p8a.eval_examples(student, tok, hard_val, VALID_PAIRS)

        # Select using only synthetic family holdouts, never exposed120.
        # Lexicographic priority:
        # 1. floor across old/new pair
        # 2. old pair preservation
        # 3. new pair
        score = (
            min(old_m["pair_accuracy"], new_m["pair_accuracy"]),
            old_m["pair_accuracy"],
            new_m["pair_accuracy"],
            new_m["act_accuracy"],
            new_m["topic_accuracy"],
        )

        print(
            f"epoch={epoch:02d}",
            f"loss={running/steps:.4f}",
            f"old_val_pair={old_m['pair_accuracy']:.4f}",
            f"new_val_pair={new_m['pair_accuracy']:.4f}",
            f"new_val_act={new_m['act_accuracy']:.4f}",
            f"new_val_topic={new_m['topic_accuracy']:.4f}",
        )

        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in student.state_dict().items()
            }
            stale = 0
        else:
            stale += 1

        if epoch >= 4 and stale >= 3:
            print("early_stopping=YES")
            break

    print("training_wall_s=", round(time.perf_counter() - started, 3))
    print("best_epoch=", best_epoch)

    if best_state is None:
        raise RuntimeError("No candidate state captured.")

    student.load_state_dict(best_state)
    student.eval()

    cand_native = p8a.eval_examples(student, tok, baseline_native_cases, VALID_PAIRS)
    cand_exp = p8a.eval_examples(student, tok, exposed_legal_no_spec, VALID_PAIRS)

    print()
    print("========== A8A5 CANDIDATE EVALUATION ==========")
    print(
        "native133_pair=", round(cand_native["pair_accuracy"], 4),
        "act=", round(cand_native["act_accuracy"], 4),
        "topic=", round(cand_native["topic_accuracy"], 4),
    )
    print(
        "exposed_legal_no_spec_pair=", round(cand_exp["pair_accuracy"], 4),
        "act=", round(cand_exp["act_accuracy"], 4),
        "topic=", round(cand_exp["topic_accuracy"], 4),
        "cases=", len(exposed_legal_no_spec),
    )

    prior_regressions = []
    for name, cases in prior_groups:
        cand = p8a.eval_examples(student, tok, cases, VALID_PAIRS)
        base_m = baseline_prior[name]
        print(
            "prior_pair",
            name,
            "baseline=", round(base_m["pair_accuracy"], 4),
            "candidate=", round(cand["pair_accuracy"], 4),
            "cases=", len(cases),
        )
        if cand["pair_accuracy"] + 1e-12 < base_m["pair_accuracy"]:
            prior_regressions.append(
                (name, base_m["pair_accuracy"], cand["pair_accuracy"])
            )

    residuals = []
    regressions_vs_baseline = []
    fixes_vs_baseline = []

    print()
    print("========== EXPOSED LEGAL TRANSITIONS ==========")
    for c, bpred, cpred in zip(
        exposed_legal_no_spec,
        baseline_exp["pair_preds"],
        cand_exp["pair_preds"],
    ):
        gold = p8a.gold_pair(c)
        b_ok = tuple(bpred) == gold
        c_ok = tuple(cpred) == gold

        if b_ok and not c_ok:
            regressions_vs_baseline.append(c.case_id)
            print(
                "REGRESSION",
                c.case_id,
                "gold=", gold,
                "a8a3=", tuple(bpred),
                "a8a5=", tuple(cpred),
                "turn=", repr(c.utterance),
                "context=", list(c.context),
            )
        elif not b_ok and c_ok:
            fixes_vs_baseline.append(c.case_id)
            print(
                "FIX",
                c.case_id,
                "gold=", gold,
                "a8a3=", tuple(bpred),
                "a8a5=", tuple(cpred),
                "turn=", repr(c.utterance),
                "context=", list(c.context),
            )
        elif not c_ok:
            residuals.append(c.case_id)
            print(
                "RESIDUAL",
                c.case_id,
                "gold=", gold,
                "a8a3=", tuple(bpred),
                "a8a5=", tuple(cpred),
                "turn=", repr(c.utterance),
                "context=", list(c.context),
            )

    # Native transitions are the hard forgetting gate.
    native_regressions = []
    native_fixes = []
    for c, bpred, cpred in zip(
        baseline_native_cases,
        baseline_native["pair_preds"],
        cand_native["pair_preds"],
    ):
        gold = p8a.gold_pair(c)
        b_ok = tuple(bpred) == gold
        c_ok = tuple(cpred) == gold
        if b_ok and not c_ok:
            native_regressions.append(c.case_id)
        elif not b_ok and c_ok:
            native_fixes.append(c.case_id)

    print()
    print("native_regression_ids=", native_regressions)
    print("native_fix_ids=", native_fixes)
    print("exposed_regression_ids=", regressions_vs_baseline)
    print("exposed_fix_ids=", fixes_vs_baseline)
    print("exposed_residual_ids=", residuals)

    # Candidate gate:
    # - preserve A8A3's native pair result exactly or improve it
    # - no historical prior pair regressions
    # - improve exposed legal natural language materially
    # - keep synthetic old/new validation healthy
    promising = all(
        (
            cand_native["pair_accuracy"] + 1e-12 >= baseline_native["pair_accuracy"],
            cand_native["act_accuracy"] + 1e-12 >= baseline_native["act_accuracy"],
            cand_native["topic_accuracy"] + 1e-12 >= baseline_native["topic_accuracy"],
            not native_regressions,
            not prior_regressions,
            cand_exp["pair_accuracy"] >= baseline_exp["pair_accuracy"] + 0.04,
            cand_exp["pair_accuracy"] >= 0.86,
            best_score[1] >= 0.90,  # old validation
            best_score[2] >= 0.90,  # new validation
        )
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "model_name": p8a.MODEL_NAME,
            "speech_acts": tuple(p8a.SPEECH_ACTS),
            "topics": tuple(p8a.TOPICS),
            "valid_pairs": VALID_PAIRS,
            "seed": SEED,
            "best_epoch": best_epoch,
            "legacy_validation_pair_accuracy": best_score[1],
            "new_validation_pair_accuracy": best_score[2],
            "native133_pair_accuracy": cand_native["pair_accuracy"],
            "native133_act_accuracy": cand_native["act_accuracy"],
            "native133_topic_accuracy": cand_native["topic_accuracy"],
            "exposed_legal_no_spec_pair_accuracy": cand_exp["pair_accuracy"],
            "baseline_exposed_legal_no_spec_pair_accuracy": baseline_exp["pair_accuracy"],
            "source_artifact": str(A8A3),
            "source_artifact_sha256": a8a3_sha_before,
            "candidate_status": "PROMISING" if promising else "NOT_GOOD_ENOUGH",
            "training_method": "weighted_legacy_replay_plus_a8a3_distillation",
        },
        OUT,
    )

    a8a3_sha_after = sha256_file(A8A3)
    source_sha_after = sha256_file(P8A)
    candidate_sha = sha256_file(OUT)

    print()
    print("========== PHASE 8A5 DECISION ==========")
    print("baseline_native133_pair=", round(baseline_native["pair_accuracy"], 4))
    print("candidate_native133_pair=", round(cand_native["pair_accuracy"], 4))
    print("baseline_exposed_legal_no_spec_pair=", round(baseline_exp["pair_accuracy"], 4))
    print("candidate_exposed_legal_no_spec_pair=", round(cand_exp["pair_accuracy"], 4))
    print(
        "exposed_pair_absolute_gain=",
        round(cand_exp["pair_accuracy"] - baseline_exp["pair_accuracy"], 4),
    )
    print("native_regressions=", native_regressions)
    print("prior_pair_regressions=", prior_regressions)
    print("remaining_exposed_legal_failure_ids=", residuals)
    print("current_a8a3_artifact_unchanged=", "YES" if a8a3_sha_before == a8a3_sha_after else "NO")
    print("phase8a_source_unchanged=", "YES" if source_sha_before == source_sha_after else "NO")
    print("candidate_artifact=", OUT)
    print("candidate_sha256=", candidate_sha)
    print("runtime_wiring_modified=NO")
    print("phase8b_retrained=NO")
    print("phase7c_retrained=NO")
    print("phase6b_retrained=NO")

    if promising:
        print("PHASE8A5_STABILITY_PRESERVING_CANDIDATE=PROMISING")
        print(
            "NEXT_ACTION=OFFLINE_LEVEL2_EVALUATION_WITH_A8A5_CANDIDATE_"
            "BEFORE_PHASE8B_REFRESH"
        )
        return 0

    print("PHASE8A5_STABILITY_PRESERVING_CANDIDATE=NOT_GOOD_ENOUGH")
    print(
        "NEXT_ACTION=ANALYZE_A8A5_TRANSITIONS; DO_NOT_RETRAIN_PHASE8B_"
        "OR_OTHER_SPECIALISTS_YET"
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
