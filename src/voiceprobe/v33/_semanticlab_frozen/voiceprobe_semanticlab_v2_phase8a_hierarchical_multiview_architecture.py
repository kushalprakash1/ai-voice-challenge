#!/usr/bin/env python3
"""SemanticLab v2 Phase 8A explicit hierarchical multi-view architecture.

WHY
---
The prior multi-view experiment proved that separately representing the turn
and its context is useful:
- zero native regressions
- zero prior regressions
- large exposed natural-language gain

Its weakness was the MONOLITHIC 20-way residual decision head.

This experiment keeps the successful frozen multi-view representation, but
replaces the single decision head with an explicit hierarchy:

    TURN/COMBINED views -> speech-act residual head (7 classes)
    TURN/CONTEXT views  -> topic residual head      (9 classes)
    ALL views           -> pair-interaction head    (20 legal pairs)

Final act/topic logits:
    frozen A8A3 logits + scaled learned residuals

Final legal-pair logits:
    legal_pair_logits(final_act, final_topic)
    + scaled pair-interaction residual

The output ontology remains the SAME frozen 20 legal speech-act/topic pairs.

DATA POLICY
-----------
Gradient training:
- original Phase8A synthetic TRAIN families
- A8A5 synthetic TRAIN families
- A8A6 synthetic TRAIN families

Model/checkpoint/scale selection:
- native 133 DEVELOPMENT corpus, with ZERO-regression hard constraint
- original Phase8A synthetic validation
- A8A5 synthetic validation
- A8A6 synthetic validation

NOT used for gradients or model/scale selection:
- exposed final 120 diagnostic
- prior exposed/adversarial corpora

Those are evaluated only after checkpoint + scales are selected.

No production/runtime/planner/telephony changes.
No frozen artifact overwrite.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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
A5SRC = HERE / "voiceprobe_semanticlab_v2_phase8a5_stability_preserving_refresh_v2.py"
A6SRC = HERE / "voiceprobe_semanticlab_v2_phase8a6_constrained_continuation.py"
EXPOSED120 = HERE / "semanticlab_v2_level2_final_unseen_holdout_120_v2_20260817.jsonl"
OUT = ROOT / "artifacts/candidates/semanticlab_v2_phase8a_hierarchical_multiview.pt"

for p in (P8A, A8A3, A5SRC, A6SRC, EXPOSED120):
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


p8a = load_mod("phase8a_hier_base", P8A)
a5src = load_mod("phase8a_hier_a5src", A5SRC)
a6src = load_mod("phase8a_hier_a6src", A6SRC)

from voiceprobe.v33.semantic_corpus import load_semanticlab_cases


SEED = 8909
random.seed(SEED)
torch.manual_seed(SEED)

EPOCHS = 120
BATCH_SIZE = 64
TEMPERATURE = 2.0

# Two independently selected correction strengths:
# - factor_scale for act/topic residuals
# - interaction_scale for legal-pair interaction residual
FACTOR_SCALE_GRID = tuple(round(i * 0.10, 2) for i in range(0, 26))       # 0..2.5
INTERACTION_SCALE_GRID = tuple(round(i * 0.10, 2) for i in range(0, 21))  # 0..2.0

CK = torch.load(A8A3, map_location="cpu")
VALID = tuple(tuple(x) for x in CK["valid_pairs"])
VALID_SET = set(VALID)
PAIR_TO_I = {pair: i for i, pair in enumerate(VALID)}
ACT_TO_I = {x: i for i, x in enumerate(p8a.SPEECH_ACTS)}
TOPIC_TO_I = {x: i for i, x in enumerate(p8a.TOPICS)}
SPEC_REVIEW = {"f2a_001"}


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(items):
    out = []
    for x in items:
        if isinstance(x, p8a.Example):
            out.append(x)
        else:
            out.append(
                p8a.Example(
                    family=x.family,
                    context=tuple(x.context),
                    turn=x.turn,
                    speech_act=x.speech_act,
                    topic=x.topic,
                )
            )
    return out


def item_parts(x):
    if isinstance(x, p8a.Example):
        return tuple(x.context), x.turn
    return tuple(x.context), x.utterance


def ser_key(x):
    context, turn = item_parts(x)
    return p8a.norm(p8a.serialize(context, turn))


def gold_pair(x):
    if isinstance(x, p8a.Example):
        return (x.speech_act, x.topic)
    return p8a.gold_pair(x)


def legal_subset(cases):
    return [c for c in cases if p8a.gold_pair(c) in VALID_SET]


def dedup(items):
    out = []
    seen = set()
    for x in items:
        key = (ser_key(x), gold_pair(x))
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


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


def build_data():
    original = p8a.build_examples()
    legacy_train = [
        x for x in original
        if x.family not in p8a.VALIDATION_FAMILIES
    ]
    legacy_val = [
        x for x in original
        if x.family in p8a.VALIDATION_FAMILIES
    ]

    blocked5, _ = a5src.blocked_surfaces()
    a5_all = a5src.build_a8a5_hard_examples()
    a5_train = canonical([
        x for x in a5_all
        if x.family not in a5src.VAL_A5
        and a5src.ser_key(x.context, x.turn) not in blocked5
    ])
    a5_val = canonical([
        x for x in a5_all
        if x.family in a5src.VAL_A5
        and a5src.ser_key(x.context, x.turn) not in blocked5
    ])

    blocked6, _ = a6src.blocked_surfaces()
    a6_all = a6src.build_hard()
    a6_train = canonical([
        x for x in a6_all
        if x.family not in a6src.VAL_FAMILIES
        and a6src.ser_key(x.context, x.turn) not in blocked6
    ])
    a6_val = canonical([
        x for x in a6_all
        if x.family in a6src.VAL_FAMILIES
        and a6src.ser_key(x.context, x.turn) not in blocked6
    ])

    legacy_train = dedup(legacy_train)
    hard_train = dedup(a5_train + a6_train)

    legacy_keys = {(ser_key(x), gold_pair(x)) for x in legacy_train}
    hard_train = [
        x for x in hard_train
        if (ser_key(x), gold_pair(x)) not in legacy_keys
    ]

    return (
        legacy_train,
        hard_train,
        dedup(legacy_val),
        dedup(a5_val),
        dedup(a6_val),
    )


@dataclass
class FrozenBatch:
    combined_h: torch.Tensor
    turn_h: torch.Tensor
    context_h: torch.Tensor
    base_act: torch.Tensor
    base_topic: torch.Tensor
    base_pair: torch.Tensor
    gold_act: torch.Tensor
    gold_topic: torch.Tensor
    gold_pair: torch.Tensor
    hard: torch.Tensor


class TensorDS(Dataset):
    def __init__(self, b):
        self.b = b

    def __len__(self):
        return len(self.b.gold_pair)

    def __getitem__(self, i):
        return (
            self.b.combined_h[i],
            self.b.turn_h[i],
            self.b.context_h[i],
            self.b.base_act[i],
            self.b.base_topic[i],
            self.b.base_pair[i],
            self.b.gold_act[i],
            self.b.gold_topic[i],
            self.b.gold_pair[i],
            self.b.hard[i],
        )


class HierarchicalFusion(nn.Module):
    def __init__(self, hidden_size, act_count, topic_count, pair_count):
        super().__init__()

        # Speech act is primarily about what the CURRENT TURN is doing,
        # while combined-turn difference exposes context-induced interpretation.
        act_dim = hidden_size * 3 + act_count
        self.act_net = nn.Sequential(
            nn.LayerNorm(act_dim),
            nn.Linear(act_dim, 256),
            nn.GELU(),
            nn.Dropout(0.16),
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, act_count),
        )

        # Topic depends on current turn plus conversational context.
        topic_dim = hidden_size * 4 + topic_count
        self.topic_net = nn.Sequential(
            nn.LayerNorm(topic_dim),
            nn.Linear(topic_dim, 256),
            nn.GELU(),
            nn.Dropout(0.16),
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, topic_count),
        )

        # Pair interaction handles context-sensitive coupling between act/topic.
        pair_dim = hidden_size * 5 + act_count + topic_count + pair_count
        self.pair_net = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, 320),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(320, 128),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(128, pair_count),
        )

        # Start as an exact zero residual => initial prediction is A8A3.
        for net in (self.act_net, self.topic_net, self.pair_net):
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def forward(
        self,
        combined_h,
        turn_h,
        context_h,
        base_act,
        base_topic,
        base_pair,
    ):
        act_probs = torch.softmax(base_act, dim=-1)
        topic_probs = torch.softmax(base_topic, dim=-1)
        pair_probs = torch.softmax(base_pair, dim=-1)

        act_features = torch.cat(
            [
                turn_h,
                combined_h,
                combined_h - turn_h,
                act_probs,
            ],
            dim=-1,
        )

        topic_features = torch.cat(
            [
                turn_h,
                context_h,
                combined_h,
                combined_h - context_h,
                topic_probs,
            ],
            dim=-1,
        )

        pair_features = torch.cat(
            [
                combined_h,
                turn_h,
                context_h,
                turn_h - context_h,
                combined_h - turn_h,
                act_probs,
                topic_probs,
                pair_probs,
            ],
            dim=-1,
        )

        return (
            self.act_net(act_features),
            self.topic_net(topic_features),
            self.pair_net(pair_features),
        )


def encode_texts(teacher, tok, texts):
    rows = []
    with torch.no_grad():
        for start in range(0, len(texts), 64):
            z = tok(
                texts[start:start+64],
                padding=True,
                truncation=True,
                max_length=p8a.MAX_LENGTH,
                return_tensors="pt",
            )
            h = teacher.encoder(
                input_ids=z["input_ids"],
                attention_mask=z["attention_mask"],
            ).last_hidden_state[:, 0, :]
            rows.append(h.cpu())
    return torch.cat(rows, dim=0)


def extract_frozen(teacher, tok, items, hard_flag=0):
    combined_texts = []
    turn_texts = []
    context_texts = []

    for x in items:
        context, turn = item_parts(x)
        ctx = " || ".join(context) if context else "<none>"

        combined_texts.append(p8a.serialize(context, turn))
        turn_texts.append("Latest clinic utterance: " + str(turn))
        context_texts.append("Recent clinic context: " + ctx)

    combined_h = encode_texts(teacher, tok, combined_texts)
    turn_h = encode_texts(teacher, tok, turn_texts)
    context_h = encode_texts(teacher, tok, context_texts)

    with torch.no_grad():
        base_act = teacher.act_head(combined_h).cpu()
        base_topic = teacher.topic_head(combined_h).cpu()
        base_pair = p8a.pair_logits(base_act, base_topic, VALID).cpu()

    gpairs = [gold_pair(x) for x in items]

    return FrozenBatch(
        combined_h=combined_h,
        turn_h=turn_h,
        context_h=context_h,
        base_act=base_act,
        base_topic=base_topic,
        base_pair=base_pair,
        gold_act=torch.tensor(
            [ACT_TO_I[a] for a, _ in gpairs],
            dtype=torch.long,
        ),
        gold_topic=torch.tensor(
            [TOPIC_TO_I[t] for _, t in gpairs],
            dtype=torch.long,
        ),
        gold_pair=torch.tensor(
            [PAIR_TO_I[p] for p in gpairs],
            dtype=torch.long,
        ),
        hard=torch.full(
            (len(items),),
            int(hard_flag),
            dtype=torch.long,
        ),
    )


def merge(a, b):
    kwargs = {}
    for field in FrozenBatch.__dataclass_fields__:
        kwargs[field] = torch.cat(
            [getattr(a, field), getattr(b, field)],
            dim=0,
        )
    return FrozenBatch(**kwargs)


def base_pred(batch):
    return batch.base_pair.argmax(dim=-1).tolist()


def forward_logits(model, batch, factor_scale, interaction_scale):
    model.eval()
    with torch.no_grad():
        ar, tr, pr = model(
            batch.combined_h,
            batch.turn_h,
            batch.context_h,
            batch.base_act,
            batch.base_topic,
            batch.base_pair,
        )
        act = batch.base_act + factor_scale * ar
        topic = batch.base_topic + factor_scale * tr
        pair = (
            p8a.pair_logits(act, topic, VALID)
            + interaction_scale * pr
        )
    return act, topic, pair


def final_pred(model, batch, factor_scale, interaction_scale):
    _, _, pair = forward_logits(
        model, batch, factor_scale, interaction_scale
    )
    return pair.argmax(dim=-1).tolist()


def pair_stats(items, pred):
    gold_ids = [PAIR_TO_I[gold_pair(x)] for x in items]

    pair_ok = [p == g for p, g in zip(pred, gold_ids)]
    act_ok = [
        VALID[p][0] == VALID[g][0]
        for p, g in zip(pred, gold_ids)
    ]
    topic_ok = [
        VALID[p][1] == VALID[g][1]
        for p, g in zip(pred, gold_ids)
    ]

    n = len(items)
    return {
        "pair": sum(pair_ok) / n,
        "act": sum(act_ok) / n,
        "topic": sum(topic_ok) / n,
        "correct": pair_ok,
    }


def row_kl(final_logits, base_logits):
    temp = TEMPERATURE
    return nn.functional.kl_div(
        nn.functional.log_softmax(final_logits / temp, dim=-1),
        nn.functional.softmax(base_logits / temp, dim=-1),
        reduction="none",
    ).sum(dim=-1) * (temp * temp)


def selection_search(model, datasets, base_native):
    best = None

    for fs in FACTOR_SCALE_GRID:
        for ps in INTERACTION_SCALE_GRID:
            metrics = {}
            predictions = {}

            for name in ("native", "legacy", "a5val", "a6val"):
                items, batch = datasets[name]
                pred = final_pred(model, batch, fs, ps)
                predictions[name] = pred
                metrics[name] = pair_stats(items, pred)

            native_items = datasets["native"][0]
            native_regressions = [
                native_items[i].case_id
                for i, was_right in enumerate(base_native["correct"])
                if was_right and not metrics["native"]["correct"][i]
            ]

            admissible = (
                not native_regressions
                and metrics["native"]["pair"] + 1e-12 >= base_native["pair"]
                and metrics["native"]["act"] + 1e-12 >= base_native["act"]
                and metrics["native"]["topic"] + 1e-12 >= base_native["topic"]
                and metrics["legacy"]["pair"] >= 0.95
            )

            if not admissible:
                continue

            natural_floor = min(
                metrics["a5val"]["pair"],
                metrics["a6val"]["pair"],
            )
            natural_avg = (
                metrics["a5val"]["pair"]
                + metrics["a6val"]["pair"]
            ) / 2.0

            score = (
                natural_floor,
                natural_avg,
                metrics["legacy"]["pair"],
                metrics["native"]["pair"],
                -(fs + ps),
            )

            row = {
                "factor_scale": fs,
                "interaction_scale": ps,
                "metrics": metrics,
                "predictions": predictions,
                "native_regressions": native_regressions,
                "score": score,
            }

            if best is None or score > best["score"]:
                best = row

    return best


def main():
    print("========== PHASE 8A EXPLICIT HIERARCHICAL MULTI-VIEW ==========")
    print("a8a3_encoder_frozen=YES")
    print("a8a3_original_heads_frozen=YES")
    print("legal_pair_ontology_frozen=YES")
    print("trainable_heads=SPEECH_ACT,TOPIC,PAIR_INTERACTION")
    print("native133_gradient_updates=NO")
    print("exposed120_gradient_updates=NO")
    print("prior_corpora_gradient_updates=NO")
    print("model_selection_uses_exposed120=NO")
    print("model_selection_uses_prior_corpora=NO")
    print("runtime_wiring=NO")
    print("telephony=DISABLED")

    sha_before = sha256_file(A8A3)

    tok = AutoTokenizer.from_pretrained(p8a.MODEL_NAME)

    teacher = p8a.DenseModel()
    teacher.load_state_dict(CK["state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    (
        legacy_train,
        hard_train,
        legacy_val,
        a5_val,
        a6_val,
    ) = build_data()

    native = list(load_semanticlab_cases())

    print("legacy_train_examples=", len(legacy_train))
    print("hard_train_examples=", len(hard_train))
    print("legacy_validation_examples=", len(legacy_val))
    print("a5_validation_examples=", len(a5_val))
    print("a6_validation_examples=", len(a6_val))
    print("native_selection_cases=", len(native))

    print()
    print("========== PRECOMPUTING FROZEN MULTI-VIEW FEATURES ==========")

    legacy_train_b = extract_frozen(
        teacher, tok, legacy_train, hard_flag=0
    )
    hard_train_b = extract_frozen(
        teacher, tok, hard_train, hard_flag=1
    )
    train_b = merge(legacy_train_b, hard_train_b)

    native_b = extract_frozen(teacher, tok, native)
    legacy_val_b = extract_frozen(teacher, tok, legacy_val)
    a5_val_b = extract_frozen(teacher, tok, a5_val)
    a6_val_b = extract_frozen(teacher, tok, a6_val)

    hidden_size = int(teacher.encoder.config.dim)

    print("hidden_size=", hidden_size)
    print("speech_act_count=", len(p8a.SPEECH_ACTS))
    print("topic_count=", len(p8a.TOPICS))
    print("legal_pair_count=", len(VALID))

    base_native = pair_stats(native, base_pred(native_b))
    base_legacy = pair_stats(legacy_val, base_pred(legacy_val_b))
    base_a5 = pair_stats(a5_val, base_pred(a5_val_b))
    base_a6 = pair_stats(a6_val, base_pred(a6_val_b))

    print()
    print("========== A8A3 BASELINES ==========")
    print(
        "native_pair=", round(base_native["pair"], 4),
        "native_act=", round(base_native["act"], 4),
        "native_topic=", round(base_native["topic"], 4),
    )
    print("legacy_val_pair=", round(base_legacy["pair"], 4))
    print("a5_val_pair=", round(base_a5["pair"], 4))
    print("a6_val_pair=", round(base_a6["pair"], 4))

    datasets = {
        "native": (native, native_b),
        "legacy": (legacy_val, legacy_val_b),
        "a5val": (a5_val, a5_val_b),
        "a6val": (a6_val, a6_val_b),
    }

    model = HierarchicalFusion(
        hidden_size=hidden_size,
        act_count=len(p8a.SPEECH_ACTS),
        topic_count=len(p8a.TOPICS),
        pair_count=len(VALID),
    )

    loader = DataLoader(
        TensorDS(train_b),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=4e-4,
        weight_decay=6e-4,
    )

    best_state = None
    best_selection = None
    best_epoch = None
    stale = 0

    print()
    print("========== HIERARCHICAL TRAINING ==========")
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0

        for (
            combined_h,
            turn_h,
            context_h,
            base_act,
            base_topic,
            base_pair,
            gold_act,
            gold_topic,
            gold_pair_ids,
            hard,
        ) in loader:
            optimizer.zero_grad()

            ar, tr, pr = model(
                combined_h,
                turn_h,
                context_h,
                base_act,
                base_topic,
                base_pair,
            )

            # Unit-scale logits for training. Scale calibration is selected
            # later on selection data, never on exposed/prior diagnostics.
            final_act = base_act + ar
            final_topic = base_topic + tr
            factor_pair = p8a.pair_logits(
                final_act,
                final_topic,
                VALID,
            )
            final_pair = factor_pair + pr

            act_ce = nn.functional.cross_entropy(
                final_act,
                gold_act,
                reduction="none",
            )
            topic_ce = nn.functional.cross_entropy(
                final_topic,
                gold_topic,
                reduction="none",
            )
            pair_ce = nn.functional.cross_entropy(
                final_pair,
                gold_pair_ids,
                reduction="none",
            )

            weights = torch.where(
                hard.bool(),
                torch.full_like(pair_ce, 3.25),
                torch.ones_like(pair_ce),
            )

            supervised = (
                (
                    0.75 * act_ce
                    + 0.75 * topic_ce
                    + 1.60 * pair_ce
                ) * weights
            ).sum() / weights.sum()

            legacy_mask = ~hard.bool()

            if bool(legacy_mask.any()):
                preserve_act = row_kl(
                    final_act[legacy_mask],
                    base_act[legacy_mask],
                ).mean()
                preserve_topic = row_kl(
                    final_topic[legacy_mask],
                    base_topic[legacy_mask],
                ).mean()
                preserve_pair = row_kl(
                    final_pair[legacy_mask],
                    base_pair[legacy_mask],
                ).mean()

                residual_l2 = (
                    ar[legacy_mask].pow(2).mean()
                    + tr[legacy_mask].pow(2).mean()
                    + pr[legacy_mask].pow(2).mean()
                )
            else:
                preserve_act = supervised * 0
                preserve_topic = supervised * 0
                preserve_pair = supervised * 0
                residual_l2 = supervised * 0

            loss = (
                supervised
                + 0.35 * preserve_act
                + 0.35 * preserve_topic
                + 0.65 * preserve_pair
                + 0.008 * residual_l2
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            optimizer.step()

            running += float(loss.item())

        selection = selection_search(
            model,
            datasets,
            base_native,
        )

        if selection is None:
            print(
                f"epoch={epoch:03d}",
                f"loss={running/len(loader):.4f}",
                "admissible_scales=NONE",
            )
            continue

        m = selection["metrics"]

        print(
            f"epoch={epoch:03d}",
            f"loss={running/len(loader):.4f}",
            f"factor_scale={selection['factor_scale']:.2f}",
            f"interaction_scale={selection['interaction_scale']:.2f}",
            f"native={m['native']['pair']:.4f}",
            f"legacy={m['legacy']['pair']:.4f}",
            f"a5val={m['a5val']['pair']:.4f}",
            f"a6val={m['a6val']['pair']:.4f}",
            "native_regressions=0",
        )

        if (
            best_selection is None
            or selection["score"] > best_selection["score"]
        ):
            best_selection = selection
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1

        if epoch >= 30 and stale >= 26:
            print("early_stopping=YES")
            break

    print(
        "training_wall_s=",
        round(time.perf_counter() - started, 3),
    )

    if best_state is None:
        print("HIERARCHICAL_PHASE8A=NO_ADMISSIBLE_CHECKPOINT")
        print("candidate_artifact_written=NO")
        print(
            "NEXT_ACTION=REVIEW_PHASE8A_ONTOLOGY_AND_TRAINING_LABEL_"
            "CONSISTENCY_BEFORE_MORE_MODEL_WORK"
        )
        return 3

    model.load_state_dict(best_state)
    model.eval()

    factor_scale = best_selection["factor_scale"]
    interaction_scale = best_selection["interaction_scale"]

    print()
    print("========== SELECTED HIERARCHICAL CANDIDATE — BEFORE EXPOSED/PRIOR ==========")
    print("best_epoch=", best_epoch)
    print("factor_scale=", factor_scale)
    print("interaction_scale=", interaction_scale)

    for name in ("native", "legacy", "a5val", "a6val"):
        m = best_selection["metrics"][name]
        print(
            name,
            "pair=", round(m["pair"], 4),
            "act=", round(m["act"], 4),
            "topic=", round(m["topic"], 4),
        )

    print(
        "native_regressions=",
        best_selection["native_regressions"],
    )

    # ================================================================
    # POST-SELECTION EXPOSED DIAGNOSTIC
    # ================================================================
    exposed_all = list(load_semanticlab_cases(EXPOSED120))
    exposed = [
        c for c in legal_subset(exposed_all)
        if c.case_id not in SPEC_REVIEW
    ]

    exposed_b = extract_frozen(teacher, tok, exposed)

    base_exp_pred = base_pred(exposed_b)
    cand_exp_pred = final_pred(
        model,
        exposed_b,
        factor_scale,
        interaction_scale,
    )

    base_exp = pair_stats(exposed, base_exp_pred)
    cand_exp = pair_stats(exposed, cand_exp_pred)

    exp_regressions = []
    exp_fixes = []
    residuals = []

    for c, bp, cp in zip(
        exposed,
        base_exp_pred,
        cand_exp_pred,
    ):
        g = PAIR_TO_I[p8a.gold_pair(c)]
        b_ok = bp == g
        c_ok = cp == g

        if b_ok and not c_ok:
            exp_regressions.append(c.case_id)
        elif not b_ok and c_ok:
            exp_fixes.append(c.case_id)

        if not c_ok:
            residuals.append(c.case_id)

    print()
    print("========== POST-SELECTION EXPOSED DIAGNOSTIC ==========")
    print("cases=", len(exposed))
    print("a8a3_pair=", round(base_exp["pair"], 4))
    print(
        "hierarchical_pair=",
        round(cand_exp["pair"], 4),
    )
    print(
        "absolute_gain=",
        round(cand_exp["pair"] - base_exp["pair"], 4),
    )
    print("regressions_vs_a8a3=", exp_regressions)
    print("fixes_vs_a8a3=", exp_fixes)
    print("remaining_failure_ids=", residuals)

    # ================================================================
    # POST-SELECTION PRIOR CORPORA
    # ================================================================
    prior_regressions = []

    print()
    print("========== POST-SELECTION PRIOR CORPORA ==========")

    for name, items in load_prior_groups():
        batch = extract_frozen(teacher, tok, items)

        bp = base_pred(batch)
        cp = final_pred(
            model,
            batch,
            factor_scale,
            interaction_scale,
        )

        bs = pair_stats(items, bp)
        cs = pair_stats(items, cp)

        print(
            name,
            "baseline=", round(bs["pair"], 4),
            "candidate=", round(cs["pair"], 4),
            "cases=", len(items),
        )

        if cs["pair"] + 1e-12 < bs["pair"]:
            prior_regressions.append(
                (name, bs["pair"], cs["pair"])
            )

    promising = all(
        (
            not best_selection["native_regressions"],
            not exp_regressions,
            not prior_regressions,
            best_selection["metrics"]["legacy"]["pair"] >= 0.95,
            best_selection["metrics"]["a5val"]["pair"] >= 0.90,
            best_selection["metrics"]["a6val"]["pair"] >= 0.90,
            cand_exp["pair"] >= 0.90,
            cand_exp["pair"] >= base_exp["pair"] + 0.07,
        )
    )

    if promising:
        OUT.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "architecture": "frozen_a8a3_hierarchical_multiview",
                "hierarchical_state_dict": best_state,
                "hidden_size": hidden_size,
                "valid_pairs": VALID,
                "speech_acts": tuple(p8a.SPEECH_ACTS),
                "topics": tuple(p8a.TOPICS),
                "factor_scale": factor_scale,
                "interaction_scale": interaction_scale,
                "a8a3_sha256": sha_before,
                "seed": SEED,
                "best_epoch": best_epoch,
                "native_pair_accuracy": best_selection["metrics"]["native"]["pair"],
                "legacy_validation_pair_accuracy": best_selection["metrics"]["legacy"]["pair"],
                "a5_validation_pair_accuracy": best_selection["metrics"]["a5val"]["pair"],
                "a6_validation_pair_accuracy": best_selection["metrics"]["a6val"]["pair"],
                "exposed_legal_no_spec_pair_accuracy": cand_exp["pair"],
                "candidate_status": "PROMISING",
            },
            OUT,
        )

        out_sha = sha256_file(OUT)
    else:
        if OUT.exists():
            OUT.unlink()
        out_sha = ""

    print()
    print("========== HIERARCHICAL PHASE 8A DECISION ==========")
    print(
        "native_regressions=",
        best_selection["native_regressions"],
    )
    print("exposed_regressions=", exp_regressions)
    print("prior_pair_regressions=", prior_regressions)
    print(
        "a8a3_unchanged=",
        "YES" if sha256_file(A8A3) == sha_before else "NO",
    )
    print("a8a3_encoder_modified=NO")
    print("a8a3_original_heads_modified=NO")
    print("runtime_wiring_modified=NO")
    print("phase8b_retrained=NO")
    print("phase7c_retrained=NO")
    print("phase6b_retrained=NO")

    if promising:
        print("candidate_artifact=", OUT)
        print("candidate_sha256=", out_sha)
        print("PHASE8A_HIERARCHICAL_MULTIVIEW=PROMISING")
        print(
            "NEXT_ACTION=OFFLINE_FULL_LEVEL2_EVALUATION_"
            "WITH_HIERARCHICAL_PHASE8A_BEFORE_PHASE8B"
        )
        return 0

    print("candidate_artifact_written=NO")
    print("PHASE8A_HIERARCHICAL_MULTIVIEW=NOT_GOOD_ENOUGH")
    print(
        "NEXT_ACTION=REVIEW_PHASE8A_ONTOLOGY_AND_TRAINING_LABEL_"
        "CONSISTENCY_BEFORE_MORE_MODEL_WORK"
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
