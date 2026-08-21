#!/usr/bin/env python3
"""
Phase 7C structured ambiguity resolution proof gate (read-only candidate proof).

This script is the architectural successor to the HARD_STOP emitted by the
frozen token/antecedent proof.  It does NOT add another scalar ambiguity
residual and it does NOT write or wire a runtime artifact.

Architecture under proof
------------------------
For in-domain (non-OOS) turns:
    context + turn
      -> Phase-7J-initialized DistilBERT specialist (top layer trainable)
      -> semantic kind head
      -> resolution-state head
      -> deterministic existing Phase 7J candidate resolver
      -> SemanticAmbiguity(kind,candidates) ONLY when state is unresolved

Internal kind ontology (shadow only; no SemanticFrame source change):
    none
    temporal_reference
    option_reference
    record_reference
    transaction_reference
    intent_next_step
    other_prior

Resolution states:
    not_applicable
    resolved_unique
    unresolved_missing_anchor
    unresolved_multiple
    unresolved_semantic

Final ambiguity is derived from resolution state, rather than learned as one
binary scalar.  Frozen OOS epoch52/scale0.9 remains authoritative and routes
through the existing frozen Phase 7J detail/candidate machinery.  Reference is
not changed or trained.

Gradient policy
---------------
Only fresh authored structured synthetic rows are used for gradients.  Exact
benchmark surfaces are filtered out.  Established1146 and exposed120 gold are
read-only selection/stability gates and never participate in gradients.

Hard approval contract
----------------------
One seed passes only if some epoch simultaneously has:
  * exact fresh structured validation final ambiguity;
  * exact fresh structured validation internal kind + resolution state;
  * exact applicable targeted ambiguity validation;
  * exact applicable metamorphic ambiguity probes and 100% invariance;
  * zero baseline-right ambiguity regressions on original V5.2 validation;
  * zero baseline-right exact ambiguity(kind,candidates) regressions across
    established1146 and exposed120;
  * at least one established/exposed or fresh capability fix;
  * frozen OOS remains exact and unchanged.

The architecture is proven only when ALL THREE deterministic seeds pass.
Otherwise this script emits a hard stop and writes nothing.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

FEAS_BASENAME = "voiceprobe_semanticlab_v2_phase7c_architecture_feasibility_gate_v1_2.py"
V52_BASENAME = "voiceprobe_semanticlab_v2_phase7c_fully_factorized_directional_residual_v5_2.py"
EXPECTED_FEAS_SHA256 = "6017b8d0c308bc992362e82d727b24f5fe5bf760801b61d004d9339deebeeabb"
EXPECTED_V52_SHA256 = "78178e2cbcf7c40c06f5f07ea5ee0848388bc42e98b623bd04f32ba6cb02f0b9"

SEEDS = (7311, 8421, 9531)
EPOCHS = 8
BATCH = 24
HEAD_LR = 4.0e-4
ENCODER_LR = 1.5e-5
WEIGHT_DECAY = 1.0e-3
MAX_LENGTH = 112
UNFREEZE_LAST_N_LAYERS = 1

OOS_EPOCH = 52
OOS_SCALE = 0.9

KINDS = (
    "none",
    "temporal_reference",
    "option_reference",
    "record_reference",
    "transaction_reference",
    "intent_next_step",
    "other_prior",
)
KIND_TO_I = {x: i for i, x in enumerate(KINDS)}

STATES = (
    "not_applicable",
    "resolved_unique",
    "unresolved_missing_anchor",
    "unresolved_multiple",
    "unresolved_semantic",
)
STATE_TO_I = {x: i for i, x in enumerate(STATES)}
UNRESOLVED_STATES = {
    "unresolved_missing_anchor",
    "unresolved_multiple",
    "unresolved_semantic",
}

DETAIL_TO_FRAME_KIND = {
    "temporal_reference": "temporal_reference",
    "option_reference": "option_reference",
    "record_reference": "record_reference",
    "transaction_reference": "transaction_reference",
    "intent_next_step": "intent",
    "other_prior": "other",
}

# This prior binary synthetic row is reference-specific.  SemanticFrame's
# current AmbiguityKind ontology has no provider-reference ambiguity kind, so
# using it to approve/reject an ambiguity architecture would be ontologically
# invalid.  It remains a reference test and is never used for ambiguity
# gradients or ambiguity scoring here.
AMBIGUITY_APPLICABILITY_EXCLUDED_FAMILIES = {
    "val_ref_provider_unresolved",
}


@dataclass(frozen=True)
class SExample:
    family: str
    context: tuple[str, ...]
    turn: str
    kind: str
    state: str
    invariance_group: str = ""

    @property
    def ambiguity(self) -> int:
        return int(self.state in UNRESOLVED_STATES)


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


def resolve_named(cli_path: str | None, basename: str) -> Path:
    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path).expanduser())
    candidates.extend([
        Path("/mnt/c/Users/llehs/Downloads") / basename,
        Path(__file__).resolve().parent / basename,
        Path.cwd() / basename,
    ])
    seen: set[Path] = set()
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if rp.is_file():
            return rp
    raise SystemExit("Could not locate " + basename + ". Checked: " + ", ".join(str(p) for p in candidates))


def norm_surface(context: Sequence[str], turn: str) -> str:
    ctx = " || ".join(" ".join(str(x).casefold().split()) for x in context)
    utt = " ".join(str(turn).casefold().split())
    return ctx + "\n" + utt


def make_asr(text: str) -> str:
    words = str(text).rstrip("?.!").split()
    if len(words) < 4:
        return "uh " + " ".join(words).lower()
    pos = min(3, len(words) - 1)
    return " ".join(words[:pos] + ["uh"] + words[pos:]).lower()


def add_s(
    rows: list[SExample],
    family: str,
    context: Sequence[str],
    turn: str,
    kind: str,
    state: str,
    *,
    invariance_group: str = "",
    add_asr: bool = False,
) -> None:
    if kind not in KIND_TO_I:
        raise ValueError(kind)
    if state not in STATE_TO_I:
        raise ValueError(state)
    if state in UNRESOLVED_STATES and kind == "none":
        raise ValueError("Unresolved state requires a non-none kind")
    base = SExample(family, tuple(context), str(turn), kind, state, invariance_group)
    rows.append(base)
    if add_asr:
        rows.append(SExample(family + "_asr", tuple(context), make_asr(turn), kind, state, invariance_group))


def build_structured_training() -> list[SExample]:
    rows: list[SExample] = []

    # Temporal: unresolved when the comparative/deictic time lacks the needed
    # anchor; resolved when a concrete prior/current anchor uniquely grounds it.
    for i, turn in enumerate([
        "Could we make the visit somewhat later?",
        "Can the appointment happen a little earlier?",
        "Is there something a bit later?",
        "Could we move it slightly sooner?",
        "Can we try a later time instead?",
        "Anything somewhat earlier?",
    ]):
        add_s(rows, "train_temporal_missing", (), turn, "temporal_reference", "unresolved_missing_anchor", invariance_group=f"tr_tm_{i}", add_asr=True)

    day_contexts = [
        "We are currently checking Wednesday availability.",
        "The search is on Friday right now.",
        "We are looking at Tuesday appointments.",
    ]
    for i, ctx in enumerate(day_contexts):
        add_s(rows, "train_temporal_day_only", (ctx,), "Is anything else available around that time?", "temporal_reference", "unresolved_missing_anchor", invariance_group=f"tr_td_{i}", add_asr=True)

    anchored = [
        ("The 1:25 PM opening is no longer available.", "What else is open around that time?"),
        ("The 10:35 AM slot was just taken.", "Is anything else available close to that time?"),
        ("The 3:45 PM appointment is booked.", "Could you check something else near that time?"),
        ("The 11:20 AM opening disappeared.", "What other openings are around that time?"),
    ]
    for i, (ctx, turn) in enumerate(anchored):
        add_s(rows, "train_temporal_resolved_context", (ctx,), turn, "temporal_reference", "resolved_unique", invariance_group=f"tr_ta_{i}", add_asr=True)

    for i, turn in enumerate([
        "Could I check something a little later this evening?",
        "Can we look a bit earlier on Thursday?",
        "Could we try somewhat later at 6:30 PM?",
        "Can you search a little sooner tomorrow morning?",
    ]):
        add_s(rows, "train_temporal_resolved_current", (), turn, "temporal_reference", "resolved_unique", invariance_group=f"tr_tc_{i}", add_asr=True)

    # Option reference: multiple live options + vague choice is unresolved;
    # one live option or explicit ordinal/comparative resolution is resolved.
    option_contexts = [
        "I can offer Tuesday at 9:25 AM or Thursday at 2:45 PM.",
        "The remaining choices are Monday morning or Friday afternoon.",
        "I have Dr. Monroe Wednesday or Dr. Iqbal Saturday.",
        "There is 10:10 AM or 3:20 PM available.",
    ]
    vague_turns = ["That choice works for me.", "Let's use that option.", "That one seems better."]
    for ci, ctx in enumerate(option_contexts):
        for ti, turn in enumerate(vague_turns):
            add_s(rows, "train_option_multiple", (ctx,), turn, "option_reference", "unresolved_multiple", invariance_group=f"tr_om_{ci}_{ti}", add_asr=(ti == 0))

    single_contexts = [
        "Only Tuesday at 11:35 AM remains available.",
        "The sole remaining choice is Thursday afternoon.",
        "Only Dr. Navarro on Monday is still open.",
        "The only open slot is 4:05 PM.",
    ]
    for ci, ctx in enumerate(single_contexts):
        for ti, turn in enumerate(["That one works.", "Use that option.", "That choice is fine."]):
            add_s(rows, "train_option_single_resolved", (ctx,), turn, "option_reference", "resolved_unique", invariance_group=f"tr_os_{ci}_{ti}", add_asr=(ti == 1))

    for ci, ctx in enumerate(option_contexts):
        for ti, turn in enumerate(["Use the first option.", "The second choice works.", "I'll take the later option."]):
            add_s(rows, "train_option_explicit_resolved", (ctx,), turn, "option_reference", "resolved_unique", invariance_group=f"tr_oe_{ci}_{ti}", add_asr=(ti == 2))

    # Record reference.
    for i, turn in enumerate([
        "It looks like that's missing from the system.",
        "That seems to be gone from the record.",
        "I don't see that there anymore.",
        "It seems like that isn't on file now.",
    ]):
        add_s(rows, "train_record_missing_anchor", (), turn, "record_reference", "unresolved_missing_anchor", invariance_group=f"tr_rm_{i}", add_asr=True)

    record_contexts = [
        "We are discussing your appointment record.",
        "The current topic is your patient profile.",
        "I am looking at the appointment entry now.",
        "We are reviewing the profile record.",
    ]
    for i, ctx in enumerate(record_contexts):
        add_s(rows, "train_record_resolved", (ctx,), "It looks like that's missing.", "record_reference", "resolved_unique", invariance_group=f"tr_rr_{i}", add_asr=True)

    # Transaction reference.
    tx_multi = [
        "I can cancel the visit or move it to another time.",
        "The available actions are reschedule or keep the current booking.",
        "I can book a new visit or cancel the existing one.",
        "We can move the appointment or leave it unchanged.",
    ]
    for i, ctx in enumerate(tx_multi):
        add_s(rows, "train_tx_multiple", (ctx,), "Should I do that now?", "transaction_reference", "unresolved_multiple", invariance_group=f"tr_xm_{i}", add_asr=True)
        add_s(rows, "train_tx_multiple", (ctx,), "Do you want me to proceed with that?", "transaction_reference", "unresolved_multiple", invariance_group=f"tr_xm2_{i}")

    tx_single = [
        "The only action under discussion is cancellation.",
        "The only pending action is rescheduling the appointment.",
        "The only action left is booking the new visit.",
        "The only pending step is keeping the current appointment.",
    ]
    for i, ctx in enumerate(tx_single):
        add_s(rows, "train_tx_single_resolved", (ctx,), "Should I do that now?", "transaction_reference", "resolved_unique", invariance_group=f"tr_xs_{i}", add_asr=True)

    # Intrinsically semantic uncertainty families.
    for i, turn in enumerate([
        "Okay, what should happen next?",
        "Alright, what comes next now?",
        "So what do we do from here?",
        "Okay, where do we go next?",
    ]):
        add_s(rows, "train_intent_semantic", (), turn, "intent_next_step", "unresolved_semantic", invariance_group=f"tr_in_{i}", add_asr=True)

    for i, (ctx, turn) in enumerate([
        ("We discussed the prior option and the insurance question.", "Which prior thing did you mean?"),
        ("We just covered the earlier slot and the profile topic.", "What were you referring to there?"),
        ("The previous exchange mentioned a schedule choice and another topic.", "Which earlier thing are you referring to?"),
    ]):
        add_s(rows, "train_other_prior", (ctx,), turn, "other_prior", "unresolved_semantic", invariance_group=f"tr_ot_{i}", add_asr=True)

    # Not-applicable controls: explicit, ordinary in-domain language that must
    # not become ambiguity merely because related nouns/verbs are present.
    controls = [
        ((), "My insurance is Blue Shield."),
        ((), "My last name is Morgan."),
        ((), "I need an in-person appointment."),
        ((), "What is the reason for changing the appointment?"),
        ((), "Can you check Friday at 3:30 PM?"),
        ((), "Please keep Thursday and search the morning."),
        ((), "Dr. Navarro works for me."),
        ((), "Can you cancel the appointment?"),
        ((), "Please reschedule the visit to Monday."),
        ((), "Is the appointment still on my account?"),
        (("The 2:10 PM slot is booked.",), "Can you search Friday instead?"),
        (("I can offer Monday or Thursday.",), "Use Thursday."),
        (("I have 9:15 AM or 1:35 PM.",), "Use the 1:35 PM slot."),
        (("Dr. Monroe is available Wednesday.",), "Book with Dr. Monroe."),
        (("The appointment is on Tuesday.",), "Keep Tuesday."),
    ]
    for i, (ctx, turn) in enumerate(controls):
        add_s(rows, "train_not_applicable", ctx, turn, "none", "not_applicable", invariance_group=f"tr_na_{i}", add_asr=(i % 3 == 0))

    return rows


def build_structured_validation() -> list[SExample]:
    rows: list[SExample] = []
    specs = [
        ("sv_temporal_missing", (), "Could the visit happen somewhat earlier?", "temporal_reference", "unresolved_missing_anchor", "sv_t1"),
        ("sv_temporal_missing_day", ("We are checking Saturday availability.",), "Anything else near that time?", "temporal_reference", "unresolved_missing_anchor", "sv_t2"),
        ("sv_temporal_resolved", ("The 12:40 PM slot is taken.",), "What else is open near that time?", "temporal_reference", "resolved_unique", "sv_t3"),
        ("sv_temporal_current_anchor", (), "Could we look a little later tonight?", "temporal_reference", "resolved_unique", "sv_t4"),
        ("sv_option_multiple", ("I can offer Wednesday at 8:35 AM or Friday at 3:55 PM.",), "That appointment choice seems fine.", "option_reference", "unresolved_multiple", "sv_o1"),
        ("sv_option_multiple2", ("The choices are Monday afternoon or Thursday morning.",), "Let's go with that one.", "option_reference", "unresolved_multiple", "sv_o2"),
        ("sv_option_single", ("Only Friday at 10:25 AM remains.",), "That one is fine.", "option_reference", "resolved_unique", "sv_o3"),
        ("sv_option_ordinal", ("I can offer Tuesday morning or Saturday afternoon.",), "Use the second option.", "option_reference", "resolved_unique", "sv_o4"),
        ("sv_record_missing", (), "That record seems to be missing now.", "record_reference", "unresolved_missing_anchor", "sv_r1"),
        ("sv_record_resolved", ("We are reviewing the appointment record.",), "That looks like it's missing.", "record_reference", "resolved_unique", "sv_r2"),
        ("sv_tx_multiple", ("I can cancel the visit or reschedule it.",), "Should I proceed with that now?", "transaction_reference", "unresolved_multiple", "sv_x1"),
        ("sv_tx_single", ("The only pending action is cancellation.",), "Should I proceed with that now?", "transaction_reference", "resolved_unique", "sv_x2"),
        ("sv_intent", (), "Alright, what happens from here?", "intent_next_step", "unresolved_semantic", "sv_i1"),
        ("sv_other", ("We discussed a prior option and a separate topic.",), "Which earlier thing did you mean?", "other_prior", "unresolved_semantic", "sv_q1"),
        ("sv_none_fact", (), "My insurance carrier is Horizon.", "none", "not_applicable", "sv_n1"),
        ("sv_none_search", (), "Can you check Sunday at 4:45 PM?", "none", "not_applicable", "sv_n2"),
        ("sv_none_tx", (), "Please cancel the visit.", "none", "not_applicable", "sv_n3"),
        ("sv_none_explicit", ("I have Monday or Wednesday.",), "Use Wednesday.", "none", "not_applicable", "sv_n4"),
    ]
    for family, ctx, turn, kind, state, inv in specs:
        add_s(rows, family, ctx, turn, kind, state, invariance_group=inv, add_asr=True if family in {"sv_temporal_missing", "sv_option_multiple", "sv_tx_multiple", "sv_none_search", "sv_record_missing", "sv_intent"} else False)
    return rows


def serialize_rows(rows: Sequence[object]) -> list[str]:
    out: list[str] = []
    for row in rows:
        context = tuple(getattr(row, "context", ()))
        turn = str(getattr(row, "turn", getattr(row, "utterance", "")))
        ctx = " || ".join(context) if context else "<none>"
        out.append(f"Recent clinic context: {ctx}\nLatest clinic utterance: {turn}")
    return out


def structured_gold_tensors(rows: Sequence[SExample]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([KIND_TO_I[x.kind] for x in rows], dtype=torch.long),
        torch.tensor([STATE_TO_I[x.state] for x in rows], dtype=torch.long),
    )


class StructuredAmbiguitySpecialist(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = copy.deepcopy(encoder)
        for p in self.encoder.parameters():
            p.requires_grad = False

        layers = None
        if hasattr(self.encoder, "transformer") and hasattr(self.encoder.transformer, "layer"):
            layers = self.encoder.transformer.layer
        elif hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer
        if layers is None or len(layers) < UNFREEZE_LAST_N_LAYERS:
            raise RuntimeError("Could not locate transformer layers for bounded top-layer unfreeze")
        for layer in layers[-UNFREEZE_LAST_N_LAYERS:]:
            for p in layer.parameters():
                p.requires_grad = True

        rep_dim = int(hidden_dim) * 2
        self.shared = nn.Sequential(
            nn.Linear(rep_dim, 160),
            nn.GELU(),
            nn.Dropout(0.12),
        )
        self.kind_head = nn.Linear(160, len(KINDS))
        self.state_head = nn.Linear(160, len(STATES))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = out[:, 0, :]
        mask = attention_mask.unsqueeze(-1).to(out.dtype)
        mean = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        h = self.shared(torch.cat([cls, mean], dim=-1))
        return self.kind_head(h), self.state_head(h)


def tokenized_dataset(tok, rows: Sequence[SExample]) -> TensorDataset:
    z = tok(serialize_rows(rows), padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    kg, sg = structured_gold_tensors(rows)
    return TensorDataset(z["input_ids"], z["attention_mask"], kg, sg)


def class_weight(labels: torch.Tensor, nclass: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=nclass).float().clamp_min(1.0)
    w = counts.sum() / counts
    return (w / w.mean()).clamp(0.5, 3.0)


def train_one_epoch(model, loader, opt, kind_w, state_w) -> dict[str, float]:
    model.train()
    total = 0.0
    ktotal = 0.0
    stotal = 0.0
    n = 0
    for ids, mask, kg, sg in loader:
        opt.zero_grad(set_to_none=True)
        kl, sl = model(ids, mask)
        lk = nn.functional.cross_entropy(kl, kg, weight=kind_w)
        ls = nn.functional.cross_entropy(sl, sg, weight=state_w)
        loss = lk + 1.15 * ls
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        bs = ids.shape[0]
        total += float(loss.item()) * bs
        ktotal += float(lk.item()) * bs
        stotal += float(ls.item()) * bs
        n += bs
    return {"loss": total/n, "kind_loss": ktotal/n, "state_loss": stotal/n}


def predict_structured(model, tok, rows: Sequence[object], batch: int = 64) -> tuple[list[str], list[str], list[float], list[float]]:
    texts = serialize_rows(rows)
    kinds: list[str] = []
    states: list[str] = []
    kconf: list[float] = []
    sconf: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            z = tok(texts[start:start+batch], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            kl, sl = model(z["input_ids"], z["attention_mask"])
            kp = torch.softmax(kl, dim=-1)
            sp = torch.softmax(sl, dim=-1)
            ki = kp.argmax(dim=-1)
            si = sp.argmax(dim=-1)
            kinds.extend(KINDS[int(i)] for i in ki.tolist())
            states.extend(STATES[int(i)] for i in si.tolist())
            kconf.extend(float(x) for x in kp.max(dim=-1).values.tolist())
            sconf.extend(float(x) for x in sp.max(dim=-1).values.tolist())
    return kinds, states, kconf, sconf


def ambiguity_from_internal(kind: str, state: str, row: object, p7jr) -> tuple[str, tuple[str, ...], str | None]:
    if state not in UNRESOLVED_STATES:
        return "none", (), None
    if kind == "none":
        return "none", (), "unresolved_state_with_none_kind"
    turn = str(getattr(row, "turn", getattr(row, "utterance", "")))
    context = tuple(getattr(row, "context", ()))
    try:
        frame_kind, cands = p7jr.ambiguity_from_detail(kind, context, turn)
    except Exception as exc:
        return "none", (), f"resolver_error:{type(exc).__name__}:{exc}"
    return str(frame_kind), tuple(str(x) for x in cands), None


def gold_case_structure(case) -> tuple[str, tuple[str, ...]]:
    amb = case.expected.get("ambiguity") or {}
    kind = str(amb.get("kind", "none") or "none")
    cands = tuple(str(x) for x in amb.get("candidates", ()) or ())
    if kind in ("", "none"):
        return "none", ()
    return kind, cands


def existing_detail_predictions(p7j, p7jr, detail_model, detail_tok, rows: Sequence[object], active_mask: Sequence[bool]) -> list[tuple[str, tuple[str, ...]]]:
    out: list[tuple[str, tuple[str, ...]]] = [("none", ()) for _ in rows]
    idx = [i for i, active in enumerate(active_mask) if active]
    if not idx:
        return out
    items = [SimpleNamespace(context=tuple(getattr(rows[i], "context", ())), turn=str(getattr(rows[i], "turn", getattr(rows[i], "utterance", "")))) for i in idx]
    preds, _ = p7j.predict_detail(detail_model, detail_tok, items)
    for i, pred, item in zip(idx, preds, items):
        try:
            k, c = p7jr.ambiguity_from_detail(pred, item.context, item.turn)
            out[i] = (str(k), tuple(str(x) for x in c))
        except Exception:
            out[i] = ("none", ())
    return out


def applicable_fexamples(rows: Sequence[object]) -> tuple[list[int], list[str]]:
    keep: list[int] = []
    excluded: list[str] = []
    for i, row in enumerate(rows):
        fam = str(getattr(row, "family", ""))
        if fam in AMBIGUITY_APPLICABILITY_EXCLUDED_FAMILIES:
            excluded.append(fam)
            continue
        keep.append(i)
    return keep, sorted(set(excluded))


def binary_gold_fexamples(rows: Sequence[object]) -> list[int]:
    return [int(getattr(x, "ambiguity")) for x in rows]


def invariance_rate(rows: Sequence[object], preds: Sequence[int]) -> float:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        g = str(getattr(row, "invariance_group", ""))
        if g:
            groups[g].append(i)
    considered = 0
    passed = 0
    for _, idx in groups.items():
        if len(idx) < 2:
            continue
        considered += 1
        vals = {int(preds[i]) for i in idx}
        if len(vals) == 1:
            passed += 1
    return passed / considered if considered else 1.0


def describe_structured_failures(rows, expected, predicted, kinds=None, states=None, limit=12):
    out = []
    for i, (g, p) in enumerate(zip(expected, predicted)):
        if g == p:
            continue
        row = rows[i]
        rec = {
            "index": i,
            "family_or_id": str(getattr(row, "family", getattr(row, "case_id", ""))),
            "gold": g,
            "pred": p,
            "turn": str(getattr(row, "turn", getattr(row, "utterance", ""))),
            "context": list(getattr(row, "context", ())),
        }
        if kinds is not None:
            rec["internal_kind"] = kinds[i]
        if states is not None:
            rec["resolution_state"] = states[i]
        out.append(rec)
        if len(out) >= limit:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feasibility-source", default=None)
    ap.add_argument("--v52-source", default=None)
    args = ap.parse_args()

    print("========== PHASE 7C STRUCTURED AMBIGUITY RESOLUTION PROOF GATE ==========")
    print("telephony=DISABLED")
    print("training=FRESH_STRUCTURED_SYNTHETIC_ONLY")
    print("candidate_artifact_write=NO")
    print("runtime_wiring_modified=NO")
    print("reference_training=NO")
    print("scalar_ambiguity_residual=NO")
    print("architecture=KIND_PLUS_RESOLUTION_STATE_PLUS_EXISTING_CANDIDATE_RESOLVER")
    print("seeds=", SEEDS)

    feas_path = resolve_named(args.feasibility_source, FEAS_BASENAME)
    v52_path = resolve_named(args.v52_source, V52_BASENAME)
    for label, path, expected in (
        ("feasibility", feas_path, EXPECTED_FEAS_SHA256),
        ("v52", v52_path, EXPECTED_V52_SHA256),
    ):
        actual = sha256_file(path)
        print(f"{label}_source=", path)
        print(f"{label}_source_sha256=", actual)
        if actual != expected:
            raise RuntimeError(f"{label} source drift expected={expected} actual={actual}")

    feas = load_mod("phase7c_feas_v12_for_structured_gate", feas_path)
    v52 = load_mod("phase7c_v52_for_structured_gate", v52_path)
    for mod, names, label in (
        (feas, ["filter_original_synthetic", "build_targeted_validation", "build_metamorphic_probes", "capture", "reconstruct_frozen_oos", "frozen_oos_pred"], "feas"),
        (v52, ["load_current_model", "build_synthetic", "load_groups", "capture_features", "gold_example_tensor", "benchmark_surfaces", "DirectionalFactorizedResidual", "runtime_for_cases"], "v52"),
    ):
        missing = [x for x in names if not hasattr(mod, x)]
        if missing:
            raise RuntimeError(f"{label} helper contract missing: {missing}")
    print("PRECHECK_IMPORTED_HELPERS=PASS")

    # Load existing Phase7J detail specialist and candidate resolver from the
    # exact current full evaluator source selected by V5.2.
    base = v52.base
    p7j = base.load_mod("structured_gate_p7j", base.P7J)
    p7jr = base.load_mod("structured_gate_p7jr", base.P7JR)
    ck7j = base.load_checkpoint(base.A7J)
    detail_model = p7j.DetailModel()
    detail_model.load_state_dict(ck7j["state_dict"])
    detail_model.eval()
    detail_tok = base.tokenizer_for(ck7j.get("model_name", getattr(p7j, "MODEL_NAME", "distilbert/distilbert-base-uncased")))
    print("PHASE7J_DETAIL_AND_RESOLVER_IMPORT=PASS")

    source_before = base.source_snapshot()
    hashes_before = {
        "v52_source": sha256_file(v52_path),
        "phase7c_source": sha256_file(base.P7C),
        "phase7c_checkpoint": sha256_file(base.A7C),
        "phase7j_source": sha256_file(base.P7J),
        "phase7j_resolver": sha256_file(base.P7JR),
        "phase7j_checkpoint": sha256_file(base.A7J),
        "phase8a_a8a3": sha256_file(base.A8A3),
    }
    artifact_before = (bool(v52.OUT.exists()), sha256_file(v52.OUT) if v52.OUT.exists() else None)

    # ------------------------------------------------------------------
    # Exact OOS replay at the already-proven RNG point.
    # ------------------------------------------------------------------
    print("========== RECONSTRUCTING AUTHORITATIVE FROZEN OOS ==========")
    torch.manual_seed(v52.SEED)
    gate_ck, gate_model, gate_tok = v52.load_current_model()
    thresholds = {f: float(gate_ck["thresholds"][f]) for f in v52.FIELDS}
    original_train, original_val = v52.build_synthetic()
    original_train, original_val, blocked, blocked_file_count = feas.filter_original_synthetic(v52, original_train, original_val)
    groups, exposed_cases = v52.load_groups()
    historical_cases = groups[0][1]
    established_cases = [c for _, cases in groups for c in cases]
    if len(established_cases) != 1146 or len(exposed_cases) != 120 or len(historical_cases) != 133:
        raise RuntimeError("Development corpus cardinality drift")

    train_x, _, train_margin, _, _, encoder_name = v52.capture_features(gate_model, gate_tok, v52.runtime_for_examples(original_train), thresholds)
    oval_x, _, oval_margin, oval_base_t, _, _ = v52.capture_features(gate_model, gate_tok, v52.runtime_for_examples(original_val), thresholds)
    hist_x, _, hist_margin, _, _, _ = v52.capture_features(gate_model, gate_tok, v52.runtime_for_cases(historical_cases), thresholds)
    replay_head = v52.DirectionalFactorizedResidual(train_x.shape[1])
    oos_head = replay_head.oos
    train_y = v52.gold_example_tensor(original_train).float()
    oos_head = feas.reconstruct_frozen_oos(v52, oos_head, train_x, train_y, train_margin, original_train)

    target_rows = feas.build_targeted_validation()
    probe_rows = feas.build_metamorphic_probes()
    structured_train = build_structured_training()
    structured_val = build_structured_validation()

    # Training novelty against every blocked benchmark surface, plus exact
    # isolation from all fresh validation/probe surfaces.
    target_surfaces = {norm_surface(x.context, x.turn) for x in target_rows}
    probe_surfaces = {norm_surface(x.context, x.turn) for x in probe_rows}
    sval_surfaces = {norm_surface(x.context, x.turn) for x in structured_val}
    train_before = len(structured_train)
    structured_train = [x for x in structured_train if norm_surface(x.context, x.turn) not in blocked and norm_surface(x.context, x.turn) not in target_surfaces and norm_surface(x.context, x.turn) not in probe_surfaces and norm_surface(x.context, x.turn) not in sval_surfaces]
    if len(structured_train) < 120:
        raise RuntimeError(f"Too few novel structured gradient rows after filtering: {len(structured_train)}")
    if len({norm_surface(x.context, x.turn) for x in structured_train}) != len(structured_train):
        raise RuntimeError("Duplicate structured training surfaces remain")
    if any(norm_surface(x.context, x.turn) in blocked for x in structured_val):
        raise RuntimeError("Structured validation overlaps benchmark surface")
    print("blocked_jsonl_files=", blocked_file_count)
    print("structured_training_rows_before_after_filter=", (train_before, len(structured_train)))
    print("structured_validation_rows=", len(structured_val))
    print("targeted_rows=", len(target_rows), "probe_rows=", len(probe_rows))
    print("training_kind_counts=", dict(Counter(x.kind for x in structured_train)))
    print("training_state_counts=", dict(Counter(x.state for x in structured_train)))
    print("FRESH_STRUCTURED_NOVELTY=PASS")

    # Capture post-replay feature sets only AFTER OOS initialization/replay.
    t_x, _, t_margin, t_base_t, _, _ = feas.capture(v52, gate_model, gate_tok, target_rows, thresholds)
    p_x, _, p_margin, p_base_t, _, _ = feas.capture(v52, gate_model, gate_tok, probe_rows, thresholds)
    sv_x, _, sv_margin, sv_base_t, _, _ = feas.capture(v52, gate_model, gate_tok, structured_val, thresholds)
    est_x, _, est_margin, est_base_t, _, _ = v52.capture_features(gate_model, gate_tok, v52.runtime_for_cases(established_cases), thresholds)
    exp_x, _, exp_margin, exp_base_t, _, _ = v52.capture_features(gate_model, gate_tok, v52.runtime_for_cases(exposed_cases), thresholds)

    oos_oval, _ = feas.frozen_oos_pred(oos_head, oval_x, oval_margin)
    oos_target, _ = feas.frozen_oos_pred(oos_head, t_x, t_margin)
    oos_probe, _ = feas.frozen_oos_pred(oos_head, p_x, p_margin)
    oos_sval, _ = feas.frozen_oos_pred(oos_head, sv_x, sv_margin)
    oos_est, _ = feas.frozen_oos_pred(oos_head, est_x, est_margin)
    oos_exp, _ = feas.frozen_oos_pred(oos_head, exp_x, exp_margin)

    # Exact already-proven OOS contract.
    oval_gold = v52.gold_example_tensor(original_val)[:, 2].long()
    est_gold_oos = v52.gold_case_tensor(established_cases)[:, 2].long()
    exp_gold_oos = v52.gold_case_tensor(exposed_cases)[:, 2].long()
    oos_checks = {
        "original_val_exact": f"{int(oos_oval.eq(oval_gold).sum())}/{len(oval_gold)}",
        "established_exact": f"{int(oos_est.eq(est_gold_oos).sum())}/{len(est_gold_oos)}",
        "exposed_exact": f"{int(oos_exp.eq(exp_gold_oos).sum())}/{len(exp_gold_oos)}",
        "targeted_in_domain_zero": int(oos_target.sum().item()) == 0,
        "probe_in_domain_zero": int(oos_probe.sum().item()) == 0,
        "structured_val_in_domain_zero": int(oos_sval.sum().item()) == 0,
    }
    print("encoder=", encoder_name)
    print("frozen_oos_epoch_scale=", (OOS_EPOCH, OOS_SCALE))
    print("frozen_oos_checks=", oos_checks)
    if oos_checks["original_val_exact"] != f"{len(oval_gold)}/{len(oval_gold)}" or oos_checks["established_exact"] != f"{len(est_gold_oos)}/{len(est_gold_oos)}" or oos_checks["exposed_exact"] != f"{len(exp_gold_oos)}/{len(exp_gold_oos)}" or not all(oos_checks[k] for k in ("targeted_in_domain_zero", "probe_in_domain_zero", "structured_val_in_domain_zero")):
        raise RuntimeError("Frozen OOS contract failed; structured tournament aborted")
    print("OOS_FREEZE_RECONSTRUCTION=PASS")

    # Base/current exact structured ambiguity for established/exposed.  This is
    # the stability reference: scalar gate decides activation, Phase7J provides
    # kind/candidates exactly as current assembly does.
    est_base_active = [bool(est_base_t[i, 1].item()) or bool(est_base_t[i, 2].item()) for i in range(len(established_cases))]
    exp_base_active = [bool(exp_base_t[i, 1].item()) or bool(exp_base_t[i, 2].item()) for i in range(len(exposed_cases))]
    base_struct_est = existing_detail_predictions(p7j, p7jr, detail_model, detail_tok, established_cases, est_base_active)
    base_struct_exp = existing_detail_predictions(p7j, p7jr, detail_model, detail_tok, exposed_cases, exp_base_active)
    gold_struct_est = [gold_case_structure(c) for c in established_cases]
    gold_struct_exp = [gold_case_structure(c) for c in exposed_cases]
    base_struct_exact = {
        "established": f"{sum(a == b for a,b in zip(base_struct_est,gold_struct_est))}/{len(established_cases)}",
        "exposed": f"{sum(a == b for a,b in zip(base_struct_exp,gold_struct_exp))}/{len(exposed_cases)}",
    }
    print("current_structured_ambiguity_exact=", base_struct_exact)

    # Existing Phase7J detail/candidate ability on gold-active in-domain cases,
    # ungated.  Diagnostic only; no gold enters training.
    def phase7j_active_exact(cases):
        # Evaluate all active, including intent/other/OOS; details are diagnostic.
        idx = [i for i,c in enumerate(cases) if gold_case_structure(c)[0] != "none"]
        active_rows = [cases[i] for i in idx]
        pred = existing_detail_predictions(p7j, p7jr, detail_model, detail_tok, active_rows, [True]*len(active_rows))
        gold = [gold_case_structure(x) for x in active_rows]
        return sum(a == b for a,b in zip(pred,gold)), len(gold)
    p7j_est_ok, p7j_est_n = phase7j_active_exact(established_cases)
    p7j_exp_ok, p7j_exp_n = phase7j_active_exact(exposed_cases)
    print("ungated_phase7j_active_structure_exact=", {"established": f"{p7j_est_ok}/{p7j_est_n}", "exposed": f"{p7j_exp_ok}/{p7j_exp_n}"})

    # Prepare frozen OOS structure on every evaluation row.  Candidate OOS
    # positives are routed through the existing detail/candidate machinery.
    frozen_oos_struct_est = existing_detail_predictions(p7j, p7jr, detail_model, detail_tok, established_cases, [bool(x) for x in oos_est.tolist()])
    frozen_oos_struct_exp = existing_detail_predictions(p7j, p7jr, detail_model, detail_tok, exposed_cases, [bool(x) for x in oos_exp.tolist()])
    frozen_oos_struct_oval = existing_detail_predictions(p7j, p7jr, detail_model, detail_tok, original_val, [bool(x) for x in oos_oval.tolist()])
    frozen_oos_struct_target = existing_detail_predictions(p7j, p7jr, detail_model, detail_tok, target_rows, [bool(x) for x in oos_target.tolist()])
    frozen_oos_struct_probe = existing_detail_predictions(p7j, p7jr, detail_model, detail_tok, probe_rows, [bool(x) for x in oos_probe.tolist()])
    frozen_oos_struct_sval = existing_detail_predictions(p7j, p7jr, detail_model, detail_tok, structured_val, [bool(x) for x in oos_sval.tolist()])

    target_keep, target_excluded = applicable_fexamples(target_rows)
    probe_keep, probe_excluded = applicable_fexamples(probe_rows)
    print("ambiguity_applicability_exclusions=", sorted(set(target_excluded + probe_excluded)))
    print("ambiguity_applicability_exclusion_reason=REFERENCE_SPECIFIC_WITH_NO_LEGAL_PROVIDER_AMBIGUITY_KIND")

    # Pre-tokenize immutable gradient data once.  The specialist copies the
    # same frozen Phase7J encoder initialization for every seed.
    train_ds = tokenized_dataset(detail_tok, structured_train)
    train_kind_gold = train_ds.tensors[2]
    train_state_gold = train_ds.tensors[3]
    kind_w = class_weight(train_kind_gold, len(KINDS))
    state_w = class_weight(train_state_gold, len(STATES))

    hidden_dim = int(detail_model.encoder.config.dim if hasattr(detail_model.encoder.config, "dim") else detail_model.encoder.config.hidden_size)
    seed_results = []

    print("========== THREE-SEED STRUCTURED SPECIALIST TOURNAMENT ==========")
    for seed in SEEDS:
        random.seed(seed)
        torch.manual_seed(seed)
        model = StructuredAmbiguitySpecialist(detail_model.encoder, hidden_dim, seed)
        enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
        head_params = [p for n,p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
        opt = torch.optim.AdamW([
            {"params": enc_params, "lr": ENCODER_LR},
            {"params": head_params, "lr": HEAD_LR},
        ], weight_decay=WEIGHT_DECAY)
        loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed))

        best_capability = None
        passing = []
        seed_start = time.perf_counter()
        print("SEED_START=", seed, "trainable_params=", sum(p.numel() for p in model.parameters() if p.requires_grad))

        for epoch in range(1, EPOCHS+1):
            losses = train_one_epoch(model, loader, opt, kind_w, state_w)

            # Fresh structured validation: exact internal semantics + final ambiguity.
            sv_kind, sv_state, _, _ = predict_structured(model, detail_tok, structured_val)
            sv_expected_internal = [(x.kind, x.state) for x in structured_val]
            sv_pred_internal = list(zip(sv_kind, sv_state))
            sv_internal_exact = sum(a == b for a,b in zip(sv_expected_internal, sv_pred_internal))
            sv_pred_struct = []
            sv_resolver_errors = 0
            for i, row in enumerate(structured_val):
                if int(oos_sval[i].item()) == 1:
                    pred = frozen_oos_struct_sval[i]
                else:
                    k,c,err = ambiguity_from_internal(sv_kind[i], sv_state[i], row, p7jr)
                    pred = (k,c)
                    sv_resolver_errors += int(err is not None)
                sv_pred_struct.append(pred)
            sv_gold_struct = []
            for row in structured_val:
                k,c,err = ambiguity_from_internal(row.kind, row.state, row, p7jr)
                if err:
                    raise RuntimeError(f"Authored structured validation resolver failure family={row.family}: {err}")
                sv_gold_struct.append((k,c))
            sv_final_exact = sum(a == b for a,b in zip(sv_pred_struct, sv_gold_struct))
            sv_inv = invariance_rate(structured_val, [int(k != "none") for k,_ in sv_pred_struct])

            # Targeted/probe final ambiguity; provider-reference-only row excluded.
            tk, ts, _, _ = predict_structured(model, detail_tok, target_rows)
            pk, ps, _, _ = predict_structured(model, detail_tok, probe_rows)
            t_pred = []
            p_pred = []
            resolver_errors = sv_resolver_errors
            for i,row in enumerate(target_rows):
                if int(oos_target[i].item()) == 1:
                    s = frozen_oos_struct_target[i]
                else:
                    k,c,err = ambiguity_from_internal(tk[i], ts[i], row, p7jr)
                    resolver_errors += int(err is not None)
                    s = (k,c)
                t_pred.append(int(s[0] != "none"))
            for i,row in enumerate(probe_rows):
                if int(oos_probe[i].item()) == 1:
                    s = frozen_oos_struct_probe[i]
                else:
                    k,c,err = ambiguity_from_internal(pk[i], ps[i], row, p7jr)
                    resolver_errors += int(err is not None)
                    s = (k,c)
                p_pred.append(int(s[0] != "none"))
            t_gold = binary_gold_fexamples(target_rows)
            p_gold = binary_gold_fexamples(probe_rows)
            t_exact = sum(t_pred[i] == t_gold[i] for i in target_keep)
            p_exact = sum(p_pred[i] == p_gold[i] for i in probe_keep)
            p_inv = invariance_rate(probe_rows, p_pred)

            # Original validation baseline-right binary stability.
            ok, os, _, _ = predict_structured(model, detail_tok, original_val)
            oval_pred = []
            for i,row in enumerate(original_val):
                if int(oos_oval[i].item()) == 1:
                    s = frozen_oos_struct_oval[i]
                else:
                    k,c,err = ambiguity_from_internal(ok[i], os[i], row, p7jr)
                    resolver_errors += int(err is not None)
                    s = (k,c)
                oval_pred.append(int(s[0] != "none"))
            oval_gold_amb = v52.gold_example_tensor(original_val)[:, 1].long().tolist()
            oval_base_amb = oval_base_t[:, 1].long().tolist()
            oval_regs = sum(1 for b,g,p in zip(oval_base_amb, oval_gold_amb, oval_pred) if b == g and p != g)
            oval_fixes = sum(1 for b,g,p in zip(oval_base_amb, oval_gold_amb, oval_pred) if b != g and p == g)

            fresh_ready = (
                sv_internal_exact == len(structured_val)
                and sv_final_exact == len(structured_val)
                and sv_inv == 1.0
                and t_exact == len(target_keep)
                and p_exact == len(probe_keep)
                and p_inv == 1.0
                and oval_regs == 0
                and resolver_errors == 0
            )

            est_regs = exp_regs = est_fixes = exp_fixes = None
            est_exact = exp_exact = None
            if fresh_ready:
                ek, es, _, _ = predict_structured(model, detail_tok, established_cases)
                xk, xs, _, _ = predict_structured(model, detail_tok, exposed_cases)
                cand_est = []
                cand_exp = []
                hard_error = False
                for i,row in enumerate(established_cases):
                    if int(oos_est[i].item()) == 1:
                        s = frozen_oos_struct_est[i]
                    else:
                        k,c,err = ambiguity_from_internal(ek[i], es[i], row, p7jr)
                        if err:
                            hard_error = True
                        s = (k,c)
                    cand_est.append(s)
                for i,row in enumerate(exposed_cases):
                    if int(oos_exp[i].item()) == 1:
                        s = frozen_oos_struct_exp[i]
                    else:
                        k,c,err = ambiguity_from_internal(xk[i], xs[i], row, p7jr)
                        if err:
                            hard_error = True
                        s = (k,c)
                    cand_exp.append(s)
                est_regs = sum(1 for b,g,p in zip(base_struct_est,gold_struct_est,cand_est) if b == g and p != g)
                exp_regs = sum(1 for b,g,p in zip(base_struct_exp,gold_struct_exp,cand_exp) if b == g and p != g)
                est_fixes = sum(1 for b,g,p in zip(base_struct_est,gold_struct_est,cand_est) if b != g and p == g)
                exp_fixes = sum(1 for b,g,p in zip(base_struct_exp,gold_struct_exp,cand_exp) if b != g and p == g)
                est_exact = sum(p == g for p,g in zip(cand_est,gold_struct_est))
                exp_exact = sum(p == g for p,g in zip(cand_exp,gold_struct_exp))
                strong = (not hard_error and est_regs == 0 and exp_regs == 0 and (est_fixes + exp_fixes + oval_fixes) > 0)
            else:
                strong = False

            row = {
                "seed": seed,
                "epoch": epoch,
                "loss": round(losses["loss"], 5),
                "kind_loss": round(losses["kind_loss"], 5),
                "state_loss": round(losses["state_loss"], 5),
                "structured_internal": f"{sv_internal_exact}/{len(structured_val)}",
                "structured_final": f"{sv_final_exact}/{len(structured_val)}",
                "targeted": f"{t_exact}/{len(target_keep)}",
                "probe": f"{p_exact}/{len(probe_keep)}",
                "probe_invariance": round(p_inv, 6),
                "oval_regs_fixes": (oval_regs, oval_fixes),
                "resolver_errors": resolver_errors,
                "est_regs_fixes": None if est_regs is None else (est_regs, est_fixes),
                "exp_regs_fixes": None if exp_regs is None else (exp_regs, exp_fixes),
                "est_exact": None if est_exact is None else f"{est_exact}/{len(established_cases)}",
                "exp_exact": None if exp_exact is None else f"{exp_exact}/{len(exposed_cases)}",
                "strong": strong,
            }
            print("EPOCH_RESULT=", row)

            capability_key = (
                sv_final_exact,
                sv_internal_exact,
                t_exact + p_exact,
                p_inv,
                -oval_regs,
                -resolver_errors,
                0 if est_regs is None else -est_regs,
                0 if exp_regs is None else -exp_regs,
            )
            if best_capability is None or capability_key > best_capability[0]:
                best_capability = (capability_key, row, copy.deepcopy(model.state_dict()), sv_pred_struct, list(zip(sv_kind,sv_state)))
            if strong:
                passing.append((row, copy.deepcopy(model.state_dict())))

        seed_pass = bool(passing)
        if seed_pass:
            passing.sort(key=lambda x: (x[0]["est_regs_fixes"][1] + x[0]["exp_regs_fixes"][1] + x[0]["oval_regs_fixes"][1], -x[0]["epoch"]), reverse=True)
            selected = passing[0][0]
        else:
            selected = None
        seed_result = {
            "seed": seed,
            "seed_pass": seed_pass,
            "selected": selected,
            "best_capability": best_capability[1] if best_capability else None,
            "wall_s": round(time.perf_counter() - seed_start, 3),
        }
        seed_results.append(seed_result)
        print("SEED_RESULT=", seed_result)

        if not seed_pass and best_capability is not None:
            _, br, state_dict, _, _ = best_capability
            model.load_state_dict(state_dict)
            sv_kind, sv_state, _, _ = predict_structured(model, detail_tok, structured_val)
            sv_pred = [ambiguity_from_internal(k,s,row,p7jr)[:2] for k,s,row in zip(sv_kind,sv_state,structured_val)]
            print("BEST_CAPABILITY_STRUCTURED_FAILURES=", describe_structured_failures(structured_val, sv_gold_struct, sv_pred, sv_kind, sv_state))

    all_pass = all(x["seed_pass"] for x in seed_results)
    print("========== STRUCTURED PROOF SUMMARY ==========")
    print("seed_pass_count=", f"{sum(x['seed_pass'] for x in seed_results)}/{len(seed_results)}")
    print("seed_results=", seed_results)

    source_after = base.source_snapshot()
    hashes_after = {
        "v52_source": sha256_file(v52_path),
        "phase7c_source": sha256_file(base.P7C),
        "phase7c_checkpoint": sha256_file(base.A7C),
        "phase7j_source": sha256_file(base.P7J),
        "phase7j_resolver": sha256_file(base.P7JR),
        "phase7j_checkpoint": sha256_file(base.A7J),
        "phase8a_a8a3": sha256_file(base.A8A3),
    }
    artifact_after = (bool(v52.OUT.exists()), sha256_file(v52.OUT) if v52.OUT.exists() else None)
    print("========== POSTFLIGHT INTEGRITY ==========")
    print("source_tree_python_unchanged=", "YES" if source_before == source_after else "NO")
    print("source_checkpoint_hashes_unchanged=", "YES" if hashes_before == hashes_after else "NO")
    print("v52_candidate_artifact_unchanged=", "YES" if artifact_before == artifact_after else "NO")
    print("candidate_artifact_written=NO")
    print("runtime_wiring_modified=NO")
    print("reference_modified=NO")
    if source_before != source_after or hashes_before != hashes_after or artifact_before != artifact_after:
        raise RuntimeError("Read-only integrity violation detected")

    print("========== AUTHORITATIVE STRUCTURED AMBIGUITY VERDICT ==========")
    print("OOS_REMAINS_FROZEN=YES")
    print("REFERENCE_REMAINS_UNCHANGED=YES")
    if all_pass:
        print("STRUCTURED_AMBIGUITY_ARCHITECTURE=PROVEN_ACROSS_ALL_THREE_SEEDS")
        print("NEXT_ACTION=RUN_ONE_INDEPENDENT_REPRODUCTION_OF_THIS_EXACT_STRUCTURED_ARCHITECTURE_BEFORE_ANY_RUNTIME_WIRING")
    else:
        print("STRUCTURED_AMBIGUITY_ARCHITECTURE=NOT_PROVEN")
        print("NEXT_ACTION=HARD_STOP_STRUCTURED_SPECIALIST__DO_NOT_RETURN_TO_SCALAR_RESIDUAL_OR_FROZEN_FEATURE_LOOPS__REVIEW_RESOLUTION_STATE_ONTOLOGY_AND_PHASE7J_KIND_COVERAGE_BEFORE_ANY_RUNTIME_CHANGE")
    print("structured_ambiguity_resolution_proof_completed=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
