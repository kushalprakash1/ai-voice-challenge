#!/usr/bin/env python3
"""
Phase 7C bounded architecture feasibility gate (V5.3 pre-approval only).

Purpose
-------
This is NOT V5.3 and does NOT create a deployment/candidate artifact.
It runs a bounded tournament over three predeclared reference/ambiguity
architectures across three deterministic seeds.  OOS is reconstructed exactly
from V5.2 epoch 52 / scale 0.9 and then frozen.

Hard contract
-------------
A field state is strong only if it:
  * passes the original+new fresh validation capability thresholds;
  * is exact on the new targeted validation cases for that field;
  * is exact on all fresh non-benchmark metamorphic probes for that field;
  * preserves every baseline-correct field decision in established1146;
  * preserves every baseline-correct field decision in exposed120.

An architecture is approved for ONE independent reproduction only when all
three deterministic seeds produce a strong reference state, a strong
ambiguity state, and a fresh joint composition.  Otherwise the verdict is a
hard stop: do not create V5.3 and do not automatically make another V5.x.

No benchmark/exposed case is used for gradients.  Development gold is used
only for selection/stability gating.  No runtime wiring or production source
is modified.  No artifact is written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

EXPECTED_V52_SHA256 = "78178e2cbcf7c40c06f5f07ea5ee0848388bc42e98b623bd04f32ba6cb02f0b9"
V52_BASENAME = "voiceprobe_semanticlab_v2_phase7c_fully_factorized_directional_residual_v5_2.py"
SEEDS = (7311, 8421, 9531)
EPOCHS = 60
LR = 2.0e-4
WEIGHT_DECAY = 1.0e-3
SCALES = tuple(round(i * 0.1, 1) for i in range(1, 31))
OOS_EPOCH = 52
OOS_SCALE = 0.9
OOS_AMBIGUITY_CONFIDENCE_THRESHOLD = 0.0

MIN_VAL_FIELD_ACC = 0.92
MIN_VAL_ACTIVE_ACC = 0.88
MIN_VAL_NEGATIVE_ACC = 0.90
MIN_VAL_JOINT_ACC = 0.86

ARCHITECTURES = (
    "context_contrast_residual",
    "gated_contrast_invariance",
    "gated_contrast_invariance_discourse",
)

REFERENCE_CONFLICT_FAMILIES_REMOVED = {
    "reference_none_provider_query",
    "reference_none_provider_query_asr",
}
AMBIGUITY_CONFLICT_FAMILIES_REMOVED = {
    "ambiguity_negative_transaction",
    "ambiguity_negative_transaction_asr",
}


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


def resolve_v52_source(cli_path: str | None) -> Path:
    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path).expanduser())
    candidates.extend(
        [
            Path("/mnt/c/Users/llehs/Downloads") / V52_BASENAME,
            Path(__file__).resolve().parent / V52_BASENAME,
            Path.cwd() / V52_BASENAME,
        ]
    )
    seen = set()
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if rp.is_file():
            return rp
    raise SystemExit(
        "Could not locate exact V5.2 source. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


@dataclass(frozen=True)
class FExample:
    family: str
    turn: str
    context: tuple[str, ...]
    reference: int
    ambiguity: int
    oos: int = 0
    invariance_group: str = ""
    contrast_group: str = ""


def make_asr(text: str) -> str:
    words = str(text).split()
    if len(words) < 4:
        return str(text).lower().rstrip("?.!")
    pos = min(3, len(words) - 1)
    return " ".join(words[:pos] + ["uh"] + words[pos:]).lower().rstrip("?.!")


def add_example(
    rows: list[FExample],
    family: str,
    turn: str,
    context: Sequence[str],
    reference: int,
    ambiguity: int,
    *,
    invariance_group: str = "",
    contrast_group: str = "",
    add_asr: bool = False,
) -> None:
    rows.append(
        FExample(
            family=family,
            turn=turn,
            context=tuple(context),
            reference=int(reference),
            ambiguity=int(ambiguity),
            invariance_group=invariance_group,
            contrast_group=contrast_group,
        )
    )
    if add_asr:
        rows.append(
            FExample(
                family=family + "_asr",
                turn=make_asr(turn),
                context=tuple(context),
                reference=int(reference),
                ambiguity=int(ambiguity),
                invariance_group=invariance_group,
                contrast_group=contrast_group,
            )
        )


def build_feasibility_training() -> list[FExample]:
    """Fresh semantic contrasts.  No benchmark strings or case IDs."""
    rows: list[FExample] = []

    # ------------------------------------------------------------------
    # Provider-pronoun antecedent contrasts + gender/name invariance.
    # Same pronoun surface is reference=1 when a provider antecedent exists,
    # but reference=0/ambiguity=1 when no provider antecedent exists.
    # ------------------------------------------------------------------
    provider_specs = [
        ("Monroe", "Wednesday at 1:40 PM"),
        ("Navarro", "Monday at 4:20 PM"),
        ("Iqbal", "Thursday at 10:50 AM"),
        ("Rivera", "Tuesday at 2:35 PM"),
    ]
    pronouns = [("she", "Does she have any other openings?"),
                ("he", "Does he have any other openings?"),
                ("they", "Do they have any other openings?")]
    for i, (name, slot) in enumerate(provider_specs):
        inv = f"train_provider_inv_{i}"
        contrast = f"train_provider_contrast_{i}"
        for pkey, turn in pronouns:
            add_example(
                rows,
                "feas_ref_provider_antecedent",
                turn,
                [f"Dr. {name} currently has {slot} available."],
                1,
                0,
                invariance_group=inv,
                contrast_group=contrast,
                add_asr=True,
            )
        # Same discourse form without a provider antecedent.
        add_example(
            rows,
            "feas_ref_provider_unresolved",
            "Does she have any other openings?",
            [f"There is an opening {slot}."],
            0,
            1,
            contrast_group=contrast,
            add_asr=True,
        )
        # Explicit-name control requires no contextual reference.
        add_example(
            rows,
            "feas_ref_provider_explicit_name",
            f"Does Dr. {name} have any other openings?",
            [],
            0,
            0,
            add_asr=True,
        )

    # ------------------------------------------------------------------
    # Time/day deictic antecedent contrasts.
    # ------------------------------------------------------------------
    time_specs = [
        ("9:35 AM", "Tuesday"),
        ("1:25 PM", "Friday"),
        ("3:45 PM", "Monday"),
        ("11:15 AM", "Thursday"),
    ]
    for i, (clock, day) in enumerate(time_specs):
        inv = f"train_time_inv_{i}"
        contrast = f"train_time_contrast_{i}"
        add_example(
            rows,
            "feas_ref_time_antecedent",
            "Could you find anything else close to that time?",
            [f"The {clock} opening on {day} is no longer available."],
            1,
            0,
            invariance_group=inv,
            contrast_group=contrast,
            add_asr=True,
        )
        add_example(
            rows,
            "feas_ref_time_unresolved",
            "Could you find anything else close to that time?",
            [f"We are checking {day} availability."],
            0,
            1,
            contrast_group=contrast,
            add_asr=True,
        )
        add_example(
            rows,
            "feas_ref_time_explicit",
            f"Could you find anything else close to {clock}?",
            [],
            0,
            0,
            add_asr=True,
        )

    day_specs = ["Tuesday", "Friday", "Monday", "Thursday"]
    unresolved_day_times = ["8:40 AM", "12:30 PM", "3:20 PM", "5:10 PM"]
    for i, day in enumerate(day_specs):
        inv = f"train_day_inv_{i}"
        contrast = f"train_day_contrast_{i}"
        add_example(
            rows,
            "feas_ref_day_antecedent",
            "Are there any other openings on that day?",
            [f"{day} is currently fully booked."],
            1,
            0,
            invariance_group=inv,
            contrast_group=contrast,
            add_asr=True,
        )
        add_example(
            rows,
            "feas_ref_day_unresolved",
            "Are there any other openings on that day?",
            [f"The {unresolved_day_times[i]} opening is no longer available."],
            0,
            1,
            contrast_group=contrast,
            add_asr=True,
        )
        add_example(
            rows,
            "feas_ref_day_explicit",
            f"Are there any other openings on {day}?",
            [],
            0,
            0,
            add_asr=True,
        )

    # ------------------------------------------------------------------
    # Resolved option/ordinal/comparative reference contrasts.
    # ------------------------------------------------------------------
    option_contexts = [
        ["I can offer 8:25 AM or 12:40 PM."],
        ["I can offer Tuesday morning or Saturday afternoon."],
        ["I can offer Dr. Moreno Monday or Dr. Kaur Friday."],
    ]
    option_turns = [
        "Would the second one work?",
        "I'll use the earlier one.",
        "Let's take the first choice.",
    ]
    for i, (ctx, turn) in enumerate(zip(option_contexts, option_turns)):
        add_example(
            rows,
            "feas_ref_resolved_option",
            turn,
            ctx,
            1,
            0,
            invariance_group=f"train_option_inv_{i}",
            contrast_group=f"train_option_contrast_{i}",
            add_asr=True,
        )
        add_example(
            rows,
            "feas_ref_unresolved_option",
            turn,
            [],
            0,
            1,
            contrast_group=f"train_option_contrast_{i}",
            add_asr=True,
        )

    # ------------------------------------------------------------------
    # Ambiguity: vague temporal vs explicit anchor.
    # ------------------------------------------------------------------
    temporal_pairs = [
        ("Could we move the appointment a little later?", "Could we move the appointment later on Wednesday afternoon?"),
        ("Could we make the visit somewhat earlier?", "Could we make the visit earlier on Friday morning?"),
        ("Can we shift it a bit later?", "Can we shift it later on Tuesday at 3:30 PM?"),
        ("Could the appointment be a little sooner?", "Could the appointment be earlier on Monday at 10:20 AM?"),
    ]
    for i, (vague, explicit) in enumerate(temporal_pairs):
        contrast = f"train_amb_temporal_contrast_{i}"
        add_example(
            rows,
            "feas_amb_temporal_vague",
            vague,
            [],
            0,
            1,
            invariance_group=f"train_amb_temporal_vague_inv_{i}",
            contrast_group=contrast,
            add_asr=True,
        )
        add_example(
            rows,
            "feas_amb_temporal_explicit",
            explicit,
            [],
            0,
            0,
            contrast_group=contrast,
            add_asr=True,
        )

    # ------------------------------------------------------------------
    # Ambiguity: transaction demonstrative depends on candidate count.
    # ------------------------------------------------------------------
    tx_contexts = [
        ["I can move the appointment or leave the current booking unchanged."],
        ["We can cancel the visit or reschedule it."],
        ["I can keep the existing time or search for another one."],
    ]
    for i, ctx in enumerate(tx_contexts):
        contrast = f"train_tx_contrast_{i}"
        add_example(
            rows,
            "feas_amb_transaction_demonstrative",
            "Should I go ahead with that now?",
            ctx,
            0,
            1,
            contrast_group=contrast,
            add_asr=True,
        )
        add_example(
            rows,
            "feas_amb_transaction_explicit",
            "Should I go ahead with the reschedule now?",
            ctx,
            0,
            0,
            contrast_group=contrast,
            add_asr=True,
        )

    # ------------------------------------------------------------------
    # Ambiguity: vague option evaluation with multiple candidates versus
    # a single candidate / explicit ordinal resolution.
    # ------------------------------------------------------------------
    eval_specs = [
        ("That slot seems acceptable to me.", "I can offer 8:45 AM or 2:20 PM."),
        ("That appointment time sounds fine.", "I can offer 10:15 AM or 4:05 PM."),
        ("That option seems workable.", "I can offer Monday morning or Thursday evening."),
    ]
    for i, (turn, ctx) in enumerate(eval_specs):
        contrast = f"train_eval_contrast_{i}"
        add_example(
            rows,
            "feas_amb_option_eval_multi",
            turn,
            [ctx],
            0,
            1,
            contrast_group=contrast,
            add_asr=True,
        )
        # Single available candidate resolves the demonstrative.
        only = re.sub(r" or .*", ".", ctx)
        add_example(
            rows,
            "feas_amb_option_eval_single",
            turn,
            [only],
            1,
            0,
            contrast_group=contrast,
            add_asr=True,
        )

    # Resolved comparative/ordinal ambiguity negatives.
    resolved = [
        ("I'll choose whichever of those two times is earlier.", ["I can offer 8:10 AM or 1:35 PM."]),
        ("The later of those two times works for me.", ["I can offer 9:25 AM or 3:50 PM."]),
        ("I'll use the second of those two choices.", ["I can offer Tuesday morning or Friday afternoon."]),
        ("Use the first option you mentioned.", ["I can offer Dr. Park Monday or Dr. Olsen Thursday."]),
    ]
    for i, (turn, ctx) in enumerate(resolved):
        add_example(
            rows,
            "feas_amb_resolved_comparative",
            turn,
            ctx,
            1,
            0,
            invariance_group=f"train_resolved_inv_{i}",
            add_asr=True,
        )

    # Broadening/search controls: semantically explicit, not ambiguous.
    for i, turn in enumerate(
        [
            "Can I widen the search to different dates and times?",
            "Please search across other days and time ranges.",
            "Could you look at more dates as well as more times?",
            "Let's broaden the availability search beyond this day and time.",
        ]
    ):
        add_example(
            rows,
            "feas_amb_broad_search_control",
            turn,
            [],
            0,
            0,
            invariance_group=f"train_broad_control_{i}",
            add_asr=True,
        )

    return rows


def build_targeted_validation() -> list[FExample]:
    rows: list[FExample] = []

    # Provider reference: new names, wording and slots.
    for i, (name, pronoun, aux, slot) in enumerate(
        [
            ("Bennett", "she", "Does", "Friday at 11:55 AM"),
            ("Sato", "he", "Does", "Wednesday at 3:15 PM"),
            ("Okafor", "they", "Do", "Monday at 9:45 AM"),
        ]
    ):
        turn = f"{aux} {pronoun} have another time available?"
        add_example(rows, "val_ref_provider", turn,
                    [f"Dr. {name} currently has {slot}."], 1, 0,
                    invariance_group="val_provider_pronoun_invariance", add_asr=False)
    add_example(rows, "val_ref_provider_unresolved",
                "Does she have another time available?",
                ["Friday at 11:55 AM is open."], 0, 1)
    add_example(rows, "val_ref_provider_explicit",
                "Does Dr. Bennett have another time available?", [], 0, 0)

    # Time/day reference.
    add_example(rows, "val_ref_time", "Is there anything else near that time?",
                ["The 10:05 AM opening on Wednesday was just taken."], 1, 0,
                invariance_group="val_time_substitution")
    add_example(rows, "val_ref_time_sub", "Is there anything else near that time?",
                ["The 2:55 PM opening on Monday was just taken."], 1, 0,
                invariance_group="val_time_substitution")
    add_example(rows, "val_ref_time_unresolved", "Is there anything else near that time?",
                ["We are looking at Monday availability."], 0, 1)
    add_example(rows, "val_ref_day", "Could you find something else that day?",
                ["Saturday currently has no remaining openings."], 1, 0,
                invariance_group="val_day_substitution")
    add_example(rows, "val_ref_day_sub", "Could you find something else that day?",
                ["Tuesday currently has no remaining openings."], 1, 0,
                invariance_group="val_day_substitution")
    add_example(rows, "val_ref_day_explicit", "Could you find something else on Saturday?", [], 0, 0)

    # Resolved comparative / ordinal.
    add_example(rows, "val_ref_comparative", "I'll take whichever of those times is earliest.",
                ["I can offer 7:50 AM, 11:20 AM, or 3:05 PM."], 1, 0,
                invariance_group="val_resolved_comparative")
    add_example(rows, "val_ref_ordinal", "Would the first one be okay?",
                ["I can offer Monday afternoon or Friday morning."], 1, 0)

    # Ambiguity temporal contrasts.
    add_example(rows, "val_amb_temporal_vague", "Could we make the appointment somewhat later?", [], 0, 1,
                invariance_group="val_temporal_vague_asr")
    add_example(rows, "val_amb_temporal_explicit", "Could we make the appointment later on Thursday morning?", [], 0, 0)
    add_example(rows, "val_amb_temporal_vague_asr", "could we uh make the appointment somewhat later", [], 0, 1,
                invariance_group="val_temporal_vague_asr")

    # Ambiguity transaction demonstrative.
    txctx = ["I can move the booking or keep the current appointment unchanged."]
    add_example(rows, "val_amb_tx_vague", "Do I continue with that now?", txctx, 0, 1)
    add_example(rows, "val_amb_tx_explicit", "Do I continue with the reschedule now?", txctx, 0, 0)

    # Ambiguity option evaluation and resolved comparison.
    add_example(rows, "val_amb_eval", "That appointment option seems reasonable.",
                ["I can offer 9:40 AM or 2:50 PM."], 0, 1)
    add_example(rows, "val_amb_resolved", "I'll take whichever of those two openings comes first.",
                ["I can offer 8:05 AM or 12:25 PM."], 1, 0,
                invariance_group="val_resolved_comparative")
    add_example(rows, "val_amb_resolved_asr", "i'll uh take whichever of those two openings comes first",
                ["I can offer 8:05 AM or 12:25 PM."], 1, 0,
                invariance_group="val_resolved_comparative")
    add_example(rows, "val_amb_broad", "Could you expand the search across several dates and time windows?", [], 0, 0)

    # Explicit visit-type controls.
    add_example(rows, "val_amb_visit_type", "Will this appointment happen in the clinic or over video?", [], 0, 0,
                invariance_group="val_visit_type_asr")
    add_example(rows, "val_amb_visit_type_asr", "will this uh appointment happen in the clinic or over video", [], 0, 0,
                invariance_group="val_visit_type_asr")

    return rows


def build_metamorphic_probes() -> list[FExample]:
    rows: list[FExample] = []

    # Pronoun/name invariance.
    ctx = ["Dr. Calder has a Thursday afternoon opening."]
    for pronoun, aux in [("she", "Does"), ("he", "Does"), ("they", "Do")]:
        add_example(rows, "probe_ref_provider_pronoun",
                    f"{aux} {pronoun} have any other times nearby?", ctx, 1, 0,
                    invariance_group="probe_provider_pronoun")
    add_example(rows, "probe_ref_provider_name",
                "Does Dr. Calder have any other times nearby?", [], 0, 0)

    # Entity substitution with equivalent discourse relation.
    add_example(rows, "probe_ref_provider_entity_a",
                "Do they have any other times nearby?",
                ["Dr. Vega has a Tuesday evening opening."], 1, 0,
                invariance_group="probe_provider_entity")
    add_example(rows, "probe_ref_provider_entity_b",
                "Do they have any other times nearby?",
                ["Dr. Desai has a Monday morning opening."], 1, 0,
                invariance_group="probe_provider_entity")

    # Time/day substitutions.
    for i, context in enumerate(
        [
            "The 9:10 AM opening on Tuesday is gone.",
            "The 1:50 PM opening on Friday is gone.",
            "The 4:30 PM opening on Wednesday is gone.",
        ]
    ):
        add_example(rows, "probe_ref_time_sub",
                    "Could you look for something else around that time?", [context], 1, 0,
                    invariance_group="probe_time_substitution")
    for day in ["Monday", "Thursday", "Saturday"]:
        add_example(rows, "probe_ref_day_sub",
                    "What other availability is there that day?",
                    [f"{day} has no remaining openings."], 1, 0,
                    invariance_group="probe_day_substitution")

    # Clean/ASR reference invariance.
    add_example(rows, "probe_ref_time_clean",
                "Can you check for another opening close to that time?",
                ["The 2:30 PM slot is unavailable."], 1, 0,
                invariance_group="probe_ref_clean_asr")
    add_example(rows, "probe_ref_time_asr",
                "can you uh check for another opening close to that time",
                ["The 2:30 PM slot is unavailable."], 1, 0,
                invariance_group="probe_ref_clean_asr")

    # Resolved comparative clean/ASR/option-count invariance.
    add_example(rows, "probe_amb_resolved_clean",
                "I'll choose whichever of those two openings is earlier.",
                ["I can offer 8:15 AM or 12:55 PM."], 1, 0,
                invariance_group="probe_resolved_comparative")
    add_example(rows, "probe_amb_resolved_asr",
                "i'll uh choose whichever of those two openings is earlier",
                ["I can offer 8:15 AM or 12:55 PM."], 1, 0,
                invariance_group="probe_resolved_comparative")
    add_example(rows, "probe_amb_resolved_three",
                "I'll choose whichever of those three openings is earliest.",
                ["I can offer 8:15 AM, 11:35 AM, or 3:45 PM."], 1, 0,
                invariance_group="probe_resolved_comparative")

    # Vague option evaluation remains ambiguous under ASR.
    add_example(rows, "probe_amb_eval_clean",
                "That appointment slot seems acceptable.",
                ["I can offer 9:05 AM or 4:15 PM."], 0, 1,
                invariance_group="probe_amb_eval_asr")
    add_example(rows, "probe_amb_eval_asr",
                "that appointment uh slot seems acceptable",
                ["I can offer 9:05 AM or 4:15 PM."], 0, 1,
                invariance_group="probe_amb_eval_asr")

    # Transaction demonstrative versus explicit operation.
    txctx = ["I can cancel the visit or move it to a new time."]
    add_example(rows, "probe_amb_tx_vague", "Should I do that now?", txctx, 0, 1)
    add_example(rows, "probe_amb_tx_explicit", "Should I reschedule the visit now?", txctx, 0, 0)

    # Vague temporal clean/ASR and explicit anchors.
    add_example(rows, "probe_amb_temporal_clean", "Could we move the visit a little sooner?", [], 0, 1,
                invariance_group="probe_amb_temporal_asr")
    add_example(rows, "probe_amb_temporal_asr", "could we uh move the visit a little sooner", [], 0, 1,
                invariance_group="probe_amb_temporal_asr")
    add_example(rows, "probe_amb_temporal_explicit", "Could we move the visit earlier on Wednesday afternoon?", [], 0, 0)
    add_example(rows, "probe_amb_temporal_explicit_sub", "Could we move the visit earlier on Friday morning?", [], 0, 0,
                invariance_group="probe_amb_explicit_sub")
    add_example(rows, "probe_amb_temporal_explicit_sub2", "Could we move the visit earlier on Tuesday evening?", [], 0, 0,
                invariance_group="probe_amb_explicit_sub")

    # Broad search / visit type hard negatives.
    add_example(rows, "probe_amb_broad", "Please widen the search to other dates and other time ranges.", [], 0, 0)
    add_example(rows, "probe_amb_visit", "Is the visit onsite or through a video appointment?", [], 0, 0)

    return rows


# ----------------------------------------------------------------------
# General structural discourse features.  These are features only; they do
# not directly route or override a field decision.
# ----------------------------------------------------------------------
WEEKDAY_RE = re.compile(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
CLOCK_RE = re.compile(r"\b(?:[0-1]?\d(?::[0-5]\d)?\s*(?:a\.?m\.?|p\.?m\.?))\b", re.I)
DAYPART_RE = re.compile(r"\b(?:today|tomorrow|morning|afternoon|evening|noon)\b", re.I)
PROVIDER_RE = re.compile(r"\b(?:dr\.?|doctor|clinician|provider)\b", re.I)
PRONOUN_RE = re.compile(r"\b(?:he|she|they|him|her|them|his|their)\b", re.I)
ANAPHOR_RE = re.compile(r"\b(?:that|this|it|one|ones|those|same)\b", re.I)
OTHER_RE = re.compile(r"\b(?:other|another|else)\b", re.I)
ORDINAL_RE = re.compile(r"\b(?:first|second|third|option\s+one|option\s+two)\b", re.I)
COMPARATIVE_RE = re.compile(r"\b(?:earlier|later|sooner|earliest|latest)\b", re.I)
ACTION_RE = re.compile(r"\b(?:choose|take|use|book|reserve|pick|go\s+with|proceed)\b", re.I)
DEICTIC_TIME_RE = re.compile(r"\b(?:that|same)\s+time\b|\baround\s+that\s+time\b|\bnear\s+that\s+time\b", re.I)
DEICTIC_DAY_RE = re.compile(r"\b(?:that|same)\s+day\b|\bon\s+that\s+day\b", re.I)


def discourse_feature_vector(turn: str, context: Sequence[str]) -> list[float]:
    t = str(turn)
    c = " ".join(str(x) for x in context)
    tl = t.lower()
    cl = c.lower()
    ctx_nonempty = float(bool(c.strip()))
    ctx_or_count = len(re.findall(r"\bor\b", cl))
    ctx_commas = c.count(",")
    ctx_option_strength = min(3.0, float(ctx_or_count + (ctx_commas >= 1))) / 3.0
    ctx_antecedent_strength = sum(
        [
            bool(PROVIDER_RE.search(c)),
            bool(WEEKDAY_RE.search(c)),
            bool(CLOCK_RE.search(c)),
            ctx_option_strength > 0,
        ]
    ) / 4.0
    explicit_turn_anchor = float(bool(WEEKDAY_RE.search(t) or CLOCK_RE.search(t) or DAYPART_RE.search(t)))
    features = [
        min(len(t.split()), 30) / 30.0,
        min(len(context), 4) / 4.0,
        min(len(c.split()), 60) / 60.0,
        float(bool(PRONOUN_RE.search(t))),
        float(bool(ANAPHOR_RE.search(t))),
        float(bool(OTHER_RE.search(t))),
        float(bool(ORDINAL_RE.search(t))),
        float(bool(COMPARATIVE_RE.search(t))),
        float(bool(WEEKDAY_RE.search(t))),
        float(bool(CLOCK_RE.search(t))),
        float(bool(DAYPART_RE.search(t))),
        float(bool(PROVIDER_RE.search(t))),
        float(bool(WEEKDAY_RE.search(c))),
        float(bool(CLOCK_RE.search(c))),
        float(bool(PROVIDER_RE.search(c))),
        ctx_option_strength,
        ctx_nonempty,
        float("?" in t),
        float(bool(ACTION_RE.search(t))),
        float(bool(DEICTIC_TIME_RE.search(t))),
        float(bool(DEICTIC_DAY_RE.search(t))),
        float(bool(PRONOUN_RE.search(t)) and bool(PROVIDER_RE.search(c))),
        float(bool(DEICTIC_TIME_RE.search(t)) and bool(CLOCK_RE.search(c))),
        float(bool(DEICTIC_DAY_RE.search(t)) and bool(WEEKDAY_RE.search(c))),
        float(bool(COMPARATIVE_RE.search(t)) and ctx_option_strength > 0),
        float(bool(ORDINAL_RE.search(t)) and ctx_option_strength > 0),
        float(bool(ANAPHOR_RE.search(t)) and bool(c.strip())),
        explicit_turn_anchor,
        ctx_antecedent_strength,
        float(bool(OTHER_RE.search(t)) and bool(c.strip())),
    ]
    return features


def discourse_features_for_examples(rows: Sequence[FExample]) -> torch.Tensor:
    return torch.tensor(
        [discourse_feature_vector(r.turn, r.context) for r in rows],
        dtype=torch.float32,
    )


def discourse_features_for_cases(cases: Sequence[object]) -> torch.Tensor:
    return torch.tensor(
        [
            discourse_feature_vector(str(c.utterance), tuple(c.context))
            for c in cases
        ],
        dtype=torch.float32,
    )


def gold_examples(rows: Sequence[object]) -> torch.Tensor:
    # Match V5.2 gold_example_tensor exactly: training targets are float32.
    # Evaluation helpers cast to long locally when discrete labels are required.
    return torch.tensor(
        [[float(r.reference), float(r.ambiguity), float(getattr(r, "oos", 0))] for r in rows],
        dtype=torch.float32,
    )


def field_stats(gold: torch.Tensor, pred: torch.Tensor) -> dict[str, float | int]:
    gold = gold.long().view(-1)
    pred = pred.long().view(-1)
    correct = pred.eq(gold)
    active = gold.eq(1)
    negative = ~active
    return {
        "n": int(len(gold)),
        "exact": int(correct.sum().item()),
        "acc": float(correct.float().mean().item()),
        "active_n": int(active.sum().item()),
        "active_acc": float((correct & active).sum().item() / max(1, active.sum().item())),
        "negative_n": int(negative.sum().item()),
        "negative_acc": float((correct & negative).sum().item() / max(1, negative.sum().item())),
    }


def field_gate(stats: dict[str, float | int]) -> bool:
    return bool(
        stats["acc"] >= MIN_VAL_FIELD_ACC
        and stats["active_acc"] >= MIN_VAL_ACTIVE_ACC
        and stats["negative_acc"] >= MIN_VAL_NEGATIVE_ACC
    )


def baseline_regressions(base_pred: torch.Tensor, cand_pred: torch.Tensor, gold: torch.Tensor) -> int:
    base_right = base_pred.long().eq(gold.long())
    cand_right = cand_pred.long().eq(gold.long())
    return int((base_right & (~cand_right)).sum().item())


def baseline_fixes(base_pred: torch.Tensor, cand_pred: torch.Tensor, gold: torch.Tensor) -> int:
    base_right = base_pred.long().eq(gold.long())
    cand_right = cand_pred.long().eq(gold.long())
    return int(((~base_right) & cand_right).sum().item())


def invariant_groups(rows: Sequence[FExample], field: str) -> list[list[int]]:
    j = 0 if field == "reference" else 1
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        group = str(getattr(r, "invariance_group", ""))
        if not group:
            continue
        groups.setdefault(group, []).append(i)
    out = []
    for _, idx in sorted(groups.items()):
        labels = {int(gold_examples([rows[i]])[0, j].item()) for i in idx}
        if len(labels) == 1 and len(idx) >= 2:
            out.append(idx)
    return out


def contrast_groups(rows: Sequence[FExample], field: str) -> list[tuple[list[int], list[int]]]:
    j = 0 if field == "reference" else 1
    by: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        group = str(getattr(r, "contrast_group", ""))
        if group:
            by.setdefault(group, []).append(i)
    out = []
    yg = gold_examples(rows)[:, j]
    for _, idx in sorted(by.items()):
        pos = [i for i in idx if int(yg[i].item()) == 1]
        neg = [i for i in idx if int(yg[i].item()) == 0]
        if pos and neg:
            out.append((pos, neg))
    return out


class ResidualHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.GELU(),
        )
        self.residual = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        r = self.residual(h).squeeze(-1)
        gate = torch.ones_like(r)
        return r, gate


class GatedResidualHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.GELU(),
        )
        self.residual = nn.Linear(128, 1)
        self.gate = nn.Linear(128, 1)
        # Conservative prior: begin with a relatively closed correction gate.
        nn.init.constant_(self.gate.bias, -1.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        r = self.residual(h).squeeze(-1)
        g = torch.sigmoid(self.gate(h).squeeze(-1))
        return r, g


def correction(head: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r, g = head(x)
    return r * g, g


def use_gated(arch: str) -> bool:
    return arch != "context_contrast_residual"


def use_invariance(arch: str) -> bool:
    return arch in {"gated_contrast_invariance", "gated_contrast_invariance_discourse"}


def use_discourse(arch: str) -> bool:
    return arch == "gated_contrast_invariance_discourse"


def build_head(arch: str, input_dim: int) -> nn.Module:
    return GatedResidualHead(input_dim) if use_gated(arch) else ResidualHead(input_dim)


def augment_x(arch: str, base_x: torch.Tensor, discourse_x: torch.Tensor) -> torch.Tensor:
    if use_discourse(arch):
        return torch.cat([base_x, discourse_x], dim=-1)
    return base_x


def invariance_loss(final_margin: torch.Tensor, groups: Sequence[Sequence[int]]) -> torch.Tensor:
    if not groups:
        return final_margin.new_tensor(0.0)
    pieces = []
    for idx in groups:
        v = final_margin[torch.tensor(idx, dtype=torch.long)]
        pieces.append(((v - v.mean()) ** 2).mean())
    return torch.stack(pieces).mean() if pieces else final_margin.new_tensor(0.0)


def contrast_loss(final_margin: torch.Tensor, groups: Sequence[tuple[Sequence[int], Sequence[int]]]) -> torch.Tensor:
    if not groups:
        return final_margin.new_tensor(0.0)
    pieces = []
    for pos_idx, neg_idx in groups:
        pos = final_margin[torch.tensor(pos_idx, dtype=torch.long)].mean()
        neg = final_margin[torch.tensor(neg_idx, dtype=torch.long)].mean()
        pieces.append(torch.relu(final_margin.new_tensor(1.0) - (pos - neg)))
    return torch.stack(pieces).mean() if pieces else final_margin.new_tensor(0.0)


def train_reference_epoch(
    arch: str,
    head: nn.Module,
    opt: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    margin: torch.Tensor,
    base_pred: torch.Tensor,
    inv_groups: Sequence[Sequence[int]],
    contrast_pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> dict[str, float]:
    head.train()
    opt.zero_grad()
    corr, gate = correction(head, x)
    logits = margin + corr
    pos = y.sum()
    neg = len(y) - pos
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(0.5, 4.0)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=pos_weight)
    corr_reg = corr.pow(2).mean()
    base_correct = base_pred.long().eq(y.long())
    trust = corr[base_correct].pow(2).mean() if base_correct.any() else corr.new_tensor(0.0)
    c_loss = contrast_loss(logits, contrast_pairs) if use_gated(arch) else logits.new_tensor(0.0)
    i_loss = invariance_loss(logits, inv_groups) if use_invariance(arch) else logits.new_tensor(0.0)
    gate_reg = gate[base_correct].mean() if use_gated(arch) and base_correct.any() else gate.new_tensor(0.0)
    loss = bce + 0.004 * corr_reg + 0.020 * trust + 0.10 * c_loss + 0.035 * i_loss + 0.003 * gate_reg
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    opt.step()
    return {
        "loss": float(loss.item()),
        "bce": float(bce.item()),
        "contrast": float(c_loss.item()),
        "invariance": float(i_loss.item()),
    }


def train_ambiguity_epoch(
    arch: str,
    promote: nn.Module,
    suppress: nn.Module,
    opt: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    margin: torch.Tensor,
    base_pred: torch.Tensor,
    inv_groups: Sequence[Sequence[int]],
    contrast_pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> dict[str, float]:
    promote.train(); suppress.train(); opt.zero_grad()
    pcorr, pgate = correction(promote, x)
    scorr, sgate = correction(suppress, x)
    routed_corr = torch.where(base_pred.eq(0), pcorr, scorr)
    routed_gate = torch.where(base_pred.eq(0), pgate, sgate)
    logits = margin + routed_corr
    pos = y.sum(); neg = len(y) - pos
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(0.5, 4.0)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=pos_weight)
    corr_reg = (pcorr.pow(2).mean() + scorr.pow(2).mean()) / 2.0
    base_correct = base_pred.long().eq(y.long())
    trust = routed_corr[base_correct].pow(2).mean() if base_correct.any() else routed_corr.new_tensor(0.0)
    c_loss = contrast_loss(logits, contrast_pairs) if use_gated(arch) else logits.new_tensor(0.0)
    i_loss = invariance_loss(logits, inv_groups) if use_invariance(arch) else logits.new_tensor(0.0)
    gate_reg = routed_gate[base_correct].mean() if use_gated(arch) and base_correct.any() else routed_gate.new_tensor(0.0)
    loss = bce + 0.004 * corr_reg + 0.020 * trust + 0.10 * c_loss + 0.035 * i_loss + 0.003 * gate_reg
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(promote.parameters()) + list(suppress.parameters()), 1.0)
    opt.step()
    return {
        "loss": float(loss.item()),
        "bce": float(bce.item()),
        "contrast": float(c_loss.item()),
        "invariance": float(i_loss.item()),
    }


def eval_correction(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    head.eval()
    with torch.no_grad():
        c, _ = correction(head, x)
    return c.detach().cpu()


def eval_amb_corrections(promote: nn.Module, suppress: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    promote.eval(); suppress.eval()
    with torch.no_grad():
        pc, _ = correction(promote, x)
        sc, _ = correction(suppress, x)
    return pc.detach().cpu(), sc.detach().cpu()


def probe_exact_and_invariant(rows: Sequence[FExample], pred: torch.Tensor, field: str) -> tuple[bool, dict[str, object]]:
    j = 0 if field == "reference" else 1
    gold = gold_examples(rows)[:, j]
    exact = pred.long().eq(gold.long())
    inv_fail = []
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r.invariance_group:
            groups.setdefault(r.invariance_group, []).append(i)
    for name, idx in sorted(groups.items()):
        vals = {int(pred[i].item()) for i in idx}
        expected = {int(gold[i].item()) for i in idx}
        if len(expected) == 1 and (len(vals) != 1 or vals != expected):
            inv_fail.append({"group": name, "pred": sorted(vals), "expected": sorted(expected)})
    return bool(exact.all().item() and not inv_fail), {
        "exact": f"{int(exact.sum().item())}/{len(rows)}",
        "fail_ids": [rows[i].family for i in range(len(rows)) if not bool(exact[i].item())],
        "invariance_failures": inv_fail,
    }


def search_reference_state(
    epoch: int,
    val_corr: torch.Tensor,
    est_corr: torch.Tensor,
    exp_corr: torch.Tensor,
    probe_corr: torch.Tensor,
    val_margin: torch.Tensor,
    est_margin: torch.Tensor,
    exp_margin: torch.Tensor,
    probe_margin: torch.Tensor,
    val_gold: torch.Tensor,
    est_gold: torch.Tensor,
    exp_gold: torch.Tensor,
    probe_rows: Sequence[FExample],
    val_base: torch.Tensor,
    est_base: torch.Tensor,
    exp_base: torch.Tensor,
    targeted_mask: torch.Tensor,
) -> list[dict[str, object]]:
    out = []
    j = 0
    for scale in SCALES:
        vp = (val_margin[:, j] + scale * val_corr >= 0).long()
        ep = (est_margin[:, j] + scale * est_corr >= 0).long()
        xp = (exp_margin[:, j] + scale * exp_corr >= 0).long()
        pp = (probe_margin[:, j] + scale * probe_corr >= 0).long()
        stats = field_stats(val_gold[:, j], vp)
        if not field_gate(stats):
            continue
        if targeted_mask.any() and not bool(vp[targeted_mask].eq(val_gold[targeted_mask, j]).all().item()):
            continue
        est_reg = baseline_regressions(est_base[:, j], ep, est_gold[:, j])
        exp_reg = baseline_regressions(exp_base[:, j], xp, exp_gold[:, j])
        if est_reg or exp_reg:
            continue
        probe_ok, probe_summary = probe_exact_and_invariant(probe_rows, pp, "reference")
        if not probe_ok:
            continue
        out.append({
            "epoch": epoch,
            "scale": scale,
            "val_stats": stats,
            "est_fixes": baseline_fixes(est_base[:, j], ep, est_gold[:, j]),
            "exp_fixes": baseline_fixes(exp_base[:, j], xp, exp_gold[:, j]),
            "probe": probe_summary,
            "val_pred": vp,
            "est_pred": ep,
            "exp_pred": xp,
            "probe_pred": pp,
        })
    return out


def zero_reg_scales_for_direction(
    corrections: torch.Tensor,
    margin: torch.Tensor,
    base: torch.Tensor,
    gold: torch.Tensor,
    *,
    base_value: int,
) -> list[float]:
    mask = base.eq(base_value) & base.eq(gold)
    if not mask.any():
        return list(SCALES)
    out = []
    for scale in SCALES:
        pred = (margin + scale * corrections >= 0).long()
        if bool(pred[mask].eq(gold[mask]).all().item()):
            out.append(scale)
    return out


def search_ambiguity_state(
    epoch: int,
    val_pc: torch.Tensor,
    val_sc: torch.Tensor,
    est_pc: torch.Tensor,
    est_sc: torch.Tensor,
    exp_pc: torch.Tensor,
    exp_sc: torch.Tensor,
    probe_pc: torch.Tensor,
    probe_sc: torch.Tensor,
    val_margin: torch.Tensor,
    est_margin: torch.Tensor,
    exp_margin: torch.Tensor,
    probe_margin: torch.Tensor,
    val_gold: torch.Tensor,
    est_gold: torch.Tensor,
    exp_gold: torch.Tensor,
    probe_rows: Sequence[FExample],
    val_base: torch.Tensor,
    est_base: torch.Tensor,
    exp_base: torch.Tensor,
    probe_base: torch.Tensor,
    targeted_mask: torch.Tensor,
    frozen_oos_val: torch.Tensor,
    frozen_oos_est: torch.Tensor,
    frozen_oos_exp: torch.Tensor,
    frozen_oos_probe: torch.Tensor,
) -> list[dict[str, object]]:
    j = 1
    # Pre-prune scales that can already regress a baseline-right case in the
    # corresponding base-ambiguity direction.  Established and exposed must
    # both be safe before any Cartesian join.
    promote_est = zero_reg_scales_for_direction(est_pc, est_margin[:, j], est_base[:, j], est_gold[:, j], base_value=0)
    promote_exp = zero_reg_scales_for_direction(exp_pc, exp_margin[:, j], exp_base[:, j], exp_gold[:, j], base_value=0)
    suppress_est = zero_reg_scales_for_direction(est_sc, est_margin[:, j], est_base[:, j], est_gold[:, j], base_value=1)
    suppress_exp = zero_reg_scales_for_direction(exp_sc, exp_margin[:, j], exp_base[:, j], exp_gold[:, j], base_value=1)
    promote_scales = sorted(set(promote_est) & set(promote_exp))
    suppress_scales = sorted(set(suppress_est) & set(suppress_exp))
    if not promote_scales or not suppress_scales:
        return []

    out = []
    for ps in promote_scales:
        vprom = val_margin[:, j] + ps * val_pc
        eprom = est_margin[:, j] + ps * est_pc
        xprom = exp_margin[:, j] + ps * exp_pc
        pprom = probe_margin[:, j] + ps * probe_pc
        for ss in suppress_scales:
            vm = torch.where(val_base[:, j].eq(0), vprom, val_margin[:, j] + ss * val_sc)
            em = torch.where(est_base[:, j].eq(0), eprom, est_margin[:, j] + ss * est_sc)
            xm = torch.where(exp_base[:, j].eq(0), xprom, exp_margin[:, j] + ss * exp_sc)
            pm = torch.where(probe_base[:, j].eq(0), pprom, probe_margin[:, j] + ss * probe_sc)
            vp = (vm >= 0).long(); ep = (em >= 0).long(); xp = (xm >= 0).long(); pp = (pm >= 0).long()
            # Exact frozen OOS -> ambiguity ontology projection.
            vp[frozen_oos_val.eq(1)] = 1
            ep[frozen_oos_est.eq(1)] = 1
            xp[frozen_oos_exp.eq(1)] = 1
            pp[frozen_oos_probe.eq(1)] = 1
            stats = field_stats(val_gold[:, j], vp)
            if not field_gate(stats):
                continue
            if targeted_mask.any() and not bool(vp[targeted_mask].eq(val_gold[targeted_mask, j]).all().item()):
                continue
            est_reg = baseline_regressions(est_base[:, j], ep, est_gold[:, j])
            exp_reg = baseline_regressions(exp_base[:, j], xp, exp_gold[:, j])
            if est_reg or exp_reg:
                continue
            probe_ok, probe_summary = probe_exact_and_invariant(probe_rows, pp, "ambiguity")
            if not probe_ok:
                continue
            out.append({
                "epoch": epoch,
                "promote_scale": ps,
                "suppress_scale": ss,
                "val_stats": stats,
                "est_fixes": baseline_fixes(est_base[:, j], ep, est_gold[:, j]),
                "exp_fixes": baseline_fixes(exp_base[:, j], xp, exp_gold[:, j]),
                "probe": probe_summary,
                "val_pred": vp,
                "est_pred": ep,
                "exp_pred": xp,
                "probe_pred": pp,
            })
    return out


def choose_best(rows: Sequence[dict[str, object]], field: str) -> dict[str, object] | None:
    if not rows:
        return None
    if field == "reference":
        return sorted(
            rows,
            key=lambda r: (
                r["val_stats"]["acc"],
                r["val_stats"]["active_acc"],
                r["val_stats"]["negative_acc"],
                r["est_fixes"] + r["exp_fixes"],
                -float(r["scale"]),
                -int(r["epoch"]),
            ),
            reverse=True,
        )[0]
    return sorted(
        rows,
        key=lambda r: (
            r["val_stats"]["acc"],
            r["val_stats"]["active_acc"],
            r["val_stats"]["negative_acc"],
            r["est_fixes"] + r["exp_fixes"],
            -(float(r["promote_scale"]) + float(r["suppress_scale"])),
            -int(r["epoch"]),
        ),
        reverse=True,
    )[0]


def case_rows_to_examples(cases: Sequence[object], v52) -> list[FExample]:
    gold = v52.gold_case_tensor(cases)
    return [
        FExample(
            family=str(c.case_id),
            turn=str(c.utterance),
            context=tuple(c.context),
            reference=int(gold[i, 0].item()),
            ambiguity=int(gold[i, 1].item()),
            oos=int(gold[i, 2].item()),
        )
        for i, c in enumerate(cases)
    ]


def reconstruct_frozen_oos(
    v52,
    oos: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_margin: torch.Tensor,
    train_rows: Sequence[object],
) -> nn.Module:
    """Train the already-exactly-initialized V5.2 OOS head to epoch 52.

    IMPORTANT: the caller must initialize DirectionalFactorizedResidual at the
    exact V5.2 replay point (after original train/val/historical feature capture).
    Re-seeding or constructing the head here would change the authoritative
    initialization because Phase8A feature capture can consume torch RNG.
    """
    indices = torch.tensor(
        [i for i, row in enumerate(train_rows) if not row.family.startswith("v3_")],
        dtype=torch.long,
    )
    x = train_x[indices]
    y = train_y[indices, 2]
    margin = train_margin[indices, 2]
    pos = y.sum(); neg = len(y) - pos
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(0.5, 4.0)
    opt = torch.optim.AdamW(oos.parameters(), lr=v52.LR, weight_decay=1e-3)

    # Reproduce the V5.2 OOS DataLoader shuffle sequence explicitly by using
    # the same generator and batch size.  No other head can influence OOS.
    ds = v52.TensorDS(x, train_y[indices], train_margin[indices])
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=v52.BATCH,
        shuffle=True,
        generator=torch.Generator().manual_seed(v52.SEED),
    )
    for epoch in range(1, OOS_EPOCH + 1):
        oos.train()
        for xb, yb, mb in loader:
            opt.zero_grad()
            residual = oos(xb)
            logits = mb[:, 2] + residual
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, yb[:, 2], pos_weight=pos_weight
            )
            reg = residual.pow(2).mean()
            loss = (bce / 3.0) + 0.012 * (reg / 4.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(oos.parameters(), 1.0)
            opt.step()
    oos.eval()
    for p in oos.parameters():
        p.requires_grad = False
    return oos


def frozen_oos_pred(oos: nn.Module, x: torch.Tensor, margin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    oos.eval()
    with torch.no_grad():
        residual = oos(x).detach().cpu()
    candidate_margin = margin[:, 2].cpu() + OOS_SCALE * residual
    pred = (candidate_margin >= 0).long()
    return pred, candidate_margin


def filter_original_synthetic(v52, train, val):
    blocked, blocked_file_count = v52.benchmark_surfaces()
    train_f = [r for r in train if v52.normalized_surface(r.context, r.turn) not in blocked]
    val_f = [r for r in val if v52.normalized_surface(r.context, r.turn) not in blocked]
    return train_f, val_f, blocked, blocked_file_count


def validate_fresh_novelty(v52, train_rows: Sequence[FExample], val_rows: Sequence[FExample], probe_rows: Sequence[FExample], blocked: set[str]) -> None:
    train_surface = {v52.normalized_surface(r.context, r.turn) for r in train_rows}
    val_surface = {v52.normalized_surface(r.context, r.turn) for r in val_rows}
    probe_surface = {v52.normalized_surface(r.context, r.turn) for r in probe_rows}
    benchmark_train = sorted(s for s in train_surface if s in blocked)
    benchmark_val = sorted(s for s in val_surface if s in blocked)
    benchmark_probe = sorted(s for s in probe_surface if s in blocked)
    if benchmark_train:
        raise RuntimeError(f"Fresh gradient rows overlap benchmark surfaces: {benchmark_train[:10]}")
    if benchmark_val or benchmark_probe:
        raise RuntimeError(
            "Fresh validation/probe overlap benchmark surfaces: "
            f"val={benchmark_val[:10]} probe={benchmark_probe[:10]}"
        )
    tv = sorted(train_surface & val_surface)
    tp = sorted(train_surface & probe_surface)
    vp = sorted(val_surface & probe_surface)
    if tv or tp or vp:
        raise RuntimeError(
            "Fresh train/validation/probe exact surface overlap: "
            f"train_val={tv[:10]} train_probe={tp[:10]} val_probe={vp[:10]}"
        )


def make_training_sets(v52, original_train: Sequence[object], fresh_train: Sequence[FExample]) -> tuple[list[object], list[object]]:
    ref_original = [
        r for r in original_train
        if not r.family.startswith("v3_")
        and not r.family.startswith("v51_oos_")
        and r.family not in REFERENCE_CONFLICT_FAMILIES_REMOVED
    ]
    amb_original = [
        r for r in original_train
        if not r.family.startswith("v51_oos_")
        and r.family not in AMBIGUITY_CONFLICT_FAMILIES_REMOVED
    ]
    return ref_original + list(fresh_train), amb_original + list(fresh_train)


def examples_runtime(v52, rows: Sequence[object]):
    return [v52.base.RuntimeTurn(context=tuple(r.context), utterance=str(r.turn)) for r in rows]


def capture(v52, model, tok, rows: Sequence[object], thresholds):
    return v52.capture_features(model, tok, examples_runtime(v52, rows), thresholds)


def run_seed(
    arch: str,
    seed: int,
    v52,
    thresholds,
    tensors: dict[str, object],
) -> dict[str, object]:
    random.seed(seed); torch.manual_seed(seed)

    train_ref_x = augment_x(arch, tensors["ref_train_x"], tensors["ref_train_disc"])
    train_amb_x = augment_x(arch, tensors["amb_train_x"], tensors["amb_train_disc"])
    val_x = augment_x(arch, tensors["val_x"], tensors["val_disc"])
    est_x = augment_x(arch, tensors["est_x"], tensors["est_disc"])
    exp_x = augment_x(arch, tensors["exp_x"], tensors["exp_disc"])
    probe_x = augment_x(arch, tensors["probe_x"], tensors["probe_disc"])

    ref_head = build_head(arch, train_ref_x.shape[1])
    amb_promote = build_head(arch, train_amb_x.shape[1])
    amb_suppress = build_head(arch, train_amb_x.shape[1])

    ref_opt = torch.optim.AdamW(ref_head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    amb_opt = torch.optim.AdamW(
        list(amb_promote.parameters()) + list(amb_suppress.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    ref_y = tensors["ref_train_y"][:, 0]
    amb_y = tensors["amb_train_y"][:, 1]
    ref_margin = tensors["ref_train_margin"][:, 0]
    amb_margin = tensors["amb_train_margin"][:, 1]
    ref_base = tensors["ref_train_base"][:, 0]
    amb_base = tensors["amb_train_base"][:, 1]

    ref_inv = invariant_groups(tensors["ref_train_rows"], "reference")
    amb_inv = invariant_groups(tensors["amb_train_rows"], "ambiguity")
    ref_contrast = contrast_groups(tensors["ref_train_rows"], "reference")
    amb_contrast = contrast_groups(tensors["amb_train_rows"], "ambiguity")

    reference_strong: list[dict[str, object]] = []
    ambiguity_strong: list[dict[str, object]] = []
    best_ref_reg = [10**9, 10**9]
    best_amb_reg = [10**9, 10**9]

    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        rl = train_reference_epoch(
            arch, ref_head, ref_opt, train_ref_x, ref_y, ref_margin, ref_base, ref_inv, ref_contrast
        )
        al = train_ambiguity_epoch(
            arch, amb_promote, amb_suppress, amb_opt, train_amb_x, amb_y, amb_margin, amb_base, amb_inv, amb_contrast
        )

        val_ref_corr = eval_correction(ref_head, val_x)
        est_ref_corr = eval_correction(ref_head, est_x)
        exp_ref_corr = eval_correction(ref_head, exp_x)
        probe_ref_corr = eval_correction(ref_head, probe_x)
        rstates = search_reference_state(
            epoch,
            val_ref_corr, est_ref_corr, exp_ref_corr, probe_ref_corr,
            tensors["val_margin"], tensors["est_margin"], tensors["exp_margin"], tensors["probe_margin"],
            tensors["val_gold"], tensors["est_gold"], tensors["exp_gold"], tensors["probe_rows"],
            tensors["val_base"], tensors["est_base"], tensors["exp_base"], tensors["targeted_ref_mask"],
        )
        reference_strong.extend(rstates)

        vpc, vsc = eval_amb_corrections(amb_promote, amb_suppress, val_x)
        epc, esc = eval_amb_corrections(amb_promote, amb_suppress, est_x)
        xpc, xsc = eval_amb_corrections(amb_promote, amb_suppress, exp_x)
        ppc, psc = eval_amb_corrections(amb_promote, amb_suppress, probe_x)
        astates = search_ambiguity_state(
            epoch,
            vpc, vsc, epc, esc, xpc, xsc, ppc, psc,
            tensors["val_margin"], tensors["est_margin"], tensors["exp_margin"], tensors["probe_margin"],
            tensors["val_gold"], tensors["est_gold"], tensors["exp_gold"], tensors["probe_rows"],
            tensors["val_base"], tensors["est_base"], tensors["exp_base"], tensors["probe_base"],
            tensors["targeted_amb_mask"],
            tensors["oos_val"], tensors["oos_est"], tensors["oos_exp"], tensors["oos_probe"],
        )
        ambiguity_strong.extend(astates)

        # Lightweight diagnostics: minimum regression counts over legal scales.
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            # Reference minimum regressions among scale choices (irrespective of fresh gate).
            rmins = [10**9, 10**9]
            for s in SCALES:
                ep = (tensors["est_margin"][:, 0] + s * est_ref_corr >= 0).long()
                xp = (tensors["exp_margin"][:, 0] + s * exp_ref_corr >= 0).long()
                rmins[0] = min(rmins[0], baseline_regressions(tensors["est_base"][:, 0], ep, tensors["est_gold"][:, 0]))
                rmins[1] = min(rmins[1], baseline_regressions(tensors["exp_base"][:, 0], xp, tensors["exp_gold"][:, 0]))
            best_ref_reg[0] = min(best_ref_reg[0], rmins[0]); best_ref_reg[1] = min(best_ref_reg[1], rmins[1])

            # Ambiguity direction-wise lower bound on regressions.
            p_est = zero_reg_scales_for_direction(epc, tensors["est_margin"][:,1], tensors["est_base"][:,1], tensors["est_gold"][:,1], base_value=0)
            p_exp = zero_reg_scales_for_direction(xpc, tensors["exp_margin"][:,1], tensors["exp_base"][:,1], tensors["exp_gold"][:,1], base_value=0)
            s_est = zero_reg_scales_for_direction(esc, tensors["est_margin"][:,1], tensors["est_base"][:,1], tensors["est_gold"][:,1], base_value=1)
            s_exp = zero_reg_scales_for_direction(xsc, tensors["exp_margin"][:,1], tensors["exp_base"][:,1], tensors["exp_gold"][:,1], base_value=1)
            amb_zero_possible = bool(set(p_est)&set(p_exp) and set(s_est)&set(s_exp))

            print(
                f"ARCH={arch} SEED={seed} epoch={epoch:02d}",
                f"ref_loss={rl['loss']:.4f}",
                f"amb_loss={al['loss']:.4f}",
                f"ref_strong_total={len(reference_strong)}",
                f"amb_strong_total={len(ambiguity_strong)}",
                f"ref_min_regs_est_exp={tuple(rmins)}",
                f"amb_zero_reg_direction_scales={'YES' if amb_zero_possible else 'NO'}",
                flush=True,
            )

    best_ref = choose_best(reference_strong, "reference")
    best_amb = choose_best(ambiguity_strong, "ambiguity")

    joint_pass = False
    joint_summary: dict[str, object] = {}
    if best_ref is not None and best_amb is not None:
        val_pred = tensors["val_base"].clone().long()
        val_pred[:, 0] = best_ref["val_pred"]
        val_pred[:, 1] = best_amb["val_pred"]
        val_pred[:, 2] = tensors["oos_val"]
        joint_correct = val_pred.eq(tensors["val_gold"]).all(dim=1)
        joint_acc = float(joint_correct.float().mean().item())
        targeted = tensors["targeted_any_mask"]
        targeted_joint = bool(joint_correct[targeted].all().item()) if targeted.any() else True
        joint_pass = bool(joint_acc >= MIN_VAL_JOINT_ACC and targeted_joint)
        joint_summary = {
            "fresh_joint": f"{int(joint_correct.sum().item())}/{len(joint_correct)}",
            "fresh_joint_acc": joint_acc,
            "targeted_joint_exact": targeted_joint,
        }

    return {
        "architecture": arch,
        "seed": seed,
        "reference_strong_states": len(reference_strong),
        "ambiguity_strong_states": len(ambiguity_strong),
        "best_reference": None if best_ref is None else {
            k: v for k, v in best_ref.items() if k not in {"val_pred", "est_pred", "exp_pred", "probe_pred"}
        },
        "best_ambiguity": None if best_amb is None else {
            k: v for k, v in best_amb.items() if k not in {"val_pred", "est_pred", "exp_pred", "probe_pred"}
        },
        "joint_pass": joint_pass,
        "joint": joint_summary,
        "seed_pass": bool(best_ref is not None and best_amb is not None and joint_pass),
        "best_reference_min_regressions_seen": tuple(best_ref_reg),
        "wall_s": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v52-source", default=None)
    args = parser.parse_args()

    print("========== PHASE 7C BOUNDED ARCHITECTURE FEASIBILITY GATE ==========")
    print("telephony=DISABLED")
    print("runtime_wiring=NO")
    print("candidate_artifact_write=NO")
    print("this_is_v5_3=NO")
    print("benchmark_cases_used_for_gradients=NO")
    print("exposed120_used_for_gradients=NO")
    print("development_gold_role=SELECTION_AND_STABILITY_ONLY")
    print("architectures=", ARCHITECTURES)
    print("seeds=", SEEDS)
    print("epochs_per_seed=", EPOCHS)
    print("hard_stop_if_no_architecture_passes_all_seeds=YES")

    v52_path = resolve_v52_source(args.v52_source)
    v52_sha = sha256_file(v52_path)
    print("v52_source=", v52_path)
    print("v52_source_sha256=", v52_sha)
    if v52_sha != EXPECTED_V52_SHA256:
        raise RuntimeError(
            "V5.2 source drift detected. Expected "
            f"{EXPECTED_V52_SHA256}, got {v52_sha}"
        )

    v52 = load_mod("phase7c_feasibility_v52", v52_path)
    source_before = v52.base.source_snapshot()
    source_hashes_before = {
        "v52_source": sha256_file(v52_path),
        "phase7c_source": sha256_file(v52.base.P7C),
        "phase7c_checkpoint": sha256_file(v52.base.A7C),
        "phase8a_a8a3": sha256_file(v52.base.A8A3),
        "phase8a_hierarchical": sha256_file(v52.c8a.g.HIER_ARTIFACT),
    }
    artifact_before = {
        "exists": bool(v52.OUT.exists()),
        "sha256": sha256_file(v52.OUT) if v52.OUT.exists() else None,
    }
    print("source_checkpoint_hashes_before=", source_hashes_before)
    print("v52_candidate_artifact_before=", artifact_before)

    # Replay-sensitive sequence begins here. Mirror V5.2/V3 exactly.
    random.seed(v52.SEED)
    torch.manual_seed(v52.SEED)
    ck, model, tok = v52.load_current_model()
    thresholds = {f: float(ck["thresholds"][f]) for f in v52.FIELDS}
    print("thresholds=", thresholds)

    original_train, original_val = v52.build_synthetic()
    original_train, original_val, blocked, blocked_files = filter_original_synthetic(v52, original_train, original_val)
    fresh_train = build_feasibility_training()
    targeted_val = build_targeted_validation()
    probes = build_metamorphic_probes()
    validate_fresh_novelty(v52, fresh_train, targeted_val, probes, blocked)
    print("blocked_jsonl_files=", blocked_files)
    print("original_train_filtered=", len(original_train))
    print("original_val_filtered=", len(original_val))
    print("fresh_feasibility_gradient_rows=", len(fresh_train))
    print("fresh_targeted_validation_rows=", len(targeted_val))
    print("fresh_metamorphic_probe_rows=", len(probes))

    ref_train_rows, amb_train_rows = make_training_sets(v52, original_train, fresh_train)
    print("reference_conflicting_synthetic_family_rows_removed=", sum(1 for r in original_train if r.family in REFERENCE_CONFLICT_FAMILIES_REMOVED))
    print("ambiguity_conflicting_synthetic_family_rows_removed=", sum(1 for r in original_train if r.family in AMBIGUITY_CONFLICT_FAMILIES_REMOVED))
    print("reference_train_rows=", len(ref_train_rows))
    print("ambiguity_train_rows=", len(amb_train_rows))

    groups, exposed_cases = v52.load_groups()
    established_cases = [c for _, cases in groups for c in cases]
    if len(established_cases) != 1146 or len(exposed_cases) != 120:
        raise RuntimeError(
            f"Development corpus cardinality drift: established={len(established_cases)} exposed={len(exposed_cases)}"
        )
    print("established_cases=", len(established_cases))
    print("exposed_cases=", len(exposed_cases))

    # Combined fresh validation = exact V5.2 fresh validation + new targeted validation.
    val_rows: list[object] = list(original_val) + list(targeted_val)
    targeted_start = len(original_val)
    targeted_ref_mask = torch.zeros(len(val_rows), dtype=torch.bool)
    targeted_amb_mask = torch.zeros(len(val_rows), dtype=torch.bool)
    targeted_any_mask = torch.zeros(len(val_rows), dtype=torch.bool)
    targeted_ref_mask[targeted_start:] = True
    targeted_amb_mask[targeted_start:] = True
    targeted_any_mask[targeted_start:] = True

    print()
    print("========== EXACT V5.2 OOS REPLAY-SENSITIVE INITIALIZATION ==========")
    # V5.2/V3 authoritative ordering:
    #   load model -> capture original train -> capture original val ->
    #   capture historical133 -> initialize DirectionalFactorizedResidual.
    # No extra capture may occur before residual-head initialization.
    orig_x, _, orig_margin, orig_base, _, encoder_name = capture(v52, model, tok, original_train, thresholds)
    orig_y = v52.gold_example_tensor(original_train)
    oos_orig_val_x, _, oos_orig_val_margin, oos_orig_val_base, _, _ = capture(
        v52, model, tok, original_val, thresholds
    )
    oos_orig_val_gold = v52.gold_example_tensor(original_val).long()
    replay_historical = list(v52.load_semanticlab_cases())
    if len(replay_historical) != 133:
        raise RuntimeError(f"Historical replay corpus drift: {len(replay_historical)} != 133")
    # Capture is intentionally retained even though its tensors are not used:
    # it is part of the exact RNG-sensitive V5.2 initialization sequence.
    v52.capture_features(
        model, tok, v52.runtime_for_cases(replay_historical), thresholds
    )
    replay_head = v52.DirectionalFactorizedResidual(orig_x.shape[1])
    oos = reconstruct_frozen_oos(
        v52, replay_head.oos, orig_x, orig_y, orig_margin, original_train
    )
    print("oos_replay_head_initialized_at_authoritative_v52_point=YES")

    print()
    print("========== CAPTURING ADDITIONAL FEASIBILITY FEATURES ==========")
    # Everything below occurs only AFTER authoritative OOS head initialization.
    ref_x, _, ref_margin, ref_base, _, _ = capture(v52, model, tok, ref_train_rows, thresholds)
    amb_x, _, amb_margin, amb_base, _, _ = capture(v52, model, tok, amb_train_rows, thresholds)
    val_x, _, val_margin, val_base, _, _ = capture(v52, model, tok, val_rows, thresholds)
    est_x, _, est_margin, est_base, _, _ = v52.capture_features(model, tok, v52.runtime_for_cases(established_cases), thresholds)
    exp_x, _, exp_margin, exp_base, _, _ = v52.capture_features(model, tok, v52.runtime_for_cases(exposed_cases), thresholds)
    probe_x, _, probe_margin, probe_base, _, _ = capture(v52, model, tok, probes, thresholds)

    val_gold = gold_examples(val_rows)
    est_gold = v52.gold_case_tensor(established_cases).long()
    exp_gold = v52.gold_case_tensor(exposed_cases).long()
    probe_gold = gold_examples(probes)
    ref_y = gold_examples(ref_train_rows)
    amb_y = gold_examples(amb_train_rows)

    print("encoder=", encoder_name)
    print("base_feature_dim=", int(ref_x.shape[1]))
    print("discourse_feature_dim=", int(discourse_features_for_examples(fresh_train).shape[1]))
    print("combined_fresh_validation_rows=", len(val_rows))

    # Discourse feature tensors.
    ref_disc = discourse_features_for_examples(ref_train_rows)
    amb_disc = discourse_features_for_examples(amb_train_rows)
    val_disc = torch.cat([
        discourse_features_for_examples([
            FExample(r.family, r.turn, tuple(r.context), int(r.reference), int(r.ambiguity), int(r.oos))
            for r in original_val
        ]),
        discourse_features_for_examples(targeted_val),
    ], dim=0)
    est_disc = discourse_features_for_cases(established_cases)
    exp_disc = discourse_features_for_cases(exposed_cases)
    probe_disc = discourse_features_for_examples(probes)

    print()
    print("========== VERIFYING FROZEN OOS ==========")
    oos_orig_val_pred, _ = frozen_oos_pred(oos, oos_orig_val_x, oos_orig_val_margin)
    oos_val, _ = frozen_oos_pred(oos, val_x, val_margin)
    oos_est, _ = frozen_oos_pred(oos, est_x, est_margin)
    oos_exp, _ = frozen_oos_pred(oos, exp_x, exp_margin)
    oos_probe, _ = frozen_oos_pred(oos, probe_x, probe_margin)

    oos_checks = {
        "original_val_exact": f"{int(oos_orig_val_pred.eq(oos_orig_val_gold[:,2]).sum().item())}/{len(original_val)}",
        "established_exact": f"{int(oos_est.eq(est_gold[:,2]).sum().item())}/{len(established_cases)}",
        "exposed_exact": f"{int(oos_exp.eq(exp_gold[:,2]).sum().item())}/{len(exposed_cases)}",
        "established_regressions": baseline_regressions(est_base[:,2], oos_est, est_gold[:,2]),
        "exposed_regressions": baseline_regressions(exp_base[:,2], oos_exp, exp_gold[:,2]),
        "established_fixes": baseline_fixes(est_base[:,2], oos_est, est_gold[:,2]),
        "exposed_fixes": baseline_fixes(exp_base[:,2], oos_exp, exp_gold[:,2]),
        "new_val_in_domain_exact": f"{int(oos_val.eq(val_gold[:,2]).sum().item())}/{len(val_rows)}",
        "probe_in_domain_exact": f"{int(oos_probe.eq(probe_gold[:,2]).sum().item())}/{len(probes)}",
    }
    print("frozen_oos_epoch_scale=", (OOS_EPOCH, OOS_SCALE))
    print("frozen_oos_checks=", oos_checks)
    expected_oos_ok = all([
        bool(oos_orig_val_pred.eq(oos_orig_val_gold[:,2]).all().item()),
        bool(oos_est.eq(est_gold[:,2]).all().item()),
        bool(oos_exp.eq(exp_gold[:,2]).all().item()),
        oos_checks["established_regressions"] == 0,
        oos_checks["exposed_regressions"] == 0,
        oos_checks["established_fixes"] == 3,
        oos_checks["exposed_fixes"] == 3,
        bool(oos_val.eq(val_gold[:,2]).all().item()),
        bool(oos_probe.eq(probe_gold[:,2]).all().item()),
    ])
    if not expected_oos_ok:
        def _mismatch_rows(rows, pred, gold, field_i=2, id_attr="family"):
            bad = []
            for i, row in enumerate(rows):
                g = int(gold[i, field_i].item())
                p = int(pred[i].item())
                if g != p:
                    bad.append({
                        "id": str(getattr(row, id_attr, getattr(row, "case_id", i))),
                        "turn": str(getattr(row, "turn", getattr(row, "utterance", ""))),
                        "gold_oos": g,
                        "pred_oos": p,
                    })
            return bad
        print("OOS_MISMATCH_original_val=", _mismatch_rows(original_val, oos_orig_val_pred, oos_orig_val_gold), flush=True)
        print("OOS_MISMATCH_combined_val=", _mismatch_rows(val_rows, oos_val, val_gold), flush=True)
        print("OOS_MISMATCH_established=", _mismatch_rows(established_cases, oos_est, est_gold, id_attr="case_id"), flush=True)
        print("OOS_MISMATCH_exposed=", _mismatch_rows(exposed_cases, oos_exp, exp_gold, id_attr="case_id"), flush=True)
        print("OOS_MISMATCH_probes=", _mismatch_rows(probes, oos_probe, probe_gold), flush=True)
        raise RuntimeError("Frozen OOS reconstruction or new in-domain OOS stability check failed; abort tournament.")
    print("OOS_FREEZE_RECONSTRUCTION=PASS")

    tensors = {
        "ref_train_rows": ref_train_rows,
        "amb_train_rows": amb_train_rows,
        "ref_train_x": ref_x,
        "amb_train_x": amb_x,
        "ref_train_disc": ref_disc,
        "amb_train_disc": amb_disc,
        "ref_train_y": ref_y,
        "amb_train_y": amb_y,
        "ref_train_margin": ref_margin,
        "amb_train_margin": amb_margin,
        "ref_train_base": ref_base,
        "amb_train_base": amb_base,
        "val_x": val_x,
        "est_x": est_x,
        "exp_x": exp_x,
        "probe_x": probe_x,
        "val_disc": val_disc,
        "est_disc": est_disc,
        "exp_disc": exp_disc,
        "probe_disc": probe_disc,
        "val_margin": val_margin,
        "est_margin": est_margin,
        "exp_margin": exp_margin,
        "probe_margin": probe_margin,
        "val_gold": val_gold,
        "est_gold": est_gold,
        "exp_gold": exp_gold,
        "probe_gold": probe_gold,
        "val_base": val_base,
        "est_base": est_base,
        "exp_base": exp_base,
        "probe_base": probe_base,
        "probe_rows": probes,
        "targeted_ref_mask": targeted_ref_mask,
        "targeted_amb_mask": targeted_amb_mask,
        "targeted_any_mask": targeted_any_mask,
        "oos_val": oos_val,
        "oos_est": oos_est,
        "oos_exp": oos_exp,
        "oos_probe": oos_probe,
    }

    print()
    print("========== BOUNDED THREE-ARCHITECTURE / THREE-SEED TOURNAMENT ==========")
    results: dict[str, list[dict[str, object]]] = {}
    for arch in ARCHITECTURES:
        print()
        print("ARCHITECTURE_START=", arch, flush=True)
        arch_results = []
        for seed in SEEDS:
            print("SEED_START=", seed, flush=True)
            result = run_seed(arch, seed, v52, thresholds, tensors)
            arch_results.append(result)
            print("SEED_RESULT=", result, flush=True)
        results[arch] = arch_results
        pass_count = sum(1 for r in arch_results if r["seed_pass"])
        print("ARCHITECTURE_SEED_PASS_COUNT=", f"{pass_count}/{len(SEEDS)}", flush=True)

    architecture_summary = {
        arch: {
            "seed_pass_count": sum(1 for r in rows if r["seed_pass"]),
            "all_seeds_pass": all(bool(r["seed_pass"]) for r in rows),
            "reference_strong_counts": [r["reference_strong_states"] for r in rows],
            "ambiguity_strong_counts": [r["ambiguity_strong_states"] for r in rows],
        }
        for arch, rows in results.items()
    }
    print()
    print("ARCHITECTURE_SUMMARY=", architecture_summary)

    winner = None
    for arch in ARCHITECTURES:  # simplest passing architecture wins
        if architecture_summary[arch]["all_seeds_pass"]:
            winner = arch
            break

    source_after = v52.base.source_snapshot()
    source_hashes_after = {
        "v52_source": sha256_file(v52_path),
        "phase7c_source": sha256_file(v52.base.P7C),
        "phase7c_checkpoint": sha256_file(v52.base.A7C),
        "phase8a_a8a3": sha256_file(v52.base.A8A3),
        "phase8a_hierarchical": sha256_file(v52.c8a.g.HIER_ARTIFACT),
    }
    artifact_after = {
        "exists": bool(v52.OUT.exists()),
        "sha256": sha256_file(v52.OUT) if v52.OUT.exists() else None,
    }
    integrity = all([
        source_before == source_after,
        source_hashes_before == source_hashes_after,
        artifact_before == artifact_after,
    ])

    print()
    print("========== POSTFLIGHT INTEGRITY ==========")
    print("source_tree_python_unchanged=", "YES" if source_before == source_after else "NO")
    print("source_checkpoint_hashes_unchanged=", "YES" if source_hashes_before == source_hashes_after else "NO")
    print("v52_candidate_artifact_unchanged=", "YES" if artifact_before == artifact_after else "NO")
    print("candidate_artifact_written=NO")
    print("runtime_wiring_modified=NO")
    print("phase8b_modified=NO")
    if not integrity:
        raise RuntimeError("Postflight integrity failure")

    print()
    print("========== AUTHORITATIVE FEASIBILITY VERDICT ==========")
    print("OOS_FROZEN=YES")
    print("OOS_FROZEN_EPOCH_SCALE=", (OOS_EPOCH, OOS_SCALE))
    if winner is not None:
        print("WINNING_ARCHITECTURE=", winner)
        print("FEASIBILITY_VERDICT=GO_FOR_ONE_INDEPENDENT_REPRODUCTION_ONLY")
        print("NEXT_ACTION=REPRODUCE_WINNING_ARCHITECTURE_FROM_SCRATCH_ONCE__IF_AND_ONLY_IF_REPRODUCTION_PASSES_THE_SAME_ZERO_REGRESSION_AND_INVARIANCE_CONTRACT_THEN_APPROVE_V5_3")
    else:
        print("WINNING_ARCHITECTURE=NONE")
        print("FEASIBILITY_VERDICT=HARD_STOP_NO_V5_3")
        print("NEXT_ACTION=DO_NOT_CREATE_V5_4__REVISIT_THE_FROZEN_REFERENCE_AMBIGUITY_REPRESENTATION_OR_FIELD_ONTOLOGY_BEFORE_ANY_MORE_RESIDUAL_TRAINING")
    print("feasibility_gate_completed=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
