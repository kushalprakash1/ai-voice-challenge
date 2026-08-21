#!/usr/bin/env python3
"""Phase 7C Composition V7 unclear-ownership proof (read-only).

This is the final bounded ownership correction following Composition V6.

Architecture under test
-----------------------
1. Reconstruct the already-approved V5.2 OOS head exactly at epoch 52/scale .9.
2. Run Phase 7J detail classification ungated, with the previously-proven
   deterministic kind-applicability corrections and noon/midday coverage.
3. Correct the specialist contracts without retraining:
   - Phase 7D is a REFERENCE-TYPE classifier, never a resolved/unresolved oracle.
   - Phase 7H alone decides whether an option reference resolves to one choice.
   - Phase 8A is supporting compatibility evidence, not a veto over stronger
     structured temporal/transaction evidence.
4. Phase 7H consumes the SAME composite candidate units produced by Phase 7J.
5. Missing evidence never creates ambiguity. Positive semantic evidence is
   required per kind.
6. The exact frozen OOS epoch-52/scale-.9 head is authoritative only for
   true OOS detail classes (generic/injection). `oos_unclear` is an ambiguity
   detail whose historical contract has gold_oos=0 and gold_ambiguity=1, so
   it is preserved only when an existing ambiguity structure is active.
7. Two fixed policies are evaluated in one run: strict corrected composition,
   then a conservative stability wrapper that preserves an existing active
   baseline unless strict composition has a positive resolution proof.
8. The V5 temporal boundary and morphology are frozen unchanged.
9. Final capability arbitration is limited to one generic ownership rule:
   an option detail with no legal option candidates may reroute to transaction
   only when the turn is an action anaphor and context contains 2+ normalized
   transaction operations.

No benchmark ID or gold label participates in inference. Gold is used only
post-inference for scoring. No source file, checkpoint, runtime wiring, or
artifact is written. No training or gradient update occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

SEP_BASENAME = "voiceprobe_semanticlab_v2_phase7c_regression_causal_separator_v1.py"
EXPECTED_SEP_SHA256 = "efa23d780dccec645d9d54434623ce8d0c287560c7e7bd89955c6469480677fb"

# Generic semantic morphology, not benchmark strings/IDs.
AXIS_NOUN_RE = re.compile(
    r"\b(?:days?|dates?|weeks?|times?|hours?|slots?|appointments?|"
    r"mornings?|afternoons?|evenings?)\b",
    re.I,
)
COMPARATIVE_RE = re.compile(r"\b(?:earlier|later|sooner)\b", re.I)
NON_TEMPORAL_OBJECT_RE = re.compile(
    r"\b(?:earlier|later|previous|prior|other)\s+"
    r"(?:one|thing|part|topic|subject|option|choice|answer|message)\b",
    re.I,
)
REPEAT_META_RE = re.compile(
    r"\b(?:repeat|say\s+(?:that|it)\s+again|again|didn['’]?t\s+(?:catch|hear))\b",
    re.I,
)

RECORD_ALLOWED_TOPICS = {"profile", "appointment_state", "other"}
TEMPORAL_ALLOWED_TOPICS = {"availability", "transaction", "other"}
TRANSACTION_ALLOWED_TOPICS = {"transaction"}
OTHER_ALLOWED_TOPICS = {"other"}


OOS_DETAILS = {"oos_generic", "oos_injection", "oos_unclear"}

# Generic ontology-level semantics. These are not benchmark strings/IDs.
IMPLICIT_LIST_RE = re.compile(
    r"\b(?:and)\b",
    re.I,
)
OPTION_LIST_SCAFFOLD_RE = re.compile(
    r"\b(?:"
    r"(?:i|we)\s+(?:have|offer)|"
    r"(?:i|we)\s+can\s+offer|"
    r"options?|choices?|both|listed|available|openings?|slots?"
    r")\b",
    re.I,
)

ORDINAL_FIRST_RE = re.compile(r"\b(?:first|1st|former)\b", re.I)
ORDINAL_SECOND_RE = re.compile(r"\b(?:second|2nd|latter)\b", re.I)
EARLIER_RE = re.compile(r"\b(?:earlier|sooner)\b", re.I)
LATER_RE = re.compile(r"\b(?:later|latest)\b", re.I)

# "later appointments/dates" names an axis. "move the appointment later" does not.
COMPARATIVE_EXPLICIT_AXIS_RE = re.compile(
    r"\b(?:earlier|later|sooner)\s+"
    r"(?:days?|dates?|times?|hours?|slots?|appointments?|"
    r"mornings?|afternoons?|evenings?)\b"
    r"|\b(?:earlier|later|sooner)\s+in\s+the\s+"
    r"(?:day|morning|afternoon|evening)\b",
    re.I,
)
TEMPORAL_CHANGE_EVENT_RE = re.compile(
    r"\b(?:"
    r"make|made|making|"
    r"do|did|doing|"
    r"move(?:d|s|ing)?|"
    r"shift(?:ed|s|ing)?|"
    r"push(?:ed|es|ing)?|"
    r"reschedul(?:e|ed|es|ing)|"
    r"chang(?:e|ed|es|ing)"
    r")\b.{0,72}\b(?:earlier|later|sooner)\b"
    r"|\b(?:appointment|visit|booking)\b"
    r".{0,72}\b(?:earlier|later|sooner)\b",
    re.I,
)

GENERIC_RECORD_REFERENCE_RE = re.compile(
    r"\b(?:that|this|which|what)\s+(?:record|entry)\b"
    r"|\b(?:record|entry)\s+(?:was|is|seems|looks)\b",
    re.I,
)

TRANSACTION_ANAPHOR_RE = re.compile(
    r"\b(?:do|proceed|continue|go\s+ahead|carry\s+on|apply)\b"
    r".{0,28}\b(?:it|that|this|change|action|request)\b",
    re.I,
)
INTERACTIVE_ACTS = {"question", "request", "confirmation"}

WEEKDAY_ORDER = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
DAYPART_ORDER = {"morning": 0, "afternoon": 1, "evening": 2}


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
    raise SystemExit(
        "Could not locate " + basename + ". Checked: " + ", ".join(map(str, candidates))
    )


def row_id(row) -> str:
    return str(getattr(row, "case_id", getattr(row, "family", "")))


def context_of(row) -> tuple[str, ...]:
    return tuple(str(x) for x in getattr(row, "context", ()))


def turn_text(row) -> str:
    return str(getattr(row, "turn", getattr(row, "utterance", "")))


def as_runtime_items(rows):
    return [SimpleNamespace(context=context_of(r), utterance=turn_text(r)) for r in rows]


def predict_ref_kinds(p7d, model, tok, rows):
    preds, probs = p7d.predict(model, tok, as_runtime_items(rows))
    return [str(x) for x in preds], probs


def predict_dense_pairs(p8a, model, tok, valid_pairs, rows):
    pairs, _independent, acts, topics = p8a.predict(
        model, tok, as_runtime_items(rows), valid_pairs
    )
    return [tuple(x) for x in pairs], [str(x) for x in acts], [str(x) for x in topics]



def visible_alternatives(context, loc) -> tuple[bool, tuple[str, ...]]:
    reasons = []
    any_visible = False
    for c in context:
        yes, reason = loc.visible_alternative_evidence(c)
        any_visible = any_visible or bool(yes)
        reasons.append(str(reason))
    return any_visible, tuple(reasons)


def structured_candidates(context, p7jr) -> tuple[str, ...]:
    try:
        return tuple(str(x) for x in p7jr.structured_context_candidates(tuple(context)))
    except Exception:
        return ()


def candidate_type_signature(candidate: str, p7jr, loc) -> tuple[str, ...]:
    try:
        return tuple(loc.candidate_component_types(str(candidate), p7jr))
    except Exception:
        return ("unknown",)


def positive_option_structure(context, candidates, p7jr, loc):
    """Return positive evidence that candidates are real alternatives.

    Explicit OR/option-list structure is strongest.  A homogeneous candidate list
    joined with AND is also legal when the context contains offer/list scaffolding.
    Heterogeneous provider+day or composite+time decompositions are rejected.
    """
    cands = tuple(str(x) for x in candidates)
    visible, reasons = visible_alternatives(context, loc)
    if len(cands) < 2:
        return False, f"candidate_count={len(cands)}", {
            "visible": visible, "reasons": reasons, "types": ()
        }

    types = tuple(candidate_type_signature(c, p7jr, loc) for c in cands)
    homogeneous = len(set(types)) == 1

    if visible:
        return True, "explicit_alternative_structure", {
            "visible": True, "reasons": reasons, "types": types
        }

    joined = " || ".join(str(x) for x in context)
    if (
        homogeneous
        and IMPLICIT_LIST_RE.search(joined)
        and OPTION_LIST_SCAFFOLD_RE.search(joined)
    ):
        return True, "implicit_homogeneous_and_list", {
            "visible": False, "reasons": reasons, "types": types
        }

    return False, "no_positive_alternative_structure", {
        "visible": visible, "reasons": reasons, "types": types
    }


def _norm_text(x: str) -> str:
    return " ".join(str(x).casefold().split())


def candidate_components(candidate: str, p7jr) -> tuple[str, ...]:
    """Extract closed semantic components from one already-structured candidate."""
    text = str(candidate)
    out = []
    for attr in ("PROVIDER", "DAY", "TIME", "DAYPART"):
        pat = getattr(p7jr, attr, None)
        if not pat:
            continue
        try:
            for m in re.finditer(pat, text, re.I):
                value = str(m.group(0)).strip()
                key = _norm_text(value)
                if key and all(_norm_text(x) != key for x in out):
                    out.append(value)
        except Exception:
            continue
    return tuple(out)


def unique_component_resolution(turn: str, candidates, p7jr):
    """Resolve a partial literal only when exactly one candidate owns the component."""
    utter = _norm_text(turn)
    matches = []
    for cand in candidates:
        owned = [
            component
            for component in candidate_components(str(cand), p7jr)
            if _norm_text(component) in utter
        ]
        if owned:
            matches.append((str(cand), tuple(owned)))
    unique = []
    for cand, comps in matches:
        if cand not in unique:
            unique.append(cand)
    if len(unique) == 1:
        return unique[0]
    return None


def temporal_candidate_key(candidate: str, p7h, p7jr):
    text = str(candidate)
    day = None
    daypart = None
    minute = p7h.time_minutes(text)

    try:
        m = re.search(p7jr.DAY, text, re.I)
        if m:
            day = WEEKDAY_ORDER.get(_norm_text(m.group(0)))
    except Exception:
        pass
    try:
        m = re.search(p7jr.DAYPART, text, re.I)
        if m:
            daypart = DAYPART_ORDER.get(_norm_text(m.group(0)))
    except Exception:
        pass

    if day is not None:
        inner = minute if minute is not None else (
            (daypart * 480 + 240) if daypart is not None else 0
        )
        return day * 1440 + inner
    if minute is not None:
        return minute
    if daypart is not None:
        return daypart
    return None


def strong_option_resolution(turn, context, candidates, p7h, p7jr):
    """Deterministic high-precision resolution before the learned Phase 7H operator."""
    cands = tuple(str(x) for x in candidates)
    if not cands:
        return "none", ""

    # Exact literal candidate first.
    literal = p7h.resolve("literal", context, cands, turn)
    if literal:
        return "literal_exact", str(literal)

    # A named subcomponent (e.g. "afternoon", provider, day) may uniquely identify
    # one composite candidate even if the full candidate string is not repeated.
    component = unique_component_resolution(turn, cands, p7jr)
    if component:
        return "literal_unique_component", str(component)

    if ORDINAL_FIRST_RE.search(turn):
        return "ordinal_first", cands[0]
    if ORDINAL_SECOND_RE.search(turn) and len(cands) >= 2:
        return "ordinal_second", cands[1]

    if EARLIER_RE.search(turn) or LATER_RE.search(turn):
        keyed = [(temporal_candidate_key(c, p7h, p7jr), c) for c in cands]
        if keyed and all(k is not None for k, _ in keyed):
            keys = [k for k, _ in keyed]
            if len(set(keys)) == len(keys):
                keyed.sort(key=lambda x: x[0])
                if EARLIER_RE.search(turn):
                    return "temporal_earlier_structured", keyed[0][1]
                return "temporal_later_structured", keyed[-1][1]

    return "none", ""


def predict_option_ops_on_structured(rows, details, p7jr, p7h, model, tok):
    """Run Phase 7H on the exact Phase 7J composite candidate units."""
    candidates_by_i: dict[int, tuple[str, ...]] = {}
    items = []
    indices = []
    for i, (row, detail) in enumerate(zip(rows, details)):
        if str(detail) != "option_reference":
            continue
        cands = structured_candidates(context_of(row), p7jr)
        candidates_by_i[i] = cands
        if not cands:
            continue
        items.append(SimpleNamespace(
            context=context_of(row),
            kind="prior_option",
            candidates=cands,
            turn=turn_text(row),
        ))
        indices.append(i)

    op_by_i: dict[int, tuple[str, str, float]] = {}
    if items:
        preds, probs = p7h.predict(model, tok, items)
        for i, item, op, pp in zip(indices, items, preds, probs):
            try:
                resolved = p7h.resolve(op, item.context, item.candidates, item.turn)
            except Exception:
                resolved = None
            op_by_i[i] = (
                str(op),
                "" if resolved is None else str(resolved),
                float(max(pp)) if pp else 0.0,
            )
    return candidates_by_i, op_by_i


def temporal_positive_evidence(comp, app, sep, context, turn, ref_kind, pair):
    """V4 temporal ambiguity boundary.

    Core rule:
      comparative temporal CHANGE + no concrete anchor/explicit axis => ambiguous
      comparative temporal CHANGE + concrete anchor/explicit axis    => resolved

    Phase 7D remains type evidence only. Phase 8A cannot veto or manufacture this
    structural boundary.
    """
    t = str(turn)
    ctx = " || ".join(context)
    act, topic = pair

    if app.NON_TEMPORAL_PRIOR_OBJECT_RE.search(t) or NON_TEMPORAL_OBJECT_RE.search(t):
        return False, "temporal:non_temporal_prior_object"

    # Explicit axis-alternative language is genuinely ambiguous even though it
    # may mention temporal vocabulary on both sides of the alternative.
    if app.TEMPORAL_AXIS_ALTERNATIVE_RE.search(t):
        return True, "temporal:explicit_axis_alternative"

    # A concrete current-turn anchor resolves the temporal dimension BEFORE any
    # earlier/later change morphology is considered.
    if (
        comp.CLOCK_RE.search(t)
        or comp.DAY_RE.search(t)
        or comp.DAYPART_RE.search(t)
        or comp.REL_DAY_RE.search(t)
    ):
        return False, "temporal:current_turn_concrete_anchor"

    # Comparative immediately naming an axis is a request on that axis rather
    # than ambiguity between day and time-of-day.
    if COMPARATIVE_EXPLICIT_AXIS_RE.search(t):
        return False, "temporal:comparative_with_explicit_axis"

    if comp.EXPLICIT_TIME_AXIS_RE.search(t) or comp.EXPLICIT_DAY_AXIS_RE.search(t):
        return False, "temporal:explicit_axis"

    # Generic change-event morphology is positive evidence. This deliberately
    # includes "make/do ... earlier/later" but excludes search/look/check verbs.
    if TEMPORAL_CHANGE_EVENT_RE.search(t):
        return True, "temporal:change_event_without_anchor"

    # Deictics are decided from actual contextual anchors.
    if comp.DEICTIC_TIME_RE.search(t):
        if not context:
            return False, "temporal:deictic_time_without_context_not_applicable"
        if comp.CLOCK_RE.search(ctx) or comp.DAYPART_RE.search(ctx):
            return False, "temporal:resolved_time_antecedent"
        if comp.DAY_RE.search(ctx) or comp.REL_DAY_RE.search(ctx):
            return True, "temporal:day_context_missing_time_anchor"
        return False, "temporal:no_positive_temporal_context"

    if comp.DEICTIC_DAY_RE.search(t):
        if not context:
            return False, "temporal:deictic_day_without_context_not_applicable"
        if comp.DAY_RE.search(ctx) or comp.REL_DAY_RE.search(ctx):
            return False, "temporal:resolved_day_antecedent"
        if comp.CLOCK_RE.search(ctx) or comp.DAYPART_RE.search(ctx):
            return True, "temporal:time_context_missing_day_anchor"
        return False, "temporal:no_positive_temporal_context"

    # Only now may metadiscourse suppress ambiguity. A genuine change-event
    # comparative has already returned above, so a noisy "clarification" act
    # cannot erase "Can we make that earlier?".
    if act in {"presence_check", "clarification"} or REPEAT_META_RE.search(t):
        return False, "temporal:metadiscourse_not_axis_ambiguity"

    return False, "temporal:no_positive_ambiguity_evidence"


def record_positive_evidence(comp, ref_kind, pair, context, turn):
    """Record ambiguity requires a genuinely vague record reference.

    Phase 7D 'unresolved' is a fallback class and therefore cannot activate this
    kind. Missing an entity is never positive evidence by itself.
    """
    act, topic = pair
    if topic not in RECORD_ALLOWED_TOPICS:
        return False, "record:topic_incompatible:" + str(topic)

    current_entities = comp.record_entities(turn)
    if len(current_entities) == 1:
        return False, "record:current_turn_entity_resolves"

    ctx_entities = comp.dedupe(x for c in context for x in comp.record_entities(c))
    if len(ctx_entities) == 1:
        return False, "record:context_entity_resolves"

    vague_record = bool(GENERIC_RECORD_REFERENCE_RE.search(str(turn)))
    if not vague_record:
        return False, "record:no_positive_vague_record_reference"

    if len(ctx_entities) >= 2:
        return True, "record:vague_reference_multiple_context_entities"

    if act in {"statement", "question", "clarification"}:
        return True, "record:vague_generic_record_without_specific_entity"

    return False, "record:speech_act_incompatible:" + str(act)


def transaction_positive_evidence(comp, ref_kind, pair, context, turn, normalize_operation):
    """Transaction ambiguity requires action-anaphor or multiple action evidence."""
    act, topic = pair
    ops = comp.context_transaction_ops(context, normalize_operation)
    interactive = act in INTERACTIVE_ACTS
    anaphor = bool(TRANSACTION_ANAPHOR_RE.search(str(turn)))

    # Multiple context actions are stronger evidence than a noisy dense topic.
    if len(ops) >= 2 and interactive:
        return True, "transaction:multiple_context_operations"

    if len(ops) == 1:
        return False, "transaction:single_context_operation_resolves"

    # Context-free "Should I do it?" / "go ahead with that change?" is a genuine
    # unresolved action reference when Phase 7J already identifies this kind.
    if anaphor and interactive:
        return True, "transaction:interactive_action_anaphor"

    return False, "transaction:no_positive_ambiguity_evidence"


def intent_positive_evidence(ref_kind, pair):
    act, topic = pair
    if ref_kind != "unresolved":
        return False, "intent:no_unresolved_reference_evidence:" + str(ref_kind)
    if (act, topic) != ("question", "other"):
        return False, "intent:pair_incompatible:" + repr((act, topic))
    return True, "intent:unresolved_question_other"


def other_positive_evidence(ref_kind, pair):
    _act, topic = pair
    if ref_kind != "unresolved":
        return False, "other:no_unresolved_reference_evidence:" + str(ref_kind)
    if topic not in OTHER_ALLOWED_TOPICS:
        return False, "other:topic_incompatible:" + str(topic)
    return True, "other:unresolved_other"


def oos_structure(detail, row, p7jr):
    if str(detail) not in OOS_DETAILS:
        return ("none", ()), "oos:detail_mismatch:" + str(detail)
    try:
        k, c = p7jr.ambiguity_from_detail(str(detail), context_of(row), turn_text(row))
        return (str(k), tuple(str(x) for x in c)), "oos:frozen_passthrough:" + str(detail)
    except Exception as exc:
        return ("none", ()), f"oos:resolver_error:{type(exc).__name__}:{exc}"


SAFE_SUPPRESSION_PREFIXES = (
    "option:strong_resolved:",
    "option:phase7h_resolved:",
    "temporal:non_temporal_prior_object",
    "temporal:current_turn_concrete_anchor",
    "temporal:comparative_with_explicit_axis",
    "temporal:explicit_axis",
    "temporal:resolved_time_antecedent",
    "temporal:resolved_day_antecedent",
    "temporal:metadiscourse_not_axis_ambiguity",
    "record:current_turn_entity_resolves",
    "record:context_entity_resolves",
    "transaction:single_context_operation_resolves",
    "oos:frozen_authority_negative",
)


def apply_stability_wrapper(baseline, strict, strict_reasons):
    """Preserve an existing active structure unless strict has resolution proof."""
    out = []
    reasons = []
    for base_value, cand_value, reason in zip(baseline, strict, strict_reasons):
        if (
            base_value[0] != "none"
            and cand_value != base_value
            and not str(reason).startswith(SAFE_SUPPRESSION_PREFIXES)
        ):
            out.append(base_value)
            reasons.append("stability_preserve:" + str(reason))
        else:
            out.append(cand_value)
            reasons.append(str(reason))
    return out, reasons



def final_capability_kind_arbitration(
    row,
    detail,
    p7jr,
    comp,
    normalize_operation,
):
    """Final generic applicability arbitration; no benchmark IDs or gold.

    The only added reroute is:
      option_reference + zero legal option candidates
      + interactive action anaphor
      + >=2 normalized context transaction operations
        -> transaction_reference

    This resolves a kind-ownership conflict. It does not infer candidates from
    benchmark labels and cannot fire when a legal option set exists.
    """
    detail = str(detail)
    if detail != "option_reference":
        return detail, "final_arbiter:unchanged"

    ctx = context_of(row)
    turn = turn_text(row)
    cands = structured_candidates(ctx, p7jr)
    if cands:
        return detail, f"final_arbiter:option_candidates_present:{len(cands)}"

    ops = comp.context_transaction_ops(ctx, normalize_operation)
    if len(ops) >= 2 and TRANSACTION_ANAPHOR_RE.search(str(turn)):
        return (
            "transaction_reference",
            "final_arbiter:option_to_transaction:"
            f"zero_option_candidates:ops={tuple(ops)}",
        )

    return detail, (
        "final_arbiter:option_kept:"
        f"candidate_count=0:ops={tuple(ops)}"
    )




def resolve_oos_detail_ownership(
    detail,
    is_frozen_oos,
    baseline_value,
    row,
    p7jr,
):
    """Resolve ownership between frozen scalar OOS and ambiguity-owned unclear.

    - oos_generic / oos_injection are true OOS classes and obey the exact frozen
      OOS head.
    - oos_unclear is not scalar OOS. It may preserve an already-active ambiguity
      structure, but an ungated Phase 7J detail may not create new ambiguity.

    This is a type/ownership rule, not a lexical detector.
    """
    detail = str(detail)

    if detail == "oos_unclear":
        if baseline_value[0] != "none":
            return baseline_value, "unclear:preserve_existing_ambiguity"
        return ("none", ()), "unclear:no_existing_ambiguity_activation"

    if detail in {"oos_generic", "oos_injection"}:
        if bool(is_frozen_oos):
            structure, reason = oos_structure(detail, row, p7jr)
            return structure, reason
        return ("none", ()), "oos:frozen_authority_negative"

    return ("none", ()), "oos:not_an_oos_detail:" + detail



def compose_v3(
    rows,
    details,
    ref_kinds,
    dense_pairs,
    frozen_oos,
    baseline,
    p7jr,
    p7h,
    op_model,
    op_tok,
    comp,
    app,
    sep,
    loc,
    normalize_operation,
):
    candidates_by_i, op_by_i = predict_option_ops_on_structured(
        rows, details, p7jr, p7h, op_model, op_tok
    )

    final = []
    reasons = []
    for i, (row, detail, ref_kind, pair, is_oos, base_value) in enumerate(
        zip(rows, details, ref_kinds, dense_pairs, frozen_oos, baseline)
    ):
        ctx = context_of(row)
        turn = turn_text(row)
        detail = str(detail)

        # V7 ownership contract: scalar OOS owns generic/injection only.
        # `oos_unclear` belongs to ambiguity and is not suppressed by a scalar
        # OOS negative when an existing ambiguity structure is already active.
        if detail in OOS_DETAILS:
            structure, reason = resolve_oos_detail_ownership(
                detail,
                bool(is_oos),
                base_value,
                row,
                p7jr,
            )
            final.append(structure)
            reasons.append(reason)
            continue

        if detail == "option_reference":
            cands = tuple(candidates_by_i.get(i, ()))
            alt_ok, alt_reason, alt_diag = positive_option_structure(ctx, cands, p7jr, loc)
            if not alt_ok:
                final.append(("none", ()))
                reasons.append(
                    "option:no_positive_alternative_structure:"
                    f"{alt_reason}:count={len(cands)}:types={alt_diag.get('types', ())}"
                )
                continue

            strong_op, strong_resolved = strong_option_resolution(
                turn, ctx, cands, p7h, p7jr
            )
            if strong_resolved:
                final.append(("none", ()))
                reasons.append(
                    f"option:strong_resolved:{strong_op}:{strong_resolved}"
                )
                continue

            op, resolved, conf = op_by_i.get(i, ("none", "", 0.0))
            if resolved:
                final.append(("none", ()))
                reasons.append(f"option:phase7h_resolved:{op}:{resolved}:p={conf:.3f}")
                continue

            # Phase 7D prior_option means "typed reference to an option"; it does
            # NOT mean the option has already been resolved.
            final.append(("option_reference", cands))
            reasons.append(
                f"option:positive_unresolved_alternatives:"
                f"refkind={ref_kind}:op={op}:count={len(cands)}:p={conf:.3f}:"
                f"structure={alt_reason}"
            )
            continue

        if detail == "temporal_reference":
            active, reason = temporal_positive_evidence(
                comp, app, sep, ctx, turn, str(ref_kind), tuple(pair)
            )
            final.append(("temporal_reference", ("time_of_day", "day")) if active else ("none", ()))
            reasons.append(reason)
            continue

        if detail == "record_reference":
            active, reason = record_positive_evidence(
                comp, str(ref_kind), tuple(pair), ctx, turn
            )
            final.append(("record_reference", ("profile", "appointment")) if active else ("none", ()))
            reasons.append(reason)
            continue

        if detail == "transaction_reference":
            active, reason = transaction_positive_evidence(
                comp, str(ref_kind), tuple(pair), ctx, turn, normalize_operation
            )
            final.append(("transaction_reference", ("book", "reschedule", "cancel")) if active else ("none", ()))
            reasons.append(reason)
            continue

        if detail == "intent_next_step":
            active, reason = intent_positive_evidence(str(ref_kind), tuple(pair))
            final.append(("intent", ("request_next_step", "acknowledgement")) if active else ("none", ()))
            reasons.append(reason)
            continue

        if detail == "other_prior":
            active, reason = other_positive_evidence(str(ref_kind), tuple(pair))
            final.append(("other", ("prior_option", "prior_topic")) if active else ("none", ()))
            reasons.append(reason)
            continue

        final.append(("none", ()))
        reasons.append("detail:not_applicable:" + detail)

    return final, reasons, candidates_by_i, op_by_i


def print_failure_evidence(
    name, rows, gold, baseline, candidate, reasons, details, ref_kinds, pairs, limit=12
):
    def ok(pred, g, exact):
        if g[0] == "__binary_active__":
            return int(pred[0] != "none") == 1
        if not exact:
            return pred[0] == g[0]
        return pred == g

    exact = name in {"established1146", "exposed120"}
    shown = 0
    for i, row in enumerate(rows):
        if ok(candidate[i], gold[i], exact):
            continue
        print("  EVIDENCE_FAIL", {
            "id": row_id(row),
            "gold": gold[i],
            "baseline": baseline[i],
            "candidate": candidate[i],
            "detail": str(details[i]),
            "phase7d": str(ref_kinds[i]),
            "phase8a_pair": tuple(pairs[i]),
            "reason": str(reasons[i]),
            "turn": turn_text(row),
            "context": list(context_of(row)),
        })
        shown += 1
        if shown >= limit:
            break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--separator-source", default=None)
    args = ap.parse_args()

    print("========== PHASE 7C COMPOSITION V5 POSITIVE-EVIDENCE PROOF ==========")
    print("telephony=DISABLED")
    print("training=NO")
    print("gradient_updates=NO")
    print("repo_modified=NO")
    print("runtime_wiring_modified=NO")
    print("new_ambiguity_head=NO")
    print("scalar_residual_training=NO")
    print("benchmark_ids_used_for_inference=NO")

    sep_path = resolve_named(args.separator_source, SEP_BASENAME)
    sep_hash = sha256_file(sep_path)
    print("separator_source=", sep_path)
    print("separator_source_sha256=", sep_hash)
    if sep_hash != EXPECTED_SEP_SHA256:
        raise RuntimeError(
            f"Separator source drift expected={EXPECTED_SEP_SHA256} actual={sep_hash}"
        )
    sep = load_mod("phase7c_compv3_separator", sep_path)

    loc_path = sep.resolve_named(None, sep.LOCALIZER_BASENAME)
    if sha256_file(loc_path) != sep.EXPECTED_LOCALIZER_SHA256:
        raise RuntimeError("Localizer source drift")
    loc = load_mod("phase7c_compv3_localizer", loc_path)

    app_path = loc.resolve_named(None, loc.APP_BASENAME)
    if sha256_file(app_path) != loc.EXPECTED_APP_SHA256:
        raise RuntimeError("Applicability source drift")
    app = load_mod("phase7c_compv3_app", app_path)

    comp_path = app.resolve_named(None, app.COMP_BASENAME)
    if sha256_file(comp_path) != app.EXPECTED_COMP_SHA256:
        raise RuntimeError("Composition source drift")
    comp = load_mod("phase7c_compv3_comp", comp_path)

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
            raise RuntimeError(f"{label} source drift expected={expected} actual={actual}")

    v52 = load_mod("phase7c_compv3_v52", v52_path)
    feas = load_mod("phase7c_compv3_feas", feas_path)
    struct = load_mod("phase7c_compv3_struct", struct_path)
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
    hashes_before = {k: sha256_file(p) for k, p in watched.items()}

    # ------------------------------------------------------------------
    # EXACT frozen OOS reconstruction. Nothing model-like is loaded before
    # the authoritative V5.2 replay-sensitive residual-head initialization.
    # ------------------------------------------------------------------
    print("\n========== EXACT FROZEN OOS RECONSTRUCTION ==========")
    random.seed(v52.SEED)
    torch.manual_seed(v52.SEED)
    gate_ck, gate_model, gate_tok = v52.load_current_model()
    thresholds = {f: float(gate_ck["thresholds"][f]) for f in v52.FIELDS}

    original_train, original_val = v52.build_synthetic()
    original_train, original_val, _blocked, _blocked_files = feas.filter_original_synthetic(
        v52, original_train, original_val
    )
    orig_x, _, orig_margin, _orig_base, _, _ = feas.capture(
        v52, gate_model, gate_tok, original_train, thresholds
    )
    orig_y = v52.gold_example_tensor(original_train)
    oval_x, _, oval_margin, _oval_base, _, _ = feas.capture(
        v52, gate_model, gate_tok, original_val, thresholds
    )
    oval_gold = v52.gold_example_tensor(original_val).long()
    historical = list(v52.load_semanticlab_cases())
    if len(historical) != 133:
        raise RuntimeError(f"Historical replay cardinality drift: {len(historical)}")
    v52.capture_features(
        gate_model, gate_tok, v52.runtime_for_cases(historical), thresholds
    )
    replay_head = v52.DirectionalFactorizedResidual(orig_x.shape[1])
    frozen_oos_head = feas.reconstruct_frozen_oos(
        v52, replay_head.oos, orig_x, orig_y, orig_margin, original_train
    )
    print("oos_replay_head_initialized_at_authoritative_v52_point=YES")

    # Corpora/fresh diagnostics are loaded only after OOS head initialization.
    groups, exposed = v52.load_groups()
    established = [c for _, cases in groups for c in cases]
    if len(established) != 1146 or len(exposed) != 120:
        raise RuntimeError(
            f"Corpus cardinality drift established={len(established)} exposed={len(exposed)}"
        )
    target = [
        x for x in feas.build_targeted_validation()
        if str(getattr(x, "family", "")) not in struct.AMBIGUITY_APPLICABILITY_EXCLUDED_FAMILIES
    ]
    probes = [
        x for x in feas.build_metamorphic_probes()
        if str(getattr(x, "family", "")) not in struct.AMBIGUITY_APPLICABILITY_EXCLUDED_FAMILIES
    ]
    structured_val = struct.build_structured_validation()

    def capture_rows(rows, are_cases):
        rt = v52.runtime_for_cases(rows) if are_cases else v52.runtime_for_examples(rows)
        return v52.capture_features(gate_model, gate_tok, rt, thresholds)

    # OOS preflight checks.
    oval_pred, _ = feas.frozen_oos_pred(frozen_oos_head, oval_x, oval_margin)
    est_x, _, est_margin, est_base_pred, _, _ = capture_rows(established, True)
    exp_x, _, exp_margin, exp_base_pred, _, _ = capture_rows(exposed, True)
    tgt_x, _, tgt_margin, tgt_base_pred, _, _ = capture_rows(target, False)
    prb_x, _, prb_margin, prb_base_pred, _, _ = capture_rows(probes, False)
    sv_x, _, sv_margin, sv_base_pred, _, _ = capture_rows(structured_val, False)

    est_oos, _ = feas.frozen_oos_pred(frozen_oos_head, est_x, est_margin)
    exp_oos, _ = feas.frozen_oos_pred(frozen_oos_head, exp_x, exp_margin)
    tgt_oos, _ = feas.frozen_oos_pred(frozen_oos_head, tgt_x, tgt_margin)
    prb_oos, _ = feas.frozen_oos_pred(frozen_oos_head, prb_x, prb_margin)
    sv_oos, _ = feas.frozen_oos_pred(frozen_oos_head, sv_x, sv_margin)

    est_gold_tensor = v52.gold_case_tensor(established).long()
    exp_gold_tensor = v52.gold_case_tensor(exposed).long()
    oos_checks = {
        "original_val": f"{int((oval_pred == oval_gold[:,2]).sum())}/{len(original_val)}",
        "established": f"{int((est_oos == est_gold_tensor[:,2]).sum())}/{len(established)}",
        "exposed": f"{int((exp_oos == exp_gold_tensor[:,2]).sum())}/{len(exposed)}",
        "targeted_in_domain_zero": int(tgt_oos.sum()) == 0,
        "probe_in_domain_zero": int(prb_oos.sum()) == 0,
        "structured_in_domain_zero": int(sv_oos.sum()) == 0,
    }
    print("frozen_oos_epoch_scale=", (feas.OOS_EPOCH, feas.OOS_SCALE))
    print("frozen_oos_checks=", oos_checks)
    oos_pass = (
        oos_checks["original_val"] == f"{len(original_val)}/{len(original_val)}"
        and oos_checks["established"] == f"{len(established)}/{len(established)}"
        and oos_checks["exposed"] == f"{len(exposed)}/{len(exposed)}"
        and oos_checks["targeted_in_domain_zero"]
        and oos_checks["probe_in_domain_zero"]
        and oos_checks["structured_in_domain_zero"]
    )
    print("OOS_FREEZE_RECONSTRUCTION=", "PASS" if oos_pass else "FAIL")
    if not oos_pass:
        raise RuntimeError("Frozen OOS preflight failed; abort Composition V5 proof")

    # ------------------------------------------------------------------
    # Load existing frozen semantic specialists only after OOS replay point.
    # ------------------------------------------------------------------
    p7j = base.load_mod("phase7c_compv3_p7j", base.P7J)
    # Keep an unmodified resolver instance solely for exact current-baseline
    # comparison. Candidate coverage extensions apply only to the V2 candidate.
    p7jr_baseline = base.load_mod("phase7c_compv3_p7jr_baseline", base.P7JR)
    p7jr = base.load_mod("phase7c_compv3_p7jr_candidate", base.P7JR)
    p7h = base.load_mod("phase7c_compv3_p7h", base.P7H)
    p7d = base.load_mod("phase7c_compv3_p7d", base.P7D)
    p8a = base.load_mod("phase7c_compv3_p8a", base.P8A)
    p8dn = base.load_mod("phase7c_compv3_p8dn", base.P8DN)

    # In-memory only: proven candidate coverage extension.
    app.install_extended_candidate_coverage(p7jr, p7h)
    comp._original_temporal_resolution = comp.temporal_resolution
    comp.temporal_resolution = lambda context, turn: app.temporal_resolution_v2(comp, context, turn)

    ck7j = base.load_checkpoint(base.A7J)
    detail_model = p7j.DetailModel()
    detail_model.load_state_dict(ck7j["state_dict"])
    detail_model.eval()
    detail_tok = base.tokenizer_for(
        ck7j.get("model_name", getattr(p7j, "MODEL_NAME", "distilbert/distilbert-base-uncased"))
    )

    ck7i = base.load_checkpoint(base.A7I)
    op_model = p7h.OperatorModel()
    op_model.load_state_dict(ck7i["state_dict"])
    op_model.eval()
    op_tok = base.tokenizer_for(
        ck7i.get("model_name", getattr(p7h, "MODEL_NAME", "distilbert/distilbert-base-uncased"))
    )

    ck7d = base.load_checkpoint(base.A7D)
    ref_model = p7d.RefKindModel()
    ref_model.load_state_dict(ck7d["state_dict"])
    ref_model.eval()
    ref_tok = base.tokenizer_for(
        ck7d.get("model_name", getattr(p7d, "MODEL_NAME", "distilbert/distilbert-base-uncased"))
    )

    ck8a = base.load_checkpoint(base.A8A3)
    dense_model = p8a.DenseModel()
    dense_model.load_state_dict(ck8a["state_dict"])
    dense_model.eval()
    dense_tok = base.tokenizer_for(
        ck8a.get("model_name", getattr(p8a, "MODEL_NAME", "distilbert/distilbert-base-uncased"))
    )
    valid_pairs = tuple(tuple(x) for x in ck8a["valid_pairs"])
    print("FROZEN_SPECIALIST_LOAD=PASS")

    # Pure pre-gold V4 temporal-boundary smokes.
    temporal_smokes = {
        "generic_do_earlier_active": temporal_positive_evidence(
            comp, app, sep, (), "Can we do something earlier?",
            "unresolved", ("question", "other")
        )[0],
        "generic_make_later_active": temporal_positive_evidence(
            comp, app, sep, (), "Could we make it later?",
            "unresolved", ("question", "availability")
        )[0],
        "generic_make_that_earlier_active_despite_clarification": temporal_positive_evidence(
            comp, app, sep, (), "Can we make that earlier?",
            "unresolved", ("clarification", "other")
        )[0],
        "inflected_shifted_later_active": temporal_positive_evidence(
            comp, app, sep, (), "Could that be shifted to something later?",
            "unresolved", ("clarification", "other")
        )[0],
        "inflected_moved_earlier_active": temporal_positive_evidence(
            comp, app, sep, (), "Could that be moved somewhat earlier?",
            "unresolved", ("question", "availability")
        )[0],
        "inflected_shifting_anchored_inactive": temporal_positive_evidence(
            comp, app, sep, (), "Could we be shifting it earlier on Thursday morning?",
            "unresolved", ("question", "availability")
        )[0],
        "anchored_daypart_inactive": temporal_positive_evidence(
            comp, app, sep, (), "Could we make the appointment later on Thursday morning?",
            "unresolved", ("offer", "transaction")
        )[0],
        "anchored_weekday_daypart_inactive": temporal_positive_evidence(
            comp, app, sep, (), "Could we move the visit earlier on Wednesday afternoon?",
            "unresolved", ("offer", "availability")
        )[0],
        "explicit_axis_inactive": temporal_positive_evidence(
            comp, app, sep, (), "Should I check later dates?",
            "unresolved", ("question", "availability")
        )[0],
        "search_later_in_day_inactive": temporal_positive_evidence(
            comp, app, sep, (), "Should I look for something later in the day?",
            "unresolved", ("question", "availability")
        )[0],
    }
    expected_smokes = {
        "generic_do_earlier_active": True,
        "generic_make_later_active": True,
        "generic_make_that_earlier_active_despite_clarification": True,
        "inflected_shifted_later_active": True,
        "inflected_moved_earlier_active": True,
        "inflected_shifting_anchored_inactive": False,
        "anchored_daypart_inactive": False,
        "anchored_weekday_daypart_inactive": False,
        "explicit_axis_inactive": False,
        "search_later_in_day_inactive": False,
    }
    if temporal_smokes != expected_smokes:
        raise RuntimeError(
            "V6 preserved temporal boundary smoke failed: "
            + repr({"actual": temporal_smokes, "expected": expected_smokes})
        )

    # Preserve the non-temporal V3 contract smokes unchanged.
    smoke_deictic = temporal_positive_evidence(
        comp, app, sep, ("We are checking Saturday availability.",),
        "Anything else near that time?", "prior_day", ("question", "availability")
    )
    smoke_record = record_positive_evidence(
        comp, "unresolved", ("statement", "appointment_state"),
        (), "That record seems to be missing."
    )
    smoke_record_explicit = record_positive_evidence(
        comp, "unresolved", ("statement", "profile"),
        (), "It looks like you're a new patient in our system."
    )
    smoke_tx = transaction_positive_evidence(
        comp, "unresolved", ("question", "availability"),
        ("I can cancel the visit or reschedule it.",),
        "Should I proceed with that now?", p8dn.normalize_operation
    )
    smoke_option_list = positive_option_structure(
        ("I have Monday and Tuesday.",), ("Monday", "Tuesday"), p7jr, loc
    )
    smoke_option_split = positive_option_structure(
        ("Dr. Calder has a Thursday afternoon opening.",),
        ("Dr. Calder", "Thursday afternoon"), p7jr, loc
    )
    if (
        not smoke_deictic[0]
        or not smoke_record[0]
        or smoke_record_explicit[0]
        or not smoke_tx[0]
        or not smoke_option_list[0]
        or smoke_option_split[0]
    ):
        raise RuntimeError(
            "Preserved V3 contract smoke failed: "
            + repr((
                smoke_deictic, smoke_record, smoke_record_explicit,
                smoke_tx, smoke_option_list, smoke_option_split,
            ))
        )
    # Final ownership/arbitration smokes.
    tx_probe = SimpleNamespace(
        context=("I can cancel the booking or reschedule it.",),
        turn="Should I carry on with that action?",
    )
    tx_corr, tx_reason = final_capability_kind_arbitration(
        tx_probe,
        "option_reference",
        p7jr,
        comp,
        p8dn.normalize_operation,
    )
    option_probe = SimpleNamespace(
        context=("I can offer Monday morning or Thursday evening.",),
        turn="Should I take that one?",
    )
    option_corr, option_reason = final_capability_kind_arbitration(
        option_probe,
        "option_reference",
        p7jr,
        comp,
        p8dn.normalize_operation,
    )
    if tx_corr != "transaction_reference":
        raise RuntimeError(
            "Final transaction arbitration smoke failed: "
            + repr((tx_corr, tx_reason))
        )
    if option_corr != "option_reference":
        raise RuntimeError(
            "Final option-preservation smoke failed: "
            + repr((option_corr, option_reason))
        )

    # OOS-authority wrapper smoke: a frozen-negative OOS suppression must not be
    # reversed by the stability wrapper.
    oos_safe, oos_safe_reasons = apply_stability_wrapper(
        [("intent", ("out_of_scope", "other"))],
        [("none", ())],
        ["oos:frozen_authority_negative"],
    )
    if oos_safe != [("none", ())]:
        raise RuntimeError(
            "Frozen OOS authority stability smoke failed: "
            + repr((oos_safe, oos_safe_reasons))
        )

    print("TEMPORAL_BOUNDARY_SMOKE=PASS")
    print("NON_TEMPORAL_CONTRACT_SMOKE=PASS")
    print("FINAL_KIND_ARBITRATION_SMOKE=PASS")
    # V7 unclear ownership smokes.
    unclear_row = SimpleNamespace(context=(), turn="Unclear utterance.")
    unclear_active, unclear_active_reason = resolve_oos_detail_ownership(
        "oos_unclear",
        False,
        ("other", ("unclear", "other")),
        unclear_row,
        p7jr,
    )
    unclear_inactive, unclear_inactive_reason = resolve_oos_detail_ownership(
        "oos_unclear",
        False,
        ("none", ()),
        unclear_row,
        p7jr,
    )
    generic_negative, generic_negative_reason = resolve_oos_detail_ownership(
        "oos_generic",
        False,
        ("intent", ("out_of_scope", "other")),
        unclear_row,
        p7jr,
    )
    if unclear_active != ("other", ("unclear", "other")):
        raise RuntimeError(
            "Unclear active ownership smoke failed: "
            + repr((unclear_active, unclear_active_reason))
        )
    if unclear_inactive != ("none", ()):
        raise RuntimeError(
            "Unclear inactive ownership smoke failed: "
            + repr((unclear_inactive, unclear_inactive_reason))
        )
    if generic_negative != ("none", ()):
        raise RuntimeError(
            "Generic frozen-negative ownership smoke failed: "
            + repr((generic_negative, generic_negative_reason))
        )

    print("FROZEN_OOS_AUTHORITY_SMOKE=PASS")
    print("UNCLEAR_AMBIGUITY_OWNERSHIP_SMOKE=PASS")

    datasets = [
        ("established1146", established, True, est_base_pred, est_oos),
        ("exposed120", exposed, True, exp_base_pred, exp_oos),
        ("targeted", target, False, tgt_base_pred, tgt_oos),
        ("metamorphic", probes, False, prb_base_pred, prb_oos),
        ("structured_val", structured_val, False, sv_base_pred, sv_oos),
    ]

    results = {}
    failure_reason_counts = Counter()
    regression_reason_counts = Counter()
    correction_counts = Counter()
    kind_coverage_failures = []

    for name, rows, are_cases, base_pred_tensor, oos_pred_tensor in datasets:
        print("\n==========", name, "==========")
        raw_details, detail_probs = comp.ungated_phase7j(
            rows, p7j, detail_model, detail_tok
        )
        details, correction_reasons = app.corrected_phase7j_details(
            rows,
            raw_details,
            detail_probs,
            p7j,
            p7jr,
            comp,
            p8dn.normalize_operation,
        )

        # Final V6 ownership arbitration. This is deterministic, generic, and
        # executes before any gold labels are consulted.
        arb_details = []
        arb_reasons = []
        for row, detail in zip(rows, details):
            corrected, why = final_capability_kind_arbitration(
                row,
                detail,
                p7jr,
                comp,
                p8dn.normalize_operation,
            )
            arb_details.append(str(corrected))
            arb_reasons.append(str(why))

        for a, b, why in zip(raw_details, details, correction_reasons):
            if str(a) != str(b):
                correction_counts[(str(a), str(b), str(why))] += 1
        for a, b, why in zip(details, arb_details, arb_reasons):
            if str(a) != str(b):
                correction_counts[(str(a), str(b), str(why))] += 1
        details = arb_details

        ref_kinds, _ref_probs = predict_ref_kinds(p7d, ref_model, ref_tok, rows)
        dense_pairs, _acts, _topics = predict_dense_pairs(
            p8a, dense_model, dense_tok, valid_pairs, rows
        )

        base_amb = [bool(int(x)) for x in base_pred_tensor[:, 1].tolist()]
        baseline = comp.current_baseline_structures(rows, base_amb, raw_details, p7jr_baseline)
        strict_candidate, strict_reasons, _cand_units, _op_state = compose_v3(
            rows,
            details,
            ref_kinds,
            dense_pairs,
            [bool(int(x)) for x in oos_pred_tensor.tolist()],
            baseline,
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
        safe_candidate, safe_reasons = apply_stability_wrapper(
            baseline, strict_candidate, strict_reasons
        )

        gold, exact_candidates = app.exact_gold_for_dataset(
            name, rows, are_cases, p7jr, comp
        )

        print("  POLICY=strict_contract")
        strict_metrics = comp.evaluate_dataset(
            name + ":strict",
            rows,
            gold,
            baseline,
            strict_candidate,
            strict_reasons,
            details,
            exact_candidates=exact_candidates,
            max_fail=6,
        )
        print("  POLICY=stability_wrapper")
        safe_metrics = comp.evaluate_dataset(
            name,
            rows,
            gold,
            baseline,
            safe_candidate,
            safe_reasons,
            details,
            exact_candidates=exact_candidates,
            max_fail=8,
        )
        results[name] = {
            "strict": strict_metrics,
            "safe": safe_metrics,
        }
        print_failure_evidence(
            name, rows, gold, baseline, safe_candidate, safe_reasons,
            details, ref_kinds, dense_pairs, limit=8
        )
        candidate = safe_candidate
        reasons = safe_reasons
        metrics = safe_metrics

        for i in metrics["failure_indices"]:
            failure_reason_counts[str(reasons[i]).split(":", 2)[0] + ":" + str(reasons[i]).split(":", 2)[1] if ":" in str(reasons[i]) else str(reasons[i])] += 1
        for i in metrics["regression_indices"]:
            regression_reason_counts[str(reasons[i]).split(":", 2)[0] + ":" + str(reasons[i]).split(":", 2)[1] if ":" in str(reasons[i]) else str(reasons[i])] += 1

        # Kind/candidate coverage check on gold-active corpus cases only, after
        # inference is immutable. This is diagnostic, not inference routing.
        if are_cases:
            for i, row in enumerate(rows):
                if comp.binary_gold(row) != 1:
                    continue
                gk, gc = gold[i]
                try:
                    pk, pc = p7jr.ambiguity_from_detail(
                        details[i], context_of(row), turn_text(row)
                    )
                    pred = (str(pk), tuple(str(x) for x in pc))
                except Exception as exc:
                    pred = ("resolver_error", ())
                if pred != (gk, gc):
                    top = sorted(
                        zip(p7j.DETAILS, detail_probs[i]),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:3]
                    kind_coverage_failures.append({
                        "dataset": name,
                        "id": row_id(row),
                        "gold": (gk, gc),
                        "pred": pred,
                        "detail": str(details[i]),
                        "top3": [(str(a), round(float(b), 4)) for a, b in top],
                    })

    print("\n========== COMPOSITION V7 DIAGNOSTIC SUMMARY ==========")
    print("kind_applicability_correction_counts=", {str(k): v for k, v in correction_counts.items()})
    print("active_kind_candidate_failure_count=", len(kind_coverage_failures))
    for item in kind_coverage_failures[:12]:
        print(" KIND_COVERAGE_FAIL", item)
    print("failure_reason_counts=", dict(failure_reason_counts))
    print("regression_reason_counts=", dict(regression_reason_counts))

    def policy_summary(policy):
        est = results["established1146"][policy]
        exp = results["exposed120"][policy]
        tgt = results["targeted"][policy]
        prb = results["metamorphic"][policy]
        sv = results["structured_val"][policy]

        zero_reg = (
            est["regressions"] == 0
            and exp["regressions"] == 0
            and tgt["regressions"] == 0
            and prb["regressions"] == 0
            and sv["regressions"] == 0
        )
        fresh_exact = (
            tgt["candidate_exact"] == tgt["n"]
            and prb["candidate_exact"] == prb["n"]
            and sv["candidate_exact"] == sv["n"]
        )
        stable_non_degrading = (
            est["candidate_exact"] >= est["base_exact"]
            and exp["candidate_exact"] >= exp["base_exact"]
        )
        has_fix = sum(results[name][policy]["fixes"] for name in results) > 0
        return {
            "est": est, "exp": exp, "tgt": tgt, "prb": prb, "sv": sv,
            "zero_reg": zero_reg,
            "fresh_exact": fresh_exact,
            "stable_non_degrading": stable_non_degrading,
            "has_fix": has_fix,
        }

    strict_summary = policy_summary("strict")
    safe_summary = policy_summary("safe")

    print("\n========== COMPOSITION V7 POLICY COMPARISON ==========")
    for label, sm in (("strict_contract", strict_summary), ("stability_wrapper", safe_summary)):
        print(label, {
            "zero_regression": sm["zero_reg"],
            "fresh_exact": sm["fresh_exact"],
            "stable_non_degrading": sm["stable_non_degrading"],
            "has_fix": sm["has_fix"],
            "established_exact": f"{sm['est']['candidate_exact']}/{sm['est']['n']}",
            "exposed_exact": f"{sm['exp']['candidate_exact']}/{sm['exp']['n']}",
            "targeted_exact": f"{sm['tgt']['candidate_exact']}/{sm['tgt']['n']}",
            "metamorphic_exact": f"{sm['prb']['candidate_exact']}/{sm['prb']['n']}",
            "structured_exact": f"{sm['sv']['candidate_exact']}/{sm['sv']['n']}",
        })

    def passes(sm):
        return (
            sm["zero_reg"]
            and sm["fresh_exact"]
            and sm["stable_non_degrading"]
            and sm["has_fix"]
            and not kind_coverage_failures
        )

    if passes(strict_summary):
        selected_policy = "strict_contract"
        selected = strict_summary
        verdict = "COMPOSITION_V7_STRICT_CONTRACT_PROVEN"
        primary = "NONE"
        next_action = (
            "RUN_EXACTLY_ONE_INDEPENDENT_COLD_REPRODUCTION_OF_THE_STRICT_CONTRACT__"
            "THEN_SHADOW_INTEGRATE_BEFORE_RUNTIME_WIRING"
        )
    elif passes(safe_summary):
        selected_policy = "stability_wrapper"
        selected = safe_summary
        verdict = "COMPOSITION_V7_STABILITY_WRAPPER_PROVEN"
        primary = "UNSAFE_SUPPRESSION_IN_STRICT_CONTRACT"
        next_action = (
            "RUN_EXACTLY_ONE_INDEPENDENT_COLD_REPRODUCTION_OF_THE_STABILITY_WRAPPER__"
            "THEN_SHADOW_INTEGRATE_WITH_BASELINE_PRESERVATION_EXPLICIT"
        )
    else:
        selected_policy = "NONE"
        selected = safe_summary
        if regression_reason_counts:
            primary = regression_reason_counts.most_common(1)[0][0]
            verdict = "COMPOSITION_V7_STABILITY_BLOCKED"
        elif not safe_summary["fresh_exact"]:
            primary = (
                failure_reason_counts.most_common(1)[0][0]
                if failure_reason_counts else "FRESH_UNDERCOVERAGE"
            )
            verdict = "COMPOSITION_V7_CAPABILITY_UNDERCOVERAGE"
        elif kind_coverage_failures:
            primary = "PHASE7J_KIND_COVERAGE"
            verdict = "COMPOSITION_V7_KIND_COVERAGE_BLOCKED"
        else:
            primary = "UNCLASSIFIED_CONTRACT_FAILURE"
            verdict = "COMPOSITION_V7_NOT_PROVEN"
        next_action = (
            "NO_TRAINING_OR_RUNTIME_PATCH__THE_REMAINING_PRINTED_CAUSE_IS_THE_ONLY_"
            "PERMITTED_NEXT_ARCHITECTURAL_TARGET"
        )

    source_after = base.source_snapshot()
    hashes_after = {k: sha256_file(p) for k, p in watched.items()}
    print("\n========== POSTFLIGHT INTEGRITY ==========")
    print("source_tree_python_unchanged=", "YES" if source_after == source_before else "NO")
    print(
        "watched_source_checkpoint_hashes_unchanged=",
        "YES" if hashes_after == hashes_before else "NO",
    )
    print("candidate_artifact_written=NO")
    print("runtime_wiring_modified=NO")
    print("training_performed=NO")
    print("reference_modified=NO")

    print("\n========== AUTHORITATIVE COMPOSITION V7 VERDICT ==========")
    print("OOS_REMAINS_FROZEN=", "YES" if oos_pass else "NO")
    print("SELECTED_POLICY=", selected_policy)
    print("ZERO_BASELINE_RIGHT_REGRESSION=", "YES" if selected["zero_reg"] else "NO")
    print("FRESH_EXACT=", "YES" if selected["fresh_exact"] else "NO")
    print(
        "STABLE_NON_DEGRADING_ESTABLISHED_EXPOSED=",
        "YES" if selected["stable_non_degrading"] else "NO",
    )
    print("PRIMARY_BLOCKER=", primary)
    print("COMPOSITION_V7_VERDICT=", verdict)
    print("NEXT_ACTION=", next_action)
    print("composition_v7_unclear_ownership_proof_completed=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
