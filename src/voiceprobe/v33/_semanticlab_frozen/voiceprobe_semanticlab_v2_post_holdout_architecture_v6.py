#!/usr/bin/env python3
"""VoiceProbe SemanticLab Level 2 architecture candidate V6.

V6 is deliberately a NARROW POST-NORMALIZATION LAYER over V5.

It preserves V5's already-proven behavior and repairs only failure families
exposed by the 120-case DEVELOPMENT adversarial audit.

No specialist retraining is performed.

Architecture additions:
1. Availability retention/search guard:
   "keep Thursday and check another time" is an availability constraint change,
   not a transaction KEEP operation, when the frozen V2 frame already carries
   concrete proposed scheduling axes.

2. Explicit transaction-search boundary:
   progressive "I'm checking/searching..." statements with transaction-search
   cues, plus interrogative "Could I check <concrete slot> for you?", are
   transaction SEARCH. SEARCH remains non-state-changing and always has
   transaction_signal=none.

3. Confirmed KEEP state:
   "the appointment is staying/remains as scheduled" normalizes to KEEP rather
   than BOOK.

4. Clause-relative scheduling comparison:
   when a failed concrete slot is followed by a concrete fallback, compare
   values per axis. Same value => retained; changed value => proposed.
   This avoids whole-turn union errors such as:
     Thursday morning -> Thursday afternoon
     Thursday morning -> Friday morning

5. Complaint idiom precedence:
   conventional complaint questions such as "What brings you in..." outrank a
   spurious visit_type prediction caused merely by the word "visit".

This file does not modify V5 or production/runtime source.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
V5_FILE = HERE / "voiceprobe_semanticlab_v2_post_holdout_architecture_v5.py"

if not V5_FILE.is_file():
    raise SystemExit(f"Missing V5 companion file: {V5_FILE}")


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v5 = load_mod("l2_v6_v5", V5_FILE)
base = v5.base
v2 = v5.v2

from voiceprobe.v33.semantic_frame import (
    AmbiguityKind,
    ConstraintAxis,
    ReferenceKind,
    SemanticAmbiguity,
    SemanticFrame,
    SemanticTopic,
    SpeechAct,
    TransactionOperation,
    TransactionSignal,
)


COMPLAINT_IDIOM_RE = re.compile(
    r"\bwhat\s+brings\s+you\s+in\b|"
    r"\bwhat\s+are\s+we\s+seeing\s+you\s+for\b|"
    r"\bwhat\s+are\s+you\s+being\s+seen\s+for\b",
    re.IGNORECASE,
)

ALT_SEARCH_VERB_RE = re.compile(
    r"\b(?:check|try|look|search)\b",
    re.IGNORECASE,
)

PROGRESSIVE_SEARCH_RE = re.compile(
    r"^\s*(?:i(?:'m|\s+am)|we(?:'re|\s+are))\s+"
    r"(?:only\s+)?(?:checking|searching)\b",
    re.IGNORECASE,
)

PROGRESSIVE_SEARCH_TARGET_RE = re.compile(
    r"\b(?:whether\b.*\bavailable\b|availability\b|opening\b|"
    r"schedule\b|for\s+you\b)",
    re.IGNORECASE,
)

QUESTION_SEARCH_FOR_YOU_RE = re.compile(
    r"^\s*(?:can|could|should|may)\s+i\s+"
    r"(?:check|search)\b.{0,80}\bfor\s+you\b",
    re.IGNORECASE,
)

ALTERNATIVE_SEARCH_CUE_RE = re.compile(
    r"\b(?:another|different|else|same\s+appointment|later\s+appointments?)\b",
    re.IGNORECASE,
)

KEEP_CONFIRMED_RE = re.compile(
    r"\b(?:appointment|visit)\b.{0,40}\b"
    r"(?:staying|stays|remaining|remains|left)\b.{0,30}\b"
    r"(?:as\s+scheduled|as\s+is|in\s+place|unchanged)\b",
    re.IGNORECASE,
)

NEGATIVE_SLOT_RE = re.compile(
    r"\b(?:full|booked|taken|unavailable|no\s+openings?|nothing)\b",
    re.IGNORECASE,
)

FALLBACK_VERB_RE = re.compile(
    r"\b(?:try|check|look|search)\b",
    re.IGNORECASE,
)

DAY_RE = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

DAYPART_RE = re.compile(
    r"\b(?:morning|afternoon|evening|night|noon|midday)\b",
    re.IGNORECASE,
)

CLOCK_RE = re.compile(
    r"\b(?:1[0-2]|0?[1-9])(?::[0-5]\d)?\s*(?:a\.?m\.?|p\.?m\.?)\b",
    re.IGNORECASE,
)

PROVIDER_RE = re.compile(
    r"\bdr\.?\s+([A-Za-z][A-Za-z'-]*)",
    re.IGNORECASE,
)


def _norm_values(values):
    return {re.sub(r"\s+", " ", x.casefold().replace(".", "")).strip() for x in values}


def _axis_values(text: str):
    day = _norm_values(DAY_RE.findall(text))
    time_values = list(DAYPART_RE.findall(text)) + list(CLOCK_RE.findall(text))
    time = _norm_values(time_values)
    provider = _norm_values(PROVIDER_RE.findall(text))
    return {
        "day": day,
        "time_of_day": time,
        "provider": provider,
    }


def _semantic_clauses(text: str):
    # Protect Dr. before punctuation splitting.
    protected = re.sub(r"\bDr\.", "Dr§", text, flags=re.IGNORECASE)
    parts = re.split(r"(?:[!?;]\s*|\.\s+)", protected)
    return [p.replace("Dr§", "Dr.").strip() for p in parts if p.strip()]


def _clause_relative_schedule(text: str, sched: dict):
    clauses = _semantic_clauses(text)
    if len(clauses) < 2:
        return sched, False

    failure_idx = None
    for idx, clause in enumerate(clauses):
        if NEGATIVE_SLOT_RE.search(clause):
            failure_idx = idx
            break
    if failure_idx is None:
        return sched, False

    fallback_clause = None
    for clause in clauses[failure_idx + 1:]:
        if FALLBACK_VERB_RE.search(clause):
            fallback_clause = clause
            break
    if fallback_clause is None:
        return sched, False

    failure_clause = clauses[failure_idx]
    before = _axis_values(failure_clause)
    after = _axis_values(fallback_clause)

    comparable = set()
    same = set()
    changed = set()

    for axis in ("day", "time_of_day", "provider"):
        if before[axis] and after[axis]:
            comparable.add(axis)
            if before[axis] == after[axis]:
                same.add(axis)
            else:
                changed.add(axis)

    if not comparable:
        return sched, False

    proposed = set(sched["proposed_changes"]) - comparable
    retained = set(sched["retained_constraints"]) - comparable

    proposed |= changed
    retained |= same
    proposed -= retained

    corrected = {
        "failed_constraints": tuple(sched["failed_constraints"]),
        "proposed_changes": base.order_axes(proposed),
        "retained_constraints": base.order_axes(retained),
    }
    return corrected, corrected != sched


def _rebuild(
    frame,
    *,
    speech_act=None,
    topic=None,
    requested_fact=None,
    schedule=None,
    transaction_operation=None,
    transaction_signal=None,
    reference=None,
    ambiguity_kind=None,
    ambiguity_candidates=None,
    selected_option=None,
):
    sched = schedule or {
        "failed_constraints": tuple(x.value for x in frame.failed_constraints),
        "proposed_changes": tuple(x.value for x in frame.proposed_changes),
        "retained_constraints": tuple(x.value for x in frame.retained_constraints),
    }

    act = speech_act if speech_act is not None else frame.speech_act.value
    top = topic if topic is not None else frame.topic.value
    fact = requested_fact if requested_fact is not None else frame.requested_fact
    op = (
        transaction_operation
        if transaction_operation is not None
        else frame.transaction_operation.value
    )
    sig = (
        transaction_signal
        if transaction_signal is not None
        else frame.transaction_signal.value
    )
    ref = reference if reference is not None else frame.reference.value
    ak = (
        ambiguity_kind
        if ambiguity_kind is not None
        else frame.ambiguity.kind.value
    )
    ac = (
        tuple(ambiguity_candidates)
        if ambiguity_candidates is not None
        else tuple(frame.ambiguity.candidates)
    )
    sel = selected_option if selected_option is not None else frame.selected_option

    return SemanticFrame(
        raw_text=frame.raw_text,
        speech_act=SpeechAct(act),
        topic=SemanticTopic(top),
        requested_fact=fact,
        failed_constraints=tuple(
            ConstraintAxis(x) for x in sched["failed_constraints"]
        ),
        proposed_changes=tuple(
            ConstraintAxis(x) for x in sched["proposed_changes"]
        ),
        retained_constraints=tuple(
            ConstraintAxis(x) for x in sched["retained_constraints"]
        ),
        offered_options=frame.offered_options,
        selected_option=sel,
        record_claims=frame.record_claims,
        transaction_operation=TransactionOperation(op),
        transaction_signal=TransactionSignal(sig),
        reference=ReferenceKind(ref),
        ambiguity=SemanticAmbiguity(
            kind=AmbiguityKind(ak),
            candidates=ac,
            detail="",
        ),
    )


def construct_v6_frames(runtime, result, checkpoints, v2_frames, v2_schedules):
    (
        v5_frames,
        v5_schedules,
        diag,
        constructor_errors,
    ) = v5.construct_v5_frames(
        runtime,
        result,
        checkpoints,
        v2_frames,
        v2_schedules,
    )

    frames = []
    schedules = []

    for i, (turn, v2_frame, frame, sched) in enumerate(
        zip(runtime, v2_frames, v5_frames, v5_schedules)
    ):
        clean = v5.strip_fillers(turn.utterance)
        current = frame
        current_sched = dict(sched)

        # 1. Complaint idiom precedence.
        if COMPLAINT_IDIOM_RE.search(clean):
            current_sched = {
                "failed_constraints": (),
                "proposed_changes": (),
                "retained_constraints": (),
            }
            current = _rebuild(
                current,
                speech_act="question",
                topic="patient_fact",
                requested_fact="complaint",
                schedule=current_sched,
                transaction_operation="none",
                transaction_signal="none",
                reference="none",
                ambiguity_kind="none",
                ambiguity_candidates=(),
                selected_option="",
            )
            diag.hit(i, "v6_complaint_idiom_precedence")

        # 2. KEEP can be a retained availability axis rather than a mutation.
        #    Trust the already-structured V2 scheduling evidence when it says
        #    availability + concrete proposal, and V5 promoted only because
        #    the word "keep" appeared before a search verb.
        if (
            v2_frame.topic.value == "availability"
            and v2_frame.transaction_operation.value == "none"
            and v2_frame.proposed_changes
            and current.transaction_operation.value == "keep"
            and ALT_SEARCH_VERB_RE.search(clean)
        ):
            current_sched = {
                "failed_constraints": tuple(
                    x.value for x in v2_frame.failed_constraints
                ),
                "proposed_changes": tuple(
                    x.value for x in v2_frame.proposed_changes
                ),
                "retained_constraints": tuple(
                    x.value for x in v2_frame.retained_constraints
                ),
            }
            current = _rebuild(
                current,
                speech_act=v2_frame.speech_act.value,
                topic="availability",
                requested_fact=v2_frame.requested_fact,
                schedule=current_sched,
                transaction_operation="none",
                transaction_signal="none",
            )
            diag.hit(i, "v6_availability_keep_search_guard")

        # 3a. Progressive transaction SEARCH statement.
        progressive_search = bool(
            PROGRESSIVE_SEARCH_RE.search(clean)
            and PROGRESSIVE_SEARCH_TARGET_RE.search(clean)
        )
        if progressive_search:
            current_sched = {
                "failed_constraints": (),
                "proposed_changes": (),
                "retained_constraints": (),
            }
            current = _rebuild(
                current,
                speech_act="statement",
                topic="transaction",
                schedule=current_sched,
                transaction_operation="search",
                transaction_signal="none",
            )
            diag.hit(i, "v6_progressive_transaction_search")

        # 3b. Interrogative transaction SEARCH with concrete target "for you".
        #     Alternative/fallback searches remain availability semantics.
        question_search = bool(
            QUESTION_SEARCH_FOR_YOU_RE.search(clean)
            and not ALTERNATIVE_SEARCH_CUE_RE.search(clean)
            and v5.ENTITY_RE.search(clean)
        )
        if question_search:
            current_sched = {
                "failed_constraints": (),
                "proposed_changes": (),
                "retained_constraints": (),
            }
            current = _rebuild(
                current,
                speech_act="question",
                topic="transaction",
                schedule=current_sched,
                transaction_operation="search",
                transaction_signal="none",
            )
            diag.hit(i, "v6_question_transaction_search")

        # 4. Confirmed appointment state "staying/remains as scheduled" is KEEP.
        if (
            current.speech_act.value == "confirmation"
            and current.topic.value == "transaction"
            and KEEP_CONFIRMED_RE.search(clean)
        ):
            current = _rebuild(
                current,
                transaction_operation="keep",
                transaction_signal="confirmed",
            )
            diag.hit(i, "v6_confirmed_keep_state")

        # 5. Clause-relative scheduling comparison.
        if (
            current.topic.value == "availability"
            and v2.NEGATIVE_AVAILABILITY_RE.search(clean)
            and v5.generalized_offer_action(clean)
        ):
            corrected_sched, changed = _clause_relative_schedule(
                clean,
                current_sched,
            )
            if changed:
                current_sched = corrected_sched
                current = _rebuild(
                    current,
                    schedule=current_sched,
                )
                diag.hit(i, "v6_clause_relative_schedule_comparison")

        frames.append(current)
        schedules.append(current_sched)

    return frames, schedules, diag, constructor_errors
