#!/usr/bin/env python3
"""VoiceProbe SemanticLab Level 2 architecture candidate V10.

V10 is a minimal narrowing of two overbroad V9 post-normalization rules over V8, derived only from the
EXPOSED/DEVELOPMENT fresh-adversarial diagnostic.

No specialist retraining.

General additions
-----------------
1. Provider-pronoun availability questions:
   - with a provider antecedent => prior_provider + question
   - without a provider antecedent => reference none
   - do not hallucinate scheduling proposals from the referenced day/time

2. Temporal ambiguity morphology and axis gating:
   vague moved/pushed/shifted earlier/later => temporal ambiguity;
   explicit time/day axis => availability proposal, not ambiguity;
   "or did you mean another day" preserves temporal ambiguity.

3. Clarification firewall:
   clarification speech acts cannot carry scheduling proposal/failure/retention.

4. Mutation permission normalization:
   "Would you like me to move/book/cancel/keep..." => transaction question
   with permission_request, scheduling cleared.

5. KEEP state cleanup:
   state-change KEEP normalization cannot retain stale record-existence claims.

6. Negative-fallback clause authority:
   "<failed slot>; keep X and try Y" restores failed axes, retains X,
   proposes axes expressed by Y, and normalizes speech act to offer.

7. Patient-fact lexical extensions:
   "brought you into the clinic" => complaint;
   polite "Could you provide/give..." fact prompts => request.

8. Prior-option interrogative guard:
   interrogative option comparisons/replacements cannot commit a selection.

9. Contextless deictic reference guard:
   "anything else around that time/day/provider" without context cannot
   manufacture scheduling changes.

10. Day-option earlier/later resolution:
    two weekday alternatives can resolve "the earlier/later one".

V10 residual corrections:
11. Mutation-permission normalization defers to an already-coherent V8
    availability fallback when both proposed and retained scheduling axes
    are present.
12. Contextless deictic cleanup fires only when the utterance itself contains
    no explicit lexical scheduling proposal axis.

This file does not modify V8, production source, runtime wiring, v0.17,
telephony, or timing.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
V8_FILE = HERE / "voiceprobe_semanticlab_v2_post_final_architecture_v8.py"

if not V8_FILE.is_file():
    raise SystemExit(f"Missing V8 companion file: {V8_FILE}")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v8 = load_mod("l2_v9_v8", V8_FILE)
v6 = v8.v6
v5 = v8.v5
v2 = v8.v2
base = v8.base


TEMPORAL_VAGUE_RE = re.compile(
    r"\b(?:make|move|moved|push|pushed|shift|shifted|change|do)\b"
    r".{0,55}\b(?:earlier|later)\b|"
    r"\b(?:earlier|later)\b.{0,55}"
    r"\b(?:make|move|moved|push|pushed|shift|shifted|change|do)\b",
    re.IGNORECASE,
)

TEMPORAL_META_DISAMBIG_RE = re.compile(
    r"\bor\s+(?:did|do|are|were)\s+you\s+mean\b.{0,40}"
    r"\b(?:another\s+day|a\s+different\s+day|the\s+day)\b",
    re.IGNORECASE,
)

PROVIDER_PRONOUN_RE = re.compile(r"\b(?:he|she|him|her)\b", re.IGNORECASE)

PROVIDER_AVAIL_QUESTION_RE = re.compile(
    r"^\s*(?:"
    r"does\s+(?:he|she)\b|"
    r"can\s+(?:he|she)\b|"
    r"could\s+(?:he|she)\b|"
    r"would\s+(?:he|she)\b|"
    r"what\s+else\s+does\s+(?:he|she)\b|"
    r"anything\b.{0,35}\bfor\s+(?:him|her)\b|"
    r"what\s+about\s+(?:him|her)\b"
    r")",
    re.IGNORECASE,
)

MUTATION_PERMISSION_RE = re.compile(
    r"^\s*(?:"
    r"would\s+you\s+like\s+me\s+to|"
    r"do\s+you\s+want\s+me\s+to|"
    r"may\s+i|can\s+i|could\s+i|should\s+i"
    r")\s+"
    r"(?P<verb>book|schedule|reschedule|move|cancel|keep|leave)\b",
    re.IGNORECASE,
)

POLITE_FACT_REQUEST_RE = re.compile(
    r"^\s*could\s+you\s+(?:please\s+)?"
    r"(?:provide|give|tell)\b",
    re.IGNORECASE,
)

COMPLAINT_INTO_RE = re.compile(
    r"\bwhat\s+(?:brought|brings)\s+you\s+"
    r"(?:in\b|into\b.{0,24}\b(?:clinic|office|practice|hospital)\b)",
    re.IGNORECASE,
)

OPTION_INTERROGATIVE_RE = re.compile(
    r"^\s*(?:can|could|would|will|does|is|are)\b",
    re.IGNORECASE,
)

CONTEXTLESS_DEICTIC_RE = re.compile(
    r"\b(?:that|the)\s+(?:time|day|provider)\b|"
    r"\baround\s+that\s+time\b|"
    r"\bthat\s+day\b",
    re.IGNORECASE,
)

KEEP_TRY_RE = re.compile(
    r"\bkeep\b(?P<keep>.+?)\band\s+try\b(?P<try>.+)$",
    re.IGNORECASE,
)

WEEKDAY_ONLY_RE = re.compile(
    r"^\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*$",
    re.IGNORECASE,
)

EARLIER_ONE_RE = re.compile(r"\bearlier\s+(?:one|option|choice)\b", re.IGNORECASE)
LATER_ONE_RE = re.compile(r"\blater\s+(?:one|option|choice)\b", re.IGNORECASE)

WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _provider_in_context(turn) -> bool:
    return any(
        re.search(r"\bdr\.?\s+[A-Za-z][A-Za-z'-]*", str(x), re.IGNORECASE)
        for x in turn.context
    )


def _mutation_op(verb: str) -> str:
    low = verb.casefold()
    if low in {"book", "schedule"}:
        return "book"
    if low in {"reschedule", "move"}:
        return "reschedule"
    if low == "cancel":
        return "cancel"
    if low in {"keep", "leave"}:
        return "keep"
    return "none"


def _schedule_of(frame):
    return {
        "failed_constraints": tuple(x.value for x in frame.failed_constraints),
        "proposed_changes": tuple(x.value for x in frame.proposed_changes),
        "retained_constraints": tuple(x.value for x in frame.retained_constraints),
    }


def _clear_schedule():
    return {
        "failed_constraints": (),
        "proposed_changes": (),
        "retained_constraints": (),
    }


def _axes_in_try_clause(text: str):
    axes = set()
    lexical = v2.proposed_axes_from_text(text)
    axes.update(lexical)

    vals = v6._axis_values(text)
    if vals["day"]:
        axes.add("day")
    if vals["time_of_day"]:
        axes.add("time_of_day")
    if vals["provider"]:
        axes.add("provider")

    return axes


def _weekday_selection(options, earlier: bool):
    if len(options) != 2:
        return ""
    parsed = []
    for opt in options:
        m = WEEKDAY_ONLY_RE.match(str(opt))
        if not m:
            return ""
        parsed.append((WEEKDAY_INDEX[m.group(1).casefold()], str(opt)))
    parsed.sort(key=lambda x: x[0])
    return parsed[0][1] if earlier else parsed[-1][1]


def construct_v10_frames(runtime, result, checkpoints, v2_frames, v2_schedules):
    (
        frames8,
        schedules8,
        diag,
        constructor_errors,
    ) = v8.construct_v8_frames(
        runtime,
        result,
        checkpoints,
        v2_frames,
        v2_schedules,
    )

    out_frames = []
    out_schedules = []

    for i, (turn, v2_frame, frame, sched8) in enumerate(
        zip(runtime, v2_frames, frames8, schedules8)
    ):
        clean = v5.strip_fillers(turn.utterance)
        current = frame
        current_sched = dict(sched8)

        # --------------------------------------------------------------
        # 1. Complaint morphology extension.
        # --------------------------------------------------------------
        if COMPLAINT_INTO_RE.search(clean):
            current_sched = _clear_schedule()
            current = v8._rebuild_v7(
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
            diag.hit(i, "v9_complaint_into_clinic")

        # --------------------------------------------------------------
        # 2. Polite fact request act.
        # --------------------------------------------------------------
        if (
            current.topic.value == "patient_fact"
            and current.requested_fact
            and POLITE_FACT_REQUEST_RE.search(clean)
        ):
            current = v8._rebuild_v7(
                current,
                speech_act="request",
            )
            diag.hit(i, "v9_polite_fact_request")

        # --------------------------------------------------------------
        # 3. Clarification firewall.
        # --------------------------------------------------------------
        if current.speech_act.value == "clarification":
            if any(current_sched.values()):
                current_sched = _clear_schedule()
                current = v8._rebuild_v7(
                    current,
                    schedule=current_sched,
                )
                diag.hit(i, "v9_clarification_scheduling_firewall")

        # --------------------------------------------------------------
        # 4. Mutation permission normalization.
        # --------------------------------------------------------------
        m_perm = MUTATION_PERMISSION_RE.search(clean)
        availability_retention_fallback = bool(
            current.topic.value == "availability"
            and current.proposed_changes
            and current.retained_constraints
        )
        if m_perm and not availability_retention_fallback:
            op = _mutation_op(m_perm.group("verb"))
            current_sched = _clear_schedule()
            current = v8._rebuild_v7(
                current,
                speech_act="question",
                topic="transaction",
                schedule=current_sched,
                transaction_operation=op,
                transaction_signal="permission_request",
                ambiguity_kind="none",
                ambiguity_candidates=(),
                reference="none",
                selected_option="",
            )
            diag.hit(i, "v10_mutation_permission_normalization")
        elif m_perm and availability_retention_fallback:
            diag.hit(i, "v10_mutation_permission_defers_to_availability_retention")

        # --------------------------------------------------------------
        # 5. Temporal ambiguity / explicit axis boundary.
        # --------------------------------------------------------------
        lexical_axes = set(v2.proposed_axes_from_text(clean))
        temporal_vague = bool(TEMPORAL_VAGUE_RE.search(clean))
        meta_disambig = bool(TEMPORAL_META_DISAMBIG_RE.search(clean))

        if (
            not turn.context
            and temporal_vague
            and not m_perm
            and (not lexical_axes or meta_disambig)
        ):
            current_sched = _clear_schedule()
            current = v8._rebuild_v7(
                current,
                speech_act="question",
                topic="other",
                schedule=current_sched,
                transaction_operation="none",
                transaction_signal="none",
                reference="none",
                ambiguity_kind="temporal_reference",
                ambiguity_candidates=("time_of_day", "day"),
                selected_option="",
            )
            diag.hit(i, "v9_temporal_ambiguity_morphology")

        elif (
            not turn.context
            and temporal_vague
            and lexical_axes
            and not meta_disambig
            and not m_perm
        ):
            current_sched = {
                "failed_constraints": (),
                "proposed_changes": base.order_axes(lexical_axes),
                "retained_constraints": (),
            }
            current = v8._rebuild_v7(
                current,
                speech_act="offer",
                topic="availability",
                schedule=current_sched,
                transaction_operation="none",
                transaction_signal="none",
                reference="none",
                ambiguity_kind="none",
                ambiguity_candidates=(),
                selected_option="",
            )
            diag.hit(i, "v9_explicit_temporal_axis_over_ambiguity")

        # --------------------------------------------------------------
        # 6. Provider-pronoun question boundary.
        # --------------------------------------------------------------
        if PROVIDER_PRONOUN_RE.search(clean) and PROVIDER_AVAIL_QUESTION_RE.search(clean):
            has_provider = _provider_in_context(turn)
            current_sched = _clear_schedule()
            current = v8._rebuild_v7(
                current,
                speech_act="question",
                topic="availability",
                schedule=current_sched,
                selected_option="",
                reference="prior_provider" if has_provider else "none",
                ambiguity_kind="none",
                ambiguity_candidates=(),
            )
            diag.hit(
                i,
                "v9_provider_pronoun_question_with_antecedent"
                if has_provider
                else "v9_provider_pronoun_without_antecedent",
            )

        # --------------------------------------------------------------
        # 7. Prior-option interrogatives cannot commit.
        # --------------------------------------------------------------
        if (
            current.reference.value == "prior_option"
            and OPTION_INTERROGATIVE_RE.search(clean)
        ):
            current = v8._rebuild_v7(
                current,
                speech_act="question",
                topic="availability",
                selected_option="",
            )
            diag.hit(i, "v9_prior_option_interrogative_no_commit")

        # --------------------------------------------------------------
        # 8. KEEP state mutation clears stale record claims.
        # --------------------------------------------------------------
        if (
            current.topic.value == "transaction"
            and current.transaction_operation.value == "keep"
            and (
                v8.KEEP_CONFIRMED_STATE_RE.search(clean)
                or v8.KEEP_PROPOSED_STATE_RE.search(clean)
            )
            and current.record_claims
        ):
            current = v8._rebuild_v7(
                current,
                record_claims=(),
            )
            diag.hit(i, "v9_keep_state_record_claim_cleanup")

        # --------------------------------------------------------------
        # 9. Negative slot + explicit keep/try clause authority.
        # --------------------------------------------------------------
        keep_try = KEEP_TRY_RE.search(clean)
        negative_axes = set(v2.failed_axes_from_negative_clauses(clean))
        if (
            current.topic.value == "availability"
            and keep_try
            and negative_axes
        ):
            retained = set(v2.retained_axes_from_text(clean, set()))
            proposed = _axes_in_try_clause(keep_try.group("try"))
            proposed -= retained

            current_sched = {
                "failed_constraints": base.order_axes(negative_axes),
                "proposed_changes": base.order_axes(proposed),
                "retained_constraints": base.order_axes(retained),
            }
            current = v8._rebuild_v7(
                current,
                speech_act="offer",
                topic="availability",
                schedule=current_sched,
                transaction_operation="none",
                transaction_signal="none",
                reference="none",
                ambiguity_kind="none",
                ambiguity_candidates=(),
                selected_option="",
            )
            diag.hit(i, "v9_negative_keep_try_clause_authority")

        # --------------------------------------------------------------
        # 10. Contextless deictic reference cannot create a proposal.
        # --------------------------------------------------------------
        if (
            not turn.context
            and current.reference.value == "none"
            and CONTEXTLESS_DEICTIC_RE.search(clean)
            and current.topic.value == "availability"
            and current.proposed_changes
            and not lexical_axes
        ):
            current_sched = _clear_schedule()
            current = v8._rebuild_v7(
                current,
                schedule=current_sched,
            )
            diag.hit(i, "v10_contextless_deictic_no_schedule_without_lexical_axis")

        # --------------------------------------------------------------
        # 11. Earlier/later among two weekday alternatives.
        # --------------------------------------------------------------
        if (
            current.reference.value == "prior_option"
            and not current.selected_option
            and turn.context
        ):
            options = v8._two_context_options(turn)
            selected = ""
            if EARLIER_ONE_RE.search(clean):
                selected = _weekday_selection(options, earlier=True)
            elif LATER_ONE_RE.search(clean):
                selected = _weekday_selection(options, earlier=False)

            if selected:
                current = v8._rebuild_v7(
                    current,
                    speech_act="confirmation",
                    topic="availability",
                    selected_option=selected,
                )
                diag.hit(i, "v9_weekday_temporal_option_selection")

        out_frames.append(current)
        out_schedules.append(current_sched)

    return out_frames, out_schedules, diag, constructor_errors
