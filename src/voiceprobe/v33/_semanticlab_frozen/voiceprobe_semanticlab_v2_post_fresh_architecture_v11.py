#!/usr/bin/env python3
"""VoiceProbe SemanticLab Level 2 architecture candidate V11.

V11 is a narrow post-normalization wrapper over V10 based only on
EXPOSED/DEVELOPMENT fresh-generalization diagnostics.

No specialist retraining.

General additions
-----------------
1. Scheduling keep/try clause authority:
   "keep <day/time/provider> and try <day/time/provider>" is availability
   scheduling even if a transaction KEEP classifier fires. Retention is
   extracted clause-locally from concrete axis values, including provider
   names.

2. Provider-pronoun availability-question extension:
   supports "What other openings does she/he have?" with provider antecedent
   precedence and contextless reference safety.

3. Deictic alternative-question normalization:
   "anything else / what else / are there other openings ... that time/day"
   is a question, not a scheduling proposal. Reference is preserved when
   context resolves it; contextless turns remain reference none.

4. Failure/fallback negative-clause boundary:
   ASR text such as "nothing is open at that time should I try another day"
   derives failed axes only from the negative prefix, not the fallback clause.

5. Lexical clarification precedence:
   explicit repeat requests suppress scheduling spillover.

6. Final interrogative option guard:
   interrogative prior-option turns cannot be re-committed by later
   earlier/later selection logic.

7. Broad availability-check normalization:
   "Can I check something later/earlier ..." remains availability scheduling,
   while frozen transaction search forms such as "... for you" stay search
   with transaction_signal none.

8. Clause-local mutation permission:
   a mutation request after an availability failure
   ("Dr. X is full; should I move your visit to Dr. Y?") becomes transaction
   permission and clears inherited scheduling failures.

This file does not modify V10, V8, production source, runtime wiring, v0.17,
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
V10_FILE = HERE / "voiceprobe_semanticlab_v2_post_fresh_architecture_v10.py"

if not V10_FILE.is_file():
    raise SystemExit(f"Missing V10 companion file: {V10_FILE}")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v10 = load_mod("l2_v11_v10", V10_FILE)
v8 = v10.v8
v6 = v10.v6
v5 = v10.v5
v2 = v10.v2
base = v10.base


PROVIDER_OTHER_OPENINGS_RE = re.compile(
    r"^\s*what\s+(?:"
    r"else|other\s+(?:openings?|times?|slots?|availability)"
    r")\s+does\s+(?:he|she)\s+have\b",
    re.IGNORECASE,
)

DEICTIC_ALT_QUESTION_RE = re.compile(
    r"^\s*(?:"
    r"anything\s+else|"
    r"what\s+else|"
    r"are\s+there\s+(?:any\s+)?other\s+(?:openings?|options?|times?|slots?)"
    r")\b.{0,70}\b(?:that\s+(?:time|day)|around\s+that\s+time|"
    r"near\s+that\s+time)\b",
    re.IGNORECASE,
)

CLARIFICATION_REPEAT_RE = re.compile(
    r"^\s*(?:sorry[\s,]*)?(?:"
    r"(?:could|can|would|will)\s+you\s+repeat\b|"
    r"repeat\b|"
    r"say\b.{0,30}\bagain\b"
    r")",
    re.IGNORECASE,
)

FALLBACK_ACTION_RE = re.compile(
    r"\b(?:can|could|should|would|may)\s+i\s+"
    r"(?:try|check|look|search)\b",
    re.IGNORECASE,
)

CLAUSE_MUTATION_PERMISSION_RE = re.compile(
    r"\b(?:"
    r"would\s+you\s+like\s+me\s+to|"
    r"do\s+you\s+want\s+me\s+to|"
    r"may\s+i|can\s+i|could\s+i|should\s+i"
    r")\s+"
    r"(?P<verb>book|schedule|reschedule|move|cancel|keep|leave)\b",
    re.IGNORECASE,
)

BROAD_AVAIL_CHECK_RE = re.compile(
    r"^\s*(?:can|could|should|may)\s+i\s+check\s+"
    r"(?:something\s+)?(?:later|earlier)\b|"
    r"^\s*(?:can|could|should|may)\s+i\s+check\s+"
    r"(?:later|earlier)\s+(?:appointments?|dates?|times?|slots?)\b",
    re.IGNORECASE,
)


def _clear_schedule():
    return {
        "failed_constraints": (),
        "proposed_changes": (),
        "retained_constraints": (),
    }


def _axes_from_values(text: str):
    axes = set()
    vals = v6._axis_values(text)
    if vals["day"]:
        axes.add("day")
    if vals["time_of_day"]:
        axes.add("time_of_day")
    if vals["provider"]:
        axes.add("provider")
    return axes


def _try_axes(text: str):
    axes = set(v2.proposed_axes_from_text(text))
    axes |= _axes_from_values(text)
    return axes


def _keep_axes(text: str):
    axes = set()
    # Retention lexicon catches phrases such as "keep the day".
    try:
        axes |= set(v2.retained_axes_from_text(text, set()))
    except Exception:
        pass
    # Concrete value extraction catches "keep Dr. Rivera", "keep Thursday",
    # and "keep 3 PM".
    axes |= _axes_from_values(text)
    return axes


def _failed_prefix_axes(text: str):
    m = FALLBACK_ACTION_RE.search(text)
    if not m:
        return set()
    prefix = text[:m.start()].strip(" ;,.")
    return set(v2.failed_axes_from_negative_clauses(prefix))


def _mutation_op(verb: str) -> str:
    return v10._mutation_op(verb)


def construct_v11_frames(runtime, result, checkpoints, v2_frames, v2_schedules):
    (
        frames10,
        schedules10,
        diag,
        constructor_errors,
    ) = v10.construct_v10_frames(
        runtime,
        result,
        checkpoints,
        v2_frames,
        v2_schedules,
    )

    frames = []
    schedules = []

    for i, (turn, frame, sched10) in enumerate(
        zip(runtime, frames10, schedules10)
    ):
        clean = v5.strip_fillers(turn.utterance)
        current = frame
        current_sched = dict(sched10)

        lexical_axes = set(v2.proposed_axes_from_text(clean))

        # --------------------------------------------------------------
        # 1. Explicit repeat requests are clarification, regardless of
        #    temporal words such as "later time".
        # --------------------------------------------------------------
        if CLARIFICATION_REPEAT_RE.search(clean):
            current_sched = _clear_schedule()
            current = v8._rebuild_v7(
                current,
                speech_act="clarification",
                topic="other",
                schedule=current_sched,
                transaction_operation="none",
                transaction_signal="none",
                reference="none",
                ambiguity_kind="none",
                ambiguity_candidates=(),
                selected_option="",
            )
            diag.hit(i, "v11_lexical_clarification_precedence")

        # --------------------------------------------------------------
        # 2. Provider-pronoun availability-question extension.
        # --------------------------------------------------------------
        if (
            v10.PROVIDER_PRONOUN_RE.search(clean)
            and PROVIDER_OTHER_OPENINGS_RE.search(clean)
        ):
            has_provider = v10._provider_in_context(turn)
            current_sched = _clear_schedule()
            current = v8._rebuild_v7(
                current,
                speech_act="question",
                topic="availability",
                schedule=current_sched,
                transaction_operation="none",
                transaction_signal="none",
                selected_option="",
                reference="prior_provider" if has_provider else "none",
                ambiguity_kind="none",
                ambiguity_candidates=(),
            )
            diag.hit(
                i,
                "v11_provider_other_openings_with_antecedent"
                if has_provider
                else "v11_provider_other_openings_without_antecedent",
            )

        # --------------------------------------------------------------
        # 3. Broad availability "check something later/earlier" must not
        #    be cross-promoted into frozen transaction search.
        # --------------------------------------------------------------
        if (
            current.transaction_operation.value == "search"
            and current.transaction_signal.value == "none"
            and BROAD_AVAIL_CHECK_RE.search(clean)
            and not re.search(r"\bfor\s+you\b", clean, re.IGNORECASE)
            and lexical_axes
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
            diag.hit(i, "v11_broad_check_is_availability")

        # --------------------------------------------------------------
        # 4. Scheduling keep/try clause authority. This runs even when
        #    V8/V10 previously converted the turn to transaction KEEP.
        # --------------------------------------------------------------
        keep_try = v10.KEEP_TRY_RE.search(clean)
        scheduling_keep_try = False
        if keep_try:
            retained = _keep_axes(keep_try.group("keep"))
            proposed = _try_axes(keep_try.group("try"))
            proposed -= retained

            if retained and proposed:
                scheduling_keep_try = True
                # Failure axes must come only from the text before "keep";
                # the try clause must not contaminate failed constraints.
                prefix = clean[:keep_try.start()].strip(" ;,.")
                failed = set(v2.failed_axes_from_negative_clauses(prefix))

                current_sched = {
                    "failed_constraints": base.order_axes(failed),
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
                diag.hit(i, "v11_scheduling_keep_try_clause_authority")

        # --------------------------------------------------------------
        # 5. Failure/fallback negative boundary for punctuation-free ASR.
        # --------------------------------------------------------------
        prefix_failed = _failed_prefix_axes(clean)
        if (
            current.topic.value == "availability"
            and prefix_failed
            and not scheduling_keep_try
        ):
            current_sched = {
                "failed_constraints": base.order_axes(prefix_failed),
                "proposed_changes": tuple(x.value for x in current.proposed_changes),
                "retained_constraints": tuple(x.value for x in current.retained_constraints),
            }
            current = v8._rebuild_v7(
                current,
                schedule=current_sched,
            )
            diag.hit(i, "v11_negative_failure_prefix_authority")

        # --------------------------------------------------------------
        # 6. Contextless/contextual deictic alternative questions.
        # --------------------------------------------------------------
        if DEICTIC_ALT_QUESTION_RE.search(clean):
            reference = current.reference.value
            if not turn.context:
                reference = "none"

            current_sched = _clear_schedule()
            current = v8._rebuild_v7(
                current,
                speech_act="question",
                topic="availability",
                schedule=current_sched,
                transaction_operation="none",
                transaction_signal="none",
                reference=reference,
                selected_option="",
                ambiguity_kind="none",
                ambiguity_candidates=(),
            )
            diag.hit(i, "v11_deictic_alternative_question")

        # --------------------------------------------------------------
        # 7. Clause-local transaction permission after a failed option.
        #    Preserve scheduling-retention fallbacks.
        # --------------------------------------------------------------
        m_mut = CLAUSE_MUTATION_PERMISSION_RE.search(clean)
        availability_retention_fallback = bool(
            current.topic.value == "availability"
            and current.proposed_changes
            and current.retained_constraints
        )

        if (
            m_mut
            and not scheduling_keep_try
            and not availability_retention_fallback
        ):
            op = _mutation_op(m_mut.group("verb"))
            current_sched = _clear_schedule()
            current = v8._rebuild_v7(
                current,
                speech_act="question",
                topic="transaction",
                schedule=current_sched,
                transaction_operation=op,
                transaction_signal="permission_request",
                reference="none",
                selected_option="",
                ambiguity_kind="none",
                ambiguity_candidates=(),
            )
            diag.hit(i, "v11_clause_mutation_permission")

        # --------------------------------------------------------------
        # 8. FINAL prior-option interrogative guard. Must run after
        #    weekday earlier/later selection from V10.
        # --------------------------------------------------------------
        if (
            current.reference.value == "prior_option"
            and v10.OPTION_INTERROGATIVE_RE.search(clean)
        ):
            current = v8._rebuild_v7(
                current,
                speech_act="question",
                topic="availability",
                selected_option="",
            )
            diag.hit(i, "v11_final_prior_option_interrogative_guard")

        frames.append(current)
        schedules.append(current_sched)

    return frames, schedules, diag, constructor_errors
