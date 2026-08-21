#!/usr/bin/env python3
"""VoiceProbe SemanticLab Level 2 architecture candidate V13.2 FINAL-CORRECTION.

V13.2 is a narrow post-normalization wrapper over V13.1 based only on the
EXPOSED/DEVELOPMENT V13 fresh 9-case residual audit.

No specialist retraining.

Final correction mechanisms
---------------------------
1. Active completed-reschedule confirmation:
   "I moved/rescheduled the visit/appointment to ..." => confirmed reschedule.

2. Negative availability prefix axis preservation:
   concrete provider/day/time values in a negative prefix remain failed even
   when V12/V13 add spoken-clock failure completion.

3. Appointment-search target cleanup:
   phrases such as "search other appointment dates" and
   "look for another appointment provider" use the explicit noun after
   "appointment" as the proposal axis; the word "appointment" itself does not
   imply time_of_day.

4. Relative CHECK axis disambiguation:
   "earlier/later today", dayparts, and "in the day" => time_of_day;
   explicit dates/days/weeks => day.

5. Appointment-state negation precedence:
   "There is no appointment showing/listed..." => appointment_missing and no
   availability scheduling failure.

6. Provider-pronoun inventory reference precedence:
   inventory questions with he/she refer to prior_provider only when a
   provider antecedent exists; otherwise reference=none even if a day/time is
   present in context.

This file does not modify V13.1, V12, production source, runtime wiring,
v0.17, telephony, or timing.
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
V13_1_FILE = HERE / "voiceprobe_semanticlab_v2_post_fresh_architecture_v13_1.py"

if not V13_1_FILE.is_file():
    raise SystemExit(f"Missing V13.1 companion file: {V13_1_FILE}")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v13 = load_mod("l2_v13_2_v13_1", V13_1_FILE)
v12 = v13.v12
v11 = v13.v11
v10 = v13.v10
v8 = v13.v8
v6 = v13.v6
v5 = v13.v5
v2 = v13.v2
base = v13.base


ACTIVE_RESCHEDULE_CONFIRMED_RE = re.compile(
    r"^\s*i(?:'ve|\s+have)?\s+"
    r"(?:moved|rescheduled)\s+"
    r"(?:the\s+|your\s+)?(?:appointment|visit)\b",
    re.IGNORECASE,
)

NEGATIVE_AVAILABILITY_STATUS_RE = re.compile(
    r"\b(?:full|booked|taken|unavailable|not\s+available|not\s+open|closed)\b",
    re.IGNORECASE,
)

APPOINTMENT_SEARCH_TARGET_RE = re.compile(
    r"^\s*(?:can|could|should|may|would)\s+i\s+"
    r"(?:look\s+for|search)\s+"
    r"(?:a\s+|an\s+)?(?:different|another|other)\s+"
    r"appointment\s+(?P<tail>.+?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)

RELATIVE_CHECK_TIME_CONTEXT_RE = re.compile(
    r"\b(?:today|tonight|this\s+(?:morning|afternoon|evening)|"
    r"in\s+the\s+(?:morning|afternoon|evening|day)|"
    r"(?:morning|afternoon|evening|noon|midday))\b",
    re.IGNORECASE,
)

RELATIVE_CHECK_DAY_CONTEXT_RE = re.compile(
    r"\b(?:another\s+day|different\s+day|"
    r"earlier\s+day|later\s+day|"
    r"earlier\s+date|later\s+date|"
    r"next\s+week|this\s+week|"
    r"earlier\s+week|later\s+week|"
    r"dates?|days?|weeks?)\b",
    re.IGNORECASE,
)

APPOINTMENT_MISSING_RE = re.compile(
    r"^\s*(?:"
    r"there\s+(?:is|isn't|isnt)\s+no?\s*|"
    r"there\s+is\s+no\s+|"
    r"i\s+(?:do\s+not|don't|dont)\s+(?:see|find)\s+|"
    r"no\s+(?:current\s+)?"
    r")"
    r"(?:current\s+)?appointment\b"
    r".{0,45}\b(?:showing|listed|found|on\s+(?:the\s+)?account|here)\b",
    re.IGNORECASE,
)

PROVIDER_PRONOUN_INVENTORY_RE = re.compile(
    r"^\s*(?:what|which)\s+other\s+"
    r"(?:appointment\s+)?(?:openings?|times?|slots?|availability)\b"
    r".{0,45}\bdoes\s+(?:he|she)\s+have\b",
    re.IGNORECASE,
)


def _clear_schedule():
    return {
        "failed_constraints": (),
        "proposed_changes": (),
        "retained_constraints": (),
    }


def _search_target_axes(tail: str):
    t = tail.casefold()
    axes = set()
    if re.search(r"\b(?:provider|providers|doctor|doctors|physician|physicians)\b", t):
        axes.add("provider")
    if re.search(r"\b(?:date|dates|day|days)\b", t):
        axes.add("day")
    if re.search(r"\b(?:time|times|slot|slots|morning|afternoon|evening)\b", t):
        axes.add("time_of_day")
    return axes


def _negative_prefix_value_axes(prefix: str):
    if not NEGATIVE_AVAILABILITY_STATUS_RE.search(prefix):
        return set()
    return set(v11._axes_from_values(prefix)) | set(v13._clock_values(prefix) and {"time_of_day"} or set())


def _relative_check_axes(clean: str, current):
    # Start with already-established proposal evidence, but then disambiguate
    # day/time from explicit relative context.
    axes = {x.value for x in current.proposed_changes}

    if RELATIVE_CHECK_TIME_CONTEXT_RE.search(clean):
        axes.add("time_of_day")
        # "in the day" / "today" is a time-of-day range, not calendar-day
        # change. Only preserve day if there is an independent explicit
        # calendar-date cue.
        independent_day = re.search(
            r"\b(?:date|dates|next\s+week|this\s+week|another\s+day|"
            r"different\s+day)\b",
            clean,
            re.IGNORECASE,
        )
        if not independent_day:
            axes.discard("day")

    if RELATIVE_CHECK_DAY_CONTEXT_RE.search(clean):
        # "in the day" is handled above as time_of_day.
        if not re.search(r"\bin\s+the\s+day\b", clean, re.IGNORECASE):
            axes.add("day")

    return axes


def construct_v13_2_frames(runtime, result, checkpoints, v2_frames, v2_schedules):
    (
        frames13,
        schedules13,
        diag,
        constructor_errors,
    ) = v13.construct_v13_frames(
        runtime,
        result,
        checkpoints,
        v2_frames,
        v2_schedules,
    )

    frames = []
    schedules = []

    for i, (turn, frame, sched13) in enumerate(
        zip(runtime, frames13, schedules13)
    ):
        clean = v5.strip_fillers(turn.utterance)
        current = frame
        current_sched = dict(sched13)

        # --------------------------------------------------------------
        # 1. Active completed reschedule.
        # --------------------------------------------------------------
        if ACTIVE_RESCHEDULE_CONFIRMED_RE.search(clean):
            current_sched = _clear_schedule()
            current = v8._rebuild_v7(
                current,
                speech_act="confirmation",
                topic="transaction",
                requested_fact="",
                schedule=current_sched,
                transaction_operation="reschedule",
                transaction_signal="confirmed",
                reference="none",
                selected_option="",
                ambiguity_kind="none",
                ambiguity_candidates=(),
            )
            diag.hit(i, "v13_2_active_reschedule_confirmed")

        # --------------------------------------------------------------
        # 2. Preserve all concrete failed axes from negative fallback prefix.
        # --------------------------------------------------------------
        fallback = v11.FALLBACK_ACTION_RE.search(clean)
        if (
            fallback
            and current.topic.value == "availability"
            and not v10.KEEP_TRY_RE.search(clean)
        ):
            prefix = clean[:fallback.start()].strip(" ;,.")
            prefix_axes = _negative_prefix_value_axes(prefix)
            if prefix_axes:
                failed = {x.value for x in current.failed_constraints} | prefix_axes
                proposed = {x.value for x in current.proposed_changes}
                retained = {x.value for x in current.retained_constraints}

                current_sched = {
                    "failed_constraints": base.order_axes(failed),
                    "proposed_changes": base.order_axes(proposed),
                    "retained_constraints": base.order_axes(retained),
                }
                current = v8._rebuild_v7(
                    current,
                    schedule=current_sched,
                )
                diag.hit(i, "v13_2_negative_prefix_axis_preservation")

        # --------------------------------------------------------------
        # 3. Explicit appointment SEARCH/LOOK target noun owns proposal axes.
        # --------------------------------------------------------------
        m_target = APPOINTMENT_SEARCH_TARGET_RE.search(clean)
        if m_target:
            axes = _search_target_axes(m_target.group("tail"))
            if axes:
                current_sched = {
                    "failed_constraints": tuple(
                        x.value for x in current.failed_constraints
                    ),
                    "proposed_changes": base.order_axes(axes),
                    "retained_constraints": (),
                }
                current = v8._rebuild_v7(
                    current,
                    speech_act="offer",
                    topic="availability",
                    requested_fact="",
                    schedule=current_sched,
                    transaction_operation="none",
                    transaction_signal="none",
                    reference="none",
                    selected_option="",
                    ambiguity_kind="none",
                    ambiguity_candidates=(),
                )
                diag.hit(i, "v13_2_appointment_search_target_axis")

        # --------------------------------------------------------------
        # 4. Relative CHECK day-vs-time disambiguation.
        # --------------------------------------------------------------
        if (
            v13.RELATIVE_CHECK_RE.search(clean)
            and not v12.EXPLICIT_TRANSACTION_SEARCH_RE.search(clean)
        ):
            axes = _relative_check_axes(clean, current)
            if axes:
                current_sched = {
                    "failed_constraints": tuple(
                        x.value for x in current.failed_constraints
                    ),
                    "proposed_changes": base.order_axes(axes),
                    "retained_constraints": (),
                }
                current = v8._rebuild_v7(
                    current,
                    speech_act="offer",
                    topic="availability",
                    requested_fact="",
                    schedule=current_sched,
                    transaction_operation="none",
                    transaction_signal="none",
                    reference="none",
                    selected_option="",
                    ambiguity_kind="none",
                    ambiguity_candidates=(),
                )
                diag.hit(i, "v13_2_relative_check_axis_disambiguation")

        # --------------------------------------------------------------
        # 5. Appointment-state missing precedence over date availability.
        # --------------------------------------------------------------
        if APPOINTMENT_MISSING_RE.search(clean):
            current_sched = _clear_schedule()
            current = v8._rebuild_v7(
                current,
                speech_act="statement",
                topic="appointment_state",
                requested_fact="",
                schedule=current_sched,
                record_claims=("appointment_missing",),
                transaction_operation="none",
                transaction_signal="none",
                reference="none",
                selected_option="",
                ambiguity_kind="none",
                ambiguity_candidates=(),
            )
            diag.hit(i, "v13_2_appointment_missing_precedence")

        # --------------------------------------------------------------
        # 6. Provider-pronoun inventory reference authority.
        # --------------------------------------------------------------
        if PROVIDER_PRONOUN_INVENTORY_RE.search(clean):
            has_provider = v10._provider_in_context(turn)
            current_sched = _clear_schedule()
            current = v8._rebuild_v7(
                current,
                speech_act="question",
                topic="availability",
                requested_fact="",
                schedule=current_sched,
                transaction_operation="none",
                transaction_signal="none",
                reference="prior_provider" if has_provider else "none",
                selected_option="",
                ambiguity_kind="none",
                ambiguity_candidates=(),
            )
            diag.hit(
                i,
                "v13_2_provider_inventory_with_antecedent"
                if has_provider
                else "v13_2_provider_inventory_without_antecedent",
            )

        frames.append(current)
        schedules.append(current_sched)

    return frames, schedules, diag, constructor_errors
