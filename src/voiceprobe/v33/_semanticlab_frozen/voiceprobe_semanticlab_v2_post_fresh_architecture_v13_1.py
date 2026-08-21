#!/usr/bin/env python3
"""VoiceProbe SemanticLab Level 2 architecture candidate V13.1.

V13 is a narrow post-normalization wrapper over V12 based only on the
EXPOSED/DEVELOPMENT V12 fresh-128 residual audit.

No specialist retraining.

General additions
-----------------
1. Inventory-question grammar extension:
   supports "what/which other appointment times..." and provider-pronoun
   variants; clears false scheduling proposals.

2. Bare TRY multi-axis completion:
   combines lexical proposal evidence with concrete day/provider/time values
   in the TRY clause, so "try Friday with Dr. Singh" proposes both axes.

3. Relative CHECK modifier extension:
   supports "check something a little/a bit earlier/later ..." while preserving
   frozen explicit transaction-search forms such as "... for you" and
   "checking whether".

4. Spoken-clock fallback comparator:
   compares clock values before/after TRY/CHECK/LOOK/SEARCH. A changed clock
   becomes time_of_day proposal and stale time retention is removed; an
   unchanged clock remains retained.

5. Passive completed-reschedule normalization:
   "The appointment was moved/rescheduled to ..." is confirmation/transaction/
   reschedule/confirmed.

This file does not modify V12, V11, production source, runtime wiring, v0.17,
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
V12_FILE = HERE / "voiceprobe_semanticlab_v2_post_fresh_architecture_v12.py"

if not V12_FILE.is_file():
    raise SystemExit(f"Missing V12 companion file: {V12_FILE}")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v12 = load_mod("l2_v13_v12", V12_FILE)
v11 = v12.v11
v10 = v12.v10
v8 = v12.v8
v6 = v12.v6
v5 = v12.v5
v2 = v12.v2
base = v12.base


INVENTORY_QUESTION_RE = re.compile(
    r"^\s*(?:what|which)\s+other\s+"
    r"(?:appointment\s+)?"
    r"(?:openings?|times?|slots?|availability)\b"
    r"(?:.{0,60}\b(?:available|open|does\s+(?:he|she)\s+have|"
    r"do\s+you\s+have|are\s+there|have)\b)?",
    re.IGNORECASE,
)

RELATIVE_CHECK_RE = re.compile(
    r"^\s*(?:can|could|should|may)\s+i\s+check\s+"
    r"(?:something\s+)?"
    r"(?:(?:a\s+(?:little|bit))\s+)?"
    r"(?P<direction>earlier|later)\b",
    re.IGNORECASE,
)

RELATIVE_TIME_ANCHOR_RE = re.compile(
    r"\b(?:time|times|slot|slots|appointment|appointments|"
    r"morning|afternoon|evening|tonight|noon|midday)\b",
    re.IGNORECASE,
)

RELATIVE_DAY_ANCHOR_RE = re.compile(
    r"\b(?:day|days|date|dates|week|weeks)\b",
    re.IGNORECASE,
)

PASSIVE_RESCHEDULE_CONFIRMED_RE = re.compile(
    r"^\s*(?:the\s+)?(?:appointment|visit)\s+"
    r"(?:was|has\s+been)\s+"
    r"(?:moved|rescheduled)\b",
    re.IGNORECASE,
)

CLOCK_TOKEN_RE = re.compile(
    r"\b(?P<clock>"
    r"(?:\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<num_mer>a\.?\s*m\.?|p\.?\s*m\.?)"
    r"|"
    r"(?P<word>one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"\s*(?P<word_mer>a\.?\s*m\.?|p\.?\s*m\.?)"
    r")\b",
    re.IGNORECASE,
)

WORD_HOUR = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12,
}


def _clear_schedule():
    return {
        "failed_constraints": (),
        "proposed_changes": (),
        "retained_constraints": (),
    }


def _clock_values(text: str):
    values = []
    for m in CLOCK_TOKEN_RE.finditer(text):
        raw = m.group("clock").casefold()
        mer = (m.group("num_mer") or m.group("word_mer") or "").casefold()
        mer = "am" if "a" in mer else "pm"

        if m.group("word"):
            hour = WORD_HOUR[m.group("word").casefold()]
            minute = 0
        else:
            num_part = re.match(r"\d{1,2}", raw)
            hour = int(num_part.group(0))
            minute = int(m.group("minute") or 0)

        hour %= 12
        if mer == "pm":
            hour += 12
        values.append(hour * 60 + minute)
    return tuple(values)


def _general_proposal_axes(text: str):
    axes = set(v2.proposed_axes_from_text(text))
    axes |= set(v12._axes_with_spoken_time(text))

    # Relative lexical anchors that lower proposal extraction may miss.
    if re.search(r"\b(?:earlier|later)\b", text, re.IGNORECASE):
        if RELATIVE_DAY_ANCHOR_RE.search(text):
            axes.add("day")
        if RELATIVE_TIME_ANCHOR_RE.search(text):
            axes.add("time_of_day")
    return axes


def construct_v13_frames(runtime, result, checkpoints, v2_frames, v2_schedules):
    (
        frames12,
        schedules12,
        diag,
        constructor_errors,
    ) = v12.construct_v12_frames(
        runtime,
        result,
        checkpoints,
        v2_frames,
        v2_schedules,
    )

    frames = []
    schedules = []

    for i, (turn, frame, sched12) in enumerate(
        zip(runtime, frames12, schedules12)
    ):
        clean = v5.strip_fillers(turn.utterance)
        current = frame
        current_sched = dict(sched12)

        # --------------------------------------------------------------
        # 1. Inventory questions: no scheduling proposal.
        # --------------------------------------------------------------
        if INVENTORY_QUESTION_RE.search(clean):
            reference = current.reference.value
            if not turn.context:
                reference = "none"

            current_sched = _clear_schedule()
            current = v8._rebuild_v7(
                current,
                speech_act="question",
                topic="availability",
                requested_fact="",
                schedule=current_sched,
                transaction_operation="none",
                transaction_signal="none",
                reference=reference,
                selected_option="",
                ambiguity_kind="none",
                ambiguity_candidates=(),
            )
            diag.hit(i, "v13_inventory_question_extension")

        # --------------------------------------------------------------
        # 2. Bare TRY multi-axis completion.
        # --------------------------------------------------------------
        if (
            v12.BARE_TRY_RE.search(clean)
            and not v10.KEEP_TRY_RE.search(clean)
        ):
            axes = _general_proposal_axes(clean)
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
                diag.hit(i, "v13_bare_try_multi_axis_completion")

        # --------------------------------------------------------------
        # 3. "check something a little earlier/later" availability.
        # --------------------------------------------------------------
        m_check = RELATIVE_CHECK_RE.search(clean)
        if (
            m_check
            and not v12.EXPLICIT_TRANSACTION_SEARCH_RE.search(clean)
        ):
            axes = _general_proposal_axes(clean)

            # If the phrase is explicitly relative but lower extraction missed
            # the anchor, infer only from an explicit day/time noun/daypart.
            if not axes:
                if RELATIVE_DAY_ANCHOR_RE.search(clean):
                    axes.add("day")
                if RELATIVE_TIME_ANCHOR_RE.search(clean):
                    axes.add("time_of_day")

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
                diag.hit(i, "v13_relative_check_availability")

        # --------------------------------------------------------------
        # 4. Spoken/numeric clock fallback comparator.
        # --------------------------------------------------------------
        fallback = v11.FALLBACK_ACTION_RE.search(clean)
        if (
            fallback
            and current.topic.value == "availability"
            and not v10.KEEP_TRY_RE.search(clean)
        ):
            prefix = clean[:fallback.start()].strip(" ;,.")
            suffix = clean[fallback.start():]

            before_times = _clock_values(prefix)
            after_times = _clock_values(suffix)

            if before_times and after_times:
                proposed = {
                    x.value for x in current.proposed_changes
                }
                retained = {
                    x.value for x in current.retained_constraints
                }
                failed = {
                    x.value for x in current.failed_constraints
                }

                before = before_times[-1]
                after = after_times[-1]

                if before != after:
                    proposed.add("time_of_day")
                    retained.discard("time_of_day")
                    diag.hit(i, "v13_changed_clock_is_proposal")
                else:
                    proposed.discard("time_of_day")
                    retained.add("time_of_day")
                    diag.hit(i, "v13_same_clock_is_retained")

                # Keep any non-time proposal axes already established, and
                # ensure explicit suffix axes are not lost.
                suffix_axes = _general_proposal_axes(suffix)
                # Non-time axis comparison remains owned by the established
                # scheduling layer.  Do not turn an already-retained day or
                # provider into a simultaneous proposal merely because that
                # value is lexically present in the fallback suffix.
                proposed |= {
                    ax
                    for ax in suffix_axes
                    if ax != "time_of_day" and ax not in retained
                }

                current_sched = {
                    "failed_constraints": base.order_axes(failed),
                    "proposed_changes": base.order_axes(proposed),
                    "retained_constraints": base.order_axes(retained),
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

        # --------------------------------------------------------------
        # 5. Passive completed reschedule.
        # --------------------------------------------------------------
        if PASSIVE_RESCHEDULE_CONFIRMED_RE.search(clean):
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
            diag.hit(i, "v13_passive_reschedule_confirmed")

        frames.append(current)
        schedules.append(current_sched)

    return frames, schedules, diag, constructor_errors
