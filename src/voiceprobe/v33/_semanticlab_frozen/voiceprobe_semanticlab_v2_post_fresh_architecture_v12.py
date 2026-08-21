#!/usr/bin/env python3
"""VoiceProbe SemanticLab Level 2 architecture candidate V12.

V12 is a narrow post-normalization wrapper over V11 based only on the
EXPOSED/DEVELOPMENT V11 fresh-128 residual audit.

No specialist retraining.

General additions
-----------------
1. Bare TRY proposal authority:
   "Could/Should/Can I try Thursday/a morning slot/a different time" is an
   availability proposal, not retention or temporal ambiguity.

2. Availability CHECK act normalization:
   broad "Can I check something earlier/later..." forms are offers even when
   lower layers already chose availability rather than transaction search.
   Frozen explicit transaction-search forms ("... for you", "checking whether")
   remain untouched.

3. Availability inventory-question cleanup:
   "What other openings/times/slots are open/available?" and provider-pronoun
   variants are questions, not scheduling proposals, and cannot retain stale
   requested_fact predictions.

4. Explicit offered-option interrogatives:
   first/second/earlier/later option questions over an explicit two-option
   offer become question + prior_option and never commit a selection.

5. Explicit offered-option confirmations:
   ordinal confirmations ("I'll take the first option") resolve the selected
   option and clear false ambiguity; weekday earlier/later choice/option forms
   override a wrong inherited selection.

6. Vague "that option works" context boundary:
   two-option explicit offers => ambiguous option reference;
   two-option non-offer enumerations => statement/other option ambiguity.

7. Negative TAKEN/full spoken-time completion:
   "that time is taken" and spoken clock forms such as "three pm is full"
   contribute time_of_day failure evidence.

8. Spoken-clock KEEP retention:
   "keep three pm and try Monday" retains time_of_day.

9. Clause-local declarative transaction mutation:
   "...; I can move/book/reschedule/cancel ..." => statement transaction
   proposed, scheduling cleared.

10. Transaction permission requested-fact cleanup:
    transaction permission frames cannot carry stale reschedule_reason or
    other requested_fact values.

This file does not modify V11, V10, V8, production source, runtime wiring,
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
V11_FILE = HERE / "voiceprobe_semanticlab_v2_post_fresh_architecture_v11.py"

if not V11_FILE.is_file():
    raise SystemExit(f"Missing V11 companion file: {V11_FILE}")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v11 = load_mod("l2_v12_v11", V11_FILE)
v10 = v11.v10
v8 = v11.v8
v6 = v11.v6
v5 = v11.v5
v2 = v11.v2
base = v11.base


BARE_TRY_RE = re.compile(
    r"^\s*(?:can|could|should|would|may)\s+i\s+try\b",
    re.IGNORECASE,
)

GENERIC_OTHER_AVAIL_RE = re.compile(
    r"^\s*what\s+other\s+"
    r"(?:openings?|times?|slots?|availability)\b"
    r".{0,50}\b(?:open|available|there|have)\b",
    re.IGNORECASE,
)

EXPLICIT_OPTION_INTERROGATIVE_RE = re.compile(
    r"^\s*(?:can|could|would|will|is|are)\b"
    r".{0,55}\b(?:first|second|earlier|later)\s+"
    r"(?:one|option|choice|slot)\b",
    re.IGNORECASE,
)

ORDINAL_CONFIRM_RE = re.compile(
    r"\b(?:take|use|choose|pick|select|go\s+with)\s+"
    r"(?:the\s+)?(?P<ordinal>first|second)\s+"
    r"(?:one|option|choice|slot)\b",
    re.IGNORECASE,
)

TEMPORAL_CHOICE_CONFIRM_RE = re.compile(
    r"^\s*(?:the\s+)?(?P<which>earlier|later)\s+"
    r"(?:one|option|choice)\b",
    re.IGNORECASE,
)

VAGUE_THAT_OPTION_RE = re.compile(
    r"\bthat\s+option(?:\s+(?:works|sounds|looks|seems)\s*"
    r"(?:good|better|fine)?)?\b",
    re.IGNORECASE,
)

SPOKEN_CLOCK_RE = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"\s*(?:a\.?\s*m\.?|p\.?\s*m\.?)\b",
    re.IGNORECASE,
)

TIME_NEGATIVE_RE = re.compile(
    r"\b(?:that\s+time|(?:at\s+)?(?:\d{1,2}(?::\d{2})?\s*"
    r"(?:a\.?m\.?|p\.?m\.?)|"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"\s*(?:a\.?\s*m\.?|p\.?\s*m\.?)))\b"
    r".{0,30}\b(?:full|booked|taken|unavailable|not\s+available|not\s+open)\b",
    re.IGNORECASE,
)

CLAUSE_DECLARATIVE_MUTATION_RE = re.compile(
    r"\bi\s+can\s+(?P<verb>book|schedule|reschedule|move|cancel)\b",
    re.IGNORECASE,
)

EXPLICIT_TRANSACTION_SEARCH_RE = re.compile(
    r"\bfor\s+you\b|\bchecking\s+whether\b|\bcheck\s+whether\b",
    re.IGNORECASE,
)


def _clear_schedule():
    return {
        "failed_constraints": (),
        "proposed_changes": (),
        "retained_constraints": (),
    }


def _axes_with_spoken_time(text: str):
    axes = set(v11._axes_from_values(text))
    if SPOKEN_CLOCK_RE.search(text):
        axes.add("time_of_day")
    return axes


def _failed_axes_with_spoken_time(text: str):
    axes = set(v2.failed_axes_from_negative_clauses(text))
    if TIME_NEGATIVE_RE.search(text):
        axes.add("time_of_day")
    return axes


def _select_ordinal(options, ordinal: str):
    if len(options) < 2:
        return ""
    if ordinal.casefold() == "first":
        return str(options[0])
    if ordinal.casefold() == "second":
        return str(options[1])
    return ""


def construct_v12_frames(runtime, result, checkpoints, v2_frames, v2_schedules):
    (
        frames11,
        schedules11,
        diag,
        constructor_errors,
    ) = v11.construct_v11_frames(
        runtime,
        result,
        checkpoints,
        v2_frames,
        v2_schedules,
    )

    frames = []
    schedules = []

    for i, (turn, frame, sched11) in enumerate(
        zip(runtime, frames11, schedules11)
    ):
        clean = v5.strip_fillers(turn.utterance)
        current = frame
        current_sched = dict(sched11)

        lexical_axes = set(v2.proposed_axes_from_text(clean))
        options = v8._two_context_options(turn)
        explicit_offer = bool(
            turn.context and v2.latest_context_is_explicit_offer(turn)
        )

        # --------------------------------------------------------------
        # 1. Bare TRY proposal authority.
        # --------------------------------------------------------------
        if (
            BARE_TRY_RE.search(clean)
            and lexical_axes
            and not v10.KEEP_TRY_RE.search(clean)
        ):
            current_sched = {
                "failed_constraints": tuple(
                    x.value for x in current.failed_constraints
                ),
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
            diag.hit(i, "v12_bare_try_proposal_authority")

        # --------------------------------------------------------------
        # 2. Broad availability CHECK act normalization.
        # --------------------------------------------------------------
        if (
            v11.BROAD_AVAIL_CHECK_RE.search(clean)
            and lexical_axes
            and not EXPLICIT_TRANSACTION_SEARCH_RE.search(clean)
        ):
            current_sched = {
                "failed_constraints": tuple(
                    x.value for x in current.failed_constraints
                ),
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
            diag.hit(i, "v12_broad_check_offer_normalization")

        # --------------------------------------------------------------
        # 3. Generic "what other openings/times..." inventory questions.
        #    Provider-pronoun-specific V11 typing remains authoritative.
        # --------------------------------------------------------------
        if (
            GENERIC_OTHER_AVAIL_RE.search(clean)
            or v11.PROVIDER_OTHER_OPENINGS_RE.search(clean)
        ):
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
            diag.hit(i, "v12_inventory_question_cleanup")

        # --------------------------------------------------------------
        # 4. Explicit offered-option interrogatives.
        # --------------------------------------------------------------
        if (
            len(options) >= 2
            and explicit_offer
            and EXPLICIT_OPTION_INTERROGATIVE_RE.search(clean)
        ):
            current_sched = _clear_schedule()
            current = v8._rebuild_v7(
                current,
                speech_act="question",
                topic="availability",
                requested_fact="",
                schedule=current_sched,
                transaction_operation="none",
                transaction_signal="none",
                reference="prior_option",
                selected_option="",
                ambiguity_kind="none",
                ambiguity_candidates=(),
            )
            diag.hit(i, "v12_explicit_option_interrogative")

        # --------------------------------------------------------------
        # 5a. Explicit ordinal confirmation selection.
        # --------------------------------------------------------------
        m_ord = ORDINAL_CONFIRM_RE.search(clean)
        if (
            len(options) >= 2
            and explicit_offer
            and m_ord
            and not EXPLICIT_OPTION_INTERROGATIVE_RE.search(clean)
        ):
            selected = _select_ordinal(options, m_ord.group("ordinal"))
            if selected:
                current = v8._rebuild_v7(
                    current,
                    speech_act="confirmation",
                    topic="availability",
                    requested_fact="",
                    selected_option=selected,
                    reference="prior_option",
                    ambiguity_kind="none",
                    ambiguity_candidates=(),
                )
                diag.hit(i, "v12_explicit_ordinal_confirmation")

        # --------------------------------------------------------------
        # 5b. Weekday earlier/later choice can override a wrong inherited
        #     selection, but only for non-interrogative confirmation forms.
        # --------------------------------------------------------------
        m_choice = TEMPORAL_CHOICE_CONFIRM_RE.search(clean)
        if (
            len(options) == 2
            and explicit_offer
            and m_choice
            and not EXPLICIT_OPTION_INTERROGATIVE_RE.search(clean)
        ):
            selected = v10._weekday_selection(
                options,
                earlier=(m_choice.group("which").casefold() == "earlier"),
            )
            if selected:
                current = v8._rebuild_v7(
                    current,
                    speech_act="confirmation",
                    topic="availability",
                    selected_option=selected,
                    reference="prior_option",
                    ambiguity_kind="none",
                    ambiguity_candidates=(),
                )
                diag.hit(i, "v12_weekday_choice_override")

        # --------------------------------------------------------------
        # 6. Vague "that option works" explicit-offer/non-offer boundary.
        # --------------------------------------------------------------
        if len(options) >= 2 and VAGUE_THAT_OPTION_RE.search(clean):
            if explicit_offer:
                current = v8._rebuild_v7(
                    current,
                    speech_act="confirmation",
                    topic="availability",
                    requested_fact="",
                    selected_option="",
                    reference="ambiguous",
                    ambiguity_kind="option_reference",
                    ambiguity_candidates=options,
                )
                diag.hit(i, "v12_vague_option_explicit_offer_ambiguity")
            else:
                current = v8._rebuild_v7(
                    current,
                    speech_act="statement",
                    topic="other",
                    requested_fact="",
                    selected_option="",
                    reference="none",
                    ambiguity_kind="option_reference",
                    ambiguity_candidates=options,
                )
                diag.hit(i, "v12_vague_option_nonoffer_ambiguity")

        # --------------------------------------------------------------
        # 7/8. KEEP...TRY with spoken clock values and negative time
        #      completion.
        # --------------------------------------------------------------
        keep_try = v10.KEEP_TRY_RE.search(clean)
        if keep_try:
            keep_axes = set(v11._keep_axes(keep_try.group("keep")))
            keep_axes |= _axes_with_spoken_time(keep_try.group("keep"))

            try_axes = set(v11._try_axes(keep_try.group("try")))
            try_axes |= _axes_with_spoken_time(keep_try.group("try"))
            try_axes -= keep_axes

            if keep_axes and try_axes:
                prefix = clean[:keep_try.start()].strip(" ;,.")
                failed = _failed_axes_with_spoken_time(prefix)

                current_sched = {
                    "failed_constraints": base.order_axes(failed),
                    "proposed_changes": base.order_axes(try_axes),
                    "retained_constraints": base.order_axes(keep_axes),
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
                diag.hit(i, "v12_spoken_clock_keep_try_authority")

        # --------------------------------------------------------------
        # 7b. Negative time evidence before a normal fallback action.
        # --------------------------------------------------------------
        m_fallback = v11.FALLBACK_ACTION_RE.search(clean)
        if (
            current.topic.value == "availability"
            and m_fallback
            and not keep_try
        ):
            prefix = clean[:m_fallback.start()].strip(" ;,.")
            failed = _failed_axes_with_spoken_time(prefix)
            if failed:
                current_sched = {
                    "failed_constraints": base.order_axes(failed),
                    "proposed_changes": tuple(
                        x.value for x in current.proposed_changes
                    ),
                    "retained_constraints": tuple(
                        x.value for x in current.retained_constraints
                    ),
                }
                current = v8._rebuild_v7(
                    current,
                    schedule=current_sched,
                )
                diag.hit(i, "v12_negative_time_prefix_completion")

        # --------------------------------------------------------------
        # 9. Clause-local declarative transaction mutation.
        # --------------------------------------------------------------
        m_decl = CLAUSE_DECLARATIVE_MUTATION_RE.search(clean)
        if m_decl and not keep_try:
            op = v10._mutation_op(m_decl.group("verb"))
            current_sched = _clear_schedule()
            current = v8._rebuild_v7(
                current,
                speech_act="statement",
                topic="transaction",
                requested_fact="",
                schedule=current_sched,
                transaction_operation=op,
                transaction_signal="proposed",
                reference="none",
                selected_option="",
                ambiguity_kind="none",
                ambiguity_candidates=(),
            )
            diag.hit(i, "v12_clause_declarative_mutation")

        # --------------------------------------------------------------
        # 10. Transaction permission cannot carry stale requested facts.
        # --------------------------------------------------------------
        if (
            current.topic.value == "transaction"
            and current.transaction_signal.value == "permission_request"
            and current.requested_fact
        ):
            current = v8._rebuild_v7(
                current,
                requested_fact="",
            )
            diag.hit(i, "v12_transaction_permission_fact_cleanup")

        frames.append(current)
        schedules.append(current_sched)

    return frames, schedules, diag, constructor_errors
