#!/usr/bin/env python3
"""VoiceProbe SemanticLab Level 2 architecture candidate V8.

V8 is a minimal residual correction of V7 behavior, implemented from the same V6 base built from the now-EXPOSED
132-case final-holdout diagnostic. That 132-case corpus is DEVELOPMENT evidence
from this point forward and is never treated as unseen again.

No specialist retraining is performed.

General architecture additions
------------------------------
1. Context-list discourse normalization:
   vague selection over two concrete options is resolved from whether the
   antecedent was an explicit offer or merely a non-offer enumeration.

2. Temporal ambiguity precedence:
   context-free "make/move/shift ... earlier/later" retains temporal ambiguity
   and cannot be erased by an erroneous dense clarification/availability act.

3. Transaction mutation normalization:
   declarative "I can book/move/cancel/keep..." => transaction statement +
   proposed signal; "appointment remains as scheduled" => confirmed KEEP;
   "will remain in place" => proposed KEEP.

4. Punctuation-free failure/fallback comparison:
   ASR turns such as "Wednesday morning is full can I try Friday morning"
   compare failed-vs-fallback values per axis exactly like punctuated turns.

5. Domain lexical precedence:
   complaint idioms including "what brought you in", and explicit
   first/last/full-name wording, can repair frozen fact-head boundary misses.

6. Reference precedence:
   rejection of an explicit offer remains prior_option; provider pronouns
   outrank a newly mentioned day; "anything else around that time" is a
   question referencing prior_time.

7. Context-option cleanup:
   discourse tails such as "are listed" / "are both available" are not part
   of the semantic option string.

8. Negative record-state normalization:
   explicit "no current appointment..." / "no profile..." controls record
   existence polarity after the record topic is already known.

9. Negative availability + explicit fallback has offer speech act even when
   the dense head emits question.

V8 residual corrections:
10. Provider-pronoun reference typing preserves an already-resolved provider
    selected_option instead of clearing it.
11. Context-option status cleanup strips terminal "are both"/"are all" even
    when an earlier parser has already removed "available/open/listed".

This file does not modify V6, production source, runtime wiring, v0.17,
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
V6_FILE = HERE / "voiceprobe_semanticlab_v2_post_holdout_architecture_v6.py"

if not V6_FILE.is_file():
    raise SystemExit(f"Missing V6 companion file: {V6_FILE}")


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v6 = load_mod("l2_v7_v6", V6_FILE)
v5 = v6.v5
v2 = v6.v2
base = v6.base

from voiceprobe.v33.semantic_frame import (
    AmbiguityKind,
    ConstraintAxis,
    RecordClaim,
    ReferenceKind,
    SemanticAmbiguity,
    SemanticFrame,
    SemanticTopic,
    SpeechAct,
    TransactionOperation,
    TransactionSignal,
)


VAGUE_CONTEXT_SELECTION_RE = re.compile(
    r"\b(?:"
    r"that\s+(?:one|choice)(?:\s+(?:sounds|looks|seems)\s+(?:good|better|fine))?|"
    r"go\s+with\s+that(?:\s+(?:one|choice))?|"
    r"let(?:'s|\s+us)\s+(?:use|take|do)\s+that(?:\s+(?:one|choice))?|"
    r"i(?:'d|\s+would)\s+(?:take|prefer)\s+that(?:\s+(?:one|choice))?"
    r")\b",
    re.IGNORECASE,
)

TEMPORAL_AMBIGUOUS_ACTION_RE = re.compile(
    r"\b(?:make|move|shift|shifted|change|do)\b.{0,45}"
    r"\b(?:earlier|later)\b|"
    r"\b(?:earlier|later)\b.{0,45}"
    r"\b(?:make|move|shift|shifted|change|do)\b",
    re.IGNORECASE,
)

I_CAN_MUTATION_RE = re.compile(
    r"^\s*(?:i|we)\s+can\s+"
    r"(?P<verb>book|reschedule|move|schedule|cancel|keep|leave)\b",
    re.IGNORECASE,
)

KEEP_CONFIRMED_STATE_RE = re.compile(
    r"\b(?:current\s+)?(?:appointment|visit)\b.{0,55}"
    r"\b(?:is\s+staying|stays|remains|is\s+remaining)\b.{0,35}"
    r"\b(?:as\s+scheduled|as\s+is|in\s+place|unchanged)\b",
    re.IGNORECASE,
)

KEEP_PROPOSED_STATE_RE = re.compile(
    r"\b(?:current\s+)?(?:appointment|visit)\b.{0,55}"
    r"\bwill\s+(?:remain|stay|be\s+left)\b.{0,35}"
    r"\b(?:as\s+scheduled|as\s+is|in\s+place|unchanged)\b",
    re.IGNORECASE,
)

COMPLAINT_IDIOM_V7_RE = re.compile(
    r"\bwhat\s+(?:brings|brought)\s+you\s+in\b|"
    r"\bwhat\s+are\s+we\s+(?:seeing|treating)\s+you\s+for\b|"
    r"\bwhat\s+are\s+you\s+(?:being\s+seen|coming\s+in)\s+for\b",
    re.IGNORECASE,
)

FIRST_NAME_RE = re.compile(r"\bfirst\s+name\b", re.IGNORECASE)
LAST_NAME_RE = re.compile(
    r"\b(?:last\s+name|family\s+name|surname)\b",
    re.IGNORECASE,
)
FULL_NAME_RE = re.compile(
    r"\b(?:full\s+name|complete\s+name)\b",
    re.IGNORECASE,
)

REJECTION_OTHER_RE = re.compile(
    r"^\s*(?:no|nope|not\s+that)\b.{0,80}"
    r"\b(?:what|anything)\b.{0,40}\b(?:other|else)\b|"
    r"^\s*(?:no|nope|not\s+that)\b.{0,80}\b(?:other|else)\b",
    re.IGNORECASE,
)

PROVIDER_PRONOUN_RE = re.compile(
    r"\b(?:he|she|him|her)\b",
    re.IGNORECASE,
)

ALT_QUESTION_RE = re.compile(
    r"^\s*(?:what|anything|any\s+other|something\s+else)\b|"
    r"\banything\s+else\b",
    re.IGNORECASE,
)

CONTEXT_STATUS_SUFFIX_RE = re.compile(
    r"\s+are(?:\s+(?:both|all))?"
    r"(?:\s+(?:available|open|listed))?\s*$|"
    r"\s+is\s+(?:available|open|listed)\s*$",
    re.IGNORECASE,
)

UNPUNCTUATED_FALLBACK_SPLIT_RE = re.compile(
    r"^(?P<before>.+?\b(?:full|booked|taken|unavailable|nothing|"
    r"no\s+openings?)\b)\s+"
    r"(?P<after>(?:can|could|should|would|may)\s+i\s+"
    r"(?:try|check|look|search)\b.+)$",
    re.IGNORECASE,
)

FALLBACK_REQUEST_RE = re.compile(
    r"\b(?:can|could|should|would|may)\s+i\s+"
    r"(?:try|check|look|search)\b",
    re.IGNORECASE,
)

APPOINTMENT_MISSING_RE = re.compile(
    r"\bno\s+(?:current\s+)?appointment\b|"
    r"\bthere\s+(?:is|isn't|is\s+not)\s+(?:no\s+)?"
    r"(?:current\s+)?appointment\b|"
    r"\b(?:can't|cannot|do\s+not|don't)\s+(?:find|see)\b.{0,30}"
    r"\bappointment\b",
    re.IGNORECASE,
)

PROFILE_MISSING_RE = re.compile(
    r"\bno\s+(?:patient\s+)?profile\b|"
    r"\bthere\s+(?:isn't|is\s+not)\b.{0,20}\bprofile\b|"
    r"\b(?:can't|cannot|do\s+not|don't)\s+(?:find|see)\b.{0,30}"
    r"\bprofile\b",
    re.IGNORECASE,
)


def _clean_context_candidate(value: str) -> str:
    out = str(value).strip(" \t\r\n,.;?!")
    out = CONTEXT_STATUS_SUFFIX_RE.sub("", out)
    out = out.strip(" \t\r\n,.;?!")
    return re.sub(r"\s+", " ", out)


def _two_context_options(turn) -> tuple[str, ...]:
    if not turn.context:
        return ()
    latest = str(turn.context[-1]).strip()
    parts = re.split(
        r"\s+(?:or|and)\s+",
        latest,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    if len(parts) != 2:
        return ()

    left = v5.clean_option_segment(parts[0], first=True, last=False)
    right = v5.clean_option_segment(parts[1], first=False, last=True)

    left = _clean_context_candidate(left)
    right = _clean_context_candidate(right)

    if (
        left
        and right
        and v5.ENTITY_RE.search(left)
        and v5.ENTITY_RE.search(right)
    ):
        return base.dedupe((left, right))
    return ()


def _rebuild_v7(
    frame,
    *,
    speech_act=None,
    topic=None,
    requested_fact=None,
    schedule=None,
    offered_options=None,
    selected_option=None,
    record_claims=None,
    transaction_operation=None,
    transaction_signal=None,
    reference=None,
    ambiguity_kind=None,
    ambiguity_candidates=None,
):
    sched = schedule or {
        "failed_constraints": tuple(x.value for x in frame.failed_constraints),
        "proposed_changes": tuple(x.value for x in frame.proposed_changes),
        "retained_constraints": tuple(x.value for x in frame.retained_constraints),
    }

    act = speech_act if speech_act is not None else frame.speech_act.value
    top = topic if topic is not None else frame.topic.value
    fact = requested_fact if requested_fact is not None else frame.requested_fact
    offered = (
        tuple(offered_options)
        if offered_options is not None
        else tuple(frame.offered_options)
    )
    selected = (
        selected_option if selected_option is not None else frame.selected_option
    )
    claims = (
        tuple(record_claims)
        if record_claims is not None
        else tuple(x.value for x in frame.record_claims)
    )
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
        offered_options=offered,
        selected_option=selected,
        record_claims=tuple(RecordClaim(x) for x in claims),
        transaction_operation=TransactionOperation(op),
        transaction_signal=TransactionSignal(sig),
        reference=ReferenceKind(ref),
        ambiguity=SemanticAmbiguity(
            kind=AmbiguityKind(ak),
            candidates=ac,
            detail="",
        ),
    )


def _mutation_op(verb: str) -> str:
    low = verb.casefold()
    if low == "book":
        return "book"
    if low in {"reschedule", "move", "schedule"}:
        return "reschedule"
    if low == "cancel":
        return "cancel"
    if low in {"keep", "leave"}:
        return "keep"
    return "none"


def _relative_schedule_v7(text: str, sched: dict):
    # First retain V6's punctuation-aware behavior.
    corrected, changed = v6._clause_relative_schedule(text, sched)
    if changed:
        return corrected, True

    # ASR often removes the punctuation between failure and fallback.
    m = UNPUNCTUATED_FALLBACK_SPLIT_RE.search(text)
    if not m:
        return sched, False

    before = v6._axis_values(m.group("before"))
    after = v6._axis_values(m.group("after"))

    comparable = set()
    same = set()
    changed_axes = set()

    for axis in ("day", "time_of_day", "provider"):
        if before[axis] and after[axis]:
            comparable.add(axis)
            if before[axis] == after[axis]:
                same.add(axis)
            else:
                changed_axes.add(axis)

    if not comparable:
        return sched, False

    proposed = set(sched["proposed_changes"]) - comparable
    retained = set(sched["retained_constraints"]) - comparable

    proposed |= changed_axes
    retained |= same
    proposed -= retained

    out = {
        "failed_constraints": tuple(sched["failed_constraints"]),
        "proposed_changes": base.order_axes(proposed),
        "retained_constraints": base.order_axes(retained),
    }
    return out, out != sched


def construct_v8_frames(runtime, result, checkpoints, v2_frames, v2_schedules):
    (
        v6_frames,
        v6_schedules,
        diag,
        constructor_errors,
    ) = v6.construct_v6_frames(
        runtime,
        result,
        checkpoints,
        v2_frames,
        v2_schedules,
    )

    frames = []
    schedules = []

    for i, (turn, v2_frame, frame, sched) in enumerate(
        zip(runtime, v2_frames, v6_frames, v6_schedules)
    ):
        clean = v5.strip_fillers(turn.utterance)
        current = frame
        current_sched = dict(sched)

        # ------------------------------------------------------------------
        # A. Domain lexical precedence.
        # ------------------------------------------------------------------
        if COMPLAINT_IDIOM_V7_RE.search(clean):
            current = _rebuild_v7(
                current,
                speech_act="question",
                topic="patient_fact",
                requested_fact="complaint",
                transaction_operation="none",
                transaction_signal="none",
            )
            diag.hit(i, "v7_complaint_idiom_inflection")

        elif current.topic.value == "patient_fact" and current.speech_act.value in {
            "question", "request"
        }:
            fact = None
            if FULL_NAME_RE.search(clean):
                fact = "full_name"
            elif FIRST_NAME_RE.search(clean):
                fact = "first_name"
            elif LAST_NAME_RE.search(clean):
                fact = "last_name"
            if fact and current.requested_fact != fact:
                current = _rebuild_v7(
                    current,
                    requested_fact=fact,
                )
                diag.hit(i, "v7_explicit_name_fact_precedence")

        # ------------------------------------------------------------------
        # B. Temporal ambiguity outranks bad act/topic or clarification veto.
        # ------------------------------------------------------------------
        temporal_source = (
            current.ambiguity.kind.value == "temporal_reference"
            or v2_frame.ambiguity.kind.value == "temporal_reference"
        )
        if (
            not turn.context
            and temporal_source
            and TEMPORAL_AMBIGUOUS_ACTION_RE.search(clean)
        ):
            current_sched = {
                "failed_constraints": (),
                "proposed_changes": (),
                "retained_constraints": (),
            }
            current = _rebuild_v7(
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
            diag.hit(i, "v7_temporal_ambiguity_precedence")

        # ------------------------------------------------------------------
        # C. Transaction mutation normalization.
        # ------------------------------------------------------------------
        m_mut = I_CAN_MUTATION_RE.search(clean)
        if m_mut:
            op = _mutation_op(m_mut.group("verb"))
            if op != "none":
                current_sched = {
                    "failed_constraints": (),
                    "proposed_changes": (),
                    "retained_constraints": (),
                }
                current = _rebuild_v7(
                    current,
                    speech_act="statement",
                    topic="transaction",
                    schedule=current_sched,
                    transaction_operation=op,
                    transaction_signal="proposed",
                    ambiguity_kind="none",
                    ambiguity_candidates=(),
                )
                diag.hit(i, "v7_declarative_mutation_proposal")

        if KEEP_PROPOSED_STATE_RE.search(clean):
            current_sched = {
                "failed_constraints": (),
                "proposed_changes": (),
                "retained_constraints": (),
            }
            current = _rebuild_v7(
                current,
                speech_act="statement",
                topic="transaction",
                schedule=current_sched,
                transaction_operation="keep",
                transaction_signal="proposed",
                ambiguity_kind="none",
                ambiguity_candidates=(),
            )
            diag.hit(i, "v7_keep_state_proposed")

        elif KEEP_CONFIRMED_STATE_RE.search(clean):
            current_sched = {
                "failed_constraints": (),
                "proposed_changes": (),
                "retained_constraints": (),
            }
            current = _rebuild_v7(
                current,
                speech_act="confirmation",
                topic="transaction",
                schedule=current_sched,
                transaction_operation="keep",
                transaction_signal="confirmed",
                ambiguity_kind="none",
                ambiguity_candidates=(),
            )
            diag.hit(i, "v7_keep_state_confirmed")

        # ------------------------------------------------------------------
        # D. Context option cleanup + vague selection boundary.
        # ------------------------------------------------------------------
        options = _two_context_options(turn)

        if current.selected_option:
            cleaned_selected = _clean_context_candidate(current.selected_option)
            if cleaned_selected != current.selected_option:
                current = _rebuild_v7(
                    current,
                    selected_option=cleaned_selected,
                )
                diag.hit(i, "v8_selected_option_suffix_cleanup")

        if current.ambiguity.candidates:
            cleaned_candidates = tuple(
                _clean_context_candidate(x)
                for x in current.ambiguity.candidates
            )
            if cleaned_candidates != tuple(current.ambiguity.candidates):
                current = _rebuild_v7(
                    current,
                    ambiguity_candidates=cleaned_candidates,
                )
                diag.hit(i, "v8_ambiguity_candidate_suffix_cleanup")

        if (
            len(options) >= 2
            and VAGUE_CONTEXT_SELECTION_RE.search(clean)
        ):
            explicit_offer = bool(
                turn.context and v2.latest_context_is_explicit_offer(turn)
            )
            if explicit_offer:
                current = _rebuild_v7(
                    current,
                    speech_act="confirmation",
                    topic="availability",
                    selected_option="",
                    reference="ambiguous",
                    ambiguity_kind="option_reference",
                    ambiguity_candidates=options,
                )
                diag.hit(i, "v7_vague_explicit_offer_ambiguity")
            else:
                current = _rebuild_v7(
                    current,
                    speech_act="statement",
                    topic="other",
                    selected_option="",
                    reference="none",
                    ambiguity_kind="option_reference",
                    ambiguity_candidates=options,
                )
                diag.hit(i, "v7_vague_nonoffer_ambiguity")

        # ------------------------------------------------------------------
        # E. Reference precedence.
        # ------------------------------------------------------------------
        if (
            turn.context
            and v2.latest_context_is_explicit_offer(turn)
            and REJECTION_OTHER_RE.search(clean)
        ):
            current = _rebuild_v7(
                current,
                speech_act="question",
                topic="availability",
                selected_option="",
                reference="prior_option",
            )
            diag.hit(i, "v7_rejection_offer_reference")

        if (
            turn.context
            and PROVIDER_PRONOUN_RE.search(clean)
            and re.search(
                r"\bdr\.?\s+[A-Za-z][A-Za-z'-]*",
                str(turn.context[-1]),
                re.IGNORECASE,
            )
        ):
            # Reference typing must not destroy a provider selection already
            # resolved by an earlier layer (e.g. "Okay, go with her.").
            current = _rebuild_v7(
                current,
                reference="prior_provider",
            )
            diag.hit(i, "v8_provider_pronoun_reference_preserve_selection")

        if (
            current.reference.value == "prior_time"
            and ALT_QUESTION_RE.search(clean)
            and current.speech_act.value != "question"
        ):
            current = _rebuild_v7(
                current,
                speech_act="question",
                topic="availability",
                selected_option="",
            )
            diag.hit(i, "v7_reference_alternative_question")

        # ------------------------------------------------------------------
        # F. Scheduling: offer act + punctuation-free relative comparison.
        # ------------------------------------------------------------------
        if (
            current.topic.value == "availability"
            and v2.NEGATIVE_AVAILABILITY_RE.search(clean)
            and FALLBACK_REQUEST_RE.search(clean)
            and current.proposed_changes
            and current.speech_act.value != "offer"
        ):
            current = _rebuild_v7(
                current,
                speech_act="offer",
            )
            diag.hit(i, "v7_negative_fallback_is_offer")

        if (
            current.topic.value == "availability"
            and v2.NEGATIVE_AVAILABILITY_RE.search(clean)
            and FALLBACK_REQUEST_RE.search(clean)
        ):
            corrected_sched, changed = _relative_schedule_v7(
                clean,
                current_sched,
            )
            if changed:
                current_sched = corrected_sched
                current = _rebuild_v7(
                    current,
                    schedule=current_sched,
                )
                diag.hit(i, "v7_relative_schedule_asr_boundary")

        # ------------------------------------------------------------------
        # G. Explicit negative record polarity.
        # ------------------------------------------------------------------
        if (
            current.topic.value == "appointment_state"
            and APPOINTMENT_MISSING_RE.search(clean)
            and tuple(x.value for x in current.record_claims)
            != ("appointment_missing",)
        ):
            current = _rebuild_v7(
                current,
                record_claims=("appointment_missing",),
            )
            diag.hit(i, "v7_appointment_missing_negation")

        if (
            current.topic.value == "profile"
            and PROFILE_MISSING_RE.search(clean)
            and tuple(x.value for x in current.record_claims)
            != ("profile_missing",)
        ):
            current = _rebuild_v7(
                current,
                record_claims=("profile_missing",),
            )
            diag.hit(i, "v7_profile_missing_negation")

        frames.append(current)
        schedules.append(current_sched)

    return frames, schedules, diag, constructor_errors
