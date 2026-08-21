#!/usr/bin/env python3
"""Read-only candidate Level 2 semantic normalization architecture audit.

NO TRAINING. NO RUNTIME WIRING. NO V0.17 MODIFICATION. NO TELEPHONY.

Purpose
-------
Take the frozen full Level 2 assembler and test a conservative second-pass
component-handoff normalization architecture over the 133-case DEVELOPMENT
corpus before retraining any specialist.

Prediction-time inputs are restricted to:
- utterance
- recent context
- frozen component predictions / clause predictions

The following are NEVER used to choose a repair:
- case_id
- category
- tags
- expected/gold frame

Gold is consulted only after every candidate frame has been constructed.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

BASE_FILE = Path(__file__).with_name(
    "voiceprobe_semanticlab_v2_full_semanticframe_eval.py"
)
if not BASE_FILE.is_file():
    raise SystemExit(
        "Missing companion full evaluator beside this script:\n"
        f"  {BASE_FILE}\n"
        "Keep both downloaded scripts in the same Downloads folder."
    )


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_mod("l2_candidate_base", BASE_FILE)

from voiceprobe.v33.semantic_corpus import load_semanticlab_cases
from voiceprobe.v33.semantic_frame import (
    AmbiguityKind,
    ConstraintAxis,
    ReferenceKind,
    SemanticAmbiguity,
    SemanticFrame,
    SemanticTopic,
    SpeechAct,
)
from voiceprobe.v33.semantic_frame_eval import evaluate_frame


AXIS_ORDER = ("day", "time_of_day", "provider")

WEEKDAY = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
CLOCK = r"(?:\b(?:1[0-2]|0?[1-9])(?::[0-5]\d)?\s*(?:a\.?m\.?|p\.?m\.?)\b)"
DAYPART = r"(?:\bmornings?\b|\bafternoons?\b|\bevenings?\b|\bnoon\b)"

# These recognizers are deliberately narrower than V1. They identify semantic
# evidence only inside an already-recognized proposal clause; they are not used
# to overwrite the frozen scheduler globally.
DAY_EVIDENCE_RE = re.compile(
    rf"(?:\b(?:days?|dates?|weekdays?|weeks?|months?|today|tomorrow|tonight)\b|"
    rf"\b{WEEKDAY}\b)",
    re.IGNORECASE,
)
TIME_EVIDENCE_RE = re.compile(
    rf"(?:\btimes?\b|{DAYPART}|{CLOCK}|"
    r"\b(?:earlier|later)\s+(?:appointments?|openings?|slots?|times?)\b|"
    r"\b(?:earlier|later)\s+in\s+the\s+day\b)",
    re.IGNORECASE,
)
PROVIDER_EVIDENCE_RE = re.compile(
    r"\b(?:providers?|doctors?|clinicians?|physicians?)\b|"
    r"\bdr\.?\s+[A-Z][A-Za-z'-]+|\bsomeone\s+else\b",
    re.IGNORECASE,
)

PROPOSAL_CUE_RE = re.compile(
    r"(?:"
    r"\b(?:another|different|other|earlier|later|instead)\b|"
    r"\b(?:try|check|look|search|broaden)\b|"
    r"\b(?:what|how)\s+about\b|"
    r"\bwant\s+me\s+to\b|"
    r"\bwould\b.{0,100}\bwork\b|"
    r"\bcould\b.{0,100}\b(?:do|come|work)\b|"
    r"\bshould\s+i\b|\bcan\s+i\b|\bi\s+can\b|"
    r"\bmaybe\b"
    r")",
    re.IGNORECASE,
)

NEGATIVE_AVAILABILITY_RE = re.compile(
    r"(?:"
    r"\bfull\b|\bfully\s+booked\b|\bbooked\s+out\b|"
    r"\bis\s+booked\b|\bare\s+booked\b|\bwas\s+booked\b|"
    r"\bunavailable\b|\bnot\s+available\b|"
    r"\bnothing\s+(?:is\s+)?(?:open|available)\b|"
    r"\bhas\s+nothing\b|\bhave\s+nothing\b|"
    r"\bno\s+(?:openings?|availability)\b|"
    r"\bdon'?t\s+have\b|\bdo\s+not\s+have\b|\bgone\b"
    r")",
    re.IGNORECASE,
)

OFFER_ACTION_RE = re.compile(
    r"(?:"
    r"\bwould\b.{0,100}\bwork\b|\bwant\s+me\s+to\b|"
    r"\bshould\s+i\b|\bcan\s+i\s+(?:check|try|look|search)\b|"
    r"\bi\s+can\s+(?:check|try|look|search)\b|"
    r"\bif\s+you(?:'d|\s+would)\s+like\b"
    r")",
    re.IGNORECASE,
)

EXPLICIT_OPTION_OFFER_RE = re.compile(
    r"(?:"
    r"\b(?:i|we)\s+(?:can\s+)?offer\b|"
    r"\b(?:i|we)\s+can\s+do\b|"
    r"\b(?:i|we)\s+have\b.{0,100}\b(?:available|open|opening)\b|"
    r"\b(?:would|could)\b.{0,100}\bwork\b"
    r")",
    re.IGNORECASE,
)

VAGUE_SELECTION_RE = re.compile(
    r"^\s*(?:"
    r"that(?:\s+one)?\s+(?:works?|sounds?\s+better)|"
    r"that\s+works?|"
    r"let'?s\s+do\s+that|"
    r"i(?:'d|\s+would)\s+(?:take|prefer)\s+that(?:\s+one)?|"
    r"go\s+with\s+that(?:\s+one)?"
    r")\s*[.!]?\s*$",
    re.IGNORECASE,
)

REJECT_ALTERNATIVE_RE = re.compile(
    r"^\s*(?:no|nope|not\s+that)\b.{0,80}\b(?:"
    r"what\s+else|anything\s+else|another\s+option|other\s+options?"
    r")\b",
    re.IGNORECASE,
)

PATIENT_FACT_REQUEST_RE = re.compile(
    r"\b(?:can|could|may)\s+i\s+(?:get|have|take|confirm|ask\s+for)\b",
    re.IGNORECASE,
)

# Direct explicit-retention language. Cross-clause anchoring is intentionally
# excluded; V1's whole-turn anchor inference was the source of the
# compound_day_time_01 regression.
RETAIN_DAY_DIRECT_RE = re.compile(
    rf"(?:"
    rf"\bsame\s+(?:day|date)\b|\bthat\s+same\s+day\b|"
    rf"\bkeep\s+(?:the\s+)?(?:day|date|{WEEKDAY})\b|"
    rf"\bleave\s+(?:the\s+)?(?:day|date)\s+alone\b"
    rf")",
    re.IGNORECASE,
)
RETAIN_TIME_DIRECT_RE = re.compile(
    r"(?:"
    r"\bsame\s+time\b|\bat\s+the\s+same\s+time\b|"
    r"\bkeep\s+(?:the\s+)?(?:time|morning|afternoon|evening)\b|"
    r"\bleave\s+(?:the\s+)?time\s+alone\b"
    r")",
    re.IGNORECASE,
)
RETAIN_PROVIDER_DIRECT_RE = re.compile(
    r"\b(?:same\s+provider|keep\s+(?:the\s+)?(?:provider|doctor)|"
    r"leave\s+(?:the\s+)?provider\s+alone)\b",
    re.IGNORECASE,
)

# Same-clause concrete anchors: "another time on Friday" retains the day;
# "another day at 3 PM" retains the time.
RETAIN_DAY_ANCHOR_RE = re.compile(
    rf"\b(?:another|different|other)\s+(?:time|appointment|opening|slot)\b"
    rf"[^,;.!?]{{0,50}}\b(?:on\s+)?{WEEKDAY}\b",
    re.IGNORECASE,
)
RETAIN_TIME_ANCHOR_RE = re.compile(
    rf"\b(?:another|different|other)\s+(?:day|date)\b"
    rf"[^,;.!?]{{0,50}}(?:{CLOCK}|{DAYPART})",
    re.IGNORECASE,
)

HORIZON_ONLY_RE = re.compile(
    r"\bsometime\b.{0,20}\b(?:next|this|following)\s+(?:week|month)\b",
    re.IGNORECASE,
)


@dataclass
class CandidateDiagnostics:
    rules: dict[int, list[str]]

    @classmethod
    def create(cls):
        return cls(rules=defaultdict(list))

    def hit(self, i: int, rule: str) -> None:
        if rule not in self.rules[i]:
            self.rules[i].append(rule)


def split_semantic_clauses(text: str) -> list[str]:
    chunks = [
        x.strip()
        for x in re.split(
            r"(?<!Dr\.)(?<!Mr\.)(?<!Ms\.)(?<!Mrs\.)(?<=[.!?])\s+|"
            r"\s*;\s*|\s*,\s*(?:and|but|so)\s+",
            text,
            flags=re.IGNORECASE,
        )
        if x.strip()
    ]
    return chunks or [text.strip()]


def axis_evidence(text: str) -> set[str]:
    out = set()
    if DAY_EVIDENCE_RE.search(text):
        out.add("day")
    if TIME_EVIDENCE_RE.search(text):
        out.add("time_of_day")
    if PROVIDER_EVIDENCE_RE.search(text):
        out.add("provider")
    return out


def proposal_axes_for_clause(clause: str) -> set[str]:
    """Return only axes explicitly proposed/varied in one clause.

    Mere availability mentions are not enough. The clause needs proposal/action
    language, and several lexical contrasts are handled before generic evidence
    so phrases such as "later in the day" stay on TIME_OF_DAY.
    """

    low = clause.casefold()
    if not PROPOSAL_CUE_RE.search(clause):
        return set()

    out = set()

    # Strong temporal relations.
    if re.search(r"\b(?:later|earlier)\s+in\s+the\s+day\b", low):
        out.add("time_of_day")

    if re.search(
        r"\b(?:another|different|other)\s+(?:times?|appointments?|openings?|slots?)\b",
        low,
    ):
        out.add("time_of_day")

    if re.search(
        r"\b(?:another|different|other|later|earlier)\s+(?:days?|dates?)\b",
        low,
    ):
        out.add("day")

    if re.search(r"\b(?:other|different|another)\s+dates?\s+and\s+times?\b", low):
        out.update(("day", "time_of_day"))

    if re.search(
        r"\b(?:mornings?|afternoons?|evenings?)\b",
        low,
    ):
        out.add("time_of_day")

    if re.search(
        r"\b(?:later|earlier)\s+(?:appointments?|openings?|slots?|times?)\b",
        low,
    ):
        out.add("time_of_day")

    if re.search(
        r"\b(?:next|this|following)\s+(?:week|month)\b",
        low,
    ):
        out.add("day")

    if re.search(r"\bsomeone\s+else\b", low):
        out.add("provider")

    if re.search(
        r"\b(?:another|different|other)\s+(?:providers?|doctors?|clinicians?|physicians?)\b",
        low,
    ):
        out.add("provider")

    # Coordinated alternatives sometimes carry the modifier only once:
    # "another provider or time", "another day or time", etc.
    if re.search(
        r"\b(?:providers?|doctors?|clinicians?|physicians?)\s+"
        r"(?:or|and)\s+(?:times?|appointments?|openings?|slots?)\b",
        low,
    ):
        out.update(("provider", "time_of_day"))
    if re.search(
        r"\b(?:days?|dates?)\s+(?:or|and)\s+"
        r"(?:times?|appointments?|openings?|slots?)\b",
        low,
    ):
        out.update(("day", "time_of_day"))
    if re.search(
        r"\b(?:times?|appointments?|openings?|slots?)\s+(?:or|and)\s+"
        r"(?:days?|dates?)\b",
        low,
    ):
        out.update(("day", "time_of_day"))

    # Explicit named candidate after a search/offer verb.
    if re.search(r"\b(?:check|try|look|search)\b", low):
        if re.search(rf"\b{WEEKDAY}\b", low):
            out.add("day")
        if re.search(rf"(?:{DAYPART}|{CLOCK})", clause, re.IGNORECASE):
            out.add("time_of_day")
        if re.search(r"\bdr\.?\s+[A-Z][A-Za-z'-]+", clause):
            out.add("provider")

    # Direct "would X work?" / "could you come X?" offers may propose concrete
    # scheduling axes even without words such as another/different.
    if re.search(r"\b(?:would|could)\b.{0,100}\b(?:work|come)\b", low):
        if re.search(rf"\b{WEEKDAY}\b", low):
            out.add("day")
        if re.search(rf"(?:{DAYPART}|{CLOCK})", clause, re.IGNORECASE):
            out.add("time_of_day")
        if re.search(r"\b(?:provider|doctor|clinician|physician)\b", low):
            out.add("provider")

    return out


def proposed_axes_from_text(text: str) -> set[str]:
    out = set()
    for clause in split_semantic_clauses(text):
        out |= proposal_axes_for_clause(clause)
    return out


def retained_axes_from_text(text: str, proposed: set[str]) -> set[str]:
    out = set()

    for clause in split_semantic_clauses(text):
        if RETAIN_DAY_DIRECT_RE.search(clause) or RETAIN_DAY_ANCHOR_RE.search(clause):
            out.add("day")
        if RETAIN_TIME_DIRECT_RE.search(clause) or RETAIN_TIME_ANCHOR_RE.search(clause):
            out.add("time_of_day")
        if RETAIN_PROVIDER_DIRECT_RE.search(clause):
            out.add("provider")

    return out & set(AXIS_ORDER)


def failed_axes_from_negative_clauses(text: str) -> set[str]:
    out = set()
    for clause in split_semantic_clauses(text):
        if NEGATIVE_AVAILABILITY_RE.search(clause):
            out |= axis_evidence(clause)
    return out


def normalize_scheduling(
    i: int,
    turn,
    gate: dict[str, int],
    base_sched: dict[str, tuple[str, ...]],
    pair: tuple[str, str],
    diag: CandidateDiagnostics,
) -> dict[str, tuple[str, ...]]:
    # Preserve the frozen Phase 7C scope boundary exactly.
    if gate.get("reference", 0) or gate.get("ambiguity", 0) or gate.get("oos", 0):
        return {
            "failed_constraints": (),
            "proposed_changes": (),
            "retained_constraints": (),
        }

    text = turn.utterance
    clauses = split_semantic_clauses(text)
    is_compound = len(clauses) > 1

    # Do not inject scheduling fields into transaction semantics. For all other
    # non-availability topics, only a genuine multi-clause turn is eligible for
    # scheduling side-channel normalization (e.g. reason question + offer).
    allow_derived = (
        pair[1] == "availability"
        or (is_compound and pair[1] != "transaction")
    )

    failed = set(base_sched["failed_constraints"])
    proposed = set(base_sched["proposed_changes"])
    retained = set(base_sched["retained_constraints"])

    if not allow_derived:
        return {
            "failed_constraints": base.order_axes(failed),
            "proposed_changes": base.order_axes(proposed),
            "retained_constraints": base.order_axes(retained),
        }

    derived_proposed = proposed_axes_from_text(text)

    # Add explicit proposals, but never globally delete frozen predictions just
    # because a lexical recognizer failed to see them.
    missing = derived_proposed - proposed
    if missing:
        proposed |= missing
        diag.hit(i, "schedule_add_explicit_proposal_axes")

    # Explicit retention is authoritative and only suppresses the SAME axis.
    explicit_retained = retained_axes_from_text(text, proposed)
    if explicit_retained:
        retained |= explicit_retained
        proposed -= explicit_retained
        diag.hit(i, "schedule_explicit_retention")

    # "Sometime next month/week" is a calendar-horizon change; "sometime" is
    # not itself a time-of-day proposal.
    if (
        HORIZON_ONLY_RE.search(text)
        and "time_of_day" in proposed
        and not re.search(rf"(?:{DAYPART}|{CLOCK})", text, re.IGNORECASE)
    ):
        proposed.discard("time_of_day")
        diag.hit(i, "schedule_calendar_horizon_not_time_of_day")

    # Complete multi-axis negative availability only for availability semantics.
    if pair[1] == "availability":
        negative_axes = failed_axes_from_negative_clauses(text)
        if negative_axes - failed:
            failed |= negative_axes
            diag.hit(i, "schedule_negative_clause_axis_completion")

    # If a failure turn explicitly offers a fallback axis, the proposal clause
    # is more trustworthy than a contradictory whole-turn proposal prediction.
    # We only replace the base proposal set when the utterance actually contains
    # negative availability plus explicit proposal language.
    if (
        pair[1] == "availability"
        and NEGATIVE_AVAILABILITY_RE.search(text)
        and derived_proposed
    ):
        fallback_axes = set(derived_proposed) - retained
        if fallback_axes and proposed != fallback_axes:
            proposed = fallback_axes
            diag.hit(i, "schedule_failure_fallback_proposal_authority")

    # Corpus semantics intentionally treat a broad two-axis alternative after a
    # local failure as proposal-only for comma-so / semicolon coordination.
    broad_compound = bool(re.search(r",\s*so\b|;", text, re.IGNORECASE))
    if broad_compound and len(derived_proposed) >= 2 and failed:
        failed.clear()
        diag.hit(i, "schedule_broad_compound_failure_suppression")

    # Final invariant: an axis cannot be both proposed and retained.
    overlap = proposed & retained
    if overlap:
        proposed -= overlap
        diag.hit(i, "schedule_retention_precedence")

    return {
        "failed_constraints": base.order_axes(failed),
        "proposed_changes": base.order_axes(proposed),
        "retained_constraints": base.order_axes(retained),
    }


def latest_context_is_explicit_offer(turn) -> bool:
    return bool(turn.context and EXPLICIT_OPTION_OFFER_RE.search(str(turn.context[-1])))


def repair_dense_pairs(runtime, result, checkpoints, diag):
    p8a = load_mod("l2_candidate_p8a", base.P8A)
    p8c = load_mod("l2_candidate_p8c", base.P8C)

    model = p8a.DenseModel()
    model.load_state_dict(checkpoints["8a"]["state_dict"])
    model.eval()
    tok = base.tokenizer_for(p8a.MODEL_NAME)
    valid_pairs = tuple(tuple(x) for x in checkpoints["8a"]["valid_pairs"])

    repaired = list(result.dense_pairs)

    for i, turn in enumerate(runtime):
        pair = repaired[i]
        text = turn.utterance

        # Declarative negative availability is a statement, not an offer.
        if (
            pair == ("offer", "availability")
            and NEGATIVE_AVAILABILITY_RE.search(text)
            and "?" not in text
            and not OFFER_ACTION_RE.search(text)
        ):
            pair = ("statement", "availability")
            diag.hit(i, "dense_negative_availability_statement")

        # Conventional indirect request form: "Can I get your full name?"
        if (
            pair == ("question", "patient_fact")
            and PATIENT_FACT_REQUEST_RE.search(text)
        ):
            pair = ("request", "patient_fact")
            diag.hit(i, "dense_patient_fact_indirect_request")

        # Option-reference ambiguity is different depending on whether the
        # prior turn was an actual offer or only a non-offer enumeration.
        if result.ambiguity_details[i] == "option_reference" and VAGUE_SELECTION_RE.search(text):
            if latest_context_is_explicit_offer(turn):
                pair = ("confirmation", "availability")
                diag.hit(i, "dense_vague_selection_after_explicit_offer")
            else:
                pair = ("statement", "other")
                diag.hit(i, "dense_vague_selection_after_nonoffer_list")

        # Compound record-status + actionable availability offer: a background
        # record statement must not hide the later offer.
        clauses = p8c.split_clauses(text)
        if len(clauses) > 1 and pair in {
            ("statement", "appointment_state"),
            ("statement", "profile"),
        }:
            items = [
                base.RuntimeTurn(context=turn.context, utterance=clause)
                for clause in clauses
            ]
            clause_pairs, _, _, _ = p8a.predict(
                model,
                tok,
                items,
                valid_pairs,
            )
            if ("offer", "availability") in clause_pairs:
                pair = ("offer", "availability")
                diag.hit(i, "dense_background_record_plus_availability_offer")

        repaired[i] = pair

    del tok, model
    gc.collect()
    return repaired


def repair_requested_facts(runtime, base_frames, checkpoints, diag):
    p8b = load_mod("l2_candidate_p8b", base.P8B)
    p8c = load_mod("l2_candidate_p8c_fact", base.P8C)

    model = p8b.Model()
    model.load_state_dict(checkpoints["8b"]["state_dict"])
    model.eval()
    tok = base.tokenizer_for(p8b.p8a.MODEL_NAME)

    out = [frame.requested_fact for frame in base_frames]

    for i, turn in enumerate(runtime):
        clauses = p8c.split_clauses(turn.utterance)
        if len(clauses) < 2:
            continue

        items = [
            p8b.Ex(
                family="candidate_runtime",
                context=turn.context,
                turn=clause,
                fact=None,
            )
            for clause in clauses
        ]
        preds, _ = p8b.predict(model, tok, items)
        positives = [str(x) for x in preds if x is not None]
        unique = list(dict.fromkeys(positives))

        if len(unique) == 1 and unique[0] != out[i]:
            out[i] = unique[0]
            diag.hit(i, "requested_fact_unique_clause_local")

    del tok, model
    gc.collect()
    return out


def repair_references(runtime, result, diag):
    p7h = load_mod("l2_candidate_p7h", base.P7H)
    out = [frame.reference.value for frame in result.frames]

    for i, turn in enumerate(runtime):
        gate = result.gate_labels[i]
        detail = result.ambiguity_details[i]

        if detail == "option_reference" and gate.get("ambiguity", 0):
            wanted = "ambiguous" if latest_context_is_explicit_offer(turn) else "none"
            if out[i] != wanted:
                out[i] = wanted
                diag.hit(i, "reference_option_ambiguity_offer_boundary")
            continue

        if (
            gate.get("reference", 0)
            and not gate.get("ambiguity", 0)
            and not gate.get("oos", 0)
            and REJECT_ALTERNATIVE_RE.search(turn.utterance)
            and latest_context_is_explicit_offer(turn)
        ):
            candidates = tuple(p7h.phase7f.benchmark_candidates((turn.context[-1],)))
            candidates = base.dedupe(candidates)
            if len(candidates) == 1 and out[i] != "prior_option":
                out[i] = "prior_option"
                diag.hit(i, "reference_reject_single_prior_offer")

    return out


def construct_candidate_frames(runtime, result, checkpoints):
    diag = CandidateDiagnostics.create()

    dense_pairs = repair_dense_pairs(runtime, result, checkpoints, diag)
    requested_facts = repair_requested_facts(runtime, result.frames, checkpoints, diag)
    references = repair_references(runtime, result, diag)

    schedules = [
        normalize_scheduling(
            i,
            turn,
            result.gate_labels[i],
            result.scheduling[i],
            dense_pairs[i],
            diag,
        )
        for i, turn in enumerate(runtime)
    ]

    frames = []
    constructor_errors = defaultdict(list)

    for i, (turn, old) in enumerate(zip(runtime, result.frames)):
        pair = dense_pairs[i]
        sched = schedules[i]

        try:
            frame = SemanticFrame(
                raw_text=turn.utterance,
                speech_act=SpeechAct(pair[0]),
                topic=SemanticTopic(pair[1]),
                requested_fact=requested_facts[i],
                failed_constraints=tuple(ConstraintAxis(x) for x in sched["failed_constraints"]),
                proposed_changes=tuple(ConstraintAxis(x) for x in sched["proposed_changes"]),
                retained_constraints=tuple(ConstraintAxis(x) for x in sched["retained_constraints"]),
                offered_options=old.offered_options,
                selected_option=old.selected_option,
                record_claims=old.record_claims,
                transaction_operation=old.transaction_operation,
                transaction_signal=old.transaction_signal,
                reference=ReferenceKind(references[i]),
                ambiguity=SemanticAmbiguity(
                    kind=AmbiguityKind(old.ambiguity.kind.value),
                    candidates=tuple(old.ambiguity.candidates),
                    detail="",
                ),
            )
        except Exception as exc:
            constructor_errors[i].append(f"{type(exc).__name__}:{exc}")
            frame = old

        frames.append(frame)

    return frames, schedules, dense_pairs, requested_facts, references, diag, dict(constructor_errors)


def field_summary(cases, failures_by_case):
    passed = Counter()
    failed = Counter()
    for failures in failures_by_case:
        bad = {f.field for f in failures}
        for field in base.FIELDS:
            (failed if field in bad else passed)[field] += 1
    return passed, failed


def frame_key(frame):
    return (
        frame.speech_act.value,
        frame.topic.value,
        frame.requested_fact,
        tuple(x.value for x in frame.failed_constraints),
        tuple(x.value for x in frame.proposed_changes),
        tuple(x.value for x in frame.retained_constraints),
        tuple(frame.offered_options),
        frame.selected_option,
        tuple(x.value for x in frame.record_claims),
        frame.transaction_operation.value,
        frame.transaction_signal.value,
        frame.reference.value,
        frame.ambiguity.kind.value,
        tuple(frame.ambiguity.candidates),
    )


def main() -> int:
    print("========== LEVEL 2 CANDIDATE NORMALIZATION ARCHITECTURE AUDIT V2 ==========")
    print("telephony=DISABLED")
    print("training=NO")
    print("repo_modified=NO")
    print("runtime_wiring_modified=NO")
    print("v0_17_modified=NO")
    print("gold_runtime_inputs=NO")
    print("case_id_runtime_inputs=NO")
    print("category_runtime_inputs=NO")
    print("tags_runtime_inputs=NO")
    print("development_only=YES")

    checkpoints = base.validate_environment()
    source_before = base.source_snapshot()

    cases = list(load_semanticlab_cases())
    runtime = [
        base.RuntimeTurn(
            context=tuple(case.context),
            utterance=str(case.utterance),
        )
        for case in cases
    ]

    # Freeze all base inference before any gold scoring.
    result = base.assemble_level2(runtime, checkpoints)
    candidate = construct_candidate_frames(runtime, result, checkpoints)
    (
        frames,
        schedules,
        dense_pairs,
        requested_facts,
        references,
        diag,
        constructor_errors,
    ) = candidate

    print("inference_and_candidate_normalization_complete=YES")
    print("gold_scoring_begins_only_now=YES")

    baseline_failures = [
        evaluate_frame(case, frame)
        for case, frame in zip(cases, result.frames)
    ]
    candidate_failures = [
        evaluate_frame(case, frame)
        for case, frame in zip(cases, frames)
    ]

    baseline_exact = sum(not x for x in baseline_failures)
    candidate_exact = sum(not x for x in candidate_failures)

    print()
    print("========== EXACT-FRAME COMPARISON ==========")
    print("cases=", len(cases))
    print("baseline_exact=", baseline_exact, "/", len(cases), "accuracy=", round(baseline_exact / len(cases), 4))
    print("candidate_exact=", candidate_exact, "/", len(cases), "accuracy=", round(candidate_exact / len(cases), 4))
    print("net_exact_gain=", candidate_exact - baseline_exact)

    improved = []
    regressed = []
    changed_still_fail = []

    for i, case in enumerate(cases):
        b = bool(baseline_failures[i])
        c = bool(candidate_failures[i])
        changed = frame_key(result.frames[i]) != frame_key(frames[i])
        if b and not c:
            improved.append(case.case_id)
        if not b and c:
            regressed.append(case.case_id)
        if changed and c:
            changed_still_fail.append(case.case_id)

    print("improved_case_ids=", improved)
    print("regressed_case_ids=", regressed)
    print("changed_but_still_failing_case_ids=", changed_still_fail)

    passed, failed = field_summary(cases, candidate_failures)
    print()
    print("========== CANDIDATE PER-FIELD ACCURACY ==========")
    for field in base.FIELDS:
        p = passed[field]
        f = failed[field]
        print(field, f"pass={p}", f"fail={f}", f"accuracy={p/(p+f):.4f}")

    critical = [i for i, case in enumerate(cases) if "critical" in case.tags]
    critical_exact = sum(not candidate_failures[i] for i in critical) / len(critical)
    critical_safety_fields = {
        "record_claims",
        "transaction_operation",
        "transaction_signal",
        "ambiguity.kind",
        "ambiguity.candidates",
    }
    safety_pass = safety_total = 0
    for i in critical:
        bad = {f.field for f in candidate_failures[i]}
        for field in critical_safety_fields:
            safety_total += 1
            safety_pass += int(field not in bad)

    print()
    print("========== CRITICAL / SAFETY ==========")
    print("critical_cases=", len(critical))
    print("critical_exact_frame_accuracy=", round(critical_exact, 4))
    print("critical_safety_field_accuracy=", round(safety_pass / safety_total, 4))
    print("base_transaction_structural_violations=", sum(len(v) for v in result.transaction_structural_violations.values()))
    print("base_record_structural_violations=", sum(len(v) for v in result.record_structural_violations.values()))

    tx_or_record_changed = []
    for i, (before, after) in enumerate(zip(result.frames, frames)):
        if (
            before.transaction_operation != after.transaction_operation
            or before.transaction_signal != after.transaction_signal
            or before.record_claims != after.record_claims
            or before.offered_options != after.offered_options
            or before.selected_option != after.selected_option
        ):
            tx_or_record_changed.append(cases[i].case_id)
    print("protected_safety_or_option_fields_changed_case_ids=", tx_or_record_changed)

    print()
    print("========== RULE HIT COUNTS ==========")
    counts = Counter(rule for rules in diag.rules.values() for rule in rules)
    for rule, count in sorted(counts.items()):
        print(rule, "hits=", count)

    print()
    print("========== ALL CHANGED CASES ==========")
    for i, case in enumerate(cases):
        if frame_key(result.frames[i]) == frame_key(frames[i]):
            continue
        before_bad = [f.field for f in baseline_failures[i]]
        after_bad = [f.field for f in candidate_failures[i]]
        print()
        print(case.case_id)
        print(" text=", repr(case.utterance))
        print(" rules=", diag.rules.get(i, []))
        print(" before_failures=", before_bad)
        print(" after_failures=", after_bad)
        print(" before_pair=", result.frames[i].speech_act.value, result.frames[i].topic.value)
        print(" after_pair=", frames[i].speech_act.value, frames[i].topic.value)
        print(" before_schedule=", result.scheduling[i])
        print(" after_schedule=", schedules[i])
        print(" before_requested_fact=", repr(result.frames[i].requested_fact), "after=", repr(requested_facts[i]))
        print(" before_reference=", result.frames[i].reference.value, "after=", references[i])

    remaining = [i for i, failures in enumerate(candidate_failures) if failures]
    print()
    print("========== REMAINING FAILURES ==========")
    if not remaining:
        print("NONE")
    for i in remaining:
        case = cases[i]
        print()
        print(case.case_id, "FAIL")
        print(" text=", repr(case.utterance))
        print(" rules=", diag.rules.get(i, []))
        for failure in candidate_failures[i]:
            print(
                " ",
                failure.field,
                "expected=",
                repr(failure.expected),
                "actual=",
                repr(failure.actual),
            )

    source_after = base.source_snapshot()
    source_unchanged = source_before == source_after

    print()
    print("========== CANDIDATE DECISION ==========")
    print("constructor_errors=", sum(len(v) for v in constructor_errors.values()))
    print("source_tree_python_unchanged=", "YES" if source_unchanged else "NO")
    print("specialist_retraining_performed=NO")
    print("runtime_wiring_performed=NO")
    print("development_corpus_not_final_holdout=YES")
    print("level2_frozen=NO")

    strong = (
        candidate_exact / len(cases) >= 0.98
        and critical_exact == 1.0
        and safety_pass == safety_total
        and not regressed
        and not tx_or_record_changed
        and not constructor_errors
        and source_unchanged
    )

    if strong:
        print("CANDIDATE_NORMALIZATION_ARCHITECTURE_V2=STRONG_DEVELOPMENT_PASS")
        print("NEXT_ACTION=REVIEW_RULE_GENERALITY_THEN_INTEGRATE_READ_ONLY_ASSEMBLER_AND_RUN_REGRESSIONS")
    else:
        print("CANDIDATE_NORMALIZATION_ARCHITECTURE_V2=NOT_STRONG")
        print("NEXT_ACTION=AUDIT_ONLY_THE_REMAINING_OR_REGRESSED_RULE_INTERACTIONS_BEFORE_RETRAINING")

    return 0 if strong else 2


if __name__ == "__main__":
    raise SystemExit(main())
