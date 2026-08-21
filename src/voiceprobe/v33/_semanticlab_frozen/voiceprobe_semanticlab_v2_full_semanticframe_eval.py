#!/usr/bin/env python3
"""Read-only full Level 2 SemanticFrame development assembler/evaluator.

Purpose
-------
Assemble one native ``SemanticFrame`` for each of the 133 SemanticLab v2
DEVELOPMENT cases using only the saved Level 2 components and deterministic
semantic adapters. Gold labels are never exposed to runtime inference.

This script:
- does not train
- does not wire or modify runtime
- does not modify the frozen v0.17 reasoner
- does not place a phone call
- does not write checkpoints or reports
- uses the 133-case corpus only after inference for evaluation

Selected architecture
---------------------
Scheduling:
    frozen Phase 6B
    gated by PREDICTED Phase 7C discourse routing

Discourse:
    frozen Phase 7C reference/ambiguity/OOS gate
    frozen Phase 7D reference kind
    frozen Phase 7I selection operator
    frozen Phase 7J ambiguity detail
    latest deterministic Phase 7J structured candidate resolver

Dense semantics:
    frozen Phase 8A3 legal act/topic pair decoder

Requested facts:
    frozen Phase 8B1

Record claims:
    frozen Phase 8C
    accepted only behind frozen Phase 8A3 clause-level legality:
        statement x profile
        statement x appointment_state

Transactions:
    frozen Phase 8A3 whole-turn transaction gate
    deterministic predicate normalizer
    deterministic signal derivation
    (no learned Phase 8D operation checkpoint is used)

Offered options:
    frozen Phase 8A3 clause semantics
    deterministic positive-availability concrete-slot extractor

Safe bare-Yes rule:
    applied only when:
      * assembled speech act == confirmation
      * latest prior clinic turn contains one explicit legal offer
      * exactly one legal structured candidate exists
    It may resolve selected_option only.
    It MUST NOT grant transaction authorization.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

# This evaluator is intentionally offline/read-only. Prevent accidental model
# downloads and Python bytecode writes while dynamically importing experiments.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import torch
from transformers import AutoTokenizer

from voiceprobe.v33.semantic_corpus import load_semanticlab_cases
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
from voiceprobe.v33.semantic_frame_eval import evaluate_frame


ROOT = Path(".").resolve()

# ---------------------------------------------------------------------------
# Selected scripts/checkpoints. These are read only.
# ---------------------------------------------------------------------------

P6B = ROOT / "voiceprobe_semanticlab_v2_phase6b_distilbert_contrastive.py"

P7C = ROOT / "voiceprobe_semanticlab_v2_phase7c_targeted_gate.py"
P7D = ROOT / "voiceprobe_semanticlab_v2_phase7d_reference_kind.py"
P7H = ROOT / "voiceprobe_semanticlab_v2_phase7h_selection_operator.py"
P7J = ROOT / "voiceprobe_semanticlab_v2_phase7j_ambiguity_detail_fixed.py"
P7JR = ROOT / "voiceprobe_semanticlab_v2_phase7j_candidate_resolver_audit.py"

P8A = ROOT / "voiceprobe_semanticlab_v2_phase8a_speech_act_topic.py"
P8B = ROOT / "voiceprobe_semanticlab_v2_phase8b_requested_fact.py"
P8C = ROOT / "voiceprobe_semanticlab_v2_phase8c_record_claims.py"
P8D = ROOT / "voiceprobe_semanticlab_v2_phase8d_transactions.py"
P8DN = ROOT / "voiceprobe_semanticlab_v2_phase8d_predicate_normalizer_audit.py"
POPT = ROOT / "voiceprobe_semanticlab_v2_offered_options_audit.py"

A6B = ROOT / "artifacts/semanticlab_v2_phase6b_distilbert_scheduling.pt"
A7C = ROOT / "artifacts/semanticlab_v2_phase7c_factorized_gate.pt"
A7D = ROOT / "artifacts/semanticlab_v2_phase7d_reference_kind.pt"
A7I = ROOT / "artifacts/semanticlab_v2_phase7i_selection_operator.pt"
A7J = ROOT / "artifacts/semanticlab_v2_phase7j_ambiguity_detail.pt"
A8A3 = ROOT / "artifacts/semanticlab_v2_phase8a3_speech_act_topic.pt"
A8B1 = ROOT / "artifacts/semanticlab_v2_phase8b1_requested_fact.pt"
A8C = ROOT / "artifacts/semanticlab_v2_phase8c_record_claims.pt"

REASONER = ROOT / "src/voiceprobe/v33/reasoner.py"

REQUIRED = (
    P6B,
    P7C,
    P7D,
    P7H,
    P7J,
    P7JR,
    P8A,
    P8B,
    P8C,
    P8D,
    P8DN,
    POPT,
    A6B,
    A7C,
    A7D,
    A7I,
    A7J,
    A8A3,
    A8B1,
    A8C,
    REASONER,
)

CHECKPOINTS = (
    A6B,
    A7C,
    A7D,
    A7I,
    A7J,
    A8A3,
    A8B1,
    A8C,
)

EXPECTED_FACTS = (
    "full_name",
    "first_name",
    "last_name",
    "dob",
    "insurance",
    "complaint",
    "reschedule_reason",
    "visit_type",
)

EXPECTED_RECORD_ENTITIES = ("profile", "appointment")
EXPECTED_RECORD_STATES = ("exists", "missing")

LEGAL_RECORD_PAIRS = {
    ("statement", "profile"),
    ("statement", "appointment_state"),
}

FIELDS = (
    "speech_act",
    "topic",
    "requested_fact",
    "failed_constraints",
    "proposed_changes",
    "retained_constraints",
    "offered_options",
    "selected_option",
    "record_claims",
    "transaction_operation",
    "transaction_signal",
    "reference",
    "ambiguity.kind",
    "ambiguity.candidates",
)

SCHEDULING_FIELDS = (
    "failed_constraints",
    "proposed_changes",
    "retained_constraints",
)

REFERENCE_AMBIGUITY_FIELDS = (
    "reference",
    "ambiguity.kind",
    "ambiguity.candidates",
)

TRANSACTION_FIELDS = (
    "transaction_operation",
    "transaction_signal",
)

CRITICAL_SAFETY_FIELDS = (
    "failed_constraints",
    "proposed_changes",
    "retained_constraints",
    "selected_option",
    "record_claims",
    "transaction_operation",
    "transaction_signal",
    "reference",
    "ambiguity.kind",
    "ambiguity.candidates",
)

MUTATING_OPERATIONS = {
    "book",
    "reschedule",
    "cancel",
    "keep",
    "create_profile",
}


@dataclass(frozen=True, slots=True)
class RuntimeTurn:
    """The ONLY whole-turn object exposed to runtime classifiers."""

    context: tuple[str, ...]
    utterance: str


@dataclass(frozen=True, slots=True)
class ClauseSpan:
    start: int
    end: int


@dataclass(slots=True)
class AssemblyResult:
    frames: list[SemanticFrame]
    assembly_violations: dict[int, list[str]]
    transaction_structural_violations: dict[int, list[str]]
    record_structural_violations: dict[int, list[str]]
    bare_yes_applied: set[int]
    gate_labels: list[dict[str, int]]
    dense_pairs: list[tuple[str, str]]
    reference_kinds: list[str]
    ambiguity_details: list[str]
    scheduling: list[dict[str, tuple[str, ...]]]


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tokenizer_for(model_name: str):
    return AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_snapshot() -> dict[str, str]:
    base = ROOT / "src/voiceprobe"
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted(base.rglob("*.py"))
    }


def pct(n: int, d: int) -> float:
    return n / d if d else 1.0


def norm_text(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def dedupe(values: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value).strip()
        key = norm_text(value)
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return tuple(out)


def order_axes(values: Iterable[str]) -> tuple[str, ...]:
    order = ("day", "time_of_day", "provider")
    present = set(values)
    return tuple(axis for axis in order if axis in present)


def print_checkpoint_manifest() -> None:
    print("========== CHECKPOINT MANIFEST ==========")
    for path in CHECKPOINTS:
        print(
            path.relative_to(ROOT),
            "sha256=" + sha256_file(path)[:20],
        )


def validate_environment() -> dict[str, Any]:
    missing = [path for path in REQUIRED if not path.is_file()]
    if missing:
        raise SystemExit(
            "Missing required Level 2 files:\n"
            + "\n".join(f"  {path}" for path in missing)
        )

    reasoner_text = REASONER.read_text(encoding="utf-8")
    if "semantic-planner v0.17." not in reasoner_text:
        raise SystemExit(
            "Frozen semantic-planner v0.17 marker was not detected. "
            "Refusing to run this evaluator against a different baseline."
        )

    ck6b = load_checkpoint(A6B)
    ck7c = load_checkpoint(A7C)
    ck7d = load_checkpoint(A7D)
    ck7i = load_checkpoint(A7I)
    ck7j = load_checkpoint(A7J)
    ck8a = load_checkpoint(A8A3)
    ck8b = load_checkpoint(A8B1)
    ck8c = load_checkpoint(A8C)

    return {
        "6b": ck6b,
        "7c": ck7c,
        "7d": ck7d,
        "7i": ck7i,
        "7j": ck7j,
        "8a": ck8a,
        "8b": ck8b,
        "8c": ck8c,
    }


def build_clauses(runtime: list[RuntimeTurn], p8c):
    flat: list[RuntimeTurn] = []
    spans: list[ClauseSpan] = []
    text_by_case: list[list[str]] = []

    for turn in runtime:
        clauses = list(p8c.split_clauses(turn.utterance))
        start = len(flat)
        for clause in clauses:
            flat.append(
                RuntimeTurn(
                    context=turn.context,
                    utterance=clause,
                )
            )
        spans.append(ClauseSpan(start=start, end=len(flat)))
        text_by_case.append(clauses)

    return flat, spans, text_by_case


def run_phase7c_gate(runtime, p7c, ck7c):
    model = p7c.phase7b.Model()
    model.load_state_dict(ck7c["state_dict"])
    model.eval()

    model_name = ck7c.get(
        "model_name",
        getattr(p7c, "MODEL_NAME", p7c.phase7b.MODEL_NAME),
    )
    tok = tokenizer_for(model_name)

    probs = p7c.raw_probs(model, tok, runtime)
    labels = p7c.decode(probs, ck7c["thresholds"])

    del probs, tok, model
    gc.collect()
    return labels


def run_phase6b_scheduling(runtime, gate_labels, p6b, ck6b):
    model = p6b.Model()
    model.load_state_dict(ck6b["state_dict"])
    model.eval()
    tok = tokenizer_for(ck6b["model_name"])

    failed_probs, relation_probs = p6b.raw_predictions(
        model,
        tok,
        [turn.utterance for turn in runtime],
    )
    raw = p6b.decode(
        failed_probs,
        relation_probs,
        dict(ck6b["failed_thresholds"]),
        dict(ck6b["relation_thresholds"]),
    )

    out: list[dict[str, tuple[str, ...]]] = []

    for pred, gate in zip(raw, gate_labels):
        diverted = bool(
            gate.get("reference", 0)
            or gate.get("ambiguity", 0)
            or gate.get("oos", 0)
        )

        if diverted:
            out.append(
                {
                    "failed_constraints": (),
                    "proposed_changes": (),
                    "retained_constraints": (),
                }
            )
            continue

        out.append(
            {
                "failed_constraints": order_axes(
                    pred.get("failed_constraints", ())
                ),
                "proposed_changes": order_axes(
                    pred.get("proposed_changes", ())
                ),
                "retained_constraints": order_axes(
                    pred.get("retained_constraints", ())
                ),
            }
        )

    del failed_probs, relation_probs, raw, tok, model
    gc.collect()
    return out


def run_phase8a_dense(runtime, flat_clauses, p8a, ck8a):
    model = p8a.DenseModel()
    model.load_state_dict(ck8a["state_dict"])
    model.eval()
    tok = tokenizer_for(p8a.MODEL_NAME)

    valid_pairs = tuple(tuple(x) for x in ck8a["valid_pairs"])

    whole_pairs, _, independent_acts, independent_topics = p8a.predict(
        model,
        tok,
        runtime,
        valid_pairs,
    )

    clause_pairs: list[tuple[str, str]] = []
    if flat_clauses:
        clause_pairs, _, _, _ = p8a.predict(
            model,
            tok,
            flat_clauses,
            valid_pairs,
        )

    del tok, model
    gc.collect()

    return (
        whole_pairs,
        independent_acts,
        independent_topics,
        clause_pairs,
        valid_pairs,
    )


def run_phase8b_requested_fact(runtime, p8b, ck8b):
    model = p8b.Model()
    model.load_state_dict(ck8b["state_dict"])
    model.eval()
    tok = tokenizer_for(p8b.p8a.MODEL_NAME)

    preds, _ = p8b.predict(model, tok, runtime)
    out = [str(value) if value is not None else "" for value in preds]

    del tok, model
    gc.collect()
    return out


def run_phase8c_record_claims(
    runtime,
    flat_clause_turns,
    spans,
    clause_pairs,
    p8c,
    ck8c,
):
    model = p8c.Model()
    model.load_state_dict(ck8c["state_dict"])
    model.eval()
    tok = tokenizer_for(p8c.p8a.MODEL_NAME)

    items = [
        p8c.Ex(
            family="runtime",
            context=turn.context,
            turn=turn.utterance,
            claim=None,
        )
        for turn in flat_clause_turns
    ]

    flat_preds, _ = p8c.predict_clauses(model, tok, items)

    claims_by_case: list[tuple[str, ...]] = []
    provenance_by_case: list[list[tuple[str, tuple[str, str]]]] = []

    for span in spans:
        claims: list[str] = []
        provenance: list[tuple[str, tuple[str, str]]] = []

        for j in range(span.start, span.end):
            pred = flat_preds[j]
            pair = clause_pairs[j]

            if pred is not None and pair in LEGAL_RECORD_PAIRS:
                if pred not in claims:
                    claims.append(pred)
                provenance.append((pred, pair))

        claims_by_case.append(tuple(sorted(claims)))
        provenance_by_case.append(provenance)

    del flat_preds, items, tok, model
    gc.collect()
    return claims_by_case, provenance_by_case


def run_phase7d_reference_kind(runtime, p7d, ck7d):
    model = p7d.RefKindModel()
    model.load_state_dict(ck7d["state_dict"])
    model.eval()

    model_name = ck7d.get(
        "model_name",
        getattr(p7d, "MODEL_NAME", "distilbert/distilbert-base-uncased"),
    )
    tok = tokenizer_for(model_name)
    preds, _ = p7d.predict(model, tok, runtime)

    del tok, model
    gc.collect()
    return [str(x) for x in preds]


def run_phase7j_ambiguity(
    runtime,
    gate_labels,
    p7j,
    p7jr,
    ck7j,
    assembly_violations,
):
    n = len(runtime)
    kinds = ["none"] * n
    candidates: list[tuple[str, ...]] = [tuple() for _ in range(n)]
    details = [""] * n

    active_indices = [
        i
        for i, gate in enumerate(gate_labels)
        if gate.get("ambiguity", 0) or gate.get("oos", 0)
    ]

    if not active_indices:
        return kinds, candidates, details

    model = p7j.DetailModel()
    model.load_state_dict(ck7j["state_dict"])
    model.eval()

    model_name = ck7j.get(
        "model_name",
        getattr(p7j, "MODEL_NAME", "distilbert/distilbert-base-uncased"),
    )
    tok = tokenizer_for(model_name)

    items = [
        SimpleNamespace(
            context=runtime[i].context,
            turn=runtime[i].utterance,
        )
        for i in active_indices
    ]

    preds, _ = p7j.predict_detail(model, tok, items)

    for i, detail in zip(active_indices, preds):
        details[i] = str(detail)
        try:
            kind, cands = p7jr.ambiguity_from_detail(
                detail,
                runtime[i].context,
                runtime[i].utterance,
            )
            kinds[i] = str(kind)
            candidates[i] = dedupe(cands)
        except Exception as exc:
            assembly_violations[i].append(
                "phase7j_candidate_resolution_error:"
                f"{type(exc).__name__}:{exc}"
            )
            # Conservative diagnostic fallback. It is visible as an assembly
            # violation and therefore can never produce a STRONG gate.
            kinds[i] = "none"
            candidates[i] = ()

    del tok, model
    gc.collect()
    return kinds, candidates, details


BARE_YES_RE = re.compile(r"^\s*yes\s*[.!]?\s*$", re.IGNORECASE)

EXPLICIT_LEGAL_OFFER_RE = re.compile(
    r"\b("
    r"would\s+you\s+like|"
    r"do\s+you\s+want(?:\s+me\s+to)?|"
    r"i\s+can\s+offer|"
    r"we\s+can\s+offer|"
    r"(?:can|may|shall|should)\s+i\s+"
    r"(?:book|schedule|reschedule|move)|"
    r"i\s+can\s+(?:book|schedule|reschedule|move)"
    r")\b",
    re.IGNORECASE,
)


def bare_yes_candidate(
    turn: RuntimeTurn,
    assembled_speech_act: str,
    p7h,
) -> str | None:
    if assembled_speech_act != "confirmation":
        return None
    if not BARE_YES_RE.fullmatch(turn.utterance):
        return None
    if not turn.context:
        return None

    previous = str(turn.context[-1])
    if not EXPLICIT_LEGAL_OFFER_RE.search(previous):
        return None

    candidates = tuple(
        p7h.phase7f.benchmark_candidates((previous,))
    )
    candidates = dedupe(candidates)

    if len(candidates) != 1:
        return None

    return candidates[0]


def run_phase7i_selected_option(
    runtime,
    gate_labels,
    reference_kinds,
    dense_pairs,
    p7h,
    ck7i,
    assembly_violations,
):
    n = len(runtime)
    selected = [""] * n
    bare_yes_applied: set[int] = set()

    # The bare-Yes rule is deterministic and is evaluated before the learned
    # selection operator. It only resolves selected_option.
    for i, turn in enumerate(runtime):
        candidate = bare_yes_candidate(
            turn,
            dense_pairs[i][0],
            p7h,
        )
        if candidate is not None:
            selected[i] = candidate
            bare_yes_applied.add(i)

    eligible: list[int] = []
    scenarios: list[Any] = []

    for i, (turn, gate, kind) in enumerate(
        zip(runtime, gate_labels, reference_kinds)
    ):
        if i in bare_yes_applied:
            continue

        if not gate.get("reference", 0):
            continue

        # Never resolve an option while the gate still says the turn is
        # ambiguous. That would silently coerce uncertainty.
        if gate.get("ambiguity", 0) or gate.get("oos", 0):
            continue

        if kind == "unresolved":
            assembly_violations[i].append(
                "unresolved_reference_kind_not_coerced"
            )
            continue

        candidates = dedupe(
            p7h.phase7f.benchmark_candidates(turn.context)
        )

        scenarios.append(
            SimpleNamespace(
                context=turn.context,
                kind=kind,
                candidates=candidates,
                turn=turn.utterance,
            )
        )
        eligible.append(i)

    if not scenarios:
        return selected, bare_yes_applied

    model = p7h.OperatorModel()
    model.load_state_dict(ck7i["state_dict"])
    model.eval()

    model_name = ck7i.get(
        "model_name",
        getattr(p7h, "MODEL_NAME", "distilbert/distilbert-base-uncased"),
    )
    tok = tokenizer_for(model_name)

    operators, _ = p7h.predict(model, tok, scenarios)

    for i, scenario, operator in zip(eligible, scenarios, operators):
        resolved = p7h.resolve(
            operator,
            scenario.context,
            scenario.candidates,
            scenario.turn,
        )
        selected[i] = "" if resolved is None else str(resolved)

    del tok, model
    gc.collect()
    return selected, bare_yes_applied


def assemble_reference(
    gate,
    predicted_kind: str,
    ambiguity_kind: str,
    assembly_violations: list[str],
) -> str:
    if not gate.get("reference", 0):
        return "none"

    if gate.get("ambiguity", 0) or gate.get("oos", 0):
        if ambiguity_kind == "none":
            assembly_violations.append(
                "reference_ambiguous_but_ambiguity_unresolved"
            )
            return "none"
        return "ambiguous"

    if predicted_kind == "unresolved":
        assembly_violations.append(
            "predicted_reference_active_but_kind_unresolved"
        )
        return "none"

    return predicted_kind


def build_offered_options(
    runtime,
    spans,
    clause_texts,
    clause_pairs,
    popt,
):
    out: list[tuple[str, ...]] = []

    for span, clauses in zip(spans, clause_texts):
        offered: list[str] = []

        for local_i, clause in enumerate(clauses):
            j = span.start + local_i
            pair = clause_pairs[j]
            positive = bool(popt.POSITIVE_AVAILABILITY.search(clause))
            negative = bool(popt.NEGATIVE_AVAILABILITY.search(clause))
            slots = popt.extract_concrete_slots(clause)

            accepted = (
                pair[1] == "availability"
                and positive
                and not negative
                and bool(slots)
            )
            if accepted:
                for slot in slots:
                    if slot not in offered:
                        offered.append(slot)

        out.append(tuple(offered))

    return out


def build_transactions(
    runtime,
    dense_pairs,
    independent_acts,
    p8d,
    p8dn,
    assembly_violations,
):
    operations: list[str] = []
    signals: list[str] = []

    for i, (turn, pair, independent_act) in enumerate(
        zip(runtime, dense_pairs, independent_acts)
    ):
        if pair[1] != "transaction":
            operations.append("none")
            signals.append("none")
            continue

        # The native frame's speech act is the ontology-valid pair act. If the
        # independent act head disagrees on a transaction turn, expose that
        # interaction rather than silently deriving consent from another act.
        if independent_act != pair[0]:
            assembly_violations[i].append(
                "dense_transaction_act_disagreement:"
                f"pair={pair[0]}:independent={independent_act}"
            )

        operation = str(p8dn.normalize_operation(turn.utterance))
        signal = p8d.derive_signal(pair[0], operation)

        if signal is None:
            assembly_violations[i].append(
                "unsupported_transaction_signal:"
                f"act={pair[0]}:operation={operation}"
            )
            signal = "none"

        operations.append(operation)
        signals.append(str(signal))

    return operations, signals


def check_structural_transaction_safety(
    dense_pair,
    speech_act,
    operation,
    signal,
    bare_yes_applied,
) -> list[str]:
    issues: list[str] = []

    if dense_pair[1] != "transaction":
        if operation != "none":
            issues.append("operation_outside_transaction_gate")
        if signal != "none":
            issues.append("signal_outside_transaction_gate")

    if operation == "none" and signal != "none":
        issues.append("authorizing_signal_without_operation")

    if operation == "search" and signal != "none":
        issues.append("read_only_search_has_authorizing_signal")

    if operation in MUTATING_OPERATIONS:
        expected_by_act = {
            "question": "permission_request",
            "statement": "proposed",
            "confirmation": "confirmed",
        }
        expected = expected_by_act.get(speech_act)
        if expected is None:
            if signal != "none":
                issues.append(
                    "mutation_signal_on_unsupported_speech_act"
                )
        elif signal != expected:
            issues.append(
                f"mutation_signal_mismatch:{signal}!={expected}"
            )

    if bare_yes_applied and (
        operation != "none" or signal != "none"
    ):
        issues.append("bare_yes_granted_transaction_authorization")

    return issues


def check_structural_record_safety(
    claims,
    provenance,
) -> list[str]:
    issues: list[str] = []

    claim_set = set(claims)
    if {
        "profile_exists",
        "profile_missing",
    }.issubset(claim_set):
        issues.append("conflicting_profile_record_claims")

    if {
        "appointment_exists",
        "appointment_missing",
    }.issubset(claim_set):
        issues.append("conflicting_appointment_record_claims")

    supported = {claim for claim, pair in provenance if pair in LEGAL_RECORD_PAIRS}
    for claim in claims:
        if claim not in supported:
            issues.append(
                f"record_claim_without_legal_clause_support:{claim}"
            )

    return issues


def safe_frame_fallback(
    turn: RuntimeTurn,
    speech_act: str,
    topic: str,
) -> SemanticFrame:
    """Construct a minimal predicted frame after a visible assembly error."""

    return SemanticFrame(
        raw_text=turn.utterance,
        speech_act=SpeechAct(speech_act),
        topic=SemanticTopic(topic),
    )


def assemble_level2(
    runtime: list[RuntimeTurn],
    checkpoints: dict[str, Any],
) -> AssemblyResult:
    # Load script modules. Their main() functions are guarded and are not run.
    p6b = load_mod("l2eval_p6b", P6B)
    p7c = load_mod("l2eval_p7c", P7C)
    p7d = load_mod("l2eval_p7d", P7D)
    p7h = load_mod("l2eval_p7h", P7H)
    p7j = load_mod("l2eval_p7j", P7J)
    p7jr = load_mod("l2eval_p7jr", P7JR)
    p8a = load_mod("l2eval_p8a", P8A)
    p8b = load_mod("l2eval_p8b", P8B)
    p8c = load_mod("l2eval_p8c", P8C)
    p8d = load_mod("l2eval_p8d", P8D)
    p8dn = load_mod("l2eval_p8dn", P8DN)
    popt = load_mod("l2eval_popt", POPT)

    # Frozen ontology/checkpoint sanity. These are metadata checks only.
    if tuple(p8b.FACTS) != EXPECTED_FACTS:
        raise RuntimeError(
            f"Phase 8B fact ontology changed: {tuple(p8b.FACTS)!r}"
        )
    if tuple(p8c.ENTITIES) != EXPECTED_RECORD_ENTITIES:
        raise RuntimeError(
            f"Phase 8C entities changed: {tuple(p8c.ENTITIES)!r}"
        )
    if tuple(p8c.STATES) != EXPECTED_RECORD_STATES:
        raise RuntimeError(
            f"Phase 8C states changed: {tuple(p8c.STATES)!r}"
        )

    n = len(runtime)
    assembly_violations: dict[int, list[str]] = defaultdict(list)
    tx_structural: dict[int, list[str]] = defaultdict(list)
    record_structural: dict[int, list[str]] = defaultdict(list)

    # ------------------------------ discourse gate ---------------------------
    gate_labels = run_phase7c_gate(
        runtime,
        p7c,
        checkpoints["7c"],
    )

    # ------------------------------ scheduling -------------------------------
    scheduling = run_phase6b_scheduling(
        runtime,
        gate_labels,
        p6b,
        checkpoints["6b"],
    )

    # Preserve native SemanticFrame's disjoint proposed/retained invariant.
    for i, sched in enumerate(scheduling):
        overlap = (
            set(sched["proposed_changes"])
            & set(sched["retained_constraints"])
        )
        if overlap:
            for axis in sorted(overlap):
                assembly_violations[i].append(
                    "axis_both_proposed_and_retained:" + axis
                )
            # Conservative invariant-preserving diagnostic behavior: an axis
            # whose relation is self-contradictory is emitted as neither.
            sched["proposed_changes"] = tuple(
                x
                for x in sched["proposed_changes"]
                if x not in overlap
            )
            sched["retained_constraints"] = tuple(
                x
                for x in sched["retained_constraints"]
                if x not in overlap
            )

    # ------------------------------ clauses ----------------------------------
    flat_clauses, spans, clause_texts = build_clauses(runtime, p8c)

    # ------------------------------ dense semantics --------------------------
    (
        dense_pairs,
        independent_acts,
        _independent_topics,
        clause_pairs,
        _valid_pairs,
    ) = run_phase8a_dense(
        runtime,
        flat_clauses,
        p8a,
        checkpoints["8a"],
    )

    # ------------------------------ requested fact ---------------------------
    requested_facts = run_phase8b_requested_fact(
        runtime,
        p8b,
        checkpoints["8b"],
    )

    # ------------------------------ record claims ----------------------------
    record_claims, record_provenance = run_phase8c_record_claims(
        runtime,
        flat_clauses,
        spans,
        clause_pairs,
        p8c,
        checkpoints["8c"],
    )

    # ------------------------------ offered options --------------------------
    offered_options = build_offered_options(
        runtime,
        spans,
        clause_texts,
        clause_pairs,
        popt,
    )

    # ------------------------------ references -------------------------------
    reference_kinds = run_phase7d_reference_kind(
        runtime,
        p7d,
        checkpoints["7d"],
    )

    # ------------------------------ ambiguity --------------------------------
    (
        ambiguity_kinds,
        ambiguity_candidates,
        ambiguity_details,
    ) = run_phase7j_ambiguity(
        runtime,
        gate_labels,
        p7j,
        p7jr,
        checkpoints["7j"],
        assembly_violations,
    )

    references: list[str] = []
    for i in range(n):
        references.append(
            assemble_reference(
                gate_labels[i],
                reference_kinds[i],
                ambiguity_kinds[i],
                assembly_violations[i],
            )
        )

    # ------------------------------ selected option --------------------------
    selected_options, bare_yes_applied = run_phase7i_selected_option(
        runtime,
        gate_labels,
        reference_kinds,
        dense_pairs,
        p7h,
        checkpoints["7i"],
        assembly_violations,
    )

    # ------------------------------ transaction ------------------------------
    transaction_operations, transaction_signals = build_transactions(
        runtime,
        dense_pairs,
        independent_acts,
        p8d,
        p8dn,
        assembly_violations,
    )

    # ------------------------------ structural safety ------------------------
    for i in range(n):
        tx_issues = check_structural_transaction_safety(
            dense_pairs[i],
            dense_pairs[i][0],
            transaction_operations[i],
            transaction_signals[i],
            i in bare_yes_applied,
        )
        if tx_issues:
            tx_structural[i].extend(tx_issues)

        rec_issues = check_structural_record_safety(
            record_claims[i],
            record_provenance[i],
        )
        if rec_issues:
            record_structural[i].extend(rec_issues)

    # ------------------------------ native frame -----------------------------
    frames: list[SemanticFrame] = []

    for i, turn in enumerate(runtime):
        pair = dense_pairs[i]
        sched = scheduling[i]

        try:
            frame = SemanticFrame(
                raw_text=turn.utterance,
                speech_act=SpeechAct(pair[0]),
                topic=SemanticTopic(pair[1]),
                requested_fact=requested_facts[i],
                failed_constraints=tuple(
                    ConstraintAxis(x)
                    for x in sched["failed_constraints"]
                ),
                proposed_changes=tuple(
                    ConstraintAxis(x)
                    for x in sched["proposed_changes"]
                ),
                retained_constraints=tuple(
                    ConstraintAxis(x)
                    for x in sched["retained_constraints"]
                ),
                offered_options=tuple(offered_options[i]),
                selected_option=selected_options[i],
                record_claims=tuple(
                    RecordClaim(x)
                    for x in record_claims[i]
                ),
                transaction_operation=TransactionOperation(
                    transaction_operations[i]
                ),
                transaction_signal=TransactionSignal(
                    transaction_signals[i]
                ),
                reference=ReferenceKind(references[i]),
                ambiguity=SemanticAmbiguity(
                    kind=AmbiguityKind(ambiguity_kinds[i]),
                    candidates=tuple(ambiguity_candidates[i]),
                    detail="",
                ),
            )
        except Exception as exc:
            assembly_violations[i].append(
                "semanticframe_constructor_error:"
                f"{type(exc).__name__}:{exc}"
            )
            # Still produce one native SemanticFrame for every development
            # case, but never conceal the fact that assembly degraded.
            frame = safe_frame_fallback(
                turn,
                pair[0],
                pair[1],
            )

        frames.append(frame)

    return AssemblyResult(
        frames=frames,
        assembly_violations=dict(assembly_violations),
        transaction_structural_violations=dict(tx_structural),
        record_structural_violations=dict(record_structural),
        bare_yes_applied=bare_yes_applied,
        gate_labels=gate_labels,
        dense_pairs=dense_pairs,
        reference_kinds=reference_kinds,
        ambiguity_details=ambiguity_details,
        scheduling=scheduling,
    )


def summarize_field_accuracy(cases, failures_by_case):
    field_pass = Counter()
    field_fail = Counter()

    for i, _case in enumerate(cases):
        failed = {f.field for f in failures_by_case[i]}
        for field in FIELDS:
            if field in failed:
                field_fail[field] += 1
            else:
                field_pass[field] += 1

    return field_pass, field_fail


def joint_accuracy(
    failures_by_case,
    fields,
    indices: Iterable[int] | None = None,
):
    field_set = set(fields)
    selected = (
        list(range(len(failures_by_case)))
        if indices is None
        else list(indices)
    )
    passed = 0
    for i in selected:
        failed = {f.field for f in failures_by_case[i]}
        passed += int(not (failed & field_set))
    return passed, len(selected)


def score_and_report(
    cases,
    result: AssemblyResult,
    source_before,
):
    # IMPORTANT: This is the first stage that reads gold expected/tags/category.
    # All component inference and SemanticFrame assembly are already complete.
    failures_by_case = [
        evaluate_frame(case, frame)
        for case, frame in zip(cases, result.frames)
    ]

    field_pass, field_fail = summarize_field_accuracy(
        cases,
        failures_by_case,
    )

    exact = sum(not failures for failures in failures_by_case)
    total = len(cases)
    total_field_pass = sum(field_pass.values())
    total_field_fail = sum(field_fail.values())

    critical_indices = [
        i
        for i, case in enumerate(cases)
        if "critical" in tuple(case.tags)
    ]

    critical_exact = sum(
        not failures_by_case[i]
        for i in critical_indices
    )

    critical_field_pass = 0
    critical_field_total = len(critical_indices) * len(FIELDS)
    critical_safety_pass = 0
    critical_safety_total = (
        len(critical_indices) * len(CRITICAL_SAFETY_FIELDS)
    )

    for i in critical_indices:
        failed = {f.field for f in failures_by_case[i]}
        critical_field_pass += sum(
            field not in failed
            for field in FIELDS
        )
        critical_safety_pass += sum(
            field not in failed
            for field in CRITICAL_SAFETY_FIELDS
        )

    sched_pass, sched_total = joint_accuracy(
        failures_by_case,
        SCHEDULING_FIELDS,
    )
    refamb_pass, refamb_total = joint_accuracy(
        failures_by_case,
        REFERENCE_AMBIGUITY_FIELDS,
    )
    tx_joint_pass, tx_joint_total = joint_accuracy(
        failures_by_case,
        TRANSACTION_FIELDS,
    )
    record_joint_pass, record_joint_total = joint_accuracy(
        failures_by_case,
        ("record_claims",),
    )

    # Safety violations combine independent structural invariants and
    # development gold mismatches on safety-critical semantic fields.
    tx_safety: dict[int, list[str]] = defaultdict(list)
    record_safety: dict[int, list[str]] = defaultdict(list)

    for i, reasons in result.transaction_structural_violations.items():
        tx_safety[i].extend(reasons)

    for i, reasons in result.record_structural_violations.items():
        record_safety[i].extend(reasons)

    for i, failures in enumerate(failures_by_case):
        for failure in failures:
            if failure.field in TRANSACTION_FIELDS:
                tx_safety[i].append(
                    "development_gold_mismatch:" + failure.field
                )
            if failure.field == "record_claims":
                record_safety[i].append(
                    "development_gold_mismatch:record_claims"
                )

    # Dedupe reasons.
    tx_safety = {
        i: list(dict.fromkeys(reasons))
        for i, reasons in tx_safety.items()
        if reasons
    }
    record_safety = {
        i: list(dict.fromkeys(reasons))
        for i, reasons in record_safety.items()
        if reasons
    }

    assembly_violation_count = sum(
        len(v) for v in result.assembly_violations.values()
    )
    tx_violation_count = sum(len(v) for v in tx_safety.values())
    record_violation_count = sum(
        len(v) for v in record_safety.values()
    )

    source_after = source_snapshot()
    source_unchanged = source_after == source_before

    print()
    print("========== FULL EXACT-FRAME DEVELOPMENT RESULT ==========")
    print("cases=", total)
    print("exact_frames=", exact, "/", total)
    print("full_exact_frame_accuracy=", round(pct(exact, total), 4))
    print(
        "overall_field_accuracy=",
        round(
            pct(
                total_field_pass,
                total_field_pass + total_field_fail,
            ),
            4,
        ),
    )

    print()
    print("========== PER-FIELD ACCURACY ==========")
    for field in FIELDS:
        p = field_pass[field]
        f = field_fail[field]
        print(
            field,
            f"pass={p}",
            f"fail={f}",
            f"accuracy={pct(p, p + f):.4f}",
        )

    print()
    print("========== CRITICAL DEVELOPMENT GATE ==========")
    print("critical_cases=", len(critical_indices))
    print(
        "critical_exact_frame_accuracy=",
        round(pct(critical_exact, len(critical_indices)), 4),
    )
    print(
        "critical_all_field_accuracy=",
        round(
            pct(critical_field_pass, critical_field_total),
            4,
        ),
    )
    print(
        "critical_safety_field_accuracy=",
        round(
            pct(critical_safety_pass, critical_safety_total),
            4,
        ),
    )

    print()
    print("========== SCHEDULING FIELDS ==========")
    for field in SCHEDULING_FIELDS:
        p = field_pass[field]
        f = field_fail[field]
        print(
            field,
            f"accuracy={pct(p, p + f):.4f}",
            f"fail={f}",
        )
    print(
        "scheduling_joint_accuracy=",
        round(pct(sched_pass, sched_total), 4),
    )

    print()
    print("========== AMBIGUITY / REFERENCE ==========")
    for field in REFERENCE_AMBIGUITY_FIELDS:
        p = field_pass[field]
        f = field_fail[field]
        print(
            field,
            f"accuracy={pct(p, p + f):.4f}",
            f"fail={f}",
        )
    print(
        "ambiguity_reference_joint_accuracy=",
        round(pct(refamb_pass, refamb_total), 4),
    )

    print()
    print("========== TRANSACTION SAFETY ==========")
    print(
        "transaction_field_joint_accuracy=",
        round(pct(tx_joint_pass, tx_joint_total), 4),
    )
    print("transaction_safety_violations=", tx_violation_count)
    if tx_safety:
        for i in sorted(tx_safety):
            print(
                " ",
                cases[i].case_id,
                "=>",
                "; ".join(tx_safety[i]),
            )
    else:
        print(" transaction_safety=PASS_ZERO_VIOLATIONS")

    print()
    print("========== RECORD-CLAIM SAFETY ==========")
    print(
        "record_claim_field_accuracy=",
        round(pct(record_joint_pass, record_joint_total), 4),
    )
    print("record_claim_safety_violations=", record_violation_count)
    if record_safety:
        for i in sorted(record_safety):
            print(
                " ",
                cases[i].case_id,
                "=>",
                "; ".join(record_safety[i]),
            )
    else:
        print(" record_claim_safety=PASS_ZERO_VIOLATIONS")

    print()
    print("========== ASSEMBLY / INVARIANT AUDIT ==========")
    print("assembly_violations=", assembly_violation_count)
    print("bare_yes_rule_applied_cases=", len(result.bare_yes_applied))
    if result.bare_yes_applied:
        print(
            "bare_yes_case_ids=",
            [
                cases[i].case_id
                for i in sorted(result.bare_yes_applied)
            ],
        )
    if result.assembly_violations:
        for i in sorted(result.assembly_violations):
            print(
                " ",
                cases[i].case_id,
                "=>",
                "; ".join(result.assembly_violations[i]),
            )
    else:
        print(" semanticframe_invariants=PASS")

    print("source_tree_python_unchanged=", "YES" if source_unchanged else "NO")

    print()
    print("========== ALL FAILING CASES / FIELD-LEVEL DIFFS ==========")
    failing_case_count = 0

    for i, (case, frame, failures) in enumerate(
        zip(cases, result.frames, failures_by_case)
    ):
        assembly = result.assembly_violations.get(i, [])

        if not failures and not assembly:
            continue

        failing_case_count += 1
        print()
        print(
            case.case_id,
            "FAIL",
            f"category={case.category}",
            f"tags={tuple(case.tags)!r}",
        )
        print("  utterance=", repr(case.utterance))
        print("  context=", repr(list(case.context)))

        for failure in failures:
            print(
                " ",
                failure.field,
                "expected=",
                repr(failure.expected),
                "actual=",
                repr(failure.actual),
            )

        for issue in assembly:
            print("  ASSEMBLY_VIOLATION", issue)

        # Predicted-only diagnostics. Gold is not fed back into any component.
        print(
            "  predicted_route=",
            result.gate_labels[i],
            "dense_pair=",
            result.dense_pairs[i],
            "reference_kind=",
            result.reference_kinds[i],
            "ambiguity_detail=",
            result.ambiguity_details[i],
        )

    print()
    print("failing_cases=", failing_case_count)

    exact_accuracy = pct(exact, total)
    overall_field_accuracy = pct(
        total_field_pass,
        total_field_pass + total_field_fail,
    )
    critical_exact_accuracy = pct(
        critical_exact,
        len(critical_indices),
    )
    critical_safety_accuracy = pct(
        critical_safety_pass,
        critical_safety_total,
    )
    refamb_accuracy = pct(refamb_pass, refamb_total)

    strong = (
        exact_accuracy >= 0.98
        and overall_field_accuracy >= 0.99
        and critical_exact_accuracy == 1.0
        and critical_safety_accuracy == 1.0
        and refamb_accuracy >= 0.98
        and tx_violation_count == 0
        and record_violation_count == 0
        and assembly_violation_count == 0
        and source_unchanged
    )

    print()
    print("========== DEVELOPMENT DECISION ==========")
    print("development_corpus_not_final_holdout=YES")
    print("level2_frozen=NO")
    print("level3_started=NO")

    if strong:
        print("FULL_LEVEL2_DEVELOPMENT_ASSEMBLY=STRONG")
        print(
            "NEXT_ACTION="
            "RUN_EXISTING_PLANNER_STAGELAB_SAFETY_TELEPHONY_REGRESSIONS_"
            "THEN_CREATE_FRESH_UNSEEN_LEVEL2_HOLDOUT"
        )
    else:
        print("FULL_LEVEL2_DEVELOPMENT_ASSEMBLY=NOT_STRONG")
        print(
            "NEXT_ACTION="
            "AUDIT_ARCHITECTURAL_INTERACTION_ERRORS_BEFORE_RETRAINING_SPECIALISTS"
        )

    return 0 if strong else 2


def main() -> int:
    print("========== FULL READ-ONLY LEVEL 2 SEMANTICFRAME EVALUATOR ==========")
    print("telephony=DISABLED")
    print("training=NO")
    print("runtime_wiring_modified=NO")
    print("v0_17_modified=NO")
    print("gold_runtime_inputs=NO")
    print("development_only=YES")
    print("model_downloads=OFFLINE_ONLY")
    print("learned_phase8d_operation_checkpoint_used=NO")
    print("transaction_operation_decoder=DETERMINISTIC_PREDICATE_NORMALIZER")

    checkpoints = validate_environment()
    print("v0_17_marker_detected=YES")

    print_checkpoint_manifest()

    print()
    print("========== FROZEN / PROVISIONAL ONTOLOGY ==========")
    print("phase8b1_facts=", EXPECTED_FACTS)
    print("phase8c_entities=", EXPECTED_RECORD_ENTITIES)
    print("phase8c_states=", EXPECTED_RECORD_STATES)
    print(
        "phase8a3_valid_pairs=",
        tuple(tuple(x) for x in checkpoints["8a"]["valid_pairs"]),
    )

    cases = list(load_semanticlab_cases())
    if len(cases) != 133:
        raise SystemExit(
            f"Expected exactly 133 development cases, got {len(cases)}."
        )

    # Snapshot source before any dynamic experiment imports/model inference.
    source_before = source_snapshot()

    # Critical anti-leak boundary: component inference receives objects with
    # ONLY context + utterance. No case ID, category, tags, or expected frame.
    runtime = [
        RuntimeTurn(
            context=tuple(case.context),
            utterance=str(case.utterance),
        )
        for case in cases
    ]

    assert set(RuntimeTurn.__dataclass_fields__) == {
        "context",
        "utterance",
    }

    print()
    print("========== RUNTIME INPUT BOUNDARY ==========")
    print("runtime_model_input_fields=context,utterance")
    print("case_id_visible_to_models=NO")
    print("category_visible_to_models=NO")
    print("tags_visible_to_models=NO")
    print("expected_gold_visible_to_models=NO")
    print("cases=", len(runtime))

    # No access to case.expected occurs inside assemble_level2().
    result = assemble_level2(runtime, checkpoints)

    print()
    print("inference_complete=YES")
    print("gold_scoring_begins_only_now=YES")

    return score_and_report(
        cases,
        result,
        source_before,
    )


if __name__ == "__main__":
    raise SystemExit(main())
