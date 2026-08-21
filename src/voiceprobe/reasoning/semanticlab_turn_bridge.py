"""Hybrid SemanticLab-v2 semantic perception for Reasoning Core v2.

The bridge is intentionally conservative:

* SemanticLab receives only committed remote-agent speech plus recent
  remote-agent history.
* Patient truth, scenario preferences, booking progress and telephony state are
  never passed into SemanticLab.
* Meanings that SemanticLab can represent losslessly are converted into the
  existing Reasoning-v2 ``TurnFrame`` contract.
* Meanings that would lose production-relevant structure fail closed to
  ``RequestedAction.CLARIFY`` while SemanticLab is enabled. The SemanticLab
  production path never wakes a second semantic LLM.
* OOS or unresolved ambiguity fails closed as ``RequestedAction.CLARIFY`` rather
  than asking the fallback model to guess.

The bridge therefore changes semantic perception only. Reasoning-v2 remains
source-of-truth for patient facts, planning, validation, booking progress,
verbalization and the telephony END_CONVERSATION safety boundary.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from time import perf_counter
from typing import Any

import httpx

from voiceprobe.reasoning.semantic_reasoner import StructuredTurnReasoner
from voiceprobe.reasoning.turn_frame import (
    RequestedAction,
    RequestedFact,
    SlotOption,
    SpeechAct,
    TurnFrame,
    WorkflowKind,
)
from voiceprobe.v33.semantic_runtime_v2 import SemanticLabV2Reasoner


_SEMANTICLAB_REASONING_V2_ENV = "VOICEPROBE_REASONING_V2_SEMANTICLAB"

_SUPPORTED_CONFIDENCE = 0.95
_SAFE_CLARIFICATION_CONFIDENCE = 1.0
_EDGE_MODEL_ENV = "VOICEPROBE_REASONING_V2_EDGE_MODEL"
_DEFAULT_EDGE_MODEL = "qwen3.5:0.8b"

_SLOT_RE = re.compile(
    r"^\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    r"\s+at\s+"
    r"((?:1[0-2]|0?[1-9])(?::[0-5][0-9])?\s*(?:AM|PM|a\.m\.|p\.m\.))\s*$",
    re.IGNORECASE,
)

_REQUESTED_FACT_ALIASES: dict[str, RequestedFact] = {
    "first_name": RequestedFact.FIRST_NAME,
    "last_name": RequestedFact.LAST_NAME,
    "full_name": RequestedFact.FULL_NAME,
    "dob": RequestedFact.DATE_OF_BIRTH,
    "date_of_birth": RequestedFact.DATE_OF_BIRTH,
    "insurance": RequestedFact.INSURANCE,
    "complaint": RequestedFact.COMPLAINT,
    "duration": RequestedFact.SYMPTOM_DURATION,
    "symptom_duration": RequestedFact.SYMPTOM_DURATION,
    "preferred_day": RequestedFact.PREFERRED_DAY,
    "preferred_time": RequestedFact.PREFERRED_TIME,
    "provider": RequestedFact.PROVIDER_PREFERENCE,
    "provider_preference": RequestedFact.PROVIDER_PREFERENCE,
    "visit_type": RequestedFact.APPOINTMENT_TYPE,
    "appointment_type": RequestedFact.APPOINTMENT_TYPE,
    "patient_status": RequestedFact.PATIENT_STATUS,
    "visited_before": RequestedFact.VISITED_BEFORE,
    "phone_number": RequestedFact.PHONE_NUMBER,
    "email": RequestedFact.EMAIL,
    "address": RequestedFact.ADDRESS,
}


def reasoning_v2_edge_model_from_environment(default_model: str) -> str:
    """Select the low-memory model used only for unresolved edge decisions.

    When the SemanticLab bridge is OFF, preserve the caller-supplied model
    exactly. When it is ON, default edge-only Ollama work to qwen3.5:0.8b,
    which is present on the validated A8/3060 setup and keeps fallback memory
    small. An explicit VOICEPROBE_REASONING_V2_EDGE_MODEL override is accepted
    when non-blank; production can point that model at the remote 3060.
    """

    if not semanticlab_reasoning_v2_enabled_from_environment():
        return str(default_model)

    value = os.environ.get(_EDGE_MODEL_ENV)
    if value is None or not value.strip():
        return _DEFAULT_EDGE_MODEL
    return " ".join(value.split())


def semanticlab_reasoning_v2_enabled_from_environment() -> bool:
    """Read the Reasoning-v2 SemanticLab feature flag.

    The convention intentionally matches ``VOICEPROBE_REASONING_V2``: unset,
    empty or exact ``0`` means OFF; exact ``1`` means ON; every other value is
    rejected so a typo cannot silently alter an authorized call path.
    """

    value = os.environ.get(_SEMANTICLAB_REASONING_V2_ENV)

    if value is None or value == "" or value == "0":
        return False

    if value == "1":
        return True

    raise ValueError(
        f"{_SEMANTICLAB_REASONING_V2_ENV} must be exactly '0' or '1'."
    )


def _normalize_slot_time(value: str) -> str:
    text = value.upper().replace(".", "")
    text = re.sub(r"\s+", " ", text).strip()
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM)", text)
    if match is None:
        return value.strip()

    hour = int(match.group(1))
    minute = match.group(2)
    suffix = match.group(3)
    return f"{hour}:{minute} {suffix}" if minute else f"{hour} {suffix}"


def _slot_from_semanticlab(value: str) -> SlotOption | None:
    """Convert only the canonical concrete-slot surface SemanticLab emits."""

    match = _SLOT_RE.fullmatch(str(value))
    if match is None:
        return None

    day = match.group(1).capitalize()
    time = _normalize_slot_time(match.group(2))
    return SlotOption(day=day, time=time)


def _canonical_slots_in_text(value: str) -> tuple[SlotOption, ...]:
    """Extract canonical weekday+clock slots from one current remote turn.

    This is value extraction only. It does not decide whether text is an offer
    or a booking confirmation; those meanings must already come from the frozen
    SemanticLab frame.
    """

    pattern = re.compile(
        r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
        r"\b(?:\s+at)?\s+"
        r"((?:1[0-2]|0?[1-9])(?::[0-5][0-9])?\s*(?:AM|PM|a\.m\.|p\.m\.))\b",
        re.IGNORECASE,
    )
    out: list[SlotOption] = []
    seen: set[tuple[str | None, str | None]] = set()
    for match in pattern.finditer(str(value)):
        slot = SlotOption(
            day=match.group(1).capitalize(),
            time=_normalize_slot_time(match.group(2)),
        )
        key = (slot.day, slot.time)
        if key not in seen:
            seen.add(key)
            out.append(slot)
    return tuple(out)


def _workflow_for_topic(topic: str) -> WorkflowKind:
    if topic == "identity":
        return WorkflowKind.IDENTITY
    if topic == "profile":
        return WorkflowKind.PROFILE_SETUP
    if topic in {"patient_fact", "reschedule_reason"}:
        return WorkflowKind.PATIENT_INTAKE
    if topic in {
        "open_intent",
        "appointment_state",
        "visit_type",
        "provider",
        "availability",
        "transaction",
        "presence",
    }:
        return WorkflowKind.SCHEDULING
    return WorkflowKind.UNKNOWN


def _speech_act(value: str) -> SpeechAct:
    mapping = {
        "greeting": SpeechAct.GREETING,
        "question": SpeechAct.QUESTION,
        "request": SpeechAct.REQUEST,
        "offer": SpeechAct.OFFER,
        "statement": SpeechAct.INFORMATION,
        "confirmation": SpeechAct.CONFIRMATION,
        "acknowledgement": SpeechAct.INFORMATION,
        "presence_check": SpeechAct.QUESTION,
        "clarification": SpeechAct.QUESTION,
        "other": SpeechAct.OTHER,
    }
    return mapping.get(value, SpeechAct.OTHER)


def _turn_frame(
    *,
    frame: Any,
    requested_action: RequestedAction,
    response_required: bool,
    requested_facts: list[RequestedFact] | None = None,
    appointment_options: list[SlotOption] | None = None,
    confirmed_appointment: SlotOption | None = None,
    booking_confirmed: bool = False,
    confidence: float = _SUPPORTED_CONFIDENCE,
) -> TurnFrame:
    return TurnFrame(
        speech_act=_speech_act(frame.speech_act.value),
        workflow=_workflow_for_topic(frame.topic.value),
        requested_action=requested_action,
        response_required=response_required,
        requested_facts=requested_facts or [],
        appointment_options=appointment_options or [],
        confirmed_appointment=confirmed_appointment,
        booking_confirmed=booking_confirmed,
        confidence=confidence,
    )


class SemanticLabHybridTurnReasoner:
    """Drop-in ``StructuredTurnReasoner`` replacement with safe fallback."""

    def __init__(
        self,
        *,
        model: str,
        url: str,
        client: httpx.Client | None = None,
        timeout_seconds: float = 20.0,
        semanticlab: Any | None = None,
        fallback: Any | None = None,
    ) -> None:
        self._semanticlab = (
            semanticlab if semanticlab is not None else SemanticLabV2Reasoner.shared()
        )

        # Prewarm during session construction. In the Asterisk path the session
        # is created before call origination, so the remote party never pays the
        # one-time SemanticLab model-construction cost on its first turn.
        self.startup_warmup_ms = 0.0
        warmup_sync = getattr(self._semanticlab, "warmup_sync", None)
        if callable(warmup_sync):
            warm_started = perf_counter()
            warmup_sync()
            self.startup_warmup_ms = (perf_counter() - warm_started) * 1000.0

        # A fallback object may be injected by unit tests, but production
        # SemanticLab mode deliberately does not instantiate a second semantic
        # LLM. Unsupported meanings fail closed to CLARIFY instead.
        self._fallback = fallback
        self.last_route = "uninitialized"
        self.last_fallback_reason = ""
        self.last_requested_fact_stability = "not_checked"
        self.last_transaction_stability = "not_checked"
        self.last_semantic_summary = ""

    def close(self) -> None:
        """Close only the owned/fallback HTTP semantic reasoner.

        ``SemanticLabV2Reasoner.shared()`` is process-shared persistent model
        state and intentionally remains alive for subsequent sessions.
        """

        close_fallback = getattr(self._fallback, "close", None)
        if callable(close_fallback):
            close_fallback()

    def _fallback_interpret(
        self,
        *,
        reason: str,
        agent_turn: str,
        recent_history: Sequence[str],
    ) -> TurnFrame:
        self.last_fallback_reason = reason

        # Tests may inject the legacy reasoner explicitly to verify compatibility,
        # but production SemanticLab mode must not wake another semantic LLM.
        if self._fallback is not None:
            self.last_route = "structured_fallback_injected"
            return self._fallback.interpret(
                agent_turn=agent_turn,
                recent_history=recent_history,
            )

        self.last_route = "semanticlab_fail_closed"
        return TurnFrame(
            speech_act=SpeechAct.OTHER,
            workflow=WorkflowKind.UNKNOWN,
            requested_action=RequestedAction.CLARIFY,
            response_required=True,
            confidence=_SAFE_CLARIFICATION_CONFIDENCE,
        )

    def interpret(
        self,
        *,
        agent_turn: str,
        recent_history: Sequence[str] = (),
    ) -> TurnFrame:
        normalized_turn = " ".join(agent_turn.split())
        if not normalized_turn:
            raise ValueError("agent_turn cannot be blank.")

        normalized_history = tuple(
            " ".join(str(item).split())
            for item in recent_history[-4:]
            if str(item).strip()
        )

        frame, oos_active = self._semanticlab.interpret_frame_sync(
            remote_turn=normalized_turn,
            recent_history=normalized_history,
        )

        requested_fact_override: str | None = None
        self.last_requested_fact_stability = "not_checked"
        self.last_transaction_stability = "not_checked"

        # A single isolated current-turn inference may be reused by the two
        # stability arbiters below. This remains the same frozen SemanticLab
        # stack; no lexical transaction classifier or second LLM is introduced.
        isolated_frame: Any | None = None
        isolated_oos: bool | None = None

        def isolated_current_turn() -> tuple[Any, bool]:
            nonlocal isolated_frame, isolated_oos
            if isolated_frame is None or isolated_oos is None:
                isolated_frame, isolated_oos = self._semanticlab.interpret_frame_sync(
                    remote_turn=normalized_turn,
                    recent_history=(),
                )
            return isolated_frame, bool(isolated_oos)

        # Transaction context-stability arbitration.
        #
        # The frozen transaction corpus already models completed BOOK/RESCHEDULE
        # confirmations. Conversation history can nevertheless suppress that
        # current-turn meaning. If the contextual frame contains exactly one
        # concrete current-turn slot but does not establish confirmation, allow
        # an isolated semantic pass to override it ONLY when that pass safely
        # establishes BOOK/RESCHEDULE + CONFIRMED. This can add a confirmation;
        # it cannot invent permission, authorization, cancellation or a slot.
        current_slots = _canonical_slots_in_text(normalized_turn)
        if (
            normalized_history
            and not oos_active
            and not frame.has_unresolved_ambiguity
            and len(current_slots) == 1
            and frame.transaction_signal.value != "confirmed"
        ):
            isolated_tx, isolated_tx_oos = isolated_current_turn()
            if (
                not isolated_tx_oos
                and not isolated_tx.has_unresolved_ambiguity
                and isolated_tx.transaction_operation.value in {"book", "reschedule"}
                and isolated_tx.transaction_signal.value == "confirmed"
            ):
                before = (
                    f"{frame.transaction_operation.value}/"
                    f"{frame.transaction_signal.value}"
                )
                after = (
                    f"{isolated_tx.transaction_operation.value}/"
                    f"{isolated_tx.transaction_signal.value}"
                )
                frame = isolated_tx
                oos_active = False
                self.last_transaction_stability = (
                    f"isolated_confirmed_override:{before}->{after}"
                )
            else:
                self.last_transaction_stability = "contextual_preserved"

        # Requested facts should be grounded in the latest remote turn whenever
        # that turn is independently interpretable. Context is still preserved
        # for genuinely elliptical turns. A second semantic-only pass is made
        # only when the contextual frame already predicts a requested fact.
        contextual_fact = str(frame.requested_fact).strip()
        if contextual_fact and normalized_history and not oos_active:
            isolated, fact_isolated_oos = isolated_current_turn()
            isolated_fact = str(isolated.requested_fact).strip()

            if (
                not fact_isolated_oos
                and not isolated.has_unresolved_ambiguity
                and isolated_fact
                and isolated_fact != contextual_fact
            ):
                requested_fact_override = isolated_fact
                self.last_requested_fact_stability = (
                    f"isolated_override:{contextual_fact}->{isolated_fact}"
                )
            else:
                self.last_requested_fact_stability = "contextual_stable"

        # Report the semantic frame actually used after stability arbitration.
        self.last_semantic_summary = (
            f"act={frame.speech_act.value};topic={frame.topic.value};"
            f"requested_fact={str(frame.requested_fact).strip() or 'none'};"
            f"transaction={frame.transaction_operation.value}/"
            f"{frame.transaction_signal.value};"
            f"records={','.join(x.value for x in frame.record_claims) or 'none'};"
            f"reference={frame.reference.value};"
            f"selected={str(frame.selected_option).strip() or 'none'}"
        )

        if oos_active or frame.has_unresolved_ambiguity:
            self.last_route = "semanticlab_safe_clarification"
            self.last_fallback_reason = ""
            return TurnFrame(
                speech_act=SpeechAct.OTHER,
                workflow=WorkflowKind.UNKNOWN,
                requested_action=RequestedAction.CLARIFY,
                response_required=True,
                confidence=_SAFE_CLARIFICATION_CONFIDENCE,
            )

        mapped = self._map_lossless(
            frame,
            requested_fact_override=requested_fact_override,
        )
        if mapped is not None:
            self.last_route = "semanticlab_native"
            self.last_fallback_reason = ""
            return mapped

        return self._fallback_interpret(
            reason=self._fallback_reason(frame),
            agent_turn=normalized_turn,
            recent_history=normalized_history,
        )

    @staticmethod
    def _fallback_reason(frame: Any) -> str:
        if frame.record_claims:
            return "record_claims_not_representable_in_turnframe"
        if frame.failed_constraints or frame.proposed_changes or frame.retained_constraints:
            return "constraint_axis_semantics_require_structured_fallback"
        if frame.reference.value != "none":
            return "typed_reference_requires_structured_fallback"
        if frame.selected_option:
            return "selected_option_requires_contextual_fallback"
        if frame.transaction_operation.value == "create_profile":
            return "profile_workflow_requirement_not_encoded"
        if frame.transaction_signal.value not in {"none", "permission_request", "confirmed"}:
            return "transaction_signal_not_losslessly_mapped"
        if frame.topic.value in {"identity", "patient_fact", "profile", "capability"}:
            return "remote_assertion_or_workflow_detail_not_representable"
        if frame.speech_act.value == "rejection":
            return "rejection_semantics_not_losslessly_mapped"
        if frame.requires_response:
            return "response_required_without_supported_requested_action"
        return "unsupported_semanticlab_frame"

    @staticmethod
    def _map_lossless(
        frame: Any,
        *,
        requested_fact_override: str | None = None,
    ) -> TurnFrame | None:
        operation = frame.transaction_operation.value
        signal = frame.transaction_signal.value

        # CONFIRMED BOOKING PRECEDENCE
        # ----------------------------
        # A transaction confirmation may also carry a redundant record claim
        # such as appointment_exists or a typed reference to the prior option.
        # Those fields must not suppress explicit booking evidence. When the
        # frozen SemanticLab frame has already established BOOK/RESCHEDULE +
        # CONFIRMED, require exactly one concrete current-turn slot (preferred),
        # or a canonical selected_option, and map it before generic information-
        # loss guards. This is semantic precedence, not phrase classification.
        if signal == "confirmed":
            if operation not in {"book", "reschedule"}:
                return None

            current_slots = _canonical_slots_in_text(frame.raw_text)
            confirmed = current_slots[0] if len(current_slots) == 1 else None
            if confirmed is None:
                confirmed = _slot_from_semanticlab(frame.selected_option)
            if confirmed is None:
                return None

            return _turn_frame(
                frame=frame,
                requested_action=RequestedAction.NONE,
                response_required=False,
                confirmed_appointment=confirmed,
                booking_confirmed=True,
            )

        # Reasoning-v2 has no native fields for SemanticLab's scheduling-axis
        # failure/change/retention structure. Do not silently erase it.
        if frame.failed_constraints or frame.proposed_changes or frame.retained_constraints:
            return None

        # Record claims can affect grounding/workflow state but do not have an
        # equivalent TurnFrame field. (Confirmed booking is handled above.)
        if frame.record_claims:
            return None

        # Profile creation needs WorkflowRequirement (optional/required/unknown),
        # which SemanticFrame does not encode. Preserve the current reasoner.
        if operation == "create_profile":
            return None

        if signal == "permission_request":
            if frame.reference.value != "none" or frame.selected_option:
                return None
            return _turn_frame(
                frame=frame,
                requested_action=RequestedAction.GRANT_PERMISSION,
                response_required=True,
            )

        if signal not in {"none"}:
            return None

        requested_fact = (
            str(requested_fact_override).strip()
            if requested_fact_override is not None
            else str(frame.requested_fact).strip()
        )
        if requested_fact:
            fact = _REQUESTED_FACT_ALIASES.get(requested_fact)
            if fact is None:
                return None
            return _turn_frame(
                frame=frame,
                requested_action=RequestedAction.ANSWER_FACT,
                response_required=True,
                requested_facts=[fact],
            )

        topic = frame.topic.value

        if topic == "visit_type" and frame.requires_response:
            return _turn_frame(
                frame=frame,
                requested_action=RequestedAction.ANSWER_FACT,
                response_required=True,
                requested_facts=[RequestedFact.APPOINTMENT_TYPE],
            )

        if topic == "provider" and frame.requires_response:
            return _turn_frame(
                frame=frame,
                requested_action=RequestedAction.ANSWER_FACT,
                response_required=True,
                requested_facts=[RequestedFact.PROVIDER_PREFERENCE],
            )

        # V3.3's proven strategic policy responds to PRESENCE_CHECK by resuming
        # the workflow and restating the goal. STATE_OBJECTIVE is the closest
        # lossless Reasoning-v2 action contract and also preserves the known
        # production opening-turn behavior without phrase matching.
        if (
            topic == "open_intent"
            or frame.speech_act.value == "presence_check"
        ) and frame.requires_response:
            return _turn_frame(
                frame=frame,
                requested_action=RequestedAction.STATE_OBJECTIVE,
                response_required=True,
            )

        if frame.speech_act.value == "clarification" and frame.requires_response:
            return _turn_frame(
                frame=frame,
                requested_action=RequestedAction.CLARIFY,
                response_required=True,
            )

        if frame.offered_options:
            options: list[SlotOption] = []
            for value in frame.offered_options:
                slot = _slot_from_semanticlab(value)
                if slot is None:
                    return None
                options.append(slot)

            return _turn_frame(
                frame=frame,
                requested_action=(
                    RequestedAction.CHOOSE_OPTION
                    if frame.requires_response
                    else RequestedAction.NONE
                ),
                response_required=bool(frame.requires_response),
                appointment_options=options,
            )

        # Source-grounded remote assertions/corrections need TurnFrame.stated_facts
        # values. SemanticFrame intentionally does not retain those values.
        if topic in {"identity", "patient_fact", "profile", "capability"}:
            return None

        # Any non-trivial selected/reference state is kept on the existing
        # semantic path unless handled by the confirmed-booking branch above.
        if frame.reference.value != "none" or frame.selected_option:
            return None

        if frame.speech_act.value == "rejection":
            return None

        if frame.requires_response:
            return None

        # Passive informational/acknowledgement/status-like turns are faithfully
        # represented by no immediate requested action. Reasoning-v2 will WAIT.
        return _turn_frame(
            frame=frame,
            requested_action=RequestedAction.NONE,
            response_required=False,
        )


def build_reasoning_v2_semantic_interpreter(
    *,
    model: str,
    url: str,
    client: httpx.Client | None = None,
    timeout_seconds: float = 20.0,
) -> Any:
    """Build the production semantic component with default behavior unchanged."""

    if semanticlab_reasoning_v2_enabled_from_environment():
        return SemanticLabHybridTurnReasoner(
            model=reasoning_v2_edge_model_from_environment(model),
            url=url,
            client=client,
            timeout_seconds=timeout_seconds,
        )

    return StructuredTurnReasoner(
        model=model,
        url=url,
        client=client,
        timeout_seconds=timeout_seconds,
    )
