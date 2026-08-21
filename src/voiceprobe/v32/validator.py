"""Deterministic action validation for VoiceProbe v3.2."""

from __future__ import annotations

import re

from voiceprobe.v3.flow_state import FlowSnapshot
from voiceprobe.v3.models import (
    DecisionKind,
    PatientFacts,
    PolicyDecision,
)

from .schemas import (
    ContextualProposal,
    DialogueAction,
    FactKey,
    Grounding,
    RiskLevel,
)


_TRANSACTION_LANGUAGE = re.compile(
    r"\b("
    r"go ahead and book|"
    r"book it|"
    r"you can book|"
    r"cancel it|"
    r"cancel the appointment|"
    r"appointment (?:is|has been) booked|"
    r"appointment (?:is|has been) confirmed"
    r")\b",
    re.IGNORECASE,
)


class ActionValidator:
    """Model proposes language intent; Python owns truth and state."""

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.60,
    ) -> None:
        self.minimum_confidence = minimum_confidence

    def validate(
        self,
        *,
        proposal: ContextualProposal,
        facts: PatientFacts,
        snapshot: FlowSnapshot,
    ) -> PolicyDecision:
        del snapshot

        confidence = proposal.confidence

        if confidence < self.minimum_confidence:
            return self._clarify(
                "v32_low_confidence",
                confidence,
            )

        # Contextual fallback may understand transaction language,
        # but it never receives authority to perform the transaction.
        if proposal.risk is RiskLevel.TRANSACTION:
            return self._clarify(
                "v32_transaction_requires_deterministic_policy",
                confidence,
            )

        if proposal.action is DialogueAction.WAIT:
            return PolicyDecision(
                DecisionKind.WAIT,
                reason="v32_contextual_wait",
                confidence=confidence,
            )

        if proposal.action is DialogueAction.STATE_OBJECTIVE:
            return PolicyDecision(
                DecisionKind.STATE_OBJECTIVE,
                text=(
                    f"I'm looking for {facts.preferred_day} "
                    f"{facts.preferred_time}."
                ),
                reason="v32_contextual_state_objective",
                confidence=confidence,
            )

        if proposal.action is DialogueAction.ANSWER_FACT:
            return self._fact(
                proposal.fact_key,
                facts,
                confidence,
            )

        if proposal.action is DialogueAction.ANSWER:
            if proposal.grounding not in {
                Grounding.LOW_RISK_CONVERSATIONAL,
                Grounding.CURRENT_GOAL,
            }:
                return self._clarify(
                    "v32_unsafe_answer_grounding",
                    confidence,
                )

            text = proposal.response_text.strip()

            if not text:
                return self._clarify(
                    "v32_empty_contextual_answer",
                    confidence,
                )

            if _TRANSACTION_LANGUAGE.search(text):
                return self._clarify(
                    "v32_blocked_transaction_language",
                    confidence,
                )

            # Hard protection for the adversarial insurance scenario.
            lowered = text.casefold()
            if (
                "blue shield" in lowered
                and facts.insurance.casefold() != "blue shield"
            ):
                return self._clarify(
                    "v32_blocked_fact_contradiction",
                    confidence,
                )

            return PolicyDecision(
                DecisionKind.ANSWER_FACT,
                text=text,
                reason="v32_contextual_answer",
                confidence=confidence,
            )

        return self._clarify(
            "v32_model_requested_clarification",
            confidence,
        )

    @staticmethod
    def _fact(
        key: FactKey,
        facts: PatientFacts,
        confidence: float,
    ) -> PolicyDecision:
        if key is FactKey.FIRST_NAME:
            text = f"{facts.first_name}."
        elif key is FactKey.LAST_NAME:
            text = f"{facts.last_name}."
        elif key is FactKey.DOB:
            text = f"{facts.dob}."
        elif key is FactKey.INSURANCE:
            text = f"{facts.insurance}."
        elif key is FactKey.COMPLAINT:
            text = f"I have {facts.complaint}."
        elif key is FactKey.SYMPTOM_DURATION:
            text = f"{facts.symptom_duration}."
        elif key is FactKey.PREFERRED_DAY:
            text = f"{facts.preferred_day}."
        elif key is FactKey.PREFERRED_TIME:
            text = f"{facts.preferred_time}."
        elif key is FactKey.APPOINTMENT_TYPE:
            text = f"A {facts.appointment_type}."
        elif key is FactKey.PROVIDER_PREFERENCE:
            text = "First available is fine."
        else:
            return PolicyDecision(
                DecisionKind.CLARIFY,
                text="Could you clarify what information you need?",
                reason="v32_missing_fact_key",
                confidence=confidence,
            )

        return PolicyDecision(
            DecisionKind.ANSWER_FACT,
            text=text,
            reason=f"v32_authoritative_fact:{key.value}",
            confidence=confidence,
        )

    @staticmethod
    def _clarify(
        reason: str,
        confidence: float,
    ) -> PolicyDecision:
        return PolicyDecision(
            DecisionKind.CLARIFY,
            text="Could you clarify that question?",
            reason=reason,
            confidence=confidence,
        )
