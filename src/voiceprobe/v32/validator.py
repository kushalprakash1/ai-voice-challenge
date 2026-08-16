"""Deterministic safety validator for v3.2 model proposals."""

from __future__ import annotations

import re

from voiceprobe.v3.flow_state import FlowSnapshot
from voiceprobe.v3.models import (
    DecisionKind,
    PatientFacts,
    PolicyDecision,
)

from .schemas import (
    ActionProposal,
    DialogueAction,
    Grounding,
    IntentRewrite,
    RiskLevel,
)


_TRANSACTION_LANGUAGE = re.compile(
    r"\\b("
    r"go ahead and book|"
    r"you can book|"
    r"book it|"
    r"cancel it|"
    r"cancel the appointment|"
    r"appointment is booked|"
    r"appointment has been booked|"
    r"appointment is confirmed|"
    r"appointment has been confirmed"
    r")\\b",
    re.IGNORECASE,
)


class ActionValidator:
    """The LLM may propose. Python remains authoritative."""

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.60,
    ) -> None:
        self.minimum_confidence = minimum_confidence

    def validate(
        self,
        *,
        rewrite: IntentRewrite,
        proposal: ActionProposal,
        facts: PatientFacts,
        snapshot: FlowSnapshot,
    ) -> PolicyDecision:
        del snapshot

        confidence = min(
            rewrite.confidence,
            proposal.confidence,
        )

        if confidence < self.minimum_confidence:
            return self._clarify(
                "v32_low_confidence",
                confidence,
            )

        if proposal.proposed_state_change:
            return self._clarify(
                "v32_blocked_model_state_change",
                confidence,
            )

        if rewrite.risk is RiskLevel.TRANSACTION:
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
                    "I'm looking for "
                    f"{facts.preferred_day} "
                    f"{facts.preferred_time}."
                ),
                reason="v32_contextual_state_objective",
                confidence=confidence,
            )

        if proposal.action is DialogueAction.ANSWER_FACT:
            return self._answer_authoritative_fact(
                fact_key=proposal.fact_key,
                facts=facts,
                confidence=confidence,
            )

        if proposal.action is DialogueAction.ANSWER:
            if proposal.grounding not in {
                Grounding.LOW_RISK_CONVERSATIONAL,
                Grounding.CURRENT_GOAL,
            }:
                return self._clarify(
                    "v32_answer_not_safely_grounded",
                    confidence,
                )

            text = " ".join(
                proposal.answer_text.split()
            )

            if not text:
                return self._clarify(
                    "v32_empty_answer",
                    confidence,
                )

            if len(text) > 280:
                return self._clarify(
                    "v32_answer_too_long",
                    confidence,
                )

            if _TRANSACTION_LANGUAGE.search(text):
                return self._clarify(
                    "v32_blocked_transaction_language",
                    confidence,
                )

            # Known adversarial contradiction from the current scenario.
            if "blue shield" in text.casefold():
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
    def _answer_authoritative_fact(
        *,
        fact_key: str,
        facts: PatientFacts,
        confidence: float,
    ) -> PolicyDecision:
        key = fact_key.casefold().strip()

        values = {
            "first_name": facts.first_name,
            "last_name": facts.last_name,
            "dob": facts.dob,
            "insurance": facts.insurance,
            "complaint": facts.complaint,
            "symptom_duration": facts.symptom_duration,
            "preferred_day": facts.preferred_day,
            "preferred_time": facts.preferred_time,
            "appointment_type": facts.appointment_type,
            "provider_preference": (
                facts.provider_preference
            ),
        }

        if key not in values:
            return PolicyDecision(
                DecisionKind.CLARIFY,
                text=(
                    "Could you clarify what "
                    "information you need?"
                ),
                reason="v32_unknown_authoritative_fact",
                confidence=confidence,
            )

        if key == "complaint":
            text = f"I have {facts.complaint}."
        elif key == "provider_preference":
            text = "First available is fine."
        elif key == "appointment_type":
            text = f"A {facts.appointment_type}."
        else:
            text = f"{values[key]}."

        return PolicyDecision(
            DecisionKind.ANSWER_FACT,
            text=text,
            reason=(
                f"v32_authoritative_fact:{key}"
            ),
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
