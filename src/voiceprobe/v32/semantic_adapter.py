"""Safe boundary between v3.2 semantics and authoritative v3 state.

The semantic model interprets language.
This module decides what trusted Python data may be spoken.

Transaction routes NEVER become PolicyDecision objects here.
"""

from __future__ import annotations

from dataclasses import dataclass

from voiceprobe.v3.models import (
    DecisionKind,
    PatientFacts,
    PolicyDecision,
)

from .semantic_frame import Focus
from .semantic_policy import (
    RoutedSemanticTurn,
    SemanticRoute,
)


@dataclass(frozen=True, slots=True)
class SemanticResolution:
    route: SemanticRoute
    decision: PolicyDecision | None

    @property
    def delegates_transaction(self) -> bool:
        return (
            self.route is SemanticRoute.TRANSACTION_GATE
            and self.decision is None
        )


def _provider_text(facts: PatientFacts) -> str:
    preference = facts.provider_preference.strip()

    if preference.casefold() == "first available":
        return "First available is fine."

    return f"{preference}."


def resolve_semantic_turn(
    routed: RoutedSemanticTurn,
    *,
    facts: PatientFacts,
) -> SemanticResolution:
    """Convert semantics to a grounded v3 response.

    Safety properties:

    * no model-provided fact values
    * no model-provided response text
    * no booking/cancel/reschedule authorization
    * no slot acceptance
    * no booking confirmation
    """

    route = routed.route

    if route is SemanticRoute.WAIT:
        return SemanticResolution(
            route=route,
            decision=PolicyDecision(
                DecisionKind.WAIT,
                reason="v32_semantic_wait",
            ),
        )

    if route is SemanticRoute.HOLD:
        return SemanticResolution(
            route=route,
            decision=PolicyDecision(
                DecisionKind.HOLD,
                reason="v32_semantic_hold",
            ),
        )

    if route is SemanticRoute.ANSWER_RESCHEDULE_REASON:
        return SemanticResolution(
            route=route,
            decision=PolicyDecision(
                DecisionKind.CONTEXTUAL_ANSWER,
                text=(
                    "That appointment time no longer works for me. "
                    f"I'd like to move it to "
                    f"{facts.preferred_day} {facts.preferred_time}."
                ),
                reason="v32_reschedule_reason",
            ),
        )

    if route is SemanticRoute.TRANSACTION_GATE:
        # Critical invariant:
        # semantic interpretation can recognize transactional language,
        # but cannot turn it into authorization.
        return SemanticResolution(
            route=route,
            decision=None,
        )

    if route is SemanticRoute.ANSWER_FACT:
        focus = routed.fact_focus

        if focus is Focus.COMPLAINT:
            return SemanticResolution(
                route=route,
                decision=PolicyDecision(
                    DecisionKind.ANSWER_COMPLAINT,
                    text=f"I have {facts.complaint}.",
                    reason="complaint_requested",
                ),
            )

        if focus is Focus.PROVIDER_PREFERENCE:
            return SemanticResolution(
                route=route,
                decision=PolicyDecision(
                    DecisionKind.ANSWER_PROVIDER_PREFERENCE,
                    text=_provider_text(facts),
                    reason="provider_preference_requested",
                ),
            )

        if focus is Focus.INSURANCE:
            return SemanticResolution(
                route=route,
                decision=PolicyDecision(
                    DecisionKind.ANSWER_FACT,
                    text=f"{facts.insurance}.",
                    reason="insurance_requested",
                ),
            )

        if focus is Focus.DOB:
            return SemanticResolution(
                route=route,
                decision=PolicyDecision(
                    DecisionKind.ANSWER_FACT,
                    text=f"{facts.dob}.",
                    reason="dob_requested",
                ),
            )

        if focus is Focus.PREFERRED_DAY:
            return SemanticResolution(
                route=route,
                decision=PolicyDecision(
                    DecisionKind.STATE_OBJECTIVE,
                    text=f"{facts.preferred_day}.",
                    reason="preferred_day_requested",
                ),
            )

        if focus is Focus.PREFERRED_TIME:
            return SemanticResolution(
                route=route,
                decision=PolicyDecision(
                    DecisionKind.STATE_OBJECTIVE,
                    text=f"{facts.preferred_time}.",
                    reason="preferred_time_requested",
                ),
            )

        # NAME is intentionally not guessed here because the current
        # semantic ontology does not yet distinguish first/full/last name.
        # Unknown authoritative grounding fails closed.
        return SemanticResolution(
            route=SemanticRoute.UNKNOWN,
            decision=None,
        )

    return SemanticResolution(
        route=SemanticRoute.UNKNOWN,
        decision=None,
    )
