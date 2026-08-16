"""Generic Qwen patient planner with deterministic validation feedback."""

from __future__ import annotations

import json
from collections.abc import Sequence

import httpx

from voiceprobe.reasoning.action_plan import (
    ActionPlan,
    PatientActionKind,
)
from voiceprobe.reasoning.constraint_validator import (
    ConstraintValidator,
    ConstraintViolation,
)
from voiceprobe.reasoning.turn_frame import (
    RequestedAction,
    TurnFrame,
)
from voiceprobe.reasoning.world_model import (
    PatientWorldModel,
)


SYSTEM_PROMPT = """\
You are the planning layer for an autonomous simulated caller.

You receive:

1. patient_world
2. remote_turn

patient_world contains immutable caller facts, objective, and constraints.

remote_turn is already a source-grounded semantic interpretation of what the
remote voice agent said.

Choose ONE ActionPlan.

CORE RULES

HARD constraints are inviolable.

Never select an appointment option that conflicts with a HARD constraint.

If the remote agent asks the caller to choose among appointment options:

- select a specific option only if it is compatible with all HARD constraints
- if none are compatible, REQUEST_ALTERNATIVE
- if required information is genuinely missing, CLARIFY

There is deliberately no generic AGREE action.

Do not respond "yes" to an option-selection question without identifying a
specific compatible option.

WAIT

If requested_action is "wait":
action = "wait"

FACT REQUESTS

If requested_action is "answer_fact":
action = "answer_fact"
facts_to_answer must contain only the requested canonical facts that the
caller can answer from patient_world.

SEARCH / WORKFLOW PERMISSION

If requested_action is "grant_permission" and the requested operation advances
the caller's stated objective without violating a hard constraint:
action = "grant_permission"

OPTION SELECTION

If requested_action is "choose_option":

1. inspect EVERY appointment option
2. compare each option to EVERY hard patient constraint
3. choose a compatible option only when all relevant hard constraints match
4. if zero compatible options remain, request an alternative

Do not relax a hard constraint merely because the remote agent provided no
matching option.

Do not alter patient_world.

VALIDATION FEEDBACK

If validation_feedback is supplied, your previous plan violated deterministic
policy. Correct the plan. Do not argue with the validator.

Return only schema-valid structured output.
"""


class PlanningFailure(RuntimeError):
    """Planner could not produce a policy-valid plan."""


class QwenPatientPlanner:
    """Generate and repair typed plans using local Qwen."""

    def __init__(
        self,
        *,
        model: str,
        url: str,
        validator: ConstraintValidator | None = None,
        client: httpx.Client | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:

        self.model = model
        self.url = url

        self.validator = (
            validator
            if validator is not None
            else ConstraintValidator()
        )

        self._owns_client = (
            client is None
        )

        self._client = (
            client
            if client is not None
            else httpx.Client(
                timeout=timeout_seconds,
            )
        )

    def close(
        self,
    ) -> None:
        if self._owns_client:
            self._client.close()

    def plan(
        self,
        *,
        world: PatientWorldModel,
        turn: TurnFrame,
        recent_actions: Sequence[ActionPlan] = (),
    ) -> tuple[
        ActionPlan,
        tuple[ConstraintViolation, ...],
    ]:
        """Generate a policy-valid patient action.

        Semantically settled turns do not need a second LLM pass.

        The semantic reasoner has already converted arbitrary natural
        language into a typed TurnFrame. If that frame mechanically
        determines the caller action, preserve that meaning directly.

        Qwen remains responsible for turns requiring actual goal reasoning,
        option evaluation, negotiation, or ambiguity resolution.
        """

        deterministic = self._deterministic_plan(
            turn=turn,
        )

        if deterministic is not None:
            violations = self.validator.validate(
                world=world,
                turn=turn,
                plan=deterministic,
            )

            if violations:
                details = "; ".join(
                    f"{item.code}: {item.detail}"
                    for item in violations
                )

                raise PlanningFailure(
                    "Deterministic plan violated policy: "
                    f"{details}"
                )

            return deterministic, ()

        first = self._plan_once(
            world=world,
            turn=turn,
            recent_actions=recent_actions,
            validation_feedback=(),
        )

        violations = (
            self.validator.validate(
                world=world,
                turn=turn,
                plan=first,
            )
        )

        if not violations:
            return first, ()

        repaired = self._plan_once(
            world=world,
            turn=turn,
            recent_actions=recent_actions,
            validation_feedback=violations,
        )

        repaired_violations = (
            self.validator.validate(
                world=world,
                turn=turn,
                plan=repaired,
            )
        )

        if repaired_violations:
            details = "; ".join(
                (
                    f"{item.code}: "
                    f"{item.detail}"
                )
                for item in repaired_violations
            )

            raise PlanningFailure(
                "Planner failed deterministic validation "
                f"after repair: {details}"
            )

        return (
            repaired,
            violations,
        )

    @staticmethod
    def _deterministic_plan(
        *,
        turn: TurnFrame,
    ) -> ActionPlan | None:
        """Resolve actions already established by structured semantics.

        This is semantic routing, not phrase matching.

        No patient name, literal receptionist wording, provider name,
        appointment time, or scenario-specific sentence appears here.
        """

        if turn.requested_action is RequestedAction.WAIT:
            return ActionPlan(
                action=PatientActionKind.WAIT,
                reason_code="semantic_turn_requires_wait",
                confidence=turn.confidence,
            )

        if (
            turn.requested_action
            is RequestedAction.ANSWER_FACT
            and turn.requested_facts
        ):
            return ActionPlan(
                action=PatientActionKind.ANSWER_FACT,
                facts_to_answer=list(
                    turn.requested_facts
                ),
                reason_code="semantic_fact_request",
                confidence=turn.confidence,
            )

        return None

    def _plan_once(
        self,
        *,
        world: PatientWorldModel,
        turn: TurnFrame,
        recent_actions: Sequence[ActionPlan],
        validation_feedback: Sequence[
            ConstraintViolation
        ],
    ) -> ActionPlan:

        schema = (
            ActionPlan.model_json_schema()
        )

        context = {
            "patient_world": (
                world.model_dump(
                    mode="json",
                )
            ),
            "remote_turn": (
                turn.model_dump(
                    mode="json",
                )
            ),
            "recent_actions": [
                action.model_dump(
                    mode="json",
                )
                for action
                in recent_actions[-4:]
            ],
            "validation_feedback": [
                {
                    "code": item.code,
                    "detail": item.detail,
                }
                for item
                in validation_feedback
            ],
        }

        response = self._client.post(
            self.url,
            json={
                "model": self.model,
                "stream": False,

                # Keep the first integration measurable.
                # We can benchmark Qwen thinking mode separately later.
                "think": False,

                "format": schema,

                "options": {
                    "temperature": 0,
                },

                "messages": [
                    {
                        "role": "system",
                        "content": (
                            SYSTEM_PROMPT
                            + "\n\nOUTPUT JSON SCHEMA:\n"
                            + json.dumps(
                                schema,
                                separators=(",", ":"),
                            )
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            context,
                            separators=(",", ":"),
                        ),
                    },
                ],
            },
        )

        response.raise_for_status()

        payload = response.json()

        try:
            content = (
                payload[
                    "message"
                ][
                    "content"
                ]
            )
        except (
            KeyError,
            TypeError,
        ) as error:
            raise RuntimeError(
                "Planner response did not contain message.content."
            ) from error

        if not isinstance(
            content,
            str,
        ):
            raise RuntimeError(
                "Planner message.content must be text."
            )

        return (
            ActionPlan.model_validate_json(
                content
            )
        )
