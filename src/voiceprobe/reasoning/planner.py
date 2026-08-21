"""Generic Qwen patient planner with deterministic validation feedback."""

from __future__ import annotations

import json
from collections.abc import Sequence

import httpx
from pydantic import ValidationError

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
    RequestedFact,
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

Caller facts themselves are authoritative and may be attached
deterministically after planning.

MULTI-INTENT TURNS

A single remote utterance may ask permission AND request caller facts.

Example:

"Would you like to create a patient profile?
I just need your first and last name."

If remote_turn.requested_action = "grant_permission", the PRIMARY action
must still be GRANT_PERMISSION, DECLINE_PERMISSION, or CLARIFY according to
workflow policy.

Do NOT change the primary action to ANSWER_FACT merely because
requested_facts is non-empty.

The deterministic caller layer can attach the requested authoritative facts
to the approved primary action.

SEARCH / WORKFLOW PERMISSION

If requested_action is "grant_permission", determine what permission is
actually being requested.

If proposed_workflow is null, evaluate the ordinary requested operation using
the workflow and the caller objective.

If proposed_workflow is present, compare the proposed supporting workflow
against the caller's objective.

GRANT permission when the proposed workflow is:

- an explicit prerequisite for achieving the caller's objective
- a reasonable enabling/setup step for the same service
- useful for completing the objective
- compatible with all hard caller constraints

DECLINE permission when it is:

- unrelated to the objective
- promotional or distracting
- a diversion into a different goal
- incompatible with caller truth or hard constraints

Use CLARIFY when there is insufficient semantic information to determine what
the proposed workflow would do.

Do not grant permission merely because the remote agent asked.

Do not decline a workflow merely because requirement = "optional".
Optional means caller consent is being requested; it does not mean the
workflow is irrelevant.

A profile or identity setup workflow for the same service can reasonably
enable a later transaction even when it is not itself the caller's final
objective.

GENERAL PRESENTED CHOICES

Some remote questions present search/workflow alternatives rather than
concrete appointment slots.

For requested_action = "choose_presented_choice":

- choose only from remote_turn.presented_choices
- preserve every HARD caller constraint
- reject explicitly conflicting branches
- a branch may omit a constraint because the caller can carry it forward
- never invent a branch
- do not answer with generic "yes"

Use action = "select_presented_choice" with selected_choice_index.

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
            world=world,
            turn=turn,
        )

        if deterministic is not None:
            deterministic = self._attach_requested_facts(
                world=world,
                turn=turn,
                plan=deterministic,
            )

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

        first = self._attach_requested_facts(
            world=world,
            turn=turn,
            plan=first,
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

        repaired = self._attach_requested_facts(
            world=world,
            turn=turn,
            plan=repaired,
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
    def _world_fact_key(
        fact: RequestedFact,
    ) -> str:
        """Map semantic fact identifiers onto generic world-state keys."""

        aliases = {
            RequestedFact.FULL_NAME: "name",
            RequestedFact.SYMPTOM_DURATION: "duration",
        }

        return aliases.get(
            fact,
            fact.value,
        )

    @classmethod
    def _world_has_fact(
        cls,
        *,
        world: PatientWorldModel,
        fact: RequestedFact,
    ) -> bool:
        """Return whether authoritative caller truth contains this fact."""

        key = cls._world_fact_key(
            fact
        )

        return (
            key in world.facts
            and world.facts[key] is not None
        )

    @classmethod
    def _resolve_requested_facts(
        cls,
        *,
        world: PatientWorldModel,
        requested: Sequence[RequestedFact],
    ) -> list[RequestedFact] | None:
        """Resolve requested semantic facts against authoritative truth.

        A joint first-name + last-name request may safely be satisfied by an
        authoritative full name when component fields are unavailable.

        We deliberately do NOT attempt to split a full name into guessed
        components.
        """

        requested_list = list(
            requested
        )

        if not requested_list:
            return []

        if all(
            cls._world_has_fact(
                world=world,
                fact=fact,
            )
            for fact in requested_list
        ):
            return requested_list

        requested_set = set(
            requested_list
        )

        joint_name_request = {
            RequestedFact.FIRST_NAME,
            RequestedFact.LAST_NAME,
        }

        if joint_name_request <= requested_set:
            if cls._world_has_fact(
                world=world,
                fact=RequestedFact.FULL_NAME,
            ):
                remaining = [
                    fact
                    for fact in requested_list
                    if fact
                    not in joint_name_request
                ]

                if all(
                    cls._world_has_fact(
                        world=world,
                        fact=fact,
                    )
                    for fact in remaining
                ):
                    return [
                        RequestedFact.FULL_NAME,
                        *remaining,
                    ]

        return None

    @classmethod
    def _attach_requested_facts(
        cls,
        *,
        world: PatientWorldModel,
        turn: TurnFrame,
        plan: ActionPlan,
    ) -> ActionPlan:
        """Derive fact disclosure exclusively from authoritative caller truth.

        The language model may choose the PRIMARY conversational action.

        It does NOT receive authority over which patient facts are disclosed.

        facts_to_answer is therefore always recomputed from:

            TurnFrame.requested_facts
            +
            PatientWorldModel

        Any facts_to_answer proposed by Qwen are ignored and replaced.

        This prevents:
        - hallucinated patient data
        - unavailable fields reaching verbalization
        - unnecessary disclosure of unrequested fields
        - model-selected substitutions for caller truth
        """

        payload = plan.model_dump(
            mode="json",
        )

        # Never trust a model-generated fact payload.
        payload["facts_to_answer"] = []

        if not turn.requested_facts:
            return ActionPlan.model_validate(
                payload
            )

        if plan.action in {
            PatientActionKind.WAIT,
            PatientActionKind.DECLINE_PERMISSION,
            PatientActionKind.END_CONVERSATION,
            PatientActionKind.CLARIFY,
        }:
            return ActionPlan.model_validate(
                payload
            )

        resolved = cls._resolve_requested_facts(
            world=world,
            requested=turn.requested_facts,
        )

        if resolved is None:
            # ANSWER_FACT cannot truthfully complete without an available
            # authoritative value, so fail closed as a clarification.
            if (
                plan.action
                is PatientActionKind.ANSWER_FACT
            ):
                return ActionPlan(
                    action=PatientActionKind.CLARIFY,
                    reason_code="requested_fact_unavailable",
                    confidence=plan.confidence,
                )

            # For a compound turn such as workflow permission + a fact
            # request, preserve the independently valid primary decision
            # but disclose no unsupported patient information.
            return ActionPlan.model_validate(
                payload
            )

        payload["facts_to_answer"] = [
            fact.value
            for fact in resolved
        ]

        return ActionPlan.model_validate(
            payload
        )

    def _deterministic_plan(
        self,
        *,
        world: PatientWorldModel,
        turn: TurnFrame,
    ) -> ActionPlan | None:
        """Resolve actions already established by structured semantics.

        This is semantic routing, not phrase matching.

        No patient name, literal receptionist wording, provider name,
        appointment time, or scenario-specific sentence appears here.
        """

        if (
            turn.booking_confirmed
            and turn.confirmed_appointment is not None
            and turn.requested_action is RequestedAction.NONE
            and not turn.response_required
        ):
            return ActionPlan(
                action=PatientActionKind.END_CONVERSATION,
                reason_code="booking_confirmed",
                confidence=turn.confidence,
            )

        if (
            turn.conversation_end_requested
            and turn.requested_action
            is RequestedAction.NONE
        ):
            return ActionPlan(
                action=PatientActionKind.END_CONVERSATION,
                reason_code="remote_conversation_end",
                confidence=turn.confidence,
            )

        if turn.requested_action is RequestedAction.WAIT:
            return ActionPlan(
                action=PatientActionKind.WAIT,
                reason_code="semantic_turn_requires_wait",
                confidence=turn.confidence,
            )

        if (
            turn.requested_action
            is RequestedAction.NONE
            and not turn.response_required
        ):
            # Passive acknowledgements or informational fragments such as
            # "Great." do not establish mission completion.
            #
            # Keep listening unless a stronger semantic signal above
            # established booking confirmation or conversation termination.
            return ActionPlan(
                action=PatientActionKind.WAIT,
                reason_code="passive_turn_requires_no_response",
                confidence=turn.confidence,
            )

        if (
            turn.requested_action
            is RequestedAction.ANSWER_FACT
            and turn.requested_facts
        ):
            resolved = self._resolve_requested_facts(
                world=world,
                requested=turn.requested_facts,
            )

            # Never invent caller information when the authoritative world
            # cannot satisfy the request.
            if resolved is None:
                return ActionPlan(
                    action=PatientActionKind.CLARIFY,
                    reason_code="requested_fact_unavailable",
                    confidence=turn.confidence,
                )

            return ActionPlan(
                action=PatientActionKind.ANSWER_FACT,
                facts_to_answer=resolved,
                reason_code="semantic_fact_request",
                confidence=turn.confidence,
            )

        if (
            turn.requested_action
            is RequestedAction.STATE_OBJECTIVE
        ):
            return ActionPlan(
                action=PatientActionKind.STATE_OBJECTIVE,
                reason_code="semantic_objective_request",
                confidence=turn.confidence,
            )

        if (
            turn.requested_action
            is RequestedAction.CHOOSE_PRESENTED_CHOICE
        ):
            compatible = (
                self.validator.compatible_presented_choice_indices(
                    world=world,
                    turn=turn,
                )
            )

            if not compatible:
                return ActionPlan(
                    action=PatientActionKind.REQUEST_ALTERNATIVE,
                    reason_code="no_compatible_presented_choice",
                    confidence=1.0,
                )

            if len(compatible) == 1:
                return ActionPlan(
                    action=PatientActionKind.SELECT_PRESENTED_CHOICE,
                    selected_choice_index=compatible[0],
                    reason_code="only_compatible_presented_choice",
                    confidence=1.0,
                )

            return None

        if (
            turn.requested_action
            is RequestedAction.CHOOSE_OPTION
        ):
            compatible = (
                self.validator.compatible_option_indices(
                    world=world,
                    turn=turn,
                )
            )

            # No offered option satisfies caller truth.
            #
            # Do not ask a language model whether an incompatible option
            # should somehow be accepted.
            if not compatible:
                return ActionPlan(
                    action=PatientActionKind.REQUEST_ALTERNATIVE,
                    reason_code="no_compatible_option",
                    confidence=1.0,
                )

            # Exactly one option satisfies every hard constraint.
            # The decision is mechanically determined.
            if len(compatible) == 1:
                return ActionPlan(
                    action=PatientActionKind.SELECT_OPTION,
                    selected_option_index=compatible[0],
                    reason_code="only_compatible_option",
                    confidence=1.0,
                )

            # Multiple policy-valid choices remain.
            # This is an actual preference/decision problem and may be
            # delegated to Qwen.
            return None

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

        messages: list[dict[str, str]] = [
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
        ]

        last_error: ValidationError | None = None

        for attempt in range(2):

            response = self._client.post(
                self.url,
                json={
                    "model": self.model,
                    "stream": False,
                    "think": False,
                    "format": schema,
                    "options": {
                        "temperature": 0,
                    },
                    "messages": messages,
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

            try:
                return (
                    ActionPlan.model_validate_json(
                        content
                    )
                )

            except ValidationError as error:
                last_error = error

                if attempt == 1:
                    break

                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous ActionPlan was structurally "
                            "invalid.\n\n"
                            "Validation error:\n"
                            f"{error}\n\n"
                            "Return a corrected ActionPlan. "
                            "SELECT_OPTION must contain a valid "
                            "selected_option_index. Do not invent an "
                            "appointment option and do not violate hard "
                            "patient constraints."
                        ),
                    }
                )

        assert last_error is not None

        raise last_error
