from __future__ import annotations

from dataclasses import dataclass

import pytest

from voiceprobe.v33.actions import ActionKind, ActionMove, ActionPlan
from voiceprobe.v33.mind import AgentMind
from voiceprobe.v33.mission import BugTarget, PatientTruth, Preference, TestMission as MissionSpec, adaptive_reschedule_mission
from voiceprobe.v33.planner import V33Planner
from voiceprobe.v33.world_model import ObservationKind, RemoteObservation


@dataclass
class ScriptedReasoner:
    observation: RemoteObservation
    candidates: tuple[ActionPlan, ...]

    async def propose(self, *, mind, remote_turn):
        return self.observation, self.candidates


def plan(kind: ActionKind, *, args=None, utterance="ok", rationale="test") -> ActionPlan:
    return ActionPlan(
        moves=(ActionMove(kind, args or {}),),
        rationale=rationale,
        utterance=utterance,
    )


@pytest.mark.asyncio
async def test_existing_patient_prefers_lookup_branch_over_profile_creation() -> None:
    mission = MissionSpec(
        mission_id="profile-branch",
        primary_goal="Test existing-profile handling",
        bug_targets=(),
        truth=PatientTruth(existing_profile=True),
        preferences=(),
    )
    observation = RemoteObservation(
        ObservationKind.PROFILE_REQUEST,
        "Would you like to create a profile?",
        True,
    )
    reasoner = ScriptedReasoner(
        observation,
        (
            plan(ActionKind.CREATE_PROFILE, utterance="Create one."),
            plan(ActionKind.REQUEST_PROFILE_LOOKUP, utterance="Please look me up first."),
        ),
    )
    planner = V33Planner(mind=AgentMind(mission), reasoner=reasoner)

    decision = await planner.decide(observation.raw_text)

    assert decision.selected.plan.has(ActionKind.REQUEST_PROFILE_LOOKUP)
    assert any("cannot_create_new_profile" in reason for item in decision.rejected for reason in item.validation.reasons)


@pytest.mark.asyncio
async def test_actionable_question_cannot_fall_into_wait() -> None:
    mission = adaptive_reschedule_mission()
    observation = RemoteObservation(
        ObservationKind.OPTION_OFFER,
        "Would you like another day or Friday morning?",
        True,
        offered_options=("Friday morning", "another weekday afternoon"),
        unavailable_constraints=("Friday afternoon",),
    )
    reasoner = ScriptedReasoner(
        observation,
        (
            plan(ActionKind.WAIT, utterance=""),
            plan(ActionKind.ASK_ALTERNATIVES, utterance="What are the closest alternatives?"),
        ),
    )
    planner = V33Planner(mind=AgentMind(mission), reasoner=reasoner)

    decision = await planner.decide(observation.raw_text)

    assert not decision.selected.plan.has(ActionKind.WAIT)
    assert decision.spoken_text
    assert any("actionable_remote_turn_cannot_wait" in item.validation.reasons for item in decision.rejected)


@pytest.mark.asyncio
async def test_preference_relaxation_uses_weight_not_friday_specific_rule() -> None:
    mission = MissionSpec(
        mission_id="weighted-relaxation",
        primary_goal="Find a valid appointment",
        bug_targets=(BugTarget.AVAILABILITY_FALLBACK,),
        truth=PatientTruth(),
        preferences=(
            Preference("day", "Friday", 0.90),
            Preference("time_of_day", "afternoon", 0.80),
        ),
    )
    observation = RemoteObservation(
        ObservationKind.AVAILABILITY_RESULT,
        "Your preferred combination is unavailable. I can change the day or the time.",
        True,
        unavailable_constraints=("day+time_of_day",),
    )
    reasoner = ScriptedReasoner(
        observation,
        (
            plan(ActionKind.RELAX_PREFERENCE, args={"key": "day"}, utterance="Try another day."),
            plan(ActionKind.RELAX_PREFERENCE, args={"key": "time_of_day"}, utterance="Keep the day and try another time."),
        ),
    )
    planner = V33Planner(mind=AgentMind(mission), reasoner=reasoner)

    decision = await planner.decide(observation.raw_text)

    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "time_of_day"


@pytest.mark.asyncio
async def test_transaction_authorization_requires_selected_and_verified_option() -> None:
    mission = adaptive_reschedule_mission()
    observation = RemoteObservation(
        ObservationKind.TRANSACTION_PERMISSION_REQUEST,
        "Should I book it?",
        True,
    )
    reasoner = ScriptedReasoner(
        observation,
        (
            plan(ActionKind.AUTHORIZE_TRANSACTION, utterance="Yes, book it."),
            ActionPlan(
                moves=(
                    ActionMove(ActionKind.WITHHOLD_AUTHORIZATION),
                    ActionMove(ActionKind.ASK_CONFIRMATION),
                ),
                rationale="Verify before committing",
                utterance="Don't book it yet. Which exact option is selected?",
            ),
        ),
    )
    planner = V33Planner(mind=AgentMind(mission), reasoner=reasoner)

    decision = await planner.decide(observation.raw_text)

    assert decision.selected.plan.has(ActionKind.WITHHOLD_AUTHORIZATION)
    assert not planner.mind.world.transaction_authorized
    assert any("transaction_authorization_requires_selected_option" in item.validation.reasons for item in decision.rejected)


@pytest.mark.asyncio
async def test_model_cannot_supply_authoritative_patient_fact_value() -> None:
    mission = adaptive_reschedule_mission()
    observation = RemoteObservation(
        ObservationKind.FACT_REQUEST,
        "What is your date of birth?",
        True,
        requested_fact="dob",
    )
    reasoner = ScriptedReasoner(
        observation,
        (
            plan(ActionKind.PROVIDE_FACT, args={"fact_key": "dob", "value": "January 1, 1900"}, utterance="January 1, 1900"),
            plan(ActionKind.PROVIDE_FACT, args={"fact_key": "dob"}, utterance="My date of birth is April 12, 1998."),
        ),
    )
    planner = V33Planner(mind=AgentMind(mission), reasoner=reasoner)

    decision = await planner.decide(observation.raw_text)

    assert decision.spoken_text == "April 12, 1998."
    assert any("model_must_not_supply_authoritative_fact_value" in item.validation.reasons for item in decision.rejected)
