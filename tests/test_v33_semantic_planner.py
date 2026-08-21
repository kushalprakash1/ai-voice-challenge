from __future__ import annotations

from dataclasses import dataclass

import pytest

from voiceprobe.v33.actions import ActionKind
from voiceprobe.v33.mind import AgentMind
from voiceprobe.v33.mission import adaptive_reschedule_mission
from voiceprobe.v33.planner import V33Planner
from voiceprobe.v33.reasoner import OllamaV33Reasoner
from voiceprobe.v33.action_generator import StrategicActionGenerator


@dataclass
class FakeBackend:
    response: dict
    last_system: str = ""
    last_prompt: str = ""

    async def generate_json(self, *, system, prompt, schema):
        self.last_system = system
        self.last_prompt = prompt
        return dict(self.response)


def semantic_response(**overrides) -> dict:
    payload = {
        "kind": "other",
        "respond": False,
        "fact": "none",
        "question_target": "none",
        "offers": [],
        "unavailable": [],
        "temporal_scope": "none",
        "search_constraints": [],
        "claims": [],
        "selected": "",
        "operation": "none",
    }
    payload.update(overrides)
    return payload


def make_reasoner(response: dict) -> OllamaV33Reasoner:
    return OllamaV33Reasoner(
        backend=FakeBackend(response),
        generator=StrategicActionGenerator(),
    )


@pytest.mark.asyncio
async def test_model_only_interprets_reschedule_reason_python_chooses_action():
    backend = FakeBackend(
        {
            "kind": "reschedule_reason_request",
            "respond": True,
            "fact": "reschedule_reason",
            "question_target": "reschedule_reason",
            "offers": [],
            "unavailable": [],
            "search_constraints": [],
            "claims": [],
            "selected": "",
            "operation": "reschedule",
        }
    )
    reasoner = OllamaV33Reasoner(
        backend=backend,
        generator=StrategicActionGenerator(),
    )
    planner = V33Planner(
        mind=AgentMind(adaptive_reschedule_mission()),
        reasoner=reasoner,
    )

    decision = await planner.decide("Why do you need to reschedule?")

    assert decision.selected.plan.has(ActionKind.PROVIDE_FACT)
    assert "no longer works" in decision.spoken_text.casefold()
    assert "menu" not in backend.last_prompt.casefold()
    assert "menu" not in backend.last_system.casefold()


@pytest.mark.asyncio
async def test_alternative_search_relaxes_relevant_lower_weight_axis():
    reasoner = make_reasoner(
        {
            "kind": "alternative_search_offer",
            "respond": True,
            "fact": "none",
            "question_target": "search_preference",
            "offers": ["Friday morning"],
            "unavailable": ["combination"],
            "search_constraints": ["time_of_day"],
            "claims": [],
            "selected": "",
            "operation": "search",
        }
    )
    planner = V33Planner(
        mind=AgentMind(adaptive_reschedule_mission()),
        reasoner=reasoner,
    )

    decision = await planner.decide(
        "No Friday afternoons. Should I check Friday mornings instead?"
    )

    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "time_of_day"
    assert decision.spoken_text.strip()


@pytest.mark.asyncio
async def test_existing_profile_branch_is_state_driven_not_default_yes():
    reasoner = make_reasoner(
        {
            "kind": "profile_request",
            "respond": True,
            "fact": "none",
            "question_target": "profile",
            "offers": [],
            "unavailable": [],
            "search_constraints": [],
            "claims": [],
            "selected": "",
            "operation": "none",
        }
    )
    planner = V33Planner(
        mind=AgentMind(adaptive_reschedule_mission()),
        reasoner=reasoner,
    )

    decision = await planner.decide("Would you like me to create a profile?")

    assert decision.selected.plan.has(ActionKind.REQUEST_PROFILE_LOOKUP) or decision.selected.plan.has(ActionKind.CLAIM_EXISTING_PROFILE)
    assert not decision.selected.plan.has(ActionKind.CREATE_PROFILE)


@pytest.mark.asyncio
async def test_presence_check_recovers_without_stale_script():
    reasoner = make_reasoner(
        {
            "kind": "presence_check",
            "respond": True,
            "fact": "none",
            "question_target": "presence",
            "offers": [],
            "unavailable": [],
            "search_constraints": [],
            "claims": [],
            "selected": "",
            "operation": "none",
        }
    )
    planner = V33Planner(
        mind=AgentMind(adaptive_reschedule_mission()),
        reasoner=reasoner,
    )

    decision = await planner.decide("Are you still there?")
    assert decision.selected.plan.has(ActionKind.RESUME_WORKFLOW)
    assert "here" in decision.spoken_text.casefold()


@pytest.mark.asyncio
async def test_question_target_repairs_reschedule_reason_without_raw_phrase_matching():
    reasoner = make_reasoner(
        semantic_response(
            kind="other",
            respond=False,
            fact="none",
            question_target="reschedule_reason",
            operation="reschedule",
        )
    )
    planner = V33Planner(
        mind=AgentMind(adaptive_reschedule_mission()),
        reasoner=reasoner,
    )

    decision = await planner.decide("arbitrary text supplied by a test double")
    assert decision.observation.kind.value == "reschedule_reason_request"
    assert decision.observation.requested_fact == "reschedule_reason"
    assert decision.observation.requires_response is True
    assert decision.selected.plan.has(ActionKind.PROVIDE_FACT)


@pytest.mark.asyncio
async def test_search_target_outranks_spurious_reschedule_fact():
    reasoner = make_reasoner(
        semantic_response(
            kind="reschedule_reason_request",
            respond=True,
            fact="reschedule_reason",
            question_target="search_preference",
            search_constraints=["time_of_day"],
            operation="search",
        )
    )
    planner = V33Planner(
        mind=AgentMind(adaptive_reschedule_mission()),
        reasoner=reasoner,
    )

    decision = await planner.decide("synthetic fallback-search question")
    assert decision.observation.kind.value == "alternative_search_offer"
    assert decision.observation.requested_fact == ""
    assert decision.observation.search_constraints == ("time_of_day",)
    assert not decision.selected.plan.has(ActionKind.PROVIDE_FACT)
    assert decision.selected.plan.has(ActionKind.RELAX_PREFERENCE) or decision.selected.plan.has(ActionKind.ASK_ALTERNATIVES)


@pytest.mark.asyncio
async def test_search_axis_can_drive_relaxation_without_concrete_offer():
    reasoner = make_reasoner(
        semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            search_constraints=["provider"],
            operation="search",
        )
    )
    planner = V33Planner(
        mind=AgentMind(adaptive_reschedule_mission()),
        reasoner=reasoner,
    )

    decision = await planner.decide("synthetic provider fallback question")
    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "provider"


def test_semantic_schema_uses_one_coarse_kind_without_redundant_question_target():
    reasoner = make_reasoner(semantic_response())
    schema = reasoner._schema(("day", "time_of_day", "provider"))
    props = schema["properties"]

    assert "kind" in props
    assert "question_target" not in props
    assert "fallback_target" in props
    assert "availability_failure" in props
    assert "record_claim" in props
    assert "search_constraints" not in props
    assert "fallback_temporal_change" not in props
    assert "fallback_temporal_retained" not in props
    assert "provider_fallback" not in props
    assert "unspecified_relaxation" not in props

@pytest.mark.asyncio
async def test_legacy_date_range_alias_normalizes_to_day_axis():
    reasoner = make_reasoner(
        semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            search_constraints=["date_range"],
            operation="search",
        )
    )
    planner = V33Planner(
        mind=AgentMind(adaptive_reschedule_mission()),
        reasoner=reasoner,
    )

    decision = await planner.decide("synthetic calendar-window fallback")
    assert decision.observation.search_constraints == ("day",)
    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "day"


@pytest.mark.asyncio
async def test_availability_result_is_actionable_even_without_direct_question():
    reasoner = make_reasoner(
        semantic_response(
            kind="availability_result",
            respond=False,
            unavailable=["combination"],
        )
    )
    planner = V33Planner(
        mind=AgentMind(adaptive_reschedule_mission()),
        reasoner=reasoner,
    )

    decision = await planner.decide("synthetic availability failure")
    assert decision.observation.requires_response is True
    assert not decision.selected.plan.has(ActionKind.WAIT)
    assert decision.selected.plan.has(ActionKind.RELAX_PREFERENCE) or decision.selected.plan.has(ActionKind.ASK_ALTERNATIVES)

def test_v12_schema_uses_one_mutually_exclusive_fallback_target():
    reasoner = make_reasoner(semantic_response())
    schema = reasoner._schema(("day", "time_of_day", "provider"))
    props = schema["properties"]

    assert set(props["fallback_target"]["enum"]) == {
        "none",
        "clock_time_or_daypart",
        "calendar_date_or_day",
        "either_time_or_calendar",
        "provider",
        "unspecified_preference",
    }
    assert set(props["availability_failure"]["enum"]) == {
        "none",
        "clock_time_or_daypart",
        "calendar_date_or_day",
        "provider",
        "combination",
    }
    assert set(props["record_claim"]["enum"]) == {
        "none",
        "profile_exists",
        "profile_missing",
        "appointment_exists",
        "appointment_missing",
    }
    assert "fallback_temporal_change" not in props
    assert "fallback_temporal_retained" not in props
    assert "fallback_temporal_relation" not in props
    assert "fallback_time_of_day" not in props
    assert "fallback_calendar" not in props
    assert "temporal_scope" not in props
    assert "search_constraints" not in props

@pytest.mark.asyncio
async def test_within_day_scope_derives_time_axis_even_if_raw_axis_conflicts():
    reasoner = make_reasoner(
        semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            temporal_scope="within_day",
            search_constraints=["day"],
            operation="search",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic temporal fallback")
    assert decision.observation.search_constraints == ("time_of_day",)


@pytest.mark.asyncio
async def test_calendar_scope_derives_day_axis_even_if_raw_axis_conflicts():
    reasoner = make_reasoner(
        semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            temporal_scope="calendar",
            search_constraints=["time_of_day"],
            operation="search",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic calendar fallback")
    assert decision.observation.search_constraints == ("day",)


@pytest.mark.asyncio
async def test_mixed_temporal_scope_derives_both_axes():
    reasoner = make_reasoner(
        semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            temporal_scope="mixed",
            operation="search",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic mixed temporal fallback")
    assert decision.observation.search_constraints == ("time_of_day", "day")


def new_semantic_response(**overrides) -> dict:
    payload = {
        "kind": "other",
        "respond": False,
        "fact": "none",
        "question_target": "none",
        "offers": [],
        "failed_temporal_scope": "none",
        "provider_unavailable": False,
        "combination_unavailable": False,
        "temporal_scope": "none",
        "provider_fallback": False,
        "unspecified_relaxation": False,
        "claims": [],
        "selected": "",
        "operation": "none",
    }
    payload.update(overrides)
    return payload


def test_v12_schema_removes_independent_axis_booleans_and_record_status_pairs():
    reasoner = make_reasoner(new_semantic_response())
    schema = reasoner._schema(("day", "time_of_day", "provider"))
    props = schema["properties"]

    assert "search_constraints" not in props
    assert "unavailable" not in props
    assert "temporal_scope" not in props
    assert "failed_temporal_scope" not in props
    assert "fallback_target" in props
    assert "availability_failure" in props
    assert "record_claim" in props

    for obsolete in {
        "failed_time_of_day",
        "failed_calendar",
        "provider_unavailable",
        "combination_unavailable",
        "fallback_temporal_change",
        "fallback_temporal_retained",
        "fallback_temporal_relation",
        "fallback_time_of_day",
        "fallback_calendar",
        "provider_fallback",
        "unspecified_relaxation",
        "profile_record_status",
        "appointment_record_status",
    }:
        assert obsolete not in props

@pytest.mark.asyncio
async def test_semantic_prompt_does_not_receive_patient_goal_or_preference_values():
    backend = FakeBackend(new_semantic_response(kind="acknowledgement"))
    reasoner = OllamaV33Reasoner(backend=backend, generator=StrategicActionGenerator())
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)

    await planner.decide("Okay, one moment.")

    prompt = backend.last_prompt.casefold()
    assert "friday" not in prompt
    assert "afternoon" not in prompt
    assert "first available" not in prompt
    assert "reschedule my appointment" not in prompt


@pytest.mark.asyncio
async def test_v08_temporal_fallback_does_not_leak_provider():
    reasoner = make_reasoner(
        new_semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            temporal_scope="calendar",
            provider_fallback=False,
            operation="search",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic calendar fallback")

    assert decision.observation.search_constraints == ("day",)
    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "day"


@pytest.mark.asyncio
async def test_v08_failed_axis_cannot_hijack_explicit_fallback_axis():
    reasoner = make_reasoner(
        new_semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            failed_temporal_scope="within_day",
            temporal_scope="calendar",
            operation="search",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic failed-time proposed-calendar fallback")

    assert decision.observation.unavailable_constraints == ("time_of_day",)
    assert decision.observation.search_constraints == ("day",)
    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "day"


@pytest.mark.asyncio
async def test_v08_provider_failure_does_not_create_provider_relaxation_when_calendar_is_proposed():
    reasoner = make_reasoner(
        new_semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            provider_unavailable=True,
            temporal_scope="calendar",
            provider_fallback=False,
            operation="search",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic provider-failed calendar-fallback")

    assert decision.observation.unavailable_constraints == ("provider",)
    assert decision.observation.search_constraints == ("day",)
    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "day"


@pytest.mark.asyncio
async def test_v08_unspecified_search_question_structurally_becomes_combination():
    reasoner = make_reasoner(
        new_semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            temporal_scope="none",
            provider_fallback=False,
            unspecified_relaxation=False,
            operation="search",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic unspecified fallback question")

    assert decision.observation.search_constraints == ("combination",)
    assert decision.selected.plan.has(ActionKind.RELAX_PREFERENCE) or decision.selected.plan.has(ActionKind.ASK_ALTERNATIVES)


@pytest.mark.asyncio
async def test_v08_availability_result_uses_failed_scope_for_proactive_replan():
    reasoner = make_reasoner(
        new_semantic_response(
            kind="availability_result",
            failed_temporal_scope="within_day",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic time availability result")

    assert decision.observation.search_constraints == ()
    assert decision.observation.unavailable_constraints == ("time_of_day",)
    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "time_of_day"


def v09_semantic_response(**overrides) -> dict:
    payload = {
        "kind": "other",
        "respond": False,
        "fact": "none",
        "question_target": "none",
        "offers": [],
        "failed_time_of_day": False,
        "failed_calendar": False,
        "provider_unavailable": False,
        "combination_unavailable": False,
        "fallback_time_of_day": False,
        "fallback_calendar": False,
        "provider_fallback": False,
        "unspecified_relaxation": False,
        "profile_record_status": "none",
        "appointment_record_status": "none",
        "selected": "",
        "operation": "none",
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_v09_named_day_can_be_retained_while_only_time_of_day_changes():
    reasoner = make_reasoner(
        v09_semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            fallback_time_of_day=True,
            fallback_calendar=False,
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic named-day retained time fallback")
    assert decision.observation.search_constraints == ("time_of_day",)
    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "time_of_day"


@pytest.mark.asyncio
async def test_v09_time_and_calendar_fallbacks_can_both_be_true():
    reasoner = make_reasoner(
        v09_semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            fallback_time_of_day=True,
            fallback_calendar=True,
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic mixed fallback")
    assert decision.observation.search_constraints == ("time_of_day", "day")


@pytest.mark.asyncio
async def test_v09_no_openings_does_not_become_missing_appointment_record():
    reasoner = make_reasoner(
        v09_semantic_response(
            kind="availability_result",
            combination_unavailable=True,
            appointment_record_status="none",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic availability-only statement")
    assert "appointment_missing" not in decision.observation.remote_claims
    assert not decision.selected.plan.has(ActionKind.CHALLENGE_REMOTE_STATE)


@pytest.mark.asyncio
async def test_v09_explicit_missing_appointment_record_remains_a_remote_claim():
    reasoner = make_reasoner(
        v09_semantic_response(
            kind="appointment_state",
            appointment_record_status="missing",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic explicit appointment-record absence")
    assert decision.observation.remote_claims == ("appointment_missing",)
    assert decision.selected.plan.has(ActionKind.CHALLENGE_REMOTE_STATE)


def v10_semantic_response(**overrides) -> dict:
    payload = {
        "kind": "other",
        "respond": False,
        "fact": "none",
        "question_target": "none",
        "offers": [],
        "failed_time_of_day": False,
        "failed_calendar": False,
        "provider_unavailable": False,
        "combination_unavailable": False,
        "fallback_temporal_relation": "none",
        "provider_fallback": False,
        "unspecified_relaxation": False,
        "profile_record_status": "none",
        "appointment_record_status": "none",
        "selected": "",
        "operation": "none",
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_v10_retained_calendar_time_change_derives_time_of_day_only():
    reasoner = make_reasoner(
        v10_semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            fallback_temporal_relation="retains_calendar_changes_time",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic retained-calendar time fallback")
    assert decision.observation.search_constraints == ("time_of_day",)
    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "time_of_day"


@pytest.mark.asyncio
async def test_v10_calendar_change_derives_day_only():
    reasoner = make_reasoner(
        v10_semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            fallback_temporal_relation="changes_calendar",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic calendar-change fallback")
    assert decision.observation.search_constraints == ("day",)
    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "day"


@pytest.mark.asyncio
async def test_v10_explicit_both_options_derives_both_axes_but_planner_can_choose_time():
    reasoner = make_reasoner(
        v10_semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            fallback_temporal_relation="offers_both",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic day-or-time fallback")
    assert decision.observation.search_constraints == ("time_of_day", "day")
    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "time_of_day"


def v11_semantic_response(**overrides) -> dict:
    payload = {
        "kind": "other",
        "respond": False,
        "fact": "none",
        "question_target": "none",
        "offers": [],
        "failed_time_of_day": False,
        "failed_calendar": False,
        "provider_unavailable": False,
        "combination_unavailable": False,
        "fallback_temporal_change": "none",
        "fallback_temporal_retained": "none",
        "provider_fallback": False,
        "unspecified_relaxation": False,
        "profile_record_status": "none",
        "appointment_record_status": "none",
        "selected": "",
        "operation": "none",
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_v11_same_time_new_day_changes_calendar_only():
    reasoner = make_reasoner(
        v11_semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            fallback_temporal_change="calendar",
            fallback_temporal_retained="time_of_day",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic same-time new-day fallback")
    assert decision.observation.search_constraints == ("day",)
    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "day"


@pytest.mark.asyncio
async def test_v11_same_day_new_time_changes_time_only():
    reasoner = make_reasoner(
        v11_semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            fallback_temporal_change="time_of_day",
            fallback_temporal_retained="calendar",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic same-day new-time fallback")
    assert decision.observation.search_constraints == ("time_of_day",)
    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "time_of_day"


@pytest.mark.asyncio
async def test_v11_explicit_time_or_calendar_preserves_both_axes():
    reasoner = make_reasoner(
        v11_semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            fallback_temporal_change="time_or_calendar",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic day-or-time fallback")
    assert decision.observation.search_constraints == ("time_of_day", "day")
    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "time_of_day"


def test_v12_prompt_schema_asks_for_one_fallback_and_one_failure_category():
    reasoner = make_reasoner(v11_semantic_response())
    schema = reasoner._schema(("day", "time_of_day", "provider"))
    fallback = schema["properties"]["fallback_target"]["description"].casefold()
    failure = schema["properties"]["availability_failure"]["description"].casefold()
    record = schema["properties"]["record_claim"]["description"].casefold()

    assert "exactly one" in fallback or "choose one" in fallback
    assert "what the clinic explicitly proposes varying" in fallback
    assert "explicitly says is unavailable" in failure
    assert "availability" in record
    assert "missing-record" in record or "missing" in record




def v12_semantic_response(**overrides) -> dict:
    payload = {
        "kind": "other",
        "respond": False,
        "fact": "none",
        "question_target": "none",
        "offers": [],
        "availability_failure": "none",
        "fallback_target": "none",
        "record_claim": "none",
        "selected": "",
        "operation": "none",
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fallback_target", "expected_search", "expected_relax"),
    [
        ("clock_time_or_daypart", ("time_of_day",), "time_of_day"),
        ("calendar_date_or_day", ("day",), "day"),
        ("either_time_or_calendar", ("time_of_day", "day"), "time_of_day"),
        ("provider", ("provider",), "provider"),
    ],
)
async def test_v12_fallback_target_maps_to_mission_axes(
    fallback_target, expected_search, expected_relax
):
    reasoner = make_reasoner(
        v12_semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            fallback_target=fallback_target,
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic v12 fallback")

    assert decision.observation.search_constraints == expected_search
    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == expected_relax


@pytest.mark.asyncio
async def test_v12_unspecified_fallback_is_combination_only():
    reasoner = make_reasoner(
        v12_semantic_response(
            kind="alternative_search_offer",
            respond=True,
            question_target="search_preference",
            fallback_target="unspecified_preference",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic unnamed preference relaxation")
    assert decision.observation.search_constraints == ("combination",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("clock_time_or_daypart", ("time_of_day",)),
        ("calendar_date_or_day", ("day",)),
        ("provider", ("provider",)),
        ("combination", ("combination",)),
        ("none", ()),
    ],
)
async def test_v12_availability_failure_is_mutually_exclusive(failure, expected):
    reasoner = make_reasoner(
        v12_semantic_response(
            kind="availability_result",
            availability_failure=failure,
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic availability result")
    assert decision.observation.unavailable_constraints == expected


@pytest.mark.asyncio
async def test_v12_availability_cannot_emit_two_record_claims():
    reasoner = make_reasoner(
        v12_semantic_response(
            kind="availability_result",
            availability_failure="combination",
            record_claim="none",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic no-match availability")
    assert decision.observation.remote_claims == ()


@pytest.mark.asyncio
async def test_v12_single_explicit_record_claim_remains_actionable():
    reasoner = make_reasoner(
        v12_semantic_response(
            kind="appointment_state",
            record_claim="appointment_missing",
        )
    )
    planner = V33Planner(mind=AgentMind(adaptive_reschedule_mission()), reasoner=reasoner)
    decision = await planner.decide("synthetic explicit missing appointment record")
    assert decision.observation.remote_claims == ("appointment_missing",)
    assert decision.selected.plan.has(ActionKind.CHALLENGE_REMOTE_STATE)

# ==================== v0.13 derived-speech-act + ambiguity tests ====================


def v13_semantic_response(**overrides) -> dict:
    payload = {
        "kind": "other",
        "respond": False,
        "fact": "none",
        "offers": [],
        "availability_failure": "none",
        "fallback_target": "none",
        "record_claim": "none",
        "selected": "",
        "operation": "none",
    }
    payload.update(overrides)
    return payload


@dataclass
class SequenceBackend:
    responses: list[dict]
    calls: int = 0
    systems: list[str] = None

    def __post_init__(self):
        if self.systems is None:
            self.systems = []

    async def generate_json(self, *, system, prompt, schema):
        self.systems.append(system)
        index = self.calls
        self.calls += 1
        if index >= len(self.responses):
            raise AssertionError("Unexpected extra semantic backend call")
        return dict(self.responses[index])


def test_v13_live_schema_removes_redundant_question_target():
    reasoner = make_reasoner(v12_semantic_response())
    schema = reasoner._schema(("day", "time_of_day", "provider"))
    props = schema["properties"]
    assert "kind" in props
    assert "question_target" not in props
    assert "fallback_target" in props
    assert "availability_failure" in props
    assert "record_claim" in props


@pytest.mark.asyncio
async def test_v13_fallback_semantics_override_spurious_option_kind():
    reasoner = make_reasoner(
        v13_semantic_response(
            kind="option_offer",
            respond=True,
            fallback_target="clock_time_or_daypart",
            offers=["mornings"],
        )
    )
    planner = V33Planner(
        mind=AgentMind(adaptive_reschedule_mission()),
        reasoner=reasoner,
    )
    decision = await planner.decide("synthetic fallback category question")
    assert decision.observation.kind.value == "alternative_search_offer"
    assert decision.observation.search_constraints == ("time_of_day",)
    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "time_of_day"


@pytest.mark.asyncio
async def test_v17_fallback_target_overrides_spurious_availability_kind_even_when_respond_false():
    reasoner = make_reasoner(
        v13_semantic_response(
            kind="availability_result",
            respond=False,
            availability_failure="clock_time_or_daypart",
            fallback_target="clock_time_or_daypart",
        )
    )
    planner = V33Planner(
        mind=AgentMind(adaptive_reschedule_mission()),
        reasoner=reasoner,
    )
    decision = await planner.decide("synthetic availability plus fallback search")
    assert decision.observation.kind.value == "alternative_search_offer"
    assert decision.observation.search_constraints == ("time_of_day",)
    move = decision.selected.plan.first(ActionKind.RELAX_PREFERENCE)
    assert move is not None
    assert move.arg("key") == "time_of_day"


@pytest.mark.asyncio
async def test_v13_fallback_semantics_override_spurious_reschedule_kind():
    reasoner = make_reasoner(
        v13_semantic_response(
            kind="reschedule_reason_request",
            respond=True,
            fact="reschedule_reason",
            fallback_target="clock_time_or_daypart",
        )
    )
    planner = V33Planner(
        mind=AgentMind(adaptive_reschedule_mission()),
        reasoner=reasoner,
    )
    decision = await planner.decide("synthetic same-calendar different-time question")
    assert decision.observation.kind.value == "alternative_search_offer"
    assert decision.observation.requested_fact == ""
    assert decision.observation.search_constraints == ("time_of_day",)
    assert not decision.selected.plan.has(ActionKind.PROVIDE_FACT)


def test_v13_temporal_adjudication_only_triggers_on_cross_axis_conflict():
    assert OllamaV33Reasoner._needs_temporal_adjudication(
        v13_semantic_response(
            availability_failure="clock_time_or_daypart",
            fallback_target="calendar_date_or_day",
        )
    )
    assert OllamaV33Reasoner._needs_temporal_adjudication(
        v13_semantic_response(
            availability_failure="calendar_date_or_day",
            fallback_target="clock_time_or_daypart",
        )
    )
    assert not OllamaV33Reasoner._needs_temporal_adjudication(
        v13_semantic_response(
            availability_failure="clock_time_or_daypart",
            fallback_target="clock_time_or_daypart",
        )
    )
    assert OllamaV33Reasoner._needs_temporal_adjudication(
        v13_semantic_response(
            availability_failure="clock_time_or_daypart",
            fallback_target="either_time_or_calendar",
        )
    )
    assert OllamaV33Reasoner._needs_temporal_adjudication(
        v13_semantic_response(
            availability_failure="none",
            fallback_target="either_time_or_calendar",
        )
    )
    # Old test-double contracts never acquire a hidden extra model call.
    legacy = v12_semantic_response(
        availability_failure="clock_time_or_daypart",
        fallback_target="calendar_date_or_day",
    )
    assert not OllamaV33Reasoner._needs_temporal_adjudication(legacy)

@pytest.mark.asyncio
async def test_v13_narrow_adjudicator_can_correct_false_calendar_fallback():
    backend = SequenceBackend([{"calendar_change": "no"}])
    reasoner = OllamaV33Reasoner(
        backend=backend,
        generator=StrategicActionGenerator(),
    )
    raw = v13_semantic_response(
        kind="alternative_search_offer",
        respond=True,
        availability_failure="clock_time_or_daypart",
        fallback_target="calendar_date_or_day",
    )
    resolved = await reasoner._adjudicate_temporal_disagreement(
        raw=raw,
        remote_turn="synthetic ambiguous temporal fallback",
    )
    assert resolved["fallback_target"] == "clock_time_or_daypart"
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_v13_narrow_adjudicator_preserves_genuine_cross_axis_calendar_fallback():
    backend = SequenceBackend([{"calendar_change": "yes"}])
    reasoner = OllamaV33Reasoner(
        backend=backend,
        generator=StrategicActionGenerator(),
    )
    raw = v13_semantic_response(
        kind="alternative_search_offer",
        respond=True,
        availability_failure="clock_time_or_daypart",
        fallback_target="calendar_date_or_day",
    )
    resolved = await reasoner._adjudicate_temporal_disagreement(
        raw=raw,
        remote_turn="synthetic explicit cross-axis fallback",
    )
    assert resolved["fallback_target"] == "calendar_date_or_day"
    assert backend.calls == 1

# ==================== v0.14 retained-vs-varied temporal tests ====================


def test_v14_ambiguous_either_target_requires_narrow_adjudication():
    raw = v13_semantic_response(
        kind="alternative_search_offer",
        respond=True,
        availability_failure="none",
        fallback_target="either_time_or_calendar",
    )
    assert OllamaV33Reasoner._needs_temporal_adjudication(raw)


@pytest.mark.asyncio
async def test_v14_adjudicator_can_reduce_either_to_time_only_when_calendar_is_retained():
    backend = SequenceBackend([{"calendar_change": "no"}])
    reasoner = OllamaV33Reasoner(
        backend=backend,
        generator=StrategicActionGenerator(),
    )
    raw = v13_semantic_response(
        kind="alternative_search_offer",
        respond=True,
        availability_failure="none",
        fallback_target="either_time_or_calendar",
    )
    resolved = await reasoner._adjudicate_temporal_disagreement(
        raw=raw,
        remote_turn="synthetic retained-calendar changed-time fallback",
    )
    assert resolved["fallback_target"] == "clock_time_or_daypart"
    assert backend.calls == 1
    assert "fixed/retained" in backend.systems[0]


@pytest.mark.asyncio
async def test_v14_adjudicator_can_reduce_either_to_calendar_only_when_time_is_retained():
    backend = SequenceBackend([{"calendar_change": "yes"}])
    reasoner = OllamaV33Reasoner(
        backend=backend,
        generator=StrategicActionGenerator(),
    )
    raw = v13_semantic_response(
        kind="alternative_search_offer",
        respond=True,
        availability_failure="none",
        fallback_target="either_time_or_calendar",
    )
    resolved = await reasoner._adjudicate_temporal_disagreement(
        raw=raw,
        remote_turn="synthetic retained-time changed-calendar fallback",
    )
    assert resolved["fallback_target"] == "calendar_date_or_day"
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_v14_adjudicator_preserves_genuine_either_axis_offer():
    backend = SequenceBackend([{"calendar_change": "both"}])
    reasoner = OllamaV33Reasoner(
        backend=backend,
        generator=StrategicActionGenerator(),
    )
    raw = v13_semantic_response(
        kind="alternative_search_offer",
        respond=True,
        availability_failure="none",
        fallback_target="either_time_or_calendar",
    )
    resolved = await reasoner._adjudicate_temporal_disagreement(
        raw=raw,
        remote_turn="synthetic genuine time-or-calendar fallback",
    )
    assert resolved["fallback_target"] == "either_time_or_calendar"
    assert backend.calls == 1


def test_v14_legacy_contract_still_bypasses_hidden_adjudication_call():
    legacy = v12_semantic_response(
        availability_failure="none",
        fallback_target="either_time_or_calendar",
    )
    assert "question_target" in legacy
    assert not OllamaV33Reasoner._needs_temporal_adjudication(legacy)
