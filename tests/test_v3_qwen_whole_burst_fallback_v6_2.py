from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from voiceprobe.v3.flow_controller import SchedulingFlowController
from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.qwen_v3_fallback import QwenV3FallbackRouter


def test_existing_same_burst_slot_state_transition_remains_unchanged():
    controller = SchedulingFlowController()
    result = controller.decide_burst(
        [
            (
                "Would you like me to check afternoon options on a different "
                "day or check with a different provider?"
            ),
            "I have Wednesday at 3:15 PM. Would that work for you?",
        ]
    )
    assert result.after.allow_earlier_week_afternoons
    assert result.decision.kind == DecisionKind.GRANT_PERMISSION
    assert result.decision.reason == "compatible_concrete_slot_offered"
    assert result.after.accepted_slot_text == "3:15 PM"


def test_existing_profile_fragment_behavior_remains_unchanged():
    controller = SchedulingFlowController()
    result = controller.decide_burst(
        [
            (
                "This call may be recorded for quality and training purposes. "
                "Thank you for calling Pivot Point Orthopedics. "
                "Would you like to create a demo patient profile?"
            ),
            "I just need your first and last name to get started.",
        ]
    )
    assert result.decision.kind == DecisionKind.ANSWER_FACT
    assert result.decision.reason == "full_name_requested"


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    async def generate_json(self, *, system, prompt, schema):
        del system, prompt, schema
        return self.outputs.pop(0)


class CapturingBackend(FakeBackend):
    async def generate_json(self, *, system, prompt, schema):
        self.prompt = json.loads(prompt)
        return await super().generate_json(system=system, prompt=prompt, schema=schema)


def observation(kind, **updates):
    base = {
        "kind": kind,
        "requested_fact": "none",
        "choice_options": [],
        "remote_dob_claim": False,
        "remote_existing_appointment_claim": False,
        "requested_day_change": False,
        "requested_time_change": False,
        "requested_provider_change": False,
        "afternoon_constraint_retained": False,
        "requires_response": True,
    }
    base.update(updates)
    return base


@pytest.mark.asyncio
async def test_open_intent_returns_grounded_objective():
    router = QwenV3FallbackRouter(
        backend=FakeBackend(
            [observation("open_intent")]
        )
    )
    d = await router.resolve(
        "How can I assist you today?",
        object(),
    )
    assert d.kind == DecisionKind.STATE_OBJECTIVE
    assert "Friday afternoon" in d.text
    assert "rephrase" not in d.text.casefold()


@pytest.mark.asyncio
async def test_multichoice_does_not_collapse_to_cancel():
    router = QwenV3FallbackRouter(
        backend=FakeBackend(
            [
                observation(
                    "appointment_choice",
                    choice_options=[
                        "new_appointment",
                        "reschedule",
                        "cancel",
                    ],
                )
            ]
        )
    )
    d = await router.resolve(
        "Book new, reschedule, or cancel?",
        object(),
    )
    assert d.kind == DecisionKind.STATE_OBJECTIVE
    assert "new appointment" in d.text.casefold()


@pytest.mark.asyncio
async def test_reschedule_confirmation_answers_yes():
    router = QwenV3FallbackRouter(
        backend=FakeBackend(
            [observation("reschedule_confirmation")]
        )
    )
    d = await router.resolve(
        "Is this the appointment you want to reschedule?",
        object(),
    )
    assert d.kind == DecisionKind.GRANT_PERMISSION
    assert "reschedule" in d.text.casefold()


@pytest.mark.asyncio
async def test_open_intent_with_remote_dob_claim_corrects_and_moves_forward():
    router = QwenV3FallbackRouter(
        backend=FakeBackend(
            [
                observation(
                    "open_intent",
                    remote_dob_claim=True,
                )
            ]
        )
    )
    d = await router.resolve(
        "Your DOB is July 4 2000. How can I help you?",
        object(),
    )
    assert "April 12, 1998" in d.text
    assert "Friday afternoon" in d.text


def test_injected_backend_is_not_auto_warmed():
    backend = FakeBackend([observation("open_intent")])
    QwenV3FallbackRouter(backend=backend)
    assert len(backend.outputs) == 1


@pytest.mark.asyncio
async def test_router_shares_only_grounded_target_location_context():
    backend = CapturingBackend(
        [observation("other", self_pay_location_action="locations_offered")]
    )
    router = QwenV3FallbackRouter(backend=backend)
    snapshot = SimpleNamespace(
        target_observations={
            "offered_locations": SimpleNamespace(
                value=("North Clinic", "South Clinic")
            ),
            "dose": SimpleNamespace(value="20 mg"),
        },
        caller_truth={"selected_location": SimpleNamespace(value="North Clinic")},
    )

    await router.resolve("Which one do you prefer?", snapshot)

    assert backend.prompt == {
        "clinic_turn": "Which one do you prefer?",
        "grounded_target_context": {
            "offered_locations": ["North Clinic", "South Clinic"],
            "last_observed_dose": "20 mg",
        },
    }
    assert "selected_location" not in json.dumps(backend.prompt)


@pytest.mark.asyncio
async def test_self_pay_acknowledgement_advances_to_locations_semantically():
    router = QwenV3FallbackRouter(
        backend=FakeBackend(
            [
                observation(
                    "acknowledgement",
                    self_pay_location_action="establish_self_pay",
                    target_acknowledges_self_pay=True,
                    extracted_location="none",
                    extracted_locations=["none"],
                    extracted_insurer="none",
                    office_hours="none",
                    hours_location="none",
                )
            ]
        )
    )

    await router.resolve("Okay, we can treat you as self-pay.", object())

    assert router.last_observation["self_pay_location_action"] == "ask_locations"
    assert router.last_observation["extracted_location"] == ""
    assert router.last_observation["extracted_locations"] == ()
    assert router.last_observation["extracted_insurer"] == ""
    assert router.last_observation["office_hours"] == ""
    assert router.last_observation["hours_location"] == ""


@pytest.mark.asyncio
async def test_compact_location_contract_leaves_contextual_office_resolution_to_python():
    router = QwenV3FallbackRouter(
        semantic_domain="self_pay_location",
        backend=FakeBackend(
            [{
                "location_action": "states_office_hours",
                "insurer": "",
                "locations": ["that office"],
                "acknowledged_location": "",
                "states_active_location": False,
                "office_hours": "closes at five",
                "requires_response": False,
            }]
        ),
    )

    await router.resolve("That office closes at five.", object())

    assert router.last_observation["self_pay_location_action"] == "confirm_location"
    assert router.last_observation["extracted_locations"] == ()
    assert router.last_observation["extracted_location"] == ""
    assert router.last_observation["hours_location"] == ""
    assert router.last_observation["office_hours"] == "closes at five"


@pytest.mark.asyncio
async def test_compact_location_contract_treats_literal_insurer_as_target_observation():
    router = QwenV3FallbackRouter(
        semantic_domain="self_pay_location",
        backend=FakeBackend(
            [{
                "location_action": "states_location",
                "insurer": "Acme Plan",
                "locations": [],
                "acknowledged_location": "",
                "states_active_location": False,
                "office_hours": "",
                "requires_response": True,
            }]
        ),
    )

    await router.resolve("Your insurer is still Acme Plan.", object())

    assert router.last_observation["self_pay_location_action"] == "provide_self_pay"
    assert router.last_observation["extracted_insurance_status"] == "specific_insurer"
    assert router.last_observation["extracted_insurer"] == "Acme Plan"


def test_multi_slot_question_is_claimed_by_deterministic_slot_owner():
    controller = SchedulingFlowController()

    # First relax Friday -> earlier weekday afternoons using an already-owned
    # deterministic branch, matching the live workflow state before the offer.
    controller.decide_burst([
        (
            "Would you like me to check afternoon options on a different "
            "day or check with a different provider?"
        )
    ])

    result = controller.decide_burst([
        (
            "I have Thursday at twelve PM, twelve forty five PM, "
            "or one thirty PM. Which works for you?"
        )
    ])

    assert result.decision.kind == DecisionKind.GRANT_PERMISSION
    assert result.decision.reason == "compatible_concrete_slot_offered"
    assert result.after.accepted_slot_text == "twelve PM"
    assert "twelve PM" in result.decision.text
    assert "five PM" not in result.decision.text


def test_multi_slot_question_cue_is_structural_not_exact_phrase():
    controller = SchedulingFlowController()
    controller.decide_burst([
        (
            "Would you like me to check afternoon options on a different "
            "day or check with a different provider?"
        )
    ])

    result = controller.decide_burst([
        (
            "Thursday has twelve PM, twelve forty five PM, or one thirty PM. "
            "Which of those should I use?"
        )
    ])

    assert result.decision.kind == DecisionKind.GRANT_PERMISSION
    assert result.decision.reason == "compatible_concrete_slot_offered"
    assert result.after.accepted_slot_text == "twelve PM"


def test_exact_single_slot_live_offer_is_owned_deterministically():
    controller = SchedulingFlowController()
    controller.tracker.relax_day_constraint_for_afternoon()

    result = controller.decide_burst([
        (
            "The next available week a afternoon slot is Thursday, August "
            "twentieth at twelve PM in Nashville with Doobie Hauser. Would "
            "you like to move your appointment to this time or hear other "
            "Thursday afternoon options?"
        )
    ])

    assert result.decision.kind == DecisionKind.GRANT_PERMISSION
    assert result.decision.reason == "compatible_concrete_slot_offered"
    assert result.after.accepted_slot_text == "twelve PM"
    assert result.decision.text == "Yes, please book the twelve PM slot."
    assert "five PM" not in result.decision.text
    assert result.after.accepted_slot_text in result.decision.text


def test_spoken_pm_parser_supports_full_clock_range_and_minutes():
    controller = SchedulingFlowController()
    controller.tracker.relax_day_constraint_for_afternoon()

    for spoken in ("six PM", "eleven fifteen PM", "twelve forty-five PM"):
        result = controller.decide_burst([
            f"Thursday has an opening at {spoken}. Would that work for you?"
        ])
        assert result.decision.reason == "compatible_concrete_slot_offered"
        assert result.after.accepted_slot_text == spoken
        assert spoken in result.decision.text
