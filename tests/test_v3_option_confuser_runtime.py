import asyncio

from voiceprobe.v3.fast_policy import RoutineSchedulingPolicy
from voiceprobe.v3.flow_controller import SchedulingFlowController
from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.personas import (
    PersonaDecisionOverlay,
    PersonaRuntime,
    get_persona,
)
from voiceprobe.v3.production import safe_production_fallback_resolver
from voiceprobe.v3.runtime import DecisionRoute, VoiceProbeV3Runtime


EXISTING_APPOINTMENT = (
    "You already have a new patient consultation booked for Tuesday, "
    "August twenty fifth at two fifteen PM. Would you like to keep this "
    "appointment, reschedule it to a different time, or cancel it?"
)

MULTIPLE_OPTIONS = (
    "Friday afternoon I have two fifteen PM, three PM, and three forty five PM. "
    "Which time works best for you?"
)


def test_selected_slot_verification_is_normal_slot_acceptance() -> None:
    decision = RoutineSchedulingPolicy().decide(
        "I have two fifteen PM selected. Is that correct?"
    )

    assert decision.kind is DecisionKind.GRANT_PERMISSION
    assert decision.reason == "compatible_concrete_slot_offered"
    assert "book" in decision.text.casefold()


def test_selected_nonfriday_slot_remains_incompatible() -> None:
    decision = RoutineSchedulingPolicy().decide(
        "I have Tuesday at two fifteen PM selected. Is that correct?"
    )

    assert decision.kind is DecisionKind.DECLINE_INCOMPATIBLE_OFFER
    assert "friday afternoon" in decision.text.casefold()


def test_option_confuser_full_production_path_has_no_fallback() -> None:
    async def scenario() -> None:
        persona = PersonaRuntime(
            get_persona("option_confuser"),
            seed=6,
            sequence_id="exclude_then_restore",
        )

        runtime = VoiceProbeV3Runtime(
            flow_controller=SchedulingFlowController(
                decision_overlay=PersonaDecisionOverlay(persona)
            ),
            fallback_resolver=safe_production_fallback_resolver,
        )

        turns = [
            EXISTING_APPOINTMENT,
            MULTIPLE_OPTIONS,
            "Okay. Which option would you prefer instead?",
            "Would you like me to book that appointment?",
            "I have two fifteen PM selected. Is that correct?",
        ]

        results = []

        for turn in turns:
            results.append(
                await runtime.process_turns([turn])
            )

        # Existing historical appointment must not complete this call.
        assert not results[0].after.complete
        assert (
            results[0].decision.reason
            == "existing_appointment_reschedule"
        )

        # Actual adversarial sequence must activate and finish.
        assert persona.selected_sequence_id == "exclude_then_restore"
        assert persona.complete

        # Nothing in the adversarial path should escape to the fallback.
        assert all(
            result.route is DecisionRoute.DETERMINISTIC
            for result in results
        )

        assert all(
            result.decision.reason
            != "production_safe_clarification"
            for result in results
        )

        final = results[-1]

        # Final persona move preserves the ordinary slot-acceptance effect.
        assert final.decision.kind is DecisionKind.GRANT_PERMISSION
        assert (
            "you can book it"
            in final.decision.text.casefold()
        )

        assert final.after.accepted_slot_text is not None
        assert (
            "two fifteen"
            in final.after.accepted_slot_text.casefold()
        )

        # Permission has been given, but PGAI has not yet confirmed execution.
        assert not final.after.complete

        confirmed = await runtime.process_turns(
            [
                (
                    "Great, your appointment is booked for Friday "
                    "at two fifteen PM."
                )
            ]
        )

        assert confirmed.decision.kind is DecisionKind.WAIT
        assert confirmed.after.complete
        assert confirmed.after.booking_confirmation_text is not None

    asyncio.run(scenario())
