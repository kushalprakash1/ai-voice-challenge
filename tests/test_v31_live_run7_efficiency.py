from voiceprobe.v3.fast_policy import RoutineSchedulingPolicy
from voiceprobe.v3.flow_controller import SchedulingFlowController
from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.personas import (
    PersonaDecisionOverlay,
    PersonaRuntime,
    get_persona,
)


LIVE_PROVIDER_1 = (
    "Thanks for sharing. Do you have a preferred provider, or are you open "
    "to any available provider for your Friday afternoon appointment?"
)

LIVE_PROVIDER_2 = (
    "Of course. Is there a specific provider you'd like to see, or would "
    "you like to see whoever is available Friday afternoon?"
)

LIVE_PROVIDER_3 = (
    "you wanna see a certain provider, or is anyone available on Friday "
    "afternoon okay for your appointment?"
)

LIVE_MULTI_OPTIONS = (
    "We have Friday afternoon openings with two providers. Would you like "
    "to see doctor Zamiel Locoski or doctor Kelly Noble, or do you want "
    "the soonest available with either?"
)

LIVE_MULTI_TIMES = (
    "For example, doctor has times at one PM, one fifteen, one thirty, "
    "and one forty five. Doctor Noble has three PM, three fifteen, and "
    "three thirty. Which works best for you?"
)


def test_exact_live_provider_variants_answer_without_clarification() -> None:
    policy = RoutineSchedulingPolicy()

    for turn in (
        LIVE_PROVIDER_1,
        LIVE_PROVIDER_2,
        LIVE_PROVIDER_3,
    ):
        decision = policy.decide(turn)

        assert decision.kind is DecisionKind.ANSWER_PROVIDER_PREFERENCE
        assert decision.reason == "provider_preference_requested"
        assert decision.text == "First available is fine."


def test_standalone_no_problem_is_silent_wait() -> None:
    decision = RoutineSchedulingPolicy().decide("No problem.")

    assert decision.kind is DecisionKind.WAIT
    assert decision.reason == "standalone_acknowledgement"
    assert not decision.text


def test_acknowledgement_plus_real_question_is_not_swallowed() -> None:
    decision = RoutineSchedulingPolicy().decide(
        "No problem. Is there a specific provider you'd like to see, "
        "or would you like whoever is available?"
    )

    assert decision.kind is DecisionKind.ANSWER_PROVIDER_PREFERENCE
    assert decision.text == "First available is fine."


def test_live_short_availability_fragment_is_hold() -> None:
    decision = RoutineSchedulingPolicy().decide(
        "Here are some Friday afternoon"
    )

    assert decision.kind is DecisionKind.HOLD
    assert decision.reason == "obvious_incomplete_asr_fragment"
    assert not decision.text


def test_active_persona_ignores_ack_and_fragment_without_aborting() -> None:
    persona = PersonaRuntime(
        get_persona("option_confuser"),
        seed=6,
        sequence_id="exclude_then_restore",
    )

    controller = SchedulingFlowController(
        decision_overlay=PersonaDecisionOverlay(persona)
    )

    activated = controller.decide_burst(
        [LIVE_MULTI_OPTIONS, LIVE_MULTI_TIMES]
    )

    assert activated.decision.kind is DecisionKind.CLARIFY
    assert "anything except the earliest" in activated.decision.text.casefold()
    assert persona.active
    assert not persona.halted

    events_after_move_one = len(persona.events)

    ack = controller.decide_burst(["No problem."])

    assert ack.decision.kind is DecisionKind.WAIT
    assert not ack.decision.text
    assert persona.active
    assert not persona.halted
    assert len(persona.events) == events_after_move_one

    fragment = controller.decide_burst(
        ["Here are some Friday afternoon"]
    )

    assert fragment.decision.kind is DecisionKind.HOLD
    assert not fragment.decision.text
    assert persona.active
    assert not persona.halted
    assert len(persona.events) == events_after_move_one

    substantive = controller.decide_burst(
        ["Okay, which option would you prefer instead?"]
    )

    assert substantive.decision.kind is DecisionKind.CLARIFY
    assert "earliest option is okay" in substantive.decision.text.casefold()
    assert persona.active
    assert not persona.halted


def test_live_predivergence_path_has_no_wasted_provider_turns() -> None:
    persona = PersonaRuntime(
        get_persona("option_confuser"),
        seed=6,
        sequence_id="exclude_then_restore",
    )

    controller = SchedulingFlowController(
        decision_overlay=PersonaDecisionOverlay(persona)
    )

    provider_1 = controller.decide_burst([LIVE_PROVIDER_1])
    provider_2 = controller.decide_burst([LIVE_PROVIDER_2])
    provider_3 = controller.decide_burst([LIVE_PROVIDER_3])

    for result in (provider_1, provider_2, provider_3):
        assert result.decision.kind is DecisionKind.ANSWER_PROVIDER_PREFERENCE
        assert result.decision.text == "First available is fine."
        assert "rephrase" not in result.decision.text.casefold()

    attack = controller.decide_burst(
        [LIVE_MULTI_OPTIONS, LIVE_MULTI_TIMES]
    )

    assert "don't book anything yet" in attack.decision.text.casefold()

    ack = controller.decide_burst(["No problem."])
    fragment = controller.decide_burst(
        ["Here are some Friday afternoon"]
    )

    assert ack.decision.kind is DecisionKind.WAIT
    assert fragment.decision.kind is DecisionKind.HOLD

    # Most important regression: neither benign remote turn may cause the
    # alternate-day search that derailed the real call.
    assert "another weekday" not in ack.decision.text.casefold()
    assert "another weekday" not in fragment.decision.text.casefold()

    assert persona.active
    assert not persona.halted


def test_live_two_provider_choice_with_soonest_option_is_deterministic() -> None:
    decision = RoutineSchedulingPolicy().decide(
        (
            "We have Friday afternoon openings with two providers. "
            "Would you like to see doctor Zamiel Locoski or doctor Kelly Noble, "
            "or do you want the soonest available with either?"
        )
    )

    assert decision.kind is DecisionKind.ANSWER_PROVIDER_PREFERENCE
    assert decision.reason == "provider_preference_requested"
    assert decision.text == "First available is fine."


def test_live_provider_intro_and_time_list_coalesce_without_fallback() -> None:
    persona = PersonaRuntime(
        get_persona("option_confuser"),
        seed=6,
        sequence_id="exclude_then_restore",
    )

    controller = SchedulingFlowController(
        decision_overlay=PersonaDecisionOverlay(persona)
    )

    result = controller.decide_burst(
        [
            (
                "We have Friday afternoon openings with two providers. "
                "Would you like to see doctor Zamiel Locoski or doctor Kelly "
                "Noble, or do you want the soonest available with either?"
            ),
            (
                "For example, doctor has times at one PM, one fifteen, "
                "one thirty, and one forty five. Doctor Noble has three PM, "
                "three fifteen, and three thirty. Which works best for you?"
            ),
        ]
    )

    assert result.decision.kind is DecisionKind.CLARIFY
    assert "anything except the earliest" in result.decision.text.casefold()
    assert result.decision.reason.startswith(
        "persona:option_confuser:exclude_then_restore:"
    )
    assert persona.active
    assert not persona.halted
