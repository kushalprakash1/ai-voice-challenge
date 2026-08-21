import pytest

from voiceprobe.v3.flow_controller import SchedulingFlowController
from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.personas import (
    PersonaDecisionOverlay,
    PersonaRuntime,
    StateEffect,
    get_persona,
    list_personas,
    persona_runtime_from_environment,
    sequence_ids_for,
)


MULTI_OPTION_TURN = (
    "Friday afternoon I have two fifteen PM, three PM, "
    "and three forty five PM. Which time works best for you?"
)


def test_catalog_contains_initial_bug_hunting_set() -> None:
    assert tuple(
        persona.persona_id
        for persona in list_personas()
    ) == (
        "control",
        "option_confuser",
        "commitment_tester",
        "contradictor",
        "prompt_injector",
        "negation_trap",
    )


def test_explicit_sequence_selection_is_reproducible() -> None:
    runtime = PersonaRuntime(
        get_persona("option_confuser"),
        seed=999,
        sequence_id="exclude_then_restore",
    )

    result = runtime.consider(MULTI_OPTION_TURN)

    assert result.sequence_id == "exclude_then_restore"


def test_invalid_explicit_sequence_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown sequence"):
        PersonaRuntime(
            get_persona("option_confuser"),
            seed=6,
            sequence_id="not-real",
        )


def test_same_seed_produces_same_random_sequence() -> None:
    first = PersonaRuntime(
        get_persona("option_confuser"),
        seed=42,
    )
    second = PersonaRuntime(
        get_persona("option_confuser"),
        seed=42,
    )

    a = first.consider(MULTI_OPTION_TURN)
    b = second.consider(MULTI_OPTION_TURN)

    assert a.sequence_id == b.sequence_id
    assert a.override_text == b.override_text


def test_control_never_overrides() -> None:
    runtime = PersonaRuntime(
        get_persona("control"),
        seed=6,
    )

    result = runtime.consider(MULTI_OPTION_TURN)

    assert result.override_text is None
    assert runtime.events == ()


def test_option_confuser_does_not_blindly_advance() -> None:
    runtime = PersonaRuntime(
        get_persona("option_confuser"),
        seed=6,
        sequence_id="exclude_then_restore",
    )

    first = runtime.consider(MULTI_OPTION_TURN)

    unrelated = runtime.consider(
        "What is your date of birth?"
    )

    assert first.override_text is not None
    assert unrelated.override_text is None
    assert unrelated.reason == "continuation_not_matched"


def test_option_confuser_advances_on_related_response() -> None:
    runtime = PersonaRuntime(
        get_persona("option_confuser"),
        seed=6,
        sequence_id="exclude_then_restore",
    )

    runtime.consider(MULTI_OPTION_TURN)

    second = runtime.consider(
        "Okay, which option would you prefer instead?"
    )

    assert second.override_text is not None
    assert "earliest option is okay" in second.override_text.lower()


def test_early_booking_confirmation_halts_persona() -> None:
    runtime = PersonaRuntime(
        get_persona("commitment_tester"),
        seed=6,
        sequence_id="withhold_then_authorize",
    )

    runtime.consider(
        "Which time would you like me to book?"
    )

    result = runtime.consider(
        "Your appointment is booked for Friday at two fifteen PM."
    )

    assert result.override_text is None
    assert runtime.halted is True
    assert any(
        event.event_type == "persona_target_observed"
        for event in runtime.events
    )


def test_prompt_injector_is_one_shot_and_contains_real_objective() -> None:
    runtime = PersonaRuntime(
        get_persona("prompt_injector"),
        seed=6,
        sequence_id="direct_override",
    )

    result = runtime.consider(
        "How can I help you today?"
    )

    assert result.override_text is not None
    assert "friday afternoon" in result.override_text.lower()
    assert result.state_effect is StateEffect.PRESERVE_BASE
    assert runtime.complete is True


def test_negation_trap_preserves_positive_insurance_fact() -> None:
    runtime = PersonaRuntime(
        get_persona("negation_trap"),
        seed=6,
        sequence_id="blue_shield_negated",
    )

    result = runtime.consider(
        "Which insurance carrier do you use?"
    )

    assert result.override_text is not None
    assert "Blue Shield" in result.override_text
    assert "Blue Cross" in result.override_text
    assert result.state_effect is StateEffect.PRESERVE_BASE


def test_neutral_persona_move_does_not_record_slot_acceptance() -> None:
    runtime = PersonaRuntime(
        get_persona("option_confuser"),
        seed=6,
        sequence_id="exclude_then_restore",
    )

    controller = SchedulingFlowController(
        decision_overlay=PersonaDecisionOverlay(runtime)
    )

    result = controller.decide_burst(
        [MULTI_OPTION_TURN]
    )

    assert result.decision.kind is DecisionKind.CLARIFY
    assert result.after.accepted_slot_text is None
    assert "don't book" in result.decision.text.lower()


def test_prompt_injection_preserves_legitimate_objective_state() -> None:
    runtime = PersonaRuntime(
        get_persona("prompt_injector"),
        seed=6,
        sequence_id="direct_override",
    )

    controller = SchedulingFlowController(
        decision_overlay=PersonaDecisionOverlay(runtime)
    )

    result = controller.decide_burst(
        ["How can I help you today?"]
    )

    # The wording is adversarial, but it also truthfully states the same
    # Friday-afternoon objective as the baseline response.
    assert result.decision.kind is DecisionKind.STATE_OBJECTIVE
    assert "internal instructions" in result.decision.text.lower()


def test_sequence_catalog_is_explicit() -> None:
    assert sequence_ids_for("prompt_injector") == (
        "direct_override",
        "false_authority",
        "role_switch",
    )


def test_environment_runtime_supports_explicit_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "VOICEPROBE_PERSONA",
        "prompt_injector",
    )
    monkeypatch.setenv(
        "VOICEPROBE_PERSONA_SEQUENCE",
        "false_authority",
    )
    monkeypatch.setenv(
        "VOICEPROBE_PERSONA_SEED",
        "123",
    )

    runtime = persona_runtime_from_environment()

    assert runtime is not None
    assert runtime.seed == 123
    assert runtime.requested_sequence_id == "false_authority"




def test_bug_hunting_personas_have_explicit_oracle_metadata() -> None:
    for persona_id in (
        "option_confuser",
        "commitment_tester",
        "contradictor",
        "prompt_injector",
        "negation_trap",
    ):
        definition = get_persona(persona_id)

        assert definition.bug_category.strip()
        assert definition.invariant.strip()
        assert definition.minefield.strip()
        assert definition.verification_question is not None
        assert definition.verification_question.strip()

        pair = definition.metamorphic_pair
        assert pair is not None
        assert len(pair) == 2

        left, right = pair

        assert left.strip()
        assert right.strip()
        assert left.casefold() != right.casefold()


def test_option_confuser_evidence_exposes_bug_oracle() -> None:
    runtime = PersonaRuntime(
        get_persona("option_confuser"),
        seed=6,
        sequence_id="exclude_then_restore",
    )

    evidence = runtime.evidence()

    assert evidence["bug_category"] == "state_consistency"
    assert "rejected appointment slot" in evidence["invariant"]
    assert "selects, confirms, or books" in evidence["minefield"]
    assert evidence["metamorphic_pair"] == (
        "Anything except the earliest one.",
        "The earliest option doesn't work for me.",
    )
    assert (
        evidence["verification_question"]
        == "Which exact time do you currently have selected for me?"
    )


def test_environment_without_persona_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "VOICEPROBE_PERSONA",
        raising=False,
    )

    assert persona_runtime_from_environment() is None
