from __future__ import annotations

import pytest

from voiceprobe.v33.semantic_frame import (
    AmbiguityKind,
    ConstraintAxis,
    RecordClaim,
    ReferenceKind,
    SemanticAmbiguity,
    SemanticFrame,
    SemanticTopic,
    SpeechAct,
    TransactionOperation,
    TransactionSignal,
    UnresolvedSemanticFrameError,
)
from voiceprobe.v33.world_model import ObservationKind


def frame(**overrides):
    values = {
        "raw_text": "synthetic remote turn",
        "speech_act": SpeechAct.STATEMENT,
    }
    values.update(overrides)
    return SemanticFrame(**values)


def test_same_day_different_time_represents_change_and_retention_separately():
    semantic = frame(
        speech_act=SpeechAct.OFFER,
        topic=SemanticTopic.AVAILABILITY,
        proposed_changes=(ConstraintAxis.TIME_OF_DAY,),
        retained_constraints=(ConstraintAxis.DAY,),
    )

    assert semantic.search_constraints == ("time_of_day",)
    assert semantic.retained_constraints == (ConstraintAxis.DAY,)
    assert (
        semantic.derive_observation_kind()
        is ObservationKind.ALTERNATIVE_SEARCH_OFFER
    )


def test_same_time_different_day_represents_inverse_minimal_pair():
    semantic = frame(
        speech_act=SpeechAct.OFFER,
        topic=SemanticTopic.AVAILABILITY,
        proposed_changes=(ConstraintAxis.DAY,),
        retained_constraints=(ConstraintAxis.TIME_OF_DAY,),
    )

    assert semantic.search_constraints == ("day",)
    assert (
        semantic.derive_observation_kind()
        is ObservationKind.ALTERNATIVE_SEARCH_OFFER
    )


def test_failed_time_and_proposed_time_are_allowed_to_overlap():
    semantic = frame(
        speech_act=SpeechAct.OFFER,
        topic=SemanticTopic.AVAILABILITY,
        failed_constraints=(ConstraintAxis.TIME_OF_DAY,),
        proposed_changes=(ConstraintAxis.TIME_OF_DAY,),
    )

    assert semantic.unavailable_constraints == ("time_of_day",)
    assert semantic.search_constraints == ("time_of_day",)


def test_changed_and_retained_same_axis_is_rejected():
    with pytest.raises(ValueError, match="both proposed-to-change and retained"):
        frame(
            proposed_changes=(ConstraintAxis.DAY,),
            retained_constraints=(ConstraintAxis.DAY,),
        )


def test_duplicate_constraint_values_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        frame(
            proposed_changes=(
                ConstraintAxis.DAY,
                ConstraintAxis.DAY,
            ),
        )


def test_alternative_search_kind_is_derived_without_respond_boolean():
    semantic = frame(
        speech_act=SpeechAct.STATEMENT,
        topic=SemanticTopic.AVAILABILITY,
        failed_constraints=(ConstraintAxis.TIME_OF_DAY,),
        proposed_changes=(ConstraintAxis.TIME_OF_DAY,),
    )

    assert semantic.requires_response is False
    assert (
        semantic.derive_observation_kind()
        is ObservationKind.ALTERNATIVE_SEARCH_OFFER
    )


def test_availability_only_without_fallback_stays_availability_result():
    semantic = frame(
        speech_act=SpeechAct.STATEMENT,
        topic=SemanticTopic.AVAILABILITY,
        failed_constraints=(
            ConstraintAxis.DAY,
            ConstraintAxis.TIME_OF_DAY,
        ),
    )

    assert (
        semantic.derive_observation_kind()
        is ObservationKind.AVAILABILITY_RESULT
    )
    assert semantic.search_constraints == ()


def test_reschedule_reason_request_is_derived_from_requested_fact():
    semantic = frame(
        speech_act=SpeechAct.QUESTION,
        topic=SemanticTopic.RESCHEDULE_REASON,
        requested_fact="reschedule_reason",
    )

    assert semantic.requires_response is True
    assert (
        semantic.derive_observation_kind()
        is ObservationKind.RESCHEDULE_REASON_REQUEST
    )


def test_generic_fact_request_is_derived_from_requested_fact():
    semantic = frame(
        speech_act=SpeechAct.QUESTION,
        topic=SemanticTopic.PATIENT_FACT,
        requested_fact="insurance",
    )

    assert (
        semantic.derive_observation_kind()
        is ObservationKind.FACT_REQUEST
    )


def test_profile_request_is_derived_from_topic_and_dialogue_act():
    semantic = frame(
        speech_act=SpeechAct.QUESTION,
        topic=SemanticTopic.PROFILE,
    )

    assert (
        semantic.derive_observation_kind()
        is ObservationKind.PROFILE_REQUEST
    )


def test_presence_check_requires_presence_topic():
    with pytest.raises(ValueError, match="PRESENCE"):
        frame(
            speech_act=SpeechAct.PRESENCE_CHECK,
            topic=SemanticTopic.OTHER,
        )

    semantic = frame(
        speech_act=SpeechAct.PRESENCE_CHECK,
        topic=SemanticTopic.PRESENCE,
    )
    assert semantic.requires_response is True
    assert (
        semantic.derive_observation_kind()
        is ObservationKind.PRESENCE_CHECK
    )


def test_record_claims_are_observations_not_patient_truth():
    semantic = frame(
        topic=SemanticTopic.APPOINTMENT_STATE,
        record_claims=(RecordClaim.APPOINTMENT_MISSING,),
    )

    assert semantic.remote_claim_values == ("appointment_missing",)
    observation = semantic.to_remote_observation()
    assert observation.remote_claims == ("appointment_missing",)


def test_transaction_permission_kind_is_derived_from_signal():
    semantic = frame(
        speech_act=SpeechAct.QUESTION,
        topic=SemanticTopic.TRANSACTION,
        transaction_operation=TransactionOperation.RESCHEDULE,
        transaction_signal=TransactionSignal.PERMISSION_REQUEST,
    )

    observation = semantic.to_remote_observation()
    assert (
        observation.kind
        is ObservationKind.TRANSACTION_PERMISSION_REQUEST
    )
    assert observation.requires_response is True
    assert observation.transaction_operation == "reschedule"


def test_transaction_confirmation_is_derived_separately_from_permission():
    semantic = frame(
        speech_act=SpeechAct.CONFIRMATION,
        topic=SemanticTopic.TRANSACTION,
        transaction_operation=TransactionOperation.RESCHEDULE,
        transaction_signal=TransactionSignal.CONFIRMED,
    )

    assert (
        semantic.derive_observation_kind()
        is ObservationKind.TRANSACTION_CONFIRMED
    )


def test_non_none_ambiguity_requires_multiple_candidates():
    with pytest.raises(ValueError, match="at least two"):
        SemanticAmbiguity(
            kind=AmbiguityKind.TEMPORAL_REFERENCE,
            candidates=("day",),
        )


def test_none_ambiguity_cannot_hide_detail():
    with pytest.raises(ValueError, match="require a non-NONE"):
        SemanticAmbiguity(
            detail="maybe a prior option",
        )


def test_ambiguous_reference_requires_explicit_ambiguity():
    with pytest.raises(ValueError, match="explicit SemanticAmbiguity"):
        frame(reference=ReferenceKind.AMBIGUOUS)


def test_unresolved_ambiguity_cannot_be_silently_converted_to_observation():
    semantic = frame(
        speech_act=SpeechAct.QUESTION,
        topic=SemanticTopic.AVAILABILITY,
        reference=ReferenceKind.AMBIGUOUS,
        ambiguity=SemanticAmbiguity(
            kind=AmbiguityKind.TEMPORAL_REFERENCE,
            candidates=("time_of_day", "day"),
            detail="The phrase could refer to either temporal axis.",
        ),
    )

    assert semantic.has_unresolved_ambiguity is True

    with pytest.raises(UnresolvedSemanticFrameError):
        semantic.to_remote_observation()


def test_resolved_prior_option_reference_is_allowed():
    semantic = frame(
        speech_act=SpeechAct.CONFIRMATION,
        topic=SemanticTopic.AVAILABILITY,
        reference=ReferenceKind.PRIOR_OPTION,
        selected_option="Friday morning",
    )

    assert semantic.has_unresolved_ambiguity is False
    assert semantic.selected_option == "Friday morning"


def test_two_axis_offer_can_represent_genuine_both_without_combination_sentinel():
    semantic = frame(
        speech_act=SpeechAct.OFFER,
        topic=SemanticTopic.AVAILABILITY,
        proposed_changes=(
            ConstraintAxis.TIME_OF_DAY,
            ConstraintAxis.DAY,
        ),
    )

    assert semantic.search_constraints == ("time_of_day", "day")
    assert "combination" not in semantic.search_constraints


def test_adapter_preserves_current_remote_observation_contract():
    semantic = frame(
        raw_text="No Friday afternoon openings. Try another time?",
        speech_act=SpeechAct.OFFER,
        topic=SemanticTopic.AVAILABILITY,
        failed_constraints=(
            ConstraintAxis.DAY,
            ConstraintAxis.TIME_OF_DAY,
        ),
        proposed_changes=(ConstraintAxis.TIME_OF_DAY,),
        retained_constraints=(ConstraintAxis.DAY,),
        record_claims=(RecordClaim.APPOINTMENT_EXISTS,),
        transaction_operation=TransactionOperation.SEARCH,
    )

    observation = semantic.to_remote_observation()

    assert observation.kind is ObservationKind.ALTERNATIVE_SEARCH_OFFER
    assert observation.raw_text.startswith("No Friday")
    assert observation.requires_response is True
    assert observation.unavailable_constraints == ("day", "time_of_day")
    assert observation.search_constraints == ("time_of_day",)
    assert observation.remote_claims == ("appointment_exists",)
    assert observation.transaction_operation == "search"
