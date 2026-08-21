from __future__ import annotations

from voiceprobe.v33.semantic_baseline_eval import (
    EvalStatus,
    evaluate_v017_case,
)
from voiceprobe.v33.semantic_corpus import SemanticLabCase
from voiceprobe.v33.world_model import ObservationKind, RemoteObservation


def case(expected):
    return SemanticLabCase(
        case_id="synthetic",
        category="test",
        utterance="synthetic",
        context=(),
        expected=expected,
        tags=(),
    )


def observation(**overrides):
    values = {
        "kind": ObservationKind.OTHER,
        "raw_text": "synthetic",
        "requires_response": False,
        "requested_fact": "",
        "offered_options": (),
        "unavailable_constraints": (),
        "remote_claims": (),
        "selected_option": "",
        "transaction_operation": "",
        "search_constraints": (),
    }
    values.update(overrides)
    return RemoteObservation(**values)


def by_field(result):
    return {item.field: item for item in result.fields}


def test_supported_axes_are_scored_exactly():
    result = evaluate_v017_case(
        case(
            {
                "failed_constraints": ["day", "time_of_day"],
                "proposed_changes": ["time_of_day"],
            }
        ),
        observation(
            kind=ObservationKind.ALTERNATIVE_SEARCH_OFFER,
            unavailable_constraints=("time_of_day", "day"),
            search_constraints=("time_of_day",),
        ),
    )

    fields = by_field(result)
    assert fields["failed_constraints"].status is EvalStatus.PASS
    assert fields["proposed_changes"].status is EvalStatus.PASS
    assert result.exact_on_supported_fields is True


def test_axis_mismatch_is_a_real_model_failure():
    result = evaluate_v017_case(
        case({"proposed_changes": ["day"]}),
        observation(search_constraints=("time_of_day",)),
    )

    field = by_field(result)["proposed_changes"]
    assert field.status is EvalStatus.FAIL
    assert result.exact_on_supported_fields is False


def test_retained_constraint_is_reported_as_architecture_gap_not_failure():
    result = evaluate_v017_case(
        case({"retained_constraints": ["day"]}),
        observation(),
    )

    field = by_field(result)["retained_constraints"]
    assert field.status is EvalStatus.UNSUPPORTED
    assert result.exact_on_supported_fields is None


def test_reference_and_ambiguity_are_explicitly_unsupported():
    result = evaluate_v017_case(
        case(
            {
                "reference": "prior_option",
                "ambiguity": {
                    "kind": "option_reference",
                    "candidates": ["A", "B"],
                },
            }
        ),
        observation(),
    )

    fields = by_field(result)
    assert fields["reference"].status is EvalStatus.UNSUPPORTED
    assert fields["ambiguity"].status is EvalStatus.UNSUPPORTED


def test_permission_and_confirmation_can_be_scored_through_legacy_kind():
    permission = evaluate_v017_case(
        case({"transaction_signal": "permission_request"}),
        observation(kind=ObservationKind.TRANSACTION_PERMISSION_REQUEST),
    )
    assert by_field(permission)["transaction_signal"].status is EvalStatus.PASS

    confirmed = evaluate_v017_case(
        case({"transaction_signal": "confirmed"}),
        observation(kind=ObservationKind.TRANSACTION_CONFIRMED),
    )
    assert by_field(confirmed)["transaction_signal"].status is EvalStatus.PASS


def test_transaction_proposal_signal_is_not_falsely_scored():
    result = evaluate_v017_case(
        case({"transaction_signal": "proposed"}),
        observation(transaction_operation="reschedule"),
    )

    field = by_field(result)["transaction_signal"]
    assert field.status is EvalStatus.UNSUPPORTED


def test_none_transaction_operation_normalizes_empty_legacy_value():
    result = evaluate_v017_case(
        case({"transaction_operation": "none"}),
        observation(transaction_operation=""),
    )

    assert by_field(result)["transaction_operation"].status is EvalStatus.PASS


def test_text_options_compare_case_insensitively_and_ignore_whitespace():
    result = evaluate_v017_case(
        case({"offered_options": ["Friday at 3 PM"]}),
        observation(offered_options=("  friday   AT 3 pm ",)),
    )

    assert by_field(result)["offered_options"].status is EvalStatus.PASS
