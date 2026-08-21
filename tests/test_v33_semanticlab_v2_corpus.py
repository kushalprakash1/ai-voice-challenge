from __future__ import annotations

from collections import Counter

from voiceprobe.v33.semantic_corpus import load_semanticlab_cases


def test_semanticlab_v2_corpus_loads_and_has_unique_ids():
    cases = load_semanticlab_cases()
    assert len(cases) >= 130
    assert len({case.case_id for case in cases}) == len(cases)


def test_required_coverage_categories_are_present():
    cases = load_semanticlab_cases()
    counts = Counter(case.category for case in cases)

    minimums = {
        "temporal_minimal_pair": 24,
        "availability_fallback": 18,
        "multi_turn_reference": 18,
        "asr_noise": 15,
        "record_state": 12,
        "transaction_consent": 12,
        "ambiguity_oos": 12,
        "compound_multi_intent": 12,
        "core_controls": 10,
    }

    for category, minimum in minimums.items():
        assert counts[category] >= minimum, (category, counts[category], minimum)


def test_critical_cases_are_a_substantial_fraction():
    cases = load_semanticlab_cases()
    critical = [case for case in cases if "critical" in case.tags]
    assert len(critical) >= 75


def test_multi_turn_cases_really_contain_context():
    cases = load_semanticlab_cases()
    refs = [case for case in cases if case.category == "multi_turn_reference"]
    assert refs
    assert all(case.context for case in refs)


def test_asr_noise_is_explicitly_tagged():
    cases = load_semanticlab_cases()
    noise = [case for case in cases if case.category == "asr_noise"]
    assert len(noise) >= 15
    assert all("asr_noise" in case.tags for case in noise)


def test_ambiguity_cases_expose_multiple_candidates():
    cases = load_semanticlab_cases()
    ambiguous = [case for case in cases if "ambiguity" in case.tags]
    assert len(ambiguous) >= 12

    for case in ambiguous:
        ambiguity = case.expected.get("ambiguity")
        assert ambiguity
        assert ambiguity["kind"] != "none"
        assert len(ambiguity["candidates"]) >= 2


def test_all_constraint_axes_are_exercised_as_changes():
    cases = load_semanticlab_cases()
    seen = {
        axis
        for case in cases
        for axis in case.expected.get("proposed_changes", ())
    }
    assert {"day", "time_of_day", "provider"} <= seen


def test_day_and_time_are_exercised_as_explicit_retained_axes():
    cases = load_semanticlab_cases()
    seen = {
        axis
        for case in cases
        for axis in case.expected.get("retained_constraints", ())
    }
    assert {"day", "time_of_day"} <= seen


def test_all_constraint_axes_are_exercised_as_failures():
    cases = load_semanticlab_cases()
    seen = {
        axis
        for case in cases
        for axis in case.expected.get("failed_constraints", ())
    }
    assert {"day", "time_of_day", "provider"} <= seen


def test_temporal_minimal_pairs_include_both_inverse_directions():
    cases = load_semanticlab_cases()
    temporal = [
        case for case in cases
        if case.category == "temporal_minimal_pair"
    ]

    same_day_change_time = [
        case for case in temporal
        if case.expected.get("proposed_changes") == ["time_of_day"]
        and case.expected.get("retained_constraints") == ["day"]
    ]
    same_time_change_day = [
        case for case in temporal
        if case.expected.get("proposed_changes") == ["day"]
        and case.expected.get("retained_constraints") == ["time_of_day"]
    ]

    assert len(same_day_change_time) >= 6
    assert len(same_time_change_day) >= 6


def test_transaction_suite_covers_proposed_permission_and_confirmed():
    cases = load_semanticlab_cases()
    tx = [case for case in cases if case.category == "transaction_consent"]
    signals = {
        case.expected.get("transaction_signal")
        for case in tx
    }
    assert {"proposed", "permission_request", "confirmed", "none"} <= signals


def test_record_suite_covers_all_record_claim_values():
    cases = load_semanticlab_cases()
    claims = {
        claim
        for case in cases
        for claim in case.expected.get("record_claims", ())
    }
    assert {
        "profile_exists",
        "profile_missing",
        "appointment_exists",
        "appointment_missing",
    } <= claims


def test_corpus_is_labels_only_and_contains_no_patient_strategy():
    cases = load_semanticlab_cases()
    forbidden = {
        "action",
        "actions",
        "patient_action",
        "utterance_to_say",
        "response_text",
        "mission_preference",
        "patient_preference",
    }

    for case in cases:
        assert not (forbidden & set(case.expected))


def test_compound_suite_contains_multi_signal_frames():
    cases = load_semanticlab_cases()
    compound = [case for case in cases if case.category == "compound_multi_intent"]

    def signal_count(case):
        exp = case.expected
        return sum(
            bool(exp.get(key))
            for key in (
                "requested_fact",
                "failed_constraints",
                "proposed_changes",
                "offered_options",
                "record_claims",
                "transaction_signal",
            )
        )

    assert sum(signal_count(case) >= 2 for case in compound) >= 8
