import pytest

from voiceprobe.v3.flow_controller import SchedulingFlowController
from voiceprobe.v3.flow_state import SchedulingFlowTracker, StateProvenance


def test_target_dose_observation_does_not_mutate_caller_truth() -> None:
    tracker = SchedulingFlowTracker(caller_truth={"dose": "10 mg"})

    snapshot = tracker.observe_target_value(
        "dose", "20 mg", evidence="I have you taking twenty milligrams."
    )

    assert snapshot.caller_truth["dose"].value == "10 mg"
    assert snapshot.caller_truth["dose"].provenance is StateProvenance.CALLER_SCENARIO
    assert snapshot.target_observations["dose"].value == "20 mg"
    assert snapshot.target_observations["dose"].provenance is StateProvenance.TARGET_OBSERVED
    assert "dose" not in snapshot.committed_dialogue


def test_new_target_observation_supersedes_only_the_old_observation() -> None:
    tracker = SchedulingFlowTracker(caller_truth={"dose": "10 mg"})
    tracker.observe_target_value("dose", "20 mg", evidence="first claim")

    snapshot = tracker.observe_target_value("dose", "40 mg", evidence="new claim")

    assert snapshot.target_observations["dose"].value == "40 mg"
    assert snapshot.target_observations["dose"].evidence == "new claim"
    assert snapshot.caller_truth["dose"].value == "10 mg"


def test_unknown_target_field_cannot_silently_become_caller_truth() -> None:
    tracker = SchedulingFlowTracker()

    snapshot = tracker.observe_target_value("unrecognized_claim", "surprise")

    assert "unrecognized_claim" in snapshot.target_observations
    assert "unrecognized_claim" not in snapshot.caller_truth
    assert "unrecognized_claim" not in snapshot.committed_dialogue


def test_selected_slot_is_grounded_and_matches_spoken_slot() -> None:
    controller = SchedulingFlowController()

    result = controller.decide_burst(
        ["I can offer Friday at 2:30 PM. Would that work for you?"]
    )

    offered = result.after.target_observations["offered_slot"]
    selected = result.after.committed_dialogue["selected_slot"]
    assert offered.value == selected.value == result.after.accepted_slot_text
    assert selected.value in result.decision.text
    assert selected.provenance is StateProvenance.COMMITTED_VALIDATED


def test_selected_slot_cannot_be_committed_without_matching_offer() -> None:
    tracker = SchedulingFlowTracker()
    tracker.observe_target_value("offered_slot", "2:30 PM")

    with pytest.raises(ValueError, match="must match target observation"):
        tracker.record_slot_acceptance("3:15 PM")
