from voiceprobe.v3.flow_nodes import build_fallback_node
from voiceprobe.v3.flow_state import FlowStage, SchedulingFlowTracker


def test_fallback_node_is_nodeconfig_compatible_shape() -> None:
    node = build_fallback_node(
        SchedulingFlowTracker().snapshot()
    )

    assert node["name"] == "voiceprobe_profile"
    assert isinstance(node["role_message"], str)
    assert node["task_messages"]
    assert node["functions"] == []


def test_fallback_node_changes_with_flow_stage() -> None:
    tracker = SchedulingFlowTracker()
    tracker.observe_remote_turn(
        "Your demo patient profile is set up."
    )

    node = build_fallback_node(
        tracker.snapshot()
    )

    # DOB remains the first uncommunicated stage because profile/identity were
    # confirmed by the remote agent.
    assert tracker.snapshot().current_stage == FlowStage.DOB
    assert node["name"] == "voiceprobe_dob"
