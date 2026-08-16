#!/usr/bin/env python3
"""Verify v3 fallback nodes against the installed Pipecat NodeConfig type."""

from __future__ import annotations

from pipecat.flows import NodeConfig

from voiceprobe.v3.flow_nodes import build_fallback_node
from voiceprobe.v3.flow_state import SchedulingFlowTracker


def main() -> None:
    node = build_fallback_node(
        SchedulingFlowTracker().snapshot()
    )

    # NodeConfig is a TypedDict, so runtime compatibility is structural.
    required = set(NodeConfig.__required_keys__)
    missing = required.difference(node)

    print("NodeConfig required keys:", sorted(required))
    print("Generated node keys:", sorted(node))

    if missing:
        raise SystemExit(
            f"Generated fallback node is missing required keys: {sorted(missing)}"
        )

    print("Pipecat NodeConfig structural compatibility: PASS")
    print("Initial node:", node["name"])


if __name__ == "__main__":
    main()
