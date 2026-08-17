#!/usr/bin/env python3

"""Offline PipecatRuntimeBridge replay with v3.2 opt-in.

No Flux websocket, Asterisk channel, or phone call is opened.
"""

from __future__ import annotations

import asyncio

from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.production import (
    PipecatRuntimeBridge,
)


class FakeSpeechFrame:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeWorker:
    def __init__(self) -> None:
        self.frames = []

    async def queue_frames(self, frames) -> None:
        self.frames.extend(frames)


async def main() -> None:
    bridge = PipecatRuntimeBridge(
        tts_frame_factory=FakeSpeechFrame,
    )

    if bridge.semantic_mode != "v32":
        raise SystemExit(
            "REFUSING: production bridge is not in v32 mode"
        )

    worker = FakeWorker()
    bridge.bind_worker(worker)

    cases = (
        (
            "deterministic_open_intent",
            "How may I help you today?",
            DecisionKind.STATE_OBJECTIVE,
            True,
        ),
        (
            "deterministic_ack",
            "No problem.",
            DecisionKind.WAIT,
            False,
        ),
        (
            "semantic_reschedule_reason",
            "What is the reason for changing your appointment?",
            DecisionKind.CONTEXTUAL_ANSWER,
            True,
        ),
        (
            "semantic_insurance",
            "Which company underwrites your health coverage?",
            DecisionKind.ANSWER_FACT,
            True,
        ),
        (
            "semantic_visit_reason",
            "What medical issue prompted this visit?",
            DecisionKind.ANSWER_COMPLAINT,
            True,
        ),
        (
            "semantic_unknown",
            "Could you unpack the metaphysics of this appointment?",
            DecisionKind.CLARIFY,
            True,
        ),
    )

    passed = 0

    print(
        "========== V3.2 PRODUCTION-BRIDGE REPLAY =========="
    )
    print("semantic_mode=", bridge.semantic_mode)

    for index, (
        name,
        utterance,
        expected_kind,
        expect_speech,
    ) in enumerate(cases, 1):

        before = (
            bridge.runtime.flow_controller
            .tracker
            .snapshot()
        )

        frames_before = len(
            worker.frames
        )

        result = await bridge.runtime.process_turns(
            [utterance],
            ingress_reason=(
                "v32_production_bridge_replay"
            ),
        )

        frames_after = len(
            worker.frames
        )

        speech_delta = (
            frames_after
            - frames_before
        )

        after = result.after

        transaction_safe = (
            before.accepted_slot_text
            == after.accepted_slot_text
            and before.booking_confirmation_text
            == after.booking_confirmation_text
            and before.complete
            == after.complete
        )

        speech_ok = (
            speech_delta == 1
            if expect_speech
            else speech_delta == 0
        )

        ok = (
            result.decision.kind
            is expected_kind
            and speech_ok
            and transaction_safe
        )

        passed += int(ok)

        print()
        print(f"[{index}] {name}")
        print(" PGAI:", utterance)
        print(
            " route=",
            result.route.value,
        )
        print(
            " decision_kind=",
            result.decision.kind.value,
        )
        print(
            " decision_reason=",
            result.decision.reason,
        )
        print(
            " decision_text=",
            result.decision.text,
        )
        print(
            " policy_latency_ms=",
            round(
                result.policy_latency_ms,
                1,
            ),
        )
        print(
            " speech_frames_added=",
            speech_delta,
        )
        print(
            " transaction_state_unchanged=",
            transaction_safe,
        )
        print(" PASS=", ok)

    snapshot = (
        bridge.runtime.flow_controller
        .tracker
        .snapshot()
    )

    print()
    print("========== SUMMARY ==========")
    print(
        f"correct={passed}/{len(cases)}"
    )
    print(
        "semantic_mode=",
        bridge.semantic_mode,
    )
    print(
        "total_speech_frames=",
        len(worker.frames),
    )
    print(
        "fallback_decisions=",
        bridge.runtime.metrics.fallback_decisions,
    )
    print(
        "accepted_slot_text=",
        snapshot.accepted_slot_text,
    )
    print(
        "booking_confirmation_text=",
        snapshot.booking_confirmation_text,
    )
    print(
        "complete=",
        snapshot.complete,
    )
    print(
        f"failed={len(cases) - passed}"
    )

    if passed != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
