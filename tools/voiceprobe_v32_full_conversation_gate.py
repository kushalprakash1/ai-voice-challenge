#!/usr/bin/env python3

"""Complete offline VoiceProbe v3.2 conversation gate.

This uses:
- production PipecatRuntimeBridge
- deterministic v3 policy
- v3.2 semantic fallback
- Option Confuser persona overlay
- fake speech sink

It opens no Flux websocket, Asterisk channel, or phone call.
"""

from __future__ import annotations

import asyncio

from voiceprobe.v3.flow_controller import (
    SchedulingFlowController,
)
from voiceprobe.v3.personas import (
    PersonaDecisionOverlay,
    PersonaRuntime,
    get_persona,
)
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


TURNS = (
    (
        "existing_appointment",
        (
            "You already have an appointment Tuesday, "
            "August 25th at 2:15 PM. Would you like to "
            "keep it, reschedule it, or cancel it?"
        ),
    ),
    (
        "reschedule_reason",
        "What is the reason for changing your appointment?",
    ),
    (
        "provider_preference",
        (
            "Do you have a provider preference, "
            "or are you okay with the first available provider?"
        ),
    ),
    (
        "multiple_options",
        (
            "Friday afternoon I have two fifteen PM, "
            "three PM, and three forty five PM. "
            "Which time works best for you?"
        ),
    ),
    (
        "option_selected",
        "Okay. I have the option you mentioned selected.",
    ),
    (
        "booking_permission",
        "Would you like me to book that appointment?",
    ),
    (
        "exact_time_verification",
        "I have two fifteen PM selected. Is that correct?",
    ),
    (
        "remote_booking_confirmation",
        (
            "Your appointment is confirmed for Friday, "
            "August 28th at 2:15 PM."
        ),
    ),
)


async def main() -> None:
    persona = PersonaRuntime(
        get_persona("option_confuser"),
        seed=6,
        sequence_id="exclude_then_restore",
    )

    controller = SchedulingFlowController(
        decision_overlay=PersonaDecisionOverlay(
            persona
        )
    )

    bridge = PipecatRuntimeBridge(
        flow_controller=controller,
        tts_frame_factory=FakeSpeechFrame,
    )

    if bridge.semantic_mode != "v32":
        raise SystemExit(
            "REFUSING: bridge must be explicitly in v32 mode"
        )

    worker = FakeWorker()
    bridge.bind_worker(worker)

    resolver = bridge.v32_semantic_resolver

    if resolver is None:
        raise SystemExit(
            "REFUSING: v32 semantic resolver is missing"
        )

    print(
        "========== V3.2 COMPLETE CONVERSATION GATE =========="
    )
    print("semantic_mode=", bridge.semantic_mode)
    print(
        "persona=option_confuser/exclude_then_restore"
    )

    results = []

    for index, (name, utterance) in enumerate(
        TURNS,
        1,
    ):
        frames_before = len(worker.frames)

        before = (
            bridge.runtime.flow_controller
            .tracker
            .snapshot()
        )

        result = await bridge.runtime.process_turns(
            [utterance],
            ingress_reason=(
                "v32_full_conversation_gate"
            ),
        )

        after = result.after

        speech_delta = (
            len(worker.frames)
            - frames_before
        )

        results.append(result)

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
            " PATIENT:",
            result.decision.text
            if result.decision.text
            else "<silence>",
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
            " accepted_slot_before=",
            before.accepted_slot_text,
        )
        print(
            " accepted_slot_after=",
            after.accepted_slot_text,
        )
        print(
            " booking_confirmation=",
            after.booking_confirmation_text,
        )
        print(
            " complete=",
            after.complete,
        )

    # --------------------------------------------------------
    # Hard invariants
    # --------------------------------------------------------

    # Existing appointment must not be mistaken for completion.
    assert (
        results[0].after.accepted_slot_text
        is None
    )
    assert not results[0].after.complete

    # Ordinary reschedule reason is solved contextually.
    assert (
        results[1].decision.kind.value
        == "contextual_answer"
    )
    assert (
        "Friday afternoon"
        in results[1].decision.text
    )

    # First three adversarial moves withhold booking permission.
    for result in (
        results[3],
        results[4],
        results[5],
    ):
        assert (
            result.after.accepted_slot_text
            is None
        )
        assert not result.after.complete

    assert (
        results[3].decision.text
        == (
            "Anything except the earliest one, "
            "but don't book anything yet."
        )
    )

    assert (
        results[4].decision.text
        == (
            "Actually, the earliest option is okay "
            "after all, but please don't book it yet."
        )
    )

    assert (
        results[5].decision.text
        == "Which exact time are you about to book?"
    )

    # Explicit final authorization is the first point where
    # the selected slot may become durable.
    assert (
        results[6].decision.text
        == (
            "Yes, that's the exact time I want. "
            "You can book it."
        )
    )

    assert (
        results[6].after.accepted_slot_text
        is not None
    )

    assert not results[6].after.complete

    # Completion requires explicit REMOTE confirmation.
    final = results[7].after

    assert final.complete
    assert final.accepted_slot_text is not None
    assert (
        final.booking_confirmation_text
        is not None
    )

    # Exactly one turn in this frozen conversation should need
    # semantic fallback: the reschedule-reason question.
    assert (
        bridge.runtime.metrics.fallback_decisions
        == 1
    )

    # Persona sequence itself completed normally.
    assert persona.complete
    assert not persona.halted

    evidence = persona.evidence()

    moves = [
        event
        for event in evidence["events"]
        if event["event_type"] == "persona_move"
    ]

    assert len(moves) == 4

    # Seven patient responses:
    # all except the final confirmation.
    assert len(worker.frames) == 7

    # Runtime observation should include actual persona speech,
    # not only baseline deterministic wording.
    history = resolver.history

    assert any(
        "Anything except the earliest one"
        in line
        for line in history
    )

    assert any(
        "Yes, that's the exact time I want"
        in line
        for line in history
    )

    print()
    print("========== FINAL INVARIANTS ==========")
    print(
        "persona_complete=",
        persona.complete,
    )
    print(
        "persona_halted=",
        persona.halted,
    )
    print(
        "persona_moves=",
        len(moves),
    )
    print(
        "fallback_decisions=",
        bridge.runtime.metrics.fallback_decisions,
    )
    print(
        "speech_frames=",
        len(worker.frames),
    )
    print(
        "accepted_slot_text=",
        final.accepted_slot_text,
    )
    print(
        "booking_confirmation_text=",
        final.booking_confirmation_text,
    )
    print(
        "complete=",
        final.complete,
    )

    print()
    print(
        "V32_COMPLETE_CONVERSATION_GATE=PASS"
    )


if __name__ == "__main__":
    asyncio.run(main())
