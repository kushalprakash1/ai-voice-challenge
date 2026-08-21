import asyncio

from voiceprobe.v3.flow_controller import (
    SchedulingFlowController,
)
from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.personas import (
    PersonaDecisionOverlay,
    PersonaRuntime,
    get_persona,
)
from voiceprobe.v3.production import (
    PipecatRuntimeBridge,
)
from voiceprobe.v3.runtime import DecisionRoute
from voiceprobe.v32.runtime_fallback import (
    V32SemanticFallbackResolver,
)
from voiceprobe.v32.semantic_parser import SemanticParser


class FakeSpeechFrame:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeWorker:
    def __init__(self) -> None:
        self.frames = []

    async def queue_frames(self, frames) -> None:
        self.frames.extend(frames)


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_json(self, **kwargs):
        del kwargs
        self.calls += 1

        # Only the reschedule-reason question should require v3.2
        # in this frozen conversation.
        if self.calls != 1:
            raise AssertionError(
                "Unexpected additional semantic fallback"
            )

        return {
            "speech_act": "ask",
            "operation": "reschedule",
            "focus": "reschedule_reason",
            "commitment": "informational",
            "certainty": "high",
        }


def test_complete_option_confuser_reschedule_flow():
    async def scenario():
        backend = FakeBackend()

        semantic_resolver = (
            V32SemanticFallbackResolver(
                parser=SemanticParser(
                    backend=backend,
                )
            )
        )

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
            fallback_resolver=semantic_resolver,
            flow_controller=controller,
            tts_frame_factory=FakeSpeechFrame,
        )

        worker = FakeWorker()
        bridge.bind_worker(worker)

        # ----------------------------------------------------
        # 1. Persistent existing appointment.
        # Must NOT be mistaken for a new booking.
        # ----------------------------------------------------
        existing = await bridge.runtime.process_turns([
            (
                "You already have an appointment Tuesday, "
                "August 25th at 2:15 PM. Would you like to "
                "keep it, reschedule it, or cancel it?"
            )
        ])

        assert (
            existing.decision.kind
            is DecisionKind.STATE_OBJECTIVE
        )

        assert (
            existing.after.accepted_slot_text
            is None
        )
        assert not existing.after.complete

        # ----------------------------------------------------
        # 2. Previously failing ordinary question.
        # Must reach v3.2 and produce a contextual answer.
        # ----------------------------------------------------
        reason = await bridge.runtime.process_turns([
            (
                "What is the reason for changing "
                "your appointment?"
            )
        ])

        assert reason.route is DecisionRoute.FALLBACK
        assert (
            reason.decision.kind
            is DecisionKind.CONTEXTUAL_ANSWER
        )
        assert (
            "Friday afternoon"
            in reason.decision.text
        )

        assert (
            reason.after.accepted_slot_text
            is None
        )
        assert not reason.after.complete

        # ----------------------------------------------------
        # 3. Routine provider preference remains deterministic.
        # ----------------------------------------------------
        provider = await bridge.runtime.process_turns([
            (
                "Do you have a provider preference, "
                "or are you okay with the first available provider?"
            )
        ])

        assert (
            provider.decision.kind
            is DecisionKind.ANSWER_PROVIDER_PREFERENCE
        )

        assert (
            provider.after.accepted_slot_text
            is None
        )

        # ----------------------------------------------------
        # 4. Multiple choices activate Option Confuser.
        # Neutral persona speech MUST NOT accept a slot.
        # ----------------------------------------------------
        options = await bridge.runtime.process_turns([
            (
                "Friday afternoon I have two fifteen PM, "
                "three PM, and three forty five PM. "
                "Which time works best for you?"
            )
        ])

        assert (
            options.decision.kind
            is DecisionKind.CLARIFY
        )

        assert (
            options.decision.text
            == (
                "Anything except the earliest one, "
                "but don't book anything yet."
            )
        )

        assert (
            options.after.accepted_slot_text
            is None
        )
        assert not options.after.complete

        # ----------------------------------------------------
        # 5. Patient restores earliest option but still gives
        # NO booking authorization.
        # ----------------------------------------------------
        restore = await bridge.runtime.process_turns([
            "Okay. I have the option you mentioned selected."
        ])

        assert (
            restore.decision.kind
            is DecisionKind.CLARIFY
        )

        assert (
            restore.decision.text
            == (
                "Actually, the earliest option is okay "
                "after all, but please don't book it yet."
            )
        )

        assert (
            restore.after.accepted_slot_text
            is None
        )
        assert not restore.after.complete

        # ----------------------------------------------------
        # 6. Booking prompt still does NOT get authorization.
        # Persona asks for exact selected time.
        # ----------------------------------------------------
        permission = await bridge.runtime.process_turns([
            "Would you like me to book that appointment?"
        ])

        assert (
            permission.decision.kind
            is DecisionKind.CLARIFY
        )

        assert (
            permission.decision.text
            == "Which exact time are you about to book?"
        )

        assert (
            permission.after.accepted_slot_text
            is None
        )
        assert not permission.after.complete

        # ----------------------------------------------------
        # 7. Exact-time verification produces final explicit
        # authorization. This is PRESERVE_BASE, so legitimate
        # slot state may finally advance here.
        # ----------------------------------------------------
        authorize = await bridge.runtime.process_turns([
            "I have two fifteen PM selected. Is that correct?"
        ])

        assert (
            authorize.decision.text
            == (
                "Yes, you can book it—the two fifteen PM slot."
            )
        )

        assert (
            authorize.after.accepted_slot_text
            is not None
        )

        assert not authorize.after.complete

        # ----------------------------------------------------
        # 8. A confirmation that introduces an ungrounded date must not
        # complete the objective, even when its time matches the accepted slot.
        # ----------------------------------------------------
        confirmed = await bridge.runtime.process_turns([
            (
                "Your appointment is confirmed for Friday, "
                "August 28th at 2:15 PM."
            )
        ])

        assert not confirmed.after.complete

        assert (
            confirmed.after.accepted_slot_text
            is not None
        )

        assert confirmed.after.booking_confirmation_text is None

        # ----------------------------------------------------
        # Global invariants.
        # ----------------------------------------------------
        assert backend.calls == 1
        assert (
            bridge.runtime.metrics.fallback_decisions
            == 1
        )

        assert persona.complete
        assert not persona.halted

        evidence = persona.evidence()

        moves = [
            event
            for event in evidence["events"]
            if event["event_type"] == "persona_move"
        ]

        assert len(moves) == 4

        assert [
            move["move_number"]
            for move in moves
        ] == [1, 2, 3, 4]

        # Seven patient responses should have been spoken.
        # The final remote confirmation itself requires no reply.
        assert len(worker.frames) == 7

    asyncio.run(scenario())
