import asyncio

from voiceprobe.v3.flow_state import FlowStage
from voiceprobe.v3.models import DecisionKind, PolicyDecision
from voiceprobe.v3.production import PipecatRuntimeBridge


class FakeSpeechFrame:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeWorker:
    def __init__(self) -> None:
        self.frames = []

    async def queue_frames(self, frames) -> None:
        self.frames.extend(frames)


class FakeSemanticRouter:
    async def resolve(self, agent_turn, snapshot):
        assert agent_turn == "What is the reason for your appointment?"
        return PolicyDecision(
            DecisionKind.ANSWER_COMPLAINT,
            text="I have right shoulder pain.",
            reason="complaint_requested",
            confidence=0.99,
        )


def test_semantic_router_seam_is_response_producing() -> None:
    async def scenario():
        bridge = PipecatRuntimeBridge(
            tts_frame_factory=FakeSpeechFrame,
            semantic_router=FakeSemanticRouter(),
        )
        worker = FakeWorker()
        bridge.bind_worker(worker)

        result = await bridge.runtime.process_turns(
            ["What is the reason for your appointment?"]
        )

        assert result.route.value == "fallback"
        assert result.decision.kind == DecisionKind.ANSWER_COMPLAINT
        assert result.response_ready is True
        assert FlowStage.VISIT_REASON not in result.before.communicated
        assert FlowStage.VISIT_REASON in result.after.communicated
        assert [frame.text for frame in worker.frames] == [
            "I have right shoulder pain."
        ]

    asyncio.run(scenario())
