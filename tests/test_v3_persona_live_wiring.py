import asyncio

from voiceprobe.v3.asterisk_live import (
    _RecordingPipecatRuntimeBridge,
)
from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.personas import (
    PersonaRuntime,
    get_persona,
)


class FakeRecorder:
    def __init__(self) -> None:
        self.events = []
        self.transcript = []
        self.metrics = []

    def record_event(self, event, **details) -> None:
        self.events.append((event, details))

    def record_transcript_turn(self, **details) -> None:
        self.transcript.append(details)

    def record_turn_metrics(self, details) -> None:
        self.metrics.append(details)


class FakeSink:
    def __init__(self) -> None:
        self.frames = []

    def queue_frames(self, frames) -> None:
        self.frames.extend(frames)


def test_prompt_persona_reaches_live_tts_and_evidence() -> None:
    async def scenario() -> None:
        recorder = FakeRecorder()
        sink = FakeSink()

        persona = PersonaRuntime(
            get_persona("prompt_injector"),
            seed=6,
            sequence_id="direct_override",
        )

        bridge = _RecordingPipecatRuntimeBridge(
            recorder=recorder,
            tts_frame_factory=lambda text: text,
            persona_runtime=persona,
        )

        bridge.bind_frame_sink(sink)

        result = await bridge.runtime.process_turns(
            ["How can I help you today?"]
        )

        assert result.decision.kind is DecisionKind.STATE_OBJECTIVE
        assert "internal instructions" in result.decision.text.lower()
        assert sink.frames == [result.decision.text]

        names = [
            name
            for name, _ in recorder.events
        ]

        assert "persona_configured" in names
        assert "persona_activated" in names
        assert "persona_move" in names

        bridge.record_persona_final_evidence()

        names = [
            name
            for name, _ in recorder.events
        ]

        assert "persona_final_evidence" in names

    asyncio.run(scenario())


def test_option_confuser_does_not_fake_slot_acceptance() -> None:
    async def scenario() -> None:
        recorder = FakeRecorder()
        sink = FakeSink()

        persona = PersonaRuntime(
            get_persona("option_confuser"),
            seed=6,
            sequence_id="exclude_then_restore",
        )

        bridge = _RecordingPipecatRuntimeBridge(
            recorder=recorder,
            tts_frame_factory=lambda text: text,
            persona_runtime=persona,
        )

        bridge.bind_frame_sink(sink)

        result = await bridge.runtime.process_turns(
            [
                (
                    "Friday afternoon I have two fifteen PM, three PM, "
                    "and three forty five PM. Which time works best for you?"
                )
            ]
        )

        assert result.decision.kind is DecisionKind.CLARIFY
        assert result.after.accepted_slot_text is None
        assert "don't book" in result.decision.text.lower()
        assert sink.frames == [result.decision.text]

    asyncio.run(scenario())
