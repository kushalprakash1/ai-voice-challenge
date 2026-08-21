from __future__ import annotations

from collections.abc import Sequence

import httpx
import pytest

from voiceprobe.agents.brain import (
    CommunicationKind,
)
from voiceprobe.autonomous_phone import (
    build_runtime_patient_session,
)
from voiceprobe.conversation.session import (
    PatientSession,
)
from voiceprobe.reasoning.action_plan import (
    ActionPlan,
    PatientActionKind,
)
from voiceprobe.reasoning.session_v2 import (
    ReasoningV2PatientSession,
    reasoning_v2_enabled_from_environment,
)
from voiceprobe.reasoning.turn_frame import (
    TurnFrame,
)
from voiceprobe.scenarios.catalog import (
    get_scenario,
)


def frame(
    *,
    requested_action: str = "none",
    response_required: bool = False,
    options: list[dict[str, object]] | None = None,
    booking_confirmed: bool = False,
    confirmed: dict[str, object] | None = None,
    end_requested: bool = False,
) -> TurnFrame:
    return TurnFrame.model_validate(
        {
            "speech_act": (
                "confirmation"
                if booking_confirmed
                else "information"
            ),
            "workflow": "scheduling",
            "requested_action": requested_action,
            "response_required": response_required,
            "requested_facts": [],
            "other_requested_facts": [],
            "stated_facts": [],
            "proposed_workflow": None,
            "appointment_options": (
                options or []
            ),
            "confirmed_appointment": confirmed,
            "booking_confirmed": booking_confirmed,
            "conversation_end_requested": end_requested,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )


class FakeSemantic:
    def __init__(
        self,
        frames: Sequence[TurnFrame],
    ) -> None:
        self.frames = list(
            frames
        )

        self.histories: list[
            tuple[str, ...]
        ] = []

    def interpret(
        self,
        *,
        agent_turn: str,
        recent_history: Sequence[str],
    ) -> TurnFrame:
        del agent_turn

        self.histories.append(
            tuple(
                recent_history
            )
        )

        if not self.frames:
            raise RuntimeError(
                "No fake semantic frame remains."
            )

        return self.frames.pop(
            0
        )

    def close(self) -> None:
        pass


class FakePlanner:
    def __init__(
        self,
        plans: Sequence[ActionPlan],
    ) -> None:
        self.plans = list(
            plans
        )

    def plan(
        self,
        *,
        world,
        turn,
        recent_actions,
    ):
        del world, turn, recent_actions

        if not self.plans:
            raise RuntimeError(
                "No fake plan remains."
            )

        return (
            self.plans.pop(
                0
            ),
            (),
        )

    def close(self) -> None:
        pass


class FakeVerbalizer:
    def __init__(
        self,
        texts: Sequence[str],
    ) -> None:
        self.texts = list(
            texts
        )

    def verbalize(
        self,
        *,
        world,
        turn,
        plan,
        corrections,
    ) -> str:
        del world, turn, plan, corrections

        if not self.texts:
            raise RuntimeError(
                "No fake verbalization remains."
            )

        value = self.texts.pop(
            0
        )

        if value == "__RAISE__":
            raise RuntimeError(
                "synthetic verbalizer failure"
            )

        return value


def make_session(
    *,
    frames: Sequence[TurnFrame],
    plans: Sequence[ActionPlan],
    texts: Sequence[str],
):
    semantic = FakeSemantic(
        frames
    )

    session = ReasoningV2PatientSession(
        scenario=get_scenario(
            "autonomous-phone-diagnostic"
        ),
        model="test-model",
        url="http://ollama.invalid/api/chat",
        semantic=semantic,
        planner=FakePlanner(
            plans
        ),
        verbalizer=FakeVerbalizer(
            texts
        ),
    )

    return session, semantic


def option(
    day: str,
    time: str,
) -> dict[str, object]:
    return {
        "day": day,
        "date_text": None,
        "time": time,
        "daypart": None,
        "provider": None,
        "appointment_type": None,
    }


def test_reasoning_v2_feature_flag_defaults_off(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "VOICEPROBE_REASONING_V2",
        raising=False,
    )

    assert (
        reasoning_v2_enabled_from_environment()
        is False
    )


def test_reasoning_v2_feature_flag_accepts_exact_values(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "VOICEPROBE_REASONING_V2",
        "0",
    )

    assert (
        reasoning_v2_enabled_from_environment()
        is False
    )

    monkeypatch.setenv(
        "VOICEPROBE_REASONING_V2",
        "1",
    )

    assert (
        reasoning_v2_enabled_from_environment()
        is True
    )

    monkeypatch.setenv(
        "VOICEPROBE_REASONING_V2",
        "true",
    )

    with pytest.raises(
        ValueError
    ):
        reasoning_v2_enabled_from_environment()


def test_runtime_factory_keeps_legacy_default(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "VOICEPROBE_REASONING_V2",
        raising=False,
    )

    with httpx.Client() as client:
        session = (
            build_runtime_patient_session(
                scenario=get_scenario(
                    "autonomous-phone-diagnostic"
                ),
                model="qwen3:14b",
                url="http://ollama.invalid/api/chat",
                client=client,
            )
        )

    assert isinstance(
        session,
        PatientSession,
    )


def test_runtime_factory_builds_v2_only_when_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "VOICEPROBE_REASONING_V2",
        "1",
    )

    with httpx.Client() as client:
        session = (
            build_runtime_patient_session(
                scenario=get_scenario(
                    "autonomous-phone-diagnostic"
                ),
                model="qwen3:14b",
                url="http://ollama.invalid/api/chat",
                client=client,
            )
        )

        assert isinstance(
            session,
            ReasoningV2PatientSession,
        )


def test_v2_wait_maps_to_media_wait_and_silence() -> None:
    session, _ = make_session(
        frames=[
            frame(
                requested_action="wait",
                response_required=False,
            )
        ],
        plans=[
            ActionPlan(
                action=PatientActionKind(
                    "wait"
                ),
                reason_code="wait",
                confidence=1.0,
            )
        ],
        texts=[],
    )

    result = session.handle_agent_turn(
        "One moment."
    )

    assert (
        result.decision.kind
        is CommunicationKind.WAIT
    )

    assert result.patient_text == ""

    assert (
        result.progress.objective_complete
        is False
    )


def test_v2_selection_updates_legacy_progress() -> None:
    session, _ = make_session(
        frames=[
            frame(
                requested_action=(
                    "choose_option"
                ),
                response_required=True,
                options=[
                    option(
                        "Friday",
                        "2:30 PM",
                    )
                ],
            )
        ],
        plans=[
            ActionPlan(
                action=PatientActionKind(
                    "select_option"
                ),
                selected_option_index=0,
                reason_code="only_option",
                confidence=1.0,
            )
        ],
        texts=[
            "Friday at 2:30 PM works for me."
        ],
    )

    result = session.handle_agent_turn(
        "How about Friday at 2:30 PM?"
    )

    assert (
        result.decision.kind
        is CommunicationKind.ACCEPT_OFFER
    )

    assert (
        result.progress.offered_day
        == "Friday"
    )

    assert (
        result.progress.offered_time
        == "2:30 PM"
    )

    assert result.progress.offer_accepted
    assert not result.progress.booking_confirmed
    assert not result.progress.objective_complete


def test_matching_booking_confirmation_completes_objective_and_can_end() -> None:
    session, _ = make_session(
        frames=[
            frame(
                requested_action="choose_option",
                response_required=True,
                options=[
                    option(
                        "Friday",
                        "2:30 PM",
                    )
                ],
            ),
            frame(
                booking_confirmed=True,
                confirmed=option(
                    "Friday",
                    "2:30 PM",
                ),
            ),
        ],
        plans=[
            ActionPlan(
                action=PatientActionKind(
                    "select_option"
                ),
                selected_option_index=0,
                reason_code="select",
                confidence=1.0,
            ),
            ActionPlan(
                action=PatientActionKind(
                    "end_conversation"
                ),
                reason_code="booking_confirmed",
                confidence=1.0,
            ),
        ],
        texts=[
            "Friday at 2:30 PM works for me.",
            "Okay, thank you. Bye.",
        ],
    )

    session.handle_agent_turn(
        "How about Friday at 2:30 PM?"
    )

    result = session.handle_agent_turn(
        "You're booked for Friday at 2:30 PM."
    )

    assert (
        result.decision.kind
        is CommunicationKind.END_CONVERSATION
    )

    assert result.progress.offer_accepted
    assert result.progress.booking_confirmed
    assert result.progress.objective_complete


def test_wrong_booking_confirmation_never_completes_or_hangs_up() -> None:
    session, _ = make_session(
        frames=[
            frame(
                requested_action="choose_option",
                response_required=True,
                options=[
                    option(
                        "Friday",
                        "2:30 PM",
                    )
                ],
            ),
            frame(
                booking_confirmed=True,
                confirmed=option(
                    "Tuesday",
                    "9:00 AM",
                ),
            ),
        ],
        plans=[
            ActionPlan(
                action=PatientActionKind(
                    "select_option"
                ),
                selected_option_index=0,
                reason_code="select",
                confidence=1.0,
            ),
            # Even if the planner proposes END, the production session
            # must not expose a hangup side effect for a conflicting slot.
            ActionPlan(
                action=PatientActionKind(
                    "end_conversation"
                ),
                reason_code="booking_confirmed",
                confidence=1.0,
            ),
        ],
        texts=[
            "Friday at 2:30 PM works for me.",
        ],
    )

    session.handle_agent_turn(
        "How about Friday at 2:30 PM?"
    )

    result = session.handle_agent_turn(
        "You're booked Tuesday at 9 AM."
    )

    assert (
        result.decision.kind
        is not CommunicationKind.END_CONVERSATION
    )

    assert (
        result.decision.kind
        is CommunicationKind.DECLINE_OFFER
    )

    assert (
        "doesn't match"
        in result.patient_text
    )

    assert result.progress.offer_accepted
    assert not result.progress.booking_confirmed
    assert not result.progress.objective_complete

    # The accepted slot remains authoritative.
    assert (
        result.progress.offered_day
        == "Friday"
    )

    assert (
        result.progress.offered_time
        == "2:30 PM"
    )


def test_remote_goodbye_before_objective_does_not_trigger_local_hangup() -> None:
    session, _ = make_session(
        frames=[
            frame(
                requested_action="none",
                response_required=False,
                end_requested=True,
            )
        ],
        plans=[
            ActionPlan(
                action=PatientActionKind(
                    "end_conversation"
                ),
                reason_code="remote_conversation_end",
                confidence=1.0,
            )
        ],
        texts=[],
    )

    result = session.handle_agent_turn(
        "Okay, bye."
    )

    # The remote side may close, but VoiceProbe must not manufacture
    # objective completion or signal a successful local END.
    assert (
        result.decision.kind
        is CommunicationKind.WAIT
    )

    assert result.patient_text == ""

    assert not result.progress.objective_complete


def test_v2_turn_commit_is_atomic_on_verbalizer_failure() -> None:
    turn = frame(
        requested_action=(
            "state_objective"
        ),
        response_required=True,
    )

    semantic = FakeSemantic(
        [
            turn,
            turn,
        ]
    )

    session = ReasoningV2PatientSession(
        scenario=get_scenario(
            "autonomous-phone-diagnostic"
        ),
        model="test-model",
        url="http://ollama.invalid/api/chat",
        semantic=semantic,
        planner=FakePlanner(
            [
                ActionPlan(
                    action=PatientActionKind(
                        "state_objective"
                    ),
                    reason_code="objective",
                    confidence=1.0,
                ),
                ActionPlan(
                    action=PatientActionKind(
                        "state_objective"
                    ),
                    reason_code="objective",
                    confidence=1.0,
                ),
            ]
        ),
        verbalizer=FakeVerbalizer(
            [
                "__RAISE__",
                (
                    "I need to schedule an appointment "
                    "for Friday afternoon."
                ),
            ]
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic verbalizer failure",
    ):
        session.handle_agent_turn(
            "How may I help you?"
        )

    assert (
        session.progress
        == session.progress.__class__()
    )

    result = session.handle_agent_turn(
        "How may I help you?"
    )

    # The failed first turn was never committed into recent history.
    assert semantic.histories == [
        (),
        (),
    ]

    assert result.patient_text


def test_v2_prefetch_is_deliberately_disabled() -> None:
    session, _ = make_session(
        frames=[],
        plans=[],
        texts=[],
    )

    assert (
        session.prefetch_agent_turn(
            "provisional text"
        )
        is False
    )

    session.invalidate_prefetch()
