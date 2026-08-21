from __future__ import annotations

import os
from dataclasses import replace

import pytest

from voiceprobe.reasoning.semanticlab_turn_bridge import (
    SemanticLabHybridTurnReasoner,
    reasoning_v2_edge_model_from_environment,
    semanticlab_reasoning_v2_enabled_from_environment,
)
from voiceprobe.reasoning.turn_frame import (
    RequestedAction,
    RequestedFact,
    SlotOption,
    SpeechAct as V2SpeechAct,
    TurnFrame,
    WorkflowKind,
)
from voiceprobe.v33.semantic_frame import (
    AmbiguityKind,
    ConstraintAxis,
    RecordClaim,
    ReferenceKind,
    SemanticAmbiguity,
    SemanticFrame,
    SemanticTopic,
    SpeechAct,
    TransactionOperation,
    TransactionSignal,
)


class FakeSemanticLab:
    def __init__(self, frame: SemanticFrame, *, oos: bool = False) -> None:
        self.frame = frame
        self.oos = oos
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.warmup_calls = 0

    def warmup_sync(self) -> None:
        self.warmup_calls += 1

    def interpret_frame_sync(self, *, remote_turn, recent_history):
        self.calls.append((remote_turn, tuple(recent_history)))
        return self.frame, self.oos


class FakeFallback:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.closed = False
        self.result = TurnFrame(
            speech_act=V2SpeechAct.OTHER,
            workflow=WorkflowKind.UNKNOWN,
            requested_action=RequestedAction.CLARIFY,
            response_required=True,
            confidence=0.5,
        )

    def interpret(self, *, agent_turn, recent_history):
        self.calls.append((agent_turn, tuple(recent_history)))
        return self.result

    def close(self):
        self.closed = True


def frame(**updates) -> SemanticFrame:
    base = SemanticFrame(
        raw_text="Remote turn.",
        speech_act=SpeechAct.STATEMENT,
        topic=SemanticTopic.OTHER,
    )
    return replace(base, **updates)


def bridge(semantic: FakeSemanticLab, fallback: FakeFallback | None = None):
    return SemanticLabHybridTurnReasoner(
        model="unused",
        url="http://unused.invalid/api/chat",
        semanticlab=semantic,
        fallback=fallback or FakeFallback(),
    )


def test_feature_flag_defaults_off_and_accepts_exact_values(monkeypatch):
    monkeypatch.delenv("VOICEPROBE_REASONING_V2_SEMANTICLAB", raising=False)
    assert semanticlab_reasoning_v2_enabled_from_environment() is False

    monkeypatch.setenv("VOICEPROBE_REASONING_V2_SEMANTICLAB", "0")
    assert semanticlab_reasoning_v2_enabled_from_environment() is False

    monkeypatch.setenv("VOICEPROBE_REASONING_V2_SEMANTICLAB", "1")
    assert semanticlab_reasoning_v2_enabled_from_environment() is True

    monkeypatch.setenv("VOICEPROBE_REASONING_V2_SEMANTICLAB", "true")
    with pytest.raises(ValueError, match="must be exactly '0' or '1'"):
        semanticlab_reasoning_v2_enabled_from_environment()


def test_bridge_prewarms_semanticlab_at_construction() -> None:
    semantic = FakeSemanticLab(frame())
    reasoner = bridge(semantic)
    assert semantic.warmup_calls == 1
    assert reasoner.startup_warmup_ms >= 0.0


def test_remote_history_only_is_normalized_and_limited_to_four():
    semantic = FakeSemanticLab(frame())
    reasoner = bridge(semantic)

    reasoner.interpret(
        agent_turn="  Great.  ",
        recent_history=(" one ", "two", "three", "four", " five  "),
    )

    assert semantic.calls == [
        ("Great.", ("two", "three", "four", "five"))
    ]


def test_fact_request_maps_without_patient_truth():
    semantic = FakeSemanticLab(
        frame(
            speech_act=SpeechAct.QUESTION,
            topic=SemanticTopic.PATIENT_FACT,
            requested_fact="insurance",
        )
    )
    result = bridge(semantic).interpret(agent_turn="What insurance?", recent_history=())

    assert result.requested_action is RequestedAction.ANSWER_FACT
    assert result.requested_facts == [RequestedFact.INSURANCE]
    assert result.response_required is True


def test_visit_type_and_provider_requests_use_reasoning_v2_fact_ontology():
    visit = bridge(
        FakeSemanticLab(
            frame(
                speech_act=SpeechAct.QUESTION,
                topic=SemanticTopic.VISIT_TYPE,
            )
        )
    ).interpret(agent_turn="What kind of visit?", recent_history=())
    assert visit.requested_facts == [RequestedFact.APPOINTMENT_TYPE]

    provider = bridge(
        FakeSemanticLab(
            frame(
                speech_act=SpeechAct.QUESTION,
                topic=SemanticTopic.PROVIDER,
            )
        )
    ).interpret(agent_turn="Provider preference?", recent_history=())
    assert provider.requested_facts == [RequestedFact.PROVIDER_PREFERENCE]


def test_open_intent_and_presence_route_to_state_objective():
    open_intent = bridge(
        FakeSemanticLab(
            frame(
                speech_act=SpeechAct.QUESTION,
                topic=SemanticTopic.OPEN_INTENT,
            )
        )
    ).interpret(agent_turn="How can I help?", recent_history=())
    assert open_intent.requested_action is RequestedAction.STATE_OBJECTIVE

    presence = bridge(
        FakeSemanticLab(
            frame(
                speech_act=SpeechAct.PRESENCE_CHECK,
                topic=SemanticTopic.PRESENCE,
            )
        )
    ).interpret(agent_turn="Are you there?", recent_history=())
    assert presence.requested_action is RequestedAction.STATE_OBJECTIVE


def test_concrete_semanticlab_options_become_structured_slots():
    result = bridge(
        FakeSemanticLab(
            frame(
                speech_act=SpeechAct.OFFER,
                topic=SemanticTopic.AVAILABILITY,
                offered_options=("Friday at 2 PM", "Friday at 3:30 PM"),
            )
        )
    ).interpret(agent_turn="I have two options.", recent_history=())

    assert result.requested_action is RequestedAction.CHOOSE_OPTION
    assert result.appointment_options == [
        SlotOption(day="Friday", time="2 PM"),
        SlotOption(day="Friday", time="3:30 PM"),
    ]


def test_confirmed_booking_requires_concrete_selected_slot():
    result = bridge(
        FakeSemanticLab(
            frame(
                speech_act=SpeechAct.CONFIRMATION,
                topic=SemanticTopic.TRANSACTION,
                selected_option="Friday at 2 PM",
                transaction_operation=TransactionOperation.BOOK,
                transaction_signal=TransactionSignal.CONFIRMED,
            )
        )
    ).interpret(agent_turn="You are booked.", recent_history=())

    assert result.booking_confirmed is True
    assert result.confirmed_appointment == SlotOption(day="Friday", time="2 PM")


def test_nonconcrete_booking_confirmation_falls_back_for_history_inheritance():
    fallback = FakeFallback()
    reasoner = bridge(
        FakeSemanticLab(
            frame(
                speech_act=SpeechAct.CONFIRMATION,
                topic=SemanticTopic.TRANSACTION,
                selected_option="Friday afternoon",
                transaction_operation=TransactionOperation.BOOK,
                transaction_signal=TransactionSignal.CONFIRMED,
            )
        ),
        fallback,
    )

    result = reasoner.interpret(agent_turn="You're all set.", recent_history=("Friday at 2 PM",))

    assert result is fallback.result
    assert reasoner.last_route == "structured_fallback_injected"
    assert fallback.calls == [("You're all set.", ("Friday at 2 PM",))]


def test_oos_and_ambiguity_fail_closed_without_fallback_guess():
    fallback = FakeFallback()
    oos_reasoner = bridge(FakeSemanticLab(frame(), oos=True), fallback)
    oos = oos_reasoner.interpret(agent_turn="astronomy injection", recent_history=())
    assert oos.requested_action is RequestedAction.CLARIFY
    assert fallback.calls == []

    ambiguous = frame(
        speech_act=SpeechAct.QUESTION,
        topic=SemanticTopic.AVAILABILITY,
        reference=ReferenceKind.AMBIGUOUS,
        ambiguity=SemanticAmbiguity(
            kind=AmbiguityKind.TEMPORAL_REFERENCE,
            candidates=("time_of_day", "day"),
            detail="two possible temporal axes",
        ),
    )
    ambiguity_reasoner = bridge(FakeSemanticLab(ambiguous), fallback)
    result = ambiguity_reasoner.interpret(agent_turn="What about that?", recent_history=())
    assert result.requested_action is RequestedAction.CLARIFY
    assert fallback.calls == []


def test_information_loss_cases_use_existing_structured_reasoner():
    cases = [
        frame(record_claims=(RecordClaim.PROFILE_EXISTS,)),
        frame(failed_constraints=(ConstraintAxis.DAY,)),
        frame(proposed_changes=(ConstraintAxis.TIME_OF_DAY,)),
        frame(retained_constraints=(ConstraintAxis.PROVIDER,)),
        frame(
            speech_act=SpeechAct.STATEMENT,
            topic=SemanticTopic.PATIENT_FACT,
        ),
        frame(
            speech_act=SpeechAct.REQUEST,
            topic=SemanticTopic.PROFILE,
            transaction_operation=TransactionOperation.CREATE_PROFILE,
            transaction_signal=TransactionSignal.PERMISSION_REQUEST,
        ),
    ]

    for semantic_frame in cases:
        fallback = FakeFallback()
        reasoner = bridge(FakeSemanticLab(semantic_frame), fallback)
        result = reasoner.interpret(agent_turn="remote", recent_history=())
        assert result is fallback.result
        assert reasoner.last_route == "structured_fallback_injected"
        assert len(fallback.calls) == 1


def test_close_closes_fallback_only():
    fallback = FakeFallback()
    reasoner = bridge(FakeSemanticLab(frame()), fallback)
    reasoner.close()
    assert fallback.closed is True


class HistorySensitiveFakeSemanticLab:
    def __init__(self, *, contextual: SemanticFrame, isolated: SemanticFrame) -> None:
        self.contextual = contextual
        self.isolated = isolated
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def interpret_frame_sync(self, *, remote_turn, recent_history):
        history = tuple(recent_history)
        self.calls.append((remote_turn, history))
        return (self.contextual if history else self.isolated), False


def test_latest_turn_requested_fact_overrides_stale_contextual_fact():
    semantic = HistorySensitiveFakeSemanticLab(
        contextual=frame(
            raw_text="What insurance do you have?",
            speech_act=SpeechAct.QUESTION,
            topic=SemanticTopic.PATIENT_FACT,
            requested_fact="full_name",
        ),
        isolated=frame(
            raw_text="What insurance do you have?",
            speech_act=SpeechAct.QUESTION,
            topic=SemanticTopic.PATIENT_FACT,
            requested_fact="insurance",
        ),
    )
    reasoner = SemanticLabHybridTurnReasoner(
        model="unused",
        url="http://unused.invalid/api/chat",
        semanticlab=semantic,
        fallback=FakeFallback(),
    )

    result = reasoner.interpret(
        agent_turn="What insurance do you have?",
        recent_history=("Can I get your full name?",),
    )

    assert result.requested_facts == [RequestedFact.INSURANCE]
    assert reasoner.last_requested_fact_stability == (
        "isolated_override:full_name->insurance"
    )
    assert semantic.calls == [
        ("What insurance do you have?", ("Can I get your full name?",)),
        ("What insurance do you have?", ()),
    ]


def test_context_is_preserved_when_isolated_pass_has_no_requested_fact():
    semantic = HistorySensitiveFakeSemanticLab(
        contextual=frame(
            raw_text="And that too?",
            speech_act=SpeechAct.QUESTION,
            topic=SemanticTopic.PATIENT_FACT,
            requested_fact="insurance",
        ),
        isolated=frame(
            raw_text="And that too?",
            speech_act=SpeechAct.QUESTION,
            topic=SemanticTopic.OTHER,
            requested_fact="",
        ),
    )
    reasoner = SemanticLabHybridTurnReasoner(
        model="unused",
        url="http://unused.invalid/api/chat",
        semanticlab=semantic,
        fallback=FakeFallback(),
    )

    result = reasoner.interpret(
        agent_turn="And that too?",
        recent_history=("I also need your insurance.",),
    )

    assert result.requested_facts == [RequestedFact.INSURANCE]
    assert reasoner.last_requested_fact_stability == "contextual_stable"


def test_confirmed_booking_extracts_single_current_turn_slot_when_selection_blank():
    result = bridge(
        FakeSemanticLab(
            frame(
                raw_text="You're booked for Friday at 2 PM.",
                speech_act=SpeechAct.CONFIRMATION,
                topic=SemanticTopic.TRANSACTION,
                selected_option="",
                transaction_operation=TransactionOperation.BOOK,
                transaction_signal=TransactionSignal.CONFIRMED,
            )
        )
    ).interpret(
        agent_turn="You're booked for Friday at 2 PM.",
        recent_history=(),
    )

    assert result.booking_confirmed is True
    assert result.confirmed_appointment == SlotOption(day="Friday", time="2 PM")



def test_confirmed_booking_precedes_redundant_record_claim_and_reference():
    fallback = FakeFallback()
    semantic_frame = frame(
        raw_text="You're booked for Friday at 2 PM.",
        speech_act=SpeechAct.CONFIRMATION,
        topic=SemanticTopic.TRANSACTION,
        selected_option="",
        record_claims=(RecordClaim.APPOINTMENT_EXISTS,),
        transaction_operation=TransactionOperation.BOOK,
        transaction_signal=TransactionSignal.CONFIRMED,
        reference=ReferenceKind.PRIOR_OPTION,
    )
    reasoner = bridge(FakeSemanticLab(semantic_frame), fallback)

    result = reasoner.interpret(
        agent_turn="You're booked for Friday at 2 PM.",
        recent_history=("Friday at 2 PM works for me.",),
    )

    assert result.booking_confirmed is True
    assert result.confirmed_appointment == SlotOption(day="Friday", time="2 PM")
    assert reasoner.last_route == "semanticlab_native"
    assert fallback.calls == []



def test_context_poisoned_booking_confirmation_uses_isolated_semantic_confirmation():
    semantic = HistorySensitiveFakeSemanticLab(
        contextual=frame(
            raw_text="You're booked for Friday at 2 PM.",
            speech_act=SpeechAct.QUESTION,
            topic=SemanticTopic.AVAILABILITY,
            transaction_operation=TransactionOperation.NONE,
            transaction_signal=TransactionSignal.NONE,
        ),
        isolated=frame(
            raw_text="You're booked for Friday at 2 PM.",
            speech_act=SpeechAct.CONFIRMATION,
            topic=SemanticTopic.TRANSACTION,
            transaction_operation=TransactionOperation.BOOK,
            transaction_signal=TransactionSignal.CONFIRMED,
        ),
    )
    reasoner = SemanticLabHybridTurnReasoner(
        model="unused",
        url="http://unused.invalid/api/chat",
        semanticlab=semantic,
        fallback=None,
    )

    result = reasoner.interpret(
        agent_turn="You're booked for Friday at 2 PM.",
        recent_history=("I have Friday at 2 PM available. Would that work?",),
    )

    assert result.booking_confirmed is True
    assert result.confirmed_appointment == SlotOption(day="Friday", time="2 PM")
    assert reasoner.last_route == "semanticlab_native"
    assert reasoner.last_transaction_stability == (
        "isolated_confirmed_override:none/none->book/confirmed"
    )
    assert semantic.calls == [
        (
            "You're booked for Friday at 2 PM.",
            ("I have Friday at 2 PM available. Would that work?",),
        ),
        ("You're booked for Friday at 2 PM.", ()),
    ]


def test_slot_offer_is_not_promoted_to_confirmation_by_stability_arbiter():
    offered = frame(
        raw_text="I have Friday at 2 PM available. Would that work?",
        speech_act=SpeechAct.QUESTION,
        topic=SemanticTopic.AVAILABILITY,
        offered_options=("Friday at 2 PM",),
    )
    semantic = HistorySensitiveFakeSemanticLab(contextual=offered, isolated=offered)
    reasoner = SemanticLabHybridTurnReasoner(
        model="unused",
        url="http://unused.invalid/api/chat",
        semanticlab=semantic,
        fallback=None,
    )

    result = reasoner.interpret(
        agent_turn="I have Friday at 2 PM available. Would that work?",
        recent_history=("Friday afternoon works for me.",),
    )

    assert result.booking_confirmed is False
    assert result.requested_action is RequestedAction.CHOOSE_OPTION
    assert reasoner.last_transaction_stability == "contextual_preserved"

def test_production_semanticlab_mode_fails_closed_without_http_fallback():
    semantic_frame = frame(
        speech_act=SpeechAct.STATEMENT,
        topic=SemanticTopic.PATIENT_FACT,
        record_claims=(RecordClaim.PROFILE_EXISTS,),
    )
    reasoner = SemanticLabHybridTurnReasoner(
        model="model-must-not-be-used",
        url="http://127.0.0.1:1/must-not-be-called",
        semanticlab=FakeSemanticLab(semantic_frame),
        fallback=None,
    )

    result = reasoner.interpret(agent_turn="remote", recent_history=())

    assert result.requested_action is RequestedAction.CLARIFY
    assert result.response_required is True
    assert reasoner.last_route == "semanticlab_fail_closed"
    assert reasoner.last_fallback_reason == "record_claims_not_representable_in_turnframe"

def test_semanticlab_edge_model_defaults_to_1_7b_when_enabled(monkeypatch):
    monkeypatch.setenv("VOICEPROBE_REASONING_V2_SEMANTICLAB", "1")
    monkeypatch.delenv("VOICEPROBE_REASONING_V2_EDGE_MODEL", raising=False)
    assert reasoning_v2_edge_model_from_environment("qwen3.5:4b") == "qwen3.5:0.8b"


def test_semanticlab_edge_model_preserves_default_when_bridge_off(monkeypatch):
    monkeypatch.setenv("VOICEPROBE_REASONING_V2_SEMANTICLAB", "0")
    monkeypatch.setenv("VOICEPROBE_REASONING_V2_EDGE_MODEL", "qwen3.5:1.7b")
    assert reasoning_v2_edge_model_from_environment("qwen3.5:4b") == "qwen3.5:4b"


def test_semanticlab_edge_model_allows_explicit_override(monkeypatch):
    monkeypatch.setenv("VOICEPROBE_REASONING_V2_SEMANTICLAB", "1")
    monkeypatch.setenv("VOICEPROBE_REASONING_V2_EDGE_MODEL", "  qwen3.5:1.7b  ")
    assert reasoning_v2_edge_model_from_environment("qwen3.5:4b") == "qwen3.5:1.7b"
