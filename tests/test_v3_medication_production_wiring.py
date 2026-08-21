from __future__ import annotations

import asyncio

from voiceprobe.v3 import production


class FakeSpeechFrame:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeWorker:
    def __init__(self) -> None:
        self.frames = []

    async def queue_frames(self, frames) -> None:
        self.frames.extend(frames)


class ScriptedQwen:
    def __init__(self) -> None:
        self.last_observation = {}
        self.calls = 0

    async def resolve(self, target_turn, snapshot):
        self.calls += 1
        del snapshot
        dose_claim = "twenty milligrams" in target_turn.casefold()
        self.last_observation = {
            "medication_action": (
                "confirm_or_correct_target_claim"
                if dose_claim
                else "provide_medication"
            ),
            "extracted_target_field": "dose" if dose_claim else "none",
            "extracted_target_value": "20 mg" if dose_claim else "",
            "target_acknowledges_correction": False,
            "requires_response": True,
        }


class SelfPayQwen:
    def __init__(self) -> None:
        self.last_observation = {}
        self.calls = 0

    async def resolve(self, target_turn, snapshot):
        self.calls += 1
        del target_turn, snapshot
        self.last_observation = {
            "self_pay_location_action": "provide_self_pay",
            "extracted_locations": (),
            "extracted_location": "",
            "target_acknowledged_location": "",
            "target_asserts_active_location": False,
            "extracted_insurance_status": "none",
            "extracted_insurer": "",
            "target_acknowledges_self_pay": False,
            "office_hours": "",
            "hours_location": "",
            "hours_context_changed": False,
            "requires_response": True,
        }


def build_medication_bridge(monkeypatch):
    monkeypatch.setenv("VOICEPROBE_SCENARIO", "medication-refill-correction")
    monkeypatch.setattr(production, "QwenV3FallbackRouter", ScriptedQwen)
    bridge = production.PipecatRuntimeBridge(
        tts_frame_factory=FakeSpeechFrame,
    )
    worker = FakeWorker()
    bridge.bind_worker(worker)
    return bridge, worker


def test_selected_medication_scenario_reaches_qwen_and_normal_response(monkeypatch):
    async def scenario():
        bridge, worker = build_medication_bridge(monkeypatch)
        result = await bridge.runtime.process_turns(
            ["What medication are you calling about?"]
        )
        assert bridge.semantic_mode == "medication_refill_qwen_v3"
        assert result.decision.reason == "medication_refill:provide_medication"
        assert result.decision.text == "The medication is lisinopril."
        assert worker.frames[0].text == result.decision.text
        assert bridge.scenario_metadata["medication_stage"] == "provide_medication"

    asyncio.run(scenario())


def test_adversarial_profile_prerequisite_bypasses_qwen_and_domain_resumes(monkeypatch):
    async def scenario():
        bridge, worker = build_medication_bridge(monkeypatch)
        profile = await bridge.runtime.process_turns(
            [
                (
                    "Would you like to create a demo patient profile? "
                    "I just need your first and last name to get started."
                )
            ]
        )
        assert profile.decision.reason.startswith("prerequisite:")
        assert "Chitragupta Subramnian Singh" in profile.decision.text
        assert bridge._qwen_v3_fallback_router.calls == 0

        medication = await bridge.runtime.process_turns(
            ["What medication are you calling about?"]
        )
        assert medication.decision.reason == "medication_refill:provide_medication"
        assert bridge._qwen_v3_fallback_router.calls == 1
        assert worker.frames[-1].text == "The medication is lisinopril."

    asyncio.run(scenario())


def test_current_name_question_outranks_future_or_historical_medication(monkeypatch):
    async def scenario():
        bridge, _ = build_medication_bridge(monkeypatch)
        turns = (
            (
                "I'll need your first and last name to create a demo patient profile. "
                "After that, I'll ask for the medication name. Can I get your first "
                "and last name to get started?"
            ),
            (
                "Thanks for sharing the medication name. To move forward, I'll need "
                "to create a demo patient profile. Can you provide your full name?"
            ),
        )
        for turn in turns:
            result = await bridge.runtime.process_turns([turn])
            assert result.decision.reason.startswith("prerequisite:")
            assert "Chitragupta Subramnian Singh" in result.decision.text
            assert "medication is" not in result.decision.text.casefold()
        assert bridge._qwen_v3_fallback_router.calls == 0

    asyncio.run(scenario())


def test_compound_prerequisites_preserve_known_domain_facts(monkeypatch):
    async def scenario():
        bridge, _ = build_medication_bridge(monkeypatch)
        cases = {
            "Can I have your name and the medication you're calling about?": (
                "Chitragupta Subramnian Singh", "lisinopril"
            ),
            "I need your date of birth and the medication name.": (
                "April 12, 1998", "lisinopril"
            ),
            "Can I get your first and last name and the pharmacy you used?": (
                "Chitragupta Subramnian Singh", "pharmacy on file"
            ),
            "Would you like to create a demo profile? Can I get your name?": (
                "Yes, please", "Chitragupta Subramnian Singh"
            ),
        }
        for turn, expected in cases.items():
            result = await bridge.runtime.process_turns([turn])
            assert result.decision.reason.startswith("prerequisite:")
            assert all(value in result.decision.text for value in expected)
        assert bridge._qwen_v3_fallback_router.calls == 0

    asyncio.run(scenario())


def test_wired_target_dose_observation_never_becomes_target_derived_truth(monkeypatch):
    async def scenario():
        bridge, worker = build_medication_bridge(monkeypatch)
        result = await bridge.runtime.process_turns(
            ["I have twenty milligrams."]
        )
        assert result.after.target_observations["dose"].value == "20 mg"
        assert result.after.caller_truth["dose"].value == "10 mg"
        assert result.after.caller_truth["dose"].value != "20 mg"
        assert worker.frames[-1].text.startswith("No, that's not right")
        assert bridge.scenario_metadata["oracle_candidate"] is False

    asyncio.run(scenario())


def test_autonomous_phone_diagnostic_does_not_activate_medication(monkeypatch):
    monkeypatch.setenv("VOICEPROBE_SCENARIO", "autonomous-phone-diagnostic")

    def forbidden_qwen():
        raise AssertionError("medication Qwen must not be constructed")

    monkeypatch.setattr(production, "QwenV3FallbackRouter", forbidden_qwen)
    bridge = production.PipecatRuntimeBridge(
        tts_frame_factory=FakeSpeechFrame,
        fallback_resolver=production.safe_production_fallback_resolver,
    )
    assert bridge.semantic_mode == "custom"
    assert bridge.scenario_metadata == {"scenario": "autonomous-phone-diagnostic"}


def test_selected_self_pay_scenario_uses_qwen_and_normal_response(monkeypatch):
    async def scenario():
        monkeypatch.setenv("VOICEPROBE_SCENARIO", "self-pay-location-switch")
        monkeypatch.setattr(production, "QwenV3FallbackRouter", SelfPayQwen)
        bridge = production.PipecatRuntimeBridge(
            tts_frame_factory=FakeSpeechFrame,
        )
        worker = FakeWorker()
        bridge.bind_worker(worker)
        result = await bridge.runtime.process_turns(
            ["What insurance do you carry?"]
        )
        assert bridge.semantic_mode == "self_pay_location_qwen_v3"
        assert result.decision.reason == "self_pay_location:establish_self_pay"
        assert worker.frames[0].text == result.decision.text
        assert "insurance_status" not in result.after.caller_truth
        await bridge.on_tts_stopped()
        assert bridge.runtime.flow_controller.tracker.snapshot().caller_truth["insurance_status"].value == "self_pay"
        metadata = bridge.scenario_metadata
        assert metadata["scenario"] == "self-pay-location-switch"
        assert metadata["scenario_stage"] == "provide_self_pay"

    asyncio.run(scenario())


def test_self_pay_profile_prerequisite_yields_back_to_domain(monkeypatch):
    async def scenario():
        monkeypatch.setenv("VOICEPROBE_SCENARIO", "self-pay-location-switch")
        monkeypatch.setattr(production, "QwenV3FallbackRouter", SelfPayQwen)
        bridge = production.PipecatRuntimeBridge(tts_frame_factory=FakeSpeechFrame)
        worker = FakeWorker()
        bridge.bind_worker(worker)
        for index, turn in enumerate((
            "Would you like to create a demo patient profile?",
            "Can I have your first and last name?",
        )):
            result = await bridge.runtime.process_turns([turn])
            assert result.decision.reason.startswith("prerequisite:")
            expected = (
                "Yes, please"
                if index == 0
                else "Chitragupta Subramnian Singh"
            )
            assert expected in result.decision.text
        assert bridge._qwen_v3_fallback_router.calls == 0
        insurance = await bridge.runtime.process_turns(
            ["What insurance do you carry?"]
        )
        assert insurance.decision.reason == "self_pay_location:establish_self_pay"
        assert bridge._qwen_v3_fallback_router.calls == 1
        assert "insurance_status" not in insurance.after.caller_truth
        await bridge.on_tts_stopped()
        assert bridge.runtime.flow_controller.tracker.snapshot().caller_truth["insurance_status"].value == "self_pay"
        assert "offered_slot" not in insurance.after.target_observations

    asyncio.run(scenario())
