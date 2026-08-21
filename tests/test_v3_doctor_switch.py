import pytest

from voiceprobe.v3.doctor_specialist import DoctorSpecialistDirectoryScenario, same_grounded_doctor
from voiceprobe.v3.flow_state import SchedulingFlowTracker


def obs(action="other", **kw):
    value = {"doctor_action": action, "reported_profile_name": "", "reported_profile_spelling": "",
             "doctors": (), "doctor_name": "", "specialty": "", "explicit_gender": "",
             "locations": (), "hours": "", "hours_location": "", "day": "",
             "multiple_doctors_capability": "", "requires_response": True}
    value.update(kw)
    return value


class Qwen:
    def __init__(self, outputs): self.outputs, self.last_observation = list(outputs), {}
    async def resolve(self, *_): self.last_observation = self.outputs.pop(0)


async def drive(outputs, turns):
    tracker = SchedulingFlowTracker()
    scenario = DoctorSpecialistDirectoryScenario(tracker=tracker, qwen=Qwen(outputs))
    decisions = []
    for turn in turns:
        decision = await scenario.resolve(turn, tracker.snapshot())
        decisions.append(decision)
        scenario.mark_decision_spoken(decision)
    return scenario, decisions


@pytest.mark.asyncio
async def test_exact_live_gender_answer_advances_and_never_repeats_gender():
    outputs = [
        obs("profile_registered"),
        obs("reports_profile_name", reported_profile_name="John Guac", reported_profile_spelling="j i a n g g"),
        obs("offers_doctors", doctors=(
            {"name": "Dubey Houser", "specialty": "quick orthopedic procedures"},
            {"name": "Doug Ross", "specialty": "advanced orthopedic procedures"},
            {"name": "Adam Bricker", "specialty": "joint and muscle conditions"},
        )),
        obs("states_specialty", doctor_name="Judy Houser", specialty="quick orthopedic procedures"),
        obs("states_gender", doctor_name="Dudi Hauser", explicit_gender="male"),
    ]
    turns = [
        "Your patient profile is set up.",
        "I have your name as John Guac. That's j i a n g g. Is that correct?",
        "Doctor Dubey Houser handles quick orthopedic procedures, Doctor Doug Ross handles advanced orthopedic procedures, and Doctor Adam Bricker handles joint and muscle conditions.",
        "Doctor Judy Houser specializes in quick orthopedic procedures.",
        "Doctor Dudi Hauser is a male doctor. Would you like to schedule an appointment with him or ask anything else?",
    ]
    scenario, decisions = await drive(outputs, turns)
    assert decisions[-1].reason == "doctor_directory:ask_location_hours"
    assert decisions[-1].text != "Is that doctor a male or female doctor?"
    assert scenario.gender_probe_complete == {"dubeyhouser"}
    assert [d.reason for d in decisions].count("doctor_directory:ask_gender") == 1
    assert {e.oracle_name for e in scenario.evidence} == {"profile_name_registration_mismatch"}


@pytest.mark.asyncio
async def test_stateful_selection_capability_switch_and_context_completion():
    outputs = [
        obs("profile_registered"),
        obs("reports_profile_name", reported_profile_name="John Guac", reported_profile_spelling="j i a n g g"),
        obs("offers_doctors", doctors=(
            {"name": "Dubey Houser", "specialty": "quick orthopedic procedures"},
            {"name": "Doug Ross", "specialty": "advanced orthopedic procedures"},
        )),
        obs("states_gender", doctor_name="Doogie Howser", explicit_gender="male"),
        obs("states_hours", locations=("Main Clinic",), hours="Monday nine to five", hours_location="Main Clinic", day="Monday"),
        obs("states_multiple_doctor_capability", multiple_doctors_capability="yes"),
        obs("asks_switch_reason"),
        obs("switch_acknowledged", doctor_name="Doug Ross"),
        obs("states_specialty", doctor_name="Doug Ross", specialty="advanced orthopedic procedures"),
        obs("states_hours", doctor_name="Doug Ross", locations=("West Clinic",), hours="Tuesday nine to four", hours_location="West Clinic", day="Tuesday"),
        obs("other", doctor_name="Doug Ross"),
    ]
    turns = [
        "Your patient profile is set up.",
        "I have your name as John Guac. That's j i a n g g.",
        "Doctor Dubey Houser handles quick orthopedic procedures and Doctor Doug Ross handles advanced orthopedic procedures.",
        "Doctor Doogie Howser is a male doctor.",
        "He works at Main Clinic Monday nine to five.",
        "Yes, you can see more than one doctor if you need to.",
        "Why would you like to switch doctors?",
        "I've switched your doctor to Doctor Doug Ross.",
        "Doctor Doug Ross specializes in advanced orthopedic procedures.",
        "Doctor Doug Ross works at West Clinic Tuesday nine to four.",
        "You have Doctor Doug Ross now.",
    ]
    scenario, decisions = await drive(outputs, turns)
    assert decisions[6].text == "I'd prefer a different doctor."
    assert scenario.active_doctor == "Doug Ross"
    assert scenario.switch_reason_spoken and scenario.switch_acknowledged
    assert scenario.final_reported_active_doctor == "Doug Ross"
    assert scenario.objective_complete
    assert not any(e.oracle_name != "profile_name_registration_mismatch" for e in scenario.evidence)


def test_asr_variants_do_not_merge_distinct_doctors():
    assert same_grounded_doctor("Dubey Houser", "Dudi Hauser")
    assert same_grounded_doctor("Dubey Houser", "Doogie Howser")
    assert not same_grounded_doctor("Dubey Houser", "Doug Ross")
