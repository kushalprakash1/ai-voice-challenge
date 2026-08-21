import asyncio
import pytest
from voiceprobe.v3.doctor_specialist import (DoctorSpecialistDirectoryScenario,
    FULL_NAME, normalize_person_name, registered_name_materially_incompatible)
from voiceprobe.v3.flow_state import SchedulingFlowTracker
from voiceprobe.v3.production import DEFAULT_PRODUCTION_FLUX_CONFIG
from voiceprobe.v3.accent import doctor_specialist_phrase_inventory

def obs(action="other", **kw):
    value={"doctor_action":action,"reported_profile_name":"","reported_profile_spelling":"","doctors":(),
           "doctor_name":"","specialty":"","explicit_gender":"","locations":(),"hours":"","hours_location":"","day":"","requires_response":True}
    value.update(kw); return value
class Qwen:
    def __init__(self, outputs): self.outputs=list(outputs); self.last_observation={}
    async def resolve(self,*_): self.last_observation=self.outputs.pop(0)

async def drive(outputs, turns):
    tracker=SchedulingFlowTracker(); scenario=DoctorSpecialistDirectoryScenario(tracker=tracker,qwen=Qwen(outputs)); decisions=[]
    for turn in turns:
        d=await scenario.resolve(turn,tracker.snapshot()); decisions.append(d); scenario.mark_decision_spoken(d)
    return scenario, decisions

@pytest.mark.asyncio
async def test_clean_grounded_milestone_flow_and_provenance():
    outputs=[obs("profile_registered"),obs("reports_profile_name",reported_profile_name=FULL_NAME,reported_profile_spelling="G Y E O N G H Y E O N G W A K"),
      obs("offers_doctors",doctors=({"name":"Min Park","specialty":"Cardiology"},{"name":"Sora Han","specialty":"Dermatology"})),
      obs("states_specialty",specialty="Cardiology"),obs("states_gender",explicit_gender="female"),
      obs("states_location",locations=("Riverside Clinic",)),obs("states_hours",hours="Monday nine to five",day="Monday"),obs("states_hours",hours="Monday nine to five",day="Monday")]
    turns=["Your profile is registered.","The name is Gyeong-hyeon Gwak, spelled G Y E O N G H Y E O N G W A K.",
      "Dr. Min Park, Cardiology, and Dr. Sora Han, Dermatology are available.","The specialty is Cardiology.",
      "Dr. Min Park is a female doctor.","That doctor works at Riverside Clinic.","The hours are Monday nine to five.","Yes, Monday nine to five."]
    s,d=await drive(outputs,turns)
    assert d[0].text.startswith("Could you repeat") and d[1].text=="What doctors or specialists are available?"
    assert s.selected_doctor=="Min Park" and not s.objective_complete
    assert not s.evidence and s.tracker.snapshot().caller_truth["full_name"].value==FULL_NAME

@pytest.mark.asyncio
async def test_exact_failed_live_profile_ack_transitions_to_cached_name_probe():
    target=("Your patient profile is set up. "
            "And your date of birth is July fourth two thousand for demo purposes. "
            "How may I help you today?")
    tracker=SchedulingFlowTracker()
    scenario=DoctorSpecialistDirectoryScenario(tracker=tracker,qwen=Qwen([obs("other")]))

    decision=await scenario.resolve(target,tracker.snapshot())
    snapshot=tracker.snapshot()

    assert snapshot.target_observations["profile_created_acknowledged"].value is True
    assert snapshot.target_observations["profile_registered"].value is True
    assert scenario.profile_registered is True
    assert decision.kind.value=="contextual_answer"
    assert decision.text=="Could you repeat the name you have on my profile and spell it for me?"
    assert decision.requires_response is True

@pytest.mark.asyncio
async def test_jack_salutation_is_supporting_target_evidence_not_caller_truth_or_oracle():
    turns=[
        "Your patient profile is set up. And your date of birth is July fourth two thousand for demo purposes. How may I help you today?",
        "Thanks, Jack. I have your date of birth as July fourth two thousand. What would you like to do next?",
    ]
    tracker=SchedulingFlowTracker()
    scenario=DoctorSpecialistDirectoryScenario(
        tracker=tracker,qwen=Qwen([obs("other"),obs("other")]))
    first=await scenario.resolve(turns[0],tracker.snapshot())
    scenario.mark_decision_spoken(first)
    second=await scenario.resolve(turns[1],tracker.snapshot())
    snapshot=tracker.snapshot()

    assert snapshot.target_observations["reported_profile_name_candidate"].value=="Jack"
    assert snapshot.caller_truth["full_name"].value==FULL_NAME
    assert not scenario.name_verified
    assert not scenario.evidence
    assert second.reason=="doctor_directory:verify_registered_name"
    assert second.text=="Could you repeat the name you have on my profile and spell it for me?"

@pytest.mark.parametrize("target",[
    "Your patient profile is set up.",
    "Your profile is set up.",
    "Your demo patient profile has been created.",
    "We successfully registered your patient profile.",
])
@pytest.mark.asyncio
async def test_equivalent_explicit_profile_acknowledgments(target):
    tracker=SchedulingFlowTracker()
    scenario=DoctorSpecialistDirectoryScenario(tracker=tracker,qwen=Qwen([obs("other")]))
    decision=await scenario.resolve(target,tracker.snapshot())
    assert scenario.profile_registered
    assert decision.reason=="doctor_directory:verify_registered_name"

@pytest.mark.asyncio
async def test_complete_exact_live_recovery_reaches_doctor_hours_without_fallbacks():
    outputs=[obs("other"),obs("other"),
      obs("reports_profile_name",reported_profile_name=FULL_NAME,reported_profile_spelling="G Y E O N G H Y E O N G W A K"),
      obs("offers_doctors",doctors=({"name":"Min Park","specialty":"Cardiology"},{"name":"Sora Han","specialty":"Dermatology"})),
      obs("states_specialty",specialty="Cardiology"),obs("states_gender",explicit_gender="female"),
      obs("states_location",locations=("Riverside Clinic",)),
      obs("states_hours",hours="Monday nine to five",day="Monday"),
      obs("states_hours",hours="Monday nine to five",day="Monday")]
    turns=[
      "Your patient profile is set up. And your date of birth is July fourth two thousand for demo purposes. How may I help you today?",
      "Thanks, Jack. I have your date of birth as July fourth two thousand. What would you like to do next?",
      "The profile name is Gyeong-hyeon Gwak, spelled G Y E O N G H Y E O N G W A K.",
      "Dr. Min Park in Cardiology and Dr. Sora Han in Dermatology are available.",
      "Dr. Min Park specializes in Cardiology.","Dr. Min Park is a female doctor.",
      "That doctor works at Riverside Clinic.","That doctor works Monday nine to five at Riverside Clinic.",
      "Yes, those are that doctor's Monday hours at Riverside Clinic."]
    scenario,decisions=await drive(outputs,turns)
    reasons=[d.reason for d in decisions]
    spoken=[d.text for d in decisions if d.text]
    snapshot=scenario.tracker.snapshot()

    assert reasons[0]=="doctor_directory:verify_registered_name"
    assert "doctor_directory:discover_specialists" in reasons
    assert "doctor_directory:select_grounded_doctor" in reasons
    assert "doctor_directory:ask_gender" in reasons
    assert "doctor_directory:ask_location_hours" in reasons
    assert "doctor_directory:ask_hours" in reasons
    assert not scenario.objective_complete  # switch milestones remain
    assert snapshot.target_observations["reported_profile_name_candidate"].value=="Jack"
    assert snapshot.caller_truth["full_name"].value==FULL_NAME
    assert not scenario.semantic_failures
    assert all(text in doctor_specialist_phrase_inventory() for text in spoken)

@pytest.mark.parametrize("candidate",["gyeong hyeon gwak","GYEONG-HYEON GWAK","Gyeonghyeon, Gwak"])
def test_name_normalization_false_positive_guards(candidate):
    assert not registered_name_materially_incompatible(candidate)
def test_material_name_mismatch(): assert registered_name_materially_incompatible("Gyeong-hyeon Kwak")

@pytest.mark.asyncio
async def test_every_conservative_oracle_and_false_positive_controls():
    outputs=[obs("profile_registered"),obs("reports_profile_name",reported_profile_name="Gyeong-hyeon Kwak"),
      obs("offers_doctors",doctors=({"name":"Min Park","specialty":"Cardiology"},)),obs("states_specialty",specialty="Cardiology"),
      obs("states_specialty",specialty="Neurology"),obs("states_gender",explicit_gender="male"),
      obs("states_gender",explicit_gender="female"),obs("states_location",locations=("East Clinic","West Clinic")),
      obs("states_hours",hours="nine to five",hours_location="East Clinic",day="Monday"),
      obs("states_hours",hours="nine to four",hours_location="East Clinic",day="Monday"),
      obs("states_location",doctor_name="Sora Han",locations=("North Clinic",))]
    turns=["Your profile is registered.","The name is Gyeong-hyeon Kwak.","Dr. Min Park, Cardiology is available.",
      "The specialty is Cardiology.","The specialty is Neurology.","Dr. Min Park is explicitly male.","Dr. Min Park is explicitly female.",
      "He works at East Clinic and West Clinic.","At East Clinic Monday hours are nine to five.","At East Clinic Monday hours are nine to four.",
      "Dr. Sora Han works at North Clinic."]
    s,_=await drive(outputs,turns); names={e.oracle_name for e in s.evidence}
    assert names=={"profile_name_registration_mismatch","specialist_identity_or_specialty_inconsistency","doctor_gender_explicit_contradiction","doctor_hours_internal_contradiction","doctor_identity_context_mismatch"}
    assert len(s.locations)>=2  # multiple legitimate locations alone were not contradictory

def test_call5_fixed_inventory_and_turn_contract():
    phrases=doctor_specialist_phrase_inventory(); assert len(phrases)==len(set(phrases))
    assert "My name is Gyeong-hyeon Gwak." in phrases
    assert DEFAULT_PRODUCTION_FLUX_CONFIG.continuation_grace_ms==3000
