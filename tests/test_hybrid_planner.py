from voiceprobe.conversation.state import (
    ActionKind,
    PatientState,
    build_initial_state,
)
from voiceprobe.planners.hybrid import (
    HybridPatientPlanner,
    ResponsePlan,
    SelectorDecision,
)
from voiceprobe.scenarios.models import PatientFacts, PatientScenario


class StubSelector:
    def __init__(self, plan: ResponsePlan) -> None:
        self.plan = plan
        self.calls = 0

    def select(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        agent_turn: str,
    ) -> SelectorDecision:
        self.calls += 1
        return SelectorDecision(plan=self.plan)


def build_scenario() -> PatientScenario:
    return PatientScenario(
        scenario_id="shoulder-friday",
        objective="Schedule an appointment for Friday afternoon.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            insurance="Blue Cross",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
    )


def test_fast_path_skips_model_selector() -> None:
    scenario = build_scenario()
    selector = StubSelector(ResponsePlan.CLARIFY)

    planner = HybridPatientPlanner(
        selector=selector,
    )

    action = planner.plan(
        scenario=scenario,
        state=build_initial_state(scenario),
        agent_turn="What insurance do you have?",
    )

    assert action.response == "I have Blue Cross."
    assert selector.calls == 0


def test_ambiguous_turn_uses_selector() -> None:
    scenario = build_scenario()
    selector = StubSelector(ResponsePlan.ANSWER_COMPLAINT_DURATION)

    planner = HybridPatientPlanner(
        selector=selector,
    )

    action = planner.plan(
        scenario=scenario,
        state=build_initial_state(scenario),
        agent_turn="Can you tell me more about what's going on?",
    )

    assert selector.calls == 1
    assert action.response == "I've had right shoulder pain for about five days."


def test_selector_can_trigger_correction() -> None:
    scenario = build_scenario()
    selector = StubSelector(ResponsePlan.CORRECT_COMPLAINT_DURATION)

    planner = HybridPatientPlanner(
        selector=selector,
    )

    agent_turn = "Okay, so your left knee has been hurting for two weeks?"

    action = planner.plan(
        scenario=scenario,
        state=build_initial_state(scenario),
        agent_turn=agent_turn,
    )

    assert action.kind is ActionKind.CORRECT
    assert "right shoulder pain" in action.response
    assert "five days" in action.response
    assert action.corrected_claim == agent_turn


def test_selector_can_request_clarification() -> None:
    scenario = build_scenario()
    selector = StubSelector(ResponsePlan.CLARIFY)

    planner = HybridPatientPlanner(
        selector=selector,
    )

    action = planner.plan(
        scenario=scenario,
        state=build_initial_state(scenario),
        agent_turn="Something unclear and unexpected.",
    )

    assert action.kind is ActionKind.CLARIFY
    assert action.response == "Sorry, could you say that again?"
