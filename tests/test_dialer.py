from voiceprobe.dialer import build_call_plan
from voiceprobe.policy import CallPolicy
from voiceprobe.safety import ALLOWED_TEST_NUMBER


def test_build_call_plan_uses_fixed_destination() -> None:
    policy = CallPolicy(originating_number="+12025550101")

    plan = build_call_plan(policy)

    assert plan.originating_number == "+12025550101"
    assert plan.destination == ALLOWED_TEST_NUMBER
    assert plan.max_duration_seconds == 180
    assert plan.dry_run is True
