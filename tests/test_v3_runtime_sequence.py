import asyncio

from voiceprobe.v3.flow_state import FlowStage
from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.runtime import VoiceProbeV3Runtime


def test_newest_call_failure_path_behaves_as_one_coherent_runtime() -> None:
    async def scenario():
        runtime = VoiceProbeV3Runtime()

        profile = await runtime.process_turns(
            [
                (
                    "Thank you for calling Pivot Point Orthopedics. "
                    "Would you like to create a demo patient profile? "
                    "I just need your first and last name to get started."
                )
            ]
        )
        assert profile.decision.kind == DecisionKind.CREATE_PROFILE

        wrong_dob = await runtime.process_turns(
            [
                (
                    "Your demo patient profile is set up and your date of birth "
                    "is July 4th, 2000. How can I help you today?"
                )
            ]
        )
        assert (
            wrong_dob.decision.kind
            == DecisionKind.CORRECT_AND_STATE_OBJECTIVE
        )

        reason = await runtime.process_turns(
            [
                "Thanks, Alex.",
                "Let me check available appointments for you on Friday afternoon.",
                "Thanks for confirming your date of birth of April 12th, 1998.",
                (
                    "Can you tell me the reason for your visit? "
                    "For example, is this for a routine checkup, "
                    "a new patient consultation, a follow-up, or something else?"
                ),
            ]
        )
        assert reason.decision.kind == DecisionKind.ANSWER_VISIT_DETAILS
        assert "shoulder" in reason.decision.text.casefold()

        provider = await runtime.process_turns(
            [
                (
                    "We have openings on Friday afternoon with two providers. "
                    "Would you prefer Dr. Zygmunt-Lukowski or Dr. Kelly Noble "
                    "or is the first available okay?"
                )
            ]
        )
        assert (
            provider.decision.kind
            == DecisionKind.ANSWER_PROVIDER_PREFERENCE
        )
        assert provider.decision.text == "First available is fine."

        following_friday = await runtime.process_turns(
            [
                "There are no Friday afternoon openings on August 21st.",
                (
                    "Would you like to see afternoon openings on Friday, "
                    "August 28th, or check other days in the future?"
                ),
            ]
        )
        assert (
            following_friday.decision.kind
            == DecisionKind.CHOOSE_SEARCH_BRANCH
        )
        assert "August 28" in following_friday.decision.text

        final = runtime.flow_controller.tracker.snapshot()

        assert FlowStage.PROFILE in final.confirmed
        assert FlowStage.DOB in final.confirmed
        assert FlowStage.VISIT_REASON in final.communicated
        assert FlowStage.APPOINTMENT_TYPE in final.communicated
        assert FlowStage.PROVIDER in final.communicated
        assert FlowStage.DATE_TIME in final.communicated
        assert not final.complete

        assert runtime.metrics.fallback_decisions == 0

    asyncio.run(scenario())
