from voiceprobe.agents.brain import (
    CommunicationKind,
    PatientBrain,
)
from voiceprobe.conversation.grounding import ground_turn_meaning
from voiceprobe.conversation.meaning import (
    AppointmentOffer,
    FactAssertion,
    TurnMeaning,
)
from voiceprobe.conversation.objective import (
    AppointmentProgress,
    record_offer_accepted,
    record_slot_offer,
)
from voiceprobe.scenarios.models import (
    PatientFacts,
    PatientScenario,
)


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


def decide(
    meaning: TurnMeaning,
    *,
    progress: AppointmentProgress | None = None,
):
    scenario = build_scenario()

    grounded = ground_turn_meaning(
        scenario=scenario,
        meaning=meaning,
    )

    return PatientBrain().decide(
        scenario=scenario,
        grounded=grounded,
        progress=progress or AppointmentProgress(),
    )


def test_answers_requested_fact() -> None:
    decision = decide(
        TurnMeaning(
            requested_facts=("insurance",),
        )
    )

    assert decision.kind is CommunicationKind.ANSWER
    assert decision.facts_to_communicate == ("insurance",)


def test_correction_has_priority_over_answering() -> None:
    decision = decide(
        TurnMeaning(
            requested_facts=("complaint", "duration"),
            stated_facts=(
                FactAssertion(
                    fact="complaint",
                    value="left knee",
                ),
                FactAssertion(
                    fact="duration",
                    value="two weeks",
                ),
            ),
        )
    )

    assert decision.kind is CommunicationKind.CORRECT
    assert decision.facts_to_communicate == (
        "complaint",
        "duration",
    )


def test_repetition_request_is_recognized() -> None:
    decision = decide(
        TurnMeaning(
            requests_repetition=True,
        )
    )

    assert decision.kind is CommunicationKind.REPEAT


def test_matching_day_only_offer_is_partial() -> None:
    decision = decide(
        TurnMeaning(
            appointment_offer=AppointmentOffer(
                day="Friday",
                time=None,
            )
        )
    )

    assert decision.kind is CommunicationKind.ACCEPT_PARTIAL_OFFER
    assert decision.offered_day == "Friday"
    assert decision.offered_time is None
    assert decision.offered_day == "Friday"


def test_wrong_day_offer_is_declined() -> None:
    decision = decide(
        TurnMeaning(
            appointment_offer=AppointmentOffer(
                day="Tuesday",
                time=None,
            )
        )
    )

    assert decision.kind is CommunicationKind.DECLINE_OFFER
    assert decision.facts_to_communicate == (
        "preferred_day",
        "preferred_time",
    )


def test_booking_confirmation_without_acceptance_is_not_complete() -> None:
    decision = decide(
        TurnMeaning(
            booking_confirmed=True,
        )
    )

    assert decision.kind is CommunicationKind.CLARIFY


def test_confirmed_booking_after_acceptance_is_acknowledged() -> None:
    progress = record_slot_offer(
        AppointmentProgress(),
        day="Friday",
        time="2:30 PM",
    )
    progress = record_offer_accepted(progress)

    decision = decide(
        TurnMeaning(
            booking_confirmed=True,
            appointment_offer=AppointmentOffer(
                day="Friday",
                time="2:30 PM",
            ),
        ),
        progress=progress,
    )

    assert decision.kind is CommunicationKind.ACKNOWLEDGE_COMPLETE


def test_uninterpretable_turn_requests_clarification() -> None:
    decision = decide(
        TurnMeaning(
            unclear=True,
        )
    )

    assert decision.kind is CommunicationKind.CLARIFY


def test_afternoon_preference_accepts_230_pm_offer() -> None:
    decision = decide(
        TurnMeaning(
            appointment_offer=AppointmentOffer(
                day="Friday",
                time="2:30 PM",
            )
        )
    )

    assert decision.kind is CommunicationKind.ACCEPT_OFFER
    assert decision.offered_day == "Friday"
    assert decision.offered_time == "2:30 PM"


def test_booking_confirmation_for_different_slot_is_not_accepted() -> None:
    progress = record_slot_offer(
        AppointmentProgress(),
        day="Friday",
        time="2:30 PM",
    )
    progress = record_offer_accepted(progress)

    decision = decide(
        TurnMeaning(
            booking_confirmed=True,
            appointment_offer=AppointmentOffer(
                day="Tuesday",
                time="9:00 AM",
            ),
        ),
        progress=progress,
    )

    assert decision.kind is CommunicationKind.DECLINE_OFFER
    assert decision.offered_day == "Tuesday"
    assert decision.offered_time == "9:00 AM"
    assert decision.facts_to_communicate == (
        "preferred_day",
        "preferred_time",
    )


def test_afternoon_accepts_asr_style_230_pm_offer() -> None:
    decision = decide(
        TurnMeaning(
            appointment_offer=AppointmentOffer(
                day="Friday",
                time="2.30 p.m.",
            )
        )
    )

    assert decision.kind is CommunicationKind.ACCEPT_OFFER
