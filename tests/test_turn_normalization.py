from voiceprobe.conversation.meaning import (
    AppointmentOffer,
    FactAssertion,
    TurnMeaning,
)
from voiceprobe.conversation.normalization import (
    normalize_turn_meaning,
)


def test_offer_derived_preference_assertions_are_removed() -> None:
    meaning = TurnMeaning(
        stated_facts=(
            FactAssertion(
                fact="preferred_day",
                value="Friday",
            ),
            FactAssertion(
                fact="preferred_time",
                value="2:30 PM",
            ),
        ),
        appointment_offer=AppointmentOffer(
            day="Friday",
            time="2:30 PM",
        ),
    )

    normalized = normalize_turn_meaning(meaning)

    assert normalized.stated_facts == ()
    assert normalized.appointment_offer is not None
    assert normalized.appointment_offer.day == "Friday"
    assert normalized.appointment_offer.time == "2:30 PM"


def test_non_scheduling_assertion_is_preserved() -> None:
    meaning = TurnMeaning(
        stated_facts=(
            FactAssertion(
                fact="insurance",
                value="Aetna",
            ),
            FactAssertion(
                fact="preferred_day",
                value="Friday",
            ),
        ),
        appointment_offer=AppointmentOffer(
            day="Friday",
            time="2:30 PM",
        ),
    )

    normalized = normalize_turn_meaning(meaning)

    assert normalized.stated_facts == (
        FactAssertion(
            fact="insurance",
            value="Aetna",
        ),
    )


def test_preference_assertion_without_offer_is_preserved() -> None:
    meaning = TurnMeaning(
        requested_facts=("preferred_day",),
        stated_facts=(
            FactAssertion(
                fact="preferred_day",
                value="Friday",
            ),
        ),
    )

    normalized = normalize_turn_meaning(meaning)

    assert normalized == meaning


def test_daypart_equivalent_to_offered_clock_is_removed() -> None:
    meaning = TurnMeaning(
        stated_facts=(
            FactAssertion(
                fact="preferred_time",
                value="afternoon",
            ),
        ),
        appointment_offer=AppointmentOffer(
            day="Friday",
            time="2:30 PM",
        ),
    )

    normalized = normalize_turn_meaning(meaning)

    assert normalized.stated_facts == ()


def test_booking_confirmation_promotes_slot_details() -> None:
    meaning = TurnMeaning(
        stated_facts=(
            FactAssertion(
                fact="preferred_day",
                value="Friday",
            ),
            FactAssertion(
                fact="preferred_time",
                value="2.30 p.m.",
            ),
        ),
        booking_confirmed=True,
    )

    normalized = normalize_turn_meaning(meaning)

    assert normalized.booking_confirmed

    assert normalized.appointment_offer == AppointmentOffer(
        day="Friday",
        time="2.30 p.m.",
    )

    assert normalized.stated_facts == ()


def test_wrong_booking_slot_is_preserved_for_validation() -> None:
    meaning = TurnMeaning(
        stated_facts=(
            FactAssertion(
                fact="preferred_day",
                value="Friday",
            ),
            FactAssertion(
                fact="preferred_time",
                value="10 a.m.",
            ),
        ),
        booking_confirmed=True,
    )

    normalized = normalize_turn_meaning(meaning)

    assert normalized.appointment_offer == AppointmentOffer(
        day="Friday",
        time="10 a.m.",
    )


def test_booking_without_slot_details_remains_slotless() -> None:
    meaning = TurnMeaning(
        booking_confirmed=True,
    )

    normalized = normalize_turn_meaning(meaning)

    assert normalized.appointment_offer is None


def test_explicit_you_are_booked_recovers_confirmation() -> None:
    meaning = TurnMeaning(
        appointment_offer=AppointmentOffer(
            day="Friday",
            time="2.30 p.m.",
        ),
        booking_confirmed=False,
    )

    normalized = normalize_turn_meaning(
        meaning,
        agent_turn=("Sorry, you're booked for Friday at 2.30 p.m."),
    )

    assert normalized.booking_confirmed


def test_offer_language_does_not_fake_booking_confirmation() -> None:
    meaning = TurnMeaning(
        appointment_offer=AppointmentOffer(
            day="Friday",
            time="2.30 p.m.",
        ),
        booking_confirmed=False,
    )

    normalized = normalize_turn_meaning(
        meaning,
        agent_turn=("I can book you for Friday at 2.30 p.m. if that works."),
    )

    assert not normalized.booking_confirmed


def test_recovers_single_duration_candidate_from_confirmation_question() -> None:
    normalized = normalize_turn_meaning(
        TurnMeaning(
            requested_facts=("duration",),
        ),
        agent_turn="And this has been happening for about three weeks?",
    )

    assert normalized.stated_facts == (
        FactAssertion(
            fact="duration",
            value="three weeks",
        ),
    )


def test_does_not_guess_between_multiple_duration_candidates() -> None:
    normalized = normalize_turn_meaning(
        TurnMeaning(
            requested_facts=("duration",),
        ),
        agent_turn="Was it three weeks or four weeks?",
    )

    assert normalized.stated_facts == ()


def test_recovers_slot_from_explicit_all_set_confirmation() -> None:
    normalized = normalize_turn_meaning(
        TurnMeaning(),
        agent_turn="You're all set for Tuesday at 9 AM.",
    )

    assert normalized.booking_confirmed
    assert normalized.appointment_offer == AppointmentOffer(
        day="Tuesday",
        time="9 AM",
    )


def test_recovers_slot_from_explicit_booked_confirmation() -> None:
    normalized = normalize_turn_meaning(
        TurnMeaning(),
        agent_turn="Great, you're booked for Friday at 2:30 PM.",
    )

    assert normalized.booking_confirmed
    assert normalized.appointment_offer == AppointmentOffer(
        day="Friday",
        time="2:30 PM",
    )


def test_recovers_explicit_terminal_goodbye() -> None:
    normalized = normalize_turn_meaning(
        TurnMeaning(),
        agent_turn="Okay, you're all set. Have a good day, goodbye.",
    )

    assert normalized.conversation_end_requested


def test_does_not_treat_nonterminal_goodbye_reference_as_call_end() -> None:
    normalized = normalize_turn_meaning(
        TurnMeaning(),
        agent_turn="Please don't say goodbye yet.",
    )

    assert not normalized.conversation_end_requested
