import pytest
from pydantic import ValidationError

from voiceprobe.conversation.meaning import (
    AppointmentOffer,
    FactAssertion,
    TurnMeaning,
)


def test_builds_requested_fact_meaning() -> None:
    meaning = TurnMeaning(
        requested_facts=("insurance",),
    )

    assert meaning.requested_facts == ("insurance",)
    assert meaning.stated_facts == ()


def test_builds_stated_fact_meaning() -> None:
    meaning = TurnMeaning(
        stated_facts=(
            FactAssertion(
                fact="complaint",
                value="left knee pain",
            ),
            FactAssertion(
                fact="duration",
                value="two weeks",
            ),
        ),
    )

    assert meaning.stated_facts[0].fact == "complaint"
    assert meaning.stated_facts[0].value == "left knee pain"

    assert meaning.stated_facts[1].fact == "duration"
    assert meaning.stated_facts[1].value == "two weeks"


def test_builds_appointment_offer() -> None:
    meaning = TurnMeaning(
        appointment_offer=AppointmentOffer(
            day="Friday",
            time="2:30 PM",
        )
    )

    assert meaning.appointment_offer is not None
    assert meaning.appointment_offer.day == "Friday"
    assert meaning.appointment_offer.time == "2:30 PM"


def test_rejects_empty_appointment_offer() -> None:
    with pytest.raises(ValidationError):
        AppointmentOffer()


def test_rejects_duplicate_requested_fact_keys() -> None:
    with pytest.raises(
        ValidationError,
        match="requested_facts cannot contain duplicates",
    ):
        TurnMeaning(
            requested_facts=(
                "insurance",
                "insurance",
            ),
        )


def test_rejects_duplicate_stated_fact_keys() -> None:
    with pytest.raises(
        ValidationError,
        match="stated_facts cannot contain duplicate fact keys",
    ):
        TurnMeaning(
            stated_facts=(
                FactAssertion(
                    fact="duration",
                    value="two weeks",
                ),
                FactAssertion(
                    fact="duration",
                    value="five days",
                ),
            ),
        )


def test_offer_can_contain_day_without_time() -> None:
    offer = AppointmentOffer(
        day="Friday",
        time=None,
    )

    assert offer.day == "Friday"
    assert offer.time is None


def test_offer_rejects_both_null() -> None:
    with pytest.raises(
        ValidationError,
        match="requires a day or time",
    ):
        AppointmentOffer(
            day=None,
            time=None,
        )


def test_fact_assertion_normalizes_whitespace() -> None:
    assertion = FactAssertion(
        fact="complaint",
        value="  left   knee pain  ",
    )

    assert assertion.value == "left knee pain"


def test_fact_assertion_rejects_blank_value() -> None:
    with pytest.raises(
        ValidationError,
        match="cannot be blank",
    ):
        FactAssertion(
            fact="complaint",
            value="   ",
        )


def test_turn_meaning_normalizes_empty_offer_to_none() -> None:
    meaning = TurnMeaning.model_validate(
        {
            "appointment_offer": {
                "day": None,
                "time": None,
            }
        }
    )

    assert meaning.appointment_offer is None
