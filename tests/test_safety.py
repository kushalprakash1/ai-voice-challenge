import pytest

from voiceprobe.safety import (
    ALLOWED_TEST_NUMBER,
    DESTINATION_ENV,
    UnsafeDestinationError,
    require_live_destination,
    validate_destination,
)


def test_allows_assessment_number() -> None:
    assert validate_destination(ALLOWED_TEST_NUMBER) == ALLOWED_TEST_NUMBER


@pytest.mark.parametrize(
    "destination",
    [
        "+12025550109",
        "+12025550101",
        "2025550100",
        "(202) 555-0100",
        "",
    ],
)
def test_rejects_every_other_destination(destination: str) -> None:
    with pytest.raises(UnsafeDestinationError):
        validate_destination(destination)


def test_live_destination_requires_explicit_configuration(monkeypatch) -> None:
    monkeypatch.delenv(DESTINATION_ENV, raising=False)
    monkeypatch.chdir("/tmp")

    with pytest.raises(UnsafeDestinationError, match=DESTINATION_ENV):
        require_live_destination()


def test_fictional_default_is_rejected_for_live_calls(monkeypatch) -> None:
    monkeypatch.setenv(DESTINATION_ENV, ALLOWED_TEST_NUMBER)

    with pytest.raises(UnsafeDestinationError, match="fictional"):
        require_live_destination()


def test_explicit_live_destination_is_accepted(monkeypatch) -> None:
    monkeypatch.setenv(DESTINATION_ENV, "+12025550123")
    assert require_live_destination() == "+12025550123"
