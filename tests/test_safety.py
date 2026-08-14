import pytest

from voiceprobe.safety import (
    ALLOWED_TEST_NUMBER,
    UnsafeDestinationError,
    validate_destination,
)


def test_allows_assessment_number() -> None:
    assert validate_destination(ALLOWED_TEST_NUMBER) == ALLOWED_TEST_NUMBER


@pytest.mark.parametrize(
    "destination",
    [
        "+18054398009",
        "+14155551212",
        "8054398008",
        "(805) 439-8008",
        "",
    ],
)
def test_rejects_every_other_destination(destination: str) -> None:
    with pytest.raises(UnsafeDestinationError):
        validate_destination(destination)
