"""Safety constraints for outbound assessment calls."""

ALLOWED_TEST_NUMBER = "+18054398008"


class UnsafeDestinationError(ValueError):
    """Raised when a call targets anything other than the assessment line."""


def validate_destination(destination: str) -> str:
    """Return the destination only when it exactly matches the allowed test line.

    The comparison is intentionally strict. VoiceProbe does not normalize,
    reformat, or guess phone numbers because doing so would weaken the outbound
    call safety boundary.
    """
    if destination != ALLOWED_TEST_NUMBER:
        raise UnsafeDestinationError(
            f"Outbound calls are restricted to {ALLOWED_TEST_NUMBER}."
        )

    return destination
