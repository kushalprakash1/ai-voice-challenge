"""Safety constraints for outbound VoiceProbe calls."""

from __future__ import annotations

import os

from dotenv import dotenv_values

EXAMPLE_DESTINATION_NUMBER = "+12025550100"
DESTINATION_ENV = "VOICEPROBE_DESTINATION_NUMBER"
# Compatibility name for dry-run plans and existing callers.
ALLOWED_TEST_NUMBER = EXAMPLE_DESTINATION_NUMBER


class UnsafeDestinationError(ValueError):
    """Raised when a call targets anything other than the configured line."""


def configured_destination() -> str | None:
    """Return the explicitly configured destination without exposing it."""
    environment_value = os.environ.get(DESTINATION_ENV, "").strip()
    if environment_value:
        return environment_value

    dotenv_value = dotenv_values(".env").get(DESTINATION_ENV)
    if isinstance(dotenv_value, str) and dotenv_value.strip():
        return dotenv_value.strip()

    return None


def destination_for_plan() -> str:
    """Use a fictional destination only for non-live planning."""
    return configured_destination() or EXAMPLE_DESTINATION_NUMBER


def require_live_destination() -> str:
    """Require an explicit destination before live authorization."""
    destination = configured_destination()
    if destination is None:
        raise UnsafeDestinationError(
            f"Live calls require {DESTINATION_ENV} to be configured explicitly."
        )
    if destination == EXAMPLE_DESTINATION_NUMBER:
        raise UnsafeDestinationError(
            f"{DESTINATION_ENV} must not use the fictional example number."
        )
    return destination


def validate_destination(destination: str) -> str:
    """Return the destination only when it matches current configuration.

    The comparison is intentionally strict. VoiceProbe does not normalize,
    reformat, or guess phone numbers because doing so would weaken the outbound
    call safety boundary.
    """
    expected = destination_for_plan()
    if destination != expected:
        raise UnsafeDestinationError(
            "Outbound destination does not match the configured destination."
        )

    return destination
