"""Policy constraints for VoiceProbe assessment calls."""

import re
from dataclasses import dataclass

from voiceprobe.safety import destination_for_plan

_E164_PATTERN = re.compile(r"^\+[1-9][0-9]{7,14}$")

DEFAULT_MAX_CALL_DURATION_SECONDS = 180
MAX_CALL_DURATION_SECONDS = 600
DEFAULT_MAX_SUITE_CALLS = 16


class InvalidCallPolicyError(ValueError):
    """Raised when an outbound call policy violates assessment constraints."""


@dataclass(frozen=True, slots=True)
class CallPolicy:
    """Immutable limits applied before any outbound call can be attempted."""

    originating_number: str
    max_call_duration_seconds: int = DEFAULT_MAX_CALL_DURATION_SECONDS
    max_suite_calls: int = DEFAULT_MAX_SUITE_CALLS
    dry_run: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.originating_number, str) or not _E164_PATTERN.fullmatch(
            self.originating_number
        ):
            raise InvalidCallPolicyError(
                "Originating number must use E.164 format, for example +12025550101."
            )

        if self.originating_number == destination_for_plan():
            raise InvalidCallPolicyError(
                "Originating number cannot be the assessment destination."
            )

        if type(self.max_call_duration_seconds) is not int:
            raise InvalidCallPolicyError(
                "Maximum call duration must be an integer number of seconds."
            )

        if not 1 <= self.max_call_duration_seconds <= MAX_CALL_DURATION_SECONDS:
            raise InvalidCallPolicyError(
                f"Maximum call duration must be between 1 and {MAX_CALL_DURATION_SECONDS} seconds."
            )

        if type(self.max_suite_calls) is not int:
            raise InvalidCallPolicyError(
                "Maximum suite size must be an integer number of calls."
            )

        if not 1 <= self.max_suite_calls <= DEFAULT_MAX_SUITE_CALLS:
            raise InvalidCallPolicyError(
                f"Maximum suite size must be between 1 and {DEFAULT_MAX_SUITE_CALLS} calls."
            )

        if type(self.dry_run) is not bool:
            raise InvalidCallPolicyError("dry_run must be a boolean.")

    @property
    def destination(self) -> str:
        """Return the only destination VoiceProbe is authorized to call."""
        return destination_for_plan()
