"""Policy constraints for VoiceProbe assessment calls."""

import re
from dataclasses import dataclass

from voiceprobe.safety import ALLOWED_TEST_NUMBER

_E164_PATTERN = re.compile(r"^\+[1-9][0-9]{7,14}$")

DEFAULT_MAX_CALL_DURATION_SECONDS = 180
DEFAULT_MAX_SUITE_CALLS = 15


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
                "Originating number must use E.164 format, for example +14155551212."
            )

        if self.originating_number == ALLOWED_TEST_NUMBER:
            raise InvalidCallPolicyError(
                "Originating number cannot be the assessment destination."
            )

        if type(self.max_call_duration_seconds) is not int:
            raise InvalidCallPolicyError(
                "Maximum call duration must be an integer number of seconds."
            )

        if not 1 <= self.max_call_duration_seconds <= 180:
            raise InvalidCallPolicyError(
                "Maximum call duration must be between 1 and 180 seconds."
            )

        if type(self.max_suite_calls) is not int:
            raise InvalidCallPolicyError(
                "Maximum suite size must be an integer number of calls."
            )

        if not 1 <= self.max_suite_calls <= 15:
            raise InvalidCallPolicyError(
                "Maximum suite size must be between 1 and 15 calls."
            )

        if type(self.dry_run) is not bool:
            raise InvalidCallPolicyError("dry_run must be a boolean.")

    @property
    def destination(self) -> str:
        """Return the only destination VoiceProbe is authorized to call."""
        return ALLOWED_TEST_NUMBER
