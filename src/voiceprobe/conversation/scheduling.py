"""Scheduling-value normalization for VoiceProbe.

Appointment reasoning should understand common clock times and dayparts
without constructing dates or relying on fragile substring matching.
"""

from __future__ import annotations

import re

_TWELVE_HOUR_PATTERN = re.compile(
    r"^(?P<hour>\d{1,2})"
    r"(?::(?P<minute>\d{2}))?"
    r"\s*(?P<period>am|pm)$",
)

_TWENTY_FOUR_HOUR_PATTERN = re.compile(
    r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})$",
)


def _normalize(value: str) -> str:
    normalized = " ".join(value.casefold().split())

    # Telephone ASR commonly renders spoken clock times as
    # "2.30 p.m." rather than "2:30 PM". Normalize those harmless
    # surface differences before deterministic time reasoning.
    normalized = (
        normalized.replace("a.m.", "am")
        .replace("p.m.", "pm")
        .replace("a.m", "am")
        .replace("p.m", "pm")
    )

    normalized = re.sub(
        r"(?<=\d)\.(?=\d{2}(?:\s|$))",
        ":",
        normalized,
    )

    return normalized


def parse_clock_minutes(value: str) -> int | None:
    """Parse a common clock expression into minutes after midnight."""
    normalized = _normalize(value)

    twelve_hour_match = _TWELVE_HOUR_PATTERN.fullmatch(normalized)

    if twelve_hour_match is not None:
        hour = int(twelve_hour_match.group("hour"))
        minute_text = twelve_hour_match.group("minute")
        minute = int(minute_text) if minute_text is not None else 0
        period = twelve_hour_match.group("period")

        if not 1 <= hour <= 12 or not 0 <= minute <= 59:
            return None

        hour %= 12

        if period == "pm":
            hour += 12

        return hour * 60 + minute

    twenty_four_hour_match = _TWENTY_FOUR_HOUR_PATTERN.fullmatch(normalized)

    if twenty_four_hour_match is None:
        return None

    hour = int(twenty_four_hour_match.group("hour"))
    minute = int(twenty_four_hour_match.group("minute"))

    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return None

    return hour * 60 + minute


def clock_matches_daypart(
    *,
    minutes: int,
    daypart: str,
) -> bool:
    """Return whether a clock time belongs to a named daypart."""
    normalized = _normalize(daypart)

    if normalized == "morning":
        return 5 * 60 <= minutes < 12 * 60

    if normalized == "afternoon":
        return 12 * 60 <= minutes < 17 * 60

    if normalized == "evening":
        return 17 * 60 <= minutes < 21 * 60

    if normalized == "night":
        return minutes >= 21 * 60 or minutes < 5 * 60

    return False


def time_matches_preference(
    *,
    preferred: str | None,
    offered: str | None,
) -> bool:
    """Compare an offered appointment time with a patient preference."""
    if preferred is None or offered is None:
        return True

    preferred_normalized = _normalize(preferred)
    offered_normalized = _normalize(offered)

    if preferred_normalized == offered_normalized:
        return True

    offered_minutes = parse_clock_minutes(offered)

    if offered_minutes is not None and preferred_normalized in {
        "morning",
        "afternoon",
        "evening",
        "night",
    }:
        return clock_matches_daypart(
            minutes=offered_minutes,
            daypart=preferred_normalized,
        )

    preferred_minutes = parse_clock_minutes(preferred)

    if preferred_minutes is not None and offered_minutes is not None:
        return preferred_minutes == offered_minutes

    return False
