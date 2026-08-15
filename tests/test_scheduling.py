from voiceprobe.conversation.scheduling import (
    parse_clock_minutes,
    time_matches_preference,
)


def test_parses_twelve_hour_clock() -> None:
    assert parse_clock_minutes("2:30 PM") == 14 * 60 + 30


def test_afternoon_matches_afternoon_clock_time() -> None:
    assert time_matches_preference(
        preferred="afternoon",
        offered="2:30 PM",
    )


def test_afternoon_rejects_morning_clock_time() -> None:
    assert not time_matches_preference(
        preferred="afternoon",
        offered="9:30 AM",
    )


def test_morning_matches_morning_clock_time() -> None:
    assert time_matches_preference(
        preferred="morning",
        offered="9:00 AM",
    )


def test_equal_explicit_times_match() -> None:
    assert time_matches_preference(
        preferred="2:30 PM",
        offered="2:30 PM",
    )


def test_unrecognized_different_values_do_not_match() -> None:
    assert not time_matches_preference(
        preferred="after lunch",
        offered="9:00 AM",
    )


def test_afternoon_matches_asr_style_230_pm() -> None:
    assert time_matches_preference(
        preferred="afternoon",
        offered="2.30 p.m.",
    )


def test_afternoon_rejects_asr_style_930_am() -> None:
    assert not time_matches_preference(
        preferred="afternoon",
        offered="9.30 a.m.",
    )


def test_parses_asr_style_period_clock() -> None:
    assert parse_clock_minutes("2.30 p.m.") == 14 * 60 + 30
