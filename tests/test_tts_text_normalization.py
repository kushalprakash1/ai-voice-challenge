from voiceprobe.tts.telephony import normalize_text_for_tts


def test_dot_clock_time_is_normalized_for_speech() -> None:
    assert (
        normalize_text_for_tts("Friday at 2.30 p.m. works for me.")
        == "Friday at 2:30 PM works for me."
    )


def test_colon_clock_time_normalizes_meridiem() -> None:
    assert (
        normalize_text_for_tts("Friday at 2:30 p.m. works for me.")
        == "Friday at 2:30 PM works for me."
    )


def test_compact_clock_time_is_normalized() -> None:
    assert (
        normalize_text_for_tts("11:30 would be too early.")
        == "11:30 would be too early."
    )

    assert (
        normalize_text_for_tts("The appointment is at 1130 a.m.")
        == "The appointment is at 11:30 AM."
    )


def test_decimal_without_meridiem_is_untouched() -> None:
    assert normalize_text_for_tts("The result was 2.30.") == "The result was 2.30."
