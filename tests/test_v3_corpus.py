from pathlib import Path

from voiceprobe.v3.corpus import load_regression_cases


def test_two_full_raw_live_logs_are_frozen_as_fixtures() -> None:
    root = Path(__file__).parent / "fixtures" / "v3_calls" / "raw"

    first = root / "call_1b2882d9-a28f-49a6-aaf6-3c413b301943.txt"
    second = root / "call_96f769c8-50b6-43cd-8a0a-39295b7791c3.txt"

    assert first.exists()
    assert second.exists()
    assert "Call UUID: 1b2882d9-a28f-49a6-aaf6-3c413b301943" in first.read_text(
        encoding="utf-8"
    )
    assert "Call UUID: 96f769c8-50b6-43cd-8a0a-39295b7791c3" in second.read_text(
        encoding="utf-8"
    )


def test_regression_corpus_contains_both_live_calls() -> None:
    cases = load_regression_cases()
    ids = {case["call_uuid"] for case in cases}

    assert ids == {
        "1b2882d9-a28f-49a6-aaf6-3c413b301943",
        "96f769c8-50b6-43cd-8a0a-39295b7791c3",
    }
    assert len(cases) >= 20
    assert sum(bool(case["critical"]) for case in cases) >= 10
