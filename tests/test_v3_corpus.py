from pathlib import Path

from voiceprobe.v3.corpus import load_regression_cases


def test_two_sanitized_terminal_examples_are_preserved() -> None:
    root = Path(__file__).parent / "fixtures" / "v3_calls" / "raw"

    first = root / "call_example_001.txt"
    second = root / "call_example_002.txt"

    assert first.exists()
    assert second.exists()
    assert "Call ID: example-call-001" in first.read_text(encoding="utf-8")
    assert "Call ID: example-call-002" in second.read_text(encoding="utf-8")


def test_regression_corpus_contains_both_live_calls() -> None:
    cases = load_regression_cases()
    ids = {case["call_uuid"] for case in cases}

    assert ids == {
        "example-call-001",
        "example-call-002",
    }
    assert len(cases) >= 20
    assert sum(bool(case["critical"]) for case in cases) >= 10
