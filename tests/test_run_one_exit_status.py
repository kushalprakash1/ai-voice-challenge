from voiceprobe.run_one import _execution_exit_code


def test_run_one_exit_code_reflects_failed_calls() -> None:
    assert _execution_exit_code(0) == 0
    assert _execution_exit_code(1) == 1
    assert _execution_exit_code(3) == 1
