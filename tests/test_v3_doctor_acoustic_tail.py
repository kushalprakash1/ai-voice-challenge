import struct

import pytest

from voiceprobe.v3 import live_turn_guard as guard


def energetic_pcm():
    return struct.pack("<160h", *([1200] * 160))


def test_post_flux_pcm_tail_invalidates_and_restarts_quiet_clock_in_baseline(monkeypatch):
    monkeypatch.setenv("VOICEPROBE_V3_LIVE", "1")
    monkeypatch.setenv("VOICEPROBE_TURN_MODE", "baseline")
    now = [100_000_000_000]
    monkeypatch.setattr(guard.time, "monotonic_ns", lambda: now[0])
    guard.reset_for_tests()
    guard.note_remote_speech_started()
    guard.note_remote_speech_ended()  # semantic/Flux completion
    guard.authorize_response_for_current_remote_turn()

    now[0] += 500_000_000  # target continues acoustically for 500 ms
    assert guard.note_remote_pcm_activity(energetic_pcm())
    assert not guard.response_authorization_is_current()
    assert not guard.claim_response_commit_if_quiet(3200)

    guard.note_remote_speech_ended()  # trailing sentence receives final EOT
    guard.authorize_response_for_current_remote_turn()
    # Quiet authorization is measured from received PCM samples, not wall time.
    assert guard.response_commit_quiet_remaining_seconds(3200) == pytest.approx(3.2)
    guard.note_remote_pcm_activity(b"\0\0" * 25_599)
    assert not guard.claim_response_commit_if_quiet(3200)
    guard.note_remote_pcm_activity(b"\0\0")
    assert guard.claim_response_commit_if_quiet(3200)


def test_silence_pcm_does_not_create_false_tail_delay(monkeypatch):
    monkeypatch.setenv("VOICEPROBE_V3_LIVE", "1")
    now = [200_000_000_000]
    monkeypatch.setattr(guard.time, "monotonic_ns", lambda: now[0])
    guard.reset_for_tests()
    guard.note_remote_speech_started()
    assert guard.note_remote_pcm_activity(energetic_pcm())
    guard.note_remote_speech_ended()
    guard.authorize_response_for_current_remote_turn()
    assert not guard.note_remote_pcm_activity(b"\0\0" * 160)
    guard.note_remote_pcm_activity(b"\0\0" * (25_600 - 160))
    assert guard.claim_response_commit_if_quiet(3200)
