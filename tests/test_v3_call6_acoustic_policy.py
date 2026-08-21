import struct
import pytest
from voiceprobe.v3 import live_turn_guard as guard

QUIET_MS=3_800; REQUIRED=30_400
def pcm(amplitude:int,samples:int=160)->bytes: return struct.pack(f'<{samples}h',*([amplitude]*samples))
def speech(samples=160,amplitude=900): return guard.note_remote_pcm_activity(pcm(amplitude,samples))
def silence(samples): guard.note_remote_pcm_activity(pcm(0,samples))
def semantic_eot(): guard.note_remote_speech_ended(); guard.authorize_response_for_current_remote_turn()
@pytest.fixture(autouse=True)
def live(monkeypatch): monkeypatch.setenv('VOICEPROBE_V3_LIVE','1'); guard.reset_for_tests()

def test_normal_requires_actual_received_silence_samples():
    guard.note_remote_speech_started(); assert speech(); semantic_eot()
    silence(REQUIRED-1); assert not guard.claim_response_commit_if_quiet(QUIET_MS)
    silence(1); assert guard.claim_response_commit_if_quiet(QUIET_MS); assert guard.final_prewrite_generation_check(QUIET_MS)

def test_long_pause_then_continuation_resets_full_counter():
    guard.note_remote_speech_started(); speech(); silence(24_000)
    assert not guard.claim_response_commit_if_quiet(QUIET_MS)
    assert speech(); semantic_eot(); silence(REQUIRED)
    assert guard.claim_response_commit_if_quiet(QUIET_MS)

def test_low_energy_tail_resets_counter():
    guard.note_remote_speech_started(); speech(); silence(800)
    assert speech(amplitude=45); semantic_eot(); silence(REQUIRED)
    assert guard.snapshot().silence_samples_since_last_remote_speech==REQUIRED
    assert guard.claim_response_commit_if_quiet(QUIET_MS)

def test_resume_at_3_7_seconds_invalidates_prepared_response():
    guard.note_remote_speech_started(); speech(); semantic_eot(); guard.note_decision_ready()
    silence(29_600); assert not guard.claim_response_commit_if_quiet(QUIET_MS)
    assert speech(); assert not guard.response_authorization_is_current()
    semantic_eot(); silence(REQUIRED); assert guard.claim_response_commit_if_quiet(QUIET_MS)

def test_tts_prepared_early_cannot_claim_during_speech():
    guard.note_remote_speech_started(); speech(); guard.note_decision_ready()
    assert not guard.claim_response_commit_if_quiet(QUIET_MS)

def test_resume_immediately_before_write_aborts_claim():
    guard.note_remote_speech_started(); speech(); semantic_eot(); silence(REQUIRED)
    assert guard.claim_response_commit_if_quiet(QUIET_MS)
    assert speech(); assert not guard.final_prewrite_generation_check(QUIET_MS)

def test_buffered_burst_wall_clock_is_irrelevant(monkeypatch):
    now=[1]; monkeypatch.setattr(guard.time,'monotonic_ns',lambda:now[0])
    guard.note_remote_speech_started(); speech(); semantic_eot(); now[0]+=99_000_000_000
    assert not guard.claim_response_commit_if_quiet(QUIET_MS)
    silence(24_000); assert speech()  # buffered continuation after only 3 s PCM
    semantic_eot(); silence(REQUIRED); assert guard.claim_response_commit_if_quiet(QUIET_MS)

def test_one_authorization_is_single_use():
    guard.note_remote_speech_started(); speech(); semantic_eot(); silence(REQUIRED)
    assert guard.claim_response_commit_if_quiet(QUIET_MS); assert guard.final_prewrite_generation_check(QUIET_MS)
    assert not guard.claim_response_commit_if_quiet(QUIET_MS)
