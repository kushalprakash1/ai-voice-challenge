"""Live remote-turn authorization in the ordered inbound PCM sample domain."""
from __future__ import annotations
import os, struct, threading, time
from collections.abc import Callable
from dataclasses import dataclass

_TRUTHY={"1","true","yes","on"}; _LOCK=threading.RLock()
_GENERATION=0; _REMOTE_SPEAKING=False; _ACOUSTIC_SPEECH_ACTIVE=False
_AUTHORIZED_GENERATION=None; _CLAIMED_GENERATION=None
_REMOTE_SPEECH_ENDED_AT_NS=None; _TARGET_LAST_AUDIO_ACTIVITY_NS=None
_STABLE_TURN_COMMIT_NS=None; _DECISION_READY_NS=None; _PLAYBACK_START_NS=None
_REMOTE_SAMPLE_CURSOR=0; _LAST_REMOTE_SPEECH_SAMPLE_INDEX=None; _REMOTE_PCM_NOISE_RMS=24.0
TURN_MODE_ENV="VOICEPROBE_TURN_MODE"; TURN_MODE_BASELINE="baseline"; TURN_MODE_CONSERVATIVE="conservative"
TELEPHONY_SAMPLE_RATE=8_000; CONSERVATIVE_COMMIT_QUIET_MS=3_800.0
REQUIRED_SILENCE_SAMPLES=30_400; REMOTE_PCM_RMS_FLOOR=55
_DETECTOR_WINDOW_SAMPLES=80; _TRAILING_HOLD_SAMPLES=TELEPHONY_SAMPLE_RATE

@dataclass(frozen=True,slots=True)
class LiveTurnGuardSnapshot:
    generation:int; remote_speaking:bool; authorized_generation:int|None; enabled:bool
    target_last_audio_activity_ns:int|None; stable_turn_commit_ns:int|None
    decision_ready_ns:int|None; playback_start_ns:int|None; flux_eot_ns:int|None
    remote_quiet_before_playback_ms:float|None; remote_sample_cursor:int
    last_remote_speech_sample_index:int|None; silence_samples_since_last_remote_speech:int
    acoustic_speech_active:bool

def turn_mode_from_environment()->str:
    value=os.getenv(TURN_MODE_ENV,TURN_MODE_BASELINE).strip().casefold()
    if value not in {TURN_MODE_BASELINE,TURN_MODE_CONSERVATIVE}: raise ValueError(f"{TURN_MODE_ENV} must be baseline or conservative")
    return value
def outbound_commit_quiet_ms_from_environment()->float: return CONSERVATIVE_COMMIT_QUIET_MS
def _enabled()->bool: return os.getenv("VOICEPROBE_V3_LIVE","").strip().casefold() in _TRUTHY
def _required_samples(quiet_ms:float)->int: return round(quiet_ms*TELEPHONY_SAMPLE_RATE/1_000)
def _silence_samples()->int:
    return 0 if _LAST_REMOTE_SPEECH_SAMPLE_INDEX is None else max(0,_REMOTE_SAMPLE_CURSOR-_LAST_REMOTE_SPEECH_SAMPLE_INDEX)

def note_remote_speech_started()->None:
    global _GENERATION,_REMOTE_SPEAKING,_REMOTE_SPEECH_ENDED_AT_NS
    if not _enabled(): return
    with _LOCK: _GENERATION+=1; _REMOTE_SPEAKING=True; _REMOTE_SPEECH_ENDED_AT_NS=None

def note_remote_pcm_activity(pcm16:bytes,*,rms_floor:int=REMOTE_PCM_RMS_FLOOR)->bool:
    """Consume every inbound sample; silence advances the cursor but not time."""
    if not _enabled() or not pcm16 or len(pcm16)%2: return False
    samples=struct.unpack(f"<{len(pcm16)//2}h",pcm16); detected=False
    global _GENERATION,_ACOUSTIC_SPEECH_ACTIVE,_REMOTE_SAMPLE_CURSOR
    global _LAST_REMOTE_SPEECH_SAMPLE_INDEX,_REMOTE_PCM_NOISE_RMS,_TARGET_LAST_AUDIO_ACTIVITY_NS
    with _LOCK:
        frame_had_speech=False
        for offset in range(0,len(samples),_DETECTOR_WINDOW_SAMPLES):
            window=samples[offset:offset+_DETECTOR_WINDOW_SAMPLES]
            rms=(sum(v*v for v in window)/len(window))**0.5
            since=(_REMOTE_SAMPLE_CURSOR-_LAST_REMOTE_SPEECH_SAMPLE_INDEX if _LAST_REMOTE_SPEECH_SAMPLE_INDEX is not None else _TRAILING_HOLD_SAMPLES+1)
            trailing=_ACOUSTIC_SPEECH_ACTIVE or since<=_TRAILING_HOLD_SAMPLES
            threshold=max(rms_floor*.55,_REMOTE_PCM_NOISE_RMS*1.7) if trailing else max(float(rms_floor),_REMOTE_PCM_NOISE_RMS*2.8)
            window_end=_REMOTE_SAMPLE_CURSOR+len(window)
            if rms>=threshold:
                _LAST_REMOTE_SPEECH_SAMPLE_INDEX=window_end; _GENERATION+=1
                frame_had_speech=True; detected=True; _TARGET_LAST_AUDIO_ACTIVITY_NS=time.monotonic_ns()
            else: _REMOTE_PCM_NOISE_RMS=.995*_REMOTE_PCM_NOISE_RMS+.005*rms
            _REMOTE_SAMPLE_CURSOR=window_end
        _ACOUSTIC_SPEECH_ACTIVE=frame_had_speech
    return detected

def note_remote_speech_ended()->None:
    global _REMOTE_SPEAKING,_REMOTE_SPEECH_ENDED_AT_NS
    if not _enabled(): return
    with _LOCK: _REMOTE_SPEAKING=False; _REMOTE_SPEECH_ENDED_AT_NS=time.monotonic_ns()
def authorize_response_for_current_remote_turn()->None:
    global _AUTHORIZED_GENERATION,_STABLE_TURN_COMMIT_NS,_CLAIMED_GENERATION
    if not _enabled(): return
    with _LOCK: _AUTHORIZED_GENERATION=_GENERATION; _CLAIMED_GENERATION=None; _STABLE_TURN_COMMIT_NS=time.monotonic_ns()
def note_decision_ready()->None:
    global _DECISION_READY_NS
    with _LOCK: _DECISION_READY_NS=time.monotonic_ns()
def note_playback_started()->None:
    global _PLAYBACK_START_NS
    with _LOCK: _PLAYBACK_START_NS=time.monotonic_ns()
def response_authorization_is_current()->bool:
    if not _enabled(): return True
    with _LOCK: return _AUTHORIZED_GENERATION is not None and _AUTHORIZED_GENERATION==_GENERATION and not _REMOTE_SPEAKING
def response_commit_quiet_remaining_seconds(quiet_ms:float)->float|None:
    """Compatibility status; irreversible authorization never uses wall time."""
    if not _enabled() or quiet_ms<=0: return 0.0 if response_authorization_is_current() else None
    with _LOCK:
        if not response_authorization_is_current(): return None
        return max(0,_required_samples(quiet_ms)-_silence_samples())/TELEPHONY_SAMPLE_RATE
def claim_response_commit_if_quiet(quiet_ms:float)->bool:
    global _AUTHORIZED_GENERATION,_CLAIMED_GENERATION
    if not _enabled(): return True
    with _LOCK:
        if (_AUTHORIZED_GENERATION is None or _AUTHORIZED_GENERATION!=_GENERATION or _REMOTE_SPEAKING or _ACOUSTIC_SPEECH_ACTIVE or _REMOTE_SPEECH_ENDED_AT_NS is None or _silence_samples()<_required_samples(quiet_ms)): return False
        _CLAIMED_GENERATION=_AUTHORIZED_GENERATION; _AUTHORIZED_GENERATION=None; return True
def final_prewrite_generation_check(quiet_ms:float=CONSERVATIVE_COMMIT_QUIET_MS)->bool:
    global _CLAIMED_GENERATION
    if not _enabled(): return True
    with _LOCK:
        current=_CLAIMED_GENERATION is not None and _CLAIMED_GENERATION==_GENERATION and not _REMOTE_SPEAKING and not _ACOUSTIC_SPEECH_ACTIVE and _silence_samples()>=_required_samples(quiet_ms)
        _CLAIMED_GENERATION=None; return current
def write_first_outbound_pcm_if_authorized(write:Callable[[],None],quiet_ms:float=CONSERVATIVE_COMMIT_QUIET_MS)->bool:
    """Validate and release the first wire write atomically with ingress state."""
    global _CLAIMED_GENERATION
    if not _enabled(): write(); return True
    with _LOCK:
        current=_CLAIMED_GENERATION is not None and _CLAIMED_GENERATION==_GENERATION and not _REMOTE_SPEAKING and not _ACOUSTIC_SPEECH_ACTIVE and _silence_samples()>=_required_samples(quiet_ms)
        _CLAIMED_GENERATION=None
        if not current: return False
        write(); return True
def snapshot()->LiveTurnGuardSnapshot:
    with _LOCK:
        silence=_silence_samples()
        return LiveTurnGuardSnapshot(_GENERATION,_REMOTE_SPEAKING,_AUTHORIZED_GENERATION,_enabled(),_TARGET_LAST_AUDIO_ACTIVITY_NS,_STABLE_TURN_COMMIT_NS,_DECISION_READY_NS,_PLAYBACK_START_NS,_REMOTE_SPEECH_ENDED_AT_NS,silence*1_000/TELEPHONY_SAMPLE_RATE,_REMOTE_SAMPLE_CURSOR,_LAST_REMOTE_SPEECH_SAMPLE_INDEX,silence,_ACOUSTIC_SPEECH_ACTIVE)
def reset_for_tests()->None:
    global _GENERATION,_REMOTE_SPEAKING,_ACOUSTIC_SPEECH_ACTIVE,_AUTHORIZED_GENERATION,_CLAIMED_GENERATION
    global _REMOTE_SPEECH_ENDED_AT_NS,_TARGET_LAST_AUDIO_ACTIVITY_NS,_STABLE_TURN_COMMIT_NS
    global _DECISION_READY_NS,_PLAYBACK_START_NS,_REMOTE_SAMPLE_CURSOR,_LAST_REMOTE_SPEECH_SAMPLE_INDEX,_REMOTE_PCM_NOISE_RMS
    with _LOCK:
        _GENERATION=0; _REMOTE_SPEAKING=False; _ACOUSTIC_SPEECH_ACTIVE=False; _AUTHORIZED_GENERATION=None; _CLAIMED_GENERATION=None
        _REMOTE_SPEECH_ENDED_AT_NS=None; _TARGET_LAST_AUDIO_ACTIVITY_NS=None; _STABLE_TURN_COMMIT_NS=None; _DECISION_READY_NS=None; _PLAYBACK_START_NS=None
        _REMOTE_SAMPLE_CURSOR=0; _LAST_REMOTE_SPEECH_SAMPLE_INDEX=None; _REMOTE_PCM_NOISE_RMS=24.0
