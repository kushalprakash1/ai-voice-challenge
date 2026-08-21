import queue
import threading

from voiceprobe.v3.audiosocket_kokoro import (
    AudioSocketKokoroSpeechTask,
    AudioSocketV3MediaBoundary,
)
from voiceprobe.v3.live_monitor import LiveAudioMonitor


def test_disabled_monitor_is_noop():
    monitor = LiveAudioMonitor(enabled=False)

    monitor.observe_inbound(b"\x01\x00")
    monitor.observe_outbound(b"\x02\x00")

    assert monitor.stats.dropped_chunks == 0


def test_full_monitor_queue_drops_instead_of_blocking():
    monitor = LiveAudioMonitor(
        enabled=True,
        queue_max_chunks=1,
    )

    monitor._queue.put_nowait(b"\x00\x00")

    monitor.observe_inbound(b"\x01\x00")

    assert monitor.stats.dropped_chunks == 1


def test_outbound_observer_failure_cannot_break_phone_send():
    sent = []

    def broken_observer(payload):
        del payload
        raise RuntimeError("local monitor failed")

    def send_audio(connection, pcm16, **kwargs):
        del connection, kwargs
        sent.append(pcm16)

    task = AudioSocketKokoroSpeechTask(
        connection=object(),
        renderer=object(),
        send_lock=threading.Lock(),
        send_audio_fn=send_audio,
        audio_observer=broken_observer,
    )

    payload = b"\x01\x00\x02\x00"

    task._send_audio(payload)

    assert sent == [payload]


def test_inbound_monitor_observes_even_during_playback_guard():
    observed = []
    submitted = []

    class SpeechTask:
        def __init__(self):
            self.playback_active = threading.Event()
            self.playback_active.set()

    boundary = AudioSocketV3MediaBoundary(
        connection=object(),
        speech_task=SpeechTask(),
        send_lock=threading.Lock(),
        audio_observer=observed.append,
    )

    payload = b"\x01\x00\x02\x00"

    forwarded = boundary.forward_inbound_pcm(
        payload,
        submit_pcm=submitted.append,
    )

    assert forwarded is False
    assert observed == [payload]
    assert submitted == []
