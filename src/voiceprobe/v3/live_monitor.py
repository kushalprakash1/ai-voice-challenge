"""Non-blocking listen-only audio monitor for VoiceProbe v3.

Both sides of the call are copied into one bounded queue and played through
one local ffplay process.

The monitor is strictly observational:
- no monitor operation may block the phone call
- full queues drop monitor audio
- ffplay failure never propagates into telephony
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import BinaryIO


ENV_LIVE_MONITOR = "VOICEPROBE_LIVE_MONITOR"

SAMPLE_RATE_HZ = 8_000
QUEUE_MAX_CHUNKS = 512

_TRUE_VALUES = frozenset(
    {
        "1",
        "true",
        "yes",
        "on",
    }
)


def live_monitor_enabled_from_environment() -> bool:
    return (
        os.environ.get(ENV_LIVE_MONITOR, "")
        .strip()
        .casefold()
        in _TRUE_VALUES
    )


@dataclass(frozen=True, slots=True)
class LiveMonitorStats:
    dropped_chunks: int


class LiveAudioMonitor:
    """One-process non-blocking monitor for inbound and outbound PCM."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        queue_max_chunks: int = QUEUE_MAX_CHUNKS,
    ) -> None:
        if queue_max_chunks <= 0:
            raise ValueError("queue_max_chunks must be positive")

        self.enabled = (
            live_monitor_enabled_from_environment()
            if enabled is None
            else bool(enabled)
        )

        self._queue: queue.Queue[bytes | None] = queue.Queue(
            maxsize=queue_max_chunks
        )

        self._process: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None

        self._lock = threading.Lock()
        self._dropped_chunks = 0
        self._entered = False

    @classmethod
    def from_environment(cls) -> "LiveAudioMonitor":
        return cls(
            enabled=live_monitor_enabled_from_environment(),
        )

    @property
    def stats(self) -> LiveMonitorStats:
        with self._lock:
            return LiveMonitorStats(
                dropped_chunks=self._dropped_chunks,
            )

    def __enter__(self) -> "LiveAudioMonitor":
        if self._entered:
            raise RuntimeError(
                "LiveAudioMonitor cannot be entered twice"
            )

        self._entered = True

        if not self.enabled:
            return self

        ffplay = shutil.which("ffplay")

        if ffplay is None:
            raise RuntimeError(
                "--live-monitor requested but ffplay was not found"
            )

        self._process = self._start_ffplay(ffplay)

        self._thread = threading.Thread(
            target=self._writer_loop,
            kwargs={
                "process": self._process,
            },
            name="voiceprobe-live-monitor",
            daemon=True,
        )
        self._thread.start()

        print(
            "VoiceProbe live monitor enabled "
            "(single mixed call stream via ffplay)"
        )

        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

    def observe_inbound(self, pcm16: bytes) -> None:
        self._observe(pcm16)

    def observe_outbound(self, pcm16: bytes) -> None:
        self._observe(pcm16)

    def _observe(self, pcm16: bytes) -> None:
        if not self.enabled or not pcm16:
            return

        try:
            self._queue.put_nowait(bytes(pcm16))
        except queue.Full:
            with self._lock:
                self._dropped_chunks += 1

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        thread = self._thread

        if thread is not None:
            thread.join(timeout=1.0)

        process = self._process

        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.5)
            except OSError:
                pass

        self._thread = None
        self._process = None

    @staticmethod
    def _start_ffplay(
        ffplay: str,
    ) -> subprocess.Popen[bytes]:
        env = os.environ.copy()

        pulse_socket = "/mnt/wslg/PulseServer"

        if (
            env.get("WSL_DISTRO_NAME")
            and os.path.exists(pulse_socket)
        ):
            env["SDL_AUDIODRIVER"] = "pulseaudio"
            env["PULSE_SERVER"] = f"unix:{pulse_socket}"

        # Deliberately keep this equivalent to the raw-PCM ffplay
        # configuration already proven to produce sound under WSLg.
        command = [
            ffplay,
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-sample_rate",
            str(SAMPLE_RATE_HZ),
            "-ch_layout",
            "mono",
            "-i",
            "pipe:0",
        ]

        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            env=env,
        )

    def _writer_loop(
        self,
        *,
        process: subprocess.Popen[bytes],
    ) -> None:
        stream: BinaryIO | None = process.stdin

        if stream is None:
            return

        try:
            while True:
                payload = self._queue.get()

                if payload is None:
                    return

                try:
                    stream.write(payload)
                    stream.flush()
                except (BrokenPipeError, OSError):
                    return
        finally:
            try:
                stream.close()
            except OSError:
                pass
