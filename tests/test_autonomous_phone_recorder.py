from __future__ import annotations

import socket
import wave
from pathlib import Path

from voiceprobe.artifacts.recorder import RunArtifactRecorder
from voiceprobe.autonomous_phone import send_audio
from voiceprobe.scenarios.models import (
    PatientFacts,
    PatientScenario,
)


def build_scenario() -> PatientScenario:
    return PatientScenario(
        scenario_id="phone-recorder-test",
        objective="Schedule an appointment.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
        ),
    )


def test_send_audio_records_successfully_sent_pcm(
    tmp_path: Path,
) -> None:
    recorder = RunArtifactRecorder(
        root=tmp_path,
        scenario=build_scenario(),
        run_id="send-audio-run",
    )

    sender, receiver = socket.socketpair()

    try:
        # Exactly one 20 ms 8 kHz PCM16 frame:
        # 8000 samples/sec * 0.020 sec = 160 samples.
        pcm16 = b"\x00\x00" * 160

        send_audio(
            sender,
            pcm16,
            recorder=recorder,
        )

        packet = receiver.recv(3 + len(pcm16))

        assert packet[0] == 0x10
        assert int.from_bytes(
            packet[1:3],
            "big",
        ) == len(pcm16)
        assert packet[3:] == pcm16

    finally:
        sender.close()
        receiver.close()
        recorder.finalize()

    with wave.open(
        str(recorder.run_dir / "outbound.wav"),
        "rb",
    ) as audio:
        assert audio.getframerate() == 8_000
        assert audio.getnframes() == 160
