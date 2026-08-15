from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from voiceprobe.artifacts.recorder import (
    RunArtifactRecorder,
)
from voiceprobe.scenarios.models import (
    PatientFacts,
    PatientScenario,
)


def build_scenario() -> PatientScenario:
    return PatientScenario(
        scenario_id="artifact-test",
        objective="Schedule an appointment.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            insurance="Blue Cross",
        ),
    )


def test_recorder_materializes_private_run_artifacts(
    tmp_path: Path,
) -> None:
    recorder = RunArtifactRecorder(
        root=tmp_path,
        scenario=build_scenario(),
        run_id="test-run",
    )

    recorder.record_event(
        "asr_final",
        text="What insurance do you have?",
    )

    recorder.record_transcript_turn(
        speaker="agent",
        text="What insurance do you have?",
    )

    recorder.record_transcript_turn(
        speaker="patient",
        text="I have Blue Cross insurance.",
        decision="answer",
    )

    recorder.record_turn_metrics(
        {
            "interpreter_seconds": 1.5,
            "verbalizer_seconds": 0.6,
            "tts_seconds": 1.2,
            "response_prep_seconds": 4.0,
            "objective_complete": False,
        }
    )

    recorder.finalize(
        call_id="test-call-id",
    )

    expected = {
        "manifest.json",
        "scenario.json",
        "events.jsonl",
        "transcript.json",
        "transcript.txt",
        "metrics.json",
        "inbound.wav",
        "outbound.wav",
        "call.wav",
        "call.ogg",
    }

    assert {path.name for path in recorder.run_dir.iterdir()} == expected


def test_transcript_json_and_text_preserve_turns(
    tmp_path: Path,
) -> None:
    recorder = RunArtifactRecorder(
        root=tmp_path,
        scenario=build_scenario(),
        run_id="transcript-run",
    )

    recorder.record_transcript_turn(
        speaker="agent",
        text="What insurance do you have?",
    )

    recorder.record_transcript_turn(
        speaker="patient",
        text="I have Blue Cross insurance.",
        decision="answer",
    )

    recorder.finalize()

    transcript = json.loads((recorder.run_dir / "transcript.json").read_text())

    assert len(transcript["turns"]) == 2
    assert transcript["turns"][0]["speaker"] == "agent"
    assert transcript["turns"][1]["speaker"] == "patient"

    text = (recorder.run_dir / "transcript.txt").read_text()

    assert "AGENT: What insurance do you have?" in text
    assert "PATIENT: I have Blue Cross insurance." in text


def test_pcm_audio_is_written_as_valid_8khz_mono_wav(
    tmp_path: Path,
) -> None:
    recorder = RunArtifactRecorder(
        root=tmp_path,
        scenario=build_scenario(),
        run_id="audio-run",
    )

    # 100 ms of silence at 8 kHz, signed 16-bit mono.
    pcm = b"\x00\x00" * 800

    recorder.record_inbound_pcm(pcm)
    recorder.record_outbound_pcm(pcm)
    recorder.finalize()

    for filename in (
        "inbound.wav",
        "outbound.wav",
    ):
        with wave.open(
            str(recorder.run_dir / filename),
            "rb",
        ) as audio:
            assert audio.getframerate() == 8_000
            assert audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
            assert audio.getnframes() == 800


def test_metrics_summary_computes_means(
    tmp_path: Path,
) -> None:
    recorder = RunArtifactRecorder(
        root=tmp_path,
        scenario=build_scenario(),
        run_id="metrics-run",
    )

    recorder.record_turn_metrics(
        {
            "response_prep_seconds": 4.0,
            "tts_seconds": 1.0,
            "objective_complete": False,
        }
    )

    recorder.record_turn_metrics(
        {
            "response_prep_seconds": 6.0,
            "tts_seconds": 2.0,
            "objective_complete": True,
        }
    )

    recorder.finalize()

    metrics = json.loads((recorder.run_dir / "metrics.json").read_text())

    summary = metrics["summary"]

    assert summary["turn_count"] == 2
    assert summary["mean_response_prep_seconds"] == 5.0
    assert summary["mean_tts_seconds"] == 1.5
    assert summary["objective_complete"] is True


def test_invalid_pcm16_payload_is_rejected(
    tmp_path: Path,
) -> None:
    recorder = RunArtifactRecorder(
        root=tmp_path,
        scenario=build_scenario(),
        run_id="invalid-pcm-run",
    )

    try:
        with pytest.raises(
            ValueError,
            match="divisible by two",
        ):
            recorder.record_inbound_pcm(b"\x00")
    finally:
        recorder.finalize()


def test_finalize_is_idempotent(
    tmp_path: Path,
) -> None:
    recorder = RunArtifactRecorder(
        root=tmp_path,
        scenario=build_scenario(),
        run_id="idempotent-run",
    )

    recorder.finalize()
    recorder.finalize()

    manifest = json.loads((recorder.run_dir / "manifest.json").read_text())

    assert manifest["status"] == "completed"
