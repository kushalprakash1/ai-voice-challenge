"""Crash-resilient private artifacts for one VoiceProbe call.

Raw development artifacts live under artifacts/, which is intentionally
gitignored. Selected fixtures can be reviewed and added deliberately.
"""

from __future__ import annotations

import json
import re
import struct
import sys
import wave
from array import array
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Self

import soundfile as sf  # type: ignore[import-untyped]

from voiceprobe.scenarios.models import PatientScenario

AUDIO_SAMPLE_RATE_HZ = 8_000
AUDIO_SAMPLE_WIDTH_BYTES = 2
AUDIO_CHANNELS = 1

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_SUMMARY_TIMING_KEYS = (
    "interpreter_seconds",
    "decision_seconds",
    "verbalizer_seconds",
    "tts_seconds",
    "endpoint_queue_seconds",
    "response_prep_seconds",
    "speech_seconds",
)


def _json_default(value: object) -> object:
    """Serialize supported project objects without silently stringifying all data."""
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)

    model_dump = getattr(value, "model_dump", None)

    if callable(model_dump):
        return model_dump(mode="json")

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp_text(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_json_atomic(
    path: Path,
    payload: object,
) -> None:
    """Write JSON through a temporary sibling so readers never see a partial file."""
    temporary = path.with_suffix(path.suffix + ".tmp")

    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=_json_default,
    )

    temporary.write_text(
        encoded + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def _open_pcm16_wave(path: Path) -> wave.Wave_write:
    writer = wave.Wave_write(str(path))

    writer.setnchannels(AUDIO_CHANNELS)
    writer.setsampwidth(AUDIO_SAMPLE_WIDTH_BYTES)
    writer.setframerate(AUDIO_SAMPLE_RATE_HZ)

    return writer


class RunArtifactRecorder:
    """Own private artifacts for exactly one call/run."""

    def __init__(
        self,
        *,
        root: Path | str,
        scenario: PatientScenario,
        run_id: str | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._lock = RLock()
        self._clock = clock

        self._scenario = scenario
        self._started_at = _utc_now()
        self._started_monotonic = self._clock()

        self.run_id = run_id or self._generate_run_id(scenario.scenario_id)

        if not _RUN_ID_RE.fullmatch(self.run_id):
            raise ValueError(
                "run_id must contain only letters, digits, '.', '_' or '-', "
                "start with an alphanumeric character, and be <= 128 characters."
            )

        self.run_dir = Path(root) / self.run_id
        self.run_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        self._events_path = self.run_dir / "events.jsonl"
        self._events_file = self._events_path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )

        self._inbound_path = self.run_dir / "inbound.wav"
        self._outbound_path = self.run_dir / "outbound.wav"
        self._call_wav_path = self.run_dir / "call.wav"
        self._call_ogg_path = self.run_dir / "call.ogg"

        self._inbound_wave = _open_pcm16_wave(self._inbound_path)
        self._outbound_wave = _open_pcm16_wave(self._outbound_path)

        self._transcript: list[dict[str, object]] = []
        self._turn_metrics: list[dict[str, object]] = []

        self._inbound_bytes = 0
        self._outbound_bytes = 0

        # Signed 32-bit-style Python integers accumulate both directions on
        # one absolute 8 kHz sample timeline. Clipping happens only when the
        # public listening artifacts are materialized at finalization.
        self._timeline_samples: list[int] = []

        # AudioSocket PCM is an ordered media stream. Multiple inbound
        # packets can already be buffered and therefore be read only
        # microseconds apart even though they represent consecutive 20 ms
        # sections of audio. Wall-clock arrival time anchors the first frame;
        # subsequent frames advance by their exact sample counts.
        self._inbound_timeline_next_sample: int | None = None

        self._finalized = False

        _write_json_atomic(
            self.run_dir / "scenario.json",
            scenario.model_dump(mode="json"),
        )

        self._write_manifest(
            status="running",
            call_id=None,
            error=None,
        )

        self.record_event(
            "run_started",
            scenario_id=scenario.scenario_id,
        )

    @staticmethod
    def _generate_run_id(
        scenario_id: str,
    ) -> str:
        timestamp = _utc_now().strftime("%Y%m%dT%H%M%S.%fZ")

        safe_scenario = re.sub(
            r"[^A-Za-z0-9._-]+",
            "-",
            scenario_id,
        ).strip("-")

        safe_scenario = safe_scenario[:48] or "scenario"

        return f"{timestamp}-{safe_scenario}"

    @property
    def elapsed_seconds(self) -> float:
        return max(
            0.0,
            self._clock() - self._started_monotonic,
        )

    def record_event(
        self,
        event: str,
        **details: object,
    ) -> None:
        """Append one immediately flushed structured event."""
        if not event.strip():
            raise ValueError("Event name cannot be blank.")

        with self._lock:
            self._ensure_open()

            payload = {
                "event": event,
                "timestamp_utc": _timestamp_text(_utc_now()),
                "elapsed_seconds": round(
                    self.elapsed_seconds,
                    6,
                ),
                "details": details,
            }

            self._events_file.write(
                json.dumps(
                    payload,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=_json_default,
                )
                + "\n"
            )

            self._events_file.flush()

    def record_transcript_turn(
        self,
        *,
        speaker: str,
        text: str,
        **metadata: object,
    ) -> None:
        """Store one finalized conversational utterance."""
        normalized_speaker = speaker.strip().casefold()
        normalized_text = " ".join(text.split())

        if normalized_speaker not in {
            "agent",
            "patient",
        }:
            raise ValueError("speaker must be 'agent' or 'patient'.")

        if not normalized_text:
            raise ValueError("Transcript text cannot be blank.")

        with self._lock:
            self._ensure_open()

            self._transcript.append(
                {
                    "index": len(self._transcript) + 1,
                    "speaker": normalized_speaker,
                    "text": normalized_text,
                    "timestamp_utc": _timestamp_text(_utc_now()),
                    "elapsed_seconds": round(
                        self.elapsed_seconds,
                        6,
                    ),
                    "metadata": metadata,
                }
            )

    def record_turn_metrics(
        self,
        metrics: Mapping[str, object],
    ) -> None:
        """Store timing/outcome measurements for one completed patient turn."""
        with self._lock:
            self._ensure_open()

            payload = dict(metrics)
            payload.setdefault(
                "turn_index",
                len(self._turn_metrics) + 1,
            )

            self._turn_metrics.append(payload)

    def record_inbound_pcm(
        self,
        payload: bytes,
    ) -> None:
        """Append one 8 kHz mono signed-16 PCM inbound frame."""
        self._record_pcm(
            payload=payload,
            writer=self._inbound_wave,
            direction="inbound",
        )

    def record_outbound_pcm(
        self,
        payload: bytes,
    ) -> None:
        """Append one 8 kHz mono signed-16 PCM outbound frame."""
        self._record_pcm(
            payload=payload,
            writer=self._outbound_wave,
            direction="outbound",
        )

    def _record_pcm(
        self,
        *,
        payload: bytes,
        writer: wave.Wave_write,
        direction: str,
    ) -> None:
        if len(payload) % AUDIO_SAMPLE_WIDTH_BYTES != 0:
            raise ValueError("PCM16 payload length must be divisible by two.")

        if not payload:
            return

        frame_sample_count = len(payload) // AUDIO_SAMPLE_WIDTH_BYTES

        with self._lock:
            self._ensure_open()

            elapsed_sample = max(
                0,
                round(self.elapsed_seconds * AUDIO_SAMPLE_RATE_HZ),
            )

            if direction == "inbound":
                if self._inbound_timeline_next_sample is None:
                    # The first frame establishes the absolute relationship
                    # between the ordered AudioSocket stream and this run's
                    # shared timeline.
                    timeline_start_sample = max(
                        0,
                        elapsed_sample - frame_sample_count,
                    )
                else:
                    # Never timestamp later inbound packets independently.
                    # Buffered socket reads may deliver consecutive media
                    # packets almost simultaneously in wall-clock time.
                    timeline_start_sample = self._inbound_timeline_next_sample

                self._inbound_timeline_next_sample = (
                    timeline_start_sample + frame_sample_count
                )

            elif direction == "outbound":
                # Outbound send_audio() is explicitly paced at telephony frame
                # cadence, so its successful send time remains an appropriate
                # shared-timeline position.
                timeline_start_sample = elapsed_sample

            else:
                raise ValueError(f"Unknown audio direction: {direction}")

            writer.writeframesraw(payload)

            if direction == "inbound":
                self._inbound_bytes += len(payload)
            else:
                self._outbound_bytes += len(payload)

            self._mix_pcm16_into_timeline(
                payload=payload,
                start_sample=timeline_start_sample,
            )

    def _mix_pcm16_into_timeline(
        self,
        *,
        payload: bytes,
        start_sample: int,
    ) -> None:
        """Mix little-endian PCM16 into the shared absolute call timeline."""
        frame_sample_count = len(payload) // AUDIO_SAMPLE_WIDTH_BYTES
        required_samples = start_sample + frame_sample_count

        if required_samples > len(self._timeline_samples):
            self._timeline_samples.extend(
                [0] * (required_samples - len(self._timeline_samples))
            )

        for offset, (sample,) in enumerate(struct.iter_unpack("<h", payload)):
            self._timeline_samples[start_sample + offset] += sample

    def _materialize_call_audio(self) -> None:
        """Write one aligned mixed WAV and one listenable OGG artifact."""
        mixed = array(
            "h",
            (
                max(
                    -32_768,
                    min(32_767, sample),
                )
                for sample in self._timeline_samples
            ),
        )

        wav_samples = array("h", mixed)

        if sys.byteorder != "little":
            wav_samples.byteswap()

        call_wave = _open_pcm16_wave(self._call_wav_path)

        try:
            call_wave.writeframes(wav_samples.tobytes())
        finally:
            call_wave.close()

        sf.write(
            self._call_ogg_path,
            mixed,
            AUDIO_SAMPLE_RATE_HZ,
            format="OGG",
            subtype="VORBIS",
        )

    def finalize(
        self,
        *,
        status: str = "completed",
        call_id: str | None = None,
        error: str | None = None,
    ) -> None:
        """Finalize WAV headers and materialize transcript/metrics/manifest."""
        with self._lock:
            if self._finalized:
                return

            self.record_event(
                "run_finalized",
                status=status,
                call_id=call_id,
                error=error,
            )

            transcript = tuple(self._transcript)
            metrics = tuple(self._turn_metrics)

            _write_json_atomic(
                self.run_dir / "transcript.json",
                {
                    "run_id": self.run_id,
                    "turns": transcript,
                },
            )

            self._write_transcript_text(transcript)

            _write_json_atomic(
                self.run_dir / "metrics.json",
                {
                    "run_id": self.run_id,
                    "turns": metrics,
                    "summary": self._build_metrics_summary(metrics),
                },
            )

            self._materialize_call_audio()

            self._inbound_wave.close()
            self._outbound_wave.close()
            self._events_file.close()

            self._write_manifest(
                status=status,
                call_id=call_id,
                error=error,
            )

            self._finalized = True

    def _build_metrics_summary(
        self,
        metrics: tuple[
            dict[str, object],
            ...,
        ],
    ) -> dict[str, object]:
        summary: dict[str, object] = {
            "turn_count": len(metrics),
            "duration_seconds": round(
                self.elapsed_seconds,
                6,
            ),
            "inbound_audio_seconds": round(
                self._audio_duration_seconds(self._inbound_bytes),
                6,
            ),
            "outbound_audio_seconds": round(
                self._audio_duration_seconds(self._outbound_bytes),
                6,
            ),
        }

        summary["call_audio_seconds"] = round(
            len(self._timeline_samples) / AUDIO_SAMPLE_RATE_HZ,
            6,
        )

        for key in _SUMMARY_TIMING_KEYS:
            values: list[float] = []

            for item in metrics:
                value = item.get(key)

                if isinstance(
                    value,
                    bool,
                ):
                    continue

                if isinstance(
                    value,
                    (int, float),
                ):
                    values.append(float(value))

            if values:
                summary[f"mean_{key}"] = round(
                    sum(values) / len(values),
                    6,
                )

        objective_values = [
            item.get("objective_complete")
            for item in metrics
            if isinstance(
                item.get("objective_complete"),
                bool,
            )
        ]

        if objective_values:
            summary["objective_complete"] = bool(objective_values[-1])

        return summary

    @staticmethod
    def _audio_duration_seconds(
        byte_count: int,
    ) -> float:
        bytes_per_second = (
            AUDIO_SAMPLE_RATE_HZ * AUDIO_SAMPLE_WIDTH_BYTES * AUDIO_CHANNELS
        )

        return byte_count / bytes_per_second

    def _write_transcript_text(
        self,
        transcript: tuple[
            dict[str, object],
            ...,
        ],
    ) -> None:
        lines: list[str] = []

        for entry in transcript:
            elapsed = entry["elapsed_seconds"]
            speaker = str(entry["speaker"]).upper()
            transcript_text = str(entry["text"])

            if isinstance(elapsed, bool) or not isinstance(
                elapsed,
                (int, float),
            ):
                raise TypeError("Transcript elapsed_seconds must be numeric.")

            elapsed_seconds = float(elapsed)

            lines.append(f"[{elapsed_seconds:08.3f}s] {speaker}: {transcript_text}")

        output = "\n".join(lines)

        if output:
            output += "\n"

        (self.run_dir / "transcript.txt").write_text(
            output,
            encoding="utf-8",
        )

    def _write_manifest(
        self,
        *,
        status: str,
        call_id: str | None,
        error: str | None,
    ) -> None:
        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "scenario_id": self._scenario.scenario_id,
            "status": status,
            "started_at_utc": _timestamp_text(self._started_at),
            "updated_at_utc": _timestamp_text(_utc_now()),
            "duration_seconds": round(
                self.elapsed_seconds,
                6,
            ),
            "call_id": call_id,
            "error": error,
            "audio": {
                "sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
                "sample_width_bytes": AUDIO_SAMPLE_WIDTH_BYTES,
                "channels": AUDIO_CHANNELS,
            },
            "artifacts": {
                "scenario": "scenario.json",
                "events": "events.jsonl",
                "transcript_json": "transcript.json",
                "transcript_text": "transcript.txt",
                "metrics": "metrics.json",
                "inbound_audio": "inbound.wav",
                "outbound_audio": "outbound.wav",
                "call_audio": "call.wav",
                "call_audio_ogg": "call.ogg",
            },
        }

        _write_json_atomic(
            self.run_dir / "manifest.json",
            payload,
        )

    def _ensure_open(self) -> None:
        if self._finalized:
            raise RuntimeError("RunArtifactRecorder is already finalized.")

    def __enter__(
        self,
    ) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, traceback

        self.finalize(
            status=("failed" if exc_value is not None else "completed"),
            error=(str(exc_value) if exc_value is not None else None),
        )
