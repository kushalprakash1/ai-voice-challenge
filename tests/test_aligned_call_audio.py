from __future__ import annotations

import json
import struct
from pathlib import Path

import soundfile as sf

from voiceprobe.artifacts.recorder import (
    AUDIO_SAMPLE_RATE_HZ,
    RunArtifactRecorder,
)
from voiceprobe.scenarios.catalog import get_scenario


class FakeClock:
    def __init__(
        self,
        value: float = 100.0,
    ) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def set(
        self,
        value: float,
    ) -> None:
        self.value = value


def pcm16(
    value: int,
    sample_count: int,
) -> bytes:
    return struct.pack(
        f"<{sample_count}h",
        *([value] * sample_count),
    )


def test_call_audio_preserves_real_timeline_silence(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    scenario = get_scenario("autonomous-phone-diagnostic")

    recorder = RunArtifactRecorder(
        root=tmp_path,
        scenario=scenario,
        run_id="aligned-timeline",
        clock=clock,
    )

    # 20 ms of patient audio beginning at run time 0.
    recorder.record_outbound_pcm(pcm16(1_000, 160))

    # Inbound frames are placed immediately before their receive timestamp.
    # Receiving this frame at 1.020 seconds therefore places it from
    # 1.000 through 1.020 seconds.
    clock.set(101.020)

    recorder.record_inbound_pcm(pcm16(2_000, 160))

    recorder.finalize()

    call, sample_rate = sf.read(
        tmp_path / "aligned-timeline" / "call.wav",
        dtype="int16",
    )

    assert sample_rate == AUDIO_SAMPLE_RATE_HZ
    assert len(call) == 8_160

    assert call[0] == 1_000
    assert call[159] == 1_000

    # The pause between speakers survives instead of being collapsed.
    assert call[160] == 0
    assert call[7_999] == 0

    assert call[8_000] == 2_000
    assert call[8_159] == 2_000

    # Raw directional streams remain available for diagnostics.
    outbound_info = sf.info(tmp_path / "aligned-timeline" / "outbound.wav")
    inbound_info = sf.info(tmp_path / "aligned-timeline" / "inbound.wav")

    assert outbound_info.frames == 160
    assert inbound_info.frames == 160

    # The compressed OGG export is independently decodable.
    ogg_info = sf.info(tmp_path / "aligned-timeline" / "call.ogg")

    assert ogg_info.samplerate == (AUDIO_SAMPLE_RATE_HZ)
    assert ogg_info.frames == 8_160

    manifest = json.loads((tmp_path / "aligned-timeline" / "manifest.json").read_text())

    assert manifest["artifacts"]["call_audio"] == ("call.wav")
    assert manifest["artifacts"]["call_audio_ogg"] == ("call.ogg")


def test_simultaneous_audio_is_mixed_with_pcm16_clipping(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    scenario = get_scenario("autonomous-phone-diagnostic")

    recorder = RunArtifactRecorder(
        root=tmp_path,
        scenario=scenario,
        run_id="aligned-overlap",
        clock=clock,
    )

    # Outbound occupies samples 0..159.
    recorder.record_outbound_pcm(pcm16(30_000, 160))

    # At receive time 20 ms, this inbound frame is placed over exactly
    # the same sample range.
    clock.set(100.020)

    recorder.record_inbound_pcm(pcm16(30_000, 160))

    recorder.finalize()

    call, sample_rate = sf.read(
        tmp_path / "aligned-overlap" / "call.wav",
        dtype="int16",
    )

    assert sample_rate == AUDIO_SAMPLE_RATE_HZ
    assert len(call) == 160

    # 30,000 + 30,000 must saturate rather than wrap around.
    assert all(sample == 32_767 for sample in call)


def test_buffered_inbound_packets_remain_sequential(
    tmp_path: Path,
) -> None:
    """Rapid socket reads must not collapse consecutive media frames."""
    clock = FakeClock()
    scenario = get_scenario("autonomous-phone-diagnostic")

    recorder = RunArtifactRecorder(
        root=tmp_path,
        scenario=scenario,
        run_id="buffered-inbound",
        clock=clock,
    )

    # Pretend three consecutive 20 ms AudioSocket packets are already
    # buffered. Python receives all three at effectively the same wall-clock
    # instant, but they still represent 60 ms of consecutive source audio.
    clock.set(100.020)

    recorder.record_inbound_pcm(pcm16(1_000, 160))
    recorder.record_inbound_pcm(pcm16(2_000, 160))
    recorder.record_inbound_pcm(pcm16(3_000, 160))

    recorder.finalize()

    call, sample_rate = sf.read(
        tmp_path / "buffered-inbound" / "call.wav",
        dtype="int16",
    )

    assert sample_rate == AUDIO_SAMPLE_RATE_HZ

    # All three packets survive as 60 ms of sequential audio instead of
    # being mixed into the same 20 ms region.
    assert len(call) == 480

    assert all(sample == 1_000 for sample in call[0:160])

    assert all(sample == 2_000 for sample in call[160:320])

    assert all(sample == 3_000 for sample in call[320:480])
