import numpy as np

from voiceprobe.tts.telephony import (
    AUDIOSOCKET_PCM8K_TYPE,
    FRAME_BYTES,
    build_audiosocket_packet,
    float_audio_to_pcm16,
    iter_pcm_frames,
    resample_to_telephony,
)


def test_resamples_24k_to_8k() -> None:
    source = np.zeros(
        24_000,
        dtype=np.float32,
    )

    result = resample_to_telephony(
        source,
        source_rate=24_000,
    )

    assert len(result) == 8_000


def test_pcm16_is_little_endian_two_bytes_per_sample() -> None:
    source = np.array(
        [0.0, 1.0, -1.0],
        dtype=np.float32,
    )

    payload = float_audio_to_pcm16(source)

    assert len(payload) == 6


def test_audio_is_clipped_before_pcm_encoding() -> None:
    source = np.array(
        [2.0, -2.0],
        dtype=np.float32,
    )

    payload = float_audio_to_pcm16(source)

    decoded = np.frombuffer(
        payload,
        dtype="<i2",
    )

    assert decoded[0] == 32767
    assert decoded[1] == -32767


def test_frames_are_twenty_milliseconds() -> None:
    payload = bytes(FRAME_BYTES * 2)

    frames = list(iter_pcm_frames(payload))

    assert len(frames) == 2
    assert all(len(frame) == FRAME_BYTES for frame in frames)


def test_final_short_frame_is_padded() -> None:
    payload = bytes(FRAME_BYTES + 10)

    frames = list(iter_pcm_frames(payload))

    assert len(frames) == 2
    assert len(frames[-1]) == FRAME_BYTES


def test_audiosocket_header_encodes_type_and_length() -> None:
    payload = bytes(FRAME_BYTES)

    packet = build_audiosocket_packet(payload)

    assert packet[0] == AUDIOSOCKET_PCM8K_TYPE
    assert (
        int.from_bytes(
            packet[1:3],
            byteorder="big",
        )
        == FRAME_BYTES
    )
    assert packet[3:] == payload


def test_audiosocket_terminate_packet_is_zero_length_hangup() -> None:
    from voiceprobe.tts.telephony import (
        build_audiosocket_terminate_packet,
    )

    assert build_audiosocket_terminate_packet() == b"\x00\x00\x00"
