"""Send Kokoro speech through an Asterisk AudioSocket connection."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from time import perf_counter

import numpy as np
from kokoro import KPipeline

from voiceprobe.tts.telephony import (
    FRAME_DURATION_SECONDS,
    build_audiosocket_packet,
    float_audio_to_pcm16,
    iter_pcm_frames,
    resample_to_telephony,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9019
DEFAULT_VOICE = "af_heart"

DEFAULT_TEXT = (
    "Hello. This is the VoiceProbe patient audio test. "
    "I've had right shoulder pain for about five days."
)


def synthesize(
    *,
    pipeline: KPipeline,
    text: str,
    voice: str,
) -> np.ndarray:
    pieces: list[np.ndarray] = []

    for _, _, audio in pipeline(
        text,
        voice=voice,
        speed=1.0,
    ):
        pieces.append(np.asarray(audio, dtype=np.float32))

    if not pieces:
        raise RuntimeError("Kokoro generated no audio.")

    return np.concatenate(pieces)


async def read_packet(
    reader: asyncio.StreamReader,
) -> tuple[int, bytes]:
    header = await reader.readexactly(3)

    message_type = header[0]

    payload_length = int.from_bytes(
        header[1:3],
        byteorder="big",
        signed=False,
    )

    payload = await reader.readexactly(payload_length)

    return message_type, payload


async def drain_inbound(
    reader: asyncio.StreamReader,
) -> None:
    try:
        while True:
            await read_packet(reader)
    except (
        asyncio.IncompleteReadError,
        ConnectionResetError,
    ):
        return


async def send_pcm_audio(
    writer: asyncio.StreamWriter,
    pcm16: bytes,
) -> None:
    loop = asyncio.get_running_loop()
    next_deadline = loop.time()

    for frame in iter_pcm_frames(pcm16):
        writer.write(build_audiosocket_packet(frame))
        await writer.drain()

        next_deadline += FRAME_DURATION_SECONDS
        delay = next_deadline - loop.time()

        if delay > 0:
            await asyncio.sleep(delay)


async def handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    pipeline: KPipeline,
    voice: str,
    text: str,
) -> None:
    print()
    print(f"AudioSocket connected from {writer.get_extra_info('peername')}")

    try:
        message_type, payload = await read_packet(reader)

        print(
            f"First AudioSocket packet: type=0x{message_type:02x}, bytes={len(payload)}"
        )

        inbound_task = asyncio.create_task(drain_inbound(reader))

        print()
        print("Answer your phone, then press ENTER here to send the Kokoro speech.")

        await asyncio.to_thread(input)

        print("Synthesizing...")

        started = perf_counter()

        audio_24k = synthesize(
            pipeline=pipeline,
            text=text,
            voice=voice,
        )

        synth_seconds = perf_counter() - started

        audio_8k = resample_to_telephony(audio_24k)

        pcm16 = float_audio_to_pcm16(audio_8k)

        duration = len(audio_8k) / 8000

        print(f"Synthesis: {synth_seconds:.3f}s")
        print(f"Audio:     {duration:.3f}s")
        print("Sending to phone...")

        await send_pcm_audio(
            writer,
            pcm16,
        )

        print("Playback complete.")

        await asyncio.sleep(0.5)

        inbound_task.cancel()

        with suppress(asyncio.CancelledError):
            await inbound_task

    except (
        asyncio.IncompleteReadError,
        ConnectionResetError,
        BrokenPipeError,
    ):
        print("AudioSocket connection closed.")

    finally:
        writer.close()

        with suppress(ConnectionError):
            await writer.wait_closed()


async def run_server(
    *,
    host: str,
    port: int,
    voice: str,
    text: str,
) -> None:
    print("Loading Kokoro...")

    started = perf_counter()

    pipeline = KPipeline(
        lang_code="a",
        repo_id="hexgrad/Kokoro-82M",
    )

    print(f"Kokoro ready in {perf_counter() - started:.3f}s")

    finished = asyncio.Event()

    async def callback(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            await handle_connection(
                reader,
                writer,
                pipeline=pipeline,
                voice=voice,
                text=text,
            )
        finally:
            finished.set()

    server = await asyncio.start_server(
        callback,
        host,
        port,
    )

    print(f"Listening on {host}:{port}")
    print("Waiting for one Asterisk AudioSocket connection...")

    async with server:
        await finished.wait()


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
    )

    args = parser.parse_args()

    asyncio.run(
        run_server(
            host=args.host,
            port=args.port,
            voice=args.voice,
            text=args.text,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
