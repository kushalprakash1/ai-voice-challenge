"""AudioSocket capture and loopback diagnostic.

Receives 8 kHz signed-linear PCM from Asterisk, records the incoming
audio to a WAV artifact, and returns the same PCM to the caller.

This remains a diagnostic tool. The production media path will replace
the loopback with ASR, dialogue policy, and synthesized speech.
"""

from __future__ import annotations

import socket
import uuid
import wave
from datetime import UTC, datetime
from pathlib import Path

HOST = "127.0.0.1"
PORT = 9019

TYPE_HANGUP = 0x00
TYPE_UUID = 0x01
TYPE_DTMF = 0x03
TYPE_PCM_8KHZ = 0x10

SAMPLE_RATE_HZ = 8_000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1

ARTIFACT_DIR = Path("artifacts/audio")


def recv_exact(connection: socket.socket, size: int) -> bytes | None:
    """Receive exactly size bytes, or None when the socket closes."""
    data = bytearray()

    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)

    return bytes(data)


def send_message(
    connection: socket.socket,
    message_type: int,
    payload: bytes,
) -> None:
    """Send one AudioSocket protocol message."""
    header = bytes([message_type]) + len(payload).to_bytes(2, "big")
    connection.sendall(header + payload)


def create_capture_path() -> Path:
    """Create a unique path for one diagnostic call."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ARTIFACT_DIR / f"audiosocket-{timestamp}.wav"


def handle_connection(connection: socket.socket) -> None:
    """Capture incoming audio while echoing it to the caller."""
    capture_path = create_capture_path()
    call_id: uuid.UUID | None = None
    audio_bytes = 0

    print(f"Recording incoming audio to {capture_path}")

    with wave.open(str(capture_path), "wb") as recording:
        recording.setnchannels(CHANNELS)
        recording.setsampwidth(SAMPLE_WIDTH_BYTES)
        recording.setframerate(SAMPLE_RATE_HZ)

        while True:
            header = recv_exact(connection, 3)
            if header is None:
                print("AudioSocket disconnected")
                break

            message_type = header[0]
            payload_length = int.from_bytes(header[1:3], "big")

            payload = recv_exact(connection, payload_length)
            if payload is None:
                print("AudioSocket disconnected during payload")
                break

            if message_type == TYPE_HANGUP:
                print("Call ended")
                break

            if message_type == TYPE_UUID:
                if len(payload) == 16:
                    call_id = uuid.UUID(bytes=payload)
                    print(f"Call UUID: {call_id}")
                continue

            if message_type == TYPE_DTMF:
                digit = payload.decode("ascii", errors="replace")
                print(f"DTMF: {digit}")
                continue

            if message_type != TYPE_PCM_8KHZ:
                continue

            recording.writeframesraw(payload)
            audio_bytes += len(payload)

            # Diagnostic loopback only.
            send_message(connection, TYPE_PCM_8KHZ, payload)

    duration_seconds = audio_bytes / (SAMPLE_RATE_HZ * SAMPLE_WIDTH_BYTES * CHANNELS)

    print(
        f"Capture complete: {capture_path} ({duration_seconds:.2f}s, call_id={call_id})"
    )


def main() -> None:
    """Run the local AudioSocket diagnostic server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)

        print(f"VoiceProbe listening on {HOST}:{PORT}")

        while True:
            connection, address = server.accept()

            with connection:
                print(f"Asterisk connected from {address}")
                handle_connection(connection)


if __name__ == "__main__":
    main()
