"""Live AudioSocket-to-Moonshine transcription for VoiceProbe.

Asterisk supplies 8 kHz signed-linear PCM through AudioSocket.
Moonshine performs streaming ASR, while TurnAssembler combines
pause-delimited ASR lines into conversational turns.
"""

from __future__ import annotations

import socket
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np
from moonshine_voice import (
    Transcriber,
    TranscriptEventListener,
    get_model_for_language,
)

from voiceprobe.conversation.turns import CompletedTurn, TurnAssembler

HOST = "127.0.0.1"
PORT = 9019

AUDIO_SAMPLE_RATE_HZ = 8_000
TURN_GAP_SECONDS = 2.0
COMPLETE_TURN_FLUSH_SECONDS = 0.65
SHORT_TURN_FLUSH_SECONDS = 1.0
SHORT_TURN_PREFETCH_DELAY_SECONDS = 0.50
INCOMPLETE_TURN_FLUSH_SECONDS = 1.8

_SINGLE_WORD_FRAGMENT_STARTERS = frozenset(
    {
        "how",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
    }
)

_INCOMPLETE_TRAILING_WORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "at",
        "because",
        "but",
        "for",
        "if",
        "in",
        "my",
        "of",
        "on",
        "or",
        "our",
        "so",
        "the",
        "their",
        "to",
        "with",
        "your",
    }
)

_SHORT_AUXILIARY_STARTERS = frozenset(
    {
        "are",
        "can",
        "could",
        "did",
        "do",
        "does",
        "has",
        "have",
        "is",
        "should",
        "was",
        "were",
        "will",
        "would",
    }
)

_SHORT_PRONOUN_TAILS = frozenset(
    {
        "he",
        "i",
        "it",
        "she",
        "they",
        "we",
        "you",
    }
)

TYPE_HANGUP = 0x00
TYPE_UUID = 0x01
TYPE_DTMF = 0x03
TYPE_PCM_8KHZ = 0x10


class ConsoleTranscriptListener(TranscriptEventListener):
    """Print ASR events and emit assembled conversational turns."""

    def __init__(
        self,
        *,
        on_turn: Callable[[CompletedTurn], None] | None = None,
        on_candidate: Callable[[str], None] | None = None,
        on_speech_activity: Callable[[], None] | None = None,
    ) -> None:
        self._assembler = TurnAssembler(
            max_gap_seconds=TURN_GAP_SECONDS,
            # Moonshine exposes explicit start/partial activity. The
            # listener's endpoint timer therefore owns live turn
            # boundaries; finalized-line processing time must not split
            # one continuously spoken utterance.
            split_on_gap=False,
        )
        self._on_turn = on_turn
        self._on_candidate = on_candidate
        self._on_speech_activity = on_speech_activity
        self._lock = threading.Lock()
        self._flush_timer: threading.Timer | None = None
        self._generation = 0
        self._candidate_timer: threading.Timer | None = None
        self._candidate_generation = 0

    def on_line_started(self, event: Any) -> None:
        print(f"[started] {event.line.text}", flush=True)
        self._note_speech_activity()

    def on_line_text_changed(self, event: Any) -> None:
        print(f"[partial] {event.line.text}", flush=True)
        self._note_speech_activity()

    def on_line_completed(self, event: Any) -> None:
        text = event.line.text
        completed_at = time.monotonic()

        print(f"[final]   {text}", flush=True)

        candidate: str | None = None
        flush_delay = self._flush_delay_for_text(text)

        with self._lock:
            self._cancel_candidate_timer_locked()

            previous = self._assembler.add_line(
                text,
                completed_at=completed_at,
            )

            if previous is not None:
                self._emit_turn(previous)

            self._schedule_flush_locked(flush_delay)

            if flush_delay == COMPLETE_TURN_FLUSH_SECONDS:
                candidate = self._assembler.pending_text
            elif (
                flush_delay == SHORT_TURN_FLUSH_SECONDS
                and self._on_candidate is not None
            ):
                pending_text = self._assembler.pending_text
                if pending_text:
                    self._schedule_candidate_locked(
                        pending_text,
                        SHORT_TURN_PREFETCH_DELAY_SECONDS,
                    )

        if candidate is not None and self._on_candidate is not None:
            self._on_candidate(candidate)

    def flush(self) -> None:
        """Immediately emit any pending conversational turn."""
        with self._lock:
            self._cancel_timer_locked()
            self._cancel_candidate_timer_locked()

            turn = self._assembler.flush()

            if turn is not None:
                self._emit_turn(turn)

    def close(self) -> None:
        """Release timer resources."""
        with self._lock:
            self._cancel_timer_locked()
            self._cancel_candidate_timer_locked()

    def _note_speech_activity(self) -> None:
        """Postpone completion and invalidate stale provisional work."""
        had_pending_turn = False

        with self._lock:
            self._cancel_candidate_timer_locked()

            if self._assembler.has_pending_turn:
                had_pending_turn = True
                self._schedule_flush_locked(INCOMPLETE_TURN_FLUSH_SECONDS)

        if had_pending_turn and self._on_speech_activity is not None:
            self._on_speech_activity()

    @staticmethod
    def _flush_delay_for_text(text: str) -> float:
        normalized = " ".join(text.split())

        if not normalized:
            return INCOMPLETE_TURN_FLUSH_SECONDS

        # Explicit continuation punctuation is the strongest evidence that
        # Moonshine finalized a line before the caller finished the turn.
        if normalized.endswith(("...", "…", ",", "-", "—", ":")):
            return INCOMPLETE_TURN_FLUSH_SECONDS

        lexical = normalized.rstrip("?!.,:;…—-").split()

        if not lexical:
            return INCOMPLETE_TURN_FLUSH_SECONDS

        # Longer finalized lines keep the proven 0.65 s endpoint and retain
        # speculative semantic prefetch.
        if len(lexical) > 2:
            return COMPLETE_TURN_FLUSH_SECONDS

        first = lexical[0].casefold().strip("'\"()[]{}")
        last = lexical[-1].casefold().strip("'\"()[]{}")

        # Moonshine can temporarily emit a bare interrogative while the
        # speaker is still forming the actual question.
        if len(lexical) == 1 and first in _SINGLE_WORD_FRAGMENT_STARTERS:
            return INCOMPLETE_TURN_FLUSH_SECONDS

        # A connective/function word at the end strongly suggests that the
        # thought is unfinished ("They and", "What about", "Friday at").
        if last in _INCOMPLETE_TRAILING_WORDS:
            return INCOMPLETE_TURN_FLUSH_SECONDS

        # Protect very short auxiliary+pronoun fragments such as "Can you?"
        # even if ASR supplied terminal punctuation.
        if (
            len(lexical) == 2
            and first in _SHORT_AUXILIARY_STARTERS
            and last in _SHORT_PRONOUN_TAILS
        ):
            return INCOMPLETE_TURN_FLUSH_SECONDS

        # A two-word auxiliary-led fragment without terminal punctuation is
        # also likely to continue ("Would Friday" -> "...work for you?").
        if (
            len(lexical) == 2
            and first in _SHORT_AUXILIARY_STARTERS
            and not normalized.endswith(("?", ".", "!"))
        ):
            return INCOMPLETE_TURN_FLUSH_SECONDS

        # Short self-contained answers/confirmations such as "Friday.",
        # "Blue Cross?", "4:30 PM?", or "Okay, bye." no longer pay the full
        # 1.8 s fragment penalty. They intentionally do not start speculative
        # Ollama work, avoiding stale GPU requests if speech resumes.
        return SHORT_TURN_FLUSH_SECONDS

    def _schedule_flush_locked(
        self,
        delay_seconds: float,
    ) -> None:
        self._cancel_timer_locked()

        self._generation += 1
        generation = self._generation

        timer = threading.Timer(
            delay_seconds,
            self._flush_if_current,
            args=(generation,),
        )
        timer.daemon = True
        self._flush_timer = timer
        timer.start()

    def _flush_if_current(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return

            self._flush_timer = None
            self._cancel_candidate_timer_locked()

            turn = self._assembler.flush()

            if turn is not None:
                self._emit_turn(turn)

    def _schedule_candidate_locked(
        self,
        candidate_text: str,
        delay_seconds: float,
    ) -> None:
        self._cancel_candidate_timer_locked()
        self._candidate_generation += 1
        generation = self._candidate_generation

        timer = threading.Timer(
            delay_seconds,
            self._emit_candidate_if_current,
            args=(generation, candidate_text),
        )
        timer.daemon = True
        self._candidate_timer = timer
        timer.start()

    def _emit_candidate_if_current(
        self,
        generation: int,
        candidate_text: str,
    ) -> None:
        candidate: str | None = None

        with self._lock:
            if generation != self._candidate_generation:
                return

            self._candidate_timer = None
            if (
                self._assembler.has_pending_turn
                and self._assembler.pending_text == candidate_text
            ):
                candidate = candidate_text

        if candidate is not None and self._on_candidate is not None:
            self._on_candidate(candidate)

    def _cancel_candidate_timer_locked(self) -> None:
        self._candidate_generation += 1
        if self._candidate_timer is not None:
            self._candidate_timer.cancel()
            self._candidate_timer = None

    def _cancel_timer_locked(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None

    def _emit_turn(self, turn: CompletedTurn) -> None:
        print()
        print(f"[TURN] {turn.text}", flush=True)
        print()

        if self._on_turn is not None:
            self._on_turn(turn)


def recv_exact(
    connection: socket.socket,
    size: int,
) -> bytes | None:
    """Receive exactly size bytes or None if the peer disconnects."""
    data = bytearray()

    while len(data) < size:
        chunk = connection.recv(size - len(data))

        if not chunk:
            return None

        data.extend(chunk)

    return bytes(data)


def pcm16_to_float32(payload: bytes) -> np.ndarray:
    """Convert little-endian signed 16-bit PCM to normalized floats."""
    samples = np.frombuffer(payload, dtype="<i2")

    return samples.astype(np.float32) / 32768.0


def handle_connection(
    connection: socket.socket,
    transcriber: Transcriber,
    listener: ConsoleTranscriptListener,
) -> None:
    """Feed one live AudioSocket call into Moonshine."""
    call_id: uuid.UUID | None = None

    transcriber.start()

    try:
        while True:
            header = recv_exact(connection, 3)

            if header is None:
                print("AudioSocket disconnected", flush=True)
                return

            message_type = header[0]
            payload_length = int.from_bytes(header[1:3], "big")

            payload = recv_exact(connection, payload_length)

            if payload is None:
                print(
                    "AudioSocket disconnected during payload",
                    flush=True,
                )
                return

            if message_type == TYPE_HANGUP:
                print("Call ended", flush=True)
                return

            if message_type == TYPE_UUID:
                if len(payload) == 16:
                    call_id = uuid.UUID(bytes=payload)
                    print(f"Call UUID: {call_id}", flush=True)

                continue

            if message_type == TYPE_DTMF:
                digit = payload.decode(
                    "ascii",
                    errors="replace",
                )
                print(f"DTMF: {digit}", flush=True)
                continue

            if message_type != TYPE_PCM_8KHZ:
                continue

            audio = pcm16_to_float32(payload)

            transcriber.add_audio(
                audio,
                AUDIO_SAMPLE_RATE_HZ,
            )

    finally:
        transcriber.stop()
        listener.flush()

        print(
            f"Transcription session complete: call_id={call_id}",
            flush=True,
        )


def build_transcriber(
    *,
    on_turn: Callable[[CompletedTurn], None] | None = None,
    on_candidate: Callable[[str], None] | None = None,
    on_speech_activity: Callable[[], None] | None = None,
) -> tuple[Transcriber, ConsoleTranscriptListener]:
    """Load Moonshine once and attach the VoiceProbe turn listener."""
    model_path, model_arch = get_model_for_language("en")

    print(f"Loading Moonshine model from {model_path}...")

    transcriber = Transcriber(
        model_path=model_path,
        model_arch=model_arch,
    )

    listener = ConsoleTranscriptListener(
        on_turn=on_turn,
        on_candidate=on_candidate,
        on_speech_activity=on_speech_activity,
    )
    transcriber.add_listener(listener)

    return transcriber, listener


def main() -> None:
    """Run the local VoiceProbe streaming-ASR server."""
    transcriber, listener = build_transcriber()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_REUSEADDR,
                1,
            )
            server.bind((HOST, PORT))
            server.listen(1)

            print(f"VoiceProbe live ASR listening on {HOST}:{PORT}")
            print("Waiting for Asterisk...")

            while True:
                connection, address = server.accept()

                with connection:
                    print(f"Asterisk connected from {address}")

                    handle_connection(
                        connection,
                        transcriber,
                        listener,
                    )

    finally:
        listener.close()
        transcriber.close()


if __name__ == "__main__":
    main()
