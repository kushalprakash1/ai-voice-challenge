"""Conversation-turn assembly for streaming ASR output.

Speech recognizers may finalize several short phrases for what a human
would consider one conversational turn. This module groups those phrase
fragments without coupling turn logic to a specific ASR implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_DUPLICATE_WINDOW_SECONDS = 0.75


def _duplicate_key(text: str) -> str:
    """Normalize punctuation-only variation in rapid ASR duplicates."""
    return text.casefold().rstrip("?!.,:;…").strip()


@dataclass(frozen=True, slots=True)
class CompletedTurn:
    """One assembled conversational turn."""

    text: str
    lines: tuple[str, ...]
    started_at: float
    completed_at: float


class TurnAssembler:
    """Combine finalized ASR lines into human-level conversational turns."""

    def __init__(
        self,
        max_gap_seconds: float = 0.9,
        *,
        split_on_gap: bool = True,
        duplicate_window_seconds: float = DEFAULT_DUPLICATE_WINDOW_SECONDS,
    ) -> None:
        if max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be greater than zero.")

        if duplicate_window_seconds < 0:
            raise ValueError("duplicate_window_seconds cannot be negative.")

        self.max_gap_seconds = max_gap_seconds
        self.split_on_gap = split_on_gap
        self.duplicate_window_seconds = duplicate_window_seconds
        self._lines: list[str] = []
        self._started_at: float | None = None
        self._last_line_at: float | None = None

    @property
    def pending_text(self) -> str | None:
        """Return the currently assembled text without finalizing it."""
        if not self._lines:
            return None

        return " ".join(self._lines)

    @property
    def has_pending_turn(self) -> bool:
        """Return whether a turn currently contains finalized ASR text."""
        return bool(self._lines)

    def add_line(
        self,
        text: str,
        *,
        completed_at: float,
    ) -> CompletedTurn | None:
        """Add one finalized ASR line.

        If the gap from the previous line is too large, the previous
        pending turn is returned before starting a new one.
        """
        normalized = " ".join(text.split())

        if not normalized:
            return None

        previous_turn: CompletedTurn | None = None

        if (
            self.split_on_gap
            and self._last_line_at is not None
            and completed_at - self._last_line_at > self.max_gap_seconds
        ):
            previous_turn = self.flush(completed_at=self._last_line_at)

        if (
            self._lines
            and self._last_line_at is not None
            and completed_at - self._last_line_at <= self.duplicate_window_seconds
            and _duplicate_key(normalized) == _duplicate_key(self._lines[-1])
        ):
            # Moonshine can emit the same short finalized phrase twice at a
            # streaming boundary. Advance the endpoint clock without adding
            # duplicate conversational content.
            self._last_line_at = completed_at
            return previous_turn

        if self._started_at is None:
            self._started_at = completed_at

        self._lines.append(normalized)
        self._last_line_at = completed_at

        return previous_turn

    def flush(
        self,
        *,
        completed_at: float | None = None,
    ) -> CompletedTurn | None:
        """Finalize and clear the currently assembled turn."""
        if not self._lines:
            return None

        if self._started_at is None or self._last_line_at is None:
            raise RuntimeError("TurnAssembler entered an inconsistent state.")

        end_time = self._last_line_at if completed_at is None else completed_at

        turn = CompletedTurn(
            text=" ".join(self._lines),
            lines=tuple(self._lines),
            started_at=self._started_at,
            completed_at=end_time,
        )

        self._lines = []
        self._started_at = None
        self._last_line_at = None

        return turn
