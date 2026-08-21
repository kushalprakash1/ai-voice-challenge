"""Real-time remote-turn arbitration for v3.3 staging and future live wiring.

This layer solves transport timing, not conversational strategy. It merges
short continuation fragments and suppresses near-duplicate repeats while a
patient response is already pending. The default continuation grace remains
900 ms; v3.3 does not lengthen it.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum


DEFAULT_CONTINUATION_GRACE_MS = 900.0


class ArbitrationKind(StrEnum):
    EMIT = "emit"
    HOLD = "hold"
    SUPPRESS_DUPLICATE = "suppress_duplicate"


@dataclass(frozen=True, slots=True)
class ArbitrationResult:
    kind: ArbitrationKind
    text: str = ""
    reason: str = ""


class V33TurnArbiter:
    def __init__(
        self,
        *,
        continuation_grace_ms: float = DEFAULT_CONTINUATION_GRACE_MS,
        duplicate_similarity: float = 0.78,
    ) -> None:
        if continuation_grace_ms < 0:
            raise ValueError("continuation_grace_ms must be non-negative")
        if not 0.0 <= duplicate_similarity <= 1.0:
            raise ValueError("duplicate_similarity must be in [0, 1]")
        self.continuation_grace_ms = continuation_grace_ms
        self.duplicate_similarity = duplicate_similarity
        self._pending = ""
        self._pending_at_ms = 0.0
        self._last_emitted = ""

    def ingest(
        self,
        text: str,
        *,
        at_ms: float,
        response_pending: bool = False,
    ) -> ArbitrationResult:
        normalized = " ".join(text.split())
        if not normalized:
            return ArbitrationResult(ArbitrationKind.HOLD, reason="empty_segment")

        if response_pending and self._last_emitted:
            similarity = self._similarity(normalized, self._last_emitted)
            if similarity >= self.duplicate_similarity:
                return ArbitrationResult(
                    ArbitrationKind.SUPPRESS_DUPLICATE,
                    reason=f"repeat_while_response_pending:{similarity:.3f}",
                )

        if self._pending:
            gap = at_ms - self._pending_at_ms
            if gap <= self.continuation_grace_ms:
                combined = f"{self._pending} {normalized}".strip()
                self._pending = ""
                if self._looks_incomplete(combined):
                    self._pending = combined
                    self._pending_at_ms = at_ms
                    return ArbitrationResult(ArbitrationKind.HOLD, reason="continued_fragment")
                self._last_emitted = combined
                return ArbitrationResult(ArbitrationKind.EMIT, combined, "merged_continuation")

            # Stale pending material should not block a later complete turn.
            self._pending = ""

        if self._looks_incomplete(normalized):
            self._pending = normalized
            self._pending_at_ms = at_ms
            return ArbitrationResult(ArbitrationKind.HOLD, reason="incomplete_fragment")

        self._last_emitted = normalized
        return ArbitrationResult(ArbitrationKind.EMIT, normalized, "complete_turn")

    def clear(self) -> None:
        self._pending = ""
        self._last_emitted = ""
        self._pending_at_ms = 0.0

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.casefold(), b.casefold()).ratio()

    @staticmethod
    def _looks_incomplete(text: str) -> bool:
        stripped = text.rstrip()
        if stripped.endswith((",", ":", "-", "—", "...", "…")):
            return True
        final = stripped.casefold().rstrip(".?!").split()
        return bool(final and final[-1] in {"and", "or", "to", "with", "for"})
