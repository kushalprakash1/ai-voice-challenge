"""Deepgram Flux ingress coordination for VoiceProbe v3.

This module intentionally has no hard dependency on Pipecat at import time.
The production adapter attaches to a DeepgramFluxSTTService via its documented
event_handler API, while unit tests can use a small fake event source.

The key behavior is burst coalescing: while VoiceProbe is preparing or playing
a response, remote end-of-turn transcripts accumulate as one conversational
burst. When VoiceProbe is ready again, the burst is collapsed to the latest
actionable request instead of replaying stale acknowledgements/status updates
through a FIFO queue.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .coalescer import ConversationBurstCoalescer
from .models import DecisionKind, PolicyDecision
from .turn_stabilizer import DEFAULT_CONTINUATION_GRACE_MS


@dataclass(frozen=True, slots=True)
class FluxIngressResult:
    """One decision emitted by the v3 ingress layer."""

    source_turns: tuple[str, ...]
    actionable_turn: str | None
    decision: PolicyDecision
    buffered_turn_count: int
    emission_reason: str

    @property
    def requires_response(self) -> bool:
        return self.decision.requires_response


class RemoteSpeechBurstBuffer:
    """Accumulate remote turns only while VoiceProbe is busy responding."""

    def __init__(
        self,
        *,
        coalescer: ConversationBurstCoalescer | None = None,
    ) -> None:
        self._coalescer = coalescer or ConversationBurstCoalescer()
        self._response_busy = False
        self._pending: list[str] = []

    @property
    def response_busy(self) -> bool:
        return self._response_busy

    @property
    def pending_turns(self) -> tuple[str, ...]:
        return tuple(self._pending)

    def mark_response_started(self) -> None:
        self._response_busy = True

    def ingest_end_of_turn(
        self,
        transcript: str,
    ) -> FluxIngressResult | None:
        """Ingest one Flux-confirmed remote end-of-turn transcript."""

        return self.ingest_turns(
            (transcript,),
            emission_reason="immediate_end_of_turn",
        )

    def ingest_turns(
        self,
        turns: tuple[str, ...],
        *,
        emission_reason: str,
    ) -> FluxIngressResult | None:
        """Ingest one stabilized conversational burst."""

        normalized = tuple(
            " ".join(turn.split())
            for turn in turns
            if turn.strip()
        )

        if not normalized:
            return None

        if self._response_busy:
            self._pending.extend(normalized)
            return None

        return self._build_result(
            normalized,
            emission_reason=emission_reason,
        )

    def mark_response_finished(
        self,
    ) -> FluxIngressResult | None:
        """Release the busy state and coalesce anything said meanwhile."""

        self._response_busy = False

        if not self._pending:
            return None

        turns = tuple(self._pending)
        self._pending.clear()

        return self._build_result(
            turns,
            emission_reason="buffered_burst_drained",
        )

    def clear_pending(self) -> tuple[str, ...]:
        """Discard pending remote turns during explicit call teardown."""

        turns = tuple(self._pending)
        self._pending.clear()
        return turns

    def _build_result(
        self,
        turns: tuple[str, ...],
        *,
        emission_reason: str,
    ) -> FluxIngressResult:
        coalesced = self._coalescer.coalesce(turns)

        return FluxIngressResult(
            source_turns=coalesced.source_turns,
            actionable_turn=coalesced.actionable_turn,
            decision=coalesced.decision,
            buffered_turn_count=max(0, len(turns) - 1),
            emission_reason=emission_reason,
        )


DecisionSink = Callable[
    [FluxIngressResult],
    Awaitable[None] | None,
]


async def _call_sink(
    sink: DecisionSink | None,
    result: FluxIngressResult | None,
) -> None:
    if sink is None or result is None:
        return

    maybe_awaitable = sink(result)

    if inspect.isawaitable(maybe_awaitable):
        await maybe_awaitable


class FluxIngressController:
    """Attach VoiceProbe v3 policy to Pipecat Deepgram Flux turn events."""

    def __init__(
        self,
        *,
        burst_buffer: RemoteSpeechBurstBuffer | None = None,
        on_decision: DecisionSink | None = None,
        continuation_grace_ms: float = DEFAULT_CONTINUATION_GRACE_MS,
    ) -> None:
        if continuation_grace_ms < 0:
            raise ValueError("continuation_grace_ms must be non-negative")

        self._burst_buffer = burst_buffer or RemoteSpeechBurstBuffer()
        self._on_decision = on_decision
        self._attached_service: Any | None = None
        self._last_start_transcript = ""
        self._turn_resumed_count = 0

        self._continuation_grace_ms = float(continuation_grace_ms)
        self._stabilized_pending: list[str] = []
        self._stabilization_task: asyncio.Task[None] | None = None

    @property
    def burst_buffer(self) -> RemoteSpeechBurstBuffer:
        return self._burst_buffer

    @property
    def attached(self) -> bool:
        return self._attached_service is not None

    @property
    def turn_resumed_count(self) -> int:
        return self._turn_resumed_count

    @property
    def continuation_grace_ms(self) -> float:
        return self._continuation_grace_ms

    @property
    def pending_stabilized_turns(self) -> tuple[str, ...]:
        return tuple(self._stabilized_pending)

    def attach(self, stt_service: Any) -> None:
        """Register against DeepgramFluxSTTService.event_handler()."""

        if self._attached_service is not None:
            raise RuntimeError(
                "FluxIngressController is already attached to an STT service."
            )

        event_handler = getattr(
            stt_service,
            "event_handler",
            None,
        )

        if not callable(event_handler):
            raise TypeError(
                "STT service does not expose Pipecat's event_handler API."
            )

        @stt_service.event_handler("on_start_of_turn")
        async def _on_start_of_turn(
            service: Any,
            transcript: str,
        ) -> None:
            del service
            self._last_start_transcript = transcript
            self._cancel_stabilization_timer()

        @stt_service.event_handler("on_turn_resumed")
        async def _on_turn_resumed(
            service: Any,
        ) -> None:
            del service
            self._turn_resumed_count += 1
            self._cancel_stabilization_timer()

        @stt_service.event_handler("on_end_of_turn")
        async def _on_end_of_turn(
            service: Any,
            transcript: str,
        ) -> None:
            del service
            await self._ingest_stabilized_end_of_turn(transcript)

        self._attached_service = stt_service

    async def _ingest_stabilized_end_of_turn(
        self,
        transcript: str,
    ) -> None:
        normalized = " ".join(transcript.split())

        if not normalized:
            return

        self._stabilized_pending.append(normalized)
        self._cancel_stabilization_timer()

        if self._continuation_grace_ms == 0:
            await self.flush_stabilized_pending()
            return

        self._stabilization_task = asyncio.create_task(
            self._delayed_stabilization_flush()
        )

    async def _delayed_stabilization_flush(self) -> None:
        try:
            await asyncio.sleep(
                self._continuation_grace_ms / 1000.0
            )
        except asyncio.CancelledError:
            return

        self._stabilization_task = None
        await self.flush_stabilized_pending()

    def _cancel_stabilization_timer(self) -> None:
        task = self._stabilization_task
        self._stabilization_task = None

        if task is not None and not task.done():
            task.cancel()

    async def flush_stabilized_pending(
        self,
    ) -> FluxIngressResult | None:
        """Release the currently stabilized remote conversational burst."""

        current_task = asyncio.current_task()
        task = self._stabilization_task
        self._stabilization_task = None

        if (
            task is not None
            and task is not current_task
            and not task.done()
        ):
            task.cancel()

        if not self._stabilized_pending:
            return None

        turns = tuple(self._stabilized_pending)
        self._stabilized_pending.clear()

        result = self._burst_buffer.ingest_turns(
            turns,
            emission_reason="stabilized_end_of_turn",
        )

        await _call_sink(
            self._on_decision,
            result,
        )
        return result

    def mark_response_started(self) -> None:
        """Call immediately before response preparation/playback begins."""

        self._burst_buffer.mark_response_started()

    async def mark_response_finished(self) -> FluxIngressResult | None:
        """Call after playback/echo guard ends and emit one coalesced decision."""

        result = self._burst_buffer.mark_response_finished()

        await _call_sink(
            self._on_decision,
            result,
        )

        return result

    def clear_pending(self) -> tuple[str, ...]:
        self._cancel_stabilization_timer()

        stabilized = tuple(self._stabilized_pending)
        self._stabilized_pending.clear()

        busy = self._burst_buffer.clear_pending()
        return stabilized + busy
