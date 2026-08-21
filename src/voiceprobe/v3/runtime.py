"""Offline-first VoiceProbe v3 conversational runtime.

The runtime joins Flux ingress with the structured scheduling flow controller.
It is intentionally transport-agnostic: the same object can be driven by real
Pipecat Flux events, deterministic transcript replays, or future recorded-audio
replays.

Routine turns stay entirely deterministic. Only decisions classified FALLBACK
are eligible for the optional fallback resolver.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Awaitable, Callable, Iterable

from .flow_controller import FlowDecision, SchedulingFlowController
from .flow_state import FlowSnapshot
from .ingress import FastStabilizationPredicate, FluxIngressController, FluxIngressResult
from .models import DecisionKind, PolicyDecision
from .turn_stabilizer import DEFAULT_CONTINUATION_GRACE_MS


class DecisionRoute(StrEnum):
    DETERMINISTIC = "deterministic"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class RuntimeDecision:
    source_turns: tuple[str, ...]
    actionable_turn: str | None
    decision: PolicyDecision
    before: FlowSnapshot
    after: FlowSnapshot
    route: DecisionRoute
    policy_latency_ms: float
    ingress_reason: str

    @property
    def requires_response(self) -> bool:
        return self.decision.requires_response

    @property
    def response_ready(self) -> bool:
        """Whether concrete response text is ready to be spoken."""
        return (
            self.decision.kind != DecisionKind.FALLBACK
            and self.decision.requires_response
            and bool(self.decision.text.strip())
        )


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    total_decisions: int
    deterministic_decisions: int
    fallback_decisions: int
    response_required: int
    waits: int
    holds: int
    total_policy_latency_ms: float
    max_policy_latency_ms: float

    @property
    def average_policy_latency_ms(self) -> float:
        if self.total_decisions == 0:
            return 0.0

        return self.total_policy_latency_ms / self.total_decisions


FallbackResolver = Callable[
    [str, FlowSnapshot],
    PolicyDecision | Awaitable[PolicyDecision],
]

RuntimeSink = Callable[
    [RuntimeDecision],
    None | Awaitable[None],
]


class VoiceProbeV3Runtime:
    """Single coordination point for live and replayed remote turns."""

    def __init__(
        self,
        *,
        flow_controller: SchedulingFlowController | None = None,
        fallback_resolver: FallbackResolver | None = None,
        on_decision: RuntimeSink | None = None,
        continuation_grace_ms: float = DEFAULT_CONTINUATION_GRACE_MS,
        fast_stabilization_predicate: FastStabilizationPredicate | None = None,
    ) -> None:
        self._flow = flow_controller or SchedulingFlowController()
        self._fallback_resolver = fallback_resolver
        self._on_decision = on_decision

        self._decisions: list[RuntimeDecision] = []
        self._total_policy_latency_ms = 0.0
        self._max_policy_latency_ms = 0.0
        self._fallback_count = 0
        self._deterministic_count = 0
        self._response_required = 0
        self._waits = 0
        self._holds = 0

        self._ingress = FluxIngressController(
            on_decision=self._handle_ingress_result,
            continuation_grace_ms=continuation_grace_ms,
            fast_stabilization_predicate=fast_stabilization_predicate,
        )

    @property
    def ingress(self) -> FluxIngressController:
        return self._ingress

    @property
    def flow_controller(self) -> SchedulingFlowController:
        return self._flow

    @property
    def decisions(self) -> tuple[RuntimeDecision, ...]:
        return tuple(self._decisions)

    @property
    def metrics(self) -> RuntimeMetrics:
        return RuntimeMetrics(
            total_decisions=len(self._decisions),
            deterministic_decisions=self._deterministic_count,
            fallback_decisions=self._fallback_count,
            response_required=self._response_required,
            waits=self._waits,
            holds=self._holds,
            total_policy_latency_ms=self._total_policy_latency_ms,
            max_policy_latency_ms=self._max_policy_latency_ms,
        )

    def attach_flux(self, stt_service: object) -> None:
        self._ingress.attach(stt_service)

    def mark_response_started(self) -> None:
        self._ingress.mark_response_started()

    async def mark_response_finished(self) -> FluxIngressResult | None:
        return await self._ingress.mark_response_finished()

    async def mark_response_suppressed(self) -> FluxIngressResult | None:
        return await self._ingress.mark_response_suppressed()

    async def process_turns(
        self,
        turns: Iterable[str],
        *,
        ingress_reason: str = "offline_replay",
    ) -> RuntimeDecision:
        """Process one remote conversational burst without Pipecat."""

        source = tuple(turn for turn in turns if turn.strip())

        if not source:
            raise ValueError("At least one non-empty remote turn is required.")

        started = time.perf_counter()
        flow_decision = self._flow.decide_burst(source)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return await self._finalize_flow_decision(
            flow_decision,
            policy_latency_ms=elapsed_ms,
            ingress_reason=ingress_reason,
        )

    async def _handle_ingress_result(
        self,
        ingress_result: FluxIngressResult,
    ) -> None:
        started = time.perf_counter()

        # The flow controller intentionally receives the entire preserved burst.
        # It updates remote-confirmation evidence before selecting the latest
        # actionable request.
        flow_decision = self._flow.decide_burst(
            ingress_result.source_turns
        )

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        await self._finalize_flow_decision(
            flow_decision,
            policy_latency_ms=elapsed_ms,
            ingress_reason=ingress_result.emission_reason,
        )

    async def _finalize_flow_decision(
        self,
        flow_decision: FlowDecision,
        *,
        policy_latency_ms: float,
        ingress_reason: str,
    ) -> RuntimeDecision:
        decision = flow_decision.decision
        after = flow_decision.after
        route = DecisionRoute.DETERMINISTIC

        if decision.kind == DecisionKind.FALLBACK:
            route = DecisionRoute.FALLBACK

            if (
                self._fallback_resolver is not None
                and flow_decision.actionable_turn is not None
            ):
                # Semantic fallback needs the complete stabilized Flux burst,
                # not only the final fragment chosen as actionable by the
                # deterministic coalescer. This preserves context such as:
                #
                #   "Would you like another afternoon day?"
                #   "or check the next available Friday."
                #
                # The deterministic route still keeps actionable_turn unchanged.
                semantic_turn = (
                    " ".join(flow_decision.source_turns)
                    if len(flow_decision.source_turns) > 1
                    else flow_decision.actionable_turn
                )

                # Measure the complete fallback path. Previously
                # policy_latency_ms only represented deterministic-policy
                # latency and therefore hid model inference time.
                fallback_started = time.perf_counter()

                maybe_decision = self._fallback_resolver(
                    semantic_turn,
                    flow_decision.before,
                )

                if inspect.isawaitable(maybe_decision):
                    decision = await maybe_decision
                else:
                    decision = maybe_decision

                policy_latency_ms += (
                    time.perf_counter() - fallback_started
                ) * 1000.0

                # Apply only the fallback's returned structured action. The
                # resolver itself never mutates flow state.
                after = self._flow.tracker.apply_decision(
                    decision
                )

        runtime_decision = RuntimeDecision(
            source_turns=flow_decision.source_turns,
            actionable_turn=flow_decision.actionable_turn,
            decision=decision,
            before=flow_decision.before,
            after=after,
            route=route,
            policy_latency_ms=policy_latency_ms,
            ingress_reason=ingress_reason,
        )

        # Stateful fallback implementations may observe the completed
        # conversational exchange. This runs for BOTH deterministic and
        # fallback decisions, allowing a contextual resolver to remember
        # ordinary preceding turns without participating in their policy.
        observer = getattr(
            self._fallback_resolver,
            "observe_exchange",
            None,
        )

        if observer is not None:
            maybe_observation = observer(
                runtime_decision.source_turns,
                runtime_decision.decision,
            )

            if inspect.isawaitable(maybe_observation):
                await maybe_observation

        self._record(runtime_decision)
        await self._emit(runtime_decision)
        return runtime_decision

    def _record(self, result: RuntimeDecision) -> None:
        self._decisions.append(result)
        self._total_policy_latency_ms += result.policy_latency_ms
        self._max_policy_latency_ms = max(
            self._max_policy_latency_ms,
            result.policy_latency_ms,
        )

        if result.route == DecisionRoute.FALLBACK:
            self._fallback_count += 1
        else:
            self._deterministic_count += 1

        if result.requires_response:
            self._response_required += 1

        if result.decision.kind == DecisionKind.WAIT:
            self._waits += 1

        if result.decision.kind == DecisionKind.HOLD:
            self._holds += 1

    async def _emit(
        self,
        result: RuntimeDecision,
    ) -> None:
        if self._on_decision is None:
            return

        maybe_awaitable = self._on_decision(result)

        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable
