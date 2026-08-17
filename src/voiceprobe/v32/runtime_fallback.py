"""Production-shaped but offline-gated v3.2 semantic fallback.

The deterministic v3 controller remains first authority.

This resolver:
- interprets only turns that v3 explicitly marked FALLBACK
- remembers prior completed exchanges
- never directly mutates flow state
- never invents authoritative patient facts
- never grants transaction authority
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from voiceprobe.v3.flow_state import FlowSnapshot
from voiceprobe.v3.models import (
    DecisionKind,
    PatientFacts,
    PolicyDecision,
)

from .ollama_backend import (
    OllamaBackend,
    OllamaConfig,
)
from .semantic_adapter import (
    SemanticResolution,
    resolve_semantic_turn,
)
from .semantic_parser import (
    SemanticParseTrace,
    SemanticParser,
)
from .semantic_policy import (
    RoutedSemanticTurn,
    SemanticRoute,
    route_semantic_frame,
)


class V32SemanticFallbackResolver:
    """Contextual semantic fallback behind deterministic v3."""

    def __init__(
        self,
        *,
        parser: SemanticParser,
        facts: PatientFacts | None = None,
        history_limit: int = 12,
    ) -> None:
        self.parser = parser
        self.facts = facts or PatientFacts()

        self._history: deque[str] = deque(
            maxlen=history_limit,
        )

        self.last_trace: SemanticParseTrace | None = None
        self.last_routed: RoutedSemanticTurn | None = None
        self.last_resolution: SemanticResolution | None = None
        self.last_backend_error: str | None = None

    @classmethod
    def from_ollama(
        cls,
        *,
        endpoint: str,
        model: str = "qwen3.5:4b",
        facts: PatientFacts | None = None,
    ) -> "V32SemanticFallbackResolver":
        backend = OllamaBackend(
            OllamaConfig(
                model=model,
                endpoint=endpoint,
                timeout_seconds=30.0,
                keep_alive="15m",
                num_ctx=1024,
                temperature=0.0,
            )
        )

        return cls(
            parser=SemanticParser(
                backend=backend,
            ),
            facts=facts,
        )

    @property
    def history(self) -> tuple[str, ...]:
        return tuple(self._history)

    def observe_exchange(
        self,
        source_turns: Iterable[str],
        decision: PolicyDecision,
    ) -> None:
        """Observe an exchange only AFTER its decision is finalized."""

        for turn in source_turns:
            normalized = " ".join(
                turn.split()
            )

            if normalized:
                self._history.append(
                    f"PGAI: {normalized}"
                )

        patient_text = " ".join(
            decision.text.split()
        )

        if patient_text:
            self._history.append(
                f"PATIENT: {patient_text}"
            )

    async def __call__(
        self,
        agent_turn: str,
        snapshot: FlowSnapshot,
    ) -> PolicyDecision:
        # Snapshot is intentionally supplied by the authoritative runtime.
        # The semantic model never receives permission to mutate it.
        del snapshot

        try:
            trace = await self.parser.parse(
                remote_turn=agent_turn,
                recent_dialogue=self.history,
            )
        except (
            ConnectionError,
            OSError,
            RuntimeError,
            TimeoutError,
            ValueError,
        ) as exc:
            # A remote reasoning node is an optional semantic aid, never
            # a reason for the telephony runtime itself to fail.
            self.last_backend_error = (
                f"{type(exc).__name__}: {exc}"
            )
            self.last_trace = None
            self.last_routed = None
            self.last_resolution = None

            return PolicyDecision(
                DecisionKind.CLARIFY,
                text=(
                    "I may have missed that. "
                    "Could you repeat the question?"
                ),
                reason="v32_semantic_backend_failure",
                confidence=0.0,
            )

        self.last_backend_error = None
        self.last_trace = trace

        if trace.validation_error is not None:
            self.last_routed = None
            self.last_resolution = None

            return PolicyDecision(
                DecisionKind.CLARIFY,
                text="Could you clarify that question?",
                reason="v32_invalid_semantic_frame",
                confidence=0.0,
            )

        routed = route_semantic_frame(
            trace.frame
        )

        self.last_routed = routed

        resolution = resolve_semantic_turn(
            routed,
            facts=self.facts,
        )

        self.last_resolution = resolution

        if resolution.decision is not None:
            return resolution.decision

        if (
            resolution.route
            is SemanticRoute.TRANSACTION_GATE
        ):
            # The semantic layer recognized transactional language,
            # but the deterministic policy did not establish a safe
            # transaction action. Fail closed instead of giving the
            # model booking/cancellation authority.
            return PolicyDecision(
                DecisionKind.CLARIFY,
                text=(
                    "Could you confirm the exact appointment action "
                    "you're asking me to authorize?"
                ),
                reason="v32_transaction_gate_fail_closed",
            )

        return PolicyDecision(
            DecisionKind.CLARIFY,
            text="Could you clarify that question?",
            reason="v32_semantic_unknown",
        )
