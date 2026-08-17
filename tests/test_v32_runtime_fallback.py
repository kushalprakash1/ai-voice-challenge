import asyncio

from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.runtime import (
    DecisionRoute,
    VoiceProbeV3Runtime,
)

from voiceprobe.v32.runtime_fallback import (
    V32SemanticFallbackResolver,
)
from voiceprobe.v32.semantic_parser import (
    SemanticParser,
)


class FakeBackend:
    def __init__(
        self,
        responses,
        *,
        delay_seconds=0.0,
    ):
        self.responses = list(responses)
        self.delay_seconds = delay_seconds
        self.calls = 0
        self.prompts = []

    async def generate_json(
        self,
        *,
        system,
        prompt,
        schema,
    ):
        del system, schema

        self.calls += 1
        self.prompts.append(prompt)

        if self.delay_seconds:
            await asyncio.sleep(
                self.delay_seconds
            )

        return self.responses.pop(0)


def semantic_resolver(
    *responses,
    delay_seconds=0.0,
):
    backend = FakeBackend(
        responses,
        delay_seconds=delay_seconds,
    )

    resolver = V32SemanticFallbackResolver(
        parser=SemanticParser(
            backend=backend,
        )
    )

    return resolver, backend


def test_runtime_fallback_uses_semantic_resolver_without_state_mutation():
    async def scenario():
        resolver, backend = semantic_resolver({
            "speech_act": "ask",
            "operation": "reschedule",
            "focus": "reschedule_reason",
            "commitment": "informational",
            "certainty": "high",
        })

        runtime = VoiceProbeV3Runtime(
            fallback_resolver=resolver,
        )

        # Deterministic exchange first. The semantic resolver should
        # OBSERVE it without being invoked.
        first = await runtime.process_turns(
            ["How may I help you today?"]
        )

        assert (
            first.route
            is DecisionRoute.DETERMINISTIC
        )

        assert backend.calls == 0

        # Novel wording: intentionally outside the lexical fast path.
        second = await runtime.process_turns([
            (
                "Could you briefly explain the scheduling conflict "
                "behind moving the existing visit?"
            )
        ])

        assert (
            second.route
            is DecisionRoute.FALLBACK
        )

        assert (
            second.decision.kind
            is DecisionKind.CONTEXTUAL_ANSWER
        )

        assert (
            second.decision.reason
            == "v32_reschedule_reason"
        )

        # Contextual explanation must not create scheduling progress.
        assert second.after == second.before

        assert (
            second.after.accepted_slot_text
            is None
        )

        assert (
            second.after.booking_confirmation_text
            is None
        )

        assert not second.after.complete
        assert backend.calls == 1

        # Prior deterministic patient speech reached the parser history.
        assert (
            "PATIENT: I need to schedule an appointment "
            "for Friday afternoon."
            in backend.prompts[0]
        )

    asyncio.run(scenario())


def test_runtime_latency_includes_semantic_fallback_time():
    async def scenario():
        resolver, _ = semantic_resolver(
            {
                "speech_act": "ask",
                "operation": "reschedule",
                "focus": "reschedule_reason",
                "commitment": "informational",
                "certainty": "high",
            },
            delay_seconds=0.03,
        )

        runtime = VoiceProbeV3Runtime(
            fallback_resolver=resolver,
        )

        result = await runtime.process_turns([
            (
                "Could you explain the scheduling conflict "
                "behind moving that existing visit?"
            )
        ])

        assert (
            result.route
            is DecisionRoute.FALLBACK
        )

        # Proves fallback inference is now visible in runtime telemetry.
        assert result.policy_latency_ms >= 20.0

        assert (
            runtime.metrics.total_policy_latency_ms
            >= 20.0
        )

    asyncio.run(scenario())


def test_transaction_semantics_fail_closed():
    async def scenario():
        resolver, _ = semantic_resolver({
            "speech_act": "ask",
            "operation": "book",
            "focus": "appointment_status",
            "commitment": "permission_request",
            "certainty": "high",
        })

        before_runtime = VoiceProbeV3Runtime()

        before = (
            before_runtime.flow_controller
            .tracker
            .snapshot()
        )

        decision = await resolver(
            (
                "Would you like me to execute the final "
                "booking operation now?"
            ),
            before,
        )

        assert (
            decision.kind
            is DecisionKind.CLARIFY
        )

        assert (
            decision.reason
            == "v32_transaction_gate_fail_closed"
        )

        after = (
            before_runtime.flow_controller
            .tracker
            .snapshot()
        )

        assert after == before

    asyncio.run(scenario())


def test_unknown_semantics_fail_closed_without_silence():
    async def scenario():
        resolver, _ = semantic_resolver({
            "speech_act": "other",
            "operation": "none",
            "focus": "other",
            "commitment": "none",
            "certainty": "low",
        })

        runtime = VoiceProbeV3Runtime(
            fallback_resolver=resolver,
        )

        result = await runtime.process_turns([
            "Could you unpack the metaphysics of this appointment?"
        ])

        assert (
            result.route
            is DecisionRoute.FALLBACK
        )

        assert (
            result.decision.kind
            is DecisionKind.CLARIFY
        )

        assert result.response_ready

        assert result.after == result.before

    asyncio.run(scenario())



def test_backend_failure_fails_closed_without_state_mutation():
    class FailingBackend:
        async def generate_json(self, **kwargs):
            del kwargs
            raise TimeoutError("reasoning node unavailable")

    async def scenario():
        resolver = V32SemanticFallbackResolver(
            parser=SemanticParser(
                backend=FailingBackend(),
            )
        )

        runtime = VoiceProbeV3Runtime(
            fallback_resolver=resolver,
        )

        result = await runtime.process_turns([
            (
                "Could you explain the scheduling conflict "
                "behind moving the existing visit?"
            )
        ])

        assert result.route is DecisionRoute.FALLBACK
        assert result.decision.kind is DecisionKind.CLARIFY
        assert (
            result.decision.reason
            == "v32_semantic_backend_failure"
        )
        assert result.response_ready
        assert result.after == result.before

        assert resolver.last_backend_error is not None
        assert "TimeoutError" in resolver.last_backend_error

    asyncio.run(scenario())
