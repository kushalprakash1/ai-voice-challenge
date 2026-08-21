#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.runtime import (
    DecisionRoute,
    VoiceProbeV3Runtime,
)

from voiceprobe.v32.runtime_fallback import (
    V32SemanticFallbackResolver,
)


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    utterance: str
    expected_kind: DecisionKind
    require_fallback: bool | None = None


CASES = (
    Case(
        name="open_intent",
        utterance="How may I help you today?",
        expected_kind=DecisionKind.STATE_OBJECTIVE,
        require_fallback=False,
    ),

    # Historical provider wording. If recent deterministic hardening
    # handles it directly, that is desirable.
    Case(
        name="historical_provider_preference",
        utterance=(
            "Of course. Is there a specific provider you'd like to "
            "see, or would you like to see whoever is available "
            "Friday afternoon?"
        ),
        expected_kind=DecisionKind.ANSWER_PROVIDER_PREFERENCE,
    ),

    Case(
        name="historical_ack",
        utterance="No problem.",
        expected_kind=DecisionKind.WAIT,
    ),

    Case(
        name="historical_fragment",
        utterance="Here are some Friday afternoon",
        expected_kind=DecisionKind.HOLD,
    ),

    # This is the ordinary question that motivated v3.2.
    Case(
        name="reschedule_reason",
        utterance=(
            "What is the reason for changing your appointment?"
        ),
        expected_kind=DecisionKind.CONTEXTUAL_ANSWER,
        require_fallback=True,
    ),

    # Novel non-lexical authoritative-fact paraphrase.
    Case(
        name="novel_insurance",
        utterance=(
            "Which company underwrites your health coverage?"
        ),
        expected_kind=DecisionKind.ANSWER_FACT,
        require_fallback=True,
    ),

    # Novel visit-purpose paraphrase.
    Case(
        name="novel_visit_reason",
        utterance=(
            "What medical issue prompted this visit?"
        ),
        expected_kind=DecisionKind.ANSWER_COMPLAINT,
        require_fallback=True,
    ),

    # Deliberately irrelevant/unknown language should still fail
    # closed into speech instead of silence.
    Case(
        name="unknown_fail_closed",
        utterance=(
            "Could you unpack the metaphysics of this appointment?"
        ),
        expected_kind=DecisionKind.CLARIFY,
        require_fallback=True,
    ),
)


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="qwen3.5:4b",
    )

    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:11434/api/chat",
    )

    args = parser.parse_args()

    resolver = (
        V32SemanticFallbackResolver.from_ollama(
            endpoint=args.endpoint,
            model=args.model,
        )
    )

    runtime = VoiceProbeV3Runtime(
        fallback_resolver=resolver,
    )

    # Warm model outside score.
    await resolver.parser.parse(
        remote_turn="Okay, thank you.",
        recent_dialogue=(),
    )

    passed = 0

    print(
        "========== V3.2 ACTUAL-RUNTIME REPLAY =========="
    )

    for index, case in enumerate(CASES, 1):
        before = (
            runtime.flow_controller
            .tracker
            .snapshot()
        )

        result = await runtime.process_turns(
            [case.utterance],
            ingress_reason="v32_runtime_replay",
        )

        after = result.after

        transaction_safe = (
            after.accepted_slot_text
            == before.accepted_slot_text
            and after.booking_confirmation_text
            == before.booking_confirmation_text
            and after.complete
            == before.complete
        )

        route_ok = True

        if case.require_fallback is True:
            route_ok = (
                result.route
                is DecisionRoute.FALLBACK
            )
        elif case.require_fallback is False:
            route_ok = (
                result.route
                is DecisionRoute.DETERMINISTIC
            )

        kind_ok = (
            result.decision.kind
            is case.expected_kind
        )

        text_ok = True

        if case.name == "novel_insurance":
            text_ok = (
                result.decision.text
                == "Blue Cross."
            )

        if case.name == "novel_visit_reason":
            text_ok = (
                "right shoulder pain"
                in result.decision.text
            )

        if case.name == "reschedule_reason":
            text_ok = (
                "Friday afternoon"
                in result.decision.text
            )

        ok = (
            route_ok
            and kind_ok
            and text_ok
            and transaction_safe
        )

        passed += int(ok)

        print()
        print(
            f"[{index}] {case.name}"
        )
        print(
            " PGAI:",
            case.utterance,
        )
        print(
            " route=",
            result.route.value,
        )
        print(
            " decision_kind=",
            result.decision.kind.value,
        )
        print(
            " decision_reason=",
            result.decision.reason,
        )
        print(
            " decision_text=",
            result.decision.text,
        )
        print(
            " policy_latency_ms=",
            round(
                result.policy_latency_ms,
                1,
            ),
        )
        print(
            " transaction_state_unchanged=",
            transaction_safe,
        )
        print(
            " PASS=",
            ok,
        )

    snapshot = (
        runtime.flow_controller
        .tracker
        .snapshot()
    )

    print()
    print("========== SUMMARY ==========")
    print(
        f"correct={passed}/{len(CASES)}"
    )
    print(
        "fallback_decisions=",
        runtime.metrics.fallback_decisions,
    )
    print(
        "deterministic_decisions=",
        runtime.metrics.deterministic_decisions,
    )
    print(
        "max_policy_latency_ms=",
        round(
            runtime.metrics.max_policy_latency_ms,
            1,
        ),
    )
    print(
        "accepted_slot_text=",
        snapshot.accepted_slot_text,
    )
    print(
        "booking_confirmation_text=",
        snapshot.booking_confirmation_text,
    )
    print(
        "complete=",
        snapshot.complete,
    )
    print(
        f"failed={len(CASES) - passed}"
    )

    if passed != len(CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
