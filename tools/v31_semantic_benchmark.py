#!/usr/bin/env python3
"""Offline/local semantic benchmark for VoiceProbe v3.1."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

from voiceprobe.v3.flow_state import FlowSnapshot, FlowStage
from voiceprobe.v3.semantic_router import SemanticIntent, V31SemanticRouter

DATA = Path("tests/data/v31_semantic_cases.jsonl")

STAGE_FOR_INTENT = {
    SemanticIntent.VISIT_REASON_REQUEST: FlowStage.VISIT_REASON,
    SemanticIntent.APPOINTMENT_TYPE_REQUEST: FlowStage.APPOINTMENT_TYPE,
    SemanticIntent.INSURANCE_REQUEST: FlowStage.INSURANCE,
    SemanticIntent.DOB_REQUEST: FlowStage.DOB,
    SemanticIntent.OPEN_ENDED_HELP: FlowStage.VISIT_REASON,
    SemanticIntent.FULL_NAME_REQUEST: FlowStage.IDENTITY,
    SemanticIntent.FIRST_NAME_REQUEST: FlowStage.IDENTITY,
    SemanticIntent.LAST_NAME_REQUEST: FlowStage.IDENTITY,
    SemanticIntent.PROVIDER_PREFERENCE_REQUEST: FlowStage.PROVIDER,
    SemanticIntent.DATE_TIME_PREFERENCE_REQUEST: FlowStage.DATE_TIME,
    SemanticIntent.PRESENCE_CHECK: FlowStage.VISIT_REASON,
}


def make_snapshot(stage: FlowStage) -> FlowSnapshot:
    return FlowSnapshot(
        communicated=frozenset(),
        confirmed=frozenset(),
        current_stage=stage,
        complete=False,
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="allow local Qwen for cases not confidently handled by prototypes",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in DATA.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    router = V31SemanticRouter()
    per_class = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
    correct = 0

    for row in rows:
        expected = SemanticIntent(row["intent"])
        state = make_snapshot(STAGE_FOR_INTENT[expected])

        if args.full:
            result = await router.classify(row["text"], state)
        else:
            result = router.scorer.classify(row["text"], state)

        predicted = result.intent

        if predicted == expected:
            correct += 1
            per_class[expected.value]["tp"] += 1
        else:
            per_class[expected.value]["fn"] += 1
            per_class[predicted.value]["fp"] += 1
            print(
                "MISS",
                f"expected={expected.value}",
                f"predicted={predicted.value}",
                f"source={result.source}",
                f"text={row['text']!r}",
            )

    f1_values = []
    for label in sorted(per_class):
        counts = per_class[label]
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        print(
            f"{label:34s} precision={precision:.3f} "
            f"recall={recall:.3f} f1={f1:.3f}"
        )

    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    accuracy = correct / len(rows)

    print()
    print(f"cases={len(rows)}")
    print(f"accuracy={accuracy:.3f}")
    print(f"macro_f1={macro_f1:.3f}")
    print(f"mode={'full-local-qwen' if args.full else 'prototype-only'}")

    if args.full and macro_f1 < 0.98:
        raise SystemExit("FULL SEMANTIC GATE FAILED: macro F1 below 0.98")

    if args.full:
        print("VOICEPROBE V3.1 FULL SEMANTIC GATE: PASS")


if __name__ == "__main__":
    asyncio.run(main())
