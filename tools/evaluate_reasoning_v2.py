"""Run deterministic semantic evaluation across historical VoiceProbe calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from replay_reasoning_v2 import (
    agent_turns_with_old_response,
    parse_transcript,
)

from voiceprobe.reasoning.action_plan import (
    ActionPlan,
)
from voiceprobe.reasoning.action_verbalizer import (
    GenericActionVerbalizer,
)
from voiceprobe.reasoning.evaluator import (
    ConversationEvaluationState,
    EvaluationSeverity,
    evaluate_turn,
)
from voiceprobe.reasoning.fact_grounding import (
    ground_fact_assertions,
)
from voiceprobe.reasoning.planner import (
    QwenPatientPlanner,
)
from voiceprobe.reasoning.semantic_reasoner import (
    StructuredTurnReasoner,
)
from voiceprobe.reasoning.world_model import (
    build_world_model,
)
from voiceprobe.scenarios.catalog import (
    get_scenario,
)


def infer_scenario(
    run_name: str,
) -> str | None:

    for scenario in (
        "autonomous-phone-diagnostic",
        "morning-slot-preference",
        "identity-insurance-check",
    ):
        if run_name.endswith(
            f"-{scenario}"
        ):
            return scenario

    return None


def evaluate_run(
    *,
    run_dir: Path,
    scenario_id: str,
    model: str,
    url: str,
) -> dict[str, object]:

    transcript = (
        run_dir
        / "transcript.txt"
    )

    scenario = get_scenario(
        scenario_id
    )

    world = build_world_model(
        scenario
    )

    semantic = StructuredTurnReasoner(
        model=model,
        url=url,
    )

    planner = QwenPatientPlanner(
        model=model,
        url=url,
    )

    verbalizer = (
        GenericActionVerbalizer()
    )

    recent_agent_history: list[
        str
    ] = []

    recent_actions: list[
        ActionPlan
    ] = []

    evaluation_state = (
        ConversationEvaluationState()
    )

    findings_json: list[
        dict[str, object]
    ] = []

    runtime_errors = 0
    turn_count = 0

    rows = agent_turns_with_old_response(
        parse_transcript(
            transcript
        )
    )

    try:
        for turn_number, (
            agent_turn,
            _old_patient,
        ) in enumerate(
            rows,
            start=1,
        ):
            turn_count += 1

            try:
                frame = semantic.interpret(
                    agent_turn=agent_turn,
                    recent_history=(
                        recent_agent_history
                    ),
                )

                grounding = (
                    ground_fact_assertions(
                        world=world,
                        turn=frame,
                    )
                )

                plan, _repaired_from = (
                    planner.plan(
                        world=world,
                        turn=frame,
                        recent_actions=(
                            recent_actions
                        ),
                    )
                )

                patient_text = (
                    verbalizer.verbalize(
                        world=world,
                        turn=frame,
                        plan=plan,
                        corrections=(
                            grounding.conflicts
                        ),
                    )
                )

                (
                    turn_findings,
                    evaluation_state,
                ) = evaluate_turn(
                    world=world,
                    turn=frame,
                    plan=plan,
                    grounding=grounding,
                    patient_text=patient_text,
                    state=evaluation_state,
                    validator=planner.validator,
                )

                for finding in turn_findings:
                    findings_json.append(
                        {
                            "run": run_dir.name,
                            "scenario": scenario_id,
                            "turn": turn_number,
                            "severity": (
                                finding.severity.value
                            ),
                            "code": finding.code,
                            "detail": finding.detail,
                            "agent_turn": agent_turn,
                            "patient_text": patient_text,
                            "requested_action": (
                                frame.requested_action.value
                            ),
                            "plan_action": (
                                plan.action.value
                            ),
                        }
                    )

                recent_actions.append(
                    plan
                )

            except Exception as error:
                runtime_errors += 1

                findings_json.append(
                    {
                        "run": run_dir.name,
                        "scenario": scenario_id,
                        "turn": turn_number,
                        "severity": "critical",
                        "code": "runtime_error",
                        "detail": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                        "agent_turn": agent_turn,
                        "patient_text": "",
                        "requested_action": None,
                        "plan_action": None,
                    }
                )

            finally:
                recent_agent_history.append(
                    agent_turn
                )

    finally:
        semantic.close()
        planner.close()

    critical = sum(
        item["severity"]
        == EvaluationSeverity.CRITICAL.value
        for item in findings_json
    )

    warnings = sum(
        item["severity"]
        == EvaluationSeverity.WARNING.value
        for item in findings_json
    )

    if critical:
        status = "CRITICAL"
    elif warnings:
        status = "WARN"
    else:
        status = "CLEAN"

    return {
        "run": run_dir.name,
        "scenario": scenario_id,
        "turns": turn_count,
        "status": status,
        "critical": critical,
        "warnings": warnings,
        "runtime_errors": runtime_errors,
        "findings": findings_json,
    }


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "artifacts/runs"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/reasoning_v2_eval"
        ),
    )

    parser.add_argument(
        "--model",
        default="qwen3:14b",
    )

    parser.add_argument(
        "--url",
        default=(
            "http://127.0.0.1:11434"
            "/api/chat"
        ),
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[
        dict[str, object]
    ] = []

    for run_dir in sorted(
        args.runs_root.iterdir()
    ):
        if not run_dir.is_dir():
            continue

        if not (
            run_dir
            / "transcript.txt"
        ).exists():
            continue

        scenario_id = infer_scenario(
            run_dir.name
        )

        if scenario_id is None:
            continue

        print(
            f"Evaluating {run_dir.name}..."
        )

        result = evaluate_run(
            run_dir=run_dir,
            scenario_id=scenario_id,
            model=args.model,
            url=args.url,
        )

        results.append(
            result
        )

        print(
            "  "
            f"{result['status']} | "
            f"turns={result['turns']} | "
            f"critical={result['critical']} | "
            f"warnings={result['warnings']}"
        )

    summary_path = (
        args.output
        / "summary.tsv"
    )

    findings_path = (
        args.output
        / "findings.jsonl"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "run\tscenario\tturns\tstatus\t"
            "critical\twarnings\truntime_errors\n"
        )

        for result in results:
            handle.write(
                f"{result['run']}\t"
                f"{result['scenario']}\t"
                f"{result['turns']}\t"
                f"{result['status']}\t"
                f"{result['critical']}\t"
                f"{result['warnings']}\t"
                f"{result['runtime_errors']}\n"
            )

    with findings_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for result in results:
            for finding in result[
                "findings"
            ]:
                handle.write(
                    json.dumps(
                        finding,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    total_turns = sum(
        int(result["turns"])
        for result in results
    )

    critical = sum(
        int(result["critical"])
        for result in results
    )

    warnings = sum(
        int(result["warnings"])
        for result in results
    )

    runtime_errors = sum(
        int(
            result[
                "runtime_errors"
            ]
        )
        for result in results
    )

    print()
    print("=" * 72)
    print("REASONING V2 EVALUATION")
    print("=" * 72)
    print(
        f"runs:           {len(results)}"
    )
    print(
        f"turns:          {total_turns}"
    )
    print(
        f"critical:       {critical}"
    )
    print(
        f"warnings:       {warnings}"
    )
    print(
        f"runtime errors: {runtime_errors}"
    )
    print(
        f"summary:        {summary_path}"
    )
    print(
        f"findings:       {findings_path}"
    )
    print("=" * 72)

    if critical:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
