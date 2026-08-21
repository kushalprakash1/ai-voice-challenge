"""Explicit one-call production entrypoint for VoiceProbe testing.

This module can authorize exactly one immutable patient scenario. It cannot
reuse or execute a multi-call suite manifest.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path

from dotenv import dotenv_values

from voiceprobe.config import Settings
from voiceprobe.execution import (
    authorize_live_execution,
    prepare_execution,
    write_execution_manifest,
)
from voiceprobe.execution_state import (
    BudgetPolicy,
    PersistentBudgetLedger,
    PersistentCallLedger,
)
from voiceprobe.policy import (
    DEFAULT_MAX_CALL_DURATION_SECONDS,
    MAX_CALL_DURATION_SECONDS,
)
from voiceprobe.runner import run_persistent_authorized_suite
from voiceprobe.safety import require_live_destination
from voiceprobe.scenarios.catalog import get_scenario, list_scenarios
from voiceprobe.suite import build_suite_plan
from voiceprobe.telephony.ami import AsteriskAMIConfig
from voiceprobe.telephony.asterisk_adapter import AsteriskAssessmentCallAdapter
from voiceprobe.v3.personas import (
    ENV_PERSONA,
    ENV_PERSONA_SEED,
    ENV_PERSONA_SEQUENCE,
    list_personas,
    sequence_ids_for,
)

DEFAULT_AMI_ENV = Path.home() / ".config/voiceprobe/ami.env"
DEFAULT_MAX_PROVIDER_RATE_PER_MINUTE_USD = Decimal("0.10")


def _required_env_value(
    values: dict[str, str | None],
    key: str,
) -> str:
    value = values.get(key)

    if value is None or not value.strip():
        raise ValueError(f"AMI configuration is missing {key!r}.")

    return value.strip()


def load_ami_config(
    path: Path = DEFAULT_AMI_ENV,
) -> AsteriskAMIConfig:
    """Load restricted localhost AMI credentials without printing secrets."""
    if not path.is_file():
        raise FileNotFoundError(f"AMI environment file does not exist: {path}")

    raw = dict(dotenv_values(path))

    username = _required_env_value(raw, "VOICEPROBE_AMI_USERNAME")
    secret = _required_env_value(raw, "VOICEPROBE_AMI_SECRET")
    host = _required_env_value(raw, "VOICEPROBE_AMI_HOST")
    port_text = _required_env_value(raw, "VOICEPROBE_AMI_PORT")

    return AsteriskAMIConfig(
        username=username,
        secret=secret,
        host=host,
        port=int(port_text),
    )


def prepare_one_call(
    *,
    settings: Settings,
    scenario_id: str,
    live_requested: bool = False,
    max_call_duration_seconds: int | None = None,
):
    """Create a fresh execution manifest containing exactly one scenario.

    The CLI's explicit --live request controls whether the prepared manifest
    is live-capable. Authorization still independently requires the live flag
    and exact confirmation token before any dialing side effect is allowed.
    """
    if type(live_requested) is not bool:
        raise TypeError("live_requested must be a boolean.")

    base_policy = settings.call_policy()

    policy = replace(
        base_policy,
        dry_run=not live_requested,
        max_call_duration_seconds=(
            base_policy.max_call_duration_seconds
            if max_call_duration_seconds is None
            else max_call_duration_seconds
        ),
    )

    scenario = get_scenario(scenario_id)

    suite = build_suite_plan(
        policy,
        scenarios=(scenario,),
    )

    if suite.call_count != 1:
        raise RuntimeError("One-call entrypoint produced a suite with call_count != 1.")

    manifest = prepare_execution(
        policy,
        suite,
    )

    if manifest.scenario_ids != (scenario_id,):
        raise RuntimeError("One-call execution manifest contains unexpected scenarios.")

    return manifest


def _execution_exit_code(failed_count: int) -> int:
    """Map suite outcome to a conventional CLI process status."""

    return 1 if failed_count > 0 else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute exactly one authorized VoiceProbe call.",
    )

    parser.add_argument(
        "--scenario",
        required=True,
        choices=tuple(scenario.scenario_id for scenario in list_scenarios()),
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Explicitly request live execution.",
    )

    parser.add_argument(
        "--live-monitor",
        action="store_true",
        help=(
            "Listen locally to both sides of the live call through ffplay. "
            "Monitoring uses one local mixed stream and is non-blocking; "
            "it never enters the call-critical media path."
        ),
    )


    parser.add_argument(
        "--persona",
        choices=tuple(
            persona.persona_id
            for persona in list_personas()
            if persona.persona_id != "control"
        ),
        default="",
        help="Optional adversarial patient persona.",
    )

    parser.add_argument(
        "--persona-sequence",
        default="",
        help="Explicit deterministic move sequence for --persona.",
    )

    parser.add_argument(
        "--persona-seed",
        type=int,
        default=6,
        help="Seed used when no explicit persona sequence is selected.",
    )

    parser.add_argument(
        "--confirm",
        default="",
        help="Exact live confirmation token.",
    )

    parser.add_argument(
        "--max-call-duration-seconds",
        type=int,
        default=DEFAULT_MAX_CALL_DURATION_SECONDS,
        help=(
            "Maximum duration for this one call in seconds. "
            f"Default is {DEFAULT_MAX_CALL_DURATION_SECONDS}; "
            f"hard maximum is {MAX_CALL_DURATION_SECONDS}."
        ),
    )

    parser.add_argument(
        "--budget-usd",
        default="5.00",
        help="Budget ceiling for this one-call execution.",
    )

    parser.add_argument(
        "--max-rate-per-minute-usd",
        default=format(
            DEFAULT_MAX_PROVIDER_RATE_PER_MINUTE_USD,
            "f",
        ),
        help=(
            "Conservative provider-rate ceiling used for reservation; "
            "default is $0.10/min."
        ),
    )

    args = parser.parse_args()

    if args.live_monitor and not args.live:
        parser.error("--live-monitor requires --live")

    if args.persona_sequence and not args.persona:
        parser.error("--persona-sequence requires --persona")

    if args.persona:
        available_sequences = sequence_ids_for(args.persona)

        if (
            args.persona_sequence
            and args.persona_sequence not in available_sequences
        ):
            parser.error(
                f"invalid --persona-sequence for {args.persona!r}; "
                f"choices are: {', '.join(available_sequences)}"
            )

    # Prevent stale shell variables from silently modifying a baseline call.
    os.environ[ENV_PERSONA] = args.persona
    os.environ[ENV_PERSONA_SEQUENCE] = args.persona_sequence
    os.environ[ENV_PERSONA_SEED] = str(args.persona_seed)

    # Keep monitoring explicitly opt-in for run_one. The actual media layer
    # reads this flag independently so the known-good adapter call signatures
    # and safety/budget plumbing remain unchanged.
    os.environ["VOICEPROBE_LIVE_MONITOR"] = (
        "1" if args.live_monitor else "0"
    )
    os.environ["VOICEPROBE_SCENARIO"] = args.scenario

    settings = Settings()  # type: ignore[call-arg]

    manifest = prepare_one_call(
        settings=settings,
        scenario_id=args.scenario,
        live_requested=args.live,
        max_call_duration_seconds=(
            args.max_call_duration_seconds
        ),
    )

    # Persist evidence of exactly what is about to cross the live boundary.
    manifest_path = write_execution_manifest(manifest)
    execution_dir = manifest_path.parent

    if not args.live:
        print(
            json.dumps(
                {
                    "execution_id": manifest.execution_id,
                    "manifest_path": str(manifest_path),
                    "call_count": manifest.call_count,
                    "scenario_id": args.scenario,
                    "dry_run": True,
                    "telephony_enabled": False,
                    "provider_call_invoked": False,
                },
                indent=2,
            )
        )
        return 0

    require_live_destination()

    authorization = authorize_live_execution(
        manifest,
        live_requested=args.live,
        confirmation_token=args.confirm,
    )

    budget_policy = BudgetPolicy(
        total_budget_usd=Decimal(args.budget_usd),
        max_provider_rate_per_minute_usd=Decimal(args.max_rate_per_minute_usd),
    )

    call_ledger = PersistentCallLedger.initialize(
        authorization,
        path=execution_dir / "calls.json",
    )

    budget_ledger = PersistentBudgetLedger.initialize(
        execution_id=manifest.execution_id,
        policy=budget_policy,
        path=execution_dir / "budget.json",
    )

    ami_config = load_ami_config()

    adapter = AsteriskAssessmentCallAdapter(
        ami_config=ami_config,
        expected_originating_number=manifest.originating_number,
    )

    try:
        result = run_persistent_authorized_suite(
            authorization,
            adapter,
            call_ledger=call_ledger,
            budget_ledger=budget_ledger,
        )
    finally:
        adapter.close()

    print(
        json.dumps(
            {
                "execution_id": result.execution_id,
                "manifest_path": str(manifest_path),
                "call_count": manifest.call_count,
                "scenario_id": args.scenario,
                "destination": manifest.destination,
                "entries": [asdict(entry) for entry in result.entries],
            },
            indent=2,
            default=str,
        )
    )

    return _execution_exit_code(result.failed_count)


if __name__ == "__main__":
    raise SystemExit(main())
