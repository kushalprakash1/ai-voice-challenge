"""Explicit one-call production entrypoint for PGAI assessment testing.

This module can authorize exactly one immutable patient scenario. It cannot
reuse or execute a multi-call suite manifest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
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
from voiceprobe.runner import run_persistent_authorized_suite
from voiceprobe.scenarios.catalog import get_scenario, list_scenarios
from voiceprobe.suite import build_suite_plan
from voiceprobe.telephony.ami import AsteriskAMIConfig
from voiceprobe.telephony.asterisk_adapter import AsteriskAssessmentCallAdapter

DEFAULT_AMI_ENV = Path.home() / ".config/voiceprobe/ami.env"


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
):
    """Create a fresh execution manifest containing exactly one scenario."""
    policy = settings.call_policy()
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute exactly one authorized PGAI assessment call.",
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
        "--confirm",
        default="",
        help="Exact live confirmation token.",
    )

    parser.add_argument(
        "--budget-usd",
        default="5.00",
        help="Execution budget ceiling; must remain below $20.",
    )

    parser.add_argument(
        "--max-rate-per-minute-usd",
        default="1.00",
        help="Conservative provider-rate assumption for reservation.",
    )

    args = parser.parse_args()

    settings = Settings()  # type: ignore[call-arg]

    manifest = prepare_one_call(
        settings=settings,
        scenario_id=args.scenario,
    )

    # Persist evidence of exactly what is about to cross the live boundary.
    manifest_path = write_execution_manifest(manifest)
    execution_dir = manifest_path.parent

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


if __name__ == "__main__":
    main()
