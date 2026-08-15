"""Explicit one-call production entrypoint for PGAI assessment testing.

This module can authorize exactly one immutable patient scenario. It cannot
reuse or execute a multi-call suite manifest.
"""

from __future__ import annotations

import argparse
import json
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
    HARD_BUDGET_CEILING_USD,
    BudgetLedger,
    BudgetPolicy,
    BudgetStateError,
    PersistentBudgetLedger,
    PersistentCallLedger,
)
from voiceprobe.runner import run_persistent_authorized_suite
from voiceprobe.scenarios.catalog import get_scenario, list_scenarios
from voiceprobe.suite import build_suite_plan
from voiceprobe.telephony.ami import AsteriskAMIConfig
from voiceprobe.telephony.asterisk_adapter import AsteriskAssessmentCallAdapter

DEFAULT_AMI_ENV = Path.home() / ".config/voiceprobe/ami.env"
DEFAULT_MAX_PROVIDER_RATE_PER_MINUTE_USD = Decimal("0.10")


def _decimal_from_budget_value(
    value: object,
    *,
    name: str,
    path: Path,
) -> Decimal:
    """Parse one persisted monetary value without silently ignoring damage."""
    if not isinstance(value, str):
        raise BudgetStateError(f"{path}: {name} must be stored as a decimal string.")

    try:
        amount = Decimal(value)
    except Exception as error:
        raise BudgetStateError(f"{path}: {name} is not a valid decimal.") from error

    if not amount.is_finite() or amount < 0:
        raise BudgetStateError(f"{path}: {name} must be a finite non-negative amount.")

    return amount


def cumulative_assessment_commitment(
    executions_root: Path,
) -> Decimal:
    """Sum every persisted assessment reservation/actual cost across runs.

    Actual provider cost replaces the conservative reservation when known.
    A malformed historical ledger is treated as a safety failure rather than
    being ignored.
    """
    total = Decimal(0)

    if not executions_root.exists():
        return total

    for budget_path in sorted(executions_root.glob("*/budget.json")):
        try:
            payload = json.loads(budget_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise BudgetStateError(
                f"Unable to read historical budget ledger: {budget_path}"
            ) from error

        if not isinstance(payload, dict):
            raise BudgetStateError(
                f"{budget_path}: budget ledger must contain an object."
            )

        entries = payload.get("entries")

        if not isinstance(entries, list):
            raise BudgetStateError(
                f"{budget_path}: budget entries must contain a list."
            )

        for entry in entries:
            if not isinstance(entry, dict):
                raise BudgetStateError(
                    f"{budget_path}: budget entry must contain an object."
                )

            actual = entry.get("actual_usd")

            if actual is not None:
                total += _decimal_from_budget_value(
                    actual,
                    name="actual_usd",
                    path=budget_path,
                )
                continue

            total += _decimal_from_budget_value(
                entry.get("reserved_usd"),
                name="reserved_usd",
                path=budget_path,
            )

    return total


def enforce_cumulative_budget(
    *,
    prior_commitment_usd: Decimal,
    next_call_reservation_usd: Decimal,
) -> None:
    """Block a call whose worst-case reservation could cross the hard cap."""
    projected = prior_commitment_usd + next_call_reservation_usd

    if projected >= HARD_BUDGET_CEILING_USD:
        raise BudgetStateError(
            "Starting this assessment call could reach or exceed the "
            "$20 cumulative challenge budget ceiling."
        )


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

    settings = Settings()  # type: ignore[call-arg]

    manifest = prepare_one_call(
        settings=settings,
        scenario_id=args.scenario,
        live_requested=args.live,
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

    # Each run_one execution owns its own atomic ledger, but the challenge
    # budget applies across every real assessment call. Scan historical
    # execution ledgers before allowing the next reservation.
    prior_commitment_usd = cumulative_assessment_commitment(execution_dir.parent)

    next_reservation = BudgetLedger(budget_policy).worst_case_call_cost(
        manifest.max_call_duration_seconds
    )

    enforce_cumulative_budget(
        prior_commitment_usd=prior_commitment_usd,
        next_call_reservation_usd=next_reservation,
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
                "prior_cumulative_commitment_usd": (prior_commitment_usd),
                "next_call_reserved_usd": (next_reservation),
                "entries": [asdict(entry) for entry in result.entries],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
