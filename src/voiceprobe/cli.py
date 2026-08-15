"""Command-line interface for VoiceProbe."""

from __future__ import annotations

import json
from dataclasses import asdict

import typer

from voiceprobe.config import Settings
from voiceprobe.dialer import build_call_plan
from voiceprobe.execution import (
    prepare_execution,
    write_execution_manifest,
)
from voiceprobe.scenarios.catalog import list_scenarios
from voiceprobe.suite import build_suite_plan

app = typer.Typer(
    no_args_is_help=True,
    help="VoiceProbe patient-agent testing toolkit.",
)

suite_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect and validate deterministic assessment suites.",
)

app.add_typer(
    suite_app,
    name="suite",
)


@app.callback()
def main() -> None:
    """Run VoiceProbe assessment and evaluation tools."""


@app.command()
def plan() -> None:
    """Display one validated outbound call plan without making a call."""
    settings = Settings()  # type: ignore[call-arg]
    call_plan = build_call_plan(settings.call_policy())

    typer.echo(
        json.dumps(
            asdict(call_plan),
            indent=2,
        )
    )


@suite_app.command("list")
def suite_list() -> None:
    """List all deterministic patient scenarios without loading call config."""
    for index, scenario in enumerate(
        list_scenarios(),
        start=1,
    ):
        targets = ", ".join(scenario.test_targets)

        typer.echo(
            f"{index:02d}. "
            f"{scenario.scenario_id} | "
            f"{scenario.facts.name} | "
            f"{scenario.facts.preferred_day} "
            f"{scenario.facts.preferred_time} | "
            f"{targets}"
        )


@suite_app.command("validate")
def suite_validate() -> None:
    """Validate the full suite and outbound safety policy without dialing."""
    settings = Settings()  # type: ignore[call-arg]

    suite_plan = build_suite_plan(settings.call_policy())

    payload = {
        "valid": True,
        "suite_id": suite_plan.suite_id,
        "scenario_count": suite_plan.call_count,
        "destination": suite_plan.destination,
        "concurrency": suite_plan.concurrency,
        "max_call_duration_seconds": (suite_plan.max_call_duration_seconds),
        "max_suite_calls": suite_plan.max_suite_calls,
        "worst_case_duration_seconds": (suite_plan.worst_case_duration_seconds),
        "worst_case_duration_minutes": (suite_plan.worst_case_duration_minutes),
        "dry_run": suite_plan.dry_run,
        "live_execution_enabled": (suite_plan.live_execution_enabled),
    }

    typer.echo(
        json.dumps(
            payload,
            indent=2,
        )
    )


@suite_app.command("prepare")
def suite_prepare() -> None:
    """Write a local execution manifest without placing any calls."""
    settings = Settings()  # type: ignore[call-arg]
    policy = settings.call_policy()
    plan = build_suite_plan(policy)

    manifest = prepare_execution(
        policy,
        plan,
    )

    manifest_path = write_execution_manifest(manifest)

    payload = {
        "prepared": True,
        "execution_id": manifest.execution_id,
        "manifest_path": str(manifest_path),
        "call_count": manifest.call_count,
        "destination": manifest.destination,
        "dry_run": manifest.dry_run,
        "live_execution": False,
    }

    typer.echo(
        json.dumps(
            payload,
            indent=2,
        )
    )


@suite_app.command("plan")
def suite_plan() -> None:
    """Display every validated suite call without executing any of them."""
    settings = Settings()  # type: ignore[call-arg]

    plan = build_suite_plan(settings.call_policy())

    typer.echo(
        json.dumps(
            asdict(plan),
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
