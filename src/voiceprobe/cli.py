"""Command-line interface for VoiceProbe."""

import json
from dataclasses import asdict

import typer

from voiceprobe.config import Settings
from voiceprobe.dialer import build_call_plan

app = typer.Typer(
    no_args_is_help=True,
    help="VoiceProbe patient-agent testing toolkit.",
)


@app.callback()
def main() -> None:
    """Run VoiceProbe assessment and evaluation tools."""


@app.command()
def plan() -> None:
    """Display the validated outbound call plan without making a call."""
    settings = Settings()  # type: ignore[call-arg]
    call_plan = build_call_plan(settings.call_policy())

    typer.echo(json.dumps(asdict(call_plan), indent=2))


if __name__ == "__main__":
    app()
