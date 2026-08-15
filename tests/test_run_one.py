from pathlib import Path

from voiceprobe.config import Settings
from voiceprobe.run_one import load_ami_config, prepare_one_call
from voiceprobe.safety import ALLOWED_TEST_NUMBER


def test_prepare_one_call_contains_exactly_requested_scenario() -> None:
    settings = Settings(
        originating_number="+18402001303",
        dry_run=True,
    )

    manifest = prepare_one_call(
        settings=settings,
        scenario_id="autonomous-phone-diagnostic",
    )

    assert manifest.call_count == 1
    assert manifest.scenario_ids == ("autonomous-phone-diagnostic",)
    assert manifest.destination == ALLOWED_TEST_NUMBER
    assert manifest.dry_run


def test_prepare_one_call_never_expands_to_catalog() -> None:
    settings = Settings(
        originating_number="+18402001303",
        dry_run=True,
    )

    manifest = prepare_one_call(
        settings=settings,
        scenario_id="repetition-clarification",
    )

    assert len(manifest.scenario_ids) == 1
    assert manifest.scenario_ids[0] == "repetition-clarification"


def test_load_ami_config_reads_voiceprobe_prefixed_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ami.env"
    path.write_text(
        (
            "VOICEPROBE_AMI_HOST=127.0.0.1\n"
            "VOICEPROBE_AMI_PORT=5038\n"
            "VOICEPROBE_AMI_USERNAME=voiceprobe-test\n"
            "VOICEPROBE_AMI_SECRET=synthetic-test-secret\n"
        ),
        encoding="utf-8",
    )

    config = load_ami_config(path)

    assert config.host == "127.0.0.1"
    assert config.port == 5038
    assert config.username == "voiceprobe-test"
    assert config.secret == "synthetic-test-secret"
