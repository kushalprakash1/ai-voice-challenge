"""Environment-backed configuration for VoiceProbe."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from voiceprobe.policy import CallPolicy


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VOICEPROBE_",
        extra="ignore",
    )

    originating_number: str
    dry_run: bool = True

    telnyx_api_key: SecretStr | None = None
    telnyx_connection_id: str | None = None

    def call_policy(self) -> CallPolicy:
        """Build the validated outbound-call policy."""
        return CallPolicy(
            originating_number=self.originating_number,
            dry_run=self.dry_run,
        )
