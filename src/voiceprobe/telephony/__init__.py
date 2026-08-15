"""Telephony control boundaries for VoiceProbe."""

from voiceprobe.telephony.ami import (
    AMIAuthenticationError,
    AMIClientError,
    AMIMessage,
    AMIOriginateError,
    AMIProtocolError,
    AsteriskAMIClient,
    AsteriskAMIConfig,
    OriginateResult,
)

__all__ = [
    "AMIAuthenticationError",
    "AMIClientError",
    "AMIMessage",
    "AMIOriginateError",
    "AMIProtocolError",
    "AsteriskAMIClient",
    "AsteriskAMIConfig",
    "OriginateResult",
]
