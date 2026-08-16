"""VoiceProbe v3.2 contextual reasoning layer.

This package is shadow-only initially.

It is not connected to live telephony.  It interprets novel clinic dialogue,
proposes bounded actions, and passes those proposals through deterministic
validation before producing an existing VoiceProbe PolicyDecision.
"""

from .fallback import ContextualFallbackResolver
from .reasoner import ContextualReasoner
from .validator import ActionValidator

__all__ = [
    "ActionValidator",
    "ContextualFallbackResolver",
    "ContextualReasoner",
]
