"""Typed action plans for VoiceProbe v3.3.

v3.3 reasons over *actions*, not canned sentences. A single spoken turn may
contain multiple moves (for example, answer a fact and then ask a steering
question), so the core unit is ActionPlan rather than one response string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


class ActionKind(StrEnum):
    WAIT = "wait"
    PROVIDE_FACT = "provide_fact"
    CONFIRM_REMOTE_FACT = "confirm_remote_fact"
    DENY_REMOTE_FACT = "deny_remote_fact"
    CREATE_PROFILE = "create_profile"
    CLAIM_EXISTING_PROFILE = "claim_existing_profile"
    REQUEST_PROFILE_LOOKUP = "request_profile_lookup"
    STATE_GOAL = "state_goal"
    ASK_QUESTION = "ask_question"
    ASK_ALTERNATIVES = "ask_alternatives"
    SELECT_OPTION = "select_option"
    REJECT_OPTION = "reject_option"
    SET_PREFERENCE = "set_preference"
    RELAX_PREFERENCE = "relax_preference"
    CHANGE_PREFERENCE = "change_preference"
    ASK_CONFIRMATION = "ask_confirmation"
    WITHHOLD_AUTHORIZATION = "withhold_authorization"
    AUTHORIZE_TRANSACTION = "authorize_transaction"
    REVOKE_AUTHORIZATION = "revoke_authorization"
    CORRECT_REMOTE_STATE = "correct_remote_state"
    CHALLENGE_REMOTE_STATE = "challenge_remote_state"
    PROMPT_INJECTION_PROBE = "prompt_injection_probe"
    CAPABILITY_PROBE = "capability_probe"
    RESUME_WORKFLOW = "resume_workflow"
    END_CALL = "end_call"


@dataclass(frozen=True, slots=True)
class ActionMove:
    kind: ActionKind
    arguments: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = {
            str(k).strip(): str(v).strip()
            for k, v in dict(self.arguments).items()
            if str(k).strip()
        }
        object.__setattr__(self, "arguments", MappingProxyType(normalized))

    def arg(self, key: str, default: str = "") -> str:
        return self.arguments.get(key, default)


@dataclass(frozen=True, slots=True)
class ActionPlan:
    moves: tuple[ActionMove, ...]
    rationale: str
    utterance: str = ""

    def __post_init__(self) -> None:
        if not self.moves:
            raise ValueError("ActionPlan requires at least one move")
        if len(self.moves) > 3:
            raise ValueError("ActionPlan is limited to three moves per turn")

    @property
    def kinds(self) -> tuple[ActionKind, ...]:
        return tuple(move.kind for move in self.moves)

    def has(self, kind: ActionKind) -> bool:
        return any(move.kind is kind for move in self.moves)

    def first(self, kind: ActionKind) -> ActionMove | None:
        for move in self.moves:
            if move.kind is kind:
                return move
        return None
