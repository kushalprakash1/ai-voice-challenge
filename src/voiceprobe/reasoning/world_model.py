"""Generic patient world model for autonomous caller reasoning.

The world model is intentionally separate from semantic perception.

Semantic perception answers:
    "What did the remote agent say?"

The world model answers:
    "Who am I, what do I know, and what am I trying to accomplish?"

The planner may inspect this model.

The semantic interpreter must not.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
)

from voiceprobe.scenarios.models import (
    PatientScenario,
)


class ConstraintStrength(StrEnum):
    """Whether a planner may violate a caller constraint."""

    HARD = "hard"
    SOFT = "soft"


class ConstraintSpec(BaseModel):
    """One generic constraint governing caller decisions."""

    model_config = ConfigDict(
        extra="forbid",
    )

    field: str
    value: str
    strength: ConstraintStrength = (
        ConstraintStrength.HARD
    )

    # Human/developer-readable explanation.
    source: str


class PatientWorldModel(BaseModel):
    """Caller truth supplied to the reasoning/planning layer."""

    model_config = ConfigDict(
        extra="forbid",
    )

    scenario_id: str
    objective: str

    # All immutable facts remain available to answer intake questions.
    facts: dict[str, Any]

    # Constraints are independent of any one patient name.
    constraints: list[ConstraintSpec]


_ANY_PROVIDER_VALUES = {
    "any",
    "any provider",
    "any available provider",
    "no preference",
    "none",
    "whoever is available",
}


def _clean(
    value: object,
) -> str | None:
    if value is None:
        return None

    normalized = " ".join(
        str(value).split()
    )

    return normalized or None


def build_world_model(
    scenario: PatientScenario,
) -> PatientWorldModel:
    """Adapt the existing VoiceProbe scenario into generic reasoning state.

    This is deliberately a migration adapter.

    Future scenarios should eventually declare constraints directly rather
    than requiring this legacy conversion step.
    """

    payload = scenario.model_dump(
        mode="json",
    )

    facts_raw = payload.get(
        "facts",
        {},
    )

    if not isinstance(
        facts_raw,
        dict,
    ):
        raise TypeError(
            "PatientScenario facts must serialize as an object."
        )

    facts = dict(
        facts_raw
    )

    constraints: list[ConstraintSpec] = []

    preferred_day = _clean(
        facts.get(
            "preferred_day"
        )
    )

    if preferred_day is not None:
        constraints.append(
            ConstraintSpec(
                field="day",
                value=preferred_day,
                strength=ConstraintStrength.HARD,
                source="preferred_day",
            )
        )

    preferred_time = _clean(
        facts.get(
            "preferred_time"
        )
    )

    if preferred_time is not None:
        constraints.append(
            ConstraintSpec(
                field="time",
                value=preferred_time,
                strength=ConstraintStrength.HARD,
                source="preferred_time",
            )
        )

    provider = _clean(
        facts.get(
            "provider_preference"
        )
    )

    # "Any provider" is freedom, not a restriction.
    if (
        provider is not None
        and provider.casefold()
        not in _ANY_PROVIDER_VALUES
    ):
        constraints.append(
            ConstraintSpec(
                field="provider",
                value=provider,
                strength=ConstraintStrength.HARD,
                source="provider_preference",
            )
        )

    scenario_id = _clean(
        payload.get(
            "scenario_id"
        )
    )

    if scenario_id is None:
        raise ValueError(
            "Scenario must contain scenario_id."
        )

    objective = _clean(
        payload.get(
            "objective"
        )
    )

    if objective is None:
        raise ValueError(
            "Scenario must contain objective."
        )

    return PatientWorldModel(
        scenario_id=scenario_id,
        objective=objective,
        facts=facts,
        constraints=constraints,
    )
