"""Ground remote-agent fact assertions against authoritative caller truth.

Semantic perception records what the remote agent CLAIMED.

This module determines whether that claim agrees with the immutable
PatientWorldModel.

No language model is involved in this comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from voiceprobe.reasoning.turn_frame import (
    AgentFactAssertion,
    RequestedFact,
    TurnFrame,
)
from voiceprobe.reasoning.world_model import (
    PatientWorldModel,
)


@dataclass(
    frozen=True,
    slots=True,
)
class FactConflict:
    """A remote assertion that conflicts with authoritative caller truth."""

    fact: RequestedFact
    asserted_value: str
    authoritative_value: str


@dataclass(
    frozen=True,
    slots=True,
)
class FactGrounding:
    """Result of grounding every remote caller-related assertion."""

    conflicts: tuple[FactConflict, ...] = ()
    matched_facts: tuple[RequestedFact, ...] = ()
    ungrounded_facts: tuple[RequestedFact, ...] = ()


_FACT_KEYS: dict[
    RequestedFact,
    str,
] = {
    RequestedFact.FIRST_NAME: "first_name",
    RequestedFact.LAST_NAME: "last_name",
    RequestedFact.FULL_NAME: "name",
    RequestedFact.DATE_OF_BIRTH: "date_of_birth",
    RequestedFact.INSURANCE: "insurance",
    RequestedFact.COMPLAINT: "complaint",
    RequestedFact.SYMPTOM_DURATION: "duration",
    RequestedFact.PREFERRED_DAY: "preferred_day",
    RequestedFact.PREFERRED_TIME: "preferred_time",
    RequestedFact.PROVIDER_PREFERENCE: "provider_preference",
    RequestedFact.APPOINTMENT_TYPE: "appointment_type",
    RequestedFact.PATIENT_STATUS: "patient_status",
    RequestedFact.VISITED_BEFORE: "visited_before",
    RequestedFact.PHONE_NUMBER: "phone_number",
    RequestedFact.EMAIL: "email",
    RequestedFact.ADDRESS: "address",
}


_ORDINAL_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)\b",
    flags=re.IGNORECASE,
)


def _normalize_text(
    value: object,
) -> str:
    """Normalize harmless surface differences without changing meaning."""

    text = " ".join(
        str(value).casefold().split()
    )

    text = text.strip(
        " .,!?:;"
    )

    return text


def _parse_date(
    value: object,
) -> str | None:
    """Canonicalize common spoken DOB representations.

    Return ISO YYYY-MM-DD when safely parseable.
    """

    text = " ".join(
        str(value).split()
    )

    text = _ORDINAL_RE.sub(
        r"\1",
        text,
    )

    formats = (
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%Y-%m-%d",
    )

    for pattern in formats:
        try:
            parsed = datetime.strptime(
                text,
                pattern,
            )
        except ValueError:
            continue

        return parsed.date().isoformat()

    return None


def _equivalent(
    *,
    fact: RequestedFact,
    asserted: object,
    authoritative: object,
) -> bool:
    """Compare one assertion using fact-appropriate normalization."""

    if fact is RequestedFact.DATE_OF_BIRTH:
        asserted_date = _parse_date(
            asserted
        )

        authoritative_date = _parse_date(
            authoritative
        )

        if (
            asserted_date is not None
            and authoritative_date is not None
        ):
            return (
                asserted_date
                == authoritative_date
            )

    if fact is RequestedFact.VISITED_BEFORE:
        if isinstance(
            authoritative,
            bool,
        ):
            asserted_text = _normalize_text(
                asserted
            )

            true_values = {
                "true",
                "yes",
                "yes i have",
                "yes i've visited before",
            }

            false_values = {
                "false",
                "no",
                "no i have not",
                "no i haven't visited before",
            }

            if asserted_text in true_values:
                return authoritative is True

            if asserted_text in false_values:
                return authoritative is False

    return (
        _normalize_text(
            asserted
        )
        ==
        _normalize_text(
            authoritative
        )
    )


def _authoritative_value(
    *,
    world: PatientWorldModel,
    assertion: AgentFactAssertion,
) -> object | None:
    key = _FACT_KEYS[
        assertion.fact
    ]

    return world.facts.get(
        key
    )


def ground_fact_assertions(
    *,
    world: PatientWorldModel,
    turn: TurnFrame,
) -> FactGrounding:
    """Compare all source-grounded remote assertions with caller truth."""

    conflicts: list[
        FactConflict
    ] = []

    matched: list[
        RequestedFact
    ] = []

    ungrounded: list[
        RequestedFact
    ] = []

    for assertion in turn.stated_facts:

        authoritative = _authoritative_value(
            world=world,
            assertion=assertion,
        )

        # If the scenario does not define the fact, do not guess whether
        # the remote assertion is right or wrong.
        if authoritative is None:
            ungrounded.append(
                assertion.fact
            )
            continue

        if _equivalent(
            fact=assertion.fact,
            asserted=assertion.value,
            authoritative=authoritative,
        ):
            matched.append(
                assertion.fact
            )
            continue

        conflicts.append(
            FactConflict(
                fact=assertion.fact,
                asserted_value=assertion.value,
                authoritative_value=str(
                    authoritative
                ),
            )
        )

    return FactGrounding(
        conflicts=tuple(
            conflicts
        ),
        matched_facts=tuple(
            matched
        ),
        ungrounded_facts=tuple(
            ungrounded
        ),
    )
