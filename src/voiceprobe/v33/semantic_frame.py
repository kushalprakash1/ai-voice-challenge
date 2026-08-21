"""SemanticLab v2 meaning contract for VoiceProbe v3.3.

This module represents *observed remote meaning only*.  It deliberately keeps
patient truth, mission preferences, strategic actions, and patient prose out of
semantic perception.

The existing v0.17 reasoner/planner is not wired to this module yet.  Phase 1
establishes the contract and invariants first so later interpreters can target a
stable representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .world_model import ObservationKind, RemoteObservation


class SpeechAct(StrEnum):
    GREETING = "greeting"
    QUESTION = "question"
    REQUEST = "request"
    OFFER = "offer"
    STATEMENT = "statement"
    CONFIRMATION = "confirmation"
    REJECTION = "rejection"
    ACKNOWLEDGEMENT = "acknowledgement"
    PRESENCE_CHECK = "presence_check"
    CLARIFICATION = "clarification"
    OTHER = "other"


class SemanticTopic(StrEnum):
    PROFILE = "profile"
    IDENTITY = "identity"
    PATIENT_FACT = "patient_fact"
    RESCHEDULE_REASON = "reschedule_reason"
    OPEN_INTENT = "open_intent"
    APPOINTMENT_STATE = "appointment_state"
    VISIT_TYPE = "visit_type"
    PROVIDER = "provider"
    AVAILABILITY = "availability"
    TRANSACTION = "transaction"
    PRESENCE = "presence"
    CAPABILITY = "capability"
    OTHER = "other"


class ConstraintAxis(StrEnum):
    DAY = "day"
    TIME_OF_DAY = "time_of_day"
    PROVIDER = "provider"


class RecordClaim(StrEnum):
    PROFILE_EXISTS = "profile_exists"
    PROFILE_MISSING = "profile_missing"
    APPOINTMENT_EXISTS = "appointment_exists"
    APPOINTMENT_MISSING = "appointment_missing"


class TransactionOperation(StrEnum):
    NONE = "none"
    BOOK = "book"
    RESCHEDULE = "reschedule"
    CANCEL = "cancel"
    KEEP = "keep"
    CREATE_PROFILE = "create_profile"
    SEARCH = "search"


class TransactionSignal(StrEnum):
    NONE = "none"
    PROPOSED = "proposed"
    PERMISSION_REQUEST = "permission_request"
    CONFIRMED = "confirmed"


class ReferenceKind(StrEnum):
    NONE = "none"
    PRIOR_OPTION = "prior_option"
    PRIOR_DAY = "prior_day"
    PRIOR_TIME = "prior_time"
    PRIOR_PROVIDER = "prior_provider"
    PRIOR_ENTITY = "prior_entity"
    AMBIGUOUS = "ambiguous"


class AmbiguityKind(StrEnum):
    NONE = "none"
    TEMPORAL_REFERENCE = "temporal_reference"
    OPTION_REFERENCE = "option_reference"
    RECORD_REFERENCE = "record_reference"
    TRANSACTION_REFERENCE = "transaction_reference"
    INTENT = "intent"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class SemanticAmbiguity:
    """Explicit uncertainty instead of forcing a single semantic guess."""

    kind: AmbiguityKind = AmbiguityKind.NONE
    candidates: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("Semantic ambiguity candidates must be unique.")

        if self.kind is AmbiguityKind.NONE:
            if self.candidates or self.detail.strip():
                raise ValueError(
                    "Ambiguity details/candidates require a non-NONE ambiguity kind."
                )
            return

        if len(self.candidates) < 2:
            raise ValueError(
                "A non-NONE semantic ambiguity must expose at least two candidates."
            )


class UnresolvedSemanticFrameError(ValueError):
    """Raised when code tries to act on a frame that still needs resolution."""


def _ensure_unique(name: str, values: tuple[object, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate values.")


@dataclass(frozen=True, slots=True)
class SemanticFrame:
    """Context-independent semantic facts extracted from one remote turn.

    Important design rule:
    - `failed_constraints` = what the remote side says is unavailable.
    - `proposed_changes` = dimensions the remote side proposes changing/searching.
    - `retained_constraints` = dimensions the remote side explicitly keeps fixed.

    `requires_response`, final ObservationKind, and patient strategy are derived;
    they are intentionally *not* independent model predictions.
    """

    raw_text: str
    speech_act: SpeechAct
    topic: SemanticTopic = SemanticTopic.OTHER

    requested_fact: str = ""

    failed_constraints: tuple[ConstraintAxis, ...] = ()
    proposed_changes: tuple[ConstraintAxis, ...] = ()
    retained_constraints: tuple[ConstraintAxis, ...] = ()

    offered_options: tuple[str, ...] = ()
    selected_option: str = ""

    record_claims: tuple[RecordClaim, ...] = ()

    transaction_operation: TransactionOperation = TransactionOperation.NONE
    transaction_signal: TransactionSignal = TransactionSignal.NONE

    reference: ReferenceKind = ReferenceKind.NONE
    ambiguity: SemanticAmbiguity = field(default_factory=SemanticAmbiguity)

    def __post_init__(self) -> None:
        if not self.raw_text.strip():
            raise ValueError("SemanticFrame.raw_text must not be empty.")

        _ensure_unique("failed_constraints", self.failed_constraints)
        _ensure_unique("proposed_changes", self.proposed_changes)
        _ensure_unique("retained_constraints", self.retained_constraints)
        _ensure_unique("offered_options", self.offered_options)
        _ensure_unique("record_claims", self.record_claims)

        changed_and_retained = (
            set(self.proposed_changes)
            & set(self.retained_constraints)
        )
        if changed_and_retained:
            labels = ", ".join(
                sorted(axis.value for axis in changed_and_retained)
            )
            raise ValueError(
                "A constraint cannot be both proposed-to-change and retained "
                f"in the same frame: {labels}"
            )

        if (
            self.speech_act is SpeechAct.PRESENCE_CHECK
            and self.topic is not SemanticTopic.PRESENCE
        ):
            raise ValueError(
                "PRESENCE_CHECK speech acts must use the PRESENCE topic."
            )

        if (
            self.reference is ReferenceKind.AMBIGUOUS
            and self.ambiguity.kind is AmbiguityKind.NONE
        ):
            raise ValueError(
                "AMBIGUOUS references require an explicit SemanticAmbiguity."
            )

    @property
    def has_unresolved_ambiguity(self) -> bool:
        return self.ambiguity.kind is not AmbiguityKind.NONE

    @property
    def requires_response(self) -> bool:
        """Derive whether the remote turn expects the patient to respond."""

        if self.transaction_signal is TransactionSignal.PERMISSION_REQUEST:
            return True

        return self.speech_act in {
            SpeechAct.QUESTION,
            SpeechAct.REQUEST,
            SpeechAct.OFFER,
            SpeechAct.PRESENCE_CHECK,
            SpeechAct.CLARIFICATION,
        }

    @property
    def search_constraints(self) -> tuple[str, ...]:
        """Expose proposed search/change dimensions for the existing world model."""

        return tuple(axis.value for axis in self.proposed_changes)

    @property
    def unavailable_constraints(self) -> tuple[str, ...]:
        return tuple(axis.value for axis in self.failed_constraints)

    @property
    def remote_claim_values(self) -> tuple[str, ...]:
        return tuple(claim.value for claim in self.record_claims)

    def derive_observation_kind(self) -> ObservationKind:
        """Derive coarse legacy kind from the richer frame.

        This is intentionally deterministic.  The semantic interpreter should
        describe meaning; it should not independently guess several redundant
        labels that can contradict each other.
        """

        if self.has_unresolved_ambiguity:
            raise UnresolvedSemanticFrameError(
                "Resolve SemanticFrame ambiguity before deriving an observation."
            )

        if self.transaction_signal is TransactionSignal.CONFIRMED:
            return ObservationKind.TRANSACTION_CONFIRMED

        if self.transaction_signal is TransactionSignal.PERMISSION_REQUEST:
            return ObservationKind.TRANSACTION_PERMISSION_REQUEST

        if self.speech_act is SpeechAct.PRESENCE_CHECK:
            return ObservationKind.PRESENCE_CHECK

        if self.requested_fact == "reschedule_reason":
            return ObservationKind.RESCHEDULE_REASON_REQUEST

        if (
            self.topic is SemanticTopic.PROFILE
            and self.requires_response
        ):
            return ObservationKind.PROFILE_REQUEST

        if self.requested_fact:
            return ObservationKind.FACT_REQUEST

        # A proposed fallback/search dimension is authoritative evidence that
        # this is an alternative-search interaction.  This intentionally
        # mirrors the structural repair made in v0.17.
        if self.proposed_changes:
            return ObservationKind.ALTERNATIVE_SEARCH_OFFER

        if self.offered_options:
            return ObservationKind.OPTION_OFFER

        if self.failed_constraints:
            return ObservationKind.AVAILABILITY_RESULT

        if self.topic is SemanticTopic.APPOINTMENT_STATE:
            return ObservationKind.APPOINTMENT_STATE

        if (
            self.topic is SemanticTopic.VISIT_TYPE
            and self.requires_response
        ):
            return ObservationKind.VISIT_TYPE_REQUEST

        if (
            self.topic is SemanticTopic.PROVIDER
            and self.requires_response
        ):
            return ObservationKind.PROVIDER_PREFERENCE_REQUEST

        if (
            self.topic is SemanticTopic.OPEN_INTENT
            and self.requires_response
        ):
            return ObservationKind.OPEN_INTENT

        if self.speech_act is SpeechAct.GREETING:
            return ObservationKind.GREETING

        if self.speech_act is SpeechAct.ACKNOWLEDGEMENT:
            return ObservationKind.ACKNOWLEDGEMENT

        if self.speech_act is SpeechAct.CLARIFICATION:
            return ObservationKind.CLARIFICATION_REQUEST

        if self.topic is SemanticTopic.CAPABILITY:
            return ObservationKind.CAPABILITY_RESPONSE

        return ObservationKind.OTHER

    def to_remote_observation(self) -> RemoteObservation:
        """Adapt a resolved SemanticFrame to the existing v3.3 world contract."""

        kind = self.derive_observation_kind()

        operation = (
            ""
            if self.transaction_operation is TransactionOperation.NONE
            else self.transaction_operation.value
        )

        return RemoteObservation(
            kind=kind,
            raw_text=self.raw_text,
            requires_response=self.requires_response,
            requested_fact=self.requested_fact,
            offered_options=self.offered_options,
            unavailable_constraints=self.unavailable_constraints,
            remote_claims=self.remote_claim_values,
            selected_option=self.selected_option,
            transaction_operation=operation,
            search_constraints=self.search_constraints,
        )
