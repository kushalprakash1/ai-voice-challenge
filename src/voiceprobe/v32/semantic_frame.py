
"""Orthogonal semantic frame for VoiceProbe v3.2.

The model interprets language only.
It does not decide VoiceProbe actions or mutate state.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class SpeechAct(StrEnum):
    ASK = "ask"
    INFORM = "inform"
    ACKNOWLEDGE = "acknowledge"
    REQUEST = "request"
    OFFER = "offer"
    FRAGMENT = "fragment"
    OTHER = "other"


class Operation(StrEnum):
    NONE = "none"
    BOOK = "book"
    RESCHEDULE = "reschedule"
    CANCEL = "cancel"
    KEEP = "keep"
    LIST_SLOTS = "list_slots"
    CHOOSE_PROVIDER = "choose_provider"


class Focus(StrEnum):
    NONE = "none"

    # Why the patient is seeking medical care / why the visit exists.
    # This is distinct from WHY an already-existing appointment is moved.
    VISIT_REASON = "visit_reason"

    # Why an existing appointment is being changed.
    RESCHEDULE_REASON = "reschedule_reason"

    INSURANCE = "insurance"
    PROVIDER_PREFERENCE = "provider_preference"
    DOB = "dob"
    NAME = "name"
    COMPLAINT = "complaint"
    PREFERRED_DAY = "preferred_day"
    PREFERRED_TIME = "preferred_time"

    APPOINTMENT_STATUS = "appointment_status"
    SLOT_OPTIONS_INTRO = "slot_options_intro"

    OTHER = "other"


class Commitment(StrEnum):
    NONE = "none"

    # Merely discussing or requesting information.
    INFORMATIONAL = "informational"

    # Clinic is asking whether it may perform a transaction.
    PERMISSION_REQUEST = "permission_request"

    # Patient explicitly gives transaction permission.
    AUTHORIZATION = "authorization"

    # Clinic reports a completed transaction.
    CONFIRMATION = "confirmation"


class Certainty(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SemanticFrame(BaseModel):
    """Strict model output.

    Notice what is deliberately absent:

    - no response text
    - no VoiceProbe action
    - no requires_response boolean
    - no transaction-state mutation
    - no numeric self-confidence
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    speech_act: SpeechAct
    operation: Operation
    focus: Focus
    commitment: Commitment
    certainty: Certainty


SEMANTIC_FRAME_SCHEMA = SemanticFrame.model_json_schema()
