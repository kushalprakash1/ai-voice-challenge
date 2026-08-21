"""Pipecat-Flows-compatible fallback node factories for VoiceProbe v3.

NodeConfig is a TypedDict in Pipecat. Returning ordinary dictionaries keeps the
stable VoiceProbe test environment independent of Pipecat while remaining
directly consumable by FlowManager in the v3 runtime environment.
"""

from __future__ import annotations

from typing import Any

from .flow_state import FlowSnapshot, FlowStage
from .models import PatientFacts


_ROLE_MESSAGE = """You are the fallback language-understanding layer for an autonomous
synthetic patient scheduling caller. Authoritative patient facts are supplied by
the application. Never invent, replace, or infer patient facts. Answer only the
latest actionable request from the remote scheduling agent. Do not restate the
overall scheduling objective unless the remote agent asks an open-ended intent
question. Do not advance the workflow unprompted. If the remote turn is only a
status update or acknowledgement, do not manufacture a response."""


_STAGE_TASKS: dict[FlowStage, str] = {
    FlowStage.PROFILE: (
        "Handle only profile-creation language. Determine whether the remote "
        "agent is asking permission to create the required demo patient profile."
    ),
    FlowStage.IDENTITY: (
        "Handle only identity collection. Determine exactly which name field "
        "the remote agent requested."
    ),
    FlowStage.DOB: (
        "Handle date-of-birth collection or correction. Preserve the "
        "authoritative DOB exactly."
    ),
    FlowStage.VISIT_REASON: (
        "Handle the reason-for-visit request. Do not substitute scheduling "
        "date/time preferences for the complaint."
    ),
    FlowStage.APPOINTMENT_TYPE: (
        "Handle visit-type choices such as new patient consultation, follow-up, "
        "or routine office visit."
    ),
    FlowStage.INSURANCE: (
        "Handle insurance questions only."
    ),
    FlowStage.DATE_TIME: (
        "Handle availability/search language while preserving the hard Friday "
        "afternoon preference."
    ),
    FlowStage.PROVIDER: (
        "Handle provider choices. The authoritative preference is first "
        "available / any available provider."
    ),
    FlowStage.SLOT: (
        "Handle concrete appointment-slot offers. Do not claim a slot is booked "
        "until the remote agent explicitly confirms it."
    ),
    FlowStage.CONFIRMATION: (
        "Verify whether a concrete accepted slot has actually been booked. "
        "Do not infer confirmation from a proposal or search status."
    ),
    FlowStage.COMPLETE: (
        "The scheduling objective is complete. Do not create additional "
        "workflow steps."
    ),
}


def build_fallback_node(
    snapshot: FlowSnapshot,
    *,
    facts: PatientFacts | None = None,
) -> dict[str, Any]:
    """Build a focused Pipecat NodeConfig-compatible dictionary."""

    patient = facts or PatientFacts()
    stage = snapshot.current_stage

    facts_block = (
        f"Authoritative facts: name={patient.first_name} {patient.last_name}; "
        f"DOB={patient.dob}; insurance={patient.insurance}; "
        f"complaint={patient.complaint}; duration={patient.symptom_duration}; "
        f"appointment_type={patient.appointment_type}; "
        f"preferred_day={patient.preferred_day}; "
        f"preferred_time={patient.preferred_time}; "
        f"provider_preference={patient.provider_preference}."
    )

    progress = (
        "Communicated stages: "
        + ", ".join(
            sorted(stage.value for stage in snapshot.communicated)
        )
        + ". Confirmed stages: "
        + ", ".join(
            sorted(stage.value for stage in snapshot.confirmed)
        )
        + "."
    )

    return {
        "name": f"voiceprobe_{stage.value}",
        "role_message": _ROLE_MESSAGE,
        "task_messages": [
            {
                "role": "developer",
                "content": _STAGE_TASKS[stage],
            },
            {
                "role": "developer",
                "content": facts_block,
            },
            {
                "role": "developer",
                "content": progress,
            },
        ],
        "functions": [],
    }
