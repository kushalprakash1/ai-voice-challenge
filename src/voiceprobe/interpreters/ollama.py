"""Ollama-backed semantic interpreter for VoiceProbe.

The model converts natural tested-agent speech into constrained semantic
data. It never writes patient responses or modifies authoritative state.
"""

from __future__ import annotations

import json
from concurrent.futures import (
    CancelledError,
    Future,
    ThreadPoolExecutor,
)
from threading import Lock
from time import perf_counter

import httpx

from voiceprobe.conversation.meaning import TurnMeaning
from voiceprobe.conversation.state import PatientState
from voiceprobe.interpreters.semantic_gate import deterministic_turn_meaning
from voiceprobe.scenarios.models import PatientScenario
from voiceprobe.target_memory import target_memory_context

DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_URL = "http://127.0.0.1:11434/api/chat"


_GENERAL_SEMANTIC_FIELDS = (
    "response_expectation",
    "question_kind",
    "workflow_direction",
    "topic",
)


def _semantic_output_schema() -> dict[str, object]:
    """Build the stricter schema used only at the model boundary."""
    schema = TurnMeaning.model_json_schema()

    required = schema.setdefault("required", [])

    if not isinstance(required, list):
        raise TypeError("TurnMeaning JSON schema has invalid required metadata.")

    properties = schema.get("properties")

    if not isinstance(properties, dict):
        raise TypeError("TurnMeaning JSON schema has no properties mapping.")

    for field_name in _GENERAL_SEMANTIC_FIELDS:
        if field_name not in required:
            required.append(field_name)

        field_schema = properties.get(field_name)

        if isinstance(field_schema, dict):
            # A default tells a structured-output model that it may omit
            # classification. At the LLM boundary these fields are mandatory.
            field_schema.pop("default", None)

    return schema


class OllamaConversationInterpreter:
    """Extract constrained conversation meaning from natural speech."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        url: str = DEFAULT_URL,
        timeout_seconds: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._url = url

        self._client = client or httpx.Client(
            timeout=timeout_seconds,
        )
        self._owns_client = client is None

        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="voiceprobe-semantic-prefetch",
        )
        self._prefetch_lock = Lock()
        self._prefetch_future: Future[TurnMeaning] | None = None
        self._prefetch_turn: str | None = None
        self._prefetch_valid = False
        self._prefetch_started_at: float | None = None

    def close(self) -> None:
        """Release speculative worker and internally owned HTTP client."""
        self.invalidate_prefetch()

        self._prefetch_executor.shutdown(
            wait=True,
            cancel_futures=True,
        )

        if self._owns_client:
            self._client.close()

    @staticmethod
    def _normalize_turn(agent_turn: str) -> str:
        return " ".join(agent_turn.split())

    def prefetch(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        agent_turn: str,
    ) -> bool:
        """Start semantic interpretation before endpoint confirmation."""
        normalized_turn = self._normalize_turn(agent_turn)

        if not normalized_turn:
            return False

        if state.scenario_id != scenario.scenario_id:
            raise ValueError("PatientState does not belong to the supplied scenario.")

        # Obvious speech acts should not launch speculative model inference.
        # The authoritative interpret() path will run the same deterministic
        # gate when the finalized turn arrives.
        if (
            deterministic_turn_meaning(
                scenario=scenario,
                agent_turn=agent_turn,
            )
            is not None
        ):
            return False

        with self._prefetch_lock:
            existing = self._prefetch_future

            if existing is not None:
                if existing.done():
                    self._prefetch_future = None
                    self._prefetch_turn = None
                    self._prefetch_valid = False
                    self._prefetch_started_at = None
                else:
                    # Never stack speculative Ollama requests. A stale
                    # request is allowed to finish before another begins.
                    return False

            self._prefetch_turn = normalized_turn
            self._prefetch_valid = True
            self._prefetch_started_at = perf_counter()

            self._prefetch_future = self._prefetch_executor.submit(
                self._interpret_uncached,
                scenario=scenario,
                state=state,
                agent_turn=agent_turn,
            )

        return True

    def invalidate_prefetch(self) -> None:
        """Prevent a speculative result from being consumed."""
        with self._prefetch_lock:
            future = self._prefetch_future

            if future is None:
                return

            self._prefetch_valid = False

            # This succeeds only if the worker has not actually begun.
            if future.cancel():
                self._prefetch_future = None
                self._prefetch_turn = None
                self._prefetch_started_at = None

    def _clear_prefetch(
        self,
        future: Future[TurnMeaning],
    ) -> None:
        with self._prefetch_lock:
            if self._prefetch_future is future:
                self._prefetch_future = None
                self._prefetch_turn = None
                self._prefetch_valid = False
                self._prefetch_started_at = None

    def interpret(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        agent_turn: str,
    ) -> TurnMeaning:
        """Use deterministic meaning, speculative output, or normal extraction."""
        normalized_turn = self._normalize_turn(agent_turn)

        if state.scenario_id != scenario.scenario_id:
            raise ValueError("PatientState does not belong to the supplied scenario.")

        deterministic = deterministic_turn_meaning(
            scenario=scenario,
            agent_turn=agent_turn,
        )

        if deterministic is not None:
            # A partial speculative request may already be running. Mark it
            # unusable; never allow it to overwrite a deterministic result.
            self.invalidate_prefetch()
            return deterministic

        with self._prefetch_lock:
            future = self._prefetch_future
            prefetched_turn = self._prefetch_turn
            valid = self._prefetch_valid
            started_at = self._prefetch_started_at

        if future is not None:
            if valid and prefetched_turn == normalized_turn:
                wait_started = perf_counter()

                try:
                    result = future.result()
                except (
                    CancelledError,
                    httpx.HTTPError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    # Speculation is optional. Record the failure and let
                    # the normal authoritative path run below.
                    print(
                        f"[PREFETCH ERROR] {type(error).__name__}: {error}",
                        flush=True,
                    )
                    self._clear_prefetch(future)
                else:
                    wait_seconds = perf_counter() - wait_started

                    total_seconds = (
                        perf_counter() - started_at
                        if started_at is not None
                        else wait_seconds
                    )

                    overlap_seconds = max(
                        0.0,
                        total_seconds - wait_seconds,
                    )

                    self._clear_prefetch(future)

                    print(
                        "[PREFETCH HIT] "
                        f"overlap={overlap_seconds:.3f}s "
                        f"remaining_wait={wait_seconds:.3f}s",
                        flush=True,
                    )

                    return result

            else:
                # Speech continued or the assembled turn changed.
                # Never use the stale semantic result. Wait for that one
                # GPU request to leave Ollama before starting another,
                # avoiding concurrent competing requests to the same model.
                try:
                    future.result()
                except (
                    CancelledError,
                    httpx.HTTPError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    print(
                        f"[PREFETCH STALE ERROR] {type(error).__name__}: {error}",
                        flush=True,
                    )

                self._clear_prefetch(future)

                print(
                    "[PREFETCH STALE] discarded",
                    flush=True,
                )

        return self._interpret_uncached(
            scenario=scenario,
            state=state,
            agent_turn=agent_turn,
        )

    def _interpret_uncached(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        agent_turn: str,
    ) -> TurnMeaning:
        """Run the proven semantic extraction request directly."""
        if state.scenario_id != scenario.scenario_id:
            raise ValueError("PatientState does not belong to the supplied scenario.")

        context = {
            "conversation_objective": scenario.objective,
            "latest_tested_agent_turn": agent_turn,
            "target_memory": target_memory_context(),
        }

        response = self._client.post(
            self._url,
            json={
                "model": self._model,
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0,
                    "num_predict": 256,
                },
                "format": _semantic_output_schema(),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a neutral semantic extraction component. "
                            "Analyze only what the tested medical scheduling "
                            "voice agent actually said. Do not decide whether "
                            "the agent is correct and do not substitute patient "
                            "ground truth. Return structured data only. "
                            "Use this patient-fact ontology: "
                            "name = the patient's full name or general identity; "
                            "first_name = specifically the patient's given or first name; "
                            "last_name = specifically the patient's family, surname, or last name; "
                            "patient_status = whether the caller is a new, returning, or existing patient; "
                            "visited_before = whether the caller has visited, been seen by, "
                            "or been a patient at this practice before; "
                            "appointment_type = the requested appointment category, such as "
                            "new-patient consultation, follow-up, routine/general office visit, "
                            "urgent concern, or procedure; "
                            "complaint = symptoms, body problem, reason for the "
                            "visit, reason for calling, or what brought them in; "
                            "duration = how long the problem has existed or when "
                            "it began; "
                            "date_of_birth = date of birth, DOB, or birthday; "
                            "insurance = insurance, coverage, insurer, carrier, "
                            "or who the patient is covered through; "
                            "preferred_day = desired appointment day or date; "
                            "preferred_time = desired appointment time, morning, "
                            "afternoon, evening, or other daypart. "
                            "requested_facts contains every fact the agent asks "
                            "the patient to provide, verify, confirm, or repeat. "
                            "The wording can be direct or indirect. "
                            "For example, 'What brought you in?' requests "
                            "complaint. 'How long has this been going on?' "
                            "requests duration. 'Who am I speaking with?' "
                            "requests name. 'Who are you covered through?' "
                            "requests insurance. 'Which day works?' requests "
                            "preferred_day. "
                            "Important hard-negative examples from observed calls: "
                            "'Would you like to create a demo patient profile?' is a "
                            "workflow_permission and does not request a name. "
                            "'For the profile, I need your first name. What should I "
                            "enter for you?' requests first_name and is NOT a workflow "
                            "permission merely because it mentions a profile. "
                            "'What type of appointment do you need: new patient "
                            "consultation, follow-up, office visit, or something else?' "
                            "requests appointment_type. "
                            "'Are you a patient, or have you visited us before?' requests "
                            "patient_status and visited_before. "
                            "'Is this a routine checkup, follow-up, urgent concern, or "
                            "specific procedure?' requests appointment_type. "
                            "stated_facts is different. Add a stated fact only "
                            "when the agent itself supplies a specific candidate "
                            "value, assumption, or summary. Do not create a "
                            "stated fact merely because a fact is being asked "
                            "about. "
                            "A confirmation question can both request and state "
                            "facts. For example, 'So your left knee has been "
                            "hurting for two weeks, right?' requests complaint "
                            "and duration and also states complaint='left knee' "
                            "and duration='two weeks'. Preserve what was spoken. "
                            "appointment_offer is null unless the agent offers "
                            "an appointment day, time, or slot. "
                            "booking_confirmed is true only when an appointment "
                            "is explicitly said to be booked or confirmed. "
                            "conversation_end_requested is true only when the "
                            "agent explicitly closes or ends the conversation, "
                            "for example 'Okay, bye', 'Goodbye', or 'Have a good "
                            "day, bye'. Do not mark ordinary acknowledgements such "
                            "as 'okay', 'great', or 'sounds good' as conversation "
                            "ending unless they clearly close the call. "
                            "requests_repetition is true when the agent wants "
                            "the patient to repeat something because it was not "
                            "heard or understood. "
                            "response_expectation describes the general form "
                            "of reply the agent is currently soliciting. Use "
                            "yes_no for questions that expect yes or no, including "
                            "permission, consent, or whether to proceed; fact when "
                            "patient information is requested; choice when the "
                            "patient must select among alternatives; acknowledgement "
                            "when a simple acknowledgement is explicitly requested; "
                            "freeform for an open-ended response; and none when the "
                            "agent is not currently soliciting a response. "
                            "topic is a short neutral description of the subject of "
                            "the current question or request. Do not put inferred "
                            "patient ground truth into topic. "
                            "These general fields never override requested_facts, "
                            "stated_facts, appointment_offer, booking_confirmed, "
                            "requests_repetition, or conversation_end_requested. "
                            "A concrete appointment slot question must still populate "
                            "appointment_offer. "
                            "question_kind describes what kind of question "
                            "requires the patient's response. Use "
                            "workflow_permission when the agent asks whether it "
                            "may perform, start, create, prepare, verify, continue, "
                            "proceed with, stop, cancel, or otherwise control a "
                            "workflow action. Use patient_attribute when the agent "
                            "asks whether the patient is or has something, especially "
                            "when that attribute is not represented by the supported "
                            "patient-fact ontology. Do not convert an unsupported "
                            "patient attribute into the closest known patient fact. "
                            "Use other when neither category fits, and none when "
                            "there is no response-triggering question. "
                            "Classify the speech act before extracting patient facts. "
                            "A question asking whether the agent should, may, or is "
                            "wanted to perform an action is workflow_permission and "
                            "yes_no. This includes forms such as 'Would you like me to "
                            "...?', 'Do you want me to ...?', 'Should I ...?', and "
                            "'May I ...?'. The object of that proposed agent action is "
                            "not automatically a patient fact being requested. For "
                            "example, asking permission to create or prepare a profile "
                            "does not itself request the patient's name, insurance, or "
                            "other profile fields. Asking permission to check appointment "
                            "availability for a stated day or time does not itself request "
                            "preferred_day or preferred_time. Populate requested_facts "
                            "only when the patient is actually being asked to provide, "
                            "state, verify, confirm, or repeat the value of that fact. "
                            "Mentions of a fact inside an action the agent proposes to "
                            "perform are not requested_facts. "
                            "Patient_attribute is for questions about whether the patient "
                            "is, has, or satisfies an attribute, not for questions about "
                            "whether the agent should perform an action. "
                            "A request to check availability is not an appointment_offer "
                            "unless the agent actually presents a concrete available slot "
                            "to the patient. "
                            "workflow_direction is determined only from the literal "
                            "direction of the agent action in the current utterance. "
                            "Do not compare it with any larger scheduling objective, "
                            "desired outcome, realism judgment, or whether a profile is "
                            "temporary, demo, test, permanent, or production. "
                            "Creating, setting up, preparing, checking, verifying, "
                            "starting, proceeding, or continuing an ordinary workflow "
                            "action is continue. Explicitly stopping, cancelling, "
                            "abandoning, terminating, or reversing the workflow is stop. "
                            "For workflow_permission, use continue when the agent is "
                            "asking to perform or proceed with a normal workflow step "
                            "such as setup, intake, profile creation, verification, "
                            "continuation, or scheduling. Use stop when the requested "
                            "action would stop, cancel, abandon, terminate, or reverse "
                            "the workflow. Use unknown only when the direction itself "
                            "cannot be determined. Use none for patient_attribute and "
                            "other questions. "
                            "Do not label a normal setup or intake step as stop merely "
                            "because the wording describes a temporary, test, or demo "
                            "profile. Classify the direction of the requested action "
                            "itself. "
                            "Every explicit question or request must classify "
                            "response_expectation according to the reply it asks for. "
                            "Do not use none merely because its subject is outside the "
                            "patient-fact ontology. Focus on the latest actionable "
                            "question or request even when it follows greetings, legal "
                            "notices, language-selection prompts, acknowledgements, or "
                            "other introductory speech. "
                            "Permission and proceed-or-not questions are yes_no. "
                            "This includes requests asking whether the agent may "
                            "continue, create or prepare something needed for intake, "
                            "verify information, or proceed with the scheduling flow. "
                            "For a yes_no workflow request, classify workflow_relation "
                            "against conversation_objective. Use advances_objective only "
                            "when agreeing directly performs the objective or when the "
                            "agent indicates that the requested action is required or "
                            "necessary to continue toward the objective. Required intake, "
                            "identity or eligibility verification, checking requested "
                            "appointment availability, and scheduling can advance the "
                            "objective when the utterance makes that relationship clear. "
                            "An optional side workflow does not automatically advance the "
                            "objective merely because it concerns setup, intake, an "
                            "account, a profile, preferences, or another ancillary action. "
                            "When the agent presents a step as optional and does not "
                            "indicate that it is required to continue toward the objective, "
                            "use none. If agreeing would stop, cancel, abandon, terminate, "
                            "or reverse progress toward the objective, use "
                            "opposes_objective. Use uncertain when the utterance does not "
                            "provide enough information to determine the relationship. "
                            "If the question asks about a patient attribute rather than "
                            "permission to perform workflow, use none unless that attribute "
                            "itself changes the objective. "
                            "Do not force genuinely unsupported patient attributes "
                            "into the closest known fact. However, first_name, last_name, "
                            "patient_status, visited_before, and appointment_type ARE "
                            "supported facts and must populate requested_facts when asked. "
                            "If some other unsupported fact is asked, leave requested_facts "
                            "empty while still classifying response expectation and topic. "
                            "Target memory is prior behavioral context only. Never treat "
                            "a target-memory entry as words spoken in the current turn. "
                            "An appointment yes/no question must still populate "
                            "appointment_offer, and a supported patient-fact question "
                            "must still populate requested_facts. Those specific "
                            "structured semantics take priority downstream. "
                            "Never infer that an understood explicit question requires "
                            "no response solely because none of the old specialized "
                            "fields apply. "
                            "unclear is true only when the utterance itself "
                            "cannot be reliably interpreted."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            context,
                            separators=(",", ":"),
                        ),
                    },
                ],
            },
        )

        response.raise_for_status()

        payload = response.json()

        try:
            content = payload["message"]["content"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                "Ollama response did not contain assistant content."
            ) from error

        if not isinstance(content, str):
            raise TypeError("Ollama assistant content was not text.")

        return TurnMeaning.model_validate_json(content)
