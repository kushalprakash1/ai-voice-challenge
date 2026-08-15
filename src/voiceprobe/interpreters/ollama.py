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
from voiceprobe.scenarios.models import PatientScenario

DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_URL = "http://127.0.0.1:11434/api/chat"


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
        """Use a matching speculative result or run normal extraction."""
        normalized_turn = self._normalize_turn(agent_turn)

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
            "latest_tested_agent_turn": agent_turn,
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
                "format": TurnMeaning.model_json_schema(),
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
                            "name = the patient's name or identity; "
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
                            "requests_repetition is true when the agent wants "
                            "the patient to repeat something because it was not "
                            "heard or understood. "
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
