"""Statistical semantic intent classifier for VoiceProbe v3.1.

This module intentionally has no LLM dependency. It is a closed-domain
classifier for scheduling dialogue built from:
- word unigram and bigram features,
- character 3/4/5-gram features,
- TF-IDF weighting,
- nearest-prototype similarity,
- class-centroid similarity,
- flow-stage priors,
- explicit out-of-domain examples.

It does not generate patient speech and cannot mutate scheduling state.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence


_TOKEN_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)

ACCEPT_SCORE = 0.36
ACCEPT_MARGIN = 0.10


@dataclass(frozen=True, slots=True)
class StatisticalIntentResult:
    intent: str
    score: float
    margin: float
    prototype_similarity: float
    centroid_similarity: float
    top_candidates: tuple[tuple[str, float], ...]


_CORPUS: Mapping[str, tuple[str, ...]] = {
    "visit_reason_request": (
        "what is the reason for your visit",
        "what is the reason for your appointment",
        "what brings you in today",
        "what brings you in",
        "why are you coming in",
        "why do you need to be seen",
        "what are we seeing you for",
        "what are you being seen for",
        "what is this appointment for",
        "what concern are you coming in for",
        "what specific concern do you have",
        "what problem are you having",
        "what symptoms are you having",
        "tell me the reason you need the appointment",
        "can you tell me why you need the appointment",
        "what issue would you like addressed",
        "what is going on that you need an appointment",
        "why would you like to be seen",
        "what is the purpose of the visit",
        "what is the medical concern today",
        "what seems to be the problem",
        "what are you looking to be seen for",
        "what is bothering you",
        "what is the reason you are scheduling",
        "what condition are we evaluating",
        "what do you need to see the doctor about",
        "what is the main issue for this visit",
        "tell me what is bringing you to the office",
    ),
    "appointment_type_request": (
        "what type of appointment do you need",
        "what kind of appointment do you need",
        "what kind of visit are we scheduling",
        "what type of visit is this",
        "is this a new patient consultation or a follow up",
        "is this a follow up visit",
        "are you scheduling a new patient consultation",
        "what appointment category is this",
        "what kind of consultation is this",
        "is this for a routine visit or a follow up",
        "is this an office visit or consultation",
        "what type of service should i schedule",
        "which visit type do you need",
        "what appointment type should i put down",
        "are you a new patient or follow up",
        "is this a consultation or follow up",
        "which appointment type should i select",
        "what kind of office visit do you need",
    ),
    "insurance_request": (
        "what insurance do you have",
        "what is your insurance provider",
        "which insurance carrier do you use",
        "who are you covered through",
        "what coverage do you have",
        "which insurer do you have",
        "what insurance plan are you on",
        "can i get your insurance information",
        "who is your health insurance with",
        "which carrier provides your coverage",
        "what company is your insurance through",
        "who is your insurer",
        "what insurance are you using for this visit",
        "which health plan do you carry",
        "who provides your medical coverage",
        "what carrier is on your insurance card",
        "which plan should i enter for insurance",
    ),
    "dob_request": (
        "what is your date of birth",
        "what is your dob",
        "what is your birthday",
        "can i have your birthday",
        "when were you born",
        "can you provide your date of birth",
        "tell me your birth date",
        "what birth date do you have",
        "may i get your dob",
        "i need your date of birth",
        "please provide your birthday",
        "can you confirm your date of birth",
        "what birthday should i put on the profile",
    ),
    "dob_assertion": (
        "your date of birth is july fourth two thousand",
        "i have your birthday as july fourth",
        "your dob is listed as july fourth",
        "i have your date of birth as july fourth 2000",
        "your birthday on file is july fourth 2000",
        "the date of birth i have is july fourth 2000",
        "your profile shows a july fourth birthday",
        "i have july fourth two thousand as your dob",
    ),
    "full_name_request": (
        "what is your full name",
        "can i have your full name",
        "what is your first and last name",
        "who am i speaking with",
        "may i have your name",
        "what name should i put down",
        "tell me your full name",
        "can i get your first and last name",
        "what is the patient name",
        "who is calling",
        "what name is this appointment under",
    ),
    "first_name_request": (
        "what is your first name",
        "can i get your first name",
        "what is your given name",
        "what should i enter for your first name",
        "may i have your first name",
        "tell me your first name",
        "which first name should i use",
        "what is the patient's first name",
    ),
    "last_name_request": (
        "what is your last name",
        "what is your surname",
        "what is your family name",
        "can i get your last name",
        "may i have your surname",
        "tell me your last name",
        "what should i enter for your last name",
        "which family name should i use",
        "what is the patient's surname",
    ),
    "provider_preference_request": (
        "do you have a provider preference",
        "do you prefer a particular doctor",
        "which provider would you like",
        "is first available okay",
        "do you have a doctor preference",
        "any preference for the physician",
        "would you like a specific provider",
        "should i choose the first available provider",
        "which doctor do you want to see",
        "does the provider matter to you",
        "are you okay with any available doctor",
    ),
    "date_time_preference_request": (
        "what day and time would you prefer",
        "when would you like the appointment",
        "what date and time works for you",
        "what day works best",
        "what time works best",
        "when are you available",
        "what appointment time do you want",
        "which day should i look at",
        "what date should i schedule",
        "do mornings or afternoons work better",
        "when would you like to come in",
        "what is your preferred appointment time",
        "which date is best for you",
        "what day would you like to be seen",
    ),
    "presence_check": (
        "are you still there",
        "can you hear me",
        "hello are you there",
        "are you there",
        "did i lose you",
        "can you still hear me",
        "are you still on the line",
        "you still there",
        "hello can you hear me",
        "are we still connected",
    ),
    "open_ended_help": (
        "how may i help you today",
        "may i help you",
        "what can i help you with",
        "how can i help you",
        "what are you calling about today",
        "what can i do for you",
        "how can we help",
        "what do you need help with",
        "what can i assist you with",
        "how may i assist you",
        "what are you calling us for",
        "what can we do for you today",
    ),
    "profile_create_request": (
        "would you like to create a demo patient profile",
        "may i set up a patient profile for you",
        "should i create your patient profile",
        "would you like me to make a profile",
        "can i create a patient profile for you",
        "do you want to set up a profile",
        "shall i create a new patient profile",
    ),
    "acknowledgement": (
        "thanks alex",
        "okay thank you",
        "great",
        "thank you",
        "got it",
        "all right",
        "perfect thank you",
        "okay",
        "sounds good",
    ),
    "status_update": (
        "let me check that for you",
        "i am looking at availability now",
        "your profile is set up",
        "one moment while i check",
        "i am checking the schedule",
        "let me see what is available",
        "please hold while i look",
        "give me a moment to check availability",
        "i am pulling up the calendar",
    ),
    "scheduling_complex": (
        "would you like a different provider or another day",
        "should i check another day or a different provider",
        "there is nothing friday afternoon would you like another day",
        "i can check earlier in the week or another provider",
        "would you like me to look at other afternoon options",
        "do you want a different day instead",
        "should i search other availability",
    ),
    "unknown": (
        "what is your address",
        "what is your phone number",
        "what pharmacy do you use",
        "do you need directions",
        "are you looking for directions",
        "what is your email address",
        "do you need a prescription refill",
        "what is your fax number",
        "what are your office hours",
        "where should i park",
        "do you want to leave a message",
        "i have a question about billing",
        "the weather is nice today",
        "we are closed on weekends",
        "what is your social security number",
        "which pharmacy should we send this to",
        "would you like to speak with billing",
        "do you need a referral",
        "what is your home address",
        "would you like directions to the clinic",
    ),
}


_STAGE_PRIORS: Mapping[str, Mapping[str, float]] = {
    "profile": {
        "profile_create_request": 1.0,
        "full_name_request": 0.8,
    },
    "identity": {
        "full_name_request": 0.8,
        "first_name_request": 1.0,
        "last_name_request": 1.0,
    },
    "dob": {
        "dob_request": 1.0,
        "dob_assertion": 1.0,
        "open_ended_help": 0.35,
    },
    "visit_reason": {
        "visit_reason_request": 1.0,
        "open_ended_help": 0.55,
        "presence_check": 0.20,
    },
    "appointment_type": {
        "appointment_type_request": 1.0,
    },
    "insurance": {
        "insurance_request": 1.0,
    },
    "date_time": {
        "date_time_preference_request": 1.0,
        "scheduling_complex": 0.65,
    },
    "provider": {
        "provider_preference_request": 1.0,
        "scheduling_complex": 0.45,
    },
    "slot": {
        "scheduling_complex": 0.55,
    },
}


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.casefold()))


def _raw_features(text: str) -> Counter[str]:
    normalized = _normalize(text)
    tokens = normalized.split()
    result: Counter[str] = Counter()

    for token in tokens:
        result[f"w:{token}"] += 1.0

    for left, right in zip(tokens, tokens[1:]):
        result[f"b:{left}_{right}"] += 1.2

    compact = f"^{normalized.replace(' ', '_')}$"
    for width in (3, 4, 5):
        for index in range(max(0, len(compact) - width + 1)):
            result[f"c{width}:{compact[index:index + width]}"] += 0.18

    return result


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def _normalize_vector(vector: Counter[str]) -> Counter[str]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0.0:
        return Counter()
    return Counter({key: value / norm for key, value in vector.items()})


class StatisticalIntentScorer:
    """TF-IDF nearest-prototype + centroid semantic classifier."""

    def __init__(self) -> None:
        documents: list[tuple[str, Counter[str]]] = []
        document_frequency: Counter[str] = Counter()

        for intent, examples in _CORPUS.items():
            for example in examples:
                vector = _raw_features(example)
                documents.append((intent, vector))
                document_frequency.update(vector.keys())

        document_count = len(documents)
        self._idf = {
            key: math.log((document_count + 1) / (count + 1)) + 1.0
            for key, count in document_frequency.items()
        }
        self._unknown_idf = math.log(document_count + 1) + 1.0

        self._prototypes: dict[str, tuple[Counter[str], ...]] = {}
        self._centroids: dict[str, Counter[str]] = {}

        for intent in _CORPUS:
            vectors = tuple(
                self._vectorize_raw(raw)
                for label, raw in documents
                if label == intent
            )
            self._prototypes[intent] = vectors

            summed: Counter[str] = Counter()
            for vector in vectors:
                summed.update(vector)

            averaged = Counter(
                {
                    key: value / len(vectors)
                    for key, value in summed.items()
                }
            )
            self._centroids[intent] = _normalize_vector(averaged)

    def _vectorize_raw(self, raw: Counter[str]) -> Counter[str]:
        weighted = Counter(
            {
                key: value * self._idf.get(key, self._unknown_idf)
                for key, value in raw.items()
            }
        )
        return _normalize_vector(weighted)

    def vectorize(self, text: str) -> Counter[str]:
        return self._vectorize_raw(_raw_features(text))

    def classify(
        self,
        text: str,
        *,
        stage: str,
    ) -> StatisticalIntentResult:
        vector = self.vectorize(text)
        scored: list[
            tuple[float, float, float, str]
        ] = []

        for intent, prototypes in self._prototypes.items():
            prototype_similarity = max(
                (_cosine(vector, item) for item in prototypes),
                default=0.0,
            )
            centroid_similarity = _cosine(
                vector,
                self._centroids[intent],
            )
            prior = _STAGE_PRIORS.get(stage, {}).get(intent, 0.0)

            score = (
                0.62 * prototype_similarity
                + 0.28 * centroid_similarity
                + 0.10 * prior
            )
            scored.append(
                (
                    score,
                    prototype_similarity,
                    centroid_similarity,
                    intent,
                )
            )

        scored.sort(reverse=True)

        top_score, top_proto, top_centroid, top_intent = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = max(0.0, top_score - second_score)

        candidates = tuple(
            (intent, score)
            for score, _, _, intent in scored[:4]
        )

        return StatisticalIntentResult(
            intent=top_intent,
            score=top_score,
            margin=margin,
            prototype_similarity=top_proto,
            centroid_similarity=top_centroid,
            top_candidates=candidates,
        )

    @staticmethod
    def accepts(result: StatisticalIntentResult) -> bool:
        return (
            result.intent != "unknown"
            and result.score >= ACCEPT_SCORE
            and result.margin >= ACCEPT_MARGIN
        )

    @staticmethod
    def confidently_unknown(result: StatisticalIntentResult) -> bool:
        return (
            result.intent == "unknown"
            and result.score >= ACCEPT_SCORE
            and result.margin >= ACCEPT_MARGIN
        )
