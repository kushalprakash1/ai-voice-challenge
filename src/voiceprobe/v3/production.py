"""Production-facing Pipecat/Flux assembly for VoiceProbe v3.

This module keeps routine reasoning deterministic while providing the narrow
adapter needed by a Pipecat task:

Flux STT events -> VoiceProbeV3Runtime -> TTSSpeakFrame -> TTS lifecycle

Pipecat imports are intentionally lazy. The repository's normal test
environment does not need Pipecat installed; the separate voiceprobe-v3
environment verifies the real integration.
"""

from __future__ import annotations

import os

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from .flow_state import FlowSnapshot
from .models import DecisionKind, PolicyDecision
from .flow_controller import SchedulingFlowController
from .runtime import (
    FallbackResolver,
    RuntimeDecision,
    VoiceProbeV3Runtime,
)
from .semantic_router import V31SemanticRouter
from .qwen_v3_fallback import (
    QwenV3FallbackRouter,
    qwen_v3_fallback_enabled_from_environment,
)
from .medication_refill import (
    MEDICATION,
    PHARMACY_PREFERENCE,
    SCENARIO_ID as MEDICATION_REFILL_SCENARIO_ID,
    MedicationRefillCorrectionScenario,
)
from .self_pay_location import (
    SCENARIO_ID as SELF_PAY_LOCATION_SCENARIO_ID,
    SelfPayLocationSwitchScenario,
)
from .doctor_specialist import (
    SCENARIO_ID as DOCTOR_SPECIALIST_SCENARIO_ID,
    DoctorDirectoryQwenRouter,
    DoctorSpecialistDirectoryScenario,
)
from .farthest_date import SCENARIO_ID as FARTHEST_DATE_SCENARIO_ID, FarthestDatePolicy
from .prerequisite import (
    PrerequisiteOverlay,
    is_high_confidence_initial_profile_prerequisite,
)
from voiceprobe.v32.runtime_fallback import (
    V32SemanticFallbackResolver,
)
from .turn_stabilizer import DEFAULT_CONTINUATION_GRACE_MS


DEFAULT_KEYTERMS = (
    "Pivot Point",
    "Alex Morgan",
    "Blue Cross",
    "new patient consultation",
)


@dataclass(frozen=True, slots=True)
class ProductionFluxConfig:
    model: str = "flux-general-en"
    sample_rate: int = 8000
    eot_threshold: float = 0.85
    eot_timeout_ms: int = 5000
    eager_eot_threshold: float | None = None
    # The remote agent can continue after an early Flux boundary. Production
    # allows more continuation time without changing the generic runtime.
    continuation_grace_ms: float = 3000.0
    keyterms: tuple[str, ...] = DEFAULT_KEYTERMS
    flux_encoding: str = "linear16"

    def validate(self) -> None:
        if self.model != "flux-general-en":
            raise ValueError("VoiceProbe v3 production model must be flux-general-en")
        if self.sample_rate != 8000:
            raise ValueError("VoiceProbe v3 production input must remain native 8 kHz")
        if self.flux_encoding != "linear16":
            raise ValueError("Deepgram Flux production encoding must be linear16")
        if not 0.0 < self.eot_threshold <= 1.0:
            raise ValueError("eot_threshold must be in (0, 1]")
        if self.eot_timeout_ms <= 0:
            raise ValueError("eot_timeout_ms must be positive")
        if self.eager_eot_threshold is not None:
            raise ValueError("EagerEndOfTurn remains disabled for the current production gate")
        if self.continuation_grace_ms < 0:
            raise ValueError("continuation_grace_ms must be non-negative")
        if not self.keyterms:
            raise ValueError("At least one production keyterm is required")


DEFAULT_PRODUCTION_FLUX_CONFIG = ProductionFluxConfig()


def safe_production_fallback_resolver(
    agent_turn: str,
    snapshot: FlowSnapshot,
) -> PolicyDecision:
    """Resolve unknown production turns without guessing or going silent.

    The deterministic fast policy still owns routine scheduling semantics.
    When that policy explicitly returns FALLBACK, production asks the remote
    side to repeat the question. CLARIFY intentionally has no flow-state
    transition, so accepted slots, booking confirmation, and scheduling
    constraints remain untouched.
    """

    del agent_turn, snapshot

    return PolicyDecision(
        kind=DecisionKind.CLARIFY,
        text="Could you please repeat that question?",
        reason="production_safe_clarification",
    )


@dataclass(frozen=True, slots=True)
class ProductionFluxBundle:
    service: Any
    config: ProductionFluxConfig


@dataclass(frozen=True, slots=True)
class ProductionPipelineBundle:
    pipeline: Any
    worker: Any
    lifecycle_processor: Any


def build_production_flux_service(
    *,
    api_key: str,
    config: ProductionFluxConfig = DEFAULT_PRODUCTION_FLUX_CONFIG,
) -> ProductionFluxBundle:
    """Instantiate the exact Deepgram Flux service used by VoiceProbe v3.

    Instantiation does not start the Pipecat pipeline or open the websocket.
    """

    if not api_key.strip():
        raise ValueError("Deepgram api_key is required")

    config.validate()

    from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService

    service = DeepgramFluxSTTService(
        api_key=api_key,
        sample_rate=config.sample_rate,
        flux_encoding=config.flux_encoding,
        settings=DeepgramFluxSTTService.Settings(
            model=config.model,
            eager_eot_threshold=config.eager_eot_threshold,
            eot_threshold=config.eot_threshold,
            eot_timeout_ms=config.eot_timeout_ms,
            keyterm=list(config.keyterms),
        ),
    )

    return ProductionFluxBundle(
        service=service,
        config=config,
    )


def _v32_semantic_enabled() -> bool:
    """Read the explicit production opt-in flag.

    Unknown values fail closed instead of accidentally selecting a
    different reasoning architecture.
    """

    raw = os.getenv(
        "VOICEPROBE_V32_SEMANTIC",
        "",
    ).strip().casefold()

    if raw in {
        "",
        "0",
        "false",
        "off",
        "no",
    }:
        return False

    if raw in {
        "1",
        "true",
        "on",
        "yes",
    }:
        return True

    raise ValueError(
        "VOICEPROBE_V32_SEMANTIC must be one of "
        "0/1, false/true, off/on, or no/yes"
    )


def _build_v32_semantic_fallback(
) -> V32SemanticFallbackResolver:
    """Construct v3.2 only after an explicit opt-in."""

    endpoint = os.getenv(
        "VOICEPROBE_V32_OLLAMA_ENDPOINT",
        "",
    ).strip()

    if not endpoint:
        raise ValueError(
            "VOICEPROBE_V32_OLLAMA_ENDPOINT is required "
            "when VOICEPROBE_V32_SEMANTIC=1"
        )

    model = os.getenv(
        "VOICEPROBE_V32_MODEL",
        "qwen3.5:4b",
    ).strip()

    if not model:
        raise ValueError(
            "VOICEPROBE_V32_MODEL must not be empty"
        )

    return V32SemanticFallbackResolver.from_ollama(
        endpoint=endpoint,
        model=model,
    )


_DEFAULT_V31_FALLBACK = object()


class PipecatRuntimeBridge:
    """Connect VoiceProbe runtime decisions to a Pipecat PipelineWorker."""

    def __init__(
        self,
        *,
        config: ProductionFluxConfig = DEFAULT_PRODUCTION_FLUX_CONFIG,
        tts_frame_factory: Callable[[str], Any] | None = None,
        fallback_resolver: FallbackResolver | object = _DEFAULT_V31_FALLBACK,
        semantic_router: V31SemanticRouter | None = None,
        flow_controller: SchedulingFlowController | None = None,
    ) -> None:
        config.validate()
        selected_scenario = os.environ.get("VOICEPROBE_SCENARIO", "").strip()
        adversarial_semantic_scenario = selected_scenario in {
            MEDICATION_REFILL_SCENARIO_ID,
            SELF_PAY_LOCATION_SCENARIO_ID,
            "self-pay-location-switch",
            DOCTOR_SPECIALIST_SCENARIO_ID,
            FARTHEST_DATE_SCENARIO_ID,
        }
        farthest_date_policy = (
            FarthestDatePolicy()
            if selected_scenario == FARTHEST_DATE_SCENARIO_ID
            else None
        )
        effective_flow = flow_controller or SchedulingFlowController(
            semantic_only=(
                adversarial_semantic_scenario
                and farthest_date_policy is None
            ),
            decision_overlay=farthest_date_policy,
        )
        self._selected_scenario = selected_scenario
        self._medication_scenario: MedicationRefillCorrectionScenario | None = None
        self._self_pay_location_scenario: SelfPayLocationSwitchScenario | None = None
        self._doctor_specialist_scenario: DoctorSpecialistDirectoryScenario | None = None
        self._farthest_date_scenario: FarthestDatePolicy | None = farthest_date_policy
        self._prerequisite_overlay: PrerequisiteOverlay | None = None

        if fallback_resolver is None:
            raise ValueError(
                "Production fallback_resolver must not be None; "
                "unresolved FALLBACK decisions must never become silence."
            )

        if fallback_resolver is _DEFAULT_V31_FALLBACK:
            if selected_scenario == FARTHEST_DATE_SCENARIO_ID:
                router = semantic_router or V31SemanticRouter(use_embeddings=True)
                resolved_fallback = router.resolve
                self._semantic_router = router
                self._v32_semantic_resolver = None
                self._qwen_v3_fallback_router = None
                self._semantic_mode = "stable_scheduler+farthest_date_policy"
            elif adversarial_semantic_scenario:
                if semantic_router is not None:
                    raise ValueError(
                        "semantic_router cannot be supplied for the medication "
                        "refill scenario."
                    )
                qwen_router = (DoctorDirectoryQwenRouter() if selected_scenario == DOCTOR_SPECIALIST_SCENARIO_ID else QwenV3FallbackRouter())
                if selected_scenario == DOCTOR_SPECIALIST_SCENARIO_ID:
                    active_scenario = DoctorSpecialistDirectoryScenario(tracker=effective_flow.tracker, qwen=qwen_router)
                    self._doctor_specialist_scenario = active_scenario
                    self._semantic_mode = "doctor_specialist_qwen_v3"
                else:
                    qwen_router.semantic_domain = (
                        "medication" if selected_scenario == MEDICATION_REFILL_SCENARIO_ID else "self_pay_location"
                    )
                if selected_scenario == MEDICATION_REFILL_SCENARIO_ID:
                    active_scenario = MedicationRefillCorrectionScenario(
                        tracker=effective_flow.tracker,
                        qwen=qwen_router,
                    )
                    self._medication_scenario = active_scenario
                    self._semantic_mode = "medication_refill_qwen_v3"
                elif selected_scenario not in {DOCTOR_SPECIALIST_SCENARIO_ID, FARTHEST_DATE_SCENARIO_ID}:
                    active_scenario = SelfPayLocationSwitchScenario(
                        tracker=effective_flow.tracker,
                        qwen=qwen_router,
                        scenario_id=selected_scenario,
                    )
                    self._self_pay_location_scenario = active_scenario
                    self._semantic_mode = "self_pay_location_qwen_v3"
                prerequisite_overlay = PrerequisiteOverlay(
                    scenario_id=selected_scenario,
                    tracker=effective_flow.tracker,
                    domain_resolver=active_scenario.resolve,
                    facts=(__import__("voiceprobe.v3.models", fromlist=["PatientFacts"]).PatientFacts(first_name="Gyeong-hyeon", last_name="Gwak") if selected_scenario == DOCTOR_SPECIALIST_SCENARIO_ID else None),
                    compound_fact_provider=(
                        lambda field: {
                            "medication": MEDICATION,
                            "pharmacy_preference": PHARMACY_PREFERENCE,
                        }.get(field)
                        if selected_scenario == MEDICATION_REFILL_SCENARIO_ID
                        else None
                    ),
                )
                self._prerequisite_overlay = prerequisite_overlay
                resolved_fallback = prerequisite_overlay.resolve
                self._semantic_router = None
                self._v32_semantic_resolver = None
                self._qwen_v3_fallback_router = qwen_router
            elif _v32_semantic_enabled():
                if semantic_router is not None:
                    raise ValueError(
                        "semantic_router cannot be supplied when "
                        "VOICEPROBE_V32_SEMANTIC is enabled."
                    )

                v32_resolver = (
                    _build_v32_semantic_fallback()
                )

                resolved_fallback: FallbackResolver = (
                    v32_resolver
                )

                self._semantic_router = None
                self._v32_semantic_resolver = v32_resolver
                self._semantic_mode = "v32"
            elif (
                semantic_router is None
                and qwen_v3_fallback_enabled_from_environment()
            ):
                qwen_router = QwenV3FallbackRouter()
                resolved_fallback = qwen_router.resolve

                self._semantic_router = None
                self._v32_semantic_resolver = None
                self._qwen_v3_fallback_router = qwen_router
                self._semantic_mode = "qwen_v3"
            else:
                router = (
                    semantic_router
                    or V31SemanticRouter(
                        use_embeddings=True,
                    )
                )

                resolved_fallback = router.resolve

                self._semantic_router: (
                    V31SemanticRouter | None
                ) = router

                self._v32_semantic_resolver = None
                self._qwen_v3_fallback_router = None
                self._semantic_mode = "v31"
        else:
            if semantic_router is not None:
                raise ValueError(
                    "semantic_router cannot be combined with a custom "
                    "fallback_resolver."
                )

            resolved_fallback = fallback_resolver
            self._semantic_router = None
            self._v32_semantic_resolver = None
            self._semantic_mode = "custom"

        if not hasattr(self, "_qwen_v3_fallback_router"):
            self._qwen_v3_fallback_router = None

        self._config = config
        self._frame_sink: Any | None = None
        self._queued_speech_count = 0
        self._pending_spoken_decision: PolicyDecision | None = None
        self._tts_frame_factory = tts_frame_factory or _default_tts_frame_factory

        self._runtime = VoiceProbeV3Runtime(
            flow_controller=effective_flow,
            fallback_resolver=resolved_fallback,
            on_decision=self._on_runtime_decision,
            continuation_grace_ms=config.continuation_grace_ms,
            fast_stabilization_predicate=(
                self._initial_prerequisite_fast_path
                if self._prerequisite_overlay is not None
                else None
            ),
        )

    def _initial_prerequisite_fast_path(self, text: str) -> bool:
        overlay = self._prerequisite_overlay
        return bool(
            overlay is not None
            and not overlay.state.profile_consent_spoken
            and not overlay.state.identity_fields_spoken
            and is_high_confidence_initial_profile_prerequisite(text)
        )

    @property
    def semantic_mode(self) -> str:
        """Active production fallback architecture."""

        return self._semantic_mode

    @property
    def scenario_metadata(self) -> dict[str, object]:
        """Existing bridge metadata for artifacts and decision-event evidence."""

        scenario = self._medication_scenario or self._self_pay_location_scenario or self._doctor_specialist_scenario or self._farthest_date_scenario
        if scenario is None:
            return {"scenario": self._selected_scenario or None}
        metadata = scenario.metadata()
        if self._prerequisite_overlay is not None:
            metadata.update(self._prerequisite_overlay.metadata())
        if self._medication_scenario is not None:
            metadata["medication_stage"] = scenario.last_semantic_action
            metadata["oracle_candidate"] = (
                self._medication_scenario.oracle.correction_retention_failure
                or self._medication_scenario.oracle.medication_state_persistence_failure
            )
        return metadata

    @property
    def objective_complete(self) -> bool:
        scenario = self._medication_scenario or self._self_pay_location_scenario or self._doctor_specialist_scenario or self._farthest_date_scenario
        if scenario is not None:
            return scenario.objective_complete
        return self._runtime.flow_controller.tracker.snapshot().complete

    @property
    def scenario_terminal(self) -> bool:
        """Whether a scenario has a grounded terminal result, successful or not."""
        if self._medication_scenario is not None:
            return self._medication_scenario.scenario_terminal
        if self._self_pay_location_scenario is not None:
            return self._self_pay_location_scenario.objective_complete
        if self._doctor_specialist_scenario is not None:
            return self._doctor_specialist_scenario.objective_complete
        if self._farthest_date_scenario is not None:
            return self._farthest_date_scenario.objective_complete
        return self.objective_complete

    @property
    def v32_semantic_resolver(
        self,
    ) -> V32SemanticFallbackResolver | None:
        return self._v32_semantic_resolver

    @property
    def config(self) -> ProductionFluxConfig:
        return self._config

    @property
    def runtime(self) -> VoiceProbeV3Runtime:
        return self._runtime

    @property
    def queued_speech_count(self) -> int:
        return self._queued_speech_count

    @property
    def frame_sink_bound(self) -> bool:
        return self._frame_sink is not None

    @property
    def worker_bound(self) -> bool:
        return self.frame_sink_bound

    def bind_frame_sink(self, sink: Any) -> None:
        if not hasattr(sink, "queue_frames"):
            raise TypeError("Pipecat frame sink must provide queue_frames(frames)")
        if self._frame_sink is not None and self._frame_sink is not sink:
            raise RuntimeError(
                "PipecatRuntimeBridge is already bound to another frame sink"
            )
        self._frame_sink = sink

    def bind_worker(self, worker: Any) -> None:
        self.bind_frame_sink(worker)

    def attach_flux(self, stt_service: Any) -> None:
        self._runtime.attach_flux(stt_service)

    async def _on_runtime_decision(
        self,
        result: RuntimeDecision,
    ) -> None:
        if not result.response_ready:
            return

        if self._frame_sink is None:
            raise RuntimeError(
                "A response became ready before a Pipecat frame sink was bound"
            )

        # Busy begins before TTS synthesis is queued, so any remote speech
        # arriving during synthesis/playback enters the existing burst buffer.
        self._runtime.mark_response_started()

        frame = self._tts_frame_factory(result.decision.text)
        maybe = self._frame_sink.queue_frames([frame])

        if inspect.isawaitable(maybe):
            await maybe

        self._queued_speech_count += 1
        self._pending_spoken_decision = result.decision

    async def on_tts_stopped(self) -> None:
        """Release response-busy state only after Pipecat reports TTS stopped."""
        scenario = self._medication_scenario or self._self_pay_location_scenario or self._doctor_specialist_scenario or self._farthest_date_scenario
        if scenario is not None and self._pending_spoken_decision is not None:
            scenario.mark_decision_spoken(self._pending_spoken_decision)
        if self._prerequisite_overlay is not None and self._pending_spoken_decision is not None:
            self._prerequisite_overlay.mark_decision_spoken(self._pending_spoken_decision)
        self._pending_spoken_decision = None
        await self._runtime.mark_response_finished()

    async def on_tts_suppressed(self) -> None:
        """Re-coalesce an unsent candidate with the remote continuation."""
        scenario = self._medication_scenario or self._self_pay_location_scenario or self._doctor_specialist_scenario or self._farthest_date_scenario
        if scenario is not None and self._pending_spoken_decision is not None:
            scenario.mark_decision_suppressed(self._pending_spoken_decision)
        self._pending_spoken_decision = None
        await self._runtime.mark_response_suppressed()

    def clear_pending(self) -> tuple[str, ...]:
        return self._runtime.ingress.clear_pending()


def _default_tts_frame_factory(text: str) -> Any:
    from pipecat.frames.frames import TTSSpeakFrame

    return TTSSpeakFrame(
        text=text,
        append_to_context=False,
    )


def build_tts_lifecycle_processor(
    bridge: PipecatRuntimeBridge,
) -> Any:
    """Create a Pipecat processor that releases busy state on TTSStoppedFrame."""

    from pipecat.frames.frames import TTSStoppedFrame
    from pipecat.processors.frame_processor import (
        FrameDirection,
        FrameProcessor,
    )

    class VoiceProbeTTSLifecycleProcessor(FrameProcessor):
        def __init__(self) -> None:
            super().__init__(name="VoiceProbeTTSLifecycleProcessor")

        async def process_frame(
            self,
            frame: Any,
            direction: FrameDirection,
        ) -> None:
            await super().process_frame(frame, direction)

            if isinstance(frame, TTSStoppedFrame):
                await bridge.on_tts_stopped()

            await self.push_frame(frame, direction)

    return VoiceProbeTTSLifecycleProcessor()


def build_production_pipeline_worker(
    *,
    transport: Any,
    stt_service: Any,
    tts_service: Any,
    bridge: PipecatRuntimeBridge,
    enable_metrics: bool = True,
    enable_usage_metrics: bool = True,
) -> ProductionPipelineBundle:
    """Build the minimal deterministic Pipecat pipeline worker for VoiceProbe v3."""

    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker

    lifecycle = build_tts_lifecycle_processor(bridge)

    pipeline = Pipeline(
        [
            transport.input(),
            stt_service,
            tts_service,
            lifecycle,
            transport.output(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=bridge.config.sample_rate,
            audio_out_sample_rate=bridge.config.sample_rate,
            enable_metrics=enable_metrics,
            enable_usage_metrics=enable_usage_metrics,
        ),
    )

    bridge.bind_worker(worker)
    bridge.attach_flux(stt_service)

    return ProductionPipelineBundle(
        pipeline=pipeline,
        worker=worker,
        lifecycle_processor=lifecycle,
    )
