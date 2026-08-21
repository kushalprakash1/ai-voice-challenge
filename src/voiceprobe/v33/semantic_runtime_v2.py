"""Feature-flagged persistent SemanticLab-v2 reasoner for VoiceProbe v3.3.

This module is intentionally isolated from the legacy OllamaV33Reasoner.
It loads the exact frozen SemanticLab proof stack packaged under
``_semanticlab_frozen`` and adapts it to the existing v3.3 Reasoner contract.

Runtime contract:
    remote-only context
      -> persistent SemanticLab-v2 stack
      -> RemoteObservation
      -> existing StrategicActionGenerator
      -> existing planner / validator / narrative director / verbalizer

The heavy semantic engine is initialized lazily and inference is executed in a
worker thread so synchronous transformer inference does not block the asyncio
telephony loop.

The package remains feature flagged by ``build_v33_reasoner`` in reasoner.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .mind import AgentMind
from .world_model import RemoteObservation


FEATURE_FLAG = "VOICEPROBE_V33_LEVEL2_RUNTIME_CANDIDATE"

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
_FROZEN_DIR = _PACKAGE_DIR / "_semanticlab_frozen"
_MANIFEST_FILE = _FROZEN_DIR / "manifest.json"

# Root sources/checkpoints intentionally remain where the proven SemanticLab
# stack expects them. The frozen package supplies only the post-architecture
# and proof-composition sources that previously lived outside the repository.
_REQUIRED_ROOT_PATHS = (
    "voiceprobe_semanticlab_v2_phase6b_distilbert_contrastive.py",
    "voiceprobe_semanticlab_v2_phase7c_targeted_gate.py",
    "voiceprobe_semanticlab_v2_phase7d_reference_kind.py",
    "voiceprobe_semanticlab_v2_phase7h_selection_operator.py",
    "voiceprobe_semanticlab_v2_phase7j_ambiguity_detail_fixed.py",
    "voiceprobe_semanticlab_v2_phase7j_candidate_resolver_audit.py",
    "voiceprobe_semanticlab_v2_phase8a_speech_act_topic.py",
    "voiceprobe_semanticlab_v2_phase8b_requested_fact.py",
    "voiceprobe_semanticlab_v2_phase8c_record_claims.py",
    "voiceprobe_semanticlab_v2_phase8d_transactions.py",
    "voiceprobe_semanticlab_v2_phase8d_predicate_normalizer_audit.py",
    "voiceprobe_semanticlab_v2_offered_options_audit.py",
    "artifacts/semanticlab_v2_phase6b_distilbert_scheduling.pt",
    "artifacts/semanticlab_v2_phase7c_factorized_gate.pt",
    "artifacts/semanticlab_v2_phase7d_reference_kind.pt",
    "artifacts/semanticlab_v2_phase7i_selection_operator.pt",
    "artifacts/semanticlab_v2_phase7j_ambiguity_detail.pt",
    "artifacts/semanticlab_v2_phase8a3_speech_act_topic.pt",
    "artifacts/semanticlab_v2_phase8b1_requested_fact.pt",
    "artifacts/semanticlab_v2_phase8c_record_claims.pt",
    "artifacts/candidates/semanticlab_v2_phase8a_hierarchical_multiview.pt",
)

_ENGINE_BASENAME = (
    "voiceprobe_semanticlab_v2_persistent_fullstack_runtime_replay_gate_v1_1.py"
)
_CACHE_GATE_BASENAME = (
    "voiceprobe_semanticlab_v2_persistent_cache_offline_gate_v1_1.py"
)
_RUNTIME_GATE_BASENAME = (
    "voiceprobe_semanticlab_v2_runtime_wiring_offline_gate_v2.py"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import frozen semantic module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_frozen(basename: str, *_unused) -> Path:
    path = (_FROZEN_DIR / str(basename)).resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing frozen semantic asset: {path}")
    return path


def _verify_frozen_assets() -> None:
    if not _MANIFEST_FILE.is_file():
        raise RuntimeError(f"Missing SemanticLab runtime manifest: {_MANIFEST_FILE}")

    manifest = json.loads(_MANIFEST_FILE.read_text(encoding="utf-8"))
    expected = dict(manifest.get("sha256") or {})
    if not expected:
        raise RuntimeError("SemanticLab frozen manifest contains no hashes.")

    unexpected = sorted(
        p.name
        for p in _FROZEN_DIR.iterdir()
        if p.is_file() and p.name != "manifest.json" and p.name not in expected
    )
    if unexpected:
        raise RuntimeError(
            "Unexpected files in frozen SemanticLab package: "
            + ", ".join(unexpected)
        )

    for basename, wanted in expected.items():
        path = _FROZEN_DIR / basename
        if not path.is_file():
            raise RuntimeError(f"Missing frozen SemanticLab asset: {basename}")
        actual = _sha256(path)
        if actual != wanted:
            raise RuntimeError(
                "Frozen SemanticLab asset drift: "
                f"{basename} expected={wanted} actual={actual}"
            )


def _verify_root_dependencies() -> None:
    missing = [
        rel for rel in _REQUIRED_ROOT_PATHS
        if not (_REPO_ROOT / rel).is_file()
    ]
    if missing:
        raise RuntimeError(
            "SemanticLab runtime root dependencies are missing: "
            + ", ".join(missing)
        )


class SemanticLabV2Reasoner:
    """Persistent local SemanticLab-v2 implementation of the v3.3 Reasoner API."""

    _shared: "SemanticLabV2Reasoner | None" = None
    _shared_guard = threading.Lock()

    def __init__(self) -> None:
        self._service = None
        self._runtime_gate = None
        self._engine = None
        self._inference_lock = threading.Lock()
        self._init_lock = threading.Lock()
        self.last_timing: dict[str, float] = {}

    @classmethod
    def shared(cls) -> "SemanticLabV2Reasoner":
        with cls._shared_guard:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    def warmup_sync(self) -> None:
        """Initialize the persistent SemanticLab stack before a live call starts."""
        self._ensure_service()

    async def warmup(self) -> None:
        # Model construction is intentionally moved off the asyncio event loop.
        await asyncio.to_thread(self._ensure_service)

    async def propose(
        self,
        *,
        mind: AgentMind,
        remote_turn: str,
    ) -> tuple[RemoteObservation, tuple[Any, ...]]:
        return await asyncio.to_thread(
            self._propose_sync,
            mind,
            remote_turn,
        )

    def _propose_sync(
        self,
        mind: AgentMind,
        remote_turn: str,
    ) -> tuple[RemoteObservation, tuple[Any, ...]]:
        self._ensure_service()
        assert self._service is not None
        assert self._runtime_gate is not None

        context = self._runtime_gate.context_from_mind(mind)

        # The persistent transformer stack is one shared mutable inference
        # lifecycle. Serialize calls even if a future caller raises concurrency.
        with self._inference_lock:
            result = self._service.interpret_one(
                context=context,
                utterance=" ".join(str(remote_turn).split()),
                mind=mind,
            )

        self.last_timing = {
            "semantic_ms": float(result.get("semantic_ms", 0.0)),
            "planner_candidate_ms": float(result.get("planner_ms", 0.0)),
            "total_ms": float(result.get("total_ms", 0.0)),
            **{
                str(k): float(v)
                for k, v in dict(result.get("component_timing") or {}).items()
            },
        }

        return result["observation"], tuple(result["plans"])

    def interpret_frame_sync(
        self,
        *,
        remote_turn: str,
        recent_history: tuple[str, ...] = (),
    ) -> tuple[Any, bool]:
        """Run only the frozen semantic stack and return final frame + OOS.

        This is the synchronous semantic-perception boundary used by the
        Reasoning-v2 bridge. It deliberately does not construct a v3.3
        ``AgentMind`` and does not invoke the strategic action generator, so
        patient truth/preferences cannot enter semantic perception.
        """

        normalized_turn = " ".join(str(remote_turn).split())
        if not normalized_turn:
            raise ValueError("remote_turn cannot be blank.")

        context = tuple(
            " ".join(str(item).split())
            for item in recent_history[-4:]
            if str(item).strip()
        )

        self._ensure_service()
        assert self._service is not None

        row = SimpleNamespace(
            context=context,
            utterance=normalized_turn,
        )

        with self._inference_lock:
            result = self._service._compose_final_frames([row])

        component = {
            str(k): float(v)
            for k, v in dict(result.get("timing") or {}).items()
        }
        self.last_timing = {
            "semantic_ms": float(sum(component.values())),
            **component,
        }

        return result["frames"][0], bool(result["oos"][0])

    def _ensure_service(self) -> None:
        if self._service is not None:
            return

        with self._init_lock:
            if self._service is not None:
                return

            _verify_frozen_assets()
            _verify_root_dependencies()

            # The proven scripts use Path(".") as their repository root.
            if Path.cwd().resolve() != _REPO_ROOT:
                raise RuntimeError(
                    "SemanticLab runtime must be initialized from repository root "
                    f"{_REPO_ROOT}; current working directory is {Path.cwd().resolve()}"
                )

            engine = _load_module(
                "voiceprobe_v33_semantic_runtime_engine",
                _resolve_frozen(_ENGINE_BASENAME),
            )
            cg = _load_module(
                "voiceprobe_v33_semantic_cache_gate",
                _resolve_frozen(_CACHE_GATE_BASENAME),
            )
            rt = _load_module(
                "voiceprobe_v33_semantic_runtime_gate",
                _resolve_frozen(_RUNTIME_GATE_BASENAME),
            )

            # Force every proof-composition resolver to the bundled frozen
            # package instead of Downloads or arbitrary current directories.
            engine.resolve_named = _resolve_frozen
            cg.resolve_named = _resolve_frozen
            rt.resolve_named = _resolve_frozen

            combined = _load_module(
                "voiceprobe_v33_semantic_combined",
                _resolve_frozen(rt.COMBINED_SHADOW_BASENAME),
            )
            combined.resolve_named = _resolve_frozen

            ownership = _load_module(
                "voiceprobe_v33_semantic_ownership",
                _resolve_frozen(rt.OWNERSHIP_PROOF_BASENAME),
            )
            presence_precedence = _load_module(
                "voiceprobe_v33_semantic_presence_precedence",
                _resolve_frozen(engine.PRESENCE_PRECEDENCE_PROOF_BASENAME),
            )

            full = _load_module(
                "voiceprobe_v33_semantic_full_eval",
                _resolve_frozen(combined.FULL_BASENAME),
            )
            v132 = _load_module(
                "voiceprobe_v33_semantic_v132",
                _resolve_frozen(rt.V13_2_BASENAME),
            )

            v7 = _load_module(
                "voiceprobe_v33_semantic_v7",
                _resolve_frozen(combined.V7_BASENAME),
            )
            v7.resolve_named = _resolve_frozen

            sep = _load_module(
                "voiceprobe_v33_semantic_sep",
                _resolve_frozen(v7.SEP_BASENAME),
            )
            sep.resolve_named = _resolve_frozen

            loc = _load_module(
                "voiceprobe_v33_semantic_loc",
                _resolve_frozen(sep.LOCALIZER_BASENAME),
            )
            loc.resolve_named = _resolve_frozen

            app = _load_module(
                "voiceprobe_v33_semantic_app",
                _resolve_frozen(loc.APP_BASENAME),
            )
            app.resolve_named = _resolve_frozen

            comp = _load_module(
                "voiceprobe_v33_semantic_comp",
                _resolve_frozen(app.COMP_BASENAME),
            )
            comp.resolve_named = _resolve_frozen

            v52 = _load_module(
                "voiceprobe_v33_semantic_v52",
                _resolve_frozen(comp.V52_BASENAME),
            )
            feas = _load_module(
                "voiceprobe_v33_semantic_feas",
                _resolve_frozen(comp.FEAS_BASENAME),
            )
            ref_v2 = _load_module(
                "voiceprobe_v33_semantic_ref_v2",
                _resolve_frozen(combined.REF_V2_BASENAME),
            )
            ref_boundary = _load_module(
                "voiceprobe_v33_semantic_ref_boundary",
                _resolve_frozen(combined.REF_BOUNDARY_BASENAME),
            )

            # Also pin nested resolution hooks before the service initializes.
            for module in (v52, feas, ref_v2, ref_boundary):
                if hasattr(module, "resolve_named"):
                    module.resolve_named = _resolve_frozen

            service = engine.PersistentFullSemanticRuntime(
                cg=cg,
                rt=rt,
                full=full,
                v132=v132,
                combined=combined,
                ownership=ownership,
                presence_precedence=presence_precedence,
                v7=v7,
                sep=sep,
                loc=loc,
                app=app,
                comp=comp,
                v52=v52,
                feas=feas,
                ref_v2=ref_v2,
                ref_boundary=ref_boundary,
            )

            self._engine = engine
            self._runtime_gate = rt
            self._service = service


def feature_enabled() -> bool:
    return os.environ.get(FEATURE_FLAG) == "1"
