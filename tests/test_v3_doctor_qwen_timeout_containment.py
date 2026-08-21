import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from voiceprobe.v3.doctor_specialist import (
    DEFAULT_DOCTOR_QWEN_TIMEOUT_SECONDS,
    DoctorDirectoryQwenRouter,
    DoctorSpecialistDirectoryScenario,
)
from voiceprobe.v3.flow_state import SchedulingFlowTracker
from voiceprobe.v3.ingress import FluxIngressController
from voiceprobe.v3.models import DecisionKind


def observation(action: str = "profile_registered") -> dict[str, object]:
    return {
        "requires_response": True,
        "doctor_action": action,
        "reported_profile_name": "",
        "reported_profile_spelling": "",
        "doctors": [],
        "doctor_name": "",
        "specialty": "",
        "explicit_gender": "",
        "locations": [],
        "hours": "",
        "hours_location": "",
        "day": "",
    }


class DelayedBackend:
    def __init__(self, delay: float, *, error: BaseException | None = None) -> None:
        self.delay = delay
        self.error = error

    async def generate_json(self, **_kwargs):
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return observation()


class BoundedBackend(DelayedBackend):
    def __init__(self, delay: float, timeout: float) -> None:
        super().__init__(delay)
        self.timeout = timeout

    async def generate_json(self, **kwargs):
        return await asyncio.wait_for(super().generate_json(**kwargs), self.timeout)


def scenario_for(backend):
    tracker = SchedulingFlowTracker()
    router = DoctorDirectoryQwenRouter(backend=backend)
    return tracker, DoctorSpecialistDirectoryScenario(tracker=tracker, qwen=router)


@pytest.mark.asyncio
async def test_qwen_below_and_near_timeout_complete() -> None:
    assert DEFAULT_DOCTOR_QWEN_TIMEOUT_SECONDS == 6.0
    for delay in (0.001, 0.04):
        tracker, scenario = scenario_for(BoundedBackend(delay, timeout=0.05))
        decision = await scenario.resolve("Your profile is registered.", tracker.snapshot())
        assert decision.kind is not DecisionKind.CLARIFY
        assert scenario.profile_registered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend",
    [
        BoundedBackend(0.05, timeout=0.005),
        DelayedBackend(0, error=TimeoutError("timed out")),
        DelayedBackend(0, error=ConnectionError("unreachable")),
    ],
)
async def test_semantic_backend_failure_is_safe_and_does_not_mutate(backend) -> None:
    tracker, scenario = scenario_for(backend)
    before = tracker.snapshot()
    decision = await scenario.resolve("Your profile is registered.", before)
    assert decision.kind is DecisionKind.CLARIFY
    assert decision.text == "Could you please repeat that question?"
    assert scenario.turn == 0
    assert not scenario.profile_registered
    assert tracker.snapshot() == before
    assert scenario.semantic_failures[-1]["event"] == "doctor_directory_semantic_failure"


@pytest.mark.asyncio
async def test_cancelled_to_thread_style_result_cannot_commit_stale_state() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)

    class ToThreadBackend:
        async def generate_json(self, **_kwargs):
            def sync_request():
                started.set()
                release.wait(timeout=1)
                finished.set()
                return observation()

            return await asyncio.get_running_loop().run_in_executor(executor, sync_request)

    tracker, scenario = scenario_for(ToThreadBackend())
    task = asyncio.create_task(scenario.resolve("Your profile is registered.", tracker.snapshot()))
    for _ in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.001)
    assert started.is_set()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(100):
        if finished.is_set():
            break
        await asyncio.sleep(0.001)
    assert finished.is_set()
    assert scenario.turn == 0
    assert not scenario.profile_registered
    executor.shutdown(wait=True)


def test_production_warmup_failure_aborts_router_construction(monkeypatch) -> None:
    class FailingBackend:
        def __init__(self, _config):
            pass

        async def generate_json(self, **_kwargs):
            raise ConnectionError("ollama unavailable")

    monkeypatch.setattr("voiceprobe.v3.doctor_specialist.OllamaBackend", FailingBackend)
    with pytest.raises(ConnectionError, match="ollama unavailable"):
        DoctorDirectoryQwenRouter()


@pytest.mark.asyncio
async def test_duplicate_flush_serializes_and_commits_once() -> None:
    active = commits = 0

    async def sink(_result):
        nonlocal active, commits
        active += 1
        assert active == 1
        await asyncio.sleep(0.01)
        commits += 1
        active -= 1

    controller = FluxIngressController(on_decision=sink, continuation_grace_ms=60_000)
    controller._stabilized_pending.append("Your profile is registered.")
    first, duplicate = await asyncio.gather(
        controller.flush_stabilized_pending(),
        controller.flush_stabilized_pending(),
    )
    assert first is not None
    assert duplicate is None
    assert commits == 1


@pytest.mark.asyncio
async def test_repeated_delayed_failures_are_retrieved_and_next_turn_recovers() -> None:
    calls = 0

    async def sink(_result):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise TimeoutError("semantic timeout")

    controller = FluxIngressController(on_decision=sink, continuation_grace_ms=1)
    for text in ("first", "second", "third"):
        controller._stabilized_pending.append(text)
        task = asyncio.create_task(controller._delayed_stabilization_flush())
        await task
    assert calls == 3
    assert len(controller._background_failures) == 2
