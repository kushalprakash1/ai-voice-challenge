import asyncio

import pytest

from voiceprobe.v3.asterisk_live import (
    _FluxReadinessGate,
    _observe_flux_connected_state,
)


class FakeFlux:
    def __init__(self, *, connected: bool = False) -> None:
        self._connection_established_event = asyncio.Event()
        if connected:
            self._connection_established_event.set()

    def connect(self) -> None:
        self._connection_established_event.set()


def test_flux_connect_after_waiter_registration() -> None:
    async def scenario() -> None:
        flux = FakeFlux()
        gate = _FluxReadinessGate()
        gate.mark_pipeline_started()
        waiter = asyncio.create_task(_observe_flux_connected_state(flux, gate))
        await asyncio.sleep(0)
        flux.connect()
        assert await waiter is True
        assert gate.ready.wait(0)

    asyncio.run(scenario())


def test_flux_connect_immediately_after_observer_installation() -> None:
    async def scenario() -> None:
        flux = FakeFlux()
        gate = _FluxReadinessGate()
        waiter = asyncio.create_task(_observe_flux_connected_state(flux, gate))
        flux.connect()
        gate.mark_pipeline_started()
        assert await waiter is True
        assert gate.ready.wait(0)

    asyncio.run(scenario())


def test_flux_already_connected_before_wait_begins() -> None:
    async def scenario() -> None:
        flux = FakeFlux(connected=True)
        gate = _FluxReadinessGate()
        gate.mark_pipeline_started()
        assert await _observe_flux_connected_state(flux, gate) is True
        assert gate.ready.wait(0)

    asyncio.run(scenario())


def test_flux_no_connection_still_times_out() -> None:
    async def scenario() -> None:
        flux = FakeFlux()
        gate = _FluxReadinessGate()
        gate.mark_pipeline_started()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                _observe_flux_connected_state(flux, gate), timeout=0.01
            )
        assert not gate.ready.is_set()

    asyncio.run(scenario())


def test_early_metrics_cannot_satisfy_or_poison_readiness() -> None:
    gate = _FluxReadinessGate()
    early_metrics = object()
    del early_metrics
    assert not gate.ready.is_set()
    gate.mark_pipeline_started()
    assert not gate.ready.is_set()
    assert gate.mark_flux_connected() is True
    assert gate.ready.wait(0)


def test_duplicate_connected_notification_completes_once() -> None:
    gate = _FluxReadinessGate()
    gate.mark_pipeline_started()
    assert gate.mark_flux_connected() is True
    assert gate.mark_flux_connected() is False
    assert gate.ready_observations == 1
