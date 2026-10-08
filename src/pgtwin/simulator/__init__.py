"""Simulator abstraction layer and adapters."""

from pgtwin.simulator.base_simulator import BaseSimulator, SimulatorConfig
from pgtwin.simulator.mock_adapter import MockSimulator
from pgtwin.simulator.replay import PushReplayController, ReplayResult


def create_simulator(cfg: dict) -> BaseSimulator:
    """Factory from a ``simulation:`` config section (graceful fallback).

    ``backend: mock | mujoco | isaac``.  Unavailable backends raise with an
    actionable message instead of failing at import time.
    """
    backend = cfg.get("backend", "mock")
    config = SimulatorConfig.from_config(cfg)
    if backend == "mock":
        return MockSimulator(config)
    if backend == "mujoco":
        from pgtwin.simulator.mujoco_adapter import MuJoCoSimulator

        return MuJoCoSimulator(config)
    if backend == "isaac":
        raise NotImplementedError(
            "Isaac Sim adapter is a Phase 3+ target; use backend 'mock' or 'mujoco'."
        )
    raise ValueError(f"unknown simulator backend '{backend}'")


__all__ = [
    "BaseSimulator",
    "SimulatorConfig",
    "MockSimulator",
    "PushReplayController",
    "ReplayResult",
    "create_simulator",
]
