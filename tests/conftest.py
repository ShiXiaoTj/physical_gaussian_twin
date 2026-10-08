"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from pgtwin.real_data.synthetic import SyntheticEpisodeGenerator, SyntheticPushConfig


@pytest.fixture(scope="session")
def synthetic_push():
    """Noise-free synthetic push episode with ground-truth friction 0.8.

    Session-scoped: generation rolls out the mock simulator once.
    """
    gen = SyntheticEpisodeGenerator(
        SyntheticPushConfig(
            episode_id="test_push",
            true_friction=0.8,
            noise_std_pos=0.0,
            hold_time=1.5,
            sim_dt=0.002,
        )
    )
    episode, obj = gen.generate()
    return episode, obj


@pytest.fixture
def rng():
    return np.random.default_rng(42)
