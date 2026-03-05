"""Archival tests for obsolete TSMC particle filter entry points."""

import pytest

import trex.smc as smc
from trex.smc.tsmc_particle_filter import TSMCLLMParticleFilter


def test_tsmc_particle_filter_not_exported_from_smc_package():
    """`trex.smc` should not export TSMC-specific particle filters."""
    assert not hasattr(smc, "TSMCLLMParticleFilter")


def test_tsmc_particle_filter_constructor_raises_archival_error():
    """Constructing the archived class should fail with an explicit message."""
    with pytest.raises(RuntimeError, match="archived"):
        TSMCLLMParticleFilter(
            config=None,
            generator=None,
            twist_scorer=None,
            reward_model=None,
        )
