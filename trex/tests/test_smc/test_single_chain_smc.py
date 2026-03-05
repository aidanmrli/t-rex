"""
Unit tests for the Stage 3 single-chain SMC core implementation.
"""

import numpy as np
import pytest

from trex.smc.single_chain_smc import SingleChainSMC


def test_beta_zero_keeps_uniform_weights_and_zero_log_z_increment():
    smc = SingleChainSMC(n_particles=4, beta=0.0, seed=11)
    smc.initialize(np.zeros(4, dtype=float))

    result = smc.step(np.array([2.0, -1.0, 0.5, 3.0], dtype=float))

    np.testing.assert_allclose(smc.normalized_weights, np.full(4, 0.25))
    np.testing.assert_allclose(smc.log_reward_prev, np.array([2.0, -1.0, 0.5, 3.0], dtype=float))
    assert result.resampled is False
    assert smc.log_z == pytest.approx(0.0)


def test_beta_one_updates_weights_without_resampling():
    smc = SingleChainSMC(n_particles=3, beta=1.0, ess_threshold=0.0, seed=7)
    smc.initialize(np.zeros(3, dtype=float))

    result = smc.step(np.log(np.array([1.0, 2.0, 4.0], dtype=float)))
    expected = np.array([1.0, 2.0, 4.0], dtype=float)
    expected /= expected.sum()

    np.testing.assert_allclose(smc.normalized_weights, expected)
    assert result.resampled is False
    assert smc.log_z == pytest.approx(np.log(7.0 / 3.0))


def test_ess_triggered_systematic_resampling_resets_weights_and_is_seeded():
    log_reward_curr = np.array([-8.0, -8.0, -8.0, 0.0], dtype=float)
    particles = ["a", "b", "c", "d"]

    smc_a = SingleChainSMC(n_particles=4, beta=1.0, ess_threshold=3.5, seed=1234)
    smc_b = SingleChainSMC(n_particles=4, beta=1.0, ess_threshold=3.5, seed=1234)
    smc_a.initialize(np.zeros(4, dtype=float))
    smc_b.initialize(np.zeros(4, dtype=float))

    result_a = smc_a.step(log_reward_curr, particles=particles)
    result_b = smc_b.step(log_reward_curr, particles=particles)

    assert result_a.resampled is True
    assert result_a.ess < 3.5
    np.testing.assert_array_equal(result_a.ancestors, result_b.ancestors)
    assert result_a.particles == result_b.particles
    np.testing.assert_allclose(smc_a.normalized_weights, np.full(4, 0.25))
    np.testing.assert_allclose(smc_a.log_reward_prev, log_reward_curr[result_a.ancestors])


def test_log_z_updates_incrementally_and_stays_finite():
    smc = SingleChainSMC(n_particles=2, beta=1.0, ess_threshold=0.0, seed=99)
    smc.initialize(np.zeros(2, dtype=float))

    first = smc.step(np.log(np.array([2.0, 4.0], dtype=float)))
    second = smc.step(np.log(np.array([8.0, 4.0], dtype=float)))
    diagnostics = smc.diagnostics()

    assert np.isfinite(smc.log_z)
    assert first.log_z == pytest.approx(np.log(3.0))
    assert second.log_z == pytest.approx(np.log(6.0))
    assert smc.log_z == pytest.approx(np.log(6.0))
    assert diagnostics["num_steps"] == 2
    np.testing.assert_allclose(diagnostics["log_z_history"], np.array([np.log(3.0), np.log(6.0)]))
