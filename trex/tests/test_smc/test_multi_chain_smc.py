"""
Unit tests for Stage 4 multi-chain SMC with hot-chain mixture proposals.
"""

import numpy as np

from trex.smc.multi_chain_smc import MultiChainSMC, MultiChainSMCConfig


def test_no_injection_behavior():
    """With lambda=0, colder chains should only use local extensions."""

    def propagate_fn(state, chain_index, particle_index, step, rng):
        del rng
        new_state = list(state)
        new_state.append((f"chain_{chain_index}", step, particle_index))
        return new_state

    def log_reward_fn(state, chain_index, particle_index, step):
        del chain_index, particle_index, step
        return -0.1 * len(state)

    config = MultiChainSMCConfig(
        betas=[0.0, 1.0],
        num_particles=4,
        num_steps=2,
        injection_probability=0.0,
        ess_threshold=0.0,
        seed=7,
    )
    smc = MultiChainSMC(config=config, propagate_fn=propagate_fn, log_reward_fn=log_reward_fn)
    result = smc.run(initial_state=[])

    cold_particles = result.particles[1]
    assert all(token[0] == "chain_1" for p in cold_particles for token in p)

    for step_diag in result.step_diagnostics:
        cold_diag = step_diag[1]
        assert cold_diag["attempted_injections"] == 0
        assert cold_diag["survived_injections"] == 0
        assert not np.any(cold_diag["injection_mask"])


def test_forced_injection_copies_full_trajectory():
    """With lambda=1, injected particles should exactly match hot-chain trajectories."""

    def propagate_fn(state, chain_index, particle_index, step, rng):
        del step, rng
        new_state = list(state)
        if chain_index == 0:
            new_state.extend([("hot", particle_index, "a"), ("hot", particle_index, "b")])
        else:
            new_state.append(("cold", particle_index, "x"))
        return new_state

    def log_reward_fn(state, chain_index, particle_index, step):
        del state, chain_index, particle_index, step
        return 0.0

    config = MultiChainSMCConfig(
        betas=[0.0, 1.0],
        num_particles=5,
        num_steps=1,
        injection_probability=1.0,
        ess_threshold=0.0,
        seed=1,
    )
    smc = MultiChainSMC(config=config, propagate_fn=propagate_fn, log_reward_fn=log_reward_fn)
    result = smc.run(initial_state=[])

    hot_trajectories = {tuple(p) for p in result.particles[0]}
    for cold_particle in result.particles[1]:
        assert tuple(cold_particle) in hot_trajectories
        assert len(cold_particle) == 2
        assert all(token[0] == "hot" for token in cold_particle)

    cold_diag = result.step_diagnostics[0][1]
    assert cold_diag["attempted_injections"] == 5


def test_injection_weight_formula():
    """Injected weights use (beta_k - beta_{k-1}) * log_R_curr in log space."""

    def propagate_fn(state, chain_index, particle_index, step, rng):
        del step, rng
        new_state = list(state)
        if chain_index == 0:
            new_state.append(("hot", particle_index))
        else:
            new_state.append(("cold", particle_index))
        return new_state

    def log_reward_fn(state, chain_index, particle_index, step):
        del chain_index, particle_index, step
        kind, idx = state[-1]
        if kind == "hot":
            return -0.5 * (idx + 1)
        return -10.0

    beta_hot = 0.0
    beta_cold = 1.0
    config = MultiChainSMCConfig(
        betas=[beta_hot, beta_cold],
        num_particles=4,
        num_steps=1,
        injection_probability=1.0,
        ess_threshold=0.0,
        seed=3,
    )
    smc = MultiChainSMC(config=config, propagate_fn=propagate_fn, log_reward_fn=log_reward_fn)
    result = smc.run(initial_state=[])

    cold_diag = result.step_diagnostics[0][1]
    assert np.all(cold_diag["injection_mask"])
    expected = (beta_cold - beta_hot) * cold_diag["log_rewards_curr"]
    np.testing.assert_allclose(cold_diag["log_incremental_weights"], expected, rtol=0.0, atol=1e-12)


def test_per_chain_ess_and_resampling():
    """Each chain should apply ESS-triggered systematic resampling independently."""

    def propagate_fn(state, chain_index, particle_index, step, rng):
        del step, rng
        new_state = list(state)
        new_state.append((chain_index, particle_index))
        return new_state

    def log_reward_fn(state, chain_index, particle_index, step):
        del chain_index, particle_index, step
        chain_idx, particle_idx = state[-1]
        if chain_idx == 0:
            return -5.0
        if particle_idx == 0:
            return 0.0
        return -12.0

    config = MultiChainSMCConfig(
        betas=[0.0, 1.0],
        num_particles=4,
        num_steps=1,
        injection_probability=0.0,
        ess_threshold=0.75,
        seed=9,
    )
    smc = MultiChainSMC(config=config, propagate_fn=propagate_fn, log_reward_fn=log_reward_fn)
    result = smc.run(initial_state=[])

    hot_diag = result.step_diagnostics[0][0]
    cold_diag = result.step_diagnostics[0][1]

    assert np.isclose(hot_diag["ess"], 4.0)
    assert hot_diag["resampled"] is False
    assert cold_diag["ess"] < 3.0
    assert cold_diag["resampled"] is True
    np.testing.assert_allclose(result.weights[1], np.full(4, 0.25), rtol=0.0, atol=1e-12)


def test_injection_diagnostics_track_survival_after_resampling():
    """Track attempted injections and survivors selected by resampling."""

    def propagate_fn(state, chain_index, particle_index, step, rng):
        del step, rng
        new_state = list(state)
        if chain_index == 0:
            new_state.append(("hot", particle_index))
        else:
            new_state.append(("cold", particle_index))
        return new_state

    def log_reward_fn(state, chain_index, particle_index, step):
        del chain_index, particle_index, step
        if state[-1][0] == "hot":
            return -9.0
        return 0.0

    config = MultiChainSMCConfig(
        betas=[0.0, 1.0],
        num_particles=8,
        num_steps=1,
        injection_probability=0.5,
        ess_threshold=0.95,
        seed=11,
    )
    smc = MultiChainSMC(config=config, propagate_fn=propagate_fn, log_reward_fn=log_reward_fn)
    result = smc.run(initial_state=[])

    cold_diag = result.step_diagnostics[0][1]
    attempted = cold_diag["attempted_injections"]
    survived = cold_diag["survived_injections"]

    assert attempted == int(np.sum(cold_diag["injection_mask"]))
    assert attempted > 0
    assert cold_diag["resampled"] is True
    expected_survived = int(np.sum(cold_diag["injection_mask"][cold_diag["ancestor_indices"]]))
    assert survived == expected_survived
    assert 0 <= survived <= attempted
    assert survived < attempted
