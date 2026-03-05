"""
Stage 4 multi-chain SMC with hot-to-cold mixture proposals.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


PropagateFn = Callable[[Any, int, int, int, np.random.Generator], Any]
LogRewardFn = Callable[[Any, int, int, int], float]


@dataclass(frozen=True)
class MultiChainSMCConfig:
    """Configuration for multi-chain SMC with hot injection."""

    betas: Sequence[float]
    num_particles: int
    num_steps: int
    injection_probability: float = 0.1
    ess_threshold: float = 0.5
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        beta_values = np.asarray(self.betas, dtype=float)
        if beta_values.ndim != 1 or beta_values.size < 1:
            raise ValueError("betas must be a non-empty 1D sequence.")
        if not np.all(np.diff(beta_values) > 0.0):
            raise ValueError("betas must be strictly increasing.")
        if not np.isclose(beta_values[0], 0.0):
            raise ValueError("Stage 4 requires beta_1 = 0.")
        if not np.isclose(beta_values[-1], 1.0):
            raise ValueError("Stage 4 requires beta_K = 1.")

        if self.num_particles <= 0:
            raise ValueError("num_particles must be positive.")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if not (0.0 <= self.injection_probability <= 1.0):
            raise ValueError("injection_probability must be in [0, 1].")
        if not (0.0 <= self.ess_threshold <= 1.0):
            raise ValueError("ess_threshold must be in [0, 1].")

    @property
    def num_chains(self) -> int:
        return len(self.betas)


@dataclass
class MultiChainSMCResult:
    """Outputs for a multi-chain SMC run."""

    particles: List[List[Any]]
    weights: List[np.ndarray]
    log_weights: List[np.ndarray]
    log_rewards_prev: List[np.ndarray]
    step_diagnostics: List[List[Dict[str, Any]]]


def _logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    if not np.isfinite(max_value):
        return max_value
    shifted = values - max_value
    return max_value + float(np.log(np.sum(np.exp(shifted))))


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    lse = _logsumexp(log_weights)
    if not np.isfinite(lse):
        return np.full(log_weights.shape, 1.0 / log_weights.size, dtype=float)

    weights = np.exp(log_weights - lse)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        return np.full(log_weights.shape, 1.0 / log_weights.size, dtype=float)
    return weights / weight_sum


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_particles = int(weights.size)
    positions = (rng.random() + np.arange(n_particles, dtype=float)) / n_particles
    cumsum = np.cumsum(weights)
    ancestors = np.searchsorted(cumsum, positions, side="left")
    return np.clip(ancestors, 0, n_particles - 1).astype(np.int64)


class MultiChainSMC:
    """
    Multi-chain SMC with adjacent-chain hot injection.

    `propagate_fn` signature:
      `propagate_fn(prev_state, chain_index, particle_index, step, rng) -> new_state`

    `log_reward_fn` signature:
      `log_reward_fn(state, chain_index, particle_index, step) -> log_reward`
    """

    def __init__(
        self,
        config: MultiChainSMCConfig,
        propagate_fn: PropagateFn,
        log_reward_fn: LogRewardFn,
    ) -> None:
        self.config = config
        self._betas = np.asarray(config.betas, dtype=float)
        self._propagate_fn = propagate_fn
        self._log_reward_fn = log_reward_fn
        self._rng = np.random.default_rng(config.seed)

    def run(self, initial_state: Any) -> MultiChainSMCResult:
        """Run SMC for all chains and return final states, weights, and diagnostics."""
        k_chains = self.config.num_chains
        n_particles = self.config.num_particles
        num_steps = self.config.num_steps

        particles: List[List[Any]] = [
            [deepcopy(initial_state) for _ in range(n_particles)] for _ in range(k_chains)
        ]
        log_rewards_prev = np.zeros((k_chains, n_particles), dtype=float)
        log_weights = np.zeros((k_chains, n_particles), dtype=float)
        step_diagnostics: List[List[Dict[str, Any]]] = []

        for step in range(num_steps):
            proposed_particles: List[List[Any]] = [[None] * n_particles for _ in range(k_chains)]
            log_rewards_curr = np.zeros((k_chains, n_particles), dtype=float)
            log_incremental_weights = np.zeros((k_chains, n_particles), dtype=float)
            injection_mask = np.zeros((k_chains, n_particles), dtype=bool)
            source_indices = np.full((k_chains, n_particles), -1, dtype=np.int64)

            # Phase 1: local propagation and local weight update.
            for chain_idx in range(k_chains):
                beta = self._betas[chain_idx]
                for particle_idx in range(n_particles):
                    state = self._propagate_fn(
                        deepcopy(particles[chain_idx][particle_idx]),
                        chain_idx,
                        particle_idx,
                        step,
                        self._rng,
                    )
                    proposed_particles[chain_idx][particle_idx] = state

                    curr_log_reward = float(self._log_reward_fn(state, chain_idx, particle_idx, step))
                    log_rewards_curr[chain_idx, particle_idx] = curr_log_reward
                    log_incremental_weights[chain_idx, particle_idx] = beta * (
                        curr_log_reward - log_rewards_prev[chain_idx, particle_idx]
                    )

            # Phase 2: hot injection from chain (k-1) to chain k.
            injection_probability = self.config.injection_probability
            if injection_probability > 0.0 and k_chains > 1:
                for chain_idx in range(1, k_chains):
                    beta_hot = self._betas[chain_idx - 1]
                    beta_cold = self._betas[chain_idx]
                    for particle_idx in range(n_particles):
                        if self._rng.random() >= injection_probability:
                            continue

                        hot_idx = int(self._rng.integers(0, n_particles))
                        proposed_particles[chain_idx][particle_idx] = deepcopy(
                            proposed_particles[chain_idx - 1][hot_idx]
                        )
                        log_rewards_curr[chain_idx, particle_idx] = log_rewards_curr[chain_idx - 1, hot_idx]
                        log_incremental_weights[chain_idx, particle_idx] = (
                            beta_cold - beta_hot
                        ) * log_rewards_curr[chain_idx, particle_idx]
                        injection_mask[chain_idx, particle_idx] = True
                        source_indices[chain_idx, particle_idx] = hot_idx

            # Accumulate incremental updates in log space.
            log_weights = log_weights + log_incremental_weights

            chain_step_diags: List[Dict[str, Any]] = []
            for chain_idx in range(k_chains):
                weights = _normalize_log_weights(log_weights[chain_idx])
                ess = float(1.0 / np.sum(weights**2))
                resampled = bool(ess < (self.config.ess_threshold * n_particles))
                ancestors = np.arange(n_particles, dtype=np.int64)

                attempted_injections = int(np.sum(injection_mask[chain_idx]))
                if resampled:
                    ancestors = _systematic_resample(weights, self._rng)
                    survived_injections = int(np.sum(injection_mask[chain_idx][ancestors]))
                    particles[chain_idx] = [
                        deepcopy(proposed_particles[chain_idx][ancestor_idx])
                        for ancestor_idx in ancestors.tolist()
                    ]
                    log_rewards_prev[chain_idx] = log_rewards_curr[chain_idx][ancestors]
                    log_weights[chain_idx] = 0.0
                else:
                    survived_injections = attempted_injections
                    particles[chain_idx] = [deepcopy(p) for p in proposed_particles[chain_idx]]
                    log_rewards_prev[chain_idx] = log_rewards_curr[chain_idx]

                chain_step_diags.append(
                    {
                        "ess": ess,
                        "resampled": resampled,
                        "attempted_injections": attempted_injections,
                        "survived_injections": survived_injections,
                        "injection_mask": injection_mask[chain_idx].copy(),
                        "ancestor_indices": ancestors.copy(),
                        "source_indices": source_indices[chain_idx].copy(),
                        "log_incremental_weights": log_incremental_weights[chain_idx].copy(),
                        "log_rewards_curr": log_rewards_curr[chain_idx].copy(),
                        "normalized_weights": weights.copy(),
                    }
                )
            step_diagnostics.append(chain_step_diags)

        final_weights = [_normalize_log_weights(log_weights[k]) for k in range(k_chains)]
        return MultiChainSMCResult(
            particles=particles,
            weights=final_weights,
            log_weights=[log_weights[k].copy() for k in range(k_chains)],
            log_rewards_prev=[log_rewards_prev[k].copy() for k in range(k_chains)],
            step_diagnostics=step_diagnostics,
        )
