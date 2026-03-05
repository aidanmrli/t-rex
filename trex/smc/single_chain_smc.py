"""
Stage 3 single-chain SMC core.

This module implements a lightweight SMC core with:
- Fixed beta incremental log-weight updates.
- ESS-based systematic resampling.
- Incremental log normalizing-constant tracking in log space.
- Cached per-particle log-reward state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np


def _logsumexp(values: np.ndarray) -> float:
    """Numerically stable log(sum(exp(values)))."""
    max_value = float(np.max(values))
    if not np.isfinite(max_value):
        return max_value
    return max_value + float(np.log(np.sum(np.exp(values - max_value))))


def _as_1d_float_array(values: Sequence[float], expected_size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape[0] != expected_size:
        raise ValueError(f"{name} must have length {expected_size}, got {array.shape[0]}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling from normalized weights."""
    n_particles = weights.shape[0]
    positions = (np.arange(n_particles, dtype=float) + rng.uniform()) / n_particles
    cumsum = np.cumsum(weights)
    ancestors = np.searchsorted(cumsum, positions, side="left")
    return np.clip(ancestors, 0, n_particles - 1).astype(np.int64)


@dataclass(frozen=True)
class StepResult:
    """Diagnostics for one SMC step."""

    step: int
    ess: float
    resampled: bool
    ancestors: Optional[np.ndarray]
    log_z: float
    normalized_weights: np.ndarray
    particles: Optional[List[Any]]


class SingleChainSMC:
    """
    Single-chain SMC core with fixed beta and ESS-triggered systematic resampling.
    """

    def __init__(
        self,
        n_particles: int,
        beta: float,
        ess_threshold: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        if n_particles <= 0:
            raise ValueError("n_particles must be positive.")
        if not np.isfinite(beta):
            raise ValueError("beta must be finite.")

        self.n_particles = int(n_particles)
        self.beta = float(beta)
        self.ess_threshold = (
            self.n_particles / 2.0 if ess_threshold is None else self._resolve_ess_threshold(ess_threshold)
        )
        self._rng = np.random.default_rng(seed)

        self.log_weights = np.zeros(self.n_particles, dtype=float)
        self.log_reward_prev: Optional[np.ndarray] = None
        self.log_reward_curr: Optional[np.ndarray] = None
        self.particles: Optional[List[Any]] = None
        self.log_z = 0.0
        self._step = 0

        self._ess_history: List[float] = []
        self._log_z_history: List[float] = []
        self._resampled_history: List[bool] = []
        self._ancestor_history: List[Optional[np.ndarray]] = []

    def _resolve_ess_threshold(self, ess_threshold: float) -> float:
        threshold = float(ess_threshold)
        if threshold < 0:
            raise ValueError("ess_threshold must be non-negative.")
        if 0 < threshold <= 1:
            return threshold * self.n_particles
        return threshold

    def _normalized_weights_from_log_weights(self) -> np.ndarray:
        log_total = _logsumexp(self.log_weights)
        return np.exp(self.log_weights - log_total)

    @property
    def normalized_weights(self) -> np.ndarray:
        """Current normalized particle weights."""
        return self._normalized_weights_from_log_weights()

    def initialize(self, log_reward_init: Sequence[float], particles: Optional[Sequence[Any]] = None) -> None:
        """
        Initialize per-particle reward cache and reset sampler state.
        """
        log_reward_init = _as_1d_float_array(log_reward_init, self.n_particles, "log_reward_init")

        self.log_weights = np.zeros(self.n_particles, dtype=float)
        self.log_reward_prev = log_reward_init.copy()
        self.log_reward_curr = log_reward_init.copy()
        self.log_z = 0.0
        self._step = 0
        self._ess_history.clear()
        self._log_z_history.clear()
        self._resampled_history.clear()
        self._ancestor_history.clear()

        if particles is None:
            self.particles = None
        else:
            if len(particles) != self.n_particles:
                raise ValueError(f"particles must have length {self.n_particles}, got {len(particles)}.")
            self.particles = list(particles)

    def step(self, log_reward_curr: Sequence[float], particles: Optional[Sequence[Any]] = None) -> StepResult:
        """
        Run one SMC reweight/resample step.

        Args:
            log_reward_curr: Per-particle log reward for current state x_{1:t}.
            particles: Optional particle payloads aligned with log_reward_curr.

        Returns:
            StepResult with ESS/resampling/log_Z diagnostics.
        """
        if self.log_reward_prev is None:
            raise RuntimeError("Call initialize() before step().")

        log_reward_curr_array = _as_1d_float_array(log_reward_curr, self.n_particles, "log_reward_curr")
        if particles is None:
            particle_list = None if self.particles is None else list(self.particles)
        else:
            if len(particles) != self.n_particles:
                raise ValueError(f"particles must have length {self.n_particles}, got {len(particles)}.")
            particle_list = list(particles)

        log_increment = self.beta * (log_reward_curr_array - self.log_reward_prev)

        # Incremental log normalizing constant update:
        # log c_t = logsumexp(log_w_prev + log_increment) - logsumexp(log_w_prev)
        # log_Z = sum_t log c_t
        prev_log_total = _logsumexp(self.log_weights)
        next_log_weights = self.log_weights + log_increment
        next_log_total = _logsumexp(next_log_weights)
        self.log_z += next_log_total - prev_log_total
        self.log_weights = next_log_weights

        normalized = self._normalized_weights_from_log_weights()
        ess = float(1.0 / np.sum(np.square(normalized)))

        resampled = bool(ess < self.ess_threshold)
        ancestors: Optional[np.ndarray] = None
        next_log_rewards = log_reward_curr_array
        next_particles = particle_list

        if resampled:
            ancestors = _systematic_resample(normalized, self._rng)
            next_log_rewards = log_reward_curr_array[ancestors]
            if particle_list is not None:
                next_particles = [particle_list[idx] for idx in ancestors]
            self.log_weights = np.zeros(self.n_particles, dtype=float)
            normalized = self._normalized_weights_from_log_weights()

        self.log_reward_curr = next_log_rewards.copy()
        self.log_reward_prev = next_log_rewards.copy()
        self.particles = None if next_particles is None else list(next_particles)

        self._step += 1
        self._ess_history.append(ess)
        self._log_z_history.append(self.log_z)
        self._resampled_history.append(resampled)
        self._ancestor_history.append(None if ancestors is None else ancestors.copy())

        return StepResult(
            step=self._step,
            ess=ess,
            resampled=resampled,
            ancestors=None if ancestors is None else ancestors.copy(),
            log_z=self.log_z,
            normalized_weights=normalized.copy(),
            particles=None if next_particles is None else list(next_particles),
        )

    def run(
        self,
        log_reward_steps: Sequence[Sequence[float]],
        particles_steps: Optional[Sequence[Sequence[Any]]] = None,
    ) -> List[StepResult]:
        """
        Run multiple SMC steps.
        """
        if particles_steps is not None and len(particles_steps) != len(log_reward_steps):
            raise ValueError("particles_steps must have the same length as log_reward_steps.")

        results: List[StepResult] = []
        if particles_steps is None:
            for rewards in log_reward_steps:
                results.append(self.step(rewards))
            return results

        for rewards, particles in zip(log_reward_steps, particles_steps):
            results.append(self.step(rewards, particles=particles))
        return results

    def diagnostics(self) -> dict:
        """Return aggregated run diagnostics."""
        return {
            "num_steps": self._step,
            "log_z": self.log_z,
            "ess_threshold": self.ess_threshold,
            "ess_history": np.asarray(self._ess_history, dtype=float),
            "log_z_history": np.asarray(self._log_z_history, dtype=float),
            "resampled_history": np.asarray(self._resampled_history, dtype=bool),
            "ancestor_history": [None if anc is None else anc.copy() for anc in self._ancestor_history],
        }
