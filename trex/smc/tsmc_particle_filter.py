"""
Twisted SMC particle filter with LLM generation and learned twist scorer.

This reuses LLMParticleFilter generation/step logic but replaces PRM scores
with twist value ratios.
"""

import logging
from collections import Counter, defaultdict
from typing import Callable, List, Optional, Protocol

import torch

from trex.smc.llm_particle_filter import LLMParticleFilter
from trex.smc.resampling import compute_ess

logger = logging.getLogger(__name__)


class TwistScorer(Protocol):
    """Protocol for twist scorers."""

    def score_texts(self, texts: List[str]) -> torch.Tensor:
        """Return ψ or logψ per text according to twist_space."""
        raise NotImplementedError


class TSMCLLMParticleFilter(LLMParticleFilter):
    """
    Twisted SMC with LLM generation and learned value function.

    Uses twist ratios for weight updates:
        w_t ∝ w_{t-1} * ψ_t / ψ_{t-1}  (prob space)
        w_t ∝ w_{t-1} * exp(logψ_t - logψ_{t-1}) (log space)
    """

    def __init__(
        self,
        config,
        generator,
        twist_scorer: TwistScorer,
        reward_model=None,
        answer_extractor: Optional[Callable[[str], str]] = None,
    ):
        super().__init__(config, generator, reward_model)
        self.twist_scorer = twist_scorer
        self.answer_extractor = answer_extractor
        self.log_space = config.twist_space == "log_prob"
        self.twist_mode = getattr(config, "twist_mode", "value")
        self.epsilon = getattr(config, "epsilon", 1e-8)
        self.log_value_min = getattr(config, "log_value_min", -1e6)
        self.warmup_steps = max(0, int(getattr(config, "warmup_steps", 0)))
        self.warmup_tokens = max(0, int(getattr(config, "warmup_tokens", 0)))

    def initialize(self, prompt: str) -> None:
        """Initialize particles, lineage scores, and prompt-state twist baselines."""
        super().initialize(prompt)
        for particle in self.particles:
            particle.metadata["twist_log_weight"] = 0.0

        # Seed prev_value at the prompt state so the first twist update is not neutralized.
        prompt_texts = [particle.text for particle in self.particles]
        prompt_values = self.twist_scorer.score_texts(prompt_texts)
        if not isinstance(prompt_values, torch.Tensor):
            prompt_values = torch.tensor(prompt_values, device=self._device)
        prompt_values = prompt_values.to(self._device).view(-1)
        if prompt_values.numel() != len(self.particles):
            raise ValueError(
                "twist_scorer returned an unexpected number of prompt values: "
                f"{prompt_values.numel()} != {len(self.particles)}"
            )
        prompt_values = self._apply_twist_mode(prompt_values)
        self._set_prev_values_metadata(prompt_values)

    def _get_prev_values_tensor(self) -> Optional[torch.Tensor]:
        prev_values = []
        for p in self.particles:
            if "prev_value" not in p.metadata:
                return None
            prev_values.append(p.metadata["prev_value"])
        if not prev_values:
            return None
        return torch.tensor(prev_values, device=self._device, dtype=torch.float32)

    def _set_prev_values_metadata(self, values: torch.Tensor) -> None:
        values_list = values.detach().to("cpu").view(-1).tolist()
        for p, v in zip(self.particles, values_list):
            p.metadata["prev_value"] = float(v)

    def _apply_twist_mode(self, values: torch.Tensor) -> torch.Tensor:
        if self.twist_mode == "value":
            return values
        if self.twist_mode == "sqrt_value":
            if self.log_space:
                return 0.5 * values
            return torch.sqrt(values.clamp(min=0.0))
        raise ValueError(f"Unknown twist_mode: {self.twist_mode}")

    def _compute_twist_values(self) -> torch.Tensor:
        n = len(self.particles)
        device = self._device

        active_indices = self._get_active_indices()
        prev_values = self._get_prev_values_tensor()
        if prev_values is None:
            prev_values = torch.zeros(n, device=device)

        current_values = prev_values.clone()

        if active_indices:
            active_texts = [self.particles[i].text for i in active_indices]
            active_values = self.twist_scorer.score_texts(active_texts)
            if not isinstance(active_values, torch.Tensor):
                active_values = torch.tensor(active_values, device=device)
            active_values = active_values.to(device).view(-1)

            current_values[active_indices] = active_values

        current_values = self._apply_twist_mode(current_values)
        return current_values

    def _update_weights_with_twist(
        self, current_values: torch.Tensor, previous_values: torch.Tensor
    ) -> torch.Tensor:
        current_values = current_values.to(self._device).float().view(-1)
        previous_values = previous_values.to(self._device).float().view(-1)

        if self.log_space:
            current_values = torch.clamp(current_values, min=self.log_value_min, max=0.0)
            previous_values = torch.clamp(previous_values, min=self.log_value_min, max=0.0)
            delta = current_values - previous_values
            log_weights = torch.log(self.get_weights() + self.epsilon) + delta
            log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
            return torch.exp(log_weights)

        ratios = current_values / (previous_values + self.epsilon)
        new_weights = self.get_weights() * ratios
        weight_sum = new_weights.sum()
        if not torch.isfinite(weight_sum) or weight_sum <= 0:
            logger.warning(
                "Twist weight sum is non-finite or zero in prob space; "
                "falling back to uniform weights for stability."
            )
            n_particles = max(1, new_weights.numel())
            return torch.full_like(new_weights, 1.0 / n_particles)
        new_weights = new_weights / weight_sum
        return new_weights

    def _accumulate_lineage_log_weights(
        self, current_values: torch.Tensor, previous_values: torch.Tensor
    ) -> None:
        """Track cumulative twist log-weights for ORM-disabled final selection."""
        current_values = current_values.to(self._device).float().view(-1)
        previous_values = previous_values.to(self._device).float().view(-1)

        if self.log_space:
            current_values = torch.clamp(current_values, min=self.log_value_min, max=0.0)
            previous_values = torch.clamp(previous_values, min=self.log_value_min, max=0.0)
            deltas = current_values - previous_values
        else:
            current_safe = torch.clamp(current_values, min=self.epsilon)
            previous_safe = torch.clamp(previous_values, min=self.epsilon)
            deltas = torch.log(current_safe) - torch.log(previous_safe)

        for particle, delta in zip(self.particles, deltas.detach().cpu().tolist()):
            particle.metadata["twist_log_weight"] = (
                float(particle.metadata.get("twist_log_weight", 0.0)) + float(delta)
            )

    def _select_best_particle_without_orm(self):
        """
        Select a stable fallback particle when ORM is disabled.

        With every-step resampling, current weights are reset to uniform. In that
        setting, use lineage twist log-weight metadata instead of current weight.
        """
        if self.config.resampling_strategy != "every_step":
            return self.get_best_particle()

        if not self.particles:
            raise ValueError("No particles initialized.")

        scores = [float(p.metadata.get("twist_log_weight", float("-inf"))) for p in self.particles]
        if all(score == float("-inf") for score in scores):
            return self.get_best_particle()
        best_idx = max(range(len(self.particles)), key=lambda idx: scores[idx])
        return self.particles[best_idx]

    def _is_in_warmup(self, active_indices: List[int]) -> bool:
        """Return True while the configured warm-up window is active."""
        if self._smc_iteration < self.warmup_steps:
            return True

        if self.config.resampling_unit != "token" or self.warmup_tokens <= 0:
            return False
        if not active_indices:
            return False

        min_tokens = min(
            int(self.particles[idx].metadata.get("generated_tokens_total", 0))
            for idx in active_indices
        )
        return min_tokens < self.warmup_tokens

    def _extract_answer(self, text: str) -> str:
        response = self._assistant_response(text).strip()
        if self.answer_extractor is None:
            return response
        extracted = self.answer_extractor(response)
        if extracted is None:
            return ""
        return str(extracted).strip()

    def select_by_majority_vote(self):
        """
        Select a particle by majority answer vote with a lineage-score tie break.
        """
        if not self.particles:
            raise ValueError("No particles initialized.")

        answer_to_indices = defaultdict(list)
        for idx, particle in enumerate(self.particles):
            answer = self._extract_answer(particle.text)
            if answer:
                answer_to_indices[answer].append(idx)

        if not answer_to_indices:
            return self._select_best_particle_without_orm()

        counts = Counter({answer: len(indices) for answer, indices in answer_to_indices.items()})
        max_count = max(counts.values())
        tied_answers = [answer for answer, count in counts.items() if count == max_count]

        def answer_best_lineage(answer: str) -> float:
            return max(
                float(self.particles[idx].metadata.get("twist_log_weight", float("-inf")))
                for idx in answer_to_indices[answer]
            )

        best_answer = max(tied_answers, key=answer_best_lineage)
        candidate_indices = answer_to_indices[best_answer]

        best_idx = max(
            candidate_indices,
            key=lambda idx: (
                float(self.particles[idx].metadata.get("twist_log_weight", float("-inf"))),
                float(self.get_weights()[idx]),
            ),
        )
        return self.particles[best_idx]

    def score_particles(self) -> torch.Tensor:
        """
        Compute current twist values for particles.

        Returns:
            Tensor of twist values (ψ or logψ), shape (n_particles,)
        """
        return self._compute_twist_values()

    def _step_by_token_chunk(self) -> bool:
        """
        TSMC step for token-interval mode: generate chunk, score, twist-update, resample.
        """
        active_indices = self._get_active_indices()
        if not active_indices:
            return False

        self._expand_token_chunk(active_indices)

        current_values = self.score_particles()
        previous_values = self._get_prev_values_tensor()
        if previous_values is None:
            previous_values = current_values.clone().detach()

        new_weights = self._update_weights_with_twist(current_values, previous_values)
        self._accumulate_lineage_log_weights(current_values, previous_values)
        self._set_prev_values_metadata(current_values)

        active_weights = new_weights[active_indices] if active_indices else new_weights

        if active_weights.numel() > 0:
            ess = compute_ess(active_weights)
        else:
            ess = None

        resampled = False
        in_warmup = self._is_in_warmup(active_indices)
        if in_warmup:
            self.set_weights(new_weights)
        elif self.config.resampling_strategy == "every_step":
            self._resample_active(active_indices, active_weights)
            resampled = True
        elif self.config.resampling_strategy == "ess_adaptive":
            if active_weights.numel() > 0:
                if ess is not None and ess < self.config.ess_threshold * len(active_indices):
                    self._resample_active(active_indices, active_weights)
                    resampled = True
                else:
                    self.set_weights(new_weights)
        else:
            self.set_weights(new_weights)

        if resampled:
            pass

        self._smc_iteration += 1

        if self._smc_iteration >= self.config.max_smc_iterations:
            for p in self.particles:
                if not p.metadata.get("finished"):
                    p.metadata["finished"] = True
                    p.metadata["max_smc_iterations_reached"] = True
            return False

        return any(not p.metadata.get("finished") for p in self.particles)

    def step(self) -> bool:
        """
        Single TSMC step: expand, score with twist, resample, inject headers.
        """
        if self.config.resampling_unit == "token":
            return self._step_by_token_chunk()

        # Complete the current reasoning step for all active particles.
        chunk_calls = 0
        while True:
            active_indices = self._get_active_indices()
            if not active_indices:
                return False

            pending_indices = self._get_pending_indices()
            if not pending_indices:
                break

            self.expand_particles()
            progressed = self._last_generation_progressed
            chunk_calls += 1

            if (not progressed) or (chunk_calls >= self.config.max_step_chunk_calls):
                for idx in pending_indices:
                    self.particles[idx].metadata["needs_step_header"] = True
                    self.particles[idx].metadata["forced_step_boundary"] = True
                break

        # In delimiter mode, inject the boundary token before scoring so the twist
        # observes the same context used for subsequent generation.
        if self.config.step_boundary_mode == "delimiter":
            self.inject_next_step_headers()

        current_values = self.score_particles()
        previous_values = self._get_prev_values_tensor()
        if previous_values is None:
            previous_values = current_values.clone().detach()

        new_weights = self._update_weights_with_twist(current_values, previous_values)
        self._accumulate_lineage_log_weights(current_values, previous_values)
        self._set_prev_values_metadata(current_values)

        active_indices = self._get_active_indices()
        active_weights = new_weights[active_indices] if active_indices else new_weights

        if active_weights.numel() > 0:
            ess = compute_ess(active_weights)
        else:
            ess = None

        resampled = False
        in_warmup = self._is_in_warmup(active_indices)
        if in_warmup:
            self.set_weights(new_weights)
        elif self.config.resampling_strategy == "every_step":
            self._resample_active(active_indices, active_weights)
            resampled = True
        elif self.config.resampling_strategy == "ess_adaptive":
            if active_weights.numel() > 0:
                if ess is not None and ess < self.config.ess_threshold * len(active_indices):
                    self._resample_active(active_indices, active_weights)
                    resampled = True
                else:
                    self.set_weights(new_weights)
        else:
            self.set_weights(new_weights)

        if resampled:
            pass

        if self.config.step_boundary_mode != "delimiter":
            self.inject_next_step_headers()

        self._smc_iteration += 1

        if self._smc_iteration >= self.config.max_smc_iterations:
            for p in self.particles:
                if not p.metadata.get("finished"):
                    p.metadata["finished"] = True
                    p.metadata["max_smc_iterations_reached"] = True
            return False

        return any(not p.metadata.get("finished") for p in self.particles)

    def select_best_by_orm(self):
        if self.reward_model is None:
            raise ValueError("Reward model is required for ORM-based selection.")
        return super().select_best_by_orm()

    def run(self):
        """Run TSMC loop and return final selected particle."""
        while self.step():
            pass

        selection_mode = getattr(self.config, "final_selection_mode", None)
        if selection_mode is None:
            selection_mode = "orm" if getattr(self.config, "use_orm_for_final", False) else "twist_weight"

        if selection_mode == "orm":
            return self.select_best_by_orm()
        if selection_mode == "majority_vote":
            return self.select_by_majority_vote()
        if selection_mode == "twist_weight":
            return self._select_best_particle_without_orm()
        raise ValueError(f"Unknown final_selection_mode: {selection_mode}")
