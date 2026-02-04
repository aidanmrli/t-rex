"""
LLM-aware Particle Filter for SMC Steering.

This module extends the base ParticleFilter with:
- LLM generation (via vLLM)
- PRM scoring for weight updates
- ORM scoring for final answer selection
- Step detection and tracking

The LLMParticleFilter implements the core SMC loop for inference-time
compute scaling in language models.
"""

import logging
import re
from copy import deepcopy
from typing import List, Optional, TYPE_CHECKING, Tuple

import torch

from trex.smc.particle_filter import ParticleFilter, Particle, SMCConfig
from trex.smc.resampling import (
    multinomial_resampling,
    systematic_resampling,
    stratified_resampling,
    compute_ess,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams
    from trex.models.reward_model import RewardModel
    from trex.baselines.smc_config import SMCSteeringConfig


class LLMParticleFilter(ParticleFilter):
    """
    Particle Filter with LLM generation and PRM scoring.
    
    Extends the base ParticleFilter to add:
    - expand_particles(): Generate next step with LLM
    - score_particles(): Score with PRM
    - select_best_by_orm(): Select best particle using ORM scores
    - step(): One SMC iteration (expand → score → resample)
    - run(): Full SMC loop with ORM selection
    
    Inherits from ParticleFilter:
    - initialize(prompt): Creates N particles with prompt
    - set_weights() / get_weights(): Weight management
    - normalize_weights(): Normalize to sum=1
    - effective_sample_size(): ESS computation
    - should_resample(): Check ESS threshold
    - resample(): Perform resampling (uses deepcopy for independence)
    - get_particle_texts(): Get all texts
    - get_best_particle(): Highest weight particle
    
    Attributes:
        config: SMCSteeringConfig with all parameters
        generator: vLLM LLM instance for text generation
        reward_model: RewardModel for PRM/ORM scoring
        _smc_iteration: Current SMC loop iteration count
    """
    
    # Regex patterns for step detection
    STEP_PATTERN = re.compile(r"## Step \d+:")
    STEP_HEADER_PATTERN = re.compile(r"## Step (\d+):")
    
    # Pattern for detecting final answer
    BOXED_PATTERN = re.compile(r"\\boxed\{")
    
    def __init__(
        self,
        config: "SMCSteeringConfig",
        generator: "LLM",
        reward_model: "RewardModel",
    ):
        """
        Initialize the LLM Particle Filter.
        
        Args:
            config: SMCSteeringConfig with all parameters
            generator: vLLM LLM instance for generation
            reward_model: RewardModel for PRM/ORM scoring
        """
        # Create SMCConfig from SMCSteeringConfig
        smc_config = SMCConfig(
            n_particles=config.n_particles,
            resampling_strategy=config.resampling_strategy,
            ess_threshold=config.ess_threshold,
            resampling_method=config.resampling_method,
            seed=config.seed,
        )
        super().__init__(smc_config)
        
        self.config = config
        self.generator = generator
        self.reward_model = reward_model
        
        # SMC loop iteration count (expand → score → resample cycles)
        # Note: Different from reasoning_step_count which tracks "## Step N:" in text
        self._smc_iteration = 0
        self._last_generation_progressed = False
        self._tokenizer = None

    def initialize(self, prompt: str) -> None:
        """
        Initialize particles with the given prompt and set PRM metadata.

        Args:
            prompt: Starting text for all particles
        """
        super().initialize(prompt)
        for p in self.particles:
            p.metadata["prm_preamble"] = prompt
            p.metadata["prm_steps"] = []
            p.metadata["prompt_truncated"] = False
            p.metadata["prompt_truncated_tokens"] = 0

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.generator.get_tokenizer()
        return self._tokenizer

    def _tokenize_prompt(self, text: str) -> Optional[List[int]]:
        tokenizer = self._get_tokenizer()
        if hasattr(tokenizer, "encode"):
            return tokenizer.encode(text, add_special_tokens=False)
        if callable(tokenizer):
            try:
                encoded = tokenizer(text, add_special_tokens=False)
                if isinstance(encoded, dict) and "input_ids" in encoded:
                    return encoded["input_ids"]
                if isinstance(encoded, list):
                    return encoded
            except Exception:
                return None
        return None

    def _decode_prompt(self, token_ids: List[int]) -> Optional[str]:
        tokenizer = self._get_tokenizer()
        if hasattr(tokenizer, "decode"):
            try:
                return tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except Exception:
                return None
        return None

    def _infer_model_max_len(self) -> Optional[int]:
        try:
            llm_engine = getattr(self.generator, "llm_engine", None)
            if llm_engine is not None:
                model_config = getattr(llm_engine, "model_config", None)
                if model_config is not None:
                    max_len = getattr(model_config, "max_model_len", None)
                    if isinstance(max_len, int) and max_len > 0:
                        return max_len
        except Exception:
            pass

        try:
            tokenizer = self._get_tokenizer()
            max_len = getattr(tokenizer, "model_max_length", None)
            if isinstance(max_len, int) and 0 < max_len < 1_000_000:
                return max_len
        except Exception:
            pass

        return None

    def _compute_prompt_max_tokens(self, max_new_tokens: int) -> Optional[int]:
        if not self.config.enable_prompt_truncation and self.config.prompt_max_tokens is None:
            return None

        prompt_max = self.config.prompt_max_tokens
        model_max = self._infer_model_max_len()
        if model_max is not None:
            model_based = max(model_max - max_new_tokens, 1)
            if prompt_max is None:
                prompt_max = model_based
            else:
                prompt_max = min(prompt_max, model_based)

        return prompt_max

    def _prepare_prompt_inputs(
        self, indices: List[int], max_new_tokens: int
    ) -> Tuple[List[str], List[object]]:
        """
        Prepare prompts for vLLM, optionally using TokensPrompt and truncating.

        Returns:
            Tuple of (prompt_texts, prompt_inputs) where prompt_inputs are either
            strings or TokensPrompt objects.
        """
        prompt_texts = []
        prompt_inputs = []
        max_prompt_tokens = self._compute_prompt_max_tokens(max_new_tokens)
        use_token_prompts = self.config.use_token_prompts

        tokenizer_needed = use_token_prompts or max_prompt_tokens is not None

        for idx in indices:
            text = self.particles[idx].text
            token_ids = None
            truncated = False

            if tokenizer_needed:
                token_ids = self._tokenize_prompt(text)
                if token_ids is not None:
                    prompt_len = len(token_ids)
                    self.particles[idx].metadata["prompt_token_len"] = prompt_len
                    self.particles[idx].metadata["prompt_max_tokens"] = max_prompt_tokens

                    if max_prompt_tokens is not None and prompt_len > max_prompt_tokens:
                        truncated_ids = token_ids[-max_prompt_tokens:]
                        decoded = self._decode_prompt(truncated_ids)
                        if decoded is not None:
                            text = decoded
                            token_ids = truncated_ids
                            truncated = True
                            self.particles[idx].metadata["prompt_truncated"] = True
                            self.particles[idx].metadata["prompt_truncated_tokens"] = (
                                prompt_len - len(truncated_ids)
                            )
                            self.particles[idx].metadata[
                                "prompt_token_len_after_truncation"
                            ] = len(truncated_ids)
                            if self.config.resampling_unit == "token":
                                self.particles[idx].metadata["prm_preamble"] = text
                                self.particles[idx].metadata["prm_steps"] = []
                        else:
                            logger.warning(
                                "Prompt truncation needed but tokenizer.decode is unavailable. "
                                "Skipping truncation; prompt may exceed model context."
                            )

                else:
                    logger.warning(
                        "Prompt tokenization unavailable. Skipping truncation checks."
                    )

            if use_token_prompts and token_ids is not None:
                from vllm.inputs import TokensPrompt

                prompt_inputs.append(TokensPrompt(prompt_token_ids=token_ids))
            else:
                prompt_inputs.append(text)

            if truncated:
                self.particles[idx].metadata["prompt_truncation_side"] = "left"

            prompt_texts.append(text)

        return prompt_texts, prompt_inputs

    def _build_sampling_params(self, max_tokens: int, stop: Optional[List[str]]):
        """
        Build vLLM SamplingParams, omitting optional fields when unset.

        Some vLLM versions do not accept None for numeric options such as top_p/top_k.
        """
        from vllm import SamplingParams

        kwargs = {
            "n": 1,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens,
            "stop": stop,
            "include_stop_str_in_output": False,
        }
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            kwargs["top_k"] = self.config.top_k

        return SamplingParams(**kwargs)
    
    @property
    def smc_iteration(self) -> int:
        """Current SMC iteration count."""
        return self._smc_iteration
    
    def _get_next_step_number(self, text: str) -> int:
        """
        Get the next step number to generate based on existing text.

        Counts existing "## Step N:" patterns in the assistant's response
        (after <|im_start|>assistant) and returns N+1.

        Args:
            text: Current particle text (may include system/user messages)

        Returns:
            Next step number (1 if no steps exist in assistant response)
        """
        # Only count steps in assistant's response, not system prompt examples
        if "<|im_start|>assistant" in text:
            response = text.split("<|im_start|>assistant")[-1]
        else:
            response = text

        matches = self.STEP_HEADER_PATTERN.findall(response)
        if not matches:
            return 1
        return max(int(m) for m in matches) + 1
    
    def _inject_step_header(self, text: str) -> str:
        """
        Inject the next step header if text ends mid-step.

        This ensures proper step numbering across expansion calls.
        The LLM needs to see the complete "## Step N:" pattern in
        context to continue correctly.

        Args:
            text: Current particle text

        Returns:
            Text with step header injected if needed
        """
        stripped = text.rstrip()
        last_line = stripped.split('\n')[-1] if '\n' in stripped else stripped

        # Check if the last line already has a complete step header (## Step N:)
        if self.STEP_PATTERN.search(last_line):
            return text

        # Check if text ends with partial step header from stop string ("## Step")
        # This can happen if stop strings are included in output upstream.
        if stripped.endswith("## Step"):
            # Complete the partial header with the correct number
            next_step = self._get_next_step_number(text)
            return text + f" {next_step}:"

        # No step header at all - inject a new one
        next_step = self._get_next_step_number(text)
        return text + f"\n\n## Step {next_step}:"
    
    def _count_reasoning_steps(self, text: str) -> int:
        """
        Count the number of reasoning steps in the assistant's response.

        Args:
            text: Text to count steps in (may include system/user messages)

        Returns:
            Number of "## Step N:" patterns found in assistant response
        """
        # Only count steps in assistant's response, not system prompt examples
        if "<|im_start|>assistant" in text:
            response = text.split("<|im_start|>assistant")[-1]
        else:
            response = text

        return len(self.STEP_HEADER_PATTERN.findall(response))
    
    def _is_finished(self, particle: Particle) -> bool:
        """
        Check if a particle is finished (has final answer or hit limits).
        
        Args:
            particle: Particle to check
            
        Returns:
            True if particle should not be expanded further
        """
        text = particle.text
        
        # Has final boxed answer
        if self.BOXED_PATTERN.search(text):
            return True
        
        # Hit max characters (not tokens - character counting is faster during generation)
        if len(text) > self.config.max_total_chars:
            return True
        
        # Hit max reasoning steps
        step_count = self._count_reasoning_steps(text)
        if step_count > self.config.max_reasoning_steps:
            return True
        
        return False

    def _get_active_indices(self) -> List[int]:
        """Indices of particles that are still active (not finished)."""
        return [i for i, p in enumerate(self.particles) if not p.metadata.get("finished")]

    def _get_pending_indices(self) -> List[int]:
        """
        Indices of particles that still need generation for the current step.

        Pending = active and not waiting for the next step header.
        """
        return [
            i
            for i, p in enumerate(self.particles)
            if (not p.metadata.get("finished")) and (not p.metadata.get("needs_step_header"))
        ]

    def _resample_active(
        self, active_indices: List[int], active_weights: torch.Tensor
    ) -> None:
        """
        Resample only the active particles, preserving finished ones.

        This keeps finished particles fixed (not duplicated or dropped) while
        resampling the active subset based on their weights.
        """
        if not active_indices:
            return

        n_active = len(active_indices)

        if self.config.resampling_method == "multinomial":
            indices = multinomial_resampling(active_weights, n_active)
        elif self.config.resampling_method == "systematic":
            indices = systematic_resampling(active_weights)
        elif self.config.resampling_method == "stratified":
            indices = stratified_resampling(active_weights)
        else:
            raise ValueError(f"Unknown resampling method: {self.config.resampling_method}")

        indices_list = indices.tolist()
        old_particles = self.particles
        resampled = [deepcopy(old_particles[active_indices[idx]]) for idx in indices_list]

        for dst_idx, new_particle in zip(active_indices, resampled):
            self.particles[dst_idx] = new_particle

        # Reset weights to uniform after resampling (standard SMC behavior).
        n_total = len(self.particles)
        uniform_weight = 1.0 / n_total
        self._weights = torch.full((n_total,), uniform_weight, device=self._device, dtype=torch.float32)
        for p in self.particles:
            p.weight = uniform_weight
    
    def expand_particles(self) -> bool:
        """
        Generate next reasoning step for all active particles.

        Uses stop sequences to detect step boundaries:
        - If model outputs "## Step", it wants to continue → keep in SMC loop
        - If model finishes without "## Step" (EOS), it's done → mark finished

        The stop string is EXCLUDED from output. We check finish_reason to know
        if the model hit the stop string or finished naturally.

        IMPORTANT: This method does NOT inject next step headers. Header injection
        happens in inject_next_step_headers() AFTER scoring. This ensures the PRM
        scores actual content, not empty step headers.

        Particles are marked as finished when:
        - Model produces EOS (natural completion with \\boxed{})
        - Text contains \\boxed{} (has final answer)
        - Max character limit reached
        - Max reasoning steps reached

        Returns:
            True if any new text was generated, False otherwise
        """
        # Only generate for pending particles (active and not waiting for next header)
        pending_indices = self._get_pending_indices()

        if not pending_indices:
            # Nothing to generate (all finished or waiting for next header)
            self._last_generation_progressed = False
            return any(not p.metadata.get("finished") for p in self.particles)

        # Generate until next step header or EOS. Exclude stop string from output.
        pending_prompt_texts, pending_prompt_inputs = self._prepare_prompt_inputs(
            pending_indices, max_new_tokens=self.config.max_tokens_per_step
        )

        # Use dynamic stop sequence when all pending particles are on the same next step.
        next_steps = {self._get_next_step_number(text) for text in pending_prompt_texts}
        if len(next_steps) == 1:
            next_step = next_steps.pop()
            stop_sequences = [f"## Step {next_step}:"]
        else:
            stop_sequences = ["## Step"]

        # Generate until next step header or EOS. Exclude stop string from output.
        sampling_params = self._build_sampling_params(
            max_tokens=self.config.max_tokens_per_step,
            stop=stop_sequences,
        )
        outputs = self.generator.generate(pending_prompt_inputs, sampling_params)

        any_progress = False

        for offset, particle_idx in enumerate(pending_indices):
            output = outputs[offset].outputs[0]
            continuation = output.text
            if continuation:
                any_progress = True

            self.particles[particle_idx].text = pending_prompt_texts[offset] + continuation

            # Check finish_reason to understand why generation stopped
            if output.finish_reason == "stop":
                # Model hit step boundary - mark for header injection after scoring
                self.particles[particle_idx].metadata["needs_step_header"] = True
                logger.debug(
                    f"Particle {particle_idx} hit step boundary, will inject header after scoring"
                )
            elif output.finish_reason == "length":
                # Max tokens reached mid-step: keep generating in the same step
                self.particles[particle_idx].metadata["finish_reason"] = "length"
                self.particles[particle_idx].metadata["truncated_step"] = True
            else:
                # Model finished naturally (EOS) or other stop condition
                self.particles[particle_idx].metadata["finished"] = True
                self.particles[particle_idx].metadata["finish_reason"] = output.finish_reason
                logger.debug(
                    f"Particle {particle_idx} finished naturally: finish_reason={output.finish_reason}"
                )

            response_token_ids = getattr(output, "token_ids", None)
            if response_token_ids is not None:
                response_length = len(response_token_ids)
                self.particles[particle_idx].metadata["response_length"] = response_length
                self.particles[particle_idx].metadata["response_clipped"] = (
                    response_length >= self.config.max_tokens_per_step
                )
            else:
                self.particles[particle_idx].metadata["response_clipped"] = (
                    output.finish_reason == "length"
                )

            # Track per-particle reasoning step count
            reasoning_step_count = self._count_reasoning_steps(self.particles[particle_idx].text)
            self.particles[particle_idx].metadata["reasoning_step_count"] = reasoning_step_count

            # Check if particle is finished (boxed answer, max chars, etc.)
            if self._is_finished(self.particles[particle_idx]):
                self.particles[particle_idx].metadata["finished"] = True

        self._last_generation_progressed = any_progress
        return any(not p.metadata.get("finished") for p in self.particles)

    def _expand_token_chunk(self, active_indices: List[int]) -> bool:
        """
        Generate a fixed token chunk for active particles (token-interval mode).

        Returns:
            True if any new text was generated, False otherwise.
        """
        if not active_indices:
            self._last_generation_progressed = False
            return False

        prompt_texts, prompt_inputs = self._prepare_prompt_inputs(
            active_indices, max_new_tokens=self.config.resample_every_tokens
        )

        sampling_params = self._build_sampling_params(
            max_tokens=self.config.resample_every_tokens,
            stop=None,
        )

        outputs = self.generator.generate(prompt_inputs, sampling_params)

        any_progress = False

        for offset, particle_idx in enumerate(active_indices):
            output = outputs[offset].outputs[0]
            continuation = output.text
            if continuation:
                any_progress = True

            self.particles[particle_idx].text = prompt_texts[offset] + continuation

            if continuation:
                steps = self.particles[particle_idx].metadata.setdefault("prm_steps", [])
                steps.append(continuation)
                self.particles[particle_idx].metadata["token_chunk_count"] = len(steps)

            if output.finish_reason == "length":
                self.particles[particle_idx].metadata["finish_reason"] = "length"
                self.particles[particle_idx].metadata["truncated_chunk"] = True
            else:
                self.particles[particle_idx].metadata["finished"] = True
                self.particles[particle_idx].metadata["finish_reason"] = output.finish_reason

            response_token_ids = getattr(output, "token_ids", None)
            if response_token_ids is not None:
                response_length = len(response_token_ids)
                self.particles[particle_idx].metadata["response_length"] = response_length
                self.particles[particle_idx].metadata["response_clipped"] = (
                    response_length >= self.config.resample_every_tokens
                )
            else:
                self.particles[particle_idx].metadata["response_clipped"] = (
                    output.finish_reason == "length"
                )

            # Check if particle is finished (boxed answer, max chars, etc.)
            if self._is_finished(self.particles[particle_idx]):
                self.particles[particle_idx].metadata["finished"] = True

        self._last_generation_progressed = any_progress
        return any_progress

    def inject_next_step_headers(self) -> None:
        """
        Inject next step headers for particles that need them.

        Called AFTER scoring to ensure PRM scores actual content, not empty headers.
        """
        for particle in self.particles:
            if particle.metadata.get("finished"):
                particle.metadata.pop("needs_step_header", None)
                continue

            if particle.metadata.pop("needs_step_header", False):
                next_step = self._get_next_step_number(particle.text)
                if next_step > self.config.max_reasoning_steps:
                    particle.metadata["finished"] = True
                    particle.metadata["max_reasoning_steps_reached"] = True
                    continue

                particle.text += f"\n\n## Step {next_step}:"
                logger.debug(
                    f"Injected Step {next_step} header after scoring"
                )
    
    def score_particles(self) -> torch.Tensor:
        """
        Score all particles using PRM on latest step.

        Formats each particle's text with separator tokens and
        extracts the score for the most recent step.

        Optimization: Only scores non-finished particles. Finished particles
        retain their last score (stored in metadata) or get a default score.

        Returns:
            Tensor of scores, shape (n_particles,), values in [0, 1],
            on the same device as the particle filter weights.
        """
        n = len(self.particles)
        device = self._device

        # Identify active (non-finished) particles
        active_indices = []
        active_texts = []
        for i, particle in enumerate(self.particles):
            if not particle.metadata.get("finished"):
                active_indices.append(i)
                if self.config.resampling_unit == "token":
                    steps = particle.metadata.get("prm_steps", [])
                    if steps:
                        steps_for_prm = list(steps)
                        preamble = particle.metadata.get("prm_preamble", "")
                        if preamble:
                            steps_for_prm[0] = preamble + steps_for_prm[0]
                        formatted = self.reward_model.format_for_prm(steps_for_prm)
                    else:
                        formatted = particle.text + self.reward_model.prm_config.step_separator_token
                else:
                    formatted = self.reward_model.format_text_for_scoring(particle.text)
                active_texts.append(formatted)

        # If no active particles, return cached scores or ones
        if not active_texts:
            scores = torch.ones(n, device=device)
            for i, particle in enumerate(self.particles):
                scores[i] = particle.metadata.get("last_prm_score", 1.0)
            return scores

        # Score only active particles
        active_scores = self.reward_model.get_latest_step_scores(
            active_texts, device=str(device)
        )

        # Build full score tensor, using cached scores for finished particles
        scores = torch.ones(n, device=device)
        for i, particle in enumerate(self.particles):
            if particle.metadata.get("finished"):
                # Use last known score or 1.0 (neutral for multiplication)
                scores[i] = particle.metadata.get("last_prm_score", 1.0)

        # Fill in active particle scores
        for idx, active_idx in enumerate(active_indices):
            score = active_scores[idx].item()
            scores[active_idx] = score
            # Cache the score for future reference
            self.particles[active_idx].metadata["last_prm_score"] = score

        logger.debug(
            f"Scored {len(active_texts)}/{n} active particles, "
            f"{n - len(active_texts)} finished (using cached scores)"
        )

        return scores
    
    def select_best_by_orm(self) -> Particle:
        """
        Select the best particle using ORM (Outcome Reward Model) scores.
        
        Scores all particles using the ORM format (single reward token at end)
        and returns the particle with the highest score.
        
        Returns:
            Particle with highest ORM score
        """
        texts = self.get_particle_texts()
        orm_scores = self.reward_model.score_orm(texts)  # Returns List[float]
        orm_scores_tensor = torch.tensor(orm_scores)
        best_idx = int(torch.argmax(orm_scores_tensor))
        
        # Store ORM scores in metadata for logging
        for i, score in enumerate(orm_scores):
            self.particles[i].metadata["orm_score"] = score
        
        return self.particles[best_idx]

    def _step_by_token_chunk(self) -> bool:
        """
        SMC step for token-interval mode: generate a fixed chunk, score, resample.

        Returns:
            True if there are particles still generating, False otherwise
        """
        import warnings

        active_indices = self._get_active_indices()
        if not active_indices:
            return False

        self._expand_token_chunk(active_indices)

        step_scores = self.score_particles()

        SCORE_EPSILON = 1e-8
        active_indices = self._get_active_indices()
        if active_indices:
            active_scores = step_scores[active_indices]
        else:
            active_scores = step_scores

        if active_scores.numel() > 0 and torch.all(active_scores < SCORE_EPSILON):
            warnings.warn(
                f"All PRM scores are near-zero (< {SCORE_EPSILON}) at SMC iteration "
                f"{self._smc_iteration}. This indicates all particles are considered "
                "completely wrong by the PRM. Possible causes: degenerate particles, "
                "PRM misbehavior, or formatting issues. "
                "Terminating SMC early - check particle content for debugging.",
                UserWarning,
                stacklevel=2
            )
            for p in self.particles:
                if not p.metadata.get("finished"):
                    p.metadata["finished"] = True
                p.metadata["degenerate_termination"] = True
            return False

        current_weights = self.get_weights()
        active_indices = self._get_active_indices()
        if active_indices:
            active_weights = current_weights[active_indices] * step_scores[active_indices]
        else:
            active_weights = current_weights * step_scores

        if active_weights.numel() > 0:
            ess = compute_ess(active_weights)
            logger.debug(
                f"SMC iteration {self._smc_iteration}: ESS={ess:.2f} "
                f"(threshold={self.config.ess_threshold * len(active_indices):.2f})"
            )
        else:
            ess = None

        resampled = False
        if self.config.resampling_strategy == "every_step":
            self._resample_active(active_indices, active_weights)
            resampled = True
        elif self.config.resampling_strategy == "ess_adaptive":
            if active_weights.numel() > 0:
                if ess is not None and ess < self.config.ess_threshold * len(active_indices):
                    self._resample_active(active_indices, active_weights)
                    resampled = True
                else:
                    new_weights = current_weights.clone()
                    for idx, weight in zip(active_indices, active_weights):
                        new_weights[idx] = weight
                    self.set_weights(new_weights)
                    self.normalize_weights()

        if resampled:
            logger.debug(f"SMC iteration {self._smc_iteration}: Resampled particles")

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
        Single SMC step: expand, score with PRM, inject headers, resample.

        Weight Update Strategy (Standard SMC Steering):
        Uses multiplicative weights: w_t = w_{t-1} × PRM(step_t)

        Important: Resampling resets weights to uniform (1/N). This is correct
        SMC behavior - the weight information is "consumed" by resampling and
        converted into particle multiplicity (high-weight particles are duplicated).

        With resampling_strategy="every_step" (default):
            - w_{t-1} is always uniform (1/N) from the previous resample
            - Effectively: w_t ∝ PRM(step_t) each step
            - Multiplicative accumulation happens implicitly through particle
              lineages (good particles survive and multiply across steps)

        With resampling_strategy="ess_adaptive":
            - Weights accumulate multiplicatively between resampling events
            - Only resets to uniform when ESS drops below threshold

        This differs from Twisted SMC which uses value function ratios:
            w_t ∝ ψ_t(s_{1:t}) / ψ_{t-1}(s_{1:t-1})
        Where ψ is the twist/value function estimate.

        Order of operations:
        1. expand_particles() - generate content until step boundary (no headers injected yet)
        2. score_particles() - score actual content
        3. resample() - based on scores (active particles only)
        4. inject_next_step_headers() - add headers AFTER resampling

        Returns:
            True if there are particles still generating, False if all finished
        """
        import warnings

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

            # Fallback: if no progress or too many chunks, force a step boundary.
            if (not progressed) or (chunk_calls >= self.config.max_step_chunk_calls):
                if not progressed:
                    warnings.warn(
                        "No generation progress detected while completing a step. "
                        "Forcing a step boundary to avoid an infinite loop. "
                        "Check stop strings and prompt format.",
                        UserWarning,
                        stacklevel=2,
                    )
                for idx in pending_indices:
                    self.particles[idx].metadata["needs_step_header"] = True
                    self.particles[idx].metadata["forced_step_boundary"] = True
                break

        # Get PRM scores for the latest step BEFORE injecting next headers
        step_scores = self.score_particles()

        # Log step scores for debugging
        logger.debug(
            f"SMC iteration {self._smc_iteration}: "
            f"step_scores min={step_scores.min():.4f}, "
            f"max={step_scores.max():.4f}, "
            f"mean={step_scores.mean():.4f}"
        )

        # Check for degenerate case: all PRM scores are effectively zero
        # Use a small tolerance for floating-point comparison
        # This means the PRM thinks ALL particles are completely wrong.
        # Continuing with uniform weights would be random - better to warn and stop.
        SCORE_EPSILON = 1e-8
        active_indices = self._get_active_indices()
        if active_indices:
            active_scores = step_scores[active_indices]
        else:
            active_scores = step_scores

        if active_scores.numel() > 0 and torch.all(active_scores < SCORE_EPSILON):
            warnings.warn(
                f"All PRM scores are near-zero (< {SCORE_EPSILON}) at SMC iteration "
                f"{self._smc_iteration}. This indicates all particles are considered "
                "completely wrong by the PRM. Possible causes: degenerate particles, "
                "PRM misbehavior, or formatting issues. "
                "Terminating SMC early - check particle content for debugging.",
                UserWarning,
                stacklevel=2
            )
            # Mark all particles as finished and return False to terminate
            for p in self.particles:
                if not p.metadata.get("finished"):
                    p.metadata["finished"] = True
                p.metadata["degenerate_termination"] = True
            return False

        # MULTIPLICATIVE weight update on active particles only: w_t = w_{t-1} × PRM(step_t)
        current_weights = self.get_weights()
        active_indices = self._get_active_indices()
        if active_indices:
            active_weights = current_weights[active_indices] * step_scores[active_indices]
        else:
            active_weights = current_weights * step_scores

        # Compute ESS on active subset for logging/decision
        if active_weights.numel() > 0:
            ess = compute_ess(active_weights)
            logger.debug(
                f"SMC iteration {self._smc_iteration}: ESS={ess:.2f} "
                f"(threshold={self.config.ess_threshold * len(active_indices):.2f})"
            )
        else:
            ess = None

        # Resample based on strategy (active particles only)
        resampled = False
        if self.config.resampling_strategy == "every_step":
            self._resample_active(active_indices, active_weights)
            resampled = True
        elif self.config.resampling_strategy == "ess_adaptive":
            if active_weights.numel() > 0:
                if ess is not None and ess < self.config.ess_threshold * len(active_indices):
                    self._resample_active(active_indices, active_weights)
                    resampled = True
                else:
                    new_weights = current_weights.clone()
                    for idx, weight in zip(active_indices, active_weights):
                        new_weights[idx] = weight
                    self.set_weights(new_weights)
                    self.normalize_weights()

        if resampled:
            logger.debug(f"SMC iteration {self._smc_iteration}: Resampled particles")

        # Inject next step headers AFTER scoring and resampling
        self.inject_next_step_headers()

        # Completed one SMC step
        self._smc_iteration += 1

        if self._smc_iteration >= self.config.max_smc_iterations:
            for p in self.particles:
                if not p.metadata.get("finished"):
                    p.metadata["finished"] = True
                    p.metadata["max_smc_iterations_reached"] = True
            return False

        # Continue if there are still active particles
        return any(not p.metadata.get("finished") for p in self.particles)
    
    def run(self) -> Particle:
        """
        Run full SMC loop, return ORM-selected best particle.
        
        Runs step() until all particles finish or max iterations reached,
        then uses ORM to select the best final answer.
        
        Returns:
            Best particle (by ORM score if use_orm_for_final=True, else by weight)
        """
        while self.step():
            pass
        
        # Final selection using ORM
        if self.config.use_orm_for_final:
            return self.select_best_by_orm()
        else:
            return self.get_best_particle()
    
    def get_summary(self) -> dict:
        """
        Get summary statistics about the current particle filter state.
        
        Returns:
            Dictionary with summary statistics
        """
        active_indices = self._get_active_indices()
        active_ess = None
        if self._weights is not None and active_indices:
            active_ess = compute_ess(self._weights[active_indices])

        return {
            "smc_iteration": self._smc_iteration,
            "n_particles": self.n_particles,
            "ess": active_ess,
            "n_finished": sum(1 for p in self.particles if p.metadata.get("finished")),
            "avg_step_count": sum(
                p.metadata.get("reasoning_step_count", 0) for p in self.particles
            ) / max(1, len(self.particles)),
        }
