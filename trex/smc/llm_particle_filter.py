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

import re
from typing import List, Optional, TYPE_CHECKING

import torch

from trex.smc.particle_filter import ParticleFilter, Particle, SMCConfig

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
    
    @property
    def smc_iteration(self) -> int:
        """Current SMC iteration count."""
        return self._smc_iteration
    
    def _get_next_step_number(self, text: str) -> int:
        """
        Get the next step number to generate based on existing text.
        
        Counts existing "## Step N:" patterns and returns N+1.
        
        Args:
            text: Current particle text
            
        Returns:
            Next step number (1 if no steps exist)
        """
        matches = self.STEP_HEADER_PATTERN.findall(text)
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
        next_step = self._get_next_step_number(text)
        stripped = text.rstrip()
        
        # Check if the last line contains a step header pattern
        # This is more robust than checking just the last N characters
        last_line = stripped.split('\n')[-1] if '\n' in stripped else stripped
        
        # Only inject if the last line doesn't already have a step header
        if not self.STEP_PATTERN.search(last_line):
            return text + f"\n\n## Step {next_step}:"
        return text
    
    def _count_reasoning_steps(self, text: str) -> int:
        """
        Count the number of reasoning steps in text.
        
        Args:
            text: Text to count steps in
            
        Returns:
            Number of "## Step N:" patterns found
        """
        return len(self.STEP_HEADER_PATTERN.findall(text))
    
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
        if step_count >= self.config.max_steps:
            return True
        
        return False
    
    def expand_particles(self) -> bool:
        """
        Generate next step for all particles until '## Step N:' or EOS.
        
        Strategy for correct step numbering:
        1. Stop generation when "## Step" is encountered
        2. INCLUDE the stop string in output (include_stop_str_in_output=True)
        3. Track step count per-particle and inject header if needed
        
        This ensures the LLM sees the complete "## Step N:" pattern in context
        for the next generation, maintaining correct step numbering.
        
        Returns:
            True if there are particles still generating, False if all finished
        """
        from vllm import SamplingParams
        
        texts = self.get_particle_texts()
        
        # Prepare prompts: inject step header if needed for particles mid-generation
        prompts = []
        active_indices = []  # Track which particles to generate for
        
        for i, text in enumerate(texts):
            if self.particles[i].metadata.get("finished"):
                # Don't generate for finished particles, but keep their text
                prompts.append(text)
            else:
                active_indices.append(i)
                if self._smc_iteration > 0:
                    # After first expansion, we may need to inject step header
                    # Only inject if text doesn't end with step pattern
                    if not self.STEP_PATTERN.search(text.rstrip()[-20:] if len(text) >= 20 else text):
                        prompts.append(self._inject_step_header(text))
                    else:
                        prompts.append(text)
                else:
                    prompts.append(text)
        
        if not active_indices:
            # All particles finished
            return False
        
        # Generate until next step marker or end
        # INCLUDE stop string in output so LLM sees complete context
        sampling_params = SamplingParams(
            n=1,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_per_step,
            stop=["## Step"],  # Stop when next step begins
            include_stop_str_in_output=True,  # Include "## Step" in output
        )
        
        # Only generate for active particles
        active_prompts = [prompts[i] for i in active_indices]
        outputs = self.generator.generate(active_prompts, sampling_params)
        
        # Map outputs back to particles
        output_idx = 0
        still_generating = False
        
        for i in range(len(self.particles)):
            if self.particles[i].metadata.get("finished"):
                continue
            
            if output_idx < len(outputs):
                continuation = outputs[output_idx].outputs[0].text
                self.particles[i].text = prompts[i] + continuation
                output_idx += 1
            
            # Track per-particle reasoning step count
            reasoning_step_count = self._count_reasoning_steps(self.particles[i].text)
            self.particles[i].metadata["reasoning_step_count"] = reasoning_step_count
            
            # Check if particle is finished
            if self._is_finished(self.particles[i]):
                self.particles[i].metadata["finished"] = True
            else:
                still_generating = True
        
        self._smc_iteration += 1
        return still_generating and self._smc_iteration < self.config.max_steps
    
    def score_particles(self) -> torch.Tensor:
        """
        Score all particles using PRM on latest step.
        
        Formats each particle's text with separator tokens and
        extracts the score for the most recent step.
        
        Returns:
            Tensor of scores, shape (n_particles,), values in [0, 1]
        """
        texts = self.get_particle_texts()
        
        # Format texts for PRM scoring
        formatted_texts = []
        for text in texts:
            formatted = self.reward_model.format_text_for_scoring(text)
            formatted_texts.append(formatted)
        
        return self.reward_model.get_latest_step_scores(formatted_texts)
    
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
    
    def step(self) -> bool:
        """
        Single SMC step: expand, score with PRM, resample.
        
        Weight Update Strategy (Standard SMC Steering):
        Uses MULTIPLICATIVE incremental weights: w_t = w_{t-1} × PRM(step_t)
        
        This differs from Twisted SMC which uses value ratios:
            w_t(s_{1:t}) = [p_0(s_t|s_{1:t-1}) / q(s_t|s_{1:t-1})] × [ψ_t(s_{1:t}) / ψ_{t-1}(s_{1:t-1})]
        Where ψ is the twist function (value estimate). When q = p_0, the
        likelihood ratio cancels, leaving only the twist ratio.
        
        Returns:
            True if there are particles still generating, False if all finished
        """
        import warnings
        
        # Expand particles (generate next step)
        should_continue = self.expand_particles()
        
        # Get PRM scores for the latest step
        step_scores = self.score_particles()
        
        # Check for degenerate case: all PRM scores are zero
        # This means the PRM thinks ALL particles are completely wrong.
        # Continuing with uniform weights would be random - better to warn and stop.
        if torch.all(step_scores == 0):
            warnings.warn(
                f"All PRM scores are zero at SMC iteration {self._smc_iteration}. "
                "This indicates all particles are considered completely wrong by the PRM. "
                "Possible causes: degenerate particles, PRM misbehavior, or formatting issues. "
                "Terminating SMC early - check particle content for debugging.",
                UserWarning,
                stacklevel=2
            )
            # Mark all particles as finished and return False to terminate
            for p in self.particles:
                p.metadata["finished"] = True
                p.metadata["degenerate_termination"] = True
            return False
        
        # MULTIPLICATIVE weight update: w_t = w_{t-1} × PRM(step_t)
        current_weights = self.get_weights()
        new_weights = current_weights * step_scores
        
        self.set_weights(new_weights)
        self.normalize_weights()
        
        # Resample if needed, (end of reasoning step or ESS is low)
        if self.should_resample():
            self.resample()
        
        return should_continue
    
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
        return {
            "smc_iteration": self._smc_iteration,
            "n_particles": self.n_particles,
            "ess": self.effective_sample_size() if self._weights is not None else None,
            "n_finished": sum(1 for p in self.particles if p.metadata.get("finished")),
            "avg_step_count": sum(
                p.metadata.get("reasoning_step_count", 0) for p in self.particles
            ) / max(1, len(self.particles)),
        }
