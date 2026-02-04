"""
Configuration for SMC Steering Baseline.

This module provides:
- SMCSteeringConfig: Configuration dataclass for SMC steering parameters
- CheckpointManager: Manages checkpointing for SLURM preemptible environments
"""

import json
import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class SMCSteeringConfig:
    """Configuration for SMC Steering Baseline."""
    
    # Generator model (for generating reasoning steps)
    generator_model_path: str = "Qwen/Qwen2.5-7B"
    generator_tp_size: int = 1
    
    # Reward model (PRM/ORM - same model, different input format)
    reward_model_path: str = "Qwen/Qwen2.5-Math-PRM-7B"
    reward_model_tp_size: int = 1
    
    # Dataset settings
    dataset_path: str = "trex/data/gsm8k_platinum_test.jsonl"
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    
    # SMC parameters
    n_particles: int = 16
    max_smc_iterations: int = 20  # Maximum SMC loop iterations (expand → score → resample)
    max_reasoning_steps: int = 15  # Maximum "## Step N:" reasoning steps per particle
    step_pattern: str = r"## Step \d+:"  # Regex for step detection
    seed: Optional[int] = None  # Random seed for reproducibility
    
    # Resampling
    resampling_method: str = "systematic"  # "multinomial", "systematic", "stratified"
    # Resampling unit:
    # - "step": Resample after a completed reasoning step (## Step N:)
    # - "token": Resample after a fixed token chunk (no step template enforced)
    resampling_unit: str = "step"
    resample_every_tokens: int = 128  # Token chunk size when resampling_unit="token"
    # Resampling strategy:
    # - "every_step": Resample after every SMC step (default, standard for SMC steering).
    #   Weights are reset to uniform after each resample. PRM scores determine which
    #   particles survive each step; accumulation happens through particle lineages.
    # - "ess_adaptive": Only resample when ESS drops below threshold. Weights
    #   accumulate multiplicatively (w_t = w_{t-1} × PRM_t) between resampling events.
    resampling_strategy: str = "every_step"
    ess_threshold: float = 0.5  # Resample when ESS < threshold * n_particles (ess_adaptive only)
    
    # Generation parameters
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens_per_step: int = 512  # Max tokens per generation call
    # Max number of generation chunks allowed to complete a single reasoning step.
    # Acts as a fallback if the model doesn't emit the next "## Step N:" header.
    max_step_chunk_calls: int = 4
    # Prompt handling
    use_token_prompts: bool = False  # Use vLLM TokensPrompt with token IDs
    enable_prompt_truncation: bool = True  # Truncate prompts that exceed context
    prompt_max_tokens: Optional[int] = None  # Optional hard cap on prompt tokens
    # Max total characters per particle (not tokens - character counting is faster)
    # Rough heuristic: 1 token ≈ 4 chars, so 2048 chars ≈ 512 tokens
    max_total_chars: int = 8192  # ~2048 tokens worth of characters
    
    # Final selection
    use_orm_for_final: bool = True  # Use ORM (not PRM) for final answer selection
    
    # Checkpointing (SLURM support)
    enable_checkpointing: bool = True
    checkpoint_file: str = "checkpoint.json"
    checkpoint_interval: int = 5  # Save every N problems
    checkpoint_time_interval: int = 600  # Save every N seconds (10 min default)
    
    # Output settings
    output_dir: str = "trex/results/smc_baseline"
    use_wandb: bool = False
    wandb_project: str = "t-rex"
    wandb_run_name: Optional[str] = None
    
    # vLLM settings
    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.9
    
    # Chat template settings
    apply_chat_template: bool = True
    system_prompt: str = """Solve the following math problem efficiently and clearly: 
- For simple problems (2 steps or fewer): Provide a concise solution with minimal explanation.  
- For complex problems (3 steps or more): Use this step-by-step format:  

## Step 1: [Concise description] 
[Brief explanation and calculations]  

## Step 2: [Concise description] 
[Brief explanation and calculations]  

Regardless of the approach, always conclude with:  

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct. 

Where [answer] is just the final number or expression that solves the problem."""
    
    def __post_init__(self):
        """Set derived values and validate configuration."""
        if self.wandb_run_name is None:
            self.wandb_run_name = f"smc_steering_n{self.n_particles}"

        # Validate resampling method
        valid_methods = ("multinomial", "systematic", "stratified")
        if self.resampling_method not in valid_methods:
            raise ValueError(
                f"Invalid resampling_method: {self.resampling_method}. "
                f"Must be one of {valid_methods}"
            )

        # Validate resampling strategy
        valid_strategies = ("every_step", "ess_adaptive")
        if self.resampling_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid resampling_strategy: {self.resampling_strategy}. "
                f"Must be one of {valid_strategies}"
            )

        # Validate resampling unit
        valid_units = ("step", "token")
        if self.resampling_unit not in valid_units:
            raise ValueError(
                f"Invalid resampling_unit: {self.resampling_unit}. "
                f"Must be one of {valid_units}"
            )

        # Validate ESS threshold
        if not 0.0 < self.ess_threshold <= 1.0:
            raise ValueError(
                f"ess_threshold must be in (0, 1], got {self.ess_threshold}"
            )

        if self.max_step_chunk_calls < 1:
            raise ValueError(
                f"max_step_chunk_calls must be >= 1, got {self.max_step_chunk_calls}"
            )

        if self.resample_every_tokens < 1:
            raise ValueError(
                f"resample_every_tokens must be >= 1, got {self.resample_every_tokens}"
            )

        if self.prompt_max_tokens is not None and self.prompt_max_tokens < 1:
            raise ValueError(
                f"prompt_max_tokens must be >= 1, got {self.prompt_max_tokens}"
            )

        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SMCSteeringConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def config_hash(self) -> str:
        """Generate a hash of the configuration for validation."""
        # Use a subset of critical fields for hashing
        critical = {
            "generator_model_path": self.generator_model_path,
            "reward_model_path": self.reward_model_path,
            "n_particles": self.n_particles,
            "max_smc_iterations": self.max_smc_iterations,
            "max_reasoning_steps": self.max_reasoning_steps,
            "max_step_chunk_calls": self.max_step_chunk_calls,
            "use_token_prompts": self.use_token_prompts,
            "enable_prompt_truncation": self.enable_prompt_truncation,
            "prompt_max_tokens": self.prompt_max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "resampling_unit": self.resampling_unit,
            "resample_every_tokens": self.resample_every_tokens,
            "resampling_method": self.resampling_method,
            "resampling_strategy": self.resampling_strategy,
            "ess_threshold": self.ess_threshold,
            "seed": self.seed,
        }
        return hashlib.md5(json.dumps(critical, sort_keys=True).encode()).hexdigest()[:8]


class CheckpointManager:
    """
    Manages checkpointing for SLURM preemptible cluster environments.
    
    Design Decision: We only checkpoint BETWEEN problems, not mid-problem.
    This simplifies the checkpoint format and avoids storing large particle states.
    
    Attributes:
        checkpoint_path: Path to the checkpoint JSON file
        state: Current checkpoint state dictionary
    """
    
    def __init__(self, checkpoint_path: str, config_hash: Optional[str] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_path: Path to save/load checkpoint file
            config_hash: Optional hash of config for validation
        """
        self.checkpoint_path = checkpoint_path
        self.config_hash = config_hash
        self.state: Dict[str, Any] = {
            "completed_idx": 0,  # Number of problems completed
            "results": [],  # Per-problem results (compact)
            "config_hash": config_hash or "",
            "finished": False,
            "last_save_time": None,
        }
        self._load()
    
    def _load(self) -> None:
        """Load existing checkpoint if available."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r") as f:
                    saved_state = json.load(f)
                
                # Validate config hash if provided
                if self.config_hash and saved_state.get("config_hash"):
                    if saved_state["config_hash"] != self.config_hash:
                        print(f"Warning: Config hash mismatch. "
                              f"Checkpoint: {saved_state['config_hash']}, "
                              f"Current: {self.config_hash}")
                        print("Starting fresh (config changed).")
                        return
                
                self.state.update(saved_state)
                print(f"Loaded checkpoint from {self.checkpoint_path}")
                print(f"  Completed problems: {self.state['completed_idx']}")
                print(f"  Finished: {self.state['finished']}")
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load checkpoint ({e}), starting fresh.")
    
    def save(self) -> None:
        """
        Save current state to checkpoint file.
        
        Uses atomic write (temp file + rename) to prevent corruption.
        """
        self.state["last_save_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
        
        # Atomic write: write to temp file, then rename
        temp_path = self.checkpoint_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(self.state, f, indent=2)
        os.rename(temp_path, self.checkpoint_path)
        
        print(f"Checkpoint saved: completed_idx={self.state['completed_idx']}, "
              f"finished={self.state['finished']}")
    
    def should_save(self, idx: int, interval: int, time_interval: int, 
                    last_save_epoch: float) -> bool:
        """
        Check if we should save a checkpoint.
        
        Args:
            idx: Current problem index (1-indexed)
            interval: Problem count interval
            time_interval: Time interval in seconds
            last_save_epoch: Epoch time of last save
            
        Returns:
            True if we should save
        """
        # Problem-count based
        if idx % interval == 0:
            return True
        
        # Time-based
        if time.time() - last_save_epoch >= time_interval:
            return True
        
        return False
    
    def is_finished(self) -> bool:
        """Check if the experiment is already finished."""
        return self.state.get("finished", False)
    
    def mark_finished(self) -> None:
        """Mark the experiment as finished and save."""
        self.state["finished"] = True
        self.save()
    
    def get_resume_index(self) -> int:
        """Get the index to resume from."""
        return self.state.get("completed_idx", 0)
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get the saved results."""
        return self.state.get("results", [])
