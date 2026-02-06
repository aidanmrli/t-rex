"""
Configuration for Twisted SMC baseline.
"""

import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from trex.baselines.smc_config import SMCSteeringConfig, CheckpointManager


@dataclass
class TSMCConfig(SMCSteeringConfig):
    """Configuration for Twisted SMC baseline."""

    # Delimiter-first defaults for TSMC (header mode remains optional).
    # Keep chat templating off by default so delimiter behavior comes from data.
    apply_chat_template: bool = False
    step_boundary_mode: str = "delimiter"
    step_delimiter: str = "\n\n"
    system_prompt: str = (
        "Solve the following math problem step by step and conclude with "
        "Therefore, the final answer is: $\\boxed{answer}$."
    )

    # Value head / twist model
    value_model_path: str = "Qwen/Qwen2.5-7B"
    value_head_path: Optional[str] = None
    value_head_type: str = "mlp"
    twist_space: str = "log_prob"  # "log_prob" or "prob"
    twist_mode: str = "value"  # "value" or "sqrt_value"
    epsilon: float = 1e-8
    log_value_min: float = -1e6
    share_base_with_generator: bool = False

    # ORM for final selection (optional)
    reward_model_path: Optional[str] = "Qwen/Qwen2.5-Math-PRM-7B"
    reward_model_tp_size: int = 1
    use_orm_for_final: bool = False
    final_selection_mode: Optional[str] = None  # "orm", "majority_vote", "twist_weight"

    # Optional warm-up before enabling resampling.
    warmup_steps: int = 0
    warmup_tokens: int = 0  # token-resampling mode only

    def __post_init__(self):
        super().__post_init__()

        if self.twist_space not in ("log_prob", "prob"):
            raise ValueError("twist_space must be 'log_prob' or 'prob'")
        if self.twist_mode not in ("value", "sqrt_value"):
            raise ValueError("twist_mode must be 'value' or 'sqrt_value'")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.log_value_min >= 0:
            raise ValueError("log_value_min must be negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.warmup_tokens < 0:
            raise ValueError("warmup_tokens must be >= 0")

        if self.final_selection_mode is None:
            self.final_selection_mode = "orm" if self.use_orm_for_final else "twist_weight"
        valid_modes = ("orm", "majority_vote", "twist_weight")
        if self.final_selection_mode not in valid_modes:
            raise ValueError(f"final_selection_mode must be one of {valid_modes}")
        # Keep legacy flag synchronized for compatibility with existing code paths.
        self.use_orm_for_final = self.final_selection_mode == "orm"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TSMCConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def config_hash(self) -> str:
        critical = {
            "generator_model_path": self.generator_model_path,
            "value_model_path": self.value_model_path,
            "value_head_path": self.value_head_path,
            "value_head_type": self.value_head_type,
            "twist_space": self.twist_space,
            "twist_mode": self.twist_mode,
            "n_particles": self.n_particles,
            "max_smc_iterations": self.max_smc_iterations,
            "max_reasoning_steps": self.max_reasoning_steps,
            "step_boundary_mode": self.step_boundary_mode,
            "step_pattern": self.step_pattern,
            "step_delimiter": self.step_delimiter,
            "resampling_unit": self.resampling_unit,
            "resample_every_tokens": self.resample_every_tokens,
            "resampling_method": self.resampling_method,
            "resampling_strategy": self.resampling_strategy,
            "ess_threshold": self.ess_threshold,
            "warmup_steps": self.warmup_steps,
            "warmup_tokens": self.warmup_tokens,
            "final_selection_mode": self.final_selection_mode,
            "seed": self.seed,
        }
        return hashlib.md5(json.dumps(critical, sort_keys=True).encode()).hexdigest()[:8]


__all__ = ["TSMCConfig", "CheckpointManager"]
