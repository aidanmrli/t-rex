"""
T-REX Baselines Package

This package contains baseline implementations for comparison with T-REX methods.
"""

from .config import BaselineConfig
from .grpo_config import GRPOConfig
from .math_reward_func import compute_score
from .grpo_reward_func import reward_func as grpo_reward_func
from .smc_config import SMCSteeringConfig, CheckpointManager

# Lazy import to avoid jsonlines dependency during test collection
def __getattr__(name):
    if name == "SMCSteeringBaseline":
        from .smc_steering_baseline import SMCSteeringBaseline
        return SMCSteeringBaseline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaselineConfig",
    "GRPOConfig",
    "compute_score",
    "grpo_reward_func",
    "SMCSteeringConfig",
    "CheckpointManager",
    "SMCSteeringBaseline",
]
