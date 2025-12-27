"""
T-REX Baselines Package

This package contains baseline implementations for comparison with T-REX methods.
"""

from .config import BaselineConfig
from .grpo_config import GRPOConfig
from .math_reward_func import compute_score
from .grpo_reward_func import reward_func as grpo_reward_func

__all__ = [
    "BaselineConfig",
    "GRPOConfig",
    "compute_score",
    "grpo_reward_func",
]

