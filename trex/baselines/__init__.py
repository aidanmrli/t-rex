"""
T-REX Baselines Package

This package contains baseline implementations for comparison with T-REX methods.
"""

from .config import BaselineConfig
from .math_reward_func import compute_score

__all__ = ["BaselineConfig", "compute_score"]
