"""
Models module for T-REX.

This module provides model wrappers for:
- RewardModel: Process/Outcome Reward Model (PRM/ORM) wrapper
- PRMConfig: Configuration for different PRM models
"""

from trex.models.prm_config import PRMConfig, QWEN_PRM_CONFIG
from trex.models.reward_model import RewardModel

__all__ = [
    "PRMConfig",
    "QWEN_PRM_CONFIG",
    "RewardModel",
]
