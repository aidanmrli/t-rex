"""
Models module for T-REX.

This module provides model wrappers for:
- RewardModel: Process/Outcome Reward Model (PRM/ORM) wrapper
- PRMConfig: Configuration for different PRM models
"""

from trex.models.prm_config import PRMConfig, QWEN_PRM_CONFIG
from trex.models.reward_model import RewardModel
from trex.models.twist_model import TwistModel
from trex.models.value_head import (
    AttentionPooledValueHead,
    LinearValueHead,
    MLPValueHead,
    ValueHead,
)

__all__ = [
    "PRMConfig",
    "QWEN_PRM_CONFIG",
    "RewardModel",
    "TwistModel",
    "ValueHead",
    "LinearValueHead",
    "MLPValueHead",
    "AttentionPooledValueHead",
]
