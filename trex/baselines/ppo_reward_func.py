"""PPO reward function - reuses GRPO reward computation with PPO method tracking.

The reward computation is identical between PPO and GRPO. The difference is in
advantage estimation (GAE with critic vs group_norm), which is handled by OpenRLHF.
This module simply sets TREX_METHOD="ppo" for efficiency tracking purposes.
"""
import os
os.environ.setdefault("TREX_METHOD", "ppo")

from trex.baselines.grpo_reward_func import (
    reward_func,
    get_tracker,
    get_verifier,
)

__all__ = ["reward_func", "get_tracker", "get_verifier"]

if __name__ == "__main__":
    # Test that reward function works
    os.environ["TREX_METHOD"] = "ppo"
    os.environ["TREX_MODEL"] = "test"
    os.environ["TREX_DATASET"] = "test"
    os.environ["TREX_EFFICIENCY_PATH"] = "/tmp/test_ppo.json"

    result = reward_func(
        ["What is 2+2? \\boxed{4}"],
        ["What is 2+2?"],
        ["4"]
    )
    print(f"Rewards: {result['rewards'].tolist()}")
    assert result['rewards'][0] == 1.0
    print("Test passed!")
