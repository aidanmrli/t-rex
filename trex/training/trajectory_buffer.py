"""
Trajectory buffer utilities for value head training.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import random


@dataclass
class Trajectory:
    """Single reasoning trajectory with reward."""

    prompt: str
    steps: List[str]
    full_text: str
    reward: float
    step_token_indices: Optional[List[int]] = None

    def get_state_reward_pairs(self) -> List[Tuple[str, float]]:
        """Return (partial_trace, reward) for each step."""
        pairs: List[Tuple[str, float]] = []
        for i in range(len(self.steps)):
            partial = self.prompt + "".join(self.steps[: i + 1])
            pairs.append((partial, float(self.reward)))
        return pairs

    def get_state_token_indices(self) -> List[int]:
        """Return token indices for step boundaries if available."""
        return list(self.step_token_indices or [])


class TrajectoryBuffer:
    """Buffer for storing and sampling trajectories."""

    def __init__(self, max_size: int = 10000):
        self.trajectories: Deque[Trajectory] = deque(maxlen=max_size)

    def add(self, trajectory: Trajectory) -> None:
        self.trajectories.append(trajectory)

    def sample(self, batch_size: int) -> List[Trajectory]:
        if not self.trajectories:
            return []
        return random.sample(self.trajectories, min(batch_size, len(self.trajectories)))

    def get_all_state_reward_pairs(self) -> List[Tuple[str, float]]:
        """Flatten all trajectories into (state, reward) pairs."""
        pairs: List[Tuple[str, float]] = []
        for traj in self.trajectories:
            pairs.extend(traj.get_state_reward_pairs())
        return pairs
