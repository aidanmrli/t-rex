"""
Unit tests for trajectory buffer utilities.
"""

from trex.training.trajectory_buffer import Trajectory, TrajectoryBuffer


def test_trajectory_state_reward_pairs():
    """Trajectory expands steps into (state, reward) pairs."""
    traj = Trajectory(
        prompt="Q: ",
        steps=["A", "B"],
        full_text="Q: A B",
        reward=1.0,
    )
    pairs = traj.get_state_reward_pairs()
    assert pairs == [("Q: A", 1.0), ("Q: AB", 1.0)]


def test_trajectory_buffer_add_and_sample():
    """TrajectoryBuffer stores trajectories and samples up to batch size."""
    buf = TrajectoryBuffer(max_size=3)
    buf.add(Trajectory(prompt="Q", steps=["A"], full_text="QA", reward=0.0))
    buf.add(Trajectory(prompt="Q", steps=["B"], full_text="QB", reward=1.0))
    sample = buf.sample(batch_size=5)
    assert len(sample) == 2
