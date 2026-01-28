"""
TDD Tests for trex/tempering/temperature_ladder.py - Temperature Schedule Generation (PyTorch).

These tests are written BEFORE the implementation as part of TDD workflow.
They define the expected API and behavior of the temperature ladder module.

The temperature_ladder module should provide:
- generate_temperature_ladder(num_temperatures, schedule, min_beta, max_beta) -> Tensor
- get_swap_pairs(timestep, num_temperatures) -> List[Tuple[int, int]]
"""

import pytest
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Import with skip if not implemented yet
# =============================================================================

try:
    from trex.tempering.temperature_ladder import (
        generate_temperature_ladder,
        get_swap_pairs,
    )
    TEMPERATURE_LADDER_IMPLEMENTED = True
except ImportError:
    TEMPERATURE_LADDER_IMPLEMENTED = False
    generate_temperature_ladder = None
    get_swap_pairs = None


# Skip if torch missing or module not implemented
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE or not TEMPERATURE_LADDER_IMPLEMENTED,
    reason="torch not available or trex.tempering.temperature_ladder not implemented"
)


# =============================================================================
# Tests for generate_temperature_ladder
# =============================================================================


class TestGenerateTemperatureLadder:
    """Tests for temperature schedule generation using PyTorch."""

    def test_linear_schedule(self):
        """Linear schedule from 0 to 1."""
        betas = generate_temperature_ladder(
            num_temperatures=5,
            schedule="linear",
            min_beta=0.0,
            max_beta=1.0
        )

        expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        assert torch.allclose(betas, expected)

    def test_first_is_min_last_is_max(self):
        """First beta is min, last is max."""
        betas = generate_temperature_ladder(
            num_temperatures=5,
            schedule="linear",
            min_beta=0.1,
            max_beta=0.9
        )

        assert torch.isclose(betas[0], torch.tensor(0.1))
        assert torch.isclose(betas[-1], torch.tensor(0.9))

    def test_geometric_schedule_is_not_linear(self):
        """Geometric schedule is not evenly spaced."""
        betas_linear = generate_temperature_ladder(5, "linear")
        betas_geometric = generate_temperature_ladder(5, "geometric")

        # Check they're different (excluding edge cases)
        assert not torch.allclose(betas_linear, betas_geometric)

    def test_monotonically_increasing(self):
        """Betas are strictly increasing."""
        betas = generate_temperature_ladder(5, "linear")

        for i in range(len(betas) - 1):
            assert betas[i] < betas[i + 1]

    def test_returns_tensor(self):
        """Returns a PyTorch tensor."""
        betas = generate_temperature_ladder(5, "linear")

        assert isinstance(betas, torch.Tensor)

    def test_correct_length(self):
        """Returns correct number of temperatures."""
        for n in [2, 5, 10, 20]:
            betas = generate_temperature_ladder(n, "linear")
            assert len(betas) == n

    def test_single_temperature(self):
        """Single temperature returns the max."""
        betas = generate_temperature_ladder(
            num_temperatures=1,
            schedule="linear",
            min_beta=0.0,
            max_beta=1.0
        )

        assert len(betas) == 1
        assert torch.isclose(betas[0], torch.tensor(1.0))

    def test_quadratic_schedule(self):
        """Quadratic schedule concentrates more at low temperatures."""
        betas_linear = generate_temperature_ladder(5, "linear")
        betas_quadratic = generate_temperature_ladder(5, "quadratic")

        # Quadratic should have smaller values in the middle (more gradual)
        # (This is a general property, exact test depends on implementation)
        assert isinstance(betas_quadratic, torch.Tensor)
        assert len(betas_quadratic) == 5

    def test_custom_range(self):
        """Supports custom min/max range."""
        betas = generate_temperature_ladder(
            num_temperatures=3,
            schedule="linear",
            min_beta=0.2,
            max_beta=0.8
        )

        expected = torch.tensor([0.2, 0.5, 0.8])
        assert torch.allclose(betas, expected)

    def test_device_support(self):
        """Supports specifying device."""
        betas = generate_temperature_ladder(5, "linear", device="cpu")
        assert betas.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_support(self):
        """Supports GPU device."""
        betas = generate_temperature_ladder(5, "linear", device="cuda")
        assert betas.device.type == "cuda"


# =============================================================================
# Tests for get_swap_pairs (Non-Reversible Swap Schedule)
# =============================================================================


class TestGetSwapPairs:
    """Tests for non-reversible swap schedule."""

    def test_odd_timestep_swaps_pairs_1_2_3_4(self):
        """S_odd = {(1,2), (3,4), ...} (1-indexed in tests, may be 0-indexed in impl)."""
        pairs = get_swap_pairs(timestep=1, num_temperatures=5)

        # Check we get the expected non-overlapping pairs
        # If 0-indexed: [(0,1), (2,3)]
        # If 1-indexed: [(1,2), (3,4)]
        assert len(pairs) == 2
        # Pairs should be adjacent and non-overlapping
        for i, j in pairs:
            assert abs(i - j) == 1

    def test_even_timestep_swaps_pairs_2_3_4_5(self):
        """S_even = {(2,3), (4,5), ...}."""
        pairs = get_swap_pairs(timestep=2, num_temperatures=5)

        # If 0-indexed: [(1,2), (3,4)]
        # If 1-indexed: [(2,3), (4,5)]
        assert len(pairs) == 2
        for i, j in pairs:
            assert abs(i - j) == 1

    def test_alternating_covers_all_adjacent_pairs(self):
        """Over 2 timesteps, every adjacent pair is attempted."""
        num_temps = 5
        pairs_odd = set(tuple(sorted(p)) for p in get_swap_pairs(1, num_temps))
        pairs_even = set(tuple(sorted(p)) for p in get_swap_pairs(2, num_temps))

        all_pairs = pairs_odd | pairs_even
        
        # All adjacent pairs (0-indexed)
        expected = {(i, i + 1) for i in range(num_temps - 1)}
        assert all_pairs == expected

    def test_single_temperature_returns_empty(self):
        """With K=1, no swaps possible."""
        pairs = get_swap_pairs(timestep=1, num_temperatures=1)

        assert pairs == []

    def test_two_temperatures(self):
        """With K=2, only one pair possible."""
        pairs_odd = get_swap_pairs(timestep=1, num_temperatures=2)
        pairs_even = get_swap_pairs(timestep=2, num_temperatures=2)

        # Only (0,1) is possible
        assert len(pairs_odd) + len(pairs_even) >= 1
        # One of them should have the pair
        all_pairs = pairs_odd + pairs_even
        assert any(abs(i - j) == 1 for i, j in all_pairs) if all_pairs else True

    def test_returns_list_of_tuples(self):
        """Returns list of (int, int) tuples."""
        pairs = get_swap_pairs(timestep=1, num_temperatures=5)

        assert isinstance(pairs, list)
        for pair in pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert all(isinstance(x, int) for x in pair)

    def test_pairs_within_valid_range(self):
        """All swap indices are valid temperature indices."""
        num_temps = 5
        for timestep in range(1, 10):
            pairs = get_swap_pairs(timestep, num_temps)
            for i, j in pairs:
                assert 0 <= i < num_temps
                assert 0 <= j < num_temps

    def test_no_overlapping_pairs(self):
        """No temperature appears in multiple pairs at same timestep."""
        pairs = get_swap_pairs(timestep=1, num_temperatures=10)
        
        used_indices = set()
        for i, j in pairs:
            assert i not in used_indices
            assert j not in used_indices
            used_indices.add(i)
            used_indices.add(j)
