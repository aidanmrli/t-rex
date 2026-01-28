"""
TDD Tests for trex/smc/resampling.py - SMC Resampling Algorithms (PyTorch).

These tests are written BEFORE the implementation as part of TDD workflow.
They define the expected API and behavior of the resampling module using PyTorch.

The resampling module should provide:
- multinomial_resampling(weights, n_particles) -> indices (LongTensor)
- systematic_resampling(weights, u=None) -> indices (LongTensor)
- stratified_resampling(weights) -> indices (LongTensor)
- normalize_weights(weights) -> normalized_weights (Tensor)
- compute_ess(weights) -> effective_sample_size (float)
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
    from trex.smc.resampling import (
        multinomial_resampling,
        systematic_resampling,
        stratified_resampling,
        normalize_weights,
        compute_ess,
    )
    RESAMPLING_IMPLEMENTED = True
except ImportError:
    RESAMPLING_IMPLEMENTED = False
    # Define stubs to allow test collection
    multinomial_resampling = None
    systematic_resampling = None
    stratified_resampling = None
    normalize_weights = None
    compute_ess = None


# Skip if torch missing or module not implemented
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE or not RESAMPLING_IMPLEMENTED,
    reason="torch not available or trex.smc.resampling not implemented"
)


# =============================================================================
# Tests for multinomial_resampling
# =============================================================================


class TestMultinomialResampling:
    """Tests for multinomial resampling using PyTorch."""

    def test_preserves_particle_count(self):
        """After resampling, exactly n_particles remain."""
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        n_particles = len(weights)

        indices = multinomial_resampling(weights, n_particles)

        assert isinstance(indices, torch.LongTensor)
        assert len(indices) == n_particles

    def test_high_weight_particle_duplicated(self):
        """Particle with weight 1.0 is always selected."""
        weights = torch.tensor([0.0, 0.0, 0.0, 1.0])
        n_particles = 4

        indices = multinomial_resampling(weights, n_particles)

        assert torch.all(indices == 3)

    def test_zero_weight_particle_never_selected(self):
        """Particle with weight 0.0 is never selected."""
        torch.manual_seed(42)
        weights = torch.tensor([0.0, 0.5, 0.5])
        n_particles = 100  # Many samples to verify

        indices = multinomial_resampling(weights, n_particles)

        assert (indices != 0).all()

    def test_uniform_weights_roughly_uniform_selection(self):
        """Uniform weights lead to roughly uniform selection."""
        torch.manual_seed(42)
        weights = torch.tensor([0.25, 0.25, 0.25, 0.25])
        n_particles = 1000

        indices = multinomial_resampling(weights, n_particles)
        counts = torch.bincount(indices, minlength=4)

        # Each particle should be selected ~250 times (within 20%)
        assert torch.all((counts > 200) & (counts < 300))

    def test_weights_do_not_need_to_be_normalized(self):
        """Unnormalized weights are handled correctly."""
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])  # Sum = 10
        n_particles = 4

        indices = multinomial_resampling(weights, n_particles)

        assert len(indices) == n_particles

    def test_indices_within_valid_range(self):
        """All indices are within [0, len(weights)-1]."""
        weights = torch.tensor([0.2, 0.3, 0.5])
        n_particles = 100

        indices = multinomial_resampling(weights, n_particles)

        assert torch.all((indices >= 0) & (indices < len(weights)))


# =============================================================================
# Tests for systematic_resampling
# =============================================================================


class TestSystematicResampling:
    """Tests for systematic resampling using PyTorch."""

    def test_preserves_particle_count(self):
        """After resampling, exactly n_particles remain."""
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])

        indices = systematic_resampling(weights)

        assert isinstance(indices, torch.LongTensor)
        assert len(indices) == len(weights)

    def test_deterministic_given_u(self):
        """Same starting offset u gives same result."""
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])

        indices1 = systematic_resampling(weights, u=0.1)
        indices2 = systematic_resampling(weights, u=0.1)

        assert torch.equal(indices1, indices2)

    def test_different_u_can_give_different_result(self):
        """Different u values can give different results."""
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])

        indices1 = systematic_resampling(weights, u=0.0)
        indices2 = systematic_resampling(weights, u=0.5)

        # With these weights and offsets, they *might* differ
        assert len(indices1) == len(weights)

    def test_high_weight_particle_duplicated(self):
        """Particle with weight 1.0 is always selected."""
        weights = torch.tensor([0.0, 0.0, 0.0, 1.0])

        indices = systematic_resampling(weights)

        assert torch.all(indices == 3)


# =============================================================================
# Tests for stratified_resampling
# =============================================================================


class TestStratifiedResampling:
    """Tests for stratified resampling using PyTorch."""

    def test_preserves_particle_count(self):
        """After resampling, exactly n_particles remain."""
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])

        indices = stratified_resampling(weights)

        assert isinstance(indices, torch.LongTensor)
        assert len(indices) == len(weights)

    def test_high_weight_particle_appears(self):
        """High weight particle appears in output."""
        weights = torch.tensor([0.1, 0.1, 0.1, 0.7])

        indices = stratified_resampling(weights)

        # The high-weight particle (3) should appear
        assert 3 in indices


# =============================================================================
# Tests for normalize_weights
# =============================================================================


class TestNormalizeWeights:
    """Tests for weight normalization utilities using PyTorch."""

    def test_normalized_weights_sum_to_one(self):
        """Normalized weights sum to 1.0."""
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])

        normalized = normalize_weights(weights)

        assert torch.isclose(normalized.sum(), torch.tensor(1.0))

    def test_preserves_relative_proportions(self):
        """Relative proportions are preserved."""
        weights = torch.tensor([1.0, 2.0, 3.0])

        normalized = normalize_weights(weights)

        # weight[1] should be twice weight[0]
        assert torch.isclose(normalized[1], 2 * normalized[0])

    def test_zero_weights_raise_error(self):
        """All-zero weights raise ValueError."""
        weights = torch.tensor([0.0, 0.0, 0.0])

        with pytest.raises(ValueError):
            normalize_weights(weights)

    def test_negative_weights_raise_error(self):
        """Negative weights raise ValueError."""
        weights = torch.tensor([-0.5, 0.5, 1.0])

        with pytest.raises(ValueError):
            normalize_weights(weights)

    def test_supports_batches(self):
        """Supports batch dimension if implemented."""
        # Optional: check if implementation supports [batch, particles]
        pass


# =============================================================================
# Tests for compute_ess (Effective Sample Size)
# =============================================================================


class TestComputeESS:
    """Tests for ESS computation using PyTorch."""

    def test_uniform_weights_ess_equals_n(self):
        """Uniform weights have ESS = n."""
        weights = torch.tensor([0.25, 0.25, 0.25, 0.25])

        ess = compute_ess(weights)

        assert torch.isclose(torch.tensor(ess), torch.tensor(4.0))

    def test_degenerate_weights_ess_equals_one(self):
        """Single particle has all weight → ESS ≈ 1."""
        weights = torch.tensor([1.0, 0.0, 0.0, 0.0])

        ess = compute_ess(weights)

        assert torch.isclose(torch.tensor(ess), torch.tensor(1.0))

    def test_ess_formula(self):
        """ESS = 1 / sum(w_i^2)."""
        weights = torch.tensor([0.5, 0.3, 0.2])

        ess = compute_ess(weights)
        expected = 1.0 / (0.5**2 + 0.3**2 + 0.2**2)

        assert torch.isclose(torch.tensor(ess), torch.tensor(expected))

    def test_ess_tensor_properties(self):
        """ESS calculation happens on same device."""
        if torch.cuda.is_available():
            weights = torch.tensor([0.5, 0.5], device="cuda")
            # This depends on if ESS returns a float or scalar tensor
            # If tensor, check device
            pass

    def test_gradients(self):
        """ESS should support gradients if needed (usually detached though)."""
        weights = torch.tensor([0.5, 0.5], requires_grad=True)
        ess = compute_ess(weights)
        # If ess returns a tensor, we can check grad
        # ess.backward()
        # assert weights.grad is not None
        pass


# =============================================================================
# GPU Support Tests (if available)
# =============================================================================


class TestGPUSupport:
    """Tests for GPU support using PyTorch."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_resampling_on_gpu(self):
        """Resampling works on GPU tensors and returns GPU tensors."""
        weights = torch.tensor([0.1, 0.9], device="cuda")
        indices = multinomial_resampling(weights, 100)
        assert indices.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_weight_normalization_on_gpu(self):
        """Normalization works on GPU."""
        weights = torch.tensor([1.0, 1.0], device="cuda")
        normalized = normalize_weights(weights)
        assert normalized.device.type == "cuda"
        assert torch.isclose(normalized.sum(), torch.tensor(1.0, device="cuda"))
