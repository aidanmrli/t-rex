"""
TDD Tests for trex/smc/twisted_smc.py - Twisted SMC Implementation (PyTorch).

These tests are written BEFORE the implementation as part of TDD workflow.
They define the expected API and behavior of the Twisted SMC module using PyTorch.

Twisted SMC (TSMC) uses a learned value function (twist) to guide sampling
towards high-reward regions. The twist function modifies importance weights
to prefer particles that are likely to lead to good outcomes.

The twisted_smc module should provide:
- compute_twisted_weights(values_t, values_t_minus_1) -> weight_ratios (Tensor)
- TwistedSMC: Main class extending ParticleFilter with twisted proposal
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
    from trex.smc.twisted_smc import (
        compute_twisted_weights,
        TwistedSMC,
        TwistedSMCConfig,
    )
    TWISTED_SMC_IMPLEMENTED = True
except ImportError:
    TWISTED_SMC_IMPLEMENTED = False
    compute_twisted_weights = None
    TwistedSMC = None
    TwistedSMCConfig = None


# Skip if torch missing or module not implemented
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE or not TWISTED_SMC_IMPLEMENTED,
    reason="torch not available or trex.smc.twisted_smc not implemented"
)


# =============================================================================
# Tests for compute_twisted_weights
# =============================================================================


class TestComputeTwistedWeights:
    """Tests for twisted importance weight computation using PyTorch."""

    def test_weight_ratio_increases_when_value_improves(self):
        """w_t > 1 when V(t) > V(t-1)."""
        values_t = torch.tensor([0.9, 0.8, 0.7])
        values_t_minus_1 = torch.tensor([0.5, 0.5, 0.5])

        weights = compute_twisted_weights(values_t, values_t_minus_1)

        assert isinstance(weights, torch.Tensor)
        assert torch.all(weights > 1.0)

    def test_weight_ratio_decreases_when_value_drops(self):
        """w_t < 1 when V(t) < V(t-1)."""
        values_t = torch.tensor([0.3, 0.2, 0.1])
        values_t_minus_1 = torch.tensor([0.5, 0.5, 0.5])

        weights = compute_twisted_weights(values_t, values_t_minus_1)

        assert torch.all(weights < 1.0)

    def test_handles_zero_previous_value(self):
        """No division by zero when V(t-1) ≈ 0."""
        values_t = torch.tensor([0.5, 0.5])
        values_t_minus_1 = torch.tensor([0.0, 1e-10])

        # Should not raise, should use epsilon
        weights = compute_twisted_weights(values_t, values_t_minus_1)

        assert not torch.any(torch.isinf(weights))
        assert not torch.any(torch.isnan(weights))

    def test_unchanged_value_gives_weight_one(self):
        """w_t = 1 when V(t) = V(t-1)."""
        values_t = torch.tensor([0.5, 0.5, 0.5])
        values_t_minus_1 = torch.tensor([0.5, 0.5, 0.5])

        weights = compute_twisted_weights(values_t, values_t_minus_1)

        assert torch.allclose(weights, torch.ones_like(weights))

    def test_negative_values_raises_without_log_space(self):
        """Negative values raise error when log_space=False (default)."""
        # In probability space, values should be non-negative
        values_t = torch.tensor([-1.0, -2.0, -3.0])
        values_t_minus_1 = torch.tensor([-2.0, -2.0, -2.0])

        with pytest.raises(ValueError, match="log_space=True"):
            compute_twisted_weights(values_t, values_t_minus_1, log_space=False)

    def test_negative_values_handled_with_log_space(self):
        """Handles negative log-probabilities when log_space=True."""
        # In log-space, values can be negative
        values_t = torch.tensor([-1.0, -2.0, -3.0])
        values_t_minus_1 = torch.tensor([-2.0, -2.0, -2.0])

        # With log_space=True, should work correctly
        weights = compute_twisted_weights(values_t, values_t_minus_1, log_space=True)

        # The first value improved (-1 > -2), so weight > 1
        # exp(-1 - (-2)) = exp(1) ≈ 2.718
        assert weights[0] > 1.0
        # Others stayed same or got worse
        assert torch.isclose(weights[1], torch.tensor(1.0))  # exp(0) = 1
        assert weights[2] < 1.0  # exp(-1) ≈ 0.368
        assert not torch.any(torch.isnan(weights))

    def test_preserves_tensor_device(self):
        """Output tensor is on the same device as input."""
        values_t = torch.tensor([0.5, 0.5])
        values_t_minus_1 = torch.tensor([0.4, 0.4])

        weights = compute_twisted_weights(values_t, values_t_minus_1)

        assert weights.device == values_t.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_support(self):
        """Works on GPU tensors."""
        values_t = torch.tensor([0.5, 0.5], device="cuda")
        values_t_minus_1 = torch.tensor([0.4, 0.4], device="cuda")

        weights = compute_twisted_weights(values_t, values_t_minus_1)

        assert weights.device.type == "cuda"

    def test_gradient_flow(self):
        """Gradients can flow through the computation."""
        values_t = torch.tensor([0.9, 0.8], requires_grad=True)
        values_t_minus_1 = torch.tensor([0.5, 0.5], requires_grad=True)

        weights = compute_twisted_weights(values_t, values_t_minus_1)
        loss = weights.sum()
        loss.backward()

        assert values_t.grad is not None
        assert values_t_minus_1.grad is not None


# =============================================================================
# Tests for TwistedSMCConfig
# =============================================================================


class TestTwistedSMCConfig:
    """Tests for TwistedSMCConfig dataclass."""

    def test_inherits_smc_config_fields(self):
        """Has all fields from base SMCConfig."""
        config = TwistedSMCConfig()

        assert hasattr(config, "n_particles")
        assert hasattr(config, "ess_threshold")
        assert hasattr(config, "resampling_method")

    def test_has_twist_specific_fields(self):
        """Has twist-specific configuration options."""
        config = TwistedSMCConfig()

        # Should have fields for the twist function
        assert hasattr(config, "use_twist")
        assert hasattr(config, "epsilon")  # For numerical stability

    def test_default_epsilon(self):
        """Default epsilon is a small positive value."""
        config = TwistedSMCConfig()

        assert config.epsilon > 0
        assert config.epsilon < 1e-3


# =============================================================================
# Tests for TwistedSMC
# =============================================================================


class TestTwistedSMCInit:
    """Tests for TwistedSMC initialization."""

    def test_can_initialize(self):
        """TwistedSMC can be initialized with config."""
        config = TwistedSMCConfig(n_particles=8)
        tsmc = TwistedSMC(config)

        assert tsmc.config.n_particles == 8

    def test_inherits_from_particle_filter(self):
        """TwistedSMC extends ParticleFilter functionality."""
        config = TwistedSMCConfig(n_particles=8)
        tsmc = TwistedSMC(config)
        tsmc.initialize(prompt="Test prompt")

        # Should have all ParticleFilter methods
        assert len(tsmc.particles) == 8
        assert hasattr(tsmc, "get_weights")
        assert hasattr(tsmc, "resample")


class TestTwistedSMCWeightUpdate:
    """Tests for TwistedSMC weight update with twist function."""

    @pytest.fixture
    def tsmc(self):
        """Create an initialized TwistedSMC."""
        config = TwistedSMCConfig(n_particles=4, use_twist=True)
        tsmc = TwistedSMC(config)
        tsmc.initialize(prompt="What is 2+2?")
        return tsmc

    def test_update_weights_with_values(self, tsmc):
        """Can update weights using value estimates."""
        current_values = torch.tensor([0.9, 0.7, 0.5, 0.3])
        previous_values = torch.tensor([0.5, 0.5, 0.5, 0.5])

        tsmc.update_weights_with_twist(current_values, previous_values)

        weights = tsmc.get_weights()
        # First particle (highest value improvement) should have highest weight
        assert weights.argmax() == 0

    def test_update_normalizes_weights(self, tsmc):
        """Weight update results in normalized weights."""
        current_values = torch.tensor([0.9, 0.7, 0.5, 0.3])
        previous_values = torch.tensor([0.5, 0.5, 0.5, 0.5])

        tsmc.update_weights_with_twist(current_values, previous_values)

        weights = tsmc.get_weights()
        assert torch.isclose(weights.sum(), torch.tensor(1.0))

    def test_stores_value_history(self, tsmc):
        """Stores previous values for next update."""
        current_values = torch.tensor([0.9, 0.7, 0.5, 0.3])

        tsmc.set_current_values(current_values)

        assert tsmc.get_previous_values() is not None
        assert torch.allclose(tsmc.get_previous_values(), current_values)


class TestTwistedSMCStep:
    """Tests for TwistedSMC step-by-step execution."""

    @pytest.fixture
    def tsmc(self):
        """Create an initialized TwistedSMC."""
        config = TwistedSMCConfig(n_particles=4, use_twist=True, ess_threshold=0.5)
        tsmc = TwistedSMC(config)
        tsmc.initialize(prompt="What is 2+2?")
        return tsmc

    def test_adaptive_resampling_with_twist(self, tsmc):
        """Resampling is triggered when ESS is low after weight update."""
        # Create highly unequal values to cause low ESS
        current_values = torch.tensor([1.0, 0.01, 0.01, 0.01])
        previous_values = torch.tensor([0.25, 0.25, 0.25, 0.25])

        tsmc.update_weights_with_twist(current_values, previous_values)

        # ESS should be low
        assert tsmc.effective_sample_size() < 2.0

    def test_twist_disabled(self):
        """When use_twist=False, behaves like standard SMC."""
        config = TwistedSMCConfig(n_particles=4, use_twist=False)
        tsmc = TwistedSMC(config)
        tsmc.initialize(prompt="Test")

        # Weights should stay uniform
        weights = tsmc.get_weights()
        assert torch.allclose(weights, torch.tensor(0.25))


# =============================================================================
# Tests for Value Function Interface
# =============================================================================


class TestValueFunctionInterface:
    """Tests for value function integration."""

    def test_set_value_function(self):
        """Can set a custom value function."""
        config = TwistedSMCConfig(n_particles=4)
        tsmc = TwistedSMC(config)

        # Mock value function
        def mock_value_fn(texts):
            return torch.tensor([0.5] * len(texts))

        tsmc.set_value_function(mock_value_fn)
        assert tsmc.value_function is not None

    def test_compute_values_uses_value_function(self):
        """compute_values() uses the set value function."""
        config = TwistedSMCConfig(n_particles=4)
        tsmc = TwistedSMC(config)
        tsmc.initialize(prompt="Test")

        # Mock value function that returns increasing values
        call_count = [0]

        def mock_value_fn(texts):
            call_count[0] += 1
            return torch.tensor([float(i) for i in range(len(texts))])

        tsmc.set_value_function(mock_value_fn)
        values = tsmc.compute_values()

        assert call_count[0] == 1
        assert len(values) == 4
