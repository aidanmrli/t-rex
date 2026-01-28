"""
TDD Tests for trex/smc/particle_filter.py - Particle Filter Implementation (PyTorch).

These tests are written BEFORE the implementation as part of TDD workflow.
They define the expected API and behavior of the ParticleFilter class using PyTorch.

The particle filter module should provide:
- SMCConfig: Configuration dataclass for SMC parameters
- Particle: Individual particle with text and value
- ParticleFilter: Main class for SMC particle filtering
"""

import pytest
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
import numpy as np


# =============================================================================
# Import with skip if not implemented yet
# =============================================================================

try:
    from trex.smc.particle_filter import (
        SMCConfig,
        Particle,
        ParticleFilter,
    )
    PARTICLE_FILTER_IMPLEMENTED = True
except ImportError:
    PARTICLE_FILTER_IMPLEMENTED = False
    SMCConfig = None
    Particle = None
    ParticleFilter = None


pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE or not PARTICLE_FILTER_IMPLEMENTED,
    reason="torch not available or trex.smc.particle_filter not implemented"
)


# =============================================================================
# Tests for SMCConfig
# =============================================================================


class TestSMCConfig:
    """Tests for SMCConfig dataclass."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = SMCConfig()
        
        assert config.n_particles >= 1
        assert 0.0 <= config.ess_threshold <= 1.0
        assert config.resampling_method in ["multinomial", "systematic", "stratified"]

    def test_custom_n_particles(self):
        """Can set custom number of particles."""
        config = SMCConfig(n_particles=32)
        
        assert config.n_particles == 32


# =============================================================================
# Tests for Particle
# =============================================================================


class TestParticle:
    """Tests for Particle dataclass."""

    def test_creation(self):
        """Can create a particle with text and weight."""
        # Note: Weights in Particle might be stored but ParticleFilter manages proper vector weights
        particle = Particle(text="Hello world", weight=0.5)
        
        assert particle.text == "Hello world"
        assert particle.weight == 0.5


# =============================================================================
# Tests for ParticleFilter Initialization
# =============================================================================


class TestParticleFilterInit:
    """Tests for ParticleFilter initialization."""

    def test_initialization_with_config(self):
        """ParticleFilter can be initialized with config."""
        config = SMCConfig(n_particles=16)
        pf = ParticleFilter(config)
        
        assert pf.config.n_particles == 16

    def test_initialize_creates_particles(self):
        """Initialize creates the specified number of particles."""
        config = SMCConfig(n_particles=16)
        pf = ParticleFilter(config)
        pf.initialize(prompt="What is 2+2?")
        
        assert len(pf.particles) == 16


class TestParticleFilterInitialize:
    """Tests for ParticleFilter.initialize() method."""

    def test_all_particles_start_with_prompt(self):
        """All particles start with the same prompt."""
        config = SMCConfig(n_particles=4)
        pf = ParticleFilter(config)
        pf.initialize(prompt="What is 2+2?")
        
        for particle in pf.particles:
            assert particle.text.startswith("What is 2+2?")

    def test_initial_weights_are_uniform(self):
        """Initial weights are uniform (all equal to 1/n)."""
        config = SMCConfig(n_particles=4)
        pf = ParticleFilter(config)
        pf.initialize(prompt="What is 2+2?")
        
        weights = pf.get_weights()
        assert isinstance(weights, torch.Tensor)
        expected = 1.0 / 4
        assert torch.allclose(weights, torch.tensor(expected))

    def test_weights_sum_to_one(self):
        """Weights sum to 1.0."""
        config = SMCConfig(n_particles=8)
        pf = ParticleFilter(config)
        pf.initialize(prompt="What is 2+2?")
        
        weights = pf.get_weights()
        assert torch.isclose(weights.sum(), torch.tensor(1.0))


# =============================================================================
# Tests for Weight Management
# =============================================================================


class TestParticleFilterWeights:
    """Tests for weight-related operations using PyTorch."""

    @pytest.fixture
    def pf(self):
        """Create an initialized particle filter."""
        config = SMCConfig(n_particles=4)
        pf = ParticleFilter(config)
        pf.initialize(prompt="What is 2+2?")
        return pf

    def test_get_weights_returns_tensor(self, pf):
        """get_weights() returns torch tensor."""
        weights = pf.get_weights()
        
        assert isinstance(weights, torch.Tensor)
        assert len(weights) == 4

    def test_set_weights_from_tensor(self, pf):
        """Can set weights from tensor."""
        new_weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        pf.set_weights(new_weights)
        
        assert torch.allclose(pf.get_weights(), new_weights)

    def test_set_weights_from_numpy(self, pf):
        """Can set weights from numpy array (converts to tensor)."""
        new_weights = np.array([0.1, 0.2, 0.3, 0.4])
        pf.set_weights(new_weights)
        
        weights = pf.get_weights()
        assert isinstance(weights, torch.Tensor)
        assert torch.allclose(weights, torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=weights.dtype))

    def test_normalize_weights(self, pf):
        """Weights can be normalized."""
        pf.set_weights(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        pf.normalize_weights()
        
        weights = pf.get_weights()
        assert torch.isclose(weights.sum(), torch.tensor(1.0))


# =============================================================================
# Tests for Resampling
# =============================================================================


class TestParticleFilterResampling:
    """Tests for resampling operations."""

    @pytest.fixture
    def pf(self):
        """Create an initialized particle filter."""
        config = SMCConfig(n_particles=4)
        pf = ParticleFilter(config)
        pf.initialize(prompt="What is 2+2?")
        return pf

    def test_resampling_preserves_count(self, pf):
        """Resampling doesn't change particle count."""
        pf.set_weights(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        
        pf.resample()
        
        assert len(pf.particles) == 4

    def test_resampling_resets_weights_to_uniform(self, pf):
        """After resampling, weights are uniform."""
        pf.set_weights(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        
        pf.resample()
        
        weights = pf.get_weights()
        assert torch.allclose(weights, torch.tensor(0.25))

    def test_high_weight_particle_duplicated(self, pf):
        """High-weight particles are duplicated during resampling."""
        # Give particle 3 all the weight
        pf.set_weights(torch.tensor([0.0, 0.0, 0.0, 1.0]))
        particle_3_text = pf.particles[3].text
        
        pf.resample()
        
        # All particles should now have particle 3's text
        for particle in pf.particles:
            assert particle.text == particle_3_text


# =============================================================================
# Tests for GPU Support
# =============================================================================


class TestParticleFilterGPU:
    """Tests for GPU support."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_particles_weights_on_device(self):
        """Weights can use GPU device."""
        config = SMCConfig(n_particles=4, device="cuda")
        pf = ParticleFilter(config)
        pf.initialize(prompt="Test")
        
        weights = pf.get_weights()
        assert weights.device.type == "cuda"
