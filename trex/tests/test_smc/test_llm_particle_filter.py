"""
Unit tests for LLMParticleFilter.

These tests verify:
- Step pattern detection
- Particle expansion (mocked LLM)
- SMC weighting and resampling
- ORM-based selection
- Particle independence after resampling
"""

import re
import pytest
import torch
from unittest.mock import MagicMock, patch


class TestStepDetection:
    """Test step pattern detection - CRITICAL for SMC timing."""
    
    def test_step_pattern_matches_valid_format(self):
        """Step pattern should match '## Step N:' format."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        pattern = LLMParticleFilter.STEP_PATTERN
        
        assert pattern.search("## Step 1: Calculate")
        assert pattern.search("## Step 2: Verify result")
        assert pattern.search("## Step 10: Multi-digit")
    
    def test_step_pattern_requires_colon(self):
        """Step pattern must have colon after number."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        pattern = LLMParticleFilter.STEP_PATTERN
        
        assert not pattern.search("## Step 1")  # Missing colon
        assert not pattern.search("## Step 1 ")  # Space instead of colon
    
    def test_step_pattern_requires_space_after_hash(self):
        """Step pattern requires space between ## and Step."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        pattern = LLMParticleFilter.STEP_PATTERN
        
        assert not pattern.search("##Step 1:")  # No space
        assert not pattern.search("# Step 1:")  # Single hash
    
    def test_step_pattern_multi_digit_numbers(self):
        """Should handle multi-digit step numbers."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        pattern = LLMParticleFilter.STEP_PATTERN
        
        assert pattern.search("## Step 100: Hundredth step")
        assert pattern.search("## Step 999:")
    
    def test_count_steps_in_text(self):
        """Should correctly count steps in text."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        from trex.baselines.smc_config import SMCSteeringConfig
        
        config = SMCSteeringConfig(n_particles=1, enable_checkpointing=False)
        
        # Create with mocks - we just need the method
        with patch('trex.smc.llm_particle_filter.LLMParticleFilter.__init__', 
                   lambda self, *args, **kwargs: None):
            pf = LLMParticleFilter.__new__(LLMParticleFilter)
            pf.STEP_HEADER_PATTERN = re.compile(r"## Step (\d+):")
            
            text = "## Step 1: First\n## Step 2: Second\n## Step 3: Third"
            count = pf._count_reasoning_steps(text)
            
            assert count == 3
    
    def test_get_next_step_number(self):
        """Should return next step number based on existing text."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        with patch('trex.smc.llm_particle_filter.LLMParticleFilter.__init__',
                   lambda self, *args, **kwargs: None):
            pf = LLMParticleFilter.__new__(LLMParticleFilter)
            pf.STEP_HEADER_PATTERN = re.compile(r"## Step (\d+):")
            
            # No steps yet
            assert pf._get_next_step_number("Just a prompt") == 1
            
            # After step 1
            assert pf._get_next_step_number("## Step 1: First\nContent") == 2
            
            # After step 3
            text = "## Step 1: A\n## Step 2: B\n## Step 3: C"
            assert pf._get_next_step_number(text) == 4


class TestParticleExpansion:
    """Test particle generation (mocked LLM)."""
    
    def test_expand_appends_generated_text(self, mock_llm, mock_reward_model, mock_smc_config, mock_output):
        """Generated text should be appended to particles."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        # Setup mock LLM
        mock_llm.generate.return_value = [
            mock_output.from_text("## Step 1: Calculate\n2+2=4\n## Step")
            for _ in range(mock_smc_config.n_particles)
        ]
        
        pf = LLMParticleFilter(mock_smc_config, mock_llm, mock_reward_model)
        pf.initialize("What is 2+2?")
        
        pf.expand_particles()
        
        assert "## Step 1" in pf.particles[0].text
        assert "Calculate" in pf.particles[0].text
    
    def test_expand_marks_finished_on_boxed(self, mock_llm, mock_reward_model, mock_smc_config, mock_output):
        """Particles with \\boxed{} should be marked finished."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        # Return boxed answer
        mock_llm.generate.return_value = [
            mock_output.from_text("Therefore, $\\boxed{4}$")
            for _ in range(mock_smc_config.n_particles)
        ]
        
        pf = LLMParticleFilter(mock_smc_config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        
        pf.expand_particles()
        
        for particle in pf.particles:
            assert particle.metadata.get("finished") is True
    
    def test_expand_returns_false_when_all_finished(self, mock_llm, mock_reward_model, mock_smc_config, mock_output):
        """expand_particles should return False when all are finished."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        mock_llm.generate.return_value = [
            mock_output.from_text("$\\boxed{4}$")
            for _ in range(mock_smc_config.n_particles)
        ]
        
        pf = LLMParticleFilter(mock_smc_config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        
        result = pf.expand_particles()
        
        assert result is False
    
    def test_expand_tracks_reasoning_step_count(self, mock_llm, mock_reward_model, mock_smc_config, mock_output):
        """Should track reasoning step count in metadata."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        mock_llm.generate.return_value = [
            mock_output.from_text("## Step 1: First\n## Step 2: Second\n## Step")
            for _ in range(mock_smc_config.n_particles)
        ]
        
        pf = LLMParticleFilter(mock_smc_config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        
        pf.expand_particles()
        
        # Should have counted 2 steps (## Step 1 and ## Step 2)
        # Note: "## Step" at end is partial, Pattern requires "## Step N:"
        assert pf.particles[0].metadata.get("reasoning_step_count", 0) >= 2


class TestSMCWeighting:
    """Test SMC weight updates."""
    
    def test_score_particles_calls_prm(self, mock_llm, mock_reward_model, mock_smc_config, mock_output):
        """score_particles should call reward model."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        pf = LLMParticleFilter(mock_smc_config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        
        scores = pf.score_particles()
        
        # Should have called format_text_for_scoring
        assert mock_reward_model.format_text_for_scoring.called
        assert mock_reward_model.get_latest_step_scores.called
    
    def test_epsilon_added_to_avoid_zero_weights(self, mock_llm, mock_reward_model, mock_smc_config, mock_output):
        """Epsilon should be added to prevent zero weights."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        # Return all zeros
        mock_reward_model.get_latest_step_scores.return_value = torch.tensor([0.0, 0.0, 0.0, 0.0])
        mock_llm.generate.return_value = [
            mock_output.from_text("step content")
            for _ in range(mock_smc_config.n_particles)
        ]
        
        pf = LLMParticleFilter(mock_smc_config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        
        # Should not raise even with zero scores
        pf.step()
        
        # Weights should be positive (epsilon added)
        weights = pf.get_weights()
        assert torch.all(weights > 0)
    
    def test_multiplicative_weight_update(self, mock_llm, mock_reward_model, mock_smc_config, mock_output):
        """Verify weights are updated multiplicatively: w_t = w_{t-1} × PRM(step_t)."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        # Mock PRM scores
        prm_scores = torch.tensor([0.8, 0.6, 0.9, 0.3])
        mock_reward_model.get_latest_step_scores.return_value = prm_scores
        
        # Mock LLM to not finish
        mock_llm.generate.return_value = [
            mock_output.from_text("step content")
            for _ in range(mock_smc_config.n_particles)
        ]
        
        pf = LLMParticleFilter(mock_smc_config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        
        # Get initial weights (uniform)
        initial_weights = pf.get_weights().clone()
        
        # Disable resampling for this test
        pf.config.ess_threshold = 0.0  # Never resample
        
        # Run one step
        pf.step()
        
        # New weights should be proportional to initial * PRM scores
        # After normalization, ratios should match
        new_weights = pf.get_weights()
        expected = (prm_scores + 1e-8)  # epsilon added in step()
        expected = expected / expected.sum()
        
        torch.testing.assert_close(new_weights, expected, rtol=1e-5, atol=1e-5)


class TestORMSelection:
    """Test ORM-based final selection."""
    
    def test_select_best_by_orm_returns_highest(self, mock_llm, mock_reward_model, mock_smc_config):
        """select_best_by_orm should return particle with highest ORM score."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        # Mock ORM to return known scores - index 2 is highest
        mock_reward_model.score_orm.return_value = [0.3, 0.5, 0.9, 0.2]
        
        pf = LLMParticleFilter(mock_smc_config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        
        # Give each particle unique text for identification
        for i, p in enumerate(pf.particles):
            p.text = f"particle_{i}"
        
        best = pf.select_best_by_orm()
        
        assert best.text == "particle_2"  # Index 2 had highest score 0.9
    
    def test_select_best_by_orm_stores_scores(self, mock_llm, mock_reward_model, mock_smc_config):
        """ORM scores should be stored in particle metadata."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        mock_reward_model.score_orm.return_value = [0.3, 0.5, 0.9, 0.2]
        
        pf = LLMParticleFilter(mock_smc_config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        
        pf.select_best_by_orm()
        
        # All particles should have ORM score in metadata
        for i, particle in enumerate(pf.particles):
            assert "orm_score" in particle.metadata
            assert particle.metadata["orm_score"] == [0.3, 0.5, 0.9, 0.2][i]


class TestParticleIndependence:
    """Test particle independence after resampling - CRITICAL for correctness."""
    
    def test_particle_independence_after_resampling(self, mock_llm, mock_reward_model, mock_smc_config):
        """
        CRITICAL: After resampling, particles must be independent.
        
        When a particle is duplicated during resampling, modifying the copy
        should NOT affect the original. This requires deepcopy, not shallow copy.
        """
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        pf = LLMParticleFilter(mock_smc_config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        
        # Give unique text to each particle
        for i, p in enumerate(pf.particles):
            p.text = f"particle_{i}"
            p.metadata["id"] = i
        
        # Set weights so particle 0 dominates (will be duplicated)
        pf.set_weights(torch.tensor([0.97, 0.01, 0.01, 0.01]))
        pf.normalize_weights()
        
        # Resample - particle 0 should be copied to most positions
        pf.resample()
        
        # Find which particles have the same text (were copied)
        texts_after = [p.text for p in pf.particles]
        
        # Modify the first particle's text and metadata
        original_text_0 = pf.particles[0].text
        pf.particles[0].text = original_text_0 + " MODIFIED"
        pf.particles[0].metadata["test_key"] = "test_value"
        
        # Other particles with the same original text should NOT be affected
        for i in range(1, len(pf.particles)):
            assert "MODIFIED" not in pf.particles[i].text, \
                f"Particle {i} was affected by modification to particle 0 (shallow copy bug)"
            assert pf.particles[i].metadata.get("test_key") != "test_value", \
                f"Particle {i} metadata was affected (shallow copy bug)"
    
    def test_resampling_resets_weights(self, mock_llm, mock_reward_model, mock_smc_config):
        """After resampling, weights should be reset to uniform."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        pf = LLMParticleFilter(mock_smc_config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        
        # Set non-uniform weights
        pf.set_weights(torch.tensor([0.7, 0.1, 0.1, 0.1]))
        
        pf.resample()
        
        # Weights should be uniform after resampling
        weights = pf.get_weights()
        expected = torch.ones(mock_smc_config.n_particles) / mock_smc_config.n_particles
        
        torch.testing.assert_close(weights, expected)


class TestSMCLoop:
    """Test full SMC loop."""
    
    def test_resampling_triggered_on_low_ess(self, mock_llm, mock_reward_model, mock_smc_config):
        """Resampling should occur when ESS drops below threshold."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        pf = LLMParticleFilter(mock_smc_config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        
        # Set highly skewed weights (low ESS)
        pf.set_weights(torch.tensor([0.99, 0.003, 0.003, 0.004]))
        pf.normalize_weights()
        
        # ESS should be low
        ess = pf.effective_sample_size()
        threshold = mock_smc_config.ess_threshold * mock_smc_config.n_particles
        
        assert ess < threshold, f"ESS {ess} should be below threshold {threshold}"
        assert pf.should_resample()
    
    def test_run_returns_best_particle(self, mock_llm, mock_reward_model, mock_smc_config, mock_output):
        """run() should return the ORM-selected best particle."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        # Make particles finish quickly
        mock_llm.generate.return_value = [
            mock_output.from_text("$\\boxed{4}$")
            for _ in range(mock_smc_config.n_particles)
        ]
        mock_reward_model.score_orm.return_value = [0.1, 0.2, 0.3, 0.8]  # Last is best
        
        pf = LLMParticleFilter(mock_smc_config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        
        best = pf.run()
        
        assert best is not None
        assert best.metadata.get("orm_score") == 0.8  # Highest ORM
    
    def test_get_summary_returns_stats(self, mock_llm, mock_reward_model, mock_smc_config, mock_output):
        """get_summary should return useful statistics."""
        from trex.smc.llm_particle_filter import LLMParticleFilter
        
        mock_llm.generate.return_value = [
            mock_output.from_text("## Step 1: Test\n$\\boxed{4}$")
            for _ in range(mock_smc_config.n_particles)
        ]
        
        pf = LLMParticleFilter(mock_smc_config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        pf.expand_particles()
        
        summary = pf.get_summary()
        
        assert "smc_iteration" in summary
        assert "n_particles" in summary
        assert "n_finished" in summary
        assert summary["n_particles"] == mock_smc_config.n_particles
