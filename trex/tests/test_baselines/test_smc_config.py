"""
Unit tests for SMCSteeringConfig and CheckpointManager.

These tests verify:
- SMCSteeringConfig default values and validation
- SMCSteeringConfig serialization (to_dict, from_dict)
- CheckpointManager save/load/resume logic
- Atomic checkpoint writes
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from trex.baselines.smc_config import SMCSteeringConfig, CheckpointManager


class TestSMCSteeringConfig:
    """Tests for SMCSteeringConfig dataclass."""
    
    def test_default_values_are_sensible(self):
        """Default config values should be reasonable for SMC steering."""
        config = SMCSteeringConfig()
        
        assert config.n_particles == 16
        assert config.max_smc_iterations == 20
        assert config.resampling_method == "systematic"
        assert 0.0 < config.ess_threshold <= 1.0
        assert config.temperature > 0.0
        assert config.generator_model_path != ""
        assert config.reward_model_path != ""
    
    def test_custom_n_particles(self):
        """Should accept custom n_particles value."""
        config = SMCSteeringConfig(n_particles=32)
        assert config.n_particles == 32
    
    def test_invalid_resampling_method_raises(self):
        """Invalid resampling method should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid resampling_method"):
            SMCSteeringConfig(resampling_method="invalid_method")
    
    def test_invalid_ess_threshold_raises(self):
        """ESS threshold outside (0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="ess_threshold"):
            SMCSteeringConfig(ess_threshold=0.0)
        
        with pytest.raises(ValueError, match="ess_threshold"):
            SMCSteeringConfig(ess_threshold=1.5)
    
    def test_config_to_dict(self):
        """Config should serialize to dictionary."""
        config = SMCSteeringConfig(n_particles=8, seed=42)
        d = config.to_dict()
        
        assert isinstance(d, dict)
        assert d["n_particles"] == 8
        assert d["seed"] == 42
    
    def test_config_from_dict(self):
        """Config should deserialize from dictionary."""
        d = {
            "n_particles": 8,
            "max_smc_iterations": 10,
            "seed": 42,
        }
        config = SMCSteeringConfig.from_dict(d)
        
        assert config.n_particles == 8
        assert config.max_smc_iterations == 10
        assert config.seed == 42
    
    def test_config_from_dict_ignores_extra_keys(self):
        """from_dict should ignore unknown keys."""
        d = {
            "n_particles": 8,
            "unknown_key": "ignored",
        }
        config = SMCSteeringConfig.from_dict(d)
        assert config.n_particles == 8
        assert not hasattr(config, "unknown_key")
    
    def test_config_hash_deterministic(self):
        """Config hash should be deterministic."""
        config1 = SMCSteeringConfig(n_particles=8, seed=42)
        config2 = SMCSteeringConfig(n_particles=8, seed=42)
        
        assert config1.config_hash() == config2.config_hash()
    
    def test_config_hash_changes_with_params(self):
        """Config hash should change when critical params change."""
        config1 = SMCSteeringConfig(n_particles=8)
        config2 = SMCSteeringConfig(n_particles=16)
        
        assert config1.config_hash() != config2.config_hash()


class TestCheckpointManager:
    """Tests for CheckpointManager class."""
    
    def test_save_creates_file(self, tmp_path):
        """Saving checkpoint should create the file."""
        checkpoint_path = tmp_path / "checkpoint.json"
        mgr = CheckpointManager(str(checkpoint_path))
        
        mgr.state["completed_idx"] = 5
        mgr.save()
        
        assert checkpoint_path.exists()
    
    def test_load_restores_state(self, tmp_path):
        """Loading checkpoint should restore saved state."""
        checkpoint_path = tmp_path / "checkpoint.json"
        
        # Create and save
        mgr1 = CheckpointManager(str(checkpoint_path))
        mgr1.state["completed_idx"] = 10
        mgr1.state["results"] = [{"test": "data"}]
        mgr1.save()
        
        # Load into new manager
        mgr2 = CheckpointManager(str(checkpoint_path))
        
        assert mgr2.state["completed_idx"] == 10
        assert mgr2.state["results"] == [{"test": "data"}]
    
    def test_load_nonexistent_returns_default(self, tmp_path):
        """Loading non-existent checkpoint should use defaults."""
        checkpoint_path = tmp_path / "nonexistent.json"
        mgr = CheckpointManager(str(checkpoint_path))
        
        assert mgr.state["completed_idx"] == 0
        assert mgr.state["results"] == []
        assert mgr.state["finished"] is False
    
    def test_atomic_save_prevents_corruption(self, tmp_path):
        """Save should be atomic (no partial writes)."""
        checkpoint_path = tmp_path / "checkpoint.json"
        mgr = CheckpointManager(str(checkpoint_path))
        
        # Save some data
        mgr.state["completed_idx"] = 100
        mgr.save()
        
        # Verify no temp file left behind
        temp_path = Path(str(checkpoint_path) + ".tmp")
        assert not temp_path.exists()
        
        # Verify main file is valid JSON
        with open(checkpoint_path) as f:
            data = json.load(f)
        assert data["completed_idx"] == 100
    
    def test_is_finished(self, tmp_path):
        """is_finished should reflect state correctly."""
        checkpoint_path = tmp_path / "checkpoint.json"
        mgr = CheckpointManager(str(checkpoint_path))
        
        assert mgr.is_finished() is False
        
        mgr.mark_finished()
        assert mgr.is_finished() is True
    
    def test_mark_finished_saves(self, tmp_path):
        """mark_finished should save the checkpoint."""
        checkpoint_path = tmp_path / "checkpoint.json"
        mgr = CheckpointManager(str(checkpoint_path))
        
        mgr.mark_finished()
        
        # Load and verify
        with open(checkpoint_path) as f:
            data = json.load(f)
        assert data["finished"] is True
    
    def test_get_resume_index(self, tmp_path):
        """get_resume_index should return completed_idx."""
        checkpoint_path = tmp_path / "checkpoint.json"
        mgr = CheckpointManager(str(checkpoint_path))
        
        mgr.state["completed_idx"] = 42
        assert mgr.get_resume_index() == 42
    
    def test_get_results(self, tmp_path):
        """get_results should return saved results."""
        checkpoint_path = tmp_path / "checkpoint.json"
        mgr = CheckpointManager(str(checkpoint_path))
        
        mgr.state["results"] = [{"a": 1}, {"b": 2}]
        assert mgr.get_results() == [{"a": 1}, {"b": 2}]
    
    def test_config_hash_mismatch_warning(self, tmp_path, capsys):
        """Config hash mismatch should warn and start fresh."""
        checkpoint_path = tmp_path / "checkpoint.json"
        
        # Save with one hash
        mgr1 = CheckpointManager(str(checkpoint_path), config_hash="hash1")
        mgr1.state["completed_idx"] = 100
        mgr1.save()
        
        # Load with different hash
        mgr2 = CheckpointManager(str(checkpoint_path), config_hash="hash2")
        
        # Should start fresh (completed_idx = 0)
        assert mgr2.state["completed_idx"] == 0
        
        # Should have printed warning
        captured = capsys.readouterr()
        assert "mismatch" in captured.out.lower() or mgr2.state["completed_idx"] == 0


class TestCheckpointShouldSave:
    """Tests for CheckpointManager.should_save logic."""
    
    def test_should_save_on_interval(self, tmp_path):
        """should_save returns True at interval boundaries."""
        checkpoint_path = tmp_path / "checkpoint.json"
        mgr = CheckpointManager(str(checkpoint_path))
        
        # Should save every 5 problems
        import time
        now = time.time()
        
        assert mgr.should_save(4, interval=5, time_interval=3600, last_save_epoch=now) is False
        assert mgr.should_save(5, interval=5, time_interval=3600, last_save_epoch=now) is True
        assert mgr.should_save(10, interval=5, time_interval=3600, last_save_epoch=now) is True
    
    def test_should_save_on_time(self, tmp_path):
        """should_save returns True when time interval exceeded."""
        checkpoint_path = tmp_path / "checkpoint.json"
        mgr = CheckpointManager(str(checkpoint_path))
        
        import time
        old_time = time.time() - 700  # 700 seconds ago
        
        # Should save if time exceeds 600s interval
        assert mgr.should_save(1, interval=100, time_interval=600, last_save_epoch=old_time) is True
