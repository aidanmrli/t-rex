"""
Unit tests for TSMCLLMParticleFilter twist weighting and metadata handling.
"""

import torch

from trex.smc.tsmc_particle_filter import TSMCLLMParticleFilter


class DummyTwistScorer:
    def __init__(self, values):
        self.values = values

    def score_texts(self, texts):
        return torch.tensor(self.values[: len(texts)], dtype=torch.float32)


def test_twist_weight_update_log_space(mock_llm, mock_tsmc_config):
    """Log-space twist updates normalize and favor higher logψ."""
    config = mock_tsmc_config
    config.twist_space = "log_prob"

    scorer = DummyTwistScorer(values=[-0.1, -1.0, -1.0, -1.0])
    pf = TSMCLLMParticleFilter(config, mock_llm, twist_scorer=scorer, reward_model=None)
    pf.initialize("prompt\n\n## Step 1:")

    current = torch.tensor([-0.1, -1.0, -1.0, -1.0])
    previous = torch.tensor([-1.0, -1.0, -1.0, -1.0])

    new_weights = pf._update_weights_with_twist(current, previous)
    assert torch.isclose(new_weights.sum(), torch.tensor(1.0))
    assert int(torch.argmax(new_weights)) == 0


def test_twist_resampling_preserves_prev_value_metadata(mock_llm, mock_tsmc_config):
    """Resampling keeps prev_value metadata aligned with selected particles."""
    config = mock_tsmc_config
    scorer = DummyTwistScorer(values=[0.5, 0.5, 0.5, 0.5])
    pf = TSMCLLMParticleFilter(config, mock_llm, twist_scorer=scorer, reward_model=None)
    pf.initialize("prompt\n\n## Step 1:")

    for idx, p in enumerate(pf.particles):
        p.metadata["prev_value"] = float(idx + 1)

    active_indices = list(range(len(pf.particles)))
    active_weights = torch.tensor([1.0, 0.0, 0.0, 0.0])
    pf._resample_active(active_indices, active_weights)

    assert all(p.metadata.get("prev_value") == 1.0 for p in pf.particles)


def test_score_particles_uses_prev_values_for_finished(mock_llm, mock_tsmc_config):
    """Finished particles retain previous values in score tensor."""
    config = mock_tsmc_config
    scorer = DummyTwistScorer(values=[0.8])
    pf = TSMCLLMParticleFilter(config, mock_llm, twist_scorer=scorer, reward_model=None)
    pf.initialize("prompt\n\n## Step 1:")

    for idx, p in enumerate(pf.particles):
        p.metadata["prev_value"] = 0.3 + 0.1 * idx

    pf.particles[0].metadata["finished"] = True

    values = pf.score_particles()
    assert torch.isclose(values[0], torch.tensor(0.3))


def test_prob_space_zero_weights_falls_back_to_uniform(mock_llm, mock_tsmc_config):
    """Prob-space update avoids NaNs when all updated weights are zero."""
    config = mock_tsmc_config
    config.twist_space = "prob"
    scorer = DummyTwistScorer(values=[0.0, 0.0, 0.0, 0.0])
    pf = TSMCLLMParticleFilter(config, mock_llm, twist_scorer=scorer, reward_model=None)
    pf.initialize("prompt\n\n## Step 1:")

    current = torch.tensor([0.0, 0.0, 0.0, 0.0])
    previous = torch.tensor([1.0, 1.0, 1.0, 1.0])
    new_weights = pf._update_weights_with_twist(current, previous)

    assert torch.all(torch.isfinite(new_weights))
    assert torch.allclose(new_weights, torch.full_like(new_weights, 0.25))


def test_fallback_selection_uses_lineage_score_every_step(mock_llm, mock_tsmc_config):
    """ORM-disabled fallback uses cumulative lineage score under every-step resampling."""
    config = mock_tsmc_config
    config.use_orm_for_final = False
    config.resampling_strategy = "every_step"
    scorer = DummyTwistScorer(values=[0.5, 0.5, 0.5, 0.5])
    pf = TSMCLLMParticleFilter(config, mock_llm, twist_scorer=scorer, reward_model=None)
    pf.initialize("prompt\n\n## Step 1:")

    for idx, particle in enumerate(pf.particles):
        particle.metadata["twist_log_weight"] = float(idx)

    best = pf._select_best_particle_without_orm()
    assert best.metadata["twist_log_weight"] == 3.0
