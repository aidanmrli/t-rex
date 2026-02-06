"""
Unit tests for TSMCLLMParticleFilter twist weighting and metadata handling.
"""

import torch

from trex.smc.tsmc_particle_filter import TSMCLLMParticleFilter


class DummyTwistScorer:
    def __init__(self, values):
        self.values = values

    def score_texts(self, texts):
        if not self.values:
            return torch.zeros(len(texts), dtype=torch.float32)
        padded = list(self.values)
        if len(padded) < len(texts):
            padded.extend([padded[-1]] * (len(texts) - len(padded)))
        return torch.tensor(padded[: len(texts)], dtype=torch.float32)


class SequencedTwistScorer:
    def __init__(self, sequences):
        self.sequences = sequences
        self.call_count = 0

    def score_texts(self, texts):
        idx = min(self.call_count, len(self.sequences) - 1)
        self.call_count += 1
        seq = self.sequences[idx]
        if len(seq) < len(texts):
            seq = list(seq) + [seq[-1]] * (len(texts) - len(seq))
        return torch.tensor(seq[: len(texts)], dtype=torch.float32)


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


def test_initialize_seeds_prompt_prev_values_for_first_update(mock_llm, mock_tsmc_config):
    """Prompt-state prev_value is seeded so first twist update is not neutralized."""
    config = mock_tsmc_config
    config.twist_space = "log_prob"

    scorer = SequencedTwistScorer(
        sequences=[
            [-1.0, -1.0, -1.0, -1.0],  # initialize(prompt)
            [-0.1, -1.0, -1.0, -1.0],  # first scored state
        ]
    )
    pf = TSMCLLMParticleFilter(config, mock_llm, twist_scorer=scorer, reward_model=None)
    pf.initialize("prompt\n\n## Step 1:")

    previous = pf._get_prev_values_tensor()
    assert previous is not None
    assert torch.allclose(previous, torch.tensor([-1.0, -1.0, -1.0, -1.0]))

    current = pf.score_particles()
    new_weights = pf._update_weights_with_twist(current, previous)
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


def test_majority_vote_selection_prefers_most_common_answer(mock_llm, mock_tsmc_config):
    """Majority vote picks the answer with highest support, tie-broken by lineage."""
    config = mock_tsmc_config
    scorer = DummyTwistScorer(values=[0.5, 0.5, 0.5, 0.5])
    pf = TSMCLLMParticleFilter(
        config,
        mock_llm,
        twist_scorer=scorer,
        reward_model=None,
        answer_extractor=lambda text: text.strip(),
    )
    pf.initialize("prompt")

    pf.particles[0].text = "42"
    pf.particles[1].text = "42"
    pf.particles[2].text = "7"
    pf.particles[3].text = "7"
    pf.particles[0].metadata["twist_log_weight"] = 1.0
    pf.particles[1].metadata["twist_log_weight"] = 2.0
    pf.particles[2].metadata["twist_log_weight"] = 0.5
    pf.particles[3].metadata["twist_log_weight"] = 0.1

    best = pf.select_by_majority_vote()
    assert best.text == "42"
    assert best.metadata["twist_log_weight"] == 2.0


def test_warmup_detection_uses_steps_and_tokens(mock_llm, mock_tsmc_config):
    """Warm-up applies by iteration and (for token mode) by per-particle token totals."""
    config = mock_tsmc_config
    config.resampling_unit = "token"
    config.warmup_steps = 2
    config.warmup_tokens = 5
    scorer = DummyTwistScorer(values=[0.5, 0.5, 0.5, 0.5])
    pf = TSMCLLMParticleFilter(config, mock_llm, twist_scorer=scorer, reward_model=None)
    pf.initialize("prompt")

    active = list(range(len(pf.particles)))
    pf._smc_iteration = 0
    assert pf._is_in_warmup(active)

    pf._smc_iteration = 2
    for particle in pf.particles:
        particle.metadata["generated_tokens_total"] = 3
    assert pf._is_in_warmup(active)

    for particle in pf.particles:
        particle.metadata["generated_tokens_total"] = 6
    assert not pf._is_in_warmup(active)
