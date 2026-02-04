"""Unit tests for value trainer rollout preprocessing."""

import sys
from types import ModuleType, SimpleNamespace

from torch import nn

from trex.training.value_trainer import ValueTrainer, ValueTrainingConfig


class _DummyTwistModel:
    def __init__(self):
        self.value_head = nn.Linear(1, 1)


class _DummyGenerator:
    def generate(self, prompts, sampling_params):
        del sampling_params
        return [SimpleNamespace(outputs=[SimpleNamespace(text=" solution")]) for _ in prompts]

    def get_tokenizer(self):
        return SimpleNamespace(apply_chat_template=lambda messages, **_: messages[-1]["content"])


class _DummyVerifier:
    def extract_answer(self, text):
        return text

    def verify(self, extracted, ground_truth):
        del extracted, ground_truth
        return True


def test_collect_rollouts_no_step_markers_does_not_duplicate_prompt(monkeypatch):
    """When generated text has no step markers, prompt should appear once in state text."""
    fake_vllm = ModuleType("vllm")
    fake_vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    trainer = ValueTrainer(
        twist_model=_DummyTwistModel(),
        config=ValueTrainingConfig(
            batch_size=1,
            num_rollouts_per_prompt=1,
            apply_chat_template=False,
        ),
    )

    trajectories = trainer.collect_rollouts(
        problems=[{"prompt": "Q:", "answer": "A"}],
        generator=_DummyGenerator(),
        verifier=_DummyVerifier(),
    )

    pairs = trajectories[0].get_state_reward_pairs()
    assert pairs[0][0] == "Q: solution"
    assert pairs[0][0].count("Q:") == 1
