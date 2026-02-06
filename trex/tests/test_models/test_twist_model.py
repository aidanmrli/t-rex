"""
Unit tests for TwistModel mapping and scoring.
"""

from types import SimpleNamespace

import torch
from torch import nn

from trex.models.twist_model import TwistModel


class DummyTokenizer:
    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=None):
        lengths = [max(1, len(t.split())) for t in texts]
        max_len = max(lengths)
        input_ids = []
        attention_mask = []
        for length in lengths:
            ids = list(range(1, length + 1))
            pad_len = max_len - length
            input_ids.append(ids + [0] * pad_len)
            attention_mask.append([1] * length + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class DummyModel(nn.Module):
    def __init__(self, hidden_size: int = 4, vocab_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=True):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, self.config.hidden_size)
        logits = torch.zeros(
            input_ids.size(0), input_ids.size(1), self.vocab_size, device=input_ids.device
        )
        hidden_states = [hidden]
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)


def test_score_texts_logits_uses_last_token():
    """score_texts_logits returns logits from the last non-padding token."""
    model = DummyModel(hidden_size=2)
    tokenizer = DummyTokenizer()
    twist = TwistModel(
        model_name_or_path="dummy",
        value_head_type="linear",
        twist_space="log_prob",
        model=model,
        tokenizer=tokenizer,
    )
    twist.value_head.linear.weight.data.fill_(1.0)
    twist.value_head.linear.bias.data.zero_()

    logits = twist.score_texts_logits(["a b", "a b c"])
    # last token ids are 2 and 3; hidden_size=2 -> logits = 2*id
    assert torch.allclose(logits, torch.tensor([4.0, 6.0]))


def test_log_prob_mapping_is_non_positive():
    """log_prob mapping returns values <= 0."""
    model = DummyModel(hidden_size=2)
    tokenizer = DummyTokenizer()
    twist = TwistModel(
        model_name_or_path="dummy",
        value_head_type="linear",
        twist_space="log_prob",
        model=model,
        tokenizer=tokenizer,
    )
    twist.value_head.linear.weight.data.zero_()
    twist.value_head.linear.bias.data.zero_()

    values = twist.score_texts(["x y"])
    assert torch.all(values <= 0.0)


def test_prob_mapping_in_range():
    """prob mapping returns values in (0, 1)."""
    model = DummyModel(hidden_size=2)
    tokenizer = DummyTokenizer()
    twist = TwistModel(
        model_name_or_path="dummy",
        value_head_type="linear",
        twist_space="prob",
        model=model,
        tokenizer=tokenizer,
    )
    twist.value_head.linear.weight.data.zero_()
    twist.value_head.linear.bias.data.zero_()

    values = twist.score_texts(["x y"])
    assert torch.all(values > 0.0)
    assert torch.all(values < 1.0)


def test_score_texts_logits_supports_explicit_token_indices():
    """Explicit token indices select the requested position per sequence."""
    model = DummyModel(hidden_size=2)
    tokenizer = DummyTokenizer()
    twist = TwistModel(
        model_name_or_path="dummy",
        value_head_type="linear",
        twist_space="log_prob",
        model=model,
        tokenizer=tokenizer,
    )
    twist.value_head.linear.weight.data.fill_(1.0)
    twist.value_head.linear.bias.data.zero_()

    logits = twist.score_texts_logits(["a b", "a b c"], token_indices=[0, 1])
    # token ids at requested positions are 1 and 2; hidden_size=2 -> logits = 2*id
    assert torch.allclose(logits, torch.tensor([2.0, 4.0]))
