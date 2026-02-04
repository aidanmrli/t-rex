"""
Twist model wrapper: base LLM + value head for Twisted SMC.

Value heads output raw logits. The wrapper maps logits to ψ or logψ
depending on twist_space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from trex.models.value_head import (
    AttentionPooledValueHead,
    LinearValueHead,
    MLPValueHead,
    ValueHead,
)

TwistSpace = Literal["log_prob", "prob"]


@dataclass
class TwistModelConfig:
    model_name_or_path: str
    value_head_type: str = "mlp"
    twist_space: TwistSpace = "log_prob"
    freeze_base_model: bool = True
    share_base_with_generator: bool = False
    max_length: Optional[int] = None
    epsilon: float = 1e-8
    log_value_min: float = -1e6


class TwistModel(nn.Module):
    """
    Base LLM with attached value head for Twisted SMC.

    The value head outputs raw logits; mapping to ψ/logψ happens here.
    """

    def __init__(
        self,
        model_name_or_path: str,
        value_head_type: str = "mlp",
        twist_space: TwistSpace = "log_prob",
        freeze_base_model: bool = True,
        share_base_with_generator: bool = False,
        max_length: Optional[int] = None,
        epsilon: float = 1e-8,
        log_value_min: float = -1e6,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[object] = None,
    ):
        super().__init__()

        if share_base_with_generator:
            raise NotImplementedError(
                "Shared-weight scorer is not implemented yet. "
                "Use a separate HF model for scoring."
            )

        self.twist_space: TwistSpace = twist_space
        self.epsilon = epsilon
        self.log_value_min = log_value_min
        self.max_length = max_length

        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
            device_map = "auto" if torch.cuda.is_available() else None
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device_map,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
        else:
            if tokenizer is None:
                raise ValueError("tokenizer must be provided when passing a custom model.")
            self.model = model
            self.tokenizer = tokenizer

        self.model.eval()

        hidden_dim = getattr(self.model.config, "hidden_size", None)
        if hidden_dim is None:
            hidden_dim = getattr(self.model.config, "n_embd", None)
        if hidden_dim is None:
            raise ValueError("Could not infer hidden size from model config.")

        value_head_type = value_head_type.lower()
        if value_head_type == "linear":
            self.value_head: ValueHead = LinearValueHead(hidden_dim)
        elif value_head_type == "mlp":
            self.value_head = MLPValueHead(hidden_dim)
        elif value_head_type in {"attention_pooled", "attention-pooled", "attention"}:
            self.value_head = AttentionPooledValueHead(hidden_dim)
        else:
            raise ValueError(f"Unknown value_head_type: {value_head_type}")

        if freeze_base_model:
            for param in self.model.parameters():
                param.requires_grad = False

    def _get_device(self) -> torch.device:
        for param in self.model.parameters():
            return param.device
        return torch.device("cpu")

    def _map_logits_to_twist(self, logits: torch.Tensor) -> torch.Tensor:
        if self.twist_space == "prob":
            values = torch.sigmoid(logits)
            return torch.clamp(values, min=self.epsilon, max=1.0 - self.epsilon)
        if self.twist_space == "log_prob":
            values = F.logsigmoid(logits)
            return torch.clamp(values, min=self.log_value_min, max=0.0)
        raise ValueError(f"Unknown twist_space: {self.twist_space}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_value_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]
        value_logits = self.value_head(hidden_states, attention_mask=attention_mask)
        value_scores = self._map_logits_to_twist(value_logits)

        if return_value_logits:
            return outputs.logits, value_scores, value_logits
        return outputs.logits, value_scores

    def _encode_texts(self, texts: List[str]) -> dict:
        max_length = self.max_length
        tokenizer_max = getattr(self.tokenizer, "model_max_length", None)
        if max_length is None and isinstance(tokenizer_max, int) and 0 < tokenizer_max < 1_000_000:
            max_length = tokenizer_max

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return encoded

    @staticmethod
    def _last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        return lengths - 1

    @staticmethod
    def _gather_last(values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # values: [B, T, ...] or [B, T]
        if values.dim() == 2:
            values = values.unsqueeze(-1)
        batch = values.shape[0]
        indices = TwistModel._last_token_indices(attention_mask)
        return values[torch.arange(batch, device=values.device), indices].squeeze(-1)

    def score_texts(self, texts: List[str]) -> torch.Tensor:
        """Return ψ or logψ for each text, using the last non-padding token."""
        encoded = self._encode_texts(texts)
        device = self._get_device()
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        _, value_scores = self.forward(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if not getattr(self.value_head, "per_token", True):
            return value_scores.view(-1)
        if attention_mask is None:
            return value_scores[:, -1].view(-1)
        return self._gather_last(value_scores, attention_mask).view(-1)

    def score_texts_logits(self, texts: List[str]) -> torch.Tensor:
        """Return raw logits for each text, using the last non-padding token."""
        encoded = self._encode_texts(texts)
        device = self._get_device()
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        _, _, value_logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_value_logits=True,
        )

        if not getattr(self.value_head, "per_token", True):
            return value_logits.view(-1)
        if attention_mask is None:
            return value_logits[:, -1].view(-1)
        return self._gather_last(value_logits, attention_mask).view(-1)

    def get_values_for_texts(self, texts: List[str]) -> torch.Tensor:
        return self.score_texts(texts)
