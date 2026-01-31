"""
Reward Model wrapper for Process/Outcome Reward Models (PRM/ORM).

This module provides:
- RewardModel: Unified wrapper for PRM and ORM scoring

The RewardModel class abstracts the differences between PRM (score each step)
and ORM (score final answer) modes, using the same underlying model with
different input formatting.

Design Note: Uses HuggingFace Transformers directly (not vLLM) because:
1. vLLM's generate() API doesn't easily expose logits at arbitrary positions
2. The official Qwen2.5-Math-PRM-7B examples use HuggingFace Transformers
3. Simpler implementation without monkey-patching vLLM internals
"""

import re
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from trex.models.prm_config import PRMConfig, QWEN_PRM_CONFIG


class RewardModel:
    """
    Model-agnostic wrapper for Process/Outcome Reward Models.
    
    Supports both PRM (score each step) and ORM (score final answer) modes.
    Default configuration is for Qwen2.5-Math-PRM-7B, but can be customized
    for other PRMs via PRMConfig.
    
    Attributes:
        model_path: Path to the HuggingFace model
        prm_config: Model-specific token and extraction configuration
        model: The loaded HuggingFace model (lazy loaded)
        tokenizer: The loaded tokenizer (lazy loaded)
    
    Example:
        >>> rm = RewardModel("Qwen/Qwen2.5-Math-PRM-7B")
        >>> # PRM mode: score each step
        >>> steps = ["## Step 1: Add\\n2+2=4", "## Step 2: Verify\\n4 is correct"]
        >>> formatted = rm.format_for_prm(steps)
        >>> scores = rm.score_prm([formatted])  # Returns [[0.95, 0.88]]
        >>> 
        >>> # ORM mode: score complete solution
        >>> solution = "## Step 1: Add\\n2+2=4\\nTherefore, the answer is $\\\\boxed{4}$"
        >>> score = rm.score_orm([solution])  # Returns [0.92]
    """
    
    # Regex pattern to detect step boundaries
    STEP_PATTERN = re.compile(r"## Step \d+:")
    
    def __init__(
        self,
        model_path: str,
        tp_size: int = 1,
        prm_config: PRMConfig = QWEN_PRM_CONFIG,
        device: Optional[str] = None,
        load_model: bool = True,
    ):
        """
        Initialize the RewardModel.
        
        Args:
            model_path: HuggingFace model path or local path
            tp_size: Tensor parallelism size (for multi-GPU)
            prm_config: Model-specific configuration
            device: Device to load model on. If None, uses "cuda" if available.
            load_model: Whether to load the model immediately. Set False for testing.
        """
        self.model_path = model_path
        self.prm_config = prm_config
        self.tp_size = tp_size
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.tokenizer = None
        self.separator_token_id = None
        
        if load_model:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the model and tokenizer from HuggingFace."""
        from transformers import AutoModel, AutoTokenizer
        
        print(f"Loading reward model from {self.model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        # Set device_map based on available hardware
        device_map = "auto" if torch.cuda.is_available() else None
        
        self.model = AutoModel.from_pretrained(
            self.model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        ).eval()
        
        # Get separator token ID
        self.separator_token_id = self.tokenizer.encode(
            self.prm_config.step_separator_token,
            add_special_tokens=False
        )[0]
        
        print(f"Reward model loaded. Separator token: {self.prm_config.step_separator_token} "
              f"(ID: {self.separator_token_id})")
    
    def format_for_prm(self, steps: List[str]) -> str:
        """
        Format steps for PRM scoring by joining with separator token.
        
        Each step gets a separator token appended. This is the format
        expected by Qwen2.5-Math-PRM-7B for process reward scoring.
        
        Args:
            steps: List of reasoning step strings
            
        Returns:
            Formatted string with separator tokens between/after steps
            
        Note:
            If steps is empty, returns just the separator token. This is
            valid but unusual - callers should typically ensure at least
            one step is present before calling this method.
            
        Example:
            >>> steps = ["## Step 1: Add\\n2+2=4", "## Step 2: Verify\\nCheck 4"]
            >>> formatted = rm.format_for_prm(steps)
            >>> # Returns: "## Step 1: Add\\n2+2=4<extra_0>## Step 2: Verify\\nCheck 4<extra_0>"
        """
        import warnings
        
        if not steps:
            warnings.warn(
                "format_for_prm called with empty steps list. "
                "This returns just a separator token, which may produce "
                "unexpected PRM scores. Consider checking for empty input upstream.",
                UserWarning,
                stacklevel=2
            )
        
        separator = self.prm_config.step_separator_token
        return separator.join(steps) + separator
    
    def format_for_orm(self, text: str) -> str:
        """
        Format complete solution for ORM scoring.
        
        ORM format has exactly one separator token at the end.
        This is for Outcome Reward Model scoring (final answer quality).
        
        Args:
            text: Complete solution text
            
        Returns:
            Text with single separator token at end
            
        Example:
            >>> text = "## Step 1: Add\\n2+2=4\\nThe answer is $\\\\boxed{4}$"
            >>> formatted = rm.format_for_orm(text)
            >>> # Returns: "## Step 1: Add\\n2+2=4\\nThe answer is $\\\\boxed{4}$<extra_0>"
        """
        separator = self.prm_config.step_separator_token
        # Remove any trailing separator tokens first, then add exactly one
        text = text.rstrip()
        while text.endswith(separator):
            text = text[:-len(separator)].rstrip()
        return text + separator
    
    def _split_into_steps_with_preamble(self, text: str) -> tuple[str, List[str]]:
        """
        Split text into preamble (prompt) and reasoning steps.

        Args:
            text: Text containing "## Step N:" patterns

        Returns:
            Tuple of (preamble, steps) where preamble is text before first step
        """
        matches = list(self.STEP_PATTERN.finditer(text))

        if not matches:
            # No step pattern found, treat entire text as preamble with no steps
            return (text.strip(), [])

        # Extract preamble (everything before first step)
        preamble = text[:matches[0].start()].strip()

        # Extract steps
        steps = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            steps.append(text[start:end].strip())

        return (preamble, steps)

    def _split_into_steps(self, text: str) -> List[str]:
        """
        Split text into reasoning steps based on step pattern.

        Args:
            text: Text containing "## Step N:" patterns

        Returns:
            List of step strings (without preamble)
        """
        _, steps = self._split_into_steps_with_preamble(text)
        if not steps:
            # No step pattern found, treat entire text as one "step"
            return [text] if text.strip() else []
        return steps
    
    def _extract_scores_from_logits(
        self,
        logits: torch.Tensor,
        token_masks: torch.Tensor
    ) -> List[List[float]]:
        """
        Extract step reward scores from model logits.
        
        This follows the official Qwen2.5-Math-PRM-7B extraction pattern.
        
        Args:
            logits: Model output, shape (batch, seq_len, num_classes)
            token_masks: Boolean mask for separator positions, shape (batch, seq_len)
            
        Returns:
            List of score lists, one per sample, scores in [0, 1]
        """
        if self.prm_config.extraction_method == "binary_softmax":
            probabilities = F.softmax(logits, dim=-1)
            probabilities = probabilities * token_masks.unsqueeze(-1)
            
            all_scores = []
            for i in range(probabilities.size(0)):
                sample = probabilities[i]  # (seq_len, num_classes)
                # Extract non-zero entries (where mask was True)
                positive_probs = sample[sample.sum(dim=-1) != 0]
                if positive_probs.numel() > 0:
                    positive_probs = positive_probs.view(-1, self.prm_config.num_classes)
                    positive_probs = positive_probs[:, self.prm_config.positive_class_idx]
                    all_scores.append(positive_probs.cpu().tolist())
                else:
                    all_scores.append([])
            
            return all_scores
        
        elif self.prm_config.extraction_method == "single_logit":
            # For models that output single scalar logit per position
            scores = torch.sigmoid(logits.squeeze(-1))
            scores = scores * token_masks.float()
            
            all_scores = []
            for i in range(scores.size(0)):
                sample_scores = scores[i][token_masks[i]].cpu().tolist()
                all_scores.append(sample_scores)
            
            return all_scores
        
        else:
            raise ValueError(f"Unknown extraction method: {self.prm_config.extraction_method}")
    
    def score_prm(self, texts: List[str]) -> List[List[float]]:
        """
        Score each step in formatted texts using PRM.
        
        Args:
            texts: List of formatted texts (with separator tokens already inserted)
            
        Returns:
            List of score lists, one per text, one score per step in [0, 1]
            
        Raises:
            RuntimeError: If model not loaded
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialize with load_model=True.")
        
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs[0]  # (batch, seq_len, num_classes)
        
        # Create mask for separator token positions
        token_masks = (inputs.input_ids == self.separator_token_id)
        
        return self._extract_scores_from_logits(logits, token_masks)
    
    def score_orm(self, texts: List[str]) -> List[float]:
        """
        Score complete solutions using ORM (takes last score only).
        
        Args:
            texts: List of complete solution texts (separator will be appended)
            
        Returns:
            List of single scores, one per text, in [0, 1]
        """
        formatted = [self.format_for_orm(t) for t in texts]
        all_scores = self.score_prm(formatted)
        # ORM uses only the final (and only) score
        return [scores[-1] if scores else 0.0 for scores in all_scores]
    
    def get_latest_step_scores(
        self, texts: List[str], device: Optional[str] = None
    ) -> torch.Tensor:
        """
        Get PRM score for just the latest step in each text.

        Used for SMC weight updates where we only care about the
        most recently generated step.

        Args:
            texts: List of texts (with separator tokens for each step)
            device: Device to place output tensor on. If None, uses model device
                    or CPU if model not loaded.

        Returns:
            Tensor of scores, shape (n_texts,), values in [0, 1]
        """
        all_scores = self.score_prm(texts)
        latest_scores = [scores[-1] if scores else 0.0 for scores in all_scores]

        # Determine output device
        if device is None:
            if self.model is not None and hasattr(self.model, 'device'):
                device = self.model.device
            else:
                device = "cpu"

        return torch.tensor(latest_scores, device=device)
    
    def format_text_for_scoring(self, text: str) -> str:
        """
        Prepare a raw text (with step patterns) for PRM scoring.

        Splits text into preamble (prompt) and steps, then formats with separator
        tokens. The preamble is prepended to the first step so the PRM has
        problem context when scoring.

        Format: preamble + step1<sep> + step2<sep> + ...

        Args:
            text: Raw text with "## Step N:" patterns

        Returns:
            Formatted text ready for score_prm()
        """
        import warnings

        if not text or not text.strip():
            warnings.warn(
                "format_text_for_scoring called with empty or whitespace-only text. "
                "This returns just a separator token, which may produce "
                "unexpected PRM scores. Consider checking for empty input upstream.",
                UserWarning,
                stacklevel=2
            )
            return self.prm_config.step_separator_token

        preamble, steps = self._split_into_steps_with_preamble(text)

        if not steps:
            # No steps found, return text with separator
            return text + self.prm_config.step_separator_token

        # Prepend preamble to first step if present
        if preamble:
            steps[0] = preamble + "\n\n" + steps[0]

        return self.format_for_prm(steps)
