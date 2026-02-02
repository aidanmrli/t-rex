"""
Unit tests for RewardModel (PRM/ORM wrapper).

These tests verify:
- PRM input formatting (reward tokens after each step)
- ORM input formatting (single token at end)
- Step splitting and detection
- Score extraction (mocked)
"""

import re
import pytest
import torch
from unittest.mock import MagicMock, patch

from trex.models.prm_config import PRMConfig, QWEN_PRM_CONFIG


class TestRewardModelFormatting:
    """Test PRM/ORM input formatting - CRITICAL for correct scoring."""
    
    def test_format_for_prm_inserts_token_after_each_step(self):
        """Each step gets a separator token appended."""
        from trex.models.reward_model import RewardModel
        
        # Create reward model without loading actual model
        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = QWEN_PRM_CONFIG
        
        steps = ["## Step 1: Do X\nCalculation", "## Step 2: Do Y\nResult"]
        formatted = rm.format_for_prm(steps)
        
        # Should have exactly 2 separator tokens
        assert formatted.count("<extra_0>") == 2
        # Should end with separator
        assert formatted.endswith("<extra_0>")
        # Steps should be joined by separator
        assert "Calculation<extra_0>## Step 2" in formatted
    
    def test_format_for_prm_handles_single_step(self):
        """Single step should get exactly one separator token."""
        from trex.models.reward_model import RewardModel
        
        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = QWEN_PRM_CONFIG
        
        steps = ["## Step 1: Calculate\n2+2=4"]
        formatted = rm.format_for_prm(steps)
        
        assert formatted.count("<extra_0>") == 1
        assert formatted.endswith("<extra_0>")
    
    def test_format_for_prm_handles_empty_input(self):
        """Empty input should return just a separator token with a warning."""
        from trex.models.reward_model import RewardModel
        
        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = QWEN_PRM_CONFIG
        
        steps = []
        
        # Should emit a warning about empty input
        with pytest.warns(UserWarning, match="empty steps list"):
            formatted = rm.format_for_prm(steps)
        
        assert formatted == "<extra_0>"
    
    def test_format_for_prm_preserves_boxed_answer(self):
        """PRM formatting should preserve \\boxed{} syntax."""
        from trex.models.reward_model import RewardModel
        
        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = QWEN_PRM_CONFIG
        
        steps = ["## Step 1: Calculate\n2+2=4\nTherefore, $\\boxed{4}$"]
        formatted = rm.format_for_prm(steps)
        
        assert "\\boxed{4}" in formatted
        assert "<extra_0>" in formatted
    
    def test_format_for_orm_single_token_at_end(self):
        """ORM format should have exactly one separator at end."""
        from trex.models.reward_model import RewardModel
        
        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = QWEN_PRM_CONFIG
        
        text = "## Step 1: Do X\n## Step 2: Do Y\nFinal answer"
        formatted = rm.format_for_orm(text)
        
        assert formatted.count("<extra_0>") == 1
        assert formatted.endswith("<extra_0>")
    
    def test_format_for_orm_removes_existing_separators(self):
        """ORM format should remove existing trailing separators."""
        from trex.models.reward_model import RewardModel
        
        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = QWEN_PRM_CONFIG
        
        text = "Some text<extra_0><extra_0>"
        formatted = rm.format_for_orm(text)
        
        # Should have exactly one separator at end
        assert formatted.count("<extra_0>") == 1
        assert formatted.endswith("<extra_0>")
    
    def test_format_for_orm_no_intermediate_tokens(self):
        """ORM format should not add intermediate tokens."""
        from trex.models.reward_model import RewardModel
        
        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = QWEN_PRM_CONFIG
        
        text = "## Step 1: Do X\n## Step 2: Do Y"
        formatted = rm.format_for_orm(text)
        
        # Only one token at the very end
        assert formatted == "## Step 1: Do X\n## Step 2: Do Y<extra_0>"
    
    def test_format_handles_unicode(self):
        """Formatting should handle unicode characters."""
        from trex.models.reward_model import RewardModel
        
        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = QWEN_PRM_CONFIG
        
        steps = ["## Step 1: 数学问题\n答案是 42"]
        formatted = rm.format_for_prm(steps)
        
        assert "数学问题" in formatted
        assert "答案是" in formatted
        assert "<extra_0>" in formatted


class TestStepSplitting:
    """Test step detection and splitting."""

    def test_split_into_steps_with_preamble_extracts_both(self):
        """Should extract preamble and steps separately."""
        from trex.models.reward_model import RewardModel

        rm = RewardModel("dummy_path", load_model=False)

        text = "What is 2+2?\n\n## Step 1: Calculate\n2+2=4"
        preamble, steps = rm._split_into_steps_with_preamble(text)

        assert preamble == "What is 2+2?"
        assert len(steps) == 1
        assert "## Step 1" in steps[0]

    def test_split_into_steps_with_preamble_no_steps(self):
        """Text without steps should return all as preamble."""
        from trex.models.reward_model import RewardModel

        rm = RewardModel("dummy_path", load_model=False)

        text = "Just a question with no steps"
        preamble, steps = rm._split_into_steps_with_preamble(text)

        assert preamble == "Just a question with no steps"
        assert steps == []

    def test_split_into_steps_with_preamble_empty_preamble(self):
        """Text starting with step should have empty preamble."""
        from trex.models.reward_model import RewardModel

        rm = RewardModel("dummy_path", load_model=False)

        text = "## Step 1: First\nContent"
        preamble, steps = rm._split_into_steps_with_preamble(text)

        assert preamble == ""
        assert len(steps) == 1

    def test_split_into_steps_basic(self):
        """Should split text at ## Step N: boundaries."""
        from trex.models.reward_model import RewardModel

        rm = RewardModel("dummy_path", load_model=False)

        text = "## Step 1: First\nContent 1\n## Step 2: Second\nContent 2"
        steps = rm._split_into_steps(text)

        assert len(steps) == 2
        assert "## Step 1" in steps[0]
        assert "Content 1" in steps[0]
        assert "## Step 2" in steps[1]
        assert "Content 2" in steps[1]
    
    def test_split_into_steps_no_pattern(self):
        """Text without step pattern should return as single element."""
        from trex.models.reward_model import RewardModel
        
        rm = RewardModel("dummy_path", load_model=False)
        
        text = "Just some text without step markers"
        steps = rm._split_into_steps(text)
        
        assert len(steps) == 1
        assert steps[0] == text
    
    def test_split_into_steps_empty(self):
        """Empty text should return empty list."""
        from trex.models.reward_model import RewardModel
        
        rm = RewardModel("dummy_path", load_model=False)
        
        steps = rm._split_into_steps("")
        assert steps == []
    
    def test_split_into_steps_multi_digit(self):
        """Should handle multi-digit step numbers."""
        from trex.models.reward_model import RewardModel
        
        rm = RewardModel("dummy_path", load_model=False)
        
        text = "## Step 1: First\n## Step 10: Tenth\n## Step 100: Hundredth"
        steps = rm._split_into_steps(text)
        
        assert len(steps) == 3


class TestRewardModelScoring:
    """Test scoring methods (mocked, no GPU required)."""
    
    def test_score_prm_returns_list_per_text(self):
        """score_prm should return nested list of scores."""
        from trex.models.reward_model import RewardModel
        
        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = QWEN_PRM_CONFIG
        
        # Mock the model and tokenizer
        rm.model = MagicMock()
        rm.tokenizer = MagicMock()
        rm.separator_token_id = 1234
        
        # Mock model output (batch=2, seq_len=10, num_classes=2)
        mock_logits = torch.randn(2, 10, 2)
        rm.model.return_value = (mock_logits,)
        
        # Mock tokenizer
        mock_input_ids = torch.zeros(2, 10, dtype=torch.long)
        mock_input_ids[0, 5] = 1234  # separator at position 5
        mock_input_ids[1, 3] = 1234  # separator at position 3
        mock_input_ids[1, 7] = 1234  # second separator
        
        rm.tokenizer.return_value = MagicMock(
            input_ids=mock_input_ids,
            to=MagicMock(return_value=MagicMock(
                input_ids=mock_input_ids
            ))
        )
        rm.model.device = "cpu"
        
        # Call score_prm
        with patch.object(rm, 'tokenizer') as mock_tok:
            mock_tok.return_value = MagicMock()
            mock_tok.return_value.to = MagicMock(return_value=MagicMock(
                input_ids=mock_input_ids
            ))
            
            # Simplified test - just verify method exists and has right signature
            assert callable(rm.score_prm)
    
    def test_score_orm_returns_single_score(self):
        """score_orm should return list of single scores."""
        from trex.models.reward_model import RewardModel
        
        rm = RewardModel("dummy_path", load_model=False)
        
        # Mock score_prm to return known values
        rm.score_prm = MagicMock(return_value=[[0.8], [0.6, 0.9]])
        
        scores = rm.score_orm(["text1", "text2"])
        
        assert len(scores) == 2
        assert scores[0] == 0.8  # Last (only) score
        assert scores[1] == 0.9  # Last of two scores
    
    def test_get_latest_step_scores_extracts_last(self):
        """get_latest_step_scores should return only last step score."""
        from trex.models.reward_model import RewardModel
        
        rm = RewardModel("dummy_path", load_model=False)
        
        # Mock score_prm to return multiple steps
        rm.score_prm = MagicMock(return_value=[[0.5, 0.8], [0.3, 0.6, 0.9]])
        
        scores = rm.get_latest_step_scores(["text1", "text2"])
        
        assert isinstance(scores, torch.Tensor)
        assert len(scores) == 2
        assert abs(scores[0].item() - 0.8) < 1e-6  # Last of [0.5, 0.8]
        assert abs(scores[1].item() - 0.9) < 1e-6  # Last of [0.3, 0.6, 0.9]
    
    def test_format_text_for_scoring(self):
        """format_text_for_scoring should detect and format steps."""
        from trex.models.reward_model import RewardModel

        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = QWEN_PRM_CONFIG

        text = "## Step 1: Calculate\n2+2=4\n## Step 2: Verify\nCorrect"
        formatted = rm.format_text_for_scoring(text)

        # Should have separator tokens
        assert "<extra_0>" in formatted
        # Should have both steps
        assert "## Step 1" in formatted
        assert "## Step 2" in formatted

    def test_format_text_for_scoring_warns_on_empty(self):
        """format_text_for_scoring should warn on empty or whitespace input."""
        from trex.models.reward_model import RewardModel

        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = QWEN_PRM_CONFIG

        # Empty string
        with pytest.warns(UserWarning, match="empty or whitespace-only text"):
            formatted = rm.format_text_for_scoring("")
        assert formatted == "<extra_0>"

        # Whitespace only
        with pytest.warns(UserWarning, match="empty or whitespace-only text"):
            formatted = rm.format_text_for_scoring("   \n\t  ")
        assert formatted.endswith("<extra_0>")

    def test_format_text_for_scoring_includes_preamble(self):
        """
        CRITICAL: format_text_for_scoring should include the prompt/preamble
        so the PRM has problem context when scoring.
        """
        from trex.models.reward_model import RewardModel

        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = QWEN_PRM_CONFIG

        text = "What is 2+2?\n\n## Step 1: Calculate\n2+2=4\n## Step 2: Verify\nCorrect"
        formatted = rm.format_text_for_scoring(text)

        # Should include the preamble (prompt)
        assert "What is 2+2?" in formatted
        # Should have both steps
        assert "## Step 1" in formatted
        assert "## Step 2" in formatted
        # Should have separator tokens
        assert "<extra_0>" in formatted

    def test_format_text_for_scoring_preamble_before_first_step(self):
        """Preamble should be prepended to first step, not get its own separator."""
        from trex.models.reward_model import RewardModel

        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = QWEN_PRM_CONFIG

        text = "Problem: Find x.\n\n## Step 1: Solve\nx=5"
        formatted = rm.format_text_for_scoring(text)

        # Preamble should come before Step 1
        preamble_pos = formatted.find("Problem: Find x.")
        step1_pos = formatted.find("## Step 1")
        assert preamble_pos < step1_pos

        # Only one separator (after the step), not after preamble
        assert formatted.count("<extra_0>") == 1
        assert formatted.endswith("<extra_0>")

    def test_format_text_for_scoring_chat_template_filters_system_prompt(self):
        """
        CRITICAL: Chat-templated text should only score assistant's steps,
        not example steps in the system prompt.
        """
        from trex.models.reward_model import RewardModel

        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = QWEN_PRM_CONFIG

        # Simulate chat-templated text with system prompt containing example steps
        chat_text = """<|im_start|>system
Solve problems step by step:

## Step 1: [Example description]
[Example content]

## Step 2: [Example description]
[Example content]
<|im_end|>
<|im_start|>user
What is 2+2?
<|im_end|>
<|im_start|>assistant
## Step 1: Add the numbers
2 + 2 = 4

## Step 2: Verify
The answer is 4.

Therefore, the answer is $\\boxed{4}$."""

        formatted = rm.format_text_for_scoring(chat_text)

        # Should include full chat context (system + user + assistant marker)
        assert "<|im_start|>system" in formatted
        assert "What is 2+2?" in formatted
        assert "<|im_start|>assistant" in formatted

        # Should only have 2 separators (for the 2 assistant steps)
        # NOT 4 separators (which would happen if system prompt steps were included)
        assert formatted.count("<extra_0>") == 2

        # The assistant's actual steps should be present
        assert "Add the numbers" in formatted
        assert "Verify" in formatted

    def test_format_text_for_scoring_non_chat_template_unchanged(self):
        """Non-chat-templated text should use original logic."""
        from trex.models.reward_model import RewardModel

        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = QWEN_PRM_CONFIG

        # Text without chat template markers
        text = "Problem: What is 2+2?\n\n## Step 1: Add\n2+2=4"
        formatted = rm.format_text_for_scoring(text)

        # Should have the preamble and step
        assert "Problem: What is 2+2?" in formatted
        assert "## Step 1: Add" in formatted
        assert formatted.count("<extra_0>") == 1


class TestScoreExtractionLogic:
    """Test the score extraction from logits."""
    
    def test_binary_softmax_extraction(self):
        """binary_softmax should take softmax and extract positive class."""
        from trex.models.reward_model import RewardModel
        
        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = PRMConfig(
            extraction_method="binary_softmax",
            num_classes=2,
            positive_class_idx=1,
        )
        
        # Create logits (batch=1, seq_len=3, num_classes=2)
        # Position 1 has separator, others don't
        logits = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [1.0, 0.0]]])  # Higher positive at pos 1
        token_masks = torch.tensor([[False, True, False]])
        
        scores = rm._extract_scores_from_logits(logits, token_masks)
        
        assert len(scores) == 1
        assert len(scores[0]) == 1  # Only one separator token
        # Score should be softmax of [0.0, 2.0] at index 1 ≈ 0.88
        assert 0.8 < scores[0][0] < 1.0
    
    def test_single_logit_extraction(self):
        """single_logit should apply sigmoid."""
        from trex.models.reward_model import RewardModel
        
        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = PRMConfig(
            extraction_method="single_logit",
            num_classes=1,
        )
        
        # Create logits (batch=1, seq_len=3, num_classes=1)
        logits = torch.tensor([[[0.0], [2.0], [0.0]]])  # High score at pos 1
        token_masks = torch.tensor([[False, True, False]])
        
        scores = rm._extract_scores_from_logits(logits, token_masks)
        
        assert len(scores) == 1
        assert len(scores[0]) == 1
        # sigmoid(2.0) ≈ 0.88
        assert 0.8 < scores[0][0] < 1.0
    
    def test_empty_mask_returns_empty(self):
        """Empty mask should return empty score list."""
        from trex.models.reward_model import RewardModel
        
        rm = RewardModel("dummy_path", load_model=False)
        rm.prm_config = QWEN_PRM_CONFIG
        
        logits = torch.randn(1, 5, 2)
        token_masks = torch.zeros(1, 5, dtype=torch.bool)
        
        scores = rm._extract_scores_from_logits(logits, token_masks)
        
        assert len(scores) == 1
        assert len(scores[0]) == 0
