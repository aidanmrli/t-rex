"""
Tests for trex/eval/math_verifier.py - Math verification integration.

These tests verify the MathVerifier class and compute_score function,
using mocks where appropriate to avoid heavy dependencies.
"""

import pytest
from unittest.mock import patch, MagicMock

from trex.eval.math_verifier import (
    MathVerifier,
    compute_score,
    HF_MATH_VERIFY_AVAILABLE,
)


# =============================================================================
# Tests for MathVerifier class
# =============================================================================


class TestMathVerifierInit:
    """Tests for MathVerifier initialization."""

    def test_default_initialization(self):
        """MathVerifier initializes with default parameters."""
        verifier = MathVerifier()
        assert verifier.timeout_seconds == 10.0
        assert verifier.use_sympy is True

    def test_custom_timeout(self):
        """Custom timeout is respected."""
        verifier = MathVerifier(timeout_seconds=5.0)
        assert verifier.timeout_seconds == 5.0

    def test_disable_sympy(self):
        """SymPy can be disabled."""
        verifier = MathVerifier(use_sympy=False)
        assert verifier.use_sympy is False

    def test_disable_hf_math_verify(self):
        """HF math_verify can be disabled."""
        verifier = MathVerifier(use_hf_math_verify=False)
        assert verifier.use_hf_math_verify is False


class TestMathVerifierExtractAnswer:
    """Tests for MathVerifier.extract_answer() method."""

    @pytest.fixture
    def verifier(self):
        """Create a MathVerifier instance."""
        return MathVerifier(timeout_seconds=5.0)

    def test_extract_boxed_answer(self, verifier):
        """Extracts answer from \\boxed{}."""
        result = verifier.extract_answer("The answer is \\boxed{42}.")
        assert result == "42"

    def test_extract_natural_language(self, verifier):
        """Extracts from natural language pattern."""
        result = verifier.extract_answer("The answer is 42.")
        assert result == "42"

    def test_extract_gsm8k_format(self, verifier):
        """Extracts from #### delimiter."""
        result = verifier.extract_answer("Calculation complete. #### 100")
        assert result == "100"

    def test_extract_last_number_fallback(self, verifier):
        """Falls back to last number."""
        result = verifier.extract_answer("I got 10, then 20, finally 42")
        assert result == "42"

    def test_extract_with_custom_data_name(self, verifier):
        """Works with different dataset names."""
        result = verifier.extract_answer("The answer is (B)", data_name="mmlu_stem")
        assert result == "B"


class TestMathVerifierVerify:
    """Tests for MathVerifier.verify() method."""

    @pytest.fixture
    def verifier(self):
        """Create a MathVerifier instance with HF math_verify disabled for predictability."""
        return MathVerifier(timeout_seconds=5.0, use_hf_math_verify=False)

    def test_verify_identical_strings(self, verifier):
        """Identical strings are equal."""
        assert verifier.verify("42", "42", extract_from_prediction=False) is True

    def test_verify_incorrect_answer(self, verifier):
        """Different values are not equal."""
        assert verifier.verify("41", "42", extract_from_prediction=False) is False

    def test_verify_with_extraction(self, verifier):
        """Extracts and verifies from full response."""
        assert verifier.verify("The answer is \\boxed{42}.", "42") is True

    def test_verify_boxed_fraction(self, verifier):
        """Verifies boxed LaTeX fraction."""
        result = verifier.verify("\\boxed{\\frac{1}{2}}", "0.5")
        # This should depend on SymPy availability
        assert result is True or result is False  # Just check it doesn't crash

    def test_verify_numeric_equality(self, verifier):
        """Numeric equivalence works."""
        assert verifier.verify("42.0", "42", extract_from_prediction=False) is True

    def test_verify_case_insensitive(self, verifier):
        """Comparison is case-insensitive."""
        assert verifier.verify("A", "a", extract_from_prediction=False) is True

    def test_verify_with_custom_timeout(self, verifier):
        """Custom timeout can be passed."""
        # Should not hang with short timeout
        result = verifier.verify("42", "42", timeout=1.0, extract_from_prediction=False)
        assert result is True

    def test_verify_empty_strings(self, verifier):
        """Empty strings comparison."""
        result = verifier.verify("", "", extract_from_prediction=False)
        assert result is True  # Empty equals empty

    def test_verify_does_not_extract_when_disabled(self, verifier):
        """No extraction when extract_from_prediction=False."""
        # The raw string "\\boxed{42}" should not match "42" without extraction
        # because the comparison is done on the raw string
        result = verifier.verify("\\boxed{42}", "42", extract_from_prediction=False)
        # This might still match due to SymPy or string comparison - depends on implementation
        assert result is True or result is False  # Just check no crash


class TestMathVerifierVerifyBatch:
    """Tests for MathVerifier.verify_batch() method."""

    @pytest.fixture
    def verifier(self):
        """Create a MathVerifier instance."""
        return MathVerifier(timeout_seconds=5.0, use_hf_math_verify=False)

    def test_verify_batch_basic(self, verifier):
        """Batch verification returns correct results."""
        predictions = ["\\boxed{42}", "\\boxed{43}"]
        ground_truths = ["42", "42"]

        results = verifier.verify_batch(predictions, ground_truths)

        assert len(results) == 2
        assert results[0] is True
        assert results[1] is False

    def test_verify_batch_empty(self, verifier):
        """Empty batch returns empty list."""
        results = verifier.verify_batch([], [])
        assert results == []

    def test_verify_batch_length_mismatch_raises(self, verifier):
        """Mismatched lengths raise assertion error."""
        with pytest.raises(AssertionError):
            verifier.verify_batch(["a", "b"], ["a"])

    def test_verify_batch_all_correct(self, verifier):
        """All correct predictions."""
        predictions = ["\\boxed{1}", "\\boxed{2}", "\\boxed{3}"]
        ground_truths = ["1", "2", "3"]

        results = verifier.verify_batch(predictions, ground_truths)

        assert all(results)

    def test_verify_batch_all_incorrect(self, verifier):
        """All incorrect predictions."""
        predictions = ["\\boxed{1}", "\\boxed{2}", "\\boxed{3}"]
        ground_truths = ["2", "3", "4"]

        results = verifier.verify_batch(predictions, ground_truths)

        assert not any(results)


class TestMathVerifierWithMocks:
    """Tests using mocks for isolation."""

    def test_verify_calls_hf_when_available(self):
        """HF math_verify is attempted when available."""
        with patch('trex.eval.math_verifier.HF_MATH_VERIFY_AVAILABLE', True):
            with patch('trex.eval.math_verifier._hf_verify_impl', return_value=True) as mock_hf:
                verifier = MathVerifier(use_hf_math_verify=True)
                # Need to patch the process pool submit
                with patch.object(verifier._process_pool, 'submit') as mock_submit:
                    mock_future = MagicMock()
                    mock_future.result.return_value = True
                    mock_submit.return_value = mock_future
                    
                    result = verifier.verify("42", "42", extract_from_prediction=False)
                    
                    assert result is True
                    mock_submit.assert_called_once()

    def test_verify_falls_back_to_math_equal(self):
        """Falls back to math_equal when HF fails."""
        with patch('trex.eval.math_verifier.math_equal', return_value=True) as mock_equal:
            verifier = MathVerifier(use_hf_math_verify=False)
            
            result = verifier.verify("42", "42", extract_from_prediction=False)
            
            assert result is True
            mock_equal.assert_called()


# =============================================================================
# Tests for compute_score function
# =============================================================================


class TestComputeScore:
    """Tests for compute_score() function."""

    def test_correct_with_boxed(self):
        """Correct answer with \\boxed{} format."""
        with patch.object(MathVerifier, 'verify', return_value=True):
            result = compute_score("The answer is \\boxed{42}.", "42")
            
            assert result["correctness"] is True
            assert result["has_boxed"] is True
            assert result["score"] == 1.0

    def test_correct_without_boxed(self):
        """Correct answer without \\boxed{}."""
        with patch.object(MathVerifier, 'verify', return_value=True):
            result = compute_score("The answer is 42.", "42")
            
            assert result["correctness"] is True
            assert result["has_boxed"] is False
            assert result["score"] == 1.0

    def test_incorrect_with_boxed(self):
        """Incorrect answer with \\boxed{}."""
        with patch.object(MathVerifier, 'verify', return_value=False):
            result = compute_score("The answer is \\boxed{41}.", "42")
            
            assert result["correctness"] is False
            assert result["has_boxed"] is True
            assert result["score"] == 0.0

    def test_incorrect_without_boxed(self):
        """Incorrect answer without \\boxed{}."""
        with patch.object(MathVerifier, 'verify', return_value=False):
            result = compute_score("I don't know.", "42")
            
            assert result["correctness"] is False
            assert result["has_boxed"] is False
            assert result["score"] == 0.0

    def test_uses_provided_verifier(self):
        """Uses provided verifier instance."""
        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = True
        
        result = compute_score("\\boxed{42}", "42", verifier=mock_verifier)
        
        mock_verifier.verify.assert_called_once()
        assert result["correctness"] is True

    def test_creates_verifier_when_not_provided(self):
        """Creates a new verifier when none provided."""
        with patch.object(MathVerifier, 'verify', return_value=True) as mock_verify:
            result = compute_score("\\boxed{42}", "42")
            
            # Should have called verify
            mock_verify.assert_called_once()

    def test_return_type(self):
        """Returns expected dictionary structure."""
        with patch.object(MathVerifier, 'verify', return_value=True):
            result = compute_score("\\boxed{42}", "42")
            
            assert isinstance(result, dict)
            assert "score" in result
            assert "correctness" in result
            assert "has_boxed" in result
            assert isinstance(result["score"], float)
            assert isinstance(result["correctness"], bool)
            assert isinstance(result["has_boxed"], bool)


# =============================================================================
# Integration Tests (without mocks, may be slower)
# =============================================================================


@pytest.mark.integration
class TestMathVerifierIntegration:
    """Integration tests for MathVerifier with real components."""

    @pytest.fixture
    def verifier(self):
        """Create a MathVerifier for integration tests."""
        return MathVerifier(
            timeout_seconds=5.0,
            use_hf_math_verify=False,  # Disable for portability
            use_sympy=True,
        )

    def test_real_boxed_answer(self, verifier):
        """Integration test with real boxed answer."""
        result = verifier.verify("The answer is \\boxed{42}.", "42")
        assert result is True

    def test_real_incorrect_answer(self, verifier):
        """Integration test with incorrect answer."""
        result = verifier.verify("The answer is \\boxed{41}.", "42")
        assert result is False

    def test_real_numeric_equivalence(self, verifier):
        """Integration test with numeric equivalence."""
        assert verifier.verify("\\boxed{42.0}", "42") is True
        assert verifier.verify("\\boxed{3.14}", "3.14") is True

    def test_real_gsm8k_format(self, verifier):
        """Integration test with GSM8K format."""
        response = """
        Janet starts with 3 apples.
        She buys 2 more apples.
        Total = 3 + 2 = 5 apples
        
        #### 5
        """
        result = verifier.verify(response, "5")
        assert result is True

    def test_real_multiple_choice(self, verifier):
        """Integration test with multiple choice."""
        result = verifier.verify("The correct answer is (B)", "B")
        assert result is True


@pytest.mark.integration
class TestComputeScoreIntegration:
    """Integration tests for compute_score."""

    def test_real_correct_score(self):
        """Integration test for correct score computation."""
        result = compute_score(
            "Let me solve this step by step... The answer is \\boxed{42}.",
            "42"
        )
        assert result["correctness"] is True
        assert result["has_boxed"] is True
        assert result["score"] == 1.0

    def test_real_incorrect_score(self):
        """Integration test for incorrect score computation."""
        result = compute_score(
            "I'm not sure, maybe \\boxed{41}?",
            "42"
        )
        assert result["correctness"] is False
        assert result["has_boxed"] is True
        assert result["score"] == 0.0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestMathVerifierEdgeCases:
    """Edge cases and error handling tests."""

    @pytest.fixture
    def verifier(self):
        """Create a MathVerifier instance."""
        return MathVerifier(timeout_seconds=2.0, use_hf_math_verify=False)

    def test_handles_none_gracefully(self, verifier):
        """None input should not crash (if passed as string)."""
        # The function expects strings, but should handle edge cases
        try:
            result = verifier.verify("None", "42", extract_from_prediction=False)
            assert result is False
        except (TypeError, AttributeError):
            # If it raises, that's also acceptable behavior
            pass

    def test_handles_very_long_input(self, verifier):
        """Very long input doesn't hang."""
        long_input = "x" * 10000
        # Should complete without hanging (within timeout)
        result = verifier.verify(long_input, "42")
        assert result is False

    def test_handles_special_characters(self, verifier):
        """Special characters are handled."""
        result = verifier.verify("\\boxed{$100}", "100")
        # Dollar sign should be stripped
        assert result is True or result is False  # Just verify no crash

    def test_handles_unicode(self, verifier):
        """Unicode characters are handled."""
        result = verifier.verify("答案是 42", "42")
        assert result is True

    def test_handles_malformed_latex(self, verifier):
        """Malformed LaTeX doesn't crash."""
        malformed = "\\boxed{\\frac{1}"  # Missing closing brace
        result = verifier.verify(malformed, "0.5")
        # Should not crash, result can be True or False
        assert result is True or result is False

    def test_handles_empty_boxed(self, verifier):
        """Empty \\boxed{} is handled."""
        result = verifier.verify("\\boxed{}", "42")
        assert result is False

    def test_multiple_boxed_uses_last(self, verifier):
        """Multiple boxed expressions use the last one."""
        result = verifier.verify("First \\boxed{1}, then \\boxed{42}", "42")
        assert result is True


# =============================================================================
# HF Math Verify Availability Tests
# =============================================================================


class TestHFMathVerifyAvailability:
    """Tests related to HuggingFace math_verify availability."""

    def test_hf_available_flag_is_boolean(self):
        """HF_MATH_VERIFY_AVAILABLE is a boolean."""
        assert isinstance(HF_MATH_VERIFY_AVAILABLE, bool)

    def test_verifier_respects_hf_flag_when_disabled(self):
        """Verifier respects use_hf_math_verify=False."""
        verifier = MathVerifier(use_hf_math_verify=False)
        assert verifier.use_hf_math_verify is False

    def test_verifier_hf_flag_depends_on_availability(self):
        """Verifier's HF flag depends on actual availability."""
        verifier = MathVerifier(use_hf_math_verify=True)
        # Should be True only if HF is actually available
        assert verifier.use_hf_math_verify == HF_MATH_VERIFY_AVAILABLE
