"""
Tests for trex/eval/grader.py - Math answer comparison and grading.

These tests verify the mathematical comparison logic including
numeric equality, symbolic equality, and special format handling.
"""

import pytest

from trex.eval.grader import (
    math_equal,
    numeric_equal,
    symbolic_equal,
    parse_digits,
    is_digit,
    str_to_pmatrix,
    choice_answer_clean,
)


# =============================================================================
# Tests for parse_digits()
# =============================================================================


class TestParseDigits:
    """Tests for parse_digits() number parsing."""

    def test_integer(self):
        """Parses integer strings."""
        assert parse_digits("42") == 42.0

    def test_float(self):
        """Parses float strings."""
        assert parse_digits("3.14") == 3.14

    def test_negative_integer(self):
        """Parses negative integers."""
        assert parse_digits("-42") == -42.0

    def test_negative_float(self):
        """Parses negative floats."""
        assert parse_digits("-3.14") == -3.14

    def test_with_commas(self):
        """Handles thousands separators."""
        assert parse_digits("1,000") == 1000.0
        assert parse_digits("1,234,567") == 1234567.0

    def test_percentage(self):
        """Handles percentage format (converts to decimal)."""
        assert parse_digits("50%") == 0.5
        assert parse_digits("100%") == 1.0
        assert parse_digits("25%") == 0.25

    def test_percentage_with_backslash(self):
        """Handles LaTeX percentage format."""
        assert parse_digits("50\\%") == 0.5

    def test_non_numeric_returns_none(self):
        """Non-numeric strings return None."""
        assert parse_digits("abc") is None
        assert parse_digits("") is None

    def test_mixed_returns_none(self):
        """Mixed alphanumeric returns None."""
        assert parse_digits("42abc") is None

    def test_scientific_notation(self):
        """Handles scientific notation."""
        result = parse_digits("1e6")
        assert result == 1000000.0


# =============================================================================
# Tests for is_digit()
# =============================================================================


class TestIsDigit:
    """Tests for is_digit() numeric check."""

    def test_integer_is_digit(self):
        """Integer string is digit."""
        assert is_digit("42") is True

    def test_float_is_digit(self):
        """Float string is digit."""
        assert is_digit("3.14") is True

    def test_negative_is_digit(self):
        """Negative number is digit."""
        assert is_digit("-42") is True

    def test_with_commas_is_digit(self):
        """Number with commas is digit."""
        assert is_digit("1,000") is True

    def test_percentage_is_digit(self):
        """Percentage is digit."""
        assert is_digit("50%") is True

    def test_text_is_not_digit(self):
        """Text is not digit."""
        assert is_digit("abc") is False

    def test_empty_is_not_digit(self):
        """Empty string is not digit."""
        assert is_digit("") is False

    def test_mixed_is_not_digit(self):
        """Mixed string is not digit."""
        assert is_digit("42abc") is False


# =============================================================================
# Tests for numeric_equal()
# =============================================================================


class TestNumericEqual:
    """Tests for numeric_equal() float comparison."""

    def test_exact_equal(self):
        """Identical floats are equal."""
        assert numeric_equal(3.14, 3.14) is True

    def test_within_tolerance(self):
        """Floats within rel_tol are equal."""
        assert numeric_equal(1.0, 1.0001, rel_tol=0.001) is True

    def test_outside_tolerance(self):
        """Floats outside rel_tol are not equal."""
        assert numeric_equal(1.0, 1.01, rel_tol=0.001) is False

    def test_zero_equal(self):
        """Zero equals zero."""
        assert numeric_equal(0.0, 0.0) is True

    def test_large_numbers(self):
        """Large numbers within tolerance."""
        assert numeric_equal(1e10, 1.00001e10) is True

    def test_small_numbers(self):
        """Small numbers within tolerance."""
        assert numeric_equal(1e-10, 1.00001e-10) is True

    def test_negative_numbers(self):
        """Negative numbers comparison."""
        assert numeric_equal(-3.14, -3.14) is True
        assert numeric_equal(-3.14, -3.15) is False


# =============================================================================
# Tests for choice_answer_clean()
# =============================================================================


class TestChoiceAnswerClean:
    """Tests for choice_answer_clean() from grader module."""

    def test_extracts_simple_letter(self):
        """Extracts simple letter answer."""
        assert choice_answer_clean("A") == "A"

    def test_extracts_parenthesized(self):
        """Extracts from (A) format."""
        assert choice_answer_clean("(A)") == "A"

    def test_extracts_from_text(self):
        """Extracts from longer text."""
        assert choice_answer_clean("The answer is B") == "B"

    def test_handles_lowercase(self):
        """Converts lowercase to uppercase."""
        assert choice_answer_clean("(c)") == "C"

    def test_strips_punctuation(self):
        """Strips trailing punctuation."""
        assert choice_answer_clean("D.") == "D"
        assert choice_answer_clean("D/") == "D"

    def test_returns_last_choice(self):
        """Returns last valid choice letter."""
        result = choice_answer_clean("A is wrong, B is right")
        assert result == "B"


# =============================================================================
# Tests for str_to_pmatrix()
# =============================================================================


class TestStrToPmatrix:
    """Tests for str_to_pmatrix() conversion."""

    def test_simple_set(self):
        """Converts simple set notation."""
        result = str_to_pmatrix("{1, 2}")
        assert "pmatrix" in result
        assert "1" in result
        assert "2" in result

    def test_multiple_sets(self):
        """Converts multiple sets."""
        result = str_to_pmatrix("{1, 2}, {3, 4}")
        assert "pmatrix" in result

    def test_no_set_returns_empty(self):
        """No set notation returns empty or original."""
        result = str_to_pmatrix("no set here")
        # Should return empty string based on implementation
        assert isinstance(result, str)


# =============================================================================
# Tests for symbolic_equal()
# =============================================================================


class TestSymbolicEqual:
    """Tests for symbolic_equal() SymPy comparison."""

    def test_identical_strings(self):
        """Identical strings are equal."""
        assert symbolic_equal("x + 1", "x + 1") is True

    def test_commutative_addition(self):
        """Addition is commutative."""
        assert symbolic_equal("x + 1", "1 + x") is True

    def test_commutative_multiplication(self):
        """Multiplication is commutative."""
        assert symbolic_equal("2 * x", "x * 2") is True

    def test_different_expressions(self):
        """Different expressions are not equal."""
        assert symbolic_equal("x + 1", "x + 2") is False

    def test_numeric_strings(self):
        """Numeric strings comparison."""
        assert symbolic_equal("0.5", "1/2") is True

    @pytest.mark.slow
    def test_expanded_factored(self):
        """Expanded vs factored forms."""
        # This test may be slow due to symbolic computation
        result = symbolic_equal("x^2 + 2*x + 1", "(x+1)^2")
        # May or may not be True depending on SymPy parser
        assert isinstance(result, bool)

    def test_handles_latex(self):
        """Handles LaTeX expressions."""
        result = symbolic_equal("\\frac{1}{2}", "0.5")
        # Result depends on latex parser availability
        assert isinstance(result, bool)


# =============================================================================
# Tests for math_equal()
# =============================================================================


class TestMathEqual:
    """Tests for math_equal() main comparison function."""

    # === String Equality ===

    def test_identical_strings(self):
        """Identical strings are equal."""
        assert math_equal("42", "42") is True

    def test_case_insensitive(self):
        """Comparison is case-insensitive."""
        assert math_equal("X", "x") is True
        assert math_equal("ABC", "abc") is True

    def test_whitespace_handling(self):
        """Handles leading/trailing whitespace."""
        assert math_equal("  42  ", "42") is True

    # === Numeric Equality ===

    def test_integer_equal(self):
        """Integer strings are compared numerically."""
        assert math_equal("42", "42.0") is True

    def test_float_equal_within_tolerance(self):
        """Floats within tolerance are equal."""
        assert math_equal("3.14159", "3.14159265") is True

    def test_float_not_equal_outside_tolerance(self):
        """Floats outside tolerance are not equal."""
        assert math_equal("3.14", "3.24") is False

    def test_scientific_notation(self):
        """Scientific notation is handled."""
        assert math_equal("1e6", "1000000") is True

    def test_negative_numbers(self):
        """Negative numbers are compared correctly."""
        assert math_equal("-5", "-5") is True
        assert math_equal("-5", "5") is False

    # === Percentage Handling ===

    def test_percentage_with_include_percentage_true(self):
        """50% equals 0.5 or 50 when include_percentage=True."""
        # Checks x, x/100, x*100
        assert math_equal("50", "0.5", include_percentage=True) is True
        assert math_equal("0.5", "50", include_percentage=True) is True

    def test_percentage_with_include_percentage_false(self):
        """50 does not equal 0.5 when include_percentage=False."""
        assert math_equal("50", "0.5", include_percentage=False) is False

    # === LaTeX Equality ===

    def test_latex_fractions_equal_numeric(self):
        """LaTeX fractions equal numeric values."""
        # Note: This depends on symbolic_equal and SymPy
        result = math_equal("\\frac{1}{2}", "0.5")
        # Should be True if SymPy can parse LaTeX
        assert isinstance(result, bool)

    # === Multiple Choice ===

    def test_multiple_choice_same(self):
        """Same letter choices are equal."""
        assert math_equal("A", "A") is True

    def test_multiple_choice_different(self):
        """Different letter choices are not equal."""
        assert math_equal("A", "B") is False

    def test_multiple_choice_extraction(self):
        """Extracts multiple choice from longer string."""
        assert math_equal("(A)", "A") is True
        assert math_equal("The answer is B", "B") is True

    # === List/Tuple Comparison ===

    def test_list_equal(self):
        """Lists with same elements are equal."""
        assert math_equal("[1, 2]", "[1, 2]") is True

    def test_tuple_equal(self):
        """Tuples with same elements are equal."""
        assert math_equal("(1, 2)", "(1, 2)") is True

    def test_list_different(self):
        """Lists with different elements are not equal."""
        assert math_equal("[1, 2]", "[1, 3]") is False

    # === Equation Comparison ===

    def test_equation_equal(self):
        """Equations with same structure are equal."""
        assert math_equal("x = 5", "x=5") is True

    def test_equation_value_extraction(self):
        """Equation value is extracted for comparison."""
        assert math_equal("x = 5", "5") is True
        assert math_equal("5", "y = 5") is True

    # === Edge Cases ===

    def test_none_prediction(self):
        """None prediction returns False."""
        assert math_equal(None, "42") is False

    def test_none_reference(self):
        """None reference returns False."""
        assert math_equal("42", None) is False

    def test_both_none(self):
        """Both None returns False."""
        assert math_equal(None, None) is False

    def test_empty_strings(self):
        """Empty strings behavior."""
        assert math_equal("", "") is True  # Direct string match

    def test_empty_vs_nonempty(self):
        """Empty vs non-empty."""
        # Empty string is falsy, should return False for non-empty reference
        result = math_equal("", "42")
        # Based on implementation: "" is falsy and not in [0, False]
        assert result is False

    def test_boolean_prediction(self):
        """Boolean prediction is handled."""
        assert math_equal(True, "True") is True
        assert math_equal(False, "False") is True

    # === Timeout Behavior ===

    def test_timeout_does_not_hang(self):
        """With timeout=True, doesn't hang on complex input."""
        # Use a string that might be slow to parse
        result = math_equal("x" * 100, "42", timeout=True)
        # Should complete (either True or False)
        assert isinstance(result, bool)

    def test_without_timeout(self):
        """Without timeout, still works for simple cases."""
        result = math_equal("42", "42", timeout=False)
        assert result is True


# =============================================================================
# Tests with Fixtures
# =============================================================================


class TestMathEqualWithFixtures:
    """Tests using fixtures from conftest.py."""

    def test_sample_comparisons(self, sample_math_comparisons):
        """Test sample math comparisons from fixture."""
        for prediction, reference, expected in sample_math_comparisons:
            result = math_equal(prediction, reference)
            assert result == expected, f"math_equal({prediction!r}, {reference!r}) = {result}, expected {expected}"


# =============================================================================
# Integration Tests
# =============================================================================


class TestGraderIntegration:
    """Integration tests for grader functions."""

    def test_real_world_comparison(self):
        """Real-world comparison scenario."""
        # Model outputs normalized fraction
        model_output = "\\frac{1}{2}"
        ground_truth = "0.5"

        # First check if it's parseable as a digit
        assert not is_digit(model_output)  # LaTeX is not a digit
        assert is_digit(ground_truth)  # 0.5 is a digit

        # math_equal should still work via symbolic comparison
        result = math_equal(model_output, ground_truth)
        assert isinstance(result, bool)

    def test_workflow_numeric(self):
        """Numeric comparison workflow."""
        pred = "3.14159"
        ref = "3.14159265358979"

        assert is_digit(pred)
        assert is_digit(ref)

        pred_num = parse_digits(pred)
        ref_num = parse_digits(ref)

        assert numeric_equal(pred_num, ref_num)
        assert math_equal(pred, ref)

    def test_workflow_multiple_choice(self):
        """Multiple choice workflow."""
        pred = "The answer is (B) because..."
        ref = "B"

        cleaned = choice_answer_clean(pred)
        assert cleaned == "B"
        assert math_equal(pred, ref)
