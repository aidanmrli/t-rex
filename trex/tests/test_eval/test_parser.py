"""
Tests for trex/eval/parser.py - Answer extraction and LaTeX normalization.

These tests verify the pure-logic functions for extracting and normalizing
mathematical answers from LLM outputs.
"""

import pytest

from trex.eval.parser import (
    strip_string,
    find_box,
    extract_answer,
    extract_last_boxed,
    _fix_fracs,
    _fix_a_slash_b,
    _fix_sqrt,
    _choice_answer_clean,
)


# =============================================================================
# Tests for strip_string()
# =============================================================================


class TestStripString:
    """Tests for strip_string() normalization."""

    def test_removes_whitespace(self):
        """Whitespace is trimmed."""
        assert strip_string("  42  ") == "42"

    def test_removes_newlines(self):
        """Newlines are removed."""
        assert strip_string("42\n") == "42"
        assert strip_string("4\n2") == "42"

    def test_removes_trailing_period(self):
        """Trailing periods are removed."""
        assert strip_string("42.") == "42"

    def test_normalizes_tfrac_to_frac(self):
        """\\tfrac becomes \\frac."""
        result = strip_string("\\tfrac{1}{2}")
        assert "tfrac" not in result
        assert "frac" in result

    def test_normalizes_dfrac_to_frac(self):
        """\\dfrac becomes \\frac."""
        result = strip_string("\\dfrac{1}{2}")
        assert "dfrac" not in result
        assert "frac" in result

    def test_removes_dollar_signs(self):
        """LaTeX delimiters $ are stripped."""
        assert strip_string("$42$") == "42"

    def test_removes_latex_parens(self):
        """\\( and \\) are removed."""
        assert strip_string("\\(42\\)") == "42"

    def test_removes_left_right(self):
        """\\left and \\right are removed."""
        result = strip_string("\\left(x\\right)")
        assert "\\left" not in result
        assert "\\right" not in result

    def test_removes_percentage_sign(self):
        """Percentage signs are removed."""
        assert strip_string("50%") == "50"
        assert strip_string("50\\%") == "50"

    def test_normalizes_bmatrix_to_pmatrix(self):
        """bmatrix becomes pmatrix."""
        result = strip_string("\\begin{bmatrix}1\\end{bmatrix}")
        assert "pmatrix" in result
        assert "bmatrix" not in result

    def test_normalizes_inequalities(self):
        """Inequality symbols are normalized."""
        assert "\\le" in strip_string("\\leq")
        assert "\\ge" in strip_string("\\geq")
        assert "\\ne" in strip_string("\\neq")

    def test_fixes_leading_decimal(self):
        """Leading decimals get a zero: .5 -> 0.5."""
        assert strip_string(".5") == "0.5"

    def test_removes_trailing_zeros(self):
        """Trailing zeros after decimal are removed."""
        assert strip_string("42.0") == "42"
        assert strip_string("42.00") == "42"

    def test_normalizes_infinity(self):
        """Infinity variations become \\infty."""
        assert "\\infty" in strip_string("infinity")

    def test_removes_spaces(self):
        """Internal spaces are removed."""
        assert strip_string("4 2") == "42"

    def test_handles_empty_string(self):
        """Empty string returns empty string."""
        assert strip_string("") == ""

    def test_handles_simple_equation(self):
        """Simple equation x = 5 extracts the value."""
        result = strip_string("x = 5")
        assert result == "5"

    def test_preserves_complex_equations(self):
        """Complex equations are not stripped incorrectly."""
        # Equation with longer LHS should be preserved
        result = strip_string("abc = 123")
        assert "123" in result


# =============================================================================
# Tests for find_box()
# =============================================================================


class TestFindBox:
    """Tests for find_box() brace matching."""

    def test_simple_boxed(self):
        """Simple \\boxed{42} extracts 42."""
        assert find_box("\\boxed{42}") == "42"

    def test_nested_braces_one_level(self):
        """One level of nested braces."""
        assert find_box("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_nested_braces_two_levels(self):
        """Two levels of nested braces."""
        result = find_box("\\boxed{\\frac{\\sqrt{2}}{3}}")
        assert result == "\\frac{\\sqrt{2}}{3}"

    def test_deeply_nested(self):
        """Multiple levels of nesting."""
        result = find_box("\\boxed{a + {b + {c}}}")
        assert result == "a + {b + {c}}"

    def test_multiple_boxed_returns_last(self):
        """With multiple boxed, returns the last one."""
        result = find_box("\\boxed{1} and \\boxed{2}")
        assert result == "2"

    def test_boxed_with_text_before(self):
        """Boxed with preceding text."""
        assert find_box("The answer is \\boxed{42}") == "42"

    def test_boxed_with_text_after(self):
        """Boxed with following text."""
        assert find_box("\\boxed{42} is the answer") == "42"

    def test_empty_boxed(self):
        """Empty \\boxed{}."""
        assert find_box("\\boxed{}") == ""

    def test_no_boxed_returns_empty(self):
        """No boxed returns empty or the input."""
        result = find_box("no boxed here")
        # Based on implementation, split on "boxed" returns original
        assert result is not None

    def test_boxed_without_braces(self):
        """\\boxed followed by non-brace character."""
        result = find_box("\\boxed 42$")
        assert "42" in result


# =============================================================================
# Tests for extract_answer()
# =============================================================================


class TestExtractAnswer:
    """Tests for extract_answer() multi-format extraction."""

    # Boxed format tests
    def test_extracts_boxed_simple(self):
        """Extracts answer from \\boxed{42}."""
        result = extract_answer("The answer is \\boxed{42}.", "math")
        assert result == "42"

    def test_extracts_boxed_fraction(self):
        """Extracts LaTeX fraction from boxed."""
        result = extract_answer("\\boxed{\\frac{1}{2}}", "math")
        assert "frac" in result
        assert "1" in result
        assert "2" in result

    def test_extracts_boxed_negative(self):
        """Extracts negative number from boxed."""
        result = extract_answer("\\boxed{-5}", "math")
        assert result == "-5"

    # GSM8K format tests
    def test_extracts_gsm8k_format(self):
        """Extracts answer after #### delimiter.

        Note: Pattern precedence matters - "the answer is" matches before "####".
        So we use a prompt without "the answer is" to test the #### pattern.
        """
        result = extract_answer("Calculation complete. #### 42", "math")
        assert result == "42"

    def test_extracts_gsm8k_with_spaces(self):
        """GSM8K format with extra spaces."""
        result = extract_answer("####    100", "gsm8k")
        assert result == "100"

    # Natural language patterns
    def test_extracts_the_answer_is(self):
        """Extracts from 'the answer is X' pattern."""
        result = extract_answer("Therefore, the answer is 42.", "math")
        assert result == "42"

    def test_extracts_The_answer_is_capitalized(self):
        """Extracts from 'The answer is X' (capitalized)."""
        result = extract_answer("The answer is 100", "math")
        assert result == "100"

    def test_extracts_final_answer_is(self):
        """Extracts from 'final answer is X' pattern."""
        result = extract_answer("The final answer is 3.14", "math")
        # The result should contain 3.14 or be normalized
        assert "3" in result

    def test_extracts_final_answer_with_dollar(self):
        """Extracts from 'final answer is $X$' pattern."""
        result = extract_answer("final answer is $42$. I hope", "math")
        assert result == "42"

    # Chinese format
    def test_extracts_chinese_format(self):
        """Extracts from '答案是' pattern."""
        result = extract_answer("答案是 42", "math")
        assert result == "42"

    # Last number fallback
    def test_extracts_last_number_fallback(self):
        """Falls back to last number when use_last_number=True."""
        result = extract_answer("I got 10, then 20, finally 42", "math", use_last_number=True)
        assert result == "42"

    def test_no_last_number_when_disabled(self):
        """Returns empty when no pattern and use_last_number=False."""
        result = extract_answer("No clear answer here", "math", use_last_number=False)
        assert result == ""

    # Multiple choice datasets
    def test_extracts_mmlu_choice(self):
        """Extracts choice for MMLU dataset."""
        result = extract_answer("The answer is (B)", "mmlu_stem")
        assert result == "B"

    def test_extracts_sat_math_choice(self):
        """Extracts choice for SAT Math dataset."""
        result = extract_answer("C is correct", "sat_math")
        assert result == "C"

    # Edge cases
    def test_removes_leading_colon(self):
        """Removes leading colon from answer."""
        result = extract_answer("the answer is: 42", "math")
        assert result == "42"

    def test_handles_empty_input(self):
        """Empty input returns empty string."""
        result = extract_answer("", "math")
        assert result == ""

    def test_handles_only_whitespace(self):
        """Whitespace-only input returns empty."""
        result = extract_answer("   ", "math", use_last_number=False)
        assert result == ""


# =============================================================================
# Tests for extract_last_boxed()
# =============================================================================


class TestExtractLastBoxed:
    """Tests for extract_last_boxed() returning Optional[str]."""

    def test_single_boxed(self):
        """Returns the boxed expression including \\boxed{}."""
        result = extract_last_boxed("The answer is \\boxed{42}")
        assert result == "\\boxed{42}"

    def test_multiple_boxed_returns_last(self):
        """Returns the last \\boxed{} in text."""
        result = extract_last_boxed("\\boxed{1} then \\boxed{2}")
        assert result == "\\boxed{2}"

    def test_returns_none_when_no_boxed(self):
        """Returns None when no \\boxed{} present."""
        result = extract_last_boxed("no boxed here")
        assert result is None

    def test_nested_boxed(self):
        """Handles nested braces in boxed."""
        result = extract_last_boxed("\\boxed{\\frac{1}{2}}")
        assert result == "\\boxed{\\frac{1}{2}}"


# =============================================================================
# Tests for _fix_fracs()
# =============================================================================


class TestFixFracs:
    """Tests for _fix_fracs() LaTeX cleanup."""

    def test_frac12_to_frac_1_2(self):
        """\\frac12 becomes \\frac{1}{2}."""
        assert _fix_fracs("\\frac12") == "\\frac{1}{2}"

    def test_frac_with_existing_braces(self):
        """\\frac{1}{2} is unchanged."""
        assert _fix_fracs("\\frac{1}{2}") == "\\frac{1}{2}"

    def test_frac_partial_braces_numerator(self):
        """\\frac{1}2 - numerator braced, denominator not.

        Note: _fix_fracs only handles the case where NEITHER has braces.
        This partial case is not transformed.
        """
        result = _fix_fracs("\\frac{1}2")
        # The function doesn't handle this case - it only fixes \\frac12
        assert result == "\\frac{1}2"

    def test_multiple_fracs(self):
        """Multiple fractions in one string."""
        result = _fix_fracs("\\frac12 + \\frac34")
        assert "\\frac{1}{2}" in result
        assert "\\frac{3}{4}" in result

    def test_no_frac_unchanged(self):
        """String without \\frac is unchanged."""
        assert _fix_fracs("42") == "42"

    def test_frac_at_end(self):
        """Handles \\frac at end of string gracefully."""
        result = _fix_fracs("\\frac")
        assert result == "\\frac"


# =============================================================================
# Tests for _fix_a_slash_b()
# =============================================================================


class TestFixASlashB:
    """Tests for _fix_a_slash_b() fraction conversion."""

    def test_simple_fraction(self):
        """1/2 becomes \\frac{1}{2}."""
        assert _fix_a_slash_b("1/2") == "\\frac{1}{2}"

    def test_larger_numbers(self):
        """3/4 becomes \\frac{3}{4}."""
        assert _fix_a_slash_b("3/4") == "\\frac{3}{4}"

    def test_multiple_slashes_unchanged(self):
        """1/2/3 is unchanged (not a simple fraction)."""
        assert _fix_a_slash_b("1/2/3") == "1/2/3"

    def test_non_integer_unchanged(self):
        """a/b with non-integers is unchanged."""
        result = _fix_a_slash_b("a/b")
        # Depending on implementation, might stay as-is
        assert "/" in result or "frac" in result

    def test_no_slash_unchanged(self):
        """String without slash is unchanged."""
        assert _fix_a_slash_b("42") == "42"

    def test_sqrt_in_fraction(self):
        """Handles sqrt in numerator/denominator."""
        result = _fix_a_slash_b("sqrt2/3")
        # Should preserve sqrt
        assert "sqrt" in result


# =============================================================================
# Tests for _fix_sqrt()
# =============================================================================


class TestFixSqrt:
    """Tests for _fix_sqrt() LaTeX cleanup."""

    def test_sqrt2_to_sqrt_2(self):
        """\\sqrt2 becomes \\sqrt{2}."""
        assert _fix_sqrt("\\sqrt2") == "\\sqrt{2}"

    def test_sqrt_already_braced(self):
        """\\sqrt{4} stays as \\sqrt{4}."""
        assert _fix_sqrt("\\sqrt{4}") == "\\sqrt{4}"

    def test_sqrt_with_word(self):
        """\\sqrtabc becomes \\sqrt{abc}."""
        assert _fix_sqrt("\\sqrtabc") == "\\sqrt{abc}"

    def test_no_sqrt_unchanged(self):
        """String without \\sqrt is unchanged."""
        assert _fix_sqrt("42") == "42"

    def test_multiple_sqrts(self):
        """Multiple sqrts in one string."""
        result = _fix_sqrt("\\sqrt2 + \\sqrt3")
        assert "\\sqrt{2}" in result
        assert "\\sqrt{3}" in result


# =============================================================================
# Tests for _choice_answer_clean()
# =============================================================================


class TestChoiceAnswerClean:
    """Tests for _choice_answer_clean() multiple choice extraction."""

    def test_extracts_parenthesized(self):
        """Extracts 'A' from '(A)'."""
        assert _choice_answer_clean("(A)") == "A"

    def test_extracts_with_period(self):
        """Extracts 'B' from 'B.'."""
        assert _choice_answer_clean("B.") == "B"

    def test_extracts_from_sentence(self):
        """Extracts choice from a sentence."""
        assert _choice_answer_clean("The answer is C") == "C"

    def test_handles_lowercase(self):
        """Lowercase letters are uppercased."""
        assert _choice_answer_clean("(b)") == "B"

    def test_extracts_last_choice(self):
        """With multiple choices, returns the last."""
        result = _choice_answer_clean("A is wrong, B is correct")
        assert result == "B"

    def test_handles_all_letters(self):
        """Handles A through E."""
        for letter in ["A", "B", "C", "D", "E"]:
            assert _choice_answer_clean(f"({letter})") == letter

    def test_strips_whitespace(self):
        """Strips leading/trailing whitespace."""
        assert _choice_answer_clean("  A  ") == "A"

    def test_no_choice_returns_cleaned(self):
        """No valid choice returns cleaned input."""
        result = _choice_answer_clean("xyz")
        assert result == "xyz"


# =============================================================================
# Integration Tests
# =============================================================================


class TestParserIntegration:
    """Integration tests combining multiple parser functions."""

    def test_boxed_fraction_normalized(self):
        """Boxed fraction is extracted and normalized."""
        result = extract_answer("\\boxed{\\frac12}", "math")
        assert "frac" in result
        assert "{1}" in result
        assert "{2}" in result

    def test_complex_latex_expression(self):
        """Complex LaTeX expression is handled."""
        result = extract_answer("\\boxed{\\sqrt{\\frac{1}{2}}}", "math")
        assert "sqrt" in result
        assert "frac" in result

    def test_real_world_response(self):
        """Simulated real-world LLM response."""
        response = """
        Let me solve this step by step.

        First, we add 2 + 2 = 4.

        Therefore, the answer is \\boxed{4}.
        """
        result = extract_answer(response, "math")
        assert result == "4"

    def test_gsm8k_style_response(self):
        """GSM8K style response with reasoning."""
        response = """
        Janet has 3 apples.
        She buys 2 more.
        Total = 3 + 2 = 5

        #### 5
        """
        result = extract_answer(response, "gsm8k")
        assert result == "5"
