# Unit Testing Plan for T-REX

**Last Updated:** 2026-01-28

**NOTE:** This plan should be updated as tests are implemented. Mark items with ✅ when complete.

---

## 1. Testing Philosophy

### 1.1 Why Unit Test This Codebase?

1. **Pure logic components** don't require LLMs: math verification, parsing, resampling, temperature schedules
2. **Cost of bugs is high**: errors in resampling or acceptance probability calculations silently poison experiments
3. **Fast feedback loop**: LLM loading takes 30+ seconds; unit tests should run in < 1 second
4. **Correctness matters**: probabilistic algorithms have known invariants we can verify

### 1.2 What to Unit Test vs. What NOT to Test

| ✅ DO Test | ❌ DON'T Test |
|-----------|--------------|
| Answer parsing/normalization | LLM output content |
| Math comparison logic | Training convergence |
| Resampling algorithms | Hyperparameter sensitivity |
| Temperature schedules | Full end-to-end pipelines |
| Weight normalization | Exact model outputs |
| Config validation | WandB logging calls |

### 1.3 Test Categories

| Category | Speed | LLM Required | Purpose |
|----------|-------|--------------|---------|
| **Unit** | < 100ms | No | Isolated function testing |
| **Integration** | < 10s | Mock | Component interaction |
| **Smoke** | < 60s | Optional | Pipeline doesn't crash |

---

## 2. Directory Structure

```
trex/
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures, pytest config
│   ├── test_eval/
│   │   ├── __init__.py
│   │   ├── test_parser.py             # Answer extraction, LaTeX normalization
│   │   ├── test_grader.py             # Math comparison, symbolic equality
│   │   └── test_math_verifier.py      # MathVerifier class, backends
│   ├── test_baselines/
│   │   ├── __init__.py
│   │   ├── test_config.py             # Config validation
│   │   ├── test_reward_functions.py   # Reward computation (mocked verifier)
│   │   └── test_checkpoint_manager.py # Checkpoint save/load
│   ├── test_smc/                      # TDD: Write before implementing
│   │   ├── __init__.py
│   │   ├── test_resampling.py         # Multinomial, systematic, stratified
│   │   ├── test_particle_filter.py    # Particle count, weight invariants
│   │   └── test_twisted_smc.py        # Twist ratio computation
│   ├── test_tempering/                # TDD: Write before implementing
│   │   ├── __init__.py
│   │   ├── test_temperature_ladder.py # Beta schedule generation
│   │   ├── test_swap_schedule.py      # Non-reversible alternating schedule
│   │   └── test_exchange.py           # Acceptance probability
│   └── test_transport/                # TDD: Write before implementing
│       ├── __init__.py
│       └── test_block_gibbs.py        # Mask selection, acceptance ratio
```

---

## 3. Existing Code: Test Specifications

### 3.1 `trex/eval/parser.py`

**Functions to Test:**
- `strip_string(string, skip_unit)` - Math expression normalization
- `extract_answer(pred_str, data_name, use_last_number)` - Multi-format extraction
- `find_box(pred_str)` - Extract `\boxed{...}` content
- `extract_last_boxed(text)` - Extract last boxed answer
- `_fix_fracs(string)` - LaTeX fraction fixes
- `_fix_a_slash_b(string)` - Convert `a/b` to LaTeX
- `_fix_sqrt(string)` - Fix `\sqrt2` → `\sqrt{2}`
- `_choice_answer_clean(pred)` - Multiple choice extraction

**Test Cases:**

```python
# test_parser.py

class TestStripString:
    """Tests for strip_string() normalization."""

    def test_removes_whitespace(self):
        """Whitespace is normalized."""
        assert strip_string("  42  ") == "42"

    def test_fixes_latex_fractions(self):
        """\\frac12 becomes \\frac{1}{2}."""
        assert strip_string("\\frac12") == "\\frac{1}{2}"

    def test_converts_inline_fractions(self):
        """1/2 becomes \\frac{1}{2}."""
        assert strip_string("1/2") == "\\frac{1}{2}"

    def test_normalizes_sqrt(self):
        """\\sqrt2 becomes \\sqrt{2}."""
        assert strip_string("\\sqrt2") == "\\sqrt{2}"

    def test_removes_dollar_signs(self):
        """LaTeX delimiters are stripped."""
        assert strip_string("$42$") == "42"

    def test_handles_units_when_skip_true(self):
        """Units are removed when skip_unit=True."""
        # Exact behavior TBD based on implementation
        pass

    def test_preserves_negative_numbers(self):
        """-42 remains -42."""
        assert strip_string("-42") == "-42"

    def test_normalizes_matrix_notation(self):
        """Matrix notation is standardized."""
        # e.g., pmatrix, bmatrix
        pass


class TestExtractAnswer:
    """Tests for extract_answer() multi-format extraction."""

    def test_extracts_boxed_answer(self):
        """Extracts answer from \\boxed{42}."""
        assert extract_answer("The answer is \\boxed{42}.", "math") == "42"

    def test_extracts_nested_boxed(self):
        """Handles nested braces in \\boxed{...}."""
        assert extract_answer("\\boxed{\\frac{1}{2}}", "math") == "\\frac{1}{2}"

    def test_extracts_gsm8k_format(self):
        """Extracts answer after #### delimiter."""
        assert extract_answer("So the answer is #### 42", "gsm8k") == "42"

    def test_extracts_the_answer_is_format(self):
        """Extracts from 'the answer is X' pattern."""
        assert extract_answer("Therefore, the answer is 42.", "math") == "42"

    def test_extracts_final_answer_format(self):
        """Extracts from 'final answer is X' pattern."""
        assert extract_answer("The final answer is 42.", "math") == "42"

    def test_extracts_last_number_when_no_pattern(self):
        """Falls back to last number when use_last_number=True."""
        assert extract_answer("I got 10, then 20, finally 42", "math", use_last_number=True) == "42"

    def test_returns_none_for_no_answer(self):
        """Returns None or empty when no answer found."""
        result = extract_answer("No answer here.", "math", use_last_number=False)
        assert result is None or result == ""

    def test_handles_multiple_choice(self):
        """Extracts letter answer for MMLU-style."""
        assert extract_answer("The answer is (B)", "mmlu") in ["B", "(B)"]

    def test_handles_chinese_format(self):
        """Extracts from '答案是' pattern."""
        assert extract_answer("答案是 42", "math") == "42"


class TestFindBox:
    """Tests for find_box() brace matching."""

    def test_simple_boxed(self):
        """Simple \\boxed{42}."""
        assert find_box("\\boxed{42}") == "42"

    def test_nested_braces(self):
        """Nested braces are matched correctly."""
        assert find_box("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_deeply_nested(self):
        """Multiple levels of nesting."""
        assert find_box("\\boxed{a + {b + {c}}}") == "a + {b + {c}}"

    def test_multiple_boxed_returns_first(self):
        """Returns first boxed content (or last, verify behavior)."""
        result = find_box("\\boxed{1} and \\boxed{2}")
        assert result in ["1", "2"]  # Verify expected behavior

    def test_unmatched_braces(self):
        """Handles malformed input gracefully."""
        # Should not crash
        result = find_box("\\boxed{unmatched")
        # Verify behavior (None, empty string, or partial)


class TestExtractLastBoxed:
    """Tests for extract_last_boxed() returning Optional[str]."""

    def test_returns_last_when_multiple(self):
        """Returns the last \\boxed{} in text."""
        assert extract_last_boxed("\\boxed{1} then \\boxed{2}") == "2"

    def test_returns_none_when_no_boxed(self):
        """Returns None when no \\boxed{} present."""
        assert extract_last_boxed("no boxed here") is None


class TestFixFracs:
    """Tests for _fix_fracs() LaTeX cleanup."""

    def test_frac12_to_frac_1_2(self):
        """\\frac12 becomes \\frac{1}{2}."""
        assert _fix_fracs("\\frac12") == "\\frac{1}{2}"

    def test_frac_with_existing_braces(self):
        """\\frac{1}{2} is unchanged."""
        assert _fix_fracs("\\frac{1}{2}") == "\\frac{1}{2}"

    def test_dfrac_also_fixed(self):
        """\\dfrac12 becomes \\dfrac{1}{2}."""
        assert _fix_fracs("\\dfrac12") == "\\dfrac{1}{2}"


class TestChoiceAnswerClean:
    """Tests for _choice_answer_clean() multiple choice extraction."""

    def test_extracts_letter(self):
        """Extracts 'A' from various formats."""
        assert _choice_answer_clean("(A)") == "A"
        assert _choice_answer_clean("A.") == "A"
        assert _choice_answer_clean("A)") == "A"

    def test_handles_lowercase(self):
        """Lowercase letters are uppercased."""
        assert _choice_answer_clean("(b)") == "B"

    def test_handles_answer_prefix(self):
        """Handles 'Answer: A' format."""
        assert _choice_answer_clean("Answer: A") == "A"
```

**Priority:** HIGH - These are the most critical pure-logic functions

---

### 3.2 `trex/eval/grader.py`

**Functions to Test:**
- `math_equal(prediction, reference, ...)` - Main comparison
- `numeric_equal(prediction, reference, rel_tol)` - Float comparison
- `symbolic_equal(a, b)` - SymPy symbolic comparison
- `parse_digits(num)` - Number parsing
- `is_digit(num)` - Numeric check
- `str_to_pmatrix(input_str)` - Set to matrix conversion
- `choice_answer_clean(pred)` - Multiple choice extraction

**Test Cases:**

```python
# test_grader.py

class TestMathEqual:
    """Tests for math_equal() comparison logic."""

    # String equality
    def test_identical_strings(self):
        """Identical strings are equal."""
        assert math_equal("42", "42") == True

    def test_case_insensitive(self):
        """Comparison is case-insensitive."""
        assert math_equal("X", "x") == True

    # Numeric equality
    def test_integer_equal(self):
        """Integer strings are compared numerically."""
        assert math_equal("42", "42.0") == True

    def test_float_equal_within_tolerance(self):
        """Floats within ~0.01% are equal."""
        assert math_equal("3.14159", "3.14159265") == True

    def test_float_not_equal_outside_tolerance(self):
        """Floats outside tolerance are not equal."""
        assert math_equal("3.14", "3.24") == False

    def test_scientific_notation(self):
        """Scientific notation is handled."""
        assert math_equal("1e6", "1000000") == True

    # Percentage handling
    def test_percentage_with_include_percentage(self):
        """50% equals 0.5 when include_percentage=True."""
        assert math_equal("50%", "0.5", include_percentage=True) == True

    def test_percentage_without_include_percentage(self):
        """50% does not equal 0.5 when include_percentage=False."""
        assert math_equal("50%", "0.5", include_percentage=False) == False

    # LaTeX equality
    def test_latex_fractions_equal(self):
        """\\frac{1}{2} equals 0.5."""
        assert math_equal("\\frac{1}{2}", "0.5") == True

    def test_latex_sqrt(self):
        """\\sqrt{4} equals 2."""
        assert math_equal("\\sqrt{4}", "2") == True

    # Multiple choice
    def test_multiple_choice_same(self):
        """Same letter choices are equal."""
        assert math_equal("A", "A") == True

    def test_multiple_choice_different(self):
        """Different letter choices are not equal."""
        assert math_equal("A", "B") == False

    # Matrix/vector
    def test_matrix_equal(self):
        """Matrices with same elements are equal."""
        # Verify exact format based on implementation
        pass

    # Equation handling
    def test_equation_form(self):
        """x = 5 style equations."""
        assert math_equal("x = 5", "5") == True  # or verify behavior

    # Edge cases
    def test_empty_strings(self):
        """Empty strings behavior."""
        # Verify: equal to each other? False?
        pass

    def test_none_handling(self):
        """None input handling."""
        # Should not crash
        pass

    def test_invalid_latex(self):
        """Malformed LaTeX doesn't crash."""
        result = math_equal("\\frac{", "0.5")
        # Should return False, not crash


class TestNumericEqual:
    """Tests for numeric_equal() float comparison."""

    def test_exact_equal(self):
        """Identical floats are equal."""
        assert numeric_equal("3.14", "3.14") == True

    def test_within_tolerance(self):
        """Floats within rel_tol are equal."""
        assert numeric_equal("1.0", "1.0001", rel_tol=0.001) == True

    def test_outside_tolerance(self):
        """Floats outside rel_tol are not equal."""
        assert numeric_equal("1.0", "1.01", rel_tol=0.001) == False

    def test_non_numeric_returns_false(self):
        """Non-numeric strings return False."""
        assert numeric_equal("abc", "123") == False


class TestSymbolicEqual:
    """Tests for symbolic_equal() SymPy comparison."""

    def test_algebraically_equal(self):
        """Algebraically equivalent expressions."""
        assert symbolic_equal("x + 1", "1 + x") == True

    def test_expanded_form(self):
        """Expanded vs factored forms."""
        assert symbolic_equal("x^2 + 2x + 1", "(x+1)^2") == True

    def test_different_expressions(self):
        """Different expressions are not equal."""
        assert symbolic_equal("x + 1", "x + 2") == False

    def test_latex_expressions(self):
        """LaTeX expressions are parsed."""
        assert symbolic_equal("\\frac{x}{2}", "0.5x") == True

    def test_timeout_does_not_hang(self):
        """Complex expressions don't hang indefinitely."""
        # This might need a mock or actual timeout test
        pass


class TestParseDigits:
    """Tests for parse_digits() number parsing."""

    def test_integer(self):
        """Parses integer strings."""
        assert parse_digits("42") == 42.0

    def test_float(self):
        """Parses float strings."""
        assert parse_digits("3.14") == 3.14

    def test_with_commas(self):
        """Handles thousands separators."""
        assert parse_digits("1,000") == 1000.0

    def test_percentage(self):
        """Handles percentage format."""
        assert parse_digits("50%") == 0.5  # or 50, verify behavior

    def test_negative(self):
        """Handles negative numbers."""
        assert parse_digits("-42") == -42.0


class TestIsDigit:
    """Tests for is_digit() numeric check."""

    def test_integer_is_digit(self):
        assert is_digit("42") == True

    def test_float_is_digit(self):
        assert is_digit("3.14") == True

    def test_negative_is_digit(self):
        assert is_digit("-42") == True

    def test_text_is_not_digit(self):
        assert is_digit("abc") == False

    def test_mixed_is_not_digit(self):
        assert is_digit("42abc") == False
```

**Priority:** HIGH - Core comparison logic

---

### 3.3 `trex/eval/math_verifier.py`

**Classes/Functions to Test:**
- `MathVerifier` class
  - `extract_answer(text, data_name, use_last_number)`
  - `verify(prediction, ground_truth, ...)`
  - `verify_batch(predictions, ground_truths, ...)`
- `compute_score(solution_str, ground_truth, verifier)`

**Test Cases:**

```python
# test_math_verifier.py

class TestMathVerifier:
    """Tests for MathVerifier class."""

    @pytest.fixture
    def verifier(self):
        """Create a MathVerifier instance."""
        return MathVerifier(timeout_seconds=5, use_sympy=True)

    def test_verify_correct_answer(self, verifier):
        """Correct answer returns True."""
        assert verifier.verify("42", "42") == True

    def test_verify_incorrect_answer(self, verifier):
        """Incorrect answer returns False."""
        assert verifier.verify("41", "42") == False

    def test_verify_with_extraction(self, verifier):
        """Extracts and verifies from full response."""
        assert verifier.verify(
            "The answer is \\boxed{42}.",
            "42",
            extract_from_prediction=True
        ) == True

    def test_verify_latex_vs_numeric(self, verifier):
        """LaTeX and numeric are compared correctly."""
        assert verifier.verify("\\frac{1}{2}", "0.5") == True

    def test_verify_batch(self, verifier):
        """Batch verification returns list of bools."""
        results = verifier.verify_batch(
            predictions=["42", "43"],
            ground_truths=["42", "42"]
        )
        assert results == [True, False]

    def test_timeout_does_not_hang(self, verifier):
        """Verification times out gracefully."""
        # Use a pathological input that could hang SymPy
        result = verifier.verify("x" * 10000, "42", timeout=1)
        # Should return False, not hang
        assert result == False or result == True  # Just verify it completes


class TestComputeScore:
    """Tests for compute_score() function."""

    def test_correct_with_boxed(self):
        """Correct answer with \\boxed{} format."""
        result = compute_score("The answer is \\boxed{42}.", "42")
        assert result["correctness"] == True
        assert result["has_boxed"] == True
        assert result["score"] == 1.0

    def test_correct_without_boxed(self):
        """Correct answer without \\boxed{}."""
        result = compute_score("The answer is 42.", "42")
        assert result["correctness"] == True
        assert result["has_boxed"] == False
        # Score depends on implementation (might be < 1.0)

    def test_incorrect_with_boxed(self):
        """Incorrect answer with \\boxed{}."""
        result = compute_score("The answer is \\boxed{41}.", "42")
        assert result["correctness"] == False
        assert result["has_boxed"] == True

    def test_no_answer_found(self):
        """No answer extractable."""
        result = compute_score("I don't know.", "42")
        assert result["correctness"] == False
```

**Priority:** HIGH - Integration point for all verification

---

## 4. TDD: Future SMC Components

These tests should be written BEFORE implementing the corresponding modules.

### 4.1 `trex/smc/resampling.py` (Phase 1.2)

**Specification from IMPLEMENTATION_PLAN.md:**
- Multinomial, systematic, stratified resampling
- Must preserve particle count
- Weights must sum to 1 after normalization

**Test Cases:**

```python
# test_resampling.py

import numpy as np
import pytest


class TestMultinomialResampling:
    """Tests for multinomial resampling."""

    def test_preserves_particle_count(self):
        """After resampling, exactly n_particles remain."""
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        n_particles = len(weights)

        indices = multinomial_resampling(weights, n_particles)

        assert len(indices) == n_particles

    def test_high_weight_particle_duplicated(self):
        """Particle with weight 1.0 is always selected."""
        weights = np.array([0.0, 0.0, 0.0, 1.0])
        n_particles = 4

        indices = multinomial_resampling(weights, n_particles)

        assert all(idx == 3 for idx in indices)

    def test_zero_weight_particle_never_selected(self):
        """Particle with weight 0.0 is never selected."""
        weights = np.array([0.0, 0.5, 0.5])
        n_particles = 100  # Many samples to verify

        indices = multinomial_resampling(weights, n_particles)

        assert 0 not in indices

    def test_uniform_weights_roughly_uniform_selection(self):
        """Uniform weights lead to roughly uniform selection."""
        np.random.seed(42)
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        n_particles = 1000

        indices = multinomial_resampling(weights, n_particles)
        counts = np.bincount(indices, minlength=4)

        # Each particle should be selected ~250 times (within 20%)
        assert all(200 < c < 300 for c in counts)

    def test_weights_do_not_need_to_be_normalized(self):
        """Unnormalized weights are handled correctly."""
        weights = np.array([1, 2, 3, 4])  # Sum = 10, not 1
        n_particles = 4

        indices = multinomial_resampling(weights, n_particles)

        assert len(indices) == n_particles


class TestSystematicResampling:
    """Tests for systematic resampling."""

    def test_preserves_particle_count(self):
        """After resampling, exactly n_particles remain."""
        weights = np.array([0.1, 0.2, 0.3, 0.4])

        indices = systematic_resampling(weights)

        assert len(indices) == len(weights)

    def test_deterministic_given_u(self):
        """Same starting offset u gives same result."""
        weights = np.array([0.1, 0.2, 0.3, 0.4])

        indices1 = systematic_resampling(weights, u=0.1)
        indices2 = systematic_resampling(weights, u=0.1)

        assert list(indices1) == list(indices2)

    def test_different_u_different_result(self):
        """Different u values can give different results."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])

        indices1 = systematic_resampling(weights, u=0.1)
        indices2 = systematic_resampling(weights, u=0.9)

        # With uniform weights, might be same, but generally can differ
        pass  # This is a property test, not strict assertion


class TestStratifiedResampling:
    """Tests for stratified resampling."""

    def test_preserves_particle_count(self):
        """After resampling, exactly n_particles remain."""
        weights = np.array([0.1, 0.2, 0.3, 0.4])

        indices = stratified_resampling(weights)

        assert len(indices) == len(weights)

    def test_each_stratum_has_one_sample(self):
        """Stratified ensures each stratum contributes."""
        # This is a property of stratified resampling
        pass


class TestWeightNormalization:
    """Tests for weight normalization utilities."""

    def test_normalized_weights_sum_to_one(self):
        """Normalized weights sum to 1.0."""
        weights = np.array([1, 2, 3, 4])

        normalized = normalize_weights(weights)

        assert np.isclose(normalized.sum(), 1.0)

    def test_zero_weights_handled(self):
        """All-zero weights don't cause division by zero."""
        weights = np.array([0.0, 0.0, 0.0])

        # Should either raise ValueError or return uniform
        with pytest.raises(ValueError):
            normalize_weights(weights)
        # OR
        # normalized = normalize_weights(weights)
        # assert np.allclose(normalized, [1/3, 1/3, 1/3])

    def test_negative_weights_rejected(self):
        """Negative weights raise error."""
        weights = np.array([-0.5, 0.5, 1.0])

        with pytest.raises(ValueError):
            normalize_weights(weights)
```

**Priority:** HIGH - Critical for SMC correctness

---

### 4.2 `trex/smc/particle_filter.py` (Phase 1.2)

**Test Cases:**

```python
# test_particle_filter.py

class TestParticleFilter:
    """Tests for ParticleFilter class."""

    def test_initialization_creates_n_particles(self):
        """Initialization creates correct number of particles."""
        config = SMCConfig(n_particles=16)
        pf = ParticleFilter(config)
        pf.initialize(prompt="What is 2+2?")

        assert pf.n_particles == 16
        assert len(pf.particles) == 16

    def test_all_particles_start_with_prompt(self):
        """All particles start with the same prompt."""
        config = SMCConfig(n_particles=4)
        pf = ParticleFilter(config)
        pf.initialize(prompt="What is 2+2?")

        for particle in pf.particles:
            assert particle.startswith("What is 2+2?")

    def test_weights_sum_to_one(self):
        """Weights are always normalized."""
        config = SMCConfig(n_particles=4)
        pf = ParticleFilter(config)
        pf.initialize(prompt="What is 2+2?")

        assert np.isclose(pf.weights.sum(), 1.0)

    def test_resampling_preserves_count(self):
        """Resampling doesn't change particle count."""
        config = SMCConfig(n_particles=4)
        pf = ParticleFilter(config)
        pf.initialize(prompt="What is 2+2?")
        pf.weights = np.array([0.1, 0.2, 0.3, 0.4])

        pf.resample()

        assert len(pf.particles) == 4

    def test_effective_sample_size_computed(self):
        """ESS is computed correctly."""
        config = SMCConfig(n_particles=4)
        pf = ParticleFilter(config)
        pf.weights = np.array([0.25, 0.25, 0.25, 0.25])

        ess = pf.effective_sample_size()

        assert ess == 4.0  # Uniform weights → ESS = n

    def test_ess_degeneracy_detection(self):
        """Degenerate weights have low ESS."""
        config = SMCConfig(n_particles=4)
        pf = ParticleFilter(config)
        pf.weights = np.array([0.97, 0.01, 0.01, 0.01])

        ess = pf.effective_sample_size()

        assert ess < 2.0  # Highly degenerate


class TestEffectiveSampleSize:
    """Tests for ESS computation."""

    def test_uniform_weights_ess_equals_n(self):
        """Uniform weights have ESS = n."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])

        ess = compute_ess(weights)

        assert np.isclose(ess, 4.0)

    def test_degenerate_weights_ess_equals_one(self):
        """Single particle has all weight → ESS ≈ 1."""
        weights = np.array([1.0, 0.0, 0.0, 0.0])

        ess = compute_ess(weights)

        assert np.isclose(ess, 1.0)

    def test_ess_formula(self):
        """ESS = 1 / sum(w_i^2)."""
        weights = np.array([0.5, 0.3, 0.2])

        ess = compute_ess(weights)
        expected = 1.0 / (0.5**2 + 0.3**2 + 0.2**2)

        assert np.isclose(ess, expected)
```

**Priority:** HIGH - Foundation for SMC

---

### 4.3 `trex/smc/twisted_smc.py` (Phase 2.3)

**Test Cases:**

```python
# test_twisted_smc.py

class TestTwistedWeights:
    """Tests for twisted importance weight computation."""

    def test_weight_ratio_increases_when_value_improves(self):
        """w_t > 1 when V(t) > V(t-1)."""
        values_t = np.array([0.9, 0.8, 0.7])
        values_t_minus_1 = np.array([0.5, 0.5, 0.5])

        weights = compute_twisted_weights(values_t, values_t_minus_1)

        assert all(w > 1.0 for w in weights)

    def test_weight_ratio_decreases_when_value_drops(self):
        """w_t < 1 when V(t) < V(t-1)."""
        values_t = np.array([0.3, 0.2, 0.1])
        values_t_minus_1 = np.array([0.5, 0.5, 0.5])

        weights = compute_twisted_weights(values_t, values_t_minus_1)

        assert all(w < 1.0 for w in weights)

    def test_handles_zero_previous_value(self):
        """No division by zero when V(t-1) ≈ 0."""
        values_t = np.array([0.5, 0.5])
        values_t_minus_1 = np.array([0.0, 1e-10])

        # Should not raise, should use epsilon
        weights = compute_twisted_weights(values_t, values_t_minus_1)

        assert not np.any(np.isinf(weights))
        assert not np.any(np.isnan(weights))

    def test_unchanged_value_gives_weight_one(self):
        """w_t = 1 when V(t) = V(t-1)."""
        values_t = np.array([0.5, 0.5, 0.5])
        values_t_minus_1 = np.array([0.5, 0.5, 0.5])

        weights = compute_twisted_weights(values_t, values_t_minus_1)

        assert np.allclose(weights, 1.0)
```

**Priority:** HIGH - Core TSMC logic

---

### 4.4 `trex/tempering/swap_schedule.py` (Phase 3.1)

**Test Cases:**

```python
# test_swap_schedule.py

class TestGetSwapPairs:
    """Tests for non-reversible swap schedule."""

    def test_odd_timestep_swaps_pairs_1_2_3_4(self):
        """S_odd = {(1,2), (3,4), ...}."""
        pairs = get_swap_pairs(timestep=1, num_temperatures=5)

        assert pairs == [(1, 2), (3, 4)]

    def test_even_timestep_swaps_pairs_2_3_4_5(self):
        """S_even = {(2,3), (4,5), ...}."""
        pairs = get_swap_pairs(timestep=2, num_temperatures=5)

        assert pairs == [(2, 3), (4, 5)]

    def test_alternating_covers_all_adjacent_pairs(self):
        """Over 2 timesteps, every adjacent pair is attempted."""
        num_temps = 5
        pairs_odd = set(get_swap_pairs(1, num_temps))
        pairs_even = set(get_swap_pairs(2, num_temps))

        all_pairs = pairs_odd | pairs_even
        expected = {(1, 2), (2, 3), (3, 4), (4, 5)}

        assert all_pairs == expected

    def test_single_temperature_returns_empty(self):
        """With K=1, no swaps possible."""
        pairs = get_swap_pairs(timestep=1, num_temperatures=1)

        assert pairs == []

    def test_two_temperatures(self):
        """With K=2, only (1,2) on odd steps."""
        pairs_odd = get_swap_pairs(timestep=1, num_temperatures=2)
        pairs_even = get_swap_pairs(timestep=2, num_temperatures=2)

        assert pairs_odd == [(1, 2)]
        assert pairs_even == []


class TestTemperatureLadder:
    """Tests for temperature schedule generation."""

    def test_linear_schedule(self):
        """Linear schedule from 0 to 1."""
        betas = generate_temperature_ladder(
            num_temperatures=5,
            schedule="linear",
            min_beta=0.0,
            max_beta=1.0
        )

        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert np.allclose(betas, expected)

    def test_first_is_min_last_is_max(self):
        """First beta is min, last is max."""
        betas = generate_temperature_ladder(
            num_temperatures=5,
            schedule="linear",
            min_beta=0.1,
            max_beta=0.9
        )

        assert betas[0] == 0.1
        assert betas[-1] == 0.9

    def test_geometric_schedule_is_not_linear(self):
        """Geometric schedule is not evenly spaced."""
        betas_linear = generate_temperature_ladder(5, "linear")
        betas_geometric = generate_temperature_ladder(5, "geometric")

        assert not np.allclose(betas_linear, betas_geometric)

    def test_monotonically_increasing(self):
        """Betas are strictly increasing."""
        betas = generate_temperature_ladder(5, "linear")

        assert all(betas[i] < betas[i+1] for i in range(len(betas)-1))
```

**Priority:** HIGH - Foundation for parallel tempering

---

### 4.5 `trex/tempering/exchange.py` (Phase 3.2)

**Test Cases:**

```python
# test_exchange.py

class TestAcceptanceRatio:
    """Tests for replica exchange acceptance probability."""

    def test_acceptance_ratio_bounded_0_to_1(self):
        """α ∈ [0, 1] for any input."""
        for _ in range(100):
            phi_x = np.random.uniform(0.01, 1.0)
            phi_x_prime = np.random.uniform(0.01, 1.0)
            beta_target = np.random.uniform(0.0, 1.0)

            alpha = compute_acceptance_ratio(phi_x, phi_x_prime, beta_target)

            assert 0.0 <= alpha <= 1.0

    def test_better_proposal_accepted_certainly(self):
        """If φ(x') > φ(x), α = 1."""
        phi_x = 0.5
        phi_x_prime = 1.0  # Better
        beta_target = 1.0

        alpha = compute_acceptance_ratio(phi_x, phi_x_prime, beta_target)

        assert alpha == 1.0

    def test_worse_proposal_accepted_probabilistically(self):
        """If φ(x') < φ(x), α < 1."""
        phi_x = 1.0
        phi_x_prime = 0.5  # Worse
        beta_target = 1.0

        alpha = compute_acceptance_ratio(phi_x, phi_x_prime, beta_target)

        assert alpha < 1.0

    def test_beta_zero_always_accepts(self):
        """At β=0 (hot), acceptance is 1."""
        alpha = compute_acceptance_ratio(
            phi_x=1.0,
            phi_x_prime=0.01,  # Much worse
            beta_target=0.0
        )

        assert alpha == 1.0  # (anything)^0 = 1

    def test_formula_correctness(self):
        """α = min(1, (φ(x')/φ(x))^β)."""
        phi_x = 0.4
        phi_x_prime = 0.8
        beta = 0.5

        alpha = compute_acceptance_ratio(phi_x, phi_x_prime, beta)
        expected = min(1.0, (0.8 / 0.4) ** 0.5)

        assert np.isclose(alpha, expected)


class TestMetropolisHastingsStep:
    """Tests for MH accept/reject step."""

    def test_always_accepts_when_alpha_is_one(self):
        """α = 1 always accepts."""
        np.random.seed(42)
        n_trials = 100

        accepts = sum(
            metropolis_hastings_accept(alpha=1.0)
            for _ in range(n_trials)
        )

        assert accepts == n_trials

    def test_never_accepts_when_alpha_is_zero(self):
        """α = 0 never accepts."""
        np.random.seed(42)
        n_trials = 100

        accepts = sum(
            metropolis_hastings_accept(alpha=0.0)
            for _ in range(n_trials)
        )

        assert accepts == 0

    def test_accepts_proportionally_to_alpha(self):
        """Acceptance rate ≈ α."""
        np.random.seed(42)
        n_trials = 10000
        alpha = 0.3

        accepts = sum(
            metropolis_hastings_accept(alpha=alpha)
            for _ in range(n_trials)
        )

        acceptance_rate = accepts / n_trials
        assert 0.25 < acceptance_rate < 0.35  # Within 5%
```

**Priority:** HIGH - Critical for parallel tempering correctness

---

## 5. Test Fixtures (`conftest.py`)

```python
# trex/tests/conftest.py

import pytest
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def sample_math_problems():
    """Sample math problems for testing."""
    return [
        {"prompt": "What is 2 + 2?", "answer": "4"},
        {"prompt": "Solve: x + 3 = 7", "answer": "4"},
        {"prompt": "What is \\frac{1}{2} + \\frac{1}{2}?", "answer": "1"},
    ]


@pytest.fixture
def sample_responses():
    """Sample model responses in various formats."""
    return [
        ("The answer is \\boxed{42}.", "42"),
        ("Therefore, the answer is 42.", "42"),
        ("#### 42", "42"),  # GSM8K format
        ("答案是 42", "42"),  # Chinese format
        ("The final answer is \\boxed{\\frac{1}{2}}.", "\\frac{1}{2}"),
    ]


@pytest.fixture
def uniform_weights():
    """Uniform weight vector."""
    return np.array([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def degenerate_weights():
    """Degenerate weight vector (one particle dominates)."""
    return np.array([0.97, 0.01, 0.01, 0.01])


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Temporary directory for checkpoint tests."""
    return tmp_path / "checkpoints"


@pytest.fixture
def mock_verifier():
    """Mock MathVerifier that returns configurable results."""
    class MockVerifier:
        def __init__(self, default_result=True):
            self.default_result = default_result
            self.call_count = 0

        def verify(self, prediction, ground_truth, **kwargs):
            self.call_count += 1
            return self.default_result

        def verify_batch(self, predictions, ground_truths, **kwargs):
            return [self.default_result] * len(predictions)

    return MockVerifier


# Mark slow tests
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


# Skip tests that require optional dependencies
@pytest.fixture
def requires_sympy():
    """Skip test if SymPy not available."""
    pytest.importorskip("sympy")


@pytest.fixture
def requires_math_verify():
    """Skip test if math_verify not available."""
    pytest.importorskip("math_verify")
```

---

## 6. pytest Configuration

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["trex/tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "requires_gpu: marks tests that require GPU",
]

[tool.coverage.run]
source = ["trex"]
omit = ["trex/tests/*", "trex/scripts/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]
```

---

## 7. Implementation Order

### Phase 1: Existing Code (Week 1)

1. [x] Create `trex/tests/` directory structure
2. [x] Create `conftest.py` with fixtures
3. [x] Implement `test_parser.py` - Answer extraction tests (77 tests)
4. [x] Implement `test_grader.py` - Math comparison tests (73 tests)
5. [ ] Implement `test_math_verifier.py` - Verifier integration tests
6. [ ] Implement `test_efficiency_tracker.py` - Metrics tracking tests

### Phase 2: SMC Foundation (TDD - Week 2)

7. [ ] Write `test_resampling.py` BEFORE implementing `resampling.py`
8. [ ] Write `test_particle_filter.py` BEFORE implementing `particle_filter.py`
9. [ ] Implement the SMC modules to pass the tests

### Phase 3: TSMC (TDD - Week 3)

10. [ ] Write `test_twisted_smc.py` BEFORE implementing
11. [ ] Implement to pass tests

### Phase 4: Parallel Tempering (TDD - Week 4)

12. [ ] Write `test_swap_schedule.py` BEFORE implementing
13. [ ] Write `test_exchange.py` BEFORE implementing
14. [ ] Implement to pass tests

---

## 8. Running Tests

```bash
# Run all tests
pytest trex/tests/ -v

# Run only fast unit tests (exclude slow/integration)
pytest trex/tests/ -v -m "not slow and not integration"

# Run specific module
pytest trex/tests/test_eval/ -v

# Run with coverage
pytest trex/tests/ --cov=trex --cov-report=html

# Run tests matching pattern
pytest trex/tests/ -v -k "parser"
```

---

## 9. CI Integration

Add to `.github/workflows/test.yml` (if using GitHub Actions):

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest trex/tests/ -v --cov=trex
```

---

## 10. Notes

- **TDD Workflow**: For new SMC/tempering modules, write tests first based on the mathematical specifications in `HIGH_LEVEL_CONTEXT.md`
- **Mocking**: Use `mock_verifier` fixture to test reward functions without loading LLMs
- **Numerical Tests**: Use `np.isclose()` or `pytest.approx()` for floating-point comparisons
- **Timeout Tests**: Mark potentially slow symbolic equality tests with `@pytest.mark.slow`
