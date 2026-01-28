"""
Shared test fixtures and pytest configuration for T-REX tests.

Fixtures defined here are automatically available to all tests
in this directory and subdirectories.
"""

import pytest
import numpy as np
from pathlib import Path


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_boxed_responses():
    """Sample model responses with \\boxed{} format."""
    return [
        ("The answer is \\boxed{42}.", "42"),
        ("Therefore, x = \\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("\\boxed{-5}", "-5"),
        ("We get \\boxed{3.14}", "3.14"),
        ("So \\boxed{x^2 + 1}", "x^2 + 1"),
    ]


@pytest.fixture
def sample_gsm8k_responses():
    """Sample GSM8K format responses with #### delimiter."""
    return [
        ("So the answer is #### 42", "42"),
        ("The total is #### 100", "100"),
        ("Mary has #### 15 apples", "15"),
    ]


@pytest.fixture
def sample_natural_language_responses():
    """Sample natural language answer patterns."""
    return [
        ("The answer is 42.", "42"),
        ("Therefore, the answer is 100", "100"),
        ("The final answer is 3.14", "3.14"),
        ("答案是 42", "42"),  # Chinese format
    ]


@pytest.fixture
def sample_latex_expressions():
    """Sample LaTeX expressions for normalization testing."""
    return [
        ("\\frac12", "\\frac{1}{2}"),
        ("\\frac{1}{2}", "\\frac{1}{2}"),
        ("\\dfrac{3}{4}", "\\frac{3}{4}"),
        ("\\sqrt2", "\\sqrt{2}"),
        ("\\sqrt{4}", "\\sqrt{4}"),
    ]


@pytest.fixture
def sample_math_comparisons():
    """Sample math comparisons (prediction, reference, expected_equal).

    Note: LaTeX vs numeric comparisons (e.g., \\frac{1}{2} vs 0.5)
    depend on SymPy's latex parser which may not always work reliably.
    Those are tested separately with appropriate skip markers.
    """
    return [
        # Identical
        ("42", "42", True),
        ("3.14", "3.14", True),
        # Numeric equivalence
        ("0.5", "0.50", True),
        ("1.0", "1", True),
        # Different values
        ("41", "42", False),
        ("0.5", "0.6", False),
        # Multiple choice
        ("A", "A", True),
        ("B", "A", False),
    ]


@pytest.fixture
def sample_multiple_choice():
    """Sample multiple choice extraction cases."""
    return [
        ("(A)", "A"),
        ("A.", "A"),
        ("The answer is B", "B"),
        ("(C) is correct", "C"),
        ("answer: d", "D"),
    ]


# =============================================================================
# Weight Vector Fixtures (for SMC tests)
# =============================================================================


@pytest.fixture
def uniform_weights():
    """Uniform weight vector (4 particles)."""
    return np.array([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def degenerate_weights():
    """Degenerate weight vector (one particle dominates)."""
    return np.array([0.97, 0.01, 0.01, 0.01])


@pytest.fixture
def varied_weights():
    """Non-uniform weight vector."""
    return np.array([0.1, 0.2, 0.3, 0.4])


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_verifier():
    """
    Factory fixture for creating mock verifiers.

    Usage:
        def test_something(mock_verifier):
            verifier = mock_verifier(default_result=True)
            assert verifier.verify("x", "y") == True
    """
    def _create_mock(default_result=True, results_map=None):
        """
        Create a mock verifier.

        Args:
            default_result: Default return value for verify().
            results_map: Dict mapping (pred, gold) -> result for specific cases.
        """
        class MockVerifier:
            def __init__(self):
                self.call_count = 0
                self.calls = []

            def verify(self, prediction, ground_truth, **kwargs):
                self.call_count += 1
                self.calls.append((prediction, ground_truth, kwargs))

                if results_map and (prediction, ground_truth) in results_map:
                    return results_map[(prediction, ground_truth)]
                return default_result

            def verify_batch(self, predictions, ground_truths, **kwargs):
                return [
                    self.verify(p, g, **kwargs)
                    for p, g in zip(predictions, ground_truths)
                ]

            def extract_answer(self, text, **kwargs):
                # Simple mock extraction - just return text
                return text

        return MockVerifier()

    return _create_mock


# =============================================================================
# Temporary File Fixtures
# =============================================================================


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Temporary directory for checkpoint tests."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def tmp_metrics_file(tmp_path):
    """Temporary file path for metrics JSON."""
    return tmp_path / "metrics.json"


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "requires_sympy: marks tests that require SymPy"
    )


# =============================================================================
# Skip Condition Fixtures
# =============================================================================


@pytest.fixture
def requires_sympy():
    """Skip test if SymPy is not available."""
    pytest.importorskip("sympy")


@pytest.fixture
def requires_regex():
    """Skip test if regex module is not available."""
    pytest.importorskip("regex")
