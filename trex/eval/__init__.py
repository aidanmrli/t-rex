"""
Evaluation utilities for T-REX math reasoning experiments.

This module provides robust answer extraction and verification for math problems,
adapted from simpleRL-reason's evaluation toolkit.
"""

from .grader import (
    math_equal,
    symbolic_equal,
    numeric_equal,
)
from .parser import (
    extract_answer,
    strip_string,
    find_box,
)
from .math_verifier import (
    MathVerifier,
    compute_score,
)

__all__ = [
    "MathVerifier",
    "compute_score",
    "math_equal",
    "symbolic_equal",
    "numeric_equal",
    "extract_answer",
    "strip_string",
    "find_box",
]
