# Adapted from simpleRL-reason (https://github.com/hkust-nlp/simpleRL-reason)

"""
Math answer verification with robust extraction and multiple comparison methods.

This module provides the main MathVerifier class that combines:
1. Robust answer extraction (boxed, natural language patterns)
2. Multiple verification backends (math_verify, SymPy, string matching)
3. Process-level timeout protection
"""

import re
import gc
import logging
import threading
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional, Callable, Any, Dict, Union

from .parser import extract_answer, extract_last_boxed
from .grader import math_equal

# Try to import HuggingFace math_verify for enhanced verification
try:
    from math_verify import parse as hf_parse, verify as hf_verify
    HF_MATH_VERIFY_AVAILABLE = True
except ImportError:
    HF_MATH_VERIFY_AVAILABLE = False


class GlobalProcessPool:
    """
    Singleton process pool for timeout-protected verification.
    
    This class manages a pool of worker processes for executing
    verification tasks with timeout protection.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, max_workers: int = 16, reset_threshold: int = 100000):
        self.max_workers = max_workers
        self.reset_threshold = reset_threshold
        self.task_counter = 0
        self.executor: Optional[ProcessPoolExecutor] = None
        self.logger = logging.getLogger(__name__)
        self._initialize_executor()
    
    def _initialize_executor(self) -> None:
        """Initialize a new ProcessPoolExecutor and reset task counter."""
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None
            gc.collect()
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.task_counter = 0
        self.logger.debug(f"Initialized ProcessPoolExecutor with {self.max_workers} workers")
    
    @classmethod
    def get_instance(cls, max_workers: int = 16, reset_threshold: int = 100000) -> 'GlobalProcessPool':
        """Get or create the singleton instance of GlobalProcessPool."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_workers=max_workers, reset_threshold=reset_threshold)
        return cls._instance
    
    def submit(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Submit a task to the executor with automatic recovery.
        
        Args:
            fn: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        
        Returns:
            Future object representing the computation.
        """
        try:
            if self.executor is None:
                with self._lock:
                    self._initialize_executor()
            return self.executor.submit(fn, *args, **kwargs)
        except (Exception, RuntimeError) as e:
            self.logger.warning(f"Process pool broken, recreating: {str(e)}")
            with self._lock:
                self._initialize_executor()
            return self.executor.submit(fn, *args, **kwargs)
    
    def shutdown(self) -> None:
        """Shutdown the process pool."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None


def _hf_verify_impl(gold: str, target: str) -> bool:
    """
    Verify using HuggingFace math_verify library.
    
    This function is designed to be called in a subprocess for timeout protection.
    """
    if not HF_MATH_VERIFY_AVAILABLE:
        return False
    
    try:
        # Wrap answers in \boxed{} if not already present
        if "\\boxed" not in target:
            boxed_target = f"\\boxed{{{target}}}"
        else:
            boxed_target = target
        
        if "\\boxed" not in gold:
            boxed_gold = f"\\boxed{{{gold}}}"
        else:
            boxed_gold = gold
        
        parsed_target = hf_parse(boxed_target)
        parsed_gold = hf_parse(boxed_gold)
        return hf_verify(gold=parsed_gold, target=parsed_target)
    except Exception as e:
        logging.debug(f"HF math_verify error: {e}")
        return False


class MathVerifier:
    """
    Robust math answer verification with multiple backends.
    
    This class provides a unified interface for verifying mathematical
    answers, supporting multiple verification methods with fallbacks.
    
    Verification order:
    1. Exact string match (after normalization)
    2. HuggingFace math_verify (if available)
    3. SymPy symbolic equality
    4. Numeric comparison with tolerance
    
    Example:
        >>> verifier = MathVerifier()
        >>> verifier.verify("\\boxed{42}", "42")
        True
        >>> verifier.verify("\\boxed{\\frac{1}{2}}", "0.5")
        True
    """
    
    def __init__(
        self,
        timeout_seconds: float = 10.0,
        max_workers: int = 16,
        use_hf_math_verify: bool = True,
        use_sympy: bool = True,
    ):
        """
        Initialize the MathVerifier.
        
        Args:
            timeout_seconds: Timeout for verification operations.
            max_workers: Maximum number of worker processes.
            use_hf_math_verify: Whether to use HuggingFace math_verify.
            use_sympy: Whether to use SymPy symbolic comparison.
        """
        self.timeout_seconds = timeout_seconds
        self.use_hf_math_verify = use_hf_math_verify and HF_MATH_VERIFY_AVAILABLE
        self.use_sympy = use_sympy
        self._process_pool = GlobalProcessPool.get_instance(max_workers=max_workers)
        self.logger = logging.getLogger(__name__)
    
    def extract_answer(
        self,
        text: str,
        data_name: str = "math",
        use_last_number: bool = True,
    ) -> str:
        """
        Extract the final answer from a model's output.
        
        Args:
            text: The model's output text.
            data_name: Dataset name for format-specific extraction.
            use_last_number: Whether to use last number as fallback.
        
        Returns:
            Extracted answer string.
        """
        return extract_answer(text, data_name=data_name, use_last_number=use_last_number)
    
    def verify(
        self,
        prediction: str,
        ground_truth: str,
        timeout: Optional[float] = None,
        extract_from_prediction: bool = True,
        extract_from_ground_truth: bool = False,
    ) -> bool:
        """
        Verify if the prediction matches the ground truth.
        
        Args:
            prediction: The model's prediction (may include full response).
            ground_truth: The expected answer.
            timeout: Override default timeout (seconds).
            extract_from_prediction: Whether to extract answer from prediction.
            extract_from_ground_truth: Whether to extract answer from ground truth.
        
        Returns:
            True if the answers match.
        """
        timeout = timeout or self.timeout_seconds
        
        # Extract answers if requested
        pred_answer = self.extract_answer(prediction) if extract_from_prediction else prediction
        gold_answer = self.extract_answer(ground_truth) if extract_from_ground_truth else ground_truth
        
        # Method 1: HuggingFace math_verify (most robust for symbolic math)
        if self.use_hf_math_verify:
            try:
                future = self._process_pool.submit(_hf_verify_impl, gold_answer, pred_answer)
                result = future.result(timeout=timeout)
                if result:
                    return True
            except FuturesTimeoutError:
                self.logger.debug(f"HF math_verify timeout for: {pred_answer} vs {gold_answer}")
            except Exception as e:
                self.logger.debug(f"HF math_verify error: {e}")
        
        # Method 2: math_equal (handles numeric, symbolic, matrices, etc.)
        try:
            if math_equal(pred_answer, gold_answer, timeout=self.use_sympy):
                return True
        except Exception as e:
            self.logger.debug(f"math_equal error: {e}")
        
        return False
    
    def verify_batch(
        self,
        predictions: list,
        ground_truths: list,
        **kwargs,
    ) -> list:
        """
        Verify a batch of predictions.
        
        Args:
            predictions: List of predictions.
            ground_truths: List of ground truth answers.
            **kwargs: Additional arguments for verify().
        
        Returns:
            List of boolean verification results.
        """
        assert len(predictions) == len(ground_truths), "Predictions and ground truths must have the same length"
        return [self.verify(p, g, **kwargs) for p, g in zip(predictions, ground_truths)]


def compute_score(
    solution_str: str,
    ground_truth: str,
    verifier: Optional[MathVerifier] = None,
) -> Dict[str, Union[float, bool]]:
    """
    Compute reward score for a model's solution.
    
    This function is compatible with OpenRLHF's reward function interface.
    
    Args:
        solution_str: The model's complete solution text.
        ground_truth: The expected answer.
        verifier: Optional MathVerifier instance (creates one if not provided).
    
    Returns:
        Dictionary with:
        - score: Float reward value (1.0 for correct, 0.0 for incorrect)
        - correctness: Boolean indicating if the answer is correct
        - has_boxed: Boolean indicating if the answer was in \\boxed{}
    """
    if verifier is None:
        verifier = MathVerifier()
    
    # Check if answer is in boxed format
    has_boxed = extract_last_boxed(solution_str) is not None
    
    # Verify the answer
    correct = verifier.verify(solution_str, ground_truth)
    
    return {
        "score": 1.0 if correct else 0.0,
        "correctness": correct,
        "has_boxed": has_boxed,
    }


if __name__ == "__main__":
    # Test the verifier
    print("Testing MathVerifier...")
    print(f"HuggingFace math_verify available: {HF_MATH_VERIFY_AVAILABLE}")
    print()
    
    verifier = MathVerifier(timeout_seconds=5.0)
    
    test_cases = [
        # (prediction, ground_truth, expected)
        ("The answer is \\boxed{42}", "42", True),
        ("\\boxed{\\frac{1}{2}}", "0.5", True),
        ("\\boxed{3.14159}", "3.14159", True),
        ("The final answer is 100 meters", "100", True),
        ("x = 5", "5", True),
        ("#### 42", "42", True),
        ("The answer is A", "A", True),
        ("\\boxed{(1, 2)}", "(1, 2)", True),
        ("\\boxed{42}", "43", False),
        ("Wrong answer", "42", False),
    ]
    
    print("Individual verification tests:")
    for pred, gold, expected in test_cases:
        result = verifier.verify(pred, gold)
        status = "✓" if result == expected else "✗"
        print(f"  {status} verify('{pred[:40]}...', '{gold}') = {result}")
    
    print("\ncompute_score test:")
    result = compute_score("Let me solve this... The answer is \\boxed{42}.", "42")
    print(f"  Score: {result}")
