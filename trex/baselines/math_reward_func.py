# Adapted from simpleRL-reason (https://github.com/hkust-nlp/simpleRL-reason)
# Reference: ReFT (Luong et al., ACL 2024)

"""
Math reward function for T-REX experiments.

Provides configurable reward computation with two modes:
- 'mix': Simple binary with format penalty
- 'independent': Separate correctness and format rewards
"""

import os
import random
import re
from typing import Dict, Optional

from trex.eval import MathVerifier
from trex.eval.parser import extract_answer, extract_last_boxed

# Environment-based configuration (compatible with Ray distributed training)
REWARD_FUNCTION_TYPE = os.environ.get('REWARD_FUNCTION_TYPE', "mix")
FORMAT_PENALTY_VALUE = float(os.environ.get('FORMAT_PENALTY_VALUE', "-1.0"))
LOG_SAMPLE_RATE = float(os.environ.get('LOG_SAMPLE_RATE', "0.05"))

# Global verifier instance
_verifier: Optional[MathVerifier] = None


def get_verifier() -> MathVerifier:
    """Get or create the global MathVerifier instance."""
    global _verifier
    if _verifier is None:
        _verifier = MathVerifier()
    return _verifier


def clean_response(text: str) -> str:
    """
    Clean model response by removing chat template tokens.
    Handles common stop tokens from various models.
    """
    # Remove everything before assistant turn marker (if present)
    text = re.sub(r'^.*?<\|im_start\|>assistant\n?', '', text, flags=re.DOTALL, count=1)
    
    # Remove stop tokens
    stop_words = ["</s>", "<|im_end|>", "<END_OF_TURN>"]

    for stop_word in stop_words:
        if stop_word in text:
            text = text.split(stop_word)[0].strip()
    
    return text


def compute_score(solution_str: str, ground_truth: str) -> Dict:
    """
    Compute reward score for a math solution.
    
    Adapted from simpleRL-reason's reward patterns.
    
    Args:
        solution_str: Full model response (may include prompt)
        ground_truth: Expected answer
    
    Returns:
        dict with 'score', 'correctness', 'has_boxed'
    """
    verifier = get_verifier()
    
    # Clean response and extract answer
    cleaned = clean_response(solution_str)
    extracted = extract_answer(cleaned)
    has_boxed = extract_last_boxed(cleaned) is not None
    
    # Verify correctness
    correct = verifier.verify(cleaned, ground_truth)
    
    # Compute reward based on mode
    if REWARD_FUNCTION_TYPE == 'mix':
        # Simple binary with format penalty
        if correct:
            score = 1.0
        elif not has_boxed:
            score = FORMAT_PENALTY_VALUE  # Penalize wrong format
        else:
            score = 0.0
    
    elif REWARD_FUNCTION_TYPE == 'independent':
        # Separate correctness and format rewards
        if correct and has_boxed:
            score = 1.0
        elif correct and not has_boxed:
            score = 0.5  # Correct but wrong format
        elif not correct and has_boxed:
            score = -0.5  # Wrong but good format
        else:
            score = FORMAT_PENALTY_VALUE  # Both wrong
    
    else:
        raise ValueError(f"Invalid REWARD_FUNCTION_TYPE: {REWARD_FUNCTION_TYPE}")
    
    # Log sample for debugging (5% rate by default)
    if random.random() < LOG_SAMPLE_RATE:
        print(f"\n[Model Response]\n{solution_str[:500]}...")
        print(f"[Ground Truth] {ground_truth}")
        print(f"[Extracted] {extracted} | [Boxed] {has_boxed}")
        print(f"[Correct] {correct} | [Score] {score}")
    
    return {"score": score, "correctness": correct, "has_boxed": has_boxed}


if __name__ == "__main__":
    # Test the reward function
    test_cases = [
        ("The answer is \\boxed{42}", "42"),
        ("Therefore x = 5", "5"),
        ("Wrong answer here", "42"),
    ]
    
    for solution, gt in test_cases:
        result = compute_score(solution, gt)
        print(f"Solution: {solution[:40]}... | GT: {gt} | Result: {result}")