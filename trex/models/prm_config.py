"""
Configuration for Process Reward Models (PRMs).

Different PRMs use different token formats for step separation and score extraction.
This module provides model-agnostic configuration to support multiple PRMs.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class PRMConfig:
    """
    Model-specific configuration for Process Reward Models.
    
    Different PRMs use different token formats for step separation and
    score extraction. This config abstracts those differences, allowing
    the SMC code to work with any supported PRM.
    
    Attributes:
        step_separator_token: Token inserted between/after reasoning steps
        extraction_method: How to extract scores from model outputs
            - "binary_softmax": 2-class logits, take softmax[:, positive_class_idx]
            - "single_logit": Single scalar logit, apply sigmoid
        num_classes: Number of output classes (for binary_softmax)
        positive_class_idx: Which class represents "correct" (for binary_softmax)
    """
    # Step separator token (inserted between/after reasoning steps)
    step_separator_token: str = "<extra_0>"  # Qwen2.5-Math-PRM-7B default
    
    # Score extraction method
    extraction_method: Literal["binary_softmax", "single_logit"] = "binary_softmax"
    
    # Number of output classes (for binary_softmax)
    num_classes: int = 2
    
    # Which class index represents "positive/correct" (for binary_softmax)
    positive_class_idx: int = 1


# Pre-defined configs for known models
QWEN_PRM_CONFIG = PRMConfig(
    step_separator_token="<extra_0>",
    extraction_method="binary_softmax",
    num_classes=2,
    positive_class_idx=1,
)

# Placeholder for future models
# MATH_SHEPHERD_CONFIG = PRMConfig(...)
