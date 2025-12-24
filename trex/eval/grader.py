# Adapted from simpleRL-reason (https://github.com/hkust-nlp/simpleRL-reason)

"""
Math answer grading utilities.

This module provides functions for comparing mathematical expressions,
supporting both numeric and symbolic equality checking.
"""

import re
import multiprocessing
from math import isclose
from typing import Union, Optional

try:
    import regex
except ImportError:
    import re as regex

try:
    from sympy import simplify, N
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

try:
    from latex2sympy2 import latex2sympy
    LATEX2SYMPY_AVAILABLE = True
except ImportError:
    LATEX2SYMPY_AVAILABLE = False


def parse_digits(num: str) -> Optional[float]:
    """Parse a string to a float, handling commas and percentages."""
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except ValueError:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except ValueError:
                pass
    return None


def is_digit(num: str) -> bool:
    """Check if a string can be parsed as a digit."""
    return parse_digits(num) is not None


def numeric_equal(prediction: float, reference: float, rel_tol: float = 1e-4) -> bool:
    """
    Check if two numbers are approximately equal.
    
    Args:
        prediction: The predicted value.
        reference: The ground truth value.
        rel_tol: Relative tolerance for comparison.
    
    Returns:
        True if the values are approximately equal.
    """
    return isclose(reference, prediction, rel_tol=rel_tol)


def str_to_pmatrix(input_str: str) -> str:
    """Convert a set-like matrix string to pmatrix format."""
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def symbolic_equal(a: str, b: str) -> bool:
    """
    Check if two mathematical expressions are symbolically equal using SymPy.
    
    Args:
        a: First expression (string).
        b: Second expression (string).
    
    Returns:
        True if the expressions are mathematically equivalent.
    """
    if not SYMPY_AVAILABLE:
        return str(a).strip() == str(b).strip()
    
    def _parse(s):
        """Try multiple parsers to convert string to SymPy expression."""
        parsers = [parse_latex, parse_expr]
        if LATEX2SYMPY_AVAILABLE:
            parsers.append(latex2sympy)
        
        for f in parsers:
            try:
                return f(s.replace("\\\\", "\\"))
            except Exception:
                try:
                    return f(s)
                except Exception:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # Direct equality
    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    # Simplify and compare
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass

    # Equation equality (check if both sides equal)
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    # Numeric evaluation
    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass

    # Matrix equality
    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass

    return False


def symbolic_equal_process(a: str, b: str, output_queue: multiprocessing.Queue) -> None:
    """Wrapper for symbolic_equal to use with multiprocessing."""
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout: float = 1.0, **kwargs) -> bool:
    """
    Call a function with a timeout using multiprocessing.
    
    Args:
        func: Function to call.
        *args: Positional arguments for the function.
        timeout: Timeout in seconds.
        **kwargs: Keyword arguments for the function.
    
    Returns:
        Result of the function, or False if timeout occurred.
    """
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    try:
        return output_queue.get_nowait()
    except Exception:
        return False


def choice_answer_clean(pred: str) -> str:
    """Clean and extract multiple choice answer from prediction string."""
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp[-1]
    else:
        pred = pred.strip().strip(".")
    pred = pred.rstrip(".").rstrip("/")
    return pred


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = True,
) -> bool:
    """
    Check if prediction equals reference mathematically.
    
    Supports:
    1. Numerical equality (with tolerance)
    2. Symbolic equality (via SymPy)
    3. Multiple choice answers
    4. Matrix/vector comparisons
    
    Args:
        prediction: The predicted answer.
        reference: The ground truth answer.
        include_percentage: Whether to check percentage equivalents (x/100, x*100).
        is_close: Whether to use approximate comparison for numbers.
        timeout: Whether to use timeout for symbolic comparison.
    
    Returns:
        True if the answers are mathematically equal.
    """
    if prediction is None or reference is None:
        return False
    
    # Direct string comparison (case-insensitive)
    if str(prediction).strip().lower() == str(reference).strip().lower():
        return True
    
    # Multiple choice handling
    if reference in ["A", "B", "C", "D", "E"] and choice_answer_clean(prediction) == reference:
        return True

    # Numerical comparison
    try:
        if is_digit(prediction) and is_digit(reference):
            pred_num = parse_digits(prediction)
            ref_num = parse_digits(reference)
            
            if include_percentage:
                gt_results = [ref_num / 100, ref_num, ref_num * 100]
            else:
                gt_results = [ref_num]
            
            for item in gt_results:
                try:
                    if is_close:
                        if numeric_equal(pred_num, item):
                            return True
                    else:
                        if item == pred_num:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # Symbolic comparison
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    # Handle pmatrix conversion
    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)

    # Clean brackets for comparison
    pred_str, ref_str = prediction, reference
    if (
        (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("("))
        or (prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("["))
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    
    if pred_str.lower() == ref_str.lower():
        return True

    # List/tuple comparison: [a, b] vs [c, d]
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(pred_parts[i].strip(), ref_parts[i].strip(), include_percentage, is_close, timeout=False)
                for i in range(len(pred_parts))
            ):
                return True

    # Matrix comparison
    if (
        (prediction.startswith("\\begin{pmatrix}") or prediction.startswith("\\begin{bmatrix}"))
        and (prediction.endswith("\\end{pmatrix}") or prediction.endswith("\\end{bmatrix}"))
        and (reference.startswith("\\begin{pmatrix}") or reference.startswith("\\begin{bmatrix}"))
        and (reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}"))
    ):
        pred_lines = [
            line.strip()
            for line in prediction[len("\\begin{pmatrix}"):-len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[len("\\begin{pmatrix}"):-len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        math_equal(pred_parts[i].strip(), ref_parts[i].strip(), include_percentage, is_close, timeout=False)
                        for i in range(len(pred_parts))
                    ):
                        matched = False
                        break
                else:
                    matched = False
                    break
        else:
            matched = False
        if matched:
            return True

    # Equation comparison: x = 5 vs x = 5
    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred_expr = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref_expr = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred_expr, ref_expr) or symbolic_equal(f"-({pred_expr})", ref_expr):
            return True
    elif prediction.count("=") == 1 and len(prediction.split("=")[0].strip()) <= 2 and "=" not in reference:
        if math_equal(prediction.split("=")[1].strip(), reference, include_percentage, is_close, timeout=False):
            return True
    elif reference.count("=") == 1 and len(reference.split("=")[0].strip()) <= 2 and "=" not in prediction:
        if math_equal(prediction, reference.split("=")[1].strip(), include_percentage, is_close, timeout=False):
            return True

    # Full symbolic equality with SymPy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference, timeout=1.0):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("0.5", "1/2", True),
        ("\\frac{1}{2}", "0.5", True),
        ("42", "42", True),
        ("3.14159", "\\pi", False),  # Without SymPy this would be False
        ("(1, 2)", "(1, 2)", True),
        ("A", "A", True),
        ("x = 5", "x=5", True),
    ]
    
    for pred, ref, expected in test_cases:
        result = math_equal(pred, ref, timeout=True)
        status = "✓" if result == expected else "✗"
        print(f"{status} math_equal('{pred}', '{ref}') = {result} (expected {expected})")
