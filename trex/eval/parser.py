# Adapted from simpleRL-reason (https://github.com/hkust-nlp/simpleRL-reason)
# Original source: Qwen Math Eval Toolkit

"""
Math answer extraction and parsing utilities.

This module provides functions for extracting and normalizing mathematical
answers from LLM outputs, handling various formats including LaTeX, boxed
answers, and natural language patterns.
"""

import re
from typing import Optional

try:
    import regex
except ImportError:
    import re as regex


# Units commonly found in math problems (from MathQA and other datasets)
UNIT_TEXTS = [
    "degree", "degrees", "mph", "kmph", "ft", "feet", "m", "meter", "meters",
    "cm", "centimeter", "centimeters", "mm", "km", "mile", "miles", "inch",
    "inches", "yard", "yards", "sq m", "sq ft", "square", "cubic", "liter",
    "liters", "gallon", "gallons", "kg", "kilogram", "kilograms", "g", "gram",
    "grams", "lb", "pound", "pounds", "second", "seconds", "sec", "minute",
    "minutes", "min", "hour", "hours", "hr", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "percent", "percentage", "dollar",
    "dollars", "cent", "cents", "rupee", "rupees", "rs", "unit", "units",
]


def _fix_fracs(string: str) -> str:
    """Fix LaTeX fractions like \\frac12 to \\frac{1}{2}."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    """Convert simple fractions like 1/2 to \\frac{1}{2}."""
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except (ValueError, AssertionError):
        return string


def _fix_sqrt(string: str) -> str:
    """Fix LaTeX sqrt like \\sqrt2 to \\sqrt{2}."""
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def strip_string(string: str, skip_unit: bool = False) -> str:
    """
    Normalize a mathematical expression string.
    
    This function applies various transformations to standardize
    mathematical expressions for comparison.
    
    Args:
        string: The input string to normalize.
        skip_unit: If True, don't remove units from the string.
    
    Returns:
        Normalized string.
    """
    string = str(string).strip()
    
    # Remove linebreaks
    string = string.replace("\n", "")
    
    # Remove trailing period
    string = string.rstrip(".")
    
    # Remove inverse spaces and backslash escapes
    string = string.replace("\\!", "")
    
    # Normalize matrix notation
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")
    
    # Normalize inequalities
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )
    
    # Remove \\left and \\right
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("\\{", "{").replace("\\}", "}")
    
    # Remove trailing \\text{...} (usually units)
    _string = re.sub(r"\\text\{.*?\}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string
    
    # Remove units
    if not skip_unit:
        for unit_text in UNIT_TEXTS:
            _string = re.sub(r"(^|\W)" + re.escape(unit_text) + r"($|\W)", r"\1\2", string, flags=re.IGNORECASE)
            if _string != "":
                string = _string
    
    # Remove degree symbols
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    
    # Remove dollar signs
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")
    
    # Replace \\text{...} with just the content
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    
    # Remove variable assignments at start
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    
    # Normalize special sets
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")
    
    # Remove percentage and percent sign
    string = string.replace("\\%", "").replace("\\%", "").replace("%", "")
    
    # Fix leading decimal points
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    
    # Handle infinity
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")
    
    # Remove 'and' and mathbf
    string = string.replace("and", "").replace("\\mathbf", "")
    
    # Remove \\mbox{...}
    string = re.sub(r"\\mbox\{.*?\}", "", string)
    
    # Normalize i/j for complex numbers
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")
    
    # Remove trailing zeros after decimal
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)
    
    if len(string) == 0:
        return string
    
    if string[0] == ".":
        string = "0" + string
    
    # Remove simple variable assignments like "k = " or "q = "
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    
    # Apply fixes
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    
    return string


def find_box(pred_str: str) -> str:
    """
    Extract content from \\boxed{...} in a string.
    
    Handles nested braces correctly.
    
    Args:
        pred_str: The string containing \\boxed{...}.
    
    Returns:
        The content inside the boxed command.
    """
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
        return a
    else:
        return ans.split("$")[0].strip()


def extract_answer(
    pred_str: str,
    data_name: str = "math",
    use_last_number: bool = True,
) -> str:
    """
    Extract the final answer from a model's prediction string.
    
    This function handles multiple answer formats:
    - \\boxed{...} (LaTeX boxed answers)
    - "The answer is..." patterns
    - GSM8K "####" format
    - Final number extraction as fallback
    
    Args:
        pred_str: The model's prediction string.
        data_name: The dataset name (affects extraction strategy).
        use_last_number: Whether to use the last number as fallback.
    
    Returns:
        The extracted answer string.
    """
    # Handle Russian characters that sometimes appear
    pred_str = pred_str.replace("\u043a\u0438", "")
    
    # Multiple choice datasets
    if data_name in ["mmlu_stem", "sat_math", "aqua", "gaokao2023"]:
        return _choice_answer_clean(pred_str)
    
    pred = ""
    
    # Pattern 1: "final answer is $...$. I hope"
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    
    # Pattern 2: \\boxed{...}
    elif "boxed" in pred_str:
        pred = find_box(pred_str)
    
    # Pattern 3: "The answer is" / "the answer is"
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    
    # Pattern 4: "final answer is"
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    
    # Pattern 5: GSM8K format "####"
    elif "####" in pred_str:
        pred = pred_str.split("####")[-1].strip()
    
    # Pattern 6: Chinese "答案是"
    elif "答案是" in pred_str:
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    
    # Pattern 7: Use last number as fallback
    elif use_last_number:
        pattern = r"-?\d*\.?\d+"
        matches = re.findall(pattern, pred_str.replace(",", ""))
        if len(matches) >= 1:
            pred = matches[-1]
        else:
            pred = ""
    else:
        pred = ""
    
    # Clean up the answer
    pred = re.sub(r"\n\s*", "", pred)
    if pred and pred[0] == ":":
        pred = pred[1:]
    if pred and pred[-1] == ".":
        pred = pred[:-1]
    if pred and pred[-1] == "/":
        pred = pred[:-1]
    
    # Apply string normalization
    skip_unit = data_name in ["carp_en", "minerva_math"]
    pred = strip_string(pred, skip_unit=skip_unit)
    
    return pred


def _choice_answer_clean(pred: str) -> str:
    """Clean and extract multiple choice answer."""
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp[-1]
    else:
        pred = pred.strip().strip(".")
    pred = pred.rstrip(".").rstrip("/")
    return pred


def extract_last_boxed(text: str) -> Optional[str]:
    """
    Extract the last \\boxed{...} content from text.
    
    Args:
        text: The text containing \\boxed{...}.
    
    Returns:
        The full \\boxed{...} string, or None if not found.
    """
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = list(re.finditer(pattern, text))
    
    if matches:
        return matches[-1].group(0)
    return None


if __name__ == "__main__":
    # Test cases
    test_cases = [
        # (input, expected_answer)
        ("The answer is \\boxed{42}", "42"),
        ("Therefore, x = \\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("So the final answer is 3.14", "3.14"),
        ("The answer is: 100 meters", "100"),
        ("#### 42", "42"),
        ("After calculation, we get x = 5. The answer is 5.", "5"),
    ]
    
    for pred_str, expected in test_cases:
        result = extract_answer(pred_str)
        status = "✓" if result == expected else "✗"
        print(f"{status} extract_answer('{pred_str[:50]}...') = '{result}' (expected '{expected}')")
