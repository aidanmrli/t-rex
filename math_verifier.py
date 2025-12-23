import re
import math
from typing import Optional, Union

class MathVerifier:
    """
    A robust and comprehensive verifier for math problems.
    Handles extraction from LLM outputs and rigorous comparison of answers.
    """

    def __init__(self):
        # Regex for finding numbers (including decimals and scientific notation)
        # Improved to better capture scientific notation and avoid splitting
        self.number_regex = re.compile(r"(?<![a-zA-Z])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?![a-zA-Z])")
        
        # Patterns for "The answer is X" type extraction
        self.answer_patterns = [
            re.compile(r"The answer is[:\s]*([\$\d\.,/%\\\w\s\(\)\{\}\^=\-\+]+?)(?:\.[\s\n]|[\n]|$)", re.IGNORECASE),
            re.compile(r"Final Answer[:\s]*([\$\d\.,/%\\\w\s\(\)\{\}\^=\-\+]+?)(?:\.[\s\n]|[\n]|$)", re.IGNORECASE),
            re.compile(r"Answer[:\s]*([\$\d\.,/%\\\w\s\(\)\{\}\^=\-\+]+?)(?:\.[\s\n]|[\n]|$)", re.IGNORECASE),
        ]

    def extract_answer(self, text: str) -> Optional[str]:
        """
        Extracts the final answer from a variety of LLM output formats.
        """
        if not text:
            return None

        # 1. Look for LaTeX \boxed{...}
        boxed_content = self._extract_boxed(text)
        if boxed_content is not None:
            return boxed_content.strip()

        # 2. Look for GSM8K #### divider
        if "####" in text:
            return text.split("####")[-1].strip()

        # 3. Look for explicit "The answer is..." patterns
        for pattern in self.answer_patterns:
            match = pattern.search(text)
            if match:
                res = match.group(1).strip()
                if "=" in res and len(res.split("=")) > 1:
                    res = res.split("=")[-1].strip()
                return res

        # 4. Handle simple equations or statements (e.g., "Area = 25\pi", "answer is 4")
        if ("=" in text or " is " in text.lower()) and len(text.strip()) < 100:
            separator = "=" if "=" in text else " is "
            parts = text.split(separator)
            potential_ans = parts[-1].strip()
            if potential_ans:
                # Strip trailing sentence punctuation
                potential_ans = potential_ans.rstrip(".").strip()
                return potential_ans

        # 5. Fallback: short purely math strings
        if len(text.strip()) < 50:
             clean_text = text.strip().lower().replace("$", "")
             if clean_text in ["pi", "e"]:
                 return clean_text
             if re.match(r"^[\d\.,/%\\\(\)\{\}\^=\-\+\s\*epi<>!\|_]*$", clean_text):
                 return text.strip()

        # 6. Handle "x 10^n" notation
        text_converted = re.sub(r"(\d)\s*[xX*×]\s*10\^?\{?([-+]?\d+)\}?", r"\1e\2", text)

        # 7. Final Fallback: last number
        numbers = self.number_regex.findall(text_converted)
        if numbers:
            return numbers[-1]

        return None

    def _extract_boxed(self, text: str) -> Optional[str]:
        """Recursive extraction of \boxed{...} content."""
        start_idx = text.find("\\boxed{")
        if start_idx == -1:
            return None
        
        balance = 0
        content_start = start_idx + 7
        for i in range(content_start, len(text)):
            char = text[i]
            if char == "{":
                balance += 1
            elif char == "}":
                if balance == 0:
                    return text[content_start:i]
                balance -= 1
        return None

    def normalize(self, s: str) -> str:
        """
        Normalizes a math string for robust comparison.
        """
        if not s:
            return ""

        s = s.lower()

        # Remove LaTeX macros
        s = re.sub(r"\\text\{.*?\}", "", s)
        s = re.sub(r"\\(?:mathbf|mathrm|mathit|acute|grave|dot|ddot|tilde|bar|vec)\{.*?\}", lambda m: m.group(0)[m.group(0).find("{")+1:-1], s)
        s = s.replace("\\%", "%").replace("$", "")
        
        # Standardize constants and sets
        s = s.replace("\\pi", "pi").replace("\\in", "in").replace("\\{", "{").replace("\\}", "}")
        
        # Strip leading variable names and assignment/membership (e.g., "x=", "area is", "y in")
        s = re.sub(r"^[a-z_]+\s*(?:=|is|in)\s*", "", s)

        # Remove units while keeping scientific 'e'
        s = re.sub(r"(?<=\d)\s*(?!pi|e[-+]?\d)[a-df-z][a-z]*", "", s)

        # Thousand separators
        s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)

        # Fractions
        s = re.sub(r"\\frac\{(.+?)\}\{(.+?)\}", r"(\1)/(\2)", s)

        # Whitespace and multiplication
        s = "".join(s.split())
        s = s.replace("*", "").replace("×", "")
        
        # Trailing dot
        if s.endswith(".") and not re.search(r"\d\.$", s):
            s = s[:-1]
            
        # Strip outermost parentheses/braces if they enclose the whole thing
        if (s.startswith("(") and s.endswith(")")) or (s.startswith("{") and s.endswith("}")):
            s = s[1:-1]

        return s

    def verify(self, prediction: str, ground_truth: str) -> bool:
        """
        Compares an LLM prediction with a ground truth string.
        """
        pred_raw = self.extract_answer(prediction)
        gold_raw = self.extract_answer(ground_truth)

        if pred_raw is None or gold_raw is None:
            return False

        # 1. Clean and normalize
        pred_norm = self.normalize(pred_raw)
        gold_norm = self.normalize(gold_raw)

        # 2. String Match
        if pred_norm == gold_norm:
            return True

        # 3. Numerical Match
        try:
            p_val = self._parse_to_float(pred_norm)
            g_val = self._parse_to_float(gold_norm)
            
            if p_val is not None and g_val is not None:
                return math.isclose(p_val, g_val, rel_tol=1e-5, abs_tol=1e-8)
        except (ValueError, OverflowError):
            pass

        return False

    def _parse_to_float(self, s: str) -> Optional[float]:
        """Attempts to parse a normalized string into a float, supporting basic math."""
        # Handle simple pi
        if s == "pi":
            return math.pi
            
        try:
            # Direct float
            return float(s)
        except ValueError:
            # Remove redundant parentheses for easier parsing
            s = s.replace("(", "").replace(")", "").replace("{", "").replace("}", "")
            
            # Handle fractions a/b
            if "/" in s:
                parts = s.split("/")
                if len(parts) == 2:
                    p1 = self._parse_to_float(parts[0])
                    p2 = self._parse_to_float(parts[1])
                    if p1 is not None and p2 is not None and p2 != 0:
                        return p1 / p2
            
            # Handle simple powers a^b
            if "^" in s:
                parts = s.split("^")
                if len(parts) == 2:
                    p1 = self._parse_to_float(parts[0])
                    p2 = self._parse_to_float(parts[1])
                    if p1 is not None and p2 is not None:
                        try:
                            return math.pow(p1, p2)
                        except (ValueError, OverflowError):
                            pass
            
            # Handle percentage
            if s.endswith("%"):
                try:
                    return float(s[:-1]) / 100.0
                except ValueError:
                    pass
                    
        return None

if __name__ == "__main__":
    # Test suite
    verifier = MathVerifier()
    
    test_cases = [
        # Basic extraction
        ("The answer is 42", "#### 42", True),
        ("The answer is \\boxed{2/3}", "#### 0.66666666", True),
        ("Final Answer: $1,000.50$", "1000.5", True),
        ("\\boxed{\\frac{1}{2}}", "0.5", True),
        ("The answer is 50%", "1/2", True),
        ("The value is \\boxed{\\pi}", "pi", True),
        ("It's 10 meters", "10", True),
        ("We found that x=5, so the answer is 5.", "5", True),
        ("Nested braces: \\boxed{\\frac{1}{2^{3}}}", "1/8", True),
        
        # Negative numbers and scales
        ("The result is -15.5", "-15.5", True),
        ("Final Answer: -1/4", "-0.25", True),
        ("The coordinate is (-3, 4)", "-3,4", True), # Might fail due to comma handling, testing robustness
        
        # Scientific notation
        ("The magnitude is 1.2e3", "1200", True),
        ("Value: 5.5 x 10^-2", "0.055", True), # Complex notation
        
        # Mixed fractions and decimals
        ("The answer is 3.5", "7/2", True),
        ("Result is 1 1/2", "1.5", False), # Heuristic choice: we don't handle mixed fractions like '1 1/2' currently
        
        # Equations in extraction
        ("Therefore, x = 10.", "10", True),
        ("We get y+2 = 4, so y=2", "2", True),
        
        # LaTeX artifacts
        ("The answer is \\boxed{x \\in \{1, 2\}}", "{1,2}", True),
        ("Area = 25\\pi", "25 * pi", True),
        
        # Zero and small values
        ("The probability is 0.00000001", "1e-8", True),
        ("The net change is zero.", "0", False), # Language processing not supported
    ]
    
    print("Running Expanded MathVerifier tests...")
    passed = 0
    for pred, gold, expected in test_cases:
        result = verifier.verify(pred, gold)
        status = "PASS" if result == expected else "FAIL"
        if result == expected:
            passed += 1
        print(f"[{status}] Pred: {pred} | Gold: {gold} | Got: {result} (Exp: {expected})")
    
    print(f"\nSummary: {passed}/{len(test_cases)} tests passed.")
