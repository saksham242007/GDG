from __future__ import annotations

import re
from typing import Tuple

from src.models.schemas import ExtractedRule, RuleComplexity


NUMERIC_PATTERN = re.compile(r"\b\d+(\.\d+)?\b")
COMPARISON_KEYWORDS = re.compile(
    r"\b(>|<|>=|<=|==|equals?|at least|greater than|less than|within|no more than|not exceed|exceed(s)?)\b",
    re.IGNORECASE,
)
DETERMINISTIC_PHRASES = re.compile(
    r"\b(must|shall|required|required to|has to|need to)\b", re.IGNORECASE
)
VAGUE_PHRASES = re.compile(
    r"\b(appropriate|reasonable|suspicious|unusual|adequate|sufficient|proper)\b",
    re.IGNORECASE,
)


def classify_rule(rule: ExtractedRule) -> Tuple[RuleComplexity, str]:
    """
    Classify a rule as simple or complex based on heuristic patterns.

    Returns:
        A tuple of (RuleComplexity, rationale string).
    """
    text = f"{rule.rule_text} {rule.condition} {rule.required_action}"

    has_number = bool(NUMERIC_PATTERN.search(text))
    has_comparison = bool(COMPARISON_KEYWORDS.search(text))
    has_deterministic = bool(DETERMINISTIC_PHRASES.search(text))
    has_vague = bool(VAGUE_PHRASES.search(text))

    # If clearly vague/contextual, treat as complex
    if has_vague and not (has_number and has_comparison):
        return RuleComplexity.COMPLEX, "Vague language without strong numeric/comparison anchors."

    # Strong numeric + comparison patterns indicate simple, SQL-checkable rules
    if (has_number and has_comparison) or (has_number and has_deterministic):
        return RuleComplexity.SIMPLE, "Numeric and/or comparison-based condition detected."

    # Default to complexity if we cannot easily map to deterministic logic
    if not (has_number or has_comparison):
        return RuleComplexity.COMPLEX, "No clear numeric or comparison patterns."

    # Fallback: honor the LLM's own classification when heuristics are inconclusive
    return rule.rule_complexity, "Heuristics inconclusive, using LLM classification."

