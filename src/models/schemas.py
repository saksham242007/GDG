from __future__ import annotations

from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


class RuleComplexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class ExtractedRule(BaseModel):
    """
    Pydantic schema representing a single compliance rule as produced by the LLM.
    """

    rule_id: str = Field(..., description="Unique identifier of the rule within the corpus.")
    rule_text: str = Field(..., description="Original text of the rule or clause.")
    condition: str = Field(..., description="Condition under which the rule applies.")
    threshold: Optional[float] = Field(
        None, description="Numeric threshold, if applicable (e.g., 10000 for amount > 10000)."
    )
    required_action: str = Field(..., description="Required control, approval, or process.")
    rule_complexity: RuleComplexity = Field(
        ..., description='Complexity classification: "simple" or "complex".'
    )
    source_pdf: Optional[str] = Field(
        None, description="Name or path of the PDF the rule was extracted from."
    )
    category: Optional[str] = Field(
        None,
        description="Optional category or section label derived from the document structure.",
    )


class ExtractedRuleList(BaseModel):
    """
    Wrapper model for a list of extracted rules from a single chunk or document.
    """

    rules: List[ExtractedRule]

