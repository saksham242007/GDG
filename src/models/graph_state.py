from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.models.evaluation_schemas import (
    AnalysisType,
    RAGComplianceResult,
    SQLComplianceResult,
)


class ComplianceGraphState(BaseModel):
    """
    Centralized state for the LangGraph compliance evaluation workflow.
    """

    # Input data
    evaluation_id: str = Field(..., description="Unique evaluation session ID")
    company_input_path: Path = Field(..., description="Path to company data")
    input_type: Optional[Literal["structured", "unstructured"]] = Field(
        None, description="Detected input type"
    )
    input_description: Optional[str] = Field(None, description="Human-readable input description")

    # Human decision gate
    human_decision: Optional[AnalysisType] = Field(
        None, description="Human-selected analysis path (SQL or RAG)"
    )
    human_operator: Optional[str] = Field(None, description="Operator who made the decision")
    decision_rationale: Optional[str] = Field(None, description="Rationale for the decision")

    # Evaluation results
    sql_result: Optional[SQLComplianceResult] = Field(None, description="SQL evaluation result")
    rag_result: Optional[RAGComplianceResult] = Field(None, description="RAG evaluation result")
    final_result: Optional[SQLComplianceResult | RAGComplianceResult] = Field(
        None, description="Final evaluation result"
    )

    # Compliance decision
    is_compliant: Optional[bool] = Field(None, description="Overall compliance status")
    compliance_score: Optional[float] = Field(None, description="Compliance score (0-100)")

    # Error handling
    error: Optional[str] = Field(None, description="Error message if evaluation failed")
    retry_count: int = Field(default=0, description="Number of retry attempts")

    # Metadata
    messages: Annotated[list, add_messages] = Field(
        default_factory=list, description="Workflow execution messages"
    )

    class Config:
        arbitrary_types_allowed = True
