from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class AnalysisType(str, Enum):
    SQL = "SQL"
    RAG = "RAG"


class HumanDecision(BaseModel):
    """
    Schema for logging human-in-the-loop decisions.
    """

    evaluation_id: str = Field(..., description="Unique evaluation session ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    analysis_type: AnalysisType = Field(..., description="Selected analysis path: SQL or RAG")
    input_type: str = Field(..., description="Type of input: structured or unstructured")
    input_source: str = Field(..., description="Path or identifier of the input data")
    human_operator: Optional[str] = Field(None, description="Optional identifier of the human operator")
    rationale: Optional[str] = Field(None, description="Optional reason for the decision")


class Violation(BaseModel):
    """
    Schema for a single compliance violation.
    """

    rule_id: str = Field(..., description="ID of the violated rule")
    violation_description: str = Field(..., description="Description of the violation")
    severity: str = Field(..., description="Severity level: low, medium, high, critical")
    affected_records: Optional[List[str]] = Field(None, description="IDs of affected records")


class SQLComplianceResult(BaseModel):
    """
    Schema for SQL-based compliance evaluation results.
    """

    evaluation_id: str = Field(..., description="Unique evaluation session ID")
    analysis_type: AnalysisType = Field(default=AnalysisType.SQL)
    compliant: bool = Field(..., description="Overall compliance status")
    violations: List[Violation] = Field(default_factory=list)
    explanation: str = Field(..., description="Human-readable explanation of the evaluation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    policy_reference: List[str] = Field(
        default_factory=list, description="List of policy rule IDs referenced"
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class RAGComplianceResult(BaseModel):
    """
    Schema for RAG-based semantic compliance evaluation results.
    """

    evaluation_id: str = Field(..., description="Unique evaluation session ID")
    analysis_type: AnalysisType = Field(default=AnalysisType.RAG)
    compliant: bool = Field(..., description="Overall compliance status")
    compliance_score: float = Field(..., ge=0.0, le=100.0, description="Compliance score (0-100)")
    explanation: str = Field(..., description="Human-readable explanation of the evaluation")
    violated_policies: List[str] = Field(
        default_factory=list, description="List of violated policy rule IDs or descriptions"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    retrieved_policy_chunks: Optional[List[str]] = Field(
        None, description="Policy chunks retrieved from RAG for context"
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class NonCompliantCase(BaseModel):
    """
    Schema for flagged non-compliant cases routed to human review.
    """

    evaluation_id: str = Field(..., description="Unique evaluation session ID")
    analysis_type: AnalysisType = Field(..., description="SQL or RAG")
    compliance_score: Optional[float] = Field(None, description="Compliance score if available")
    violations: List[Violation] = Field(default_factory=list)
    explanation: str = Field(..., description="Reason for flagging")
    flagged_at: datetime = Field(default_factory=datetime.now)
    review_status: str = Field(default="pending", description="pending, reviewed, resolved")
