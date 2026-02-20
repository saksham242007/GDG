from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    AUTO_LOG = "AUTO_LOG"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MARKED_COMPLIANT = "marked_compliant"


class ViolationRecord(BaseModel):
    """
    Schema for a single evaluated record with violation status.
    """

    record_id: str = Field(..., description="Unique identifier for the record")
    analysis_type: str = Field(..., description="SQL or RAG")
    violation: bool = Field(..., description="Whether violation was detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    explanation: str = Field(..., description="Explanation of the violation or compliance")
    cited_rule_ids: List[str] = Field(default_factory=list, description="Policy rule IDs referenced")
    timestamp: datetime = Field(default_factory=datetime.now)
    tier: Optional[str] = Field(None, description="SQL or RAG tier")


class DecisionRecord(BaseModel):
    """
    Schema for confidence-based decision routing.
    """

    record_id: str = Field(..., description="Unique identifier for the record")
    tier: str = Field(..., description="SQL or RAG")
    violation: bool = Field(..., description="Whether violation was detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    decision: DecisionType = Field(..., description="AUTO_LOG or HUMAN_REVIEW")
    timestamp: datetime = Field(default_factory=datetime.now)
    explanation: Optional[str] = Field(None, description="Decision rationale")


class HumanReviewCase(BaseModel):
    """
    Schema for cases requiring human review.
    """

    record_id: str = Field(..., description="Unique identifier for the record")
    violation: bool = Field(..., description="Whether violation was detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    explanation: str = Field(..., description="Violation explanation")
    cited_rule_ids: List[str] = Field(default_factory=list, description="Policy rule IDs")
    analysis_type: str = Field(..., description="SQL or RAG")
    record_details: Optional[dict] = Field(None, description="Additional record context")
    created_at: datetime = Field(default_factory=datetime.now)
    review_status: ReviewStatus = Field(default=ReviewStatus.PENDING)


class ReviewDecision(BaseModel):
    """
    Schema for human review decisions.
    """

    record_id: str = Field(..., description="Record being reviewed")
    human_decision: ReviewStatus = Field(..., description="Approved, rejected, or marked compliant")
    review_comment: Optional[str] = Field(None, description="Reviewer comment")
    reviewer_id: str = Field(..., description="ID of the human reviewer")
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence_override: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Optional confidence adjustment"
    )


class HumanFeedback(BaseModel):
    """
    Schema for storing human feedback for model improvement.
    """

    record_id: str = Field(..., description="Record ID")
    original_prediction: ViolationRecord = Field(..., description="Original model prediction")
    human_decision: ReviewDecision = Field(..., description="Human review decision")
    feedback_type: str = Field(..., description="correction, clarification, edge_case")
    model_version: Optional[str] = Field(None, description="Model version used")
    stored_at: datetime = Field(default_factory=datetime.now)


class DailySummary(BaseModel):
    """
    Schema for daily compliance summary report.
    """

    date: str = Field(..., description="Report date (YYYY-MM-DD)")
    total_records_processed: int = Field(..., description="Total records evaluated")
    sql_violations_count: int = Field(default=0, description="SQL tier violations")
    rag_violations_count: int = Field(default=0, description="RAG tier violations")
    human_review_count: int = Field(default=0, description="Cases sent to human review")
    auto_logged_count: int = Field(default=0, description="High-confidence auto-logged violations")
    compliance_rate: float = Field(..., ge=0.0, le=100.0, description="Compliance rate percentage")
    severity_breakdown: dict = Field(
        default_factory=dict, description="Violations by severity level"
    )
    trend_data: Optional[dict] = Field(None, description="Historical trend comparison")
    generated_at: datetime = Field(default_factory=datetime.now)
