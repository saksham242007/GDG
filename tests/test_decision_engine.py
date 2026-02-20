"""
Unit tests for the Decision Engine module.

Tests confidence-based routing of violations to auto-log or human review.
"""

import pytest
from pathlib import Path

from src.models.evaluation_schemas import (
    AnalysisType,
    SQLComplianceResult,
    Violation,
)
from src.models.review_schemas import DecisionType, ViolationRecord
from src.decision.decision_engine import (
    create_violation_record,
    make_confidence_decision,
)


class TestCreateViolationRecord:
    """Tests for converting evaluation results to ViolationRecords."""

    def test_sql_non_compliant_result(self):
        """Non-compliant SQL result should produce a violation record with violation=True."""
        result = SQLComplianceResult(
            evaluation_id="EVAL-001",
            analysis_type=AnalysisType.SQL,
            compliant=False,
            explanation="Amount exceeded threshold",
            violations=[
                Violation(
                    rule_id="RULE-001",
                    violation_description="Threshold breach",
                    severity="high",
                ),
            ],
            policy_references=["RULE-001"],
            confidence=0.85,
        )
        record = create_violation_record("REC-001", result)
        assert isinstance(record, ViolationRecord)
        assert record.violation is True
        assert record.record_id == "REC-001"

    def test_sql_compliant_result(self):
        """Compliant SQL result should produce a violation record with violation=False."""
        result = SQLComplianceResult(
            evaluation_id="EVAL-002",
            analysis_type=AnalysisType.SQL,
            compliant=True,
            explanation="All good",
            violations=[],
            policy_references=[],
            confidence=0.95,
        )
        record = create_violation_record("REC-002", result)
        assert record.violation is False


class TestMakeConfidenceDecision:
    """Tests for confidence-based routing decisions."""

    def test_high_confidence_auto_logged(self):
        """Violations >= 0.75 confidence should be auto-logged."""
        record = ViolationRecord(
            record_id="REC-001",
            analysis_type="SQL",
            violation=True,
            confidence=0.85,
            explanation="High confidence violation",
            cited_rule_ids=["RULE-1"],
            tier="SQL",
        )
        decision = make_confidence_decision(record)
        assert decision.decision == DecisionType.AUTO_LOG

    def test_low_confidence_human_review(self):
        """Violations < 0.75 confidence should go to human review."""
        record = ViolationRecord(
            record_id="REC-002",
            analysis_type="RAG",
            violation=True,
            confidence=0.5,
            explanation="Low confidence violation",
            cited_rule_ids=["POL-A"],
            tier="RAG",
        )
        decision = make_confidence_decision(record)
        assert decision.decision == DecisionType.HUMAN_REVIEW

    def test_boundary_confidence(self):
        """Exactly at threshold (0.75) should be auto-logged."""
        record = ViolationRecord(
            record_id="REC-003",
            analysis_type="SQL",
            violation=True,
            confidence=0.75,
            explanation="Boundary case",
            cited_rule_ids=["RULE-2"],
            tier="SQL",
        )
        decision = make_confidence_decision(record)
        assert decision.decision == DecisionType.AUTO_LOG

    def test_zero_confidence_human_review(self):
        """Zero confidence should always go to human review."""
        record = ViolationRecord(
            record_id="REC-004",
            analysis_type="RAG",
            violation=True,
            confidence=0.0,
            explanation="No confidence",
            cited_rule_ids=[],
            tier="RAG",
        )
        decision = make_confidence_decision(record)
        assert decision.decision == DecisionType.HUMAN_REVIEW
