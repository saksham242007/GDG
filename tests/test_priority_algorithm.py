"""
Unit tests for the Priority Algorithm module.

Tests the scoring logic for review case prioritization.
"""

import pytest
from datetime import datetime

from src.models.review_schemas import HumanReviewCase, ReviewStatus
from src.review.priority_algorithm import (
    calculate_priority_score,
    prioritize_review_queue,
    get_high_priority_cases,
)


def _make_case(
    confidence: float = 0.5,
    violation: bool = True,
    analysis_type: str = "SQL",
    review_status: ReviewStatus = ReviewStatus.PENDING,
) -> HumanReviewCase:
    """Helper to create a minimal HumanReviewCase for testing."""
    return HumanReviewCase(
        record_id=f"REC-{id(confidence)}",
        violation=violation,
        confidence=confidence,
        explanation="Test case",
        cited_rule_ids=["RULE-1"],
        analysis_type=analysis_type,
        review_status=review_status,
    )


class TestCalculatePriorityScore:
    """Tests for the priority score calculation."""

    def test_low_confidence_higher_priority(self):
        """Lower confidence should produce a higher priority score."""
        high_conf = calculate_priority_score(_make_case(confidence=0.95))
        low_conf = calculate_priority_score(_make_case(confidence=0.3))
        assert low_conf > high_conf

    def test_violation_increases_priority(self):
        """Violations should have higher priority than non-violations."""
        violation = calculate_priority_score(_make_case(violation=True))
        no_violation = calculate_priority_score(_make_case(violation=False))
        assert violation > no_violation

    def test_score_is_non_negative(self):
        """Priority score should always be >= 0."""
        score = calculate_priority_score(_make_case(confidence=0.99, violation=False))
        assert score >= 0

    def test_score_capped_at_100(self):
        """Priority score should not exceed 100."""
        score = calculate_priority_score(_make_case(confidence=0.0, violation=True))
        assert score <= 100

    def test_rag_cases_higher_priority(self):
        """RAG analysis type should get a priority boost (10 points)."""
        sql_score = calculate_priority_score(_make_case(analysis_type="SQL"))
        rag_score = calculate_priority_score(_make_case(analysis_type="RAG"))
        assert rag_score > sql_score


class TestPrioritizeReviewQueue:
    """Tests for sorting review cases by priority."""

    def test_sorted_highest_first(self):
        """Cases should be sorted with highest priority first."""
        cases = [
            _make_case(confidence=0.9, violation=False),
            _make_case(confidence=0.2, violation=True),
            _make_case(confidence=0.5, violation=True),
        ]
        sorted_cases = prioritize_review_queue(cases)
        scores = [calculate_priority_score(c) for c in sorted_cases]
        assert scores == sorted(scores, reverse=True)

    def test_empty_list(self):
        """Empty list should return empty list."""
        assert prioritize_review_queue([]) == []


class TestGetHighPriorityCases:
    """Tests for filtering high-priority cases."""

    def test_filters_above_threshold(self):
        cases = [
            _make_case(confidence=0.1, violation=True),   # High priority
            _make_case(confidence=0.9, violation=False),  # Low priority
        ]
        high = get_high_priority_cases(cases, threshold=40.0)
        assert len(high) >= 1
        # The high-confidence non-violation should be filtered out
        assert len(high) < len(cases)
