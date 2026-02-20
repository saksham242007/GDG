from __future__ import annotations

from datetime import datetime
from typing import List

from src.models.review_schemas import HumanReviewCase, ReviewStatus


def calculate_priority_score(case: HumanReviewCase) -> float:
    """
    Calculate priority score for a review case.

    Higher score = higher priority.

    Factors:
    - Confidence (lower confidence = higher priority)
    - Violation severity (if available)
    - Age of case (older = higher priority)
    - Analysis type (RAG may need more attention)

    Returns:
        Priority score (0-100, higher = more urgent)
    """
    score = 0.0

    # Confidence factor: Lower confidence = higher priority
    # If confidence is 0.5, score = 50; if 0.3, score = 70
    confidence_factor = (1.0 - case.confidence) * 50
    score += confidence_factor

    # Violation factor: If violation detected, add priority
    if case.violation:
        score += 20

    # Age factor: Older cases get higher priority
    if case.created_at:
        age_days = (datetime.now() - case.created_at).days
        age_factor = min(age_days * 2, 20)  # Max 20 points for age
        score += age_factor

    # Analysis type factor: RAG cases may need more attention
    if case.analysis_type == "RAG":
        score += 10

    # Severity factor (if available in record_details)
    if case.record_details:
        severity = case.record_details.get("severity", "").lower()
        if severity == "critical":
            score += 30
        elif severity == "high":
            score += 20
        elif severity == "medium":
            score += 10

    return min(score, 100.0)  # Cap at 100


def prioritize_review_queue(cases: List[HumanReviewCase]) -> List[HumanReviewCase]:
    """
    Sort review cases by priority (highest first).

    Args:
        cases: List of review cases

    Returns:
        Sorted list (highest priority first)
    """
    # Calculate priority for each case
    cases_with_priority = [(case, calculate_priority_score(case)) for case in cases]

    # Sort by priority (descending)
    cases_with_priority.sort(key=lambda x: x[1], reverse=True)

    # Return sorted cases
    return [case for case, _ in cases_with_priority]


def get_high_priority_cases(
    cases: List[HumanReviewCase],
    threshold: float = 60.0,
) -> List[HumanReviewCase]:
    """
    Filter cases above priority threshold.

    Args:
        cases: List of review cases
        threshold: Minimum priority score

    Returns:
        High-priority cases
    """
    prioritized = prioritize_review_queue(cases)
    return [case for case in prioritized if calculate_priority_score(case) >= threshold]
