from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.models.evaluation_schemas import RAGComplianceResult, SQLComplianceResult
from src.models.review_schemas import DecisionRecord, DecisionType, ViolationRecord
from src.utils.logging_config import get_logger


logger = get_logger(__name__)

CONFIDENCE_THRESHOLD = 0.75


def create_violation_record(
    record_id: str,
    result: SQLComplianceResult | RAGComplianceResult,
) -> ViolationRecord:
    """
    Convert evaluation result into a ViolationRecord.
    """
    violation = False
    if isinstance(result, SQLComplianceResult):
        violation = not result.compliant
        cited_rules = result.policy_reference
    else:  # RAGComplianceResult
        violation = result.compliance_score < 75.0
        cited_rules = result.violated_policies

    return ViolationRecord(
        record_id=record_id,
        analysis_type=result.analysis_type.value,
        violation=violation,
        confidence=result.confidence,
        explanation=result.explanation,
        cited_rule_ids=cited_rules,
        tier=result.analysis_type.value,
    )


def make_confidence_decision(
    violation_record: ViolationRecord,
) -> DecisionRecord:
    """
    Apply confidence-based decision logic:
    - confidence >= 0.75 → AUTO_LOG
    - confidence < 0.75 → HUMAN_REVIEW
    """
    decision = (
        DecisionType.AUTO_LOG
        if violation_record.confidence >= CONFIDENCE_THRESHOLD
        else DecisionType.HUMAN_REVIEW
    )

    explanation = (
        f"High confidence ({violation_record.confidence:.2f}) - auto-logging violation"
        if decision == DecisionType.AUTO_LOG
        else f"Low confidence ({violation_record.confidence:.2f}) - requires human review"
    )

    return DecisionRecord(
        record_id=violation_record.record_id,
        tier=violation_record.tier or violation_record.analysis_type,
        violation=violation_record.violation,
        confidence=violation_record.confidence,
        decision=decision,
        explanation=explanation,
    )


def route_violation(
    violation_record: ViolationRecord,
    high_confidence_path: Path,
    review_path: Path,
) -> DecisionRecord:
    """
    Route violation based on confidence to appropriate storage.
    """
    decision = make_confidence_decision(violation_record)

    # Ensure directories exist
    high_confidence_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)

    if decision.decision == DecisionType.AUTO_LOG:
        # Store in high-confidence violations
        violations = []
        if high_confidence_path.exists():
            try:
                with high_confidence_path.open("r", encoding="utf-8") as f:
                    violations = json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Could not read existing violations file, starting fresh.")

        violations.append(violation_record.model_dump(mode="json", exclude_none=True))

        with high_confidence_path.open("w", encoding="utf-8") as f:
            json.dump(violations, f, indent=2, ensure_ascii=False, default=str)

        logger.info("Auto-logged violation: %s (confidence: %.2f)", violation_record.record_id, violation_record.confidence)

    else:  # HUMAN_REVIEW
        # Store in review queue
        review_cases = []
        if review_path.exists():
            try:
                with review_path.open("r", encoding="utf-8") as f:
                    review_cases = json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Could not read existing review file, starting fresh.")

        from src.models.review_schemas import HumanReviewCase, ReviewStatus

        review_case = HumanReviewCase(
            record_id=violation_record.record_id,
            violation=violation_record.violation,
            confidence=violation_record.confidence,
            explanation=violation_record.explanation,
            cited_rule_ids=violation_record.cited_rule_ids,
            analysis_type=violation_record.analysis_type,
            review_status=ReviewStatus.PENDING,
        )

        review_cases.append(review_case.model_dump(mode="json", exclude_none=True))

        with review_path.open("w", encoding="utf-8") as f:
            json.dump(review_cases, f, indent=2, ensure_ascii=False, default=str)

        logger.info("Routed to human review: %s (confidence: %.2f)", violation_record.record_id, violation_record.confidence)

    return decision


def process_evaluation_results(
    results: List[tuple[str, SQLComplianceResult | RAGComplianceResult]],
    high_confidence_path: Path,
    review_path: Path,
) -> List[DecisionRecord]:
    """
    Process a batch of evaluation results and route them based on confidence.

    Args:
        results: List of (record_id, result) tuples
        high_confidence_path: Path to high-confidence violations JSON
        review_path: Path to human review queue JSON

    Returns:
        List of DecisionRecords
    """
    decisions: List[DecisionRecord] = []

    for record_id, result in results:
        violation_record = create_violation_record(record_id, result)
        decision = route_violation(violation_record, high_confidence_path, review_path)
        decisions.append(decision)

    logger.info(
        "Processed %d results: %d auto-logged, %d sent to review",
        len(results),
        sum(1 for d in decisions if d.decision == DecisionType.AUTO_LOG),
        sum(1 for d in decisions if d.decision == DecisionType.HUMAN_REVIEW),
    )

    return decisions
