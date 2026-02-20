from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from src.models.review_schemas import HumanFeedback, ReviewDecision, ViolationRecord
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


def store_human_feedback(
    original_record: ViolationRecord,
    review_decision: ReviewDecision,
    feedback_dataset_path: Path,
    model_version: Optional[str] = None,
) -> None:
    """
    Store human feedback for model improvement and future training.

    Args:
        original_record: Original violation record from model
        review_decision: Human review decision
        feedback_dataset_path: Path to feedback dataset JSON
        model_version: Optional model version identifier
    """
    # Determine feedback type
    feedback_type = "correction"
    if review_decision.human_decision.value == "rejected":
        feedback_type = "false_positive"
    elif review_decision.human_decision.value == "marked_compliant":
        feedback_type = "false_negative"

    feedback = HumanFeedback(
        record_id=original_record.record_id,
        original_prediction=original_record,
        human_decision=review_decision,
        feedback_type=feedback_type,
        model_version=model_version,
    )

    # Load existing feedback
    feedback_dataset = []
    if feedback_dataset_path.exists():
        try:
            with feedback_dataset_path.open("r", encoding="utf-8") as f:
                feedback_dataset = json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Could not read existing feedback dataset, starting fresh.")

    # Append new feedback
    feedback_dataset.append(feedback.model_dump(mode="json", exclude_none=True))

    # Save feedback dataset
    feedback_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with feedback_dataset_path.open("w", encoding="utf-8") as f:
        json.dump(feedback_dataset, f, indent=2, ensure_ascii=False, default=str)

    logger.info(
        "Stored human feedback: %s (type: %s)",
        original_record.record_id,
        feedback_type,
    )


def load_feedback_dataset(feedback_dataset_path: Path) -> List[HumanFeedback]:
    """
    Load the human feedback dataset for analysis or training.
    """
    if not feedback_dataset_path.exists():
        return []

    try:
        with feedback_dataset_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return [HumanFeedback.model_validate(item) for item in data]

    except (json.JSONDecodeError, IOError) as e:
        logger.error("Failed to load feedback dataset: %s", e)
        return []


def generate_feedback_summary(feedback_dataset_path: Path) -> dict:
    """
    Generate summary statistics from feedback dataset.
    """
    feedbacks = load_feedback_dataset(feedback_dataset_path)

    if not feedbacks:
        return {
            "total_feedback_records": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "corrections": 0,
        }

    false_positives = sum(1 for f in feedbacks if f.feedback_type == "false_positive")
    false_negatives = sum(1 for f in feedbacks if f.feedback_type == "false_negative")
    corrections = sum(1 for f in feedbacks if f.feedback_type == "correction")

    return {
        "total_feedback_records": len(feedbacks),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "corrections": corrections,
        "false_positive_rate": false_positives / len(feedbacks) if feedbacks else 0.0,
        "false_negative_rate": false_negatives / len(feedbacks) if feedbacks else 0.0,
    }
