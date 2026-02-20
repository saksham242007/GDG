from __future__ import annotations

from pathlib import Path
from typing import List

from src.models.review_schemas import HumanReviewCase, ReviewDecision, ReviewStatus
from src.review.priority_algorithm import prioritize_review_queue
from src.review.review_interface import (
    display_review_case,
    load_review_cases,
    save_review_decision,
)
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


def create_review_batches(
    cases: List[HumanReviewCase],
    batch_size: int = 10,
) -> List[List[HumanReviewCase]]:
    """
    Create batches of review cases for efficient processing.

    Args:
        cases: List of review cases
        batch_size: Number of cases per batch

    Returns:
        List of batches (each batch is a list of cases)
    """
    # Prioritize cases first
    prioritized = prioritize_review_queue(cases)

    # Create batches
    batches: List[List[HumanReviewCase]] = []
    for i in range(0, len(prioritized), batch_size):
        batch = prioritized[i : i + batch_size]
        batches.append(batch)

    logger.info("Created %d review batches (batch size: %d)", len(batches), batch_size)
    return batches


def process_batch_review(
    batch: List[HumanReviewCase],
    review_path: Path,
    reviewed_path: Path,
    reviewer_id: str,
) -> List[ReviewDecision]:
    """
    Process a batch of review cases efficiently.

    Args:
        batch: List of cases in this batch
        review_path: Path to review queue JSON
        reviewed_path: Path to reviewed cases JSON
        reviewer_id: ID of reviewer

    Returns:
        List of review decisions made
    """
    decisions: List[ReviewDecision] = []

    print(f"\n{'=' * 80}")
    print(f"BATCH REVIEW - Processing {len(batch)} cases")
    print(f"{'=' * 80}")

    for idx, case in enumerate(batch, start=1):
        display_review_case(case, idx - 1, len(batch))

        print("\nQuick decision options:")
        print("  1. Approve violation")
        print("  2. Reject violation")
        print("  3. Mark compliant")
        print("  4. Skip (review later)")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            decision_status = ReviewStatus.APPROVED
        elif choice == "2":
            decision_status = ReviewStatus.REJECTED
        elif choice == "3":
            decision_status = ReviewStatus.MARKED_COMPLIANT
        else:
            continue  # Skip

        comment = input("Comment (optional, press Enter to skip): ").strip() or None

        decision = ReviewDecision(
            record_id=case.record_id,
            human_decision=decision_status,
            review_comment=comment,
            reviewer_id=reviewer_id,
        )

        save_review_decision(decision, review_path, reviewed_path)
        decisions.append(decision)

        print(f"✓ Decision saved for {case.record_id}\n")

    logger.info("Batch review complete: %d decisions made", len(decisions))
    return decisions


def run_batch_review_workflow(
    review_path: Path,
    reviewed_path: Path,
    batch_size: int = 10,
    reviewer_id: str = "batch_reviewer",
) -> None:
    """
    Run batch review workflow with prioritization.

    Args:
        review_path: Path to review queue
        reviewed_path: Path to reviewed cases
        batch_size: Cases per batch
        reviewer_id: Reviewer identifier
    """
    logger.info("Starting batch review workflow")

    cases = load_review_cases(review_path)

    if not cases:
        print("No pending review cases.")
        return

    # Create batches
    batches = create_review_batches(cases, batch_size)

    print(f"\nTotal cases: {len(cases)}")
    print(f"Batches: {len(batches)}")
    print(f"Batch size: {batch_size}")

    for batch_num, batch in enumerate(batches, start=1):
        print(f"\n{'=' * 80}")
        print(f"BATCH {batch_num} of {len(batches)}")
        print(f"{'=' * 80}")

        decisions = process_batch_review(batch, review_path, reviewed_path, reviewer_id)

        if batch_num < len(batches):
            continue_batch = input("\nContinue to next batch? (y/n): ").strip().lower()
            if continue_batch != "y":
                break

    logger.info("Batch review workflow completed")
