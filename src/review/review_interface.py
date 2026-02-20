from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.models.review_schemas import HumanReviewCase, ReviewDecision, ReviewStatus
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


def load_review_cases(review_path: Path) -> List[HumanReviewCase]:
    """
    Load pending review cases from JSON file.
    """
    if not review_path.exists():
        return []

    try:
        with review_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        cases = [HumanReviewCase.model_validate(item) for item in data]
        # Filter to only pending cases
        return [c for c in cases if c.review_status == ReviewStatus.PENDING]

    except (json.JSONDecodeError, IOError) as e:
        logger.error("Failed to load review cases: %s", e)
        return []


def display_review_case(case: HumanReviewCase, index: int, total: int) -> None:
    """
    Display a review case in a formatted way.
    """
    print("\n" + "=" * 80)
    print(f"REVIEW CASE {index + 1} of {total}")
    print("=" * 80)
    print(f"Record ID: {case.record_id}")
    print(f"Analysis Type: {case.analysis_type}")
    print(f"Violation Detected: {'YES' if case.violation else 'NO'}")
    print(f"Confidence Score: {case.confidence:.2%}")
    print(f"\nExplanation:")
    print(f"  {case.explanation}")
    print(f"\nCited Policy Rules:")
    for rule_id in case.cited_rule_ids:
        print(f"  - {rule_id}")
    if case.record_details:
        print(f"\nRecord Details:")
        for key, value in case.record_details.items():
            print(f"  {key}: {value}")
    print("=" * 80)


def prompt_review_decision(case: HumanReviewCase) -> ReviewDecision:
    """
    CLI prompt for human review decision.
    """
    print("\nSelect your decision:")
    print("  1. Approve violation (confirm violation exists)")
    print("  2. Reject violation (false positive)")
    print("  3. Mark as compliant (no violation)")
    print("  4. Skip (review later)")

    while True:
        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            decision_status = ReviewStatus.APPROVED
            break
        elif choice == "2":
            decision_status = ReviewStatus.REJECTED
            break
        elif choice == "3":
            decision_status = ReviewStatus.MARKED_COMPLIANT
            break
        elif choice == "4":
            return None  # Skip this case
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

    reviewer_id = input("\nEnter your reviewer ID: ").strip()
    if not reviewer_id:
        reviewer_id = "anonymous"

    comment = input("Enter review comment (optional, press Enter to skip): ").strip()
    if not comment:
        comment = None

    return ReviewDecision(
        record_id=case.record_id,
        human_decision=decision_status,
        review_comment=comment,
        reviewer_id=reviewer_id,
    )


def save_review_decision(
    decision: ReviewDecision,
    review_path: Path,
    reviewed_path: Path,
) -> None:
    """
    Save review decision and update review cases.
    """
    # Load all review cases
    all_cases = []
    if review_path.exists():
        try:
            with review_path.open("r", encoding="utf-8") as f:
                all_cases = json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Could not read review cases file")

    # Update the case status
    for case_data in all_cases:
        if case_data["record_id"] == decision.record_id:
            case_data["review_status"] = decision.human_decision.value
            case_data["reviewed_at"] = decision.timestamp.isoformat()
            case_data["reviewer_id"] = decision.reviewer_id
            if decision.review_comment:
                case_data["review_comment"] = decision.review_comment
            break

    # Save updated cases
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", encoding="utf-8") as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False, default=str)

    # Save to reviewed cases log
    reviewed_cases = []
    if reviewed_path.exists():
        try:
            with reviewed_path.open("r", encoding="utf-8") as f:
                reviewed_cases = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    reviewed_cases.append(decision.model_dump(mode="json", exclude_none=True))

    reviewed_path.parent.mkdir(parents=True, exist_ok=True)
    with reviewed_path.open("w", encoding="utf-8") as f:
        json.dump(reviewed_cases, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Review decision saved: %s -> %s", decision.record_id, decision.human_decision.value)


def run_cli_review_interface(
    review_path: Path,
    reviewed_path: Path,
) -> None:
    """
    Run the CLI-based human review interface.
    """
    logger.info("Starting CLI Review Interface")

    cases = load_review_cases(review_path)

    if not cases:
        print("\nNo pending review cases found.")
        return

    print(f"\nFound {len(cases)} pending review case(s).")

    for idx, case in enumerate(cases):
        display_review_case(case, idx, len(cases))
        decision = prompt_review_decision(case)

        if decision is None:
            print("Skipping this case...")
            continue

        save_review_decision(decision, review_path, reviewed_path)
        print(f"\n✓ Decision saved for {case.record_id}")

        if idx < len(cases) - 1:
            continue_review = input("\nContinue to next case? (y/n): ").strip().lower()
            if continue_review != "y":
                break

    logger.info("CLI Review Interface completed")


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    review_path = PROJECT_ROOT / "outputs" / "review" / "needs_review.json"
    reviewed_path = PROJECT_ROOT / "outputs" / "review" / "reviewed_cases.json"

    run_cli_review_interface(review_path, reviewed_path)
