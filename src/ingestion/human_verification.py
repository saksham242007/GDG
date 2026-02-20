from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.models.schemas import ExtractedRule
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class VerificationStatus:
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"
    NEEDS_REVISION = "needs_revision"


def display_rule_for_verification(rule: ExtractedRule, index: int, total: int) -> None:
    """
    Display a rule for human verification.
    """
    print("\n" + "=" * 80)
    print(f"RULE VERIFICATION {index + 1} of {total}")
    print("=" * 80)
    print(f"Rule ID: {rule.rule_id}")
    print(f"Complexity: {rule.rule_complexity.value}")
    print(f"Source PDF: {rule.source_pdf or 'Unknown'}")
    print(f"Category: {rule.category or 'Uncategorized'}")
    print(f"\nRule Text:")
    print(f"  {rule.rule_text}")
    print(f"\nCondition:")
    print(f"  {rule.condition}")
    if rule.threshold is not None:
        print(f"\nThreshold: {rule.threshold}")
    print(f"\nRequired Action:")
    print(f"  {rule.required_action}")
    print("=" * 80)


def prompt_verification_decision(rule: ExtractedRule) -> tuple[str, Optional[str]]:
    """
    Prompt human for verification decision.

    Returns:
        Tuple of (status, comment)
    """
    print("\nSelect verification decision:")
    print("  1. Approve (rule is valid and ready)")
    print("  2. Reject (rule is invalid or incorrect)")
    print("  3. Needs Revision (rule needs modification)")
    print("  4. Skip (review later)")

    while True:
        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            return VerificationStatus.APPROVED, None
        elif choice == "2":
            comment = input("Enter rejection reason: ").strip()
            return VerificationStatus.REJECTED, comment
        elif choice == "3":
            comment = input("Enter revision notes: ").strip()
            return VerificationStatus.NEEDS_REVISION, comment
        elif choice == "4":
            return VerificationStatus.PENDING, None
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def verify_rules_batch(
    rules: List[ExtractedRule],
    verification_log_path: Path,
    auto_approve: bool = False,
) -> tuple[List[ExtractedRule], List[ExtractedRule]]:
    """
    Human verification workflow for a batch of rules.

    Args:
        rules: List of rules to verify
        verification_log_path: Path to save verification log
        auto_approve: If True, auto-approve all rules (for testing)

    Returns:
        Tuple of (approved_rules, rejected_rules)
    """
    if auto_approve:
        logger.info("Auto-approving all %d rules (auto_approve=True)", len(rules))
        return rules, []

    approved_rules: List[ExtractedRule] = []
    rejected_rules: List[ExtractedRule] = []
    verification_log: List[dict] = []

    print(f"\n{'=' * 80}")
    print(f"HUMAN VERIFICATION REQUIRED")
    print(f"{'=' * 80}")
    print(f"Total rules to verify: {len(rules)}")
    print(f"{'=' * 80}\n")

    for idx, rule in enumerate(rules):
        display_rule_for_verification(rule, idx, len(rules))

        status, comment = prompt_verification_decision(rule)

        verification_entry = {
            "rule_id": rule.rule_id,
            "status": status,
            "comment": comment,
            "verified_at": datetime.now().isoformat(),
            "verifier": input("Enter your name/ID (optional): ").strip() or "anonymous",
        }
        verification_log.append(verification_entry)

        if status == VerificationStatus.APPROVED:
            approved_rules.append(rule)
            logger.info("Rule %s approved", rule.rule_id)
        elif status == VerificationStatus.REJECTED:
            rejected_rules.append(rule)
            logger.info("Rule %s rejected: %s", rule.rule_id, comment)
        elif status == VerificationStatus.NEEDS_REVISION:
            rejected_rules.append(rule)
            logger.info("Rule %s needs revision: %s", rule.rule_id, comment)
        else:  # PENDING
            logger.info("Rule %s skipped (pending)", rule.rule_id)

        if idx < len(rules) - 1:
            continue_review = input("\nContinue to next rule? (y/n): ").strip().lower()
            if continue_review != "y":
                break

    # Save verification log
    verification_log_path.parent.mkdir(parents=True, exist_ok=True)
    if verification_log_path.exists():
        try:
            with verification_log_path.open("r", encoding="utf-8") as f:
                existing_log = json.load(f)
            verification_log = existing_log + verification_log
        except (json.JSONDecodeError, IOError):
            pass

    with verification_log_path.open("w", encoding="utf-8") as f:
        json.dump(verification_log, f, indent=2, ensure_ascii=False, default=str)

    logger.info(
        "Verification complete: %d approved, %d rejected/pending",
        len(approved_rules),
        len(rejected_rules),
    )

    return approved_rules, rejected_rules
