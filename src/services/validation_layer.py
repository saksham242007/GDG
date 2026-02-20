from __future__ import annotations

from typing import List

from src.models.schemas import ExtractedRule
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when rule validation fails."""

    pass


def validate_rule(rule: ExtractedRule) -> tuple[bool, List[str]]:
    """
    Validate a single extracted rule for consistency and completeness.

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors: List[str] = []

    # Check required fields
    if not rule.rule_id or len(rule.rule_id.strip()) == 0:
        errors.append("Rule ID is missing or empty")

    if not rule.rule_text or len(rule.rule_text.strip()) < 10:
        errors.append("Rule text is too short or missing")

    if not rule.condition or len(rule.condition.strip()) == 0:
        errors.append("Condition is missing or empty")

    if not rule.required_action or len(rule.required_action.strip()) == 0:
        errors.append("Required action is missing or empty")

    # Validate threshold if provided
    if rule.threshold is not None:
        if not isinstance(rule.threshold, (int, float)):
            errors.append("Threshold must be numeric")
        elif rule.threshold < 0:
            errors.append("Threshold cannot be negative")

    # Check for duplicate rule IDs (would need context, but basic check)
    if len(rule.rule_id) < 3:
        errors.append("Rule ID is too short")

    # Validate complexity classification
    if rule.rule_complexity.value not in ["simple", "complex"]:
        errors.append(f"Invalid complexity: {rule.rule_complexity.value}")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_rules_batch(rules: List[ExtractedRule]) -> tuple[List[ExtractedRule], List[tuple[ExtractedRule, List[str]]]]:
    """
    Validate a batch of rules.

    Returns:
        Tuple of (valid_rules, invalid_rules_with_errors)
    """
    valid_rules: List[ExtractedRule] = []
    invalid_rules: List[tuple[ExtractedRule, List[str]]] = []

    for rule in rules:
        is_valid, errors = validate_rule(rule)
        if is_valid:
            valid_rules.append(rule)
        else:
            invalid_rules.append((rule, errors))
            logger.warning(
                "Rule %s failed validation: %s",
                rule.rule_id,
                "; ".join(errors),
            )

    logger.info(
        "Validation complete: %d valid, %d invalid rules",
        len(valid_rules),
        len(invalid_rules),
    )

    return valid_rules, invalid_rules


def check_rule_consistency(rules: List[ExtractedRule]) -> List[str]:
    """
    Check for consistency issues across rules (e.g., duplicate IDs, conflicting rules).

    Returns:
        List of consistency warnings
    """
    warnings: List[str] = []

    # Check for duplicate rule IDs
    rule_ids = [r.rule_id for r in rules]
    duplicates = [rid for rid in rule_ids if rule_ids.count(rid) > 1]
    if duplicates:
        warnings.append(f"Duplicate rule IDs found: {set(duplicates)}")

    # Check for conflicting thresholds (same condition, different thresholds)
    condition_thresholds: dict[str, List[float]] = {}
    for rule in rules:
        if rule.threshold is not None:
            key = rule.condition.lower().strip()
            if key not in condition_thresholds:
                condition_thresholds[key] = []
            condition_thresholds[key].append(rule.threshold)

    for condition, thresholds in condition_thresholds.items():
        if len(set(thresholds)) > 1:
            warnings.append(
                f"Conflicting thresholds for condition '{condition}': {thresholds}"
            )

    return warnings
