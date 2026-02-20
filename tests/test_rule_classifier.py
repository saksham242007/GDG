"""
Unit tests for the Rule Classifier module.

Tests the heuristic-based classifier that determines if a rule is
'simple' (deterministic, SQL-checkable) or 'complex' (contextual/semantic).
"""

import pytest
from src.models.schemas import ExtractedRule, RuleComplexity
from src.services.rule_classifier import classify_rule


class TestRuleClassifier:
    """Tests for the classify_rule function."""

    def test_simple_rule_with_threshold(self):
        """Rules with numeric thresholds should be classified as SIMPLE."""
        rule = ExtractedRule(
            rule_id="TEST-001",
            rule_text="Transactions exceeding $10,000 must be reported.",
            condition="transaction_amount > 10000",
            threshold=10000.0,
            required_action="Report to compliance officer",
            rule_complexity=RuleComplexity.SIMPLE,
        )
        result = classify_rule(rule)
        assert result[0] == RuleComplexity.SIMPLE

    def test_keyword_rule_without_numeric_threshold(self):
        """Rules without numeric patterns are classified as COMPLEX by the heuristic."""
        rule = ExtractedRule(
            rule_id="TEST-002",
            rule_text="All invoices must be approved before payment.",
            condition="invoice_status == 'approved'",
            threshold=None,
            required_action="Require manager approval",
            rule_complexity=RuleComplexity.SIMPLE,
        )
        result = classify_rule(rule)
        # The classifier uses numeric/comparison pattern detection,
        # so keyword-only conditions are classified as COMPLEX
        assert result[0] == RuleComplexity.COMPLEX

    def test_complex_rule_with_vague_language(self):
        """Rules with vague/contextual language should be classified as COMPLEX."""
        rule = ExtractedRule(
            rule_id="TEST-003",
            rule_text="Employees should generally follow best practices for data handling.",
            condition="subjective assessment of data handling practices",
            threshold=None,
            required_action="Review practices periodically",
            rule_complexity=RuleComplexity.COMPLEX,
        )
        result = classify_rule(rule)
        assert result[0] == RuleComplexity.COMPLEX

    def test_complex_rule_with_semantic_keywords(self):
        """Rules mentioning 'reasonable' or 'context' should be COMPLEX."""
        rule = ExtractedRule(
            rule_id="TEST-004",
            rule_text="A reasonable effort must be made to ensure compliance.",
            condition="reasonable effort in context",
            threshold=None,
            required_action="Ensure reasonable measures",
            rule_complexity=RuleComplexity.COMPLEX,
        )
        result = classify_rule(rule)
        assert result[0] == RuleComplexity.COMPLEX

    def test_null_check_without_numeric_pattern(self):
        """Rules with null checks but no numeric patterns are COMPLEX."""
        rule = ExtractedRule(
            rule_id="TEST-005",
            rule_text="All purchase orders must have a valid PO number.",
            condition="po_number is not null",
            threshold=None,
            required_action="must have PO number",
            rule_complexity=RuleComplexity.SIMPLE,
        )
        result = classify_rule(rule)
        assert result[0] == RuleComplexity.COMPLEX
