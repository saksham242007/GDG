"""
Unit tests for the shared utility functions.

Tests detect_input_type() and flag_non_compliant_case().
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.utils.shared import detect_input_type, flag_non_compliant_case
from src.models.evaluation_schemas import (
    AnalysisType,
    RAGComplianceResult,
    SQLComplianceResult,
    Violation,
)


class TestDetectInputType:
    """Tests for the detect_input_type function."""

    def test_sqlite_database(self, tmp_path):
        db_file = tmp_path / "company.db"
        db_file.touch()
        input_type, desc = detect_input_type(db_file)
        assert input_type == "structured"
        assert "SQL database" in desc

    def test_sqlite_extension(self, tmp_path):
        db_file = tmp_path / "data.sqlite"
        db_file.touch()
        input_type, desc = detect_input_type(db_file)
        assert input_type == "structured"

    def test_text_file(self, tmp_path):
        txt_file = tmp_path / "report.txt"
        txt_file.touch()
        input_type, desc = detect_input_type(txt_file)
        assert input_type == "unstructured"
        assert "Text" in desc

    def test_pdf_file(self, tmp_path):
        pdf_file = tmp_path / "policy.pdf"
        pdf_file.touch()
        input_type, desc = detect_input_type(pdf_file)
        assert input_type == "unstructured"

    def test_log_file(self, tmp_path):
        log_file = tmp_path / "audit.log"
        log_file.touch()
        input_type, desc = detect_input_type(log_file)
        assert input_type == "unstructured"

    def test_unknown_extension(self, tmp_path):
        xyz_file = tmp_path / "data.xyz"
        xyz_file.touch()
        input_type, desc = detect_input_type(xyz_file)
        assert input_type == "unstructured"
        assert ".xyz" in desc

    def test_csv_file(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.touch()
        input_type, desc = detect_input_type(csv_file)
        assert input_type == "structured"
        assert "CSV" in desc


class TestFlagNonCompliantCase:
    """Tests for the flag_non_compliant_case function."""

    def test_compliant_result_skipped(self, tmp_path):
        """Compliant results should NOT be flagged."""
        flags_path = tmp_path / "flags.json"
        result = SQLComplianceResult(
            evaluation_id="EVAL-001",
            analysis_type=AnalysisType.SQL,
            compliant=True,
            explanation="All rules passed",
            violations=[],
            policy_references=[],
            confidence=0.9,
        )
        flag_non_compliant_case(result, flags_path)
        assert not flags_path.exists()

    def test_non_compliant_sql_result_flagged(self, tmp_path):
        """Non-compliant SQL results should be written to flags file."""
        flags_path = tmp_path / "flags.json"
        result = SQLComplianceResult(
            evaluation_id="EVAL-002",
            analysis_type=AnalysisType.SQL,
            compliant=False,
            explanation="Threshold exceeded",
            violations=[
                Violation(
                    rule_id="RULE-001",
                    violation_description="Amount exceeds limit",
                    severity="high",
                )
            ],
            policy_references=["RULE-001"],
            confidence=0.8,
        )
        flag_non_compliant_case(result, flags_path)
        assert flags_path.exists()
        flags = json.loads(flags_path.read_text())
        assert len(flags) == 1
        assert flags[0]["evaluation_id"] == "EVAL-002"

    def test_non_compliant_rag_result_flagged(self, tmp_path):
        """Non-compliant RAG results should be flagged with correct structure."""
        flags_path = tmp_path / "flags.json"
        result = RAGComplianceResult(
            evaluation_id="EVAL-003",
            analysis_type=AnalysisType.RAG,
            compliant=False,
            compliance_score=35.0,
            explanation="Policy violation detected",
            violated_policies=["POLICY-A", "POLICY-B"],
            confidence=0.85,
        )
        flag_non_compliant_case(result, flags_path)
        flags = json.loads(flags_path.read_text())
        assert len(flags) == 1
        assert flags[0]["analysis_type"] == "RAG"
        assert len(flags[0]["violations"]) == 2

    def test_multiple_flags_append(self, tmp_path):
        """Multiple calls should append to the same flags file."""
        flags_path = tmp_path / "flags.json"

        for i in range(3):
            result = SQLComplianceResult(
                evaluation_id=f"EVAL-{i}",
                analysis_type=AnalysisType.SQL,
                compliant=False,
                explanation=f"Violation {i}",
                violations=[
                    Violation(
                        rule_id=f"RULE-{i}",
                        violation_description=f"Issue {i}",
                        severity="medium",
                    )
                ],
                policy_references=[f"RULE-{i}"],
                confidence=0.7,
            )
            flag_non_compliant_case(result, flags_path)

        flags = json.loads(flags_path.read_text())
        assert len(flags) == 3
