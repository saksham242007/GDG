"""
Shared utility functions used across the compliance pipeline modules.

Centralizes common logic like input type detection and non-compliant case flagging
to eliminate code duplication.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from src.models.evaluation_schemas import (
    NonCompliantCase,
    RAGComplianceResult,
    SQLComplianceResult,
    Violation,
)
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


def detect_input_type(input_path: Path) -> tuple[str, str]:
    """
    Detect whether input is structured (SQL database) or unstructured (text file).

    Args:
        input_path: Path to the input file

    Returns:
        Tuple of (input_type, description)
    """
    suffix = input_path.suffix.lower()
    if suffix in (".db", ".sqlite"):
        return "structured", "SQL database"
    elif suffix in (".csv",):
        return "structured", "CSV file"
    elif suffix in (".txt", ".log", ".md", ".pdf", ".docx"):
        return "unstructured", "Text/document file"
    else:
        return "unstructured", f"File ({suffix})"


def flag_non_compliant_case(
    result: Union[SQLComplianceResult, RAGComplianceResult],
    flags_path: Path,
) -> None:
    """
    Flag non-compliant cases and persist to JSON for human review queue.

    Args:
        result: Compliance evaluation result (SQL or RAG)
        flags_path: Path to the flags JSON file
    """
    if result.compliant:
        return

    violations: list[Violation] = []
    if isinstance(result, SQLComplianceResult):
        violations = result.violations
    elif isinstance(result, RAGComplianceResult):
        violations = [
            Violation(
                rule_id=policy_id,
                violation_description=f"Policy violation: {policy_id}",
                severity="high",
            )
            for policy_id in result.violated_policies
        ]

    non_compliant_case = NonCompliantCase(
        evaluation_id=result.evaluation_id,
        analysis_type=result.analysis_type,
        compliance_score=(
            result.compliance_score if isinstance(result, RAGComplianceResult) else None
        ),
        violations=violations,
        explanation=result.explanation,
        review_status="pending",
    )

    # Load existing flags
    flags = []
    if flags_path.exists():
        try:
            with flags_path.open("r", encoding="utf-8") as f:
                flags = json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Could not read existing flags file, starting fresh.")

    flags.append(non_compliant_case.model_dump(mode="json", exclude_none=True))

    # Save flags
    flags_path.parent.mkdir(parents=True, exist_ok=True)
    with flags_path.open("w", encoding="utf-8") as f:
        json.dump(flags, f, indent=2, ensure_ascii=False, default=str)

    logger.warning(
        "Non-compliant case flagged: %s (Score: %s)",
        result.evaluation_id,
        (
            f"{result.compliance_score:.1f}%"
            if isinstance(result, RAGComplianceResult)
            else "N/A"
        ),
    )
