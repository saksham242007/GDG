from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models.evaluation_schemas import AnalysisType, HumanDecision
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


def prompt_human_decision(
    evaluation_id: str,
    input_type: str,
    input_source: str,
    audit_log_path: Path,
) -> AnalysisType:
    """
    Human-in-the-Loop decision gate: prompts user to choose SQL or RAG analysis path.

    This function MUST pause execution and wait for human input before proceeding.

    Args:
        evaluation_id: Unique evaluation session ID
        input_type: Type of input ("structured" or "unstructured")
        input_source: Path or identifier of the input data
        audit_log_path: Path to the audit log file for human decisions

    Returns:
        Selected AnalysisType (SQL or RAG)
    """
    print("\n" + "=" * 80)
    print("HUMAN DECISION GATE - Compliance Evaluation Routing")
    print("=" * 80)
    print(f"Evaluation ID: {evaluation_id}")
    print(f"Input Type: {input_type}")
    print(f"Input Source: {input_source}")
    print("\nPlease select the analysis path:")
    print("\n  Option A: SQL Rule Engine")
    print("    → Fast deterministic checks against structured rules")
    print("    → Suitable for: Database tables, transactions, structured records")
    print("\n  Option B: Policy RAG + LLM Semantic Analysis")
    print("    → Semantic evaluation using policy embeddings")
    print("    → Suitable for: Documents, logs, text files, unstructured data")
    print("\n" + "-" * 80)

    while True:
        choice = input("\nEnter your choice (A for SQL, B for RAG): ").strip().upper()

        if choice == "A":
            selected_type = AnalysisType.SQL
            break
        elif choice == "B":
            selected_type = AnalysisType.RAG
            break
        else:
            print("Invalid choice. Please enter 'A' for SQL or 'B' for RAG.")

    # Optional: prompt for operator name and rationale
    operator = input("\nEnter operator name (optional, press Enter to skip): ").strip()
    if not operator:
        operator = None

    rationale = input("Enter decision rationale (optional, press Enter to skip): ").strip()
    if not rationale:
        rationale = None

    # Log the human decision
    decision = HumanDecision(
        evaluation_id=evaluation_id,
        timestamp=datetime.now(),
        analysis_type=selected_type,
        input_type=input_type,
        input_source=input_source,
        human_operator=operator,
        rationale=rationale,
    )

    log_human_decision(decision, audit_log_path)

    logger.info(
        "Human decision recorded: %s selected for evaluation %s",
        selected_type.value,
        evaluation_id,
    )

    return selected_type


def log_human_decision(decision: HumanDecision, audit_log_path: Path) -> None:
    """
    Persist human decision to audit log file (append mode).
    """
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing decisions if file exists
    decisions = []
    if audit_log_path.exists():
        try:
            with audit_log_path.open("r", encoding="utf-8") as f:
                decisions = json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Could not read existing audit log, starting fresh.")

    # Append new decision
    decisions.append(decision.model_dump(mode="json", exclude_none=True))

    # Write back
    with audit_log_path.open("w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Human decision logged to %s", audit_log_path)
