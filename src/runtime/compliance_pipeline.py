from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from src.models.evaluation_schemas import (
    AnalysisType,
    RAGComplianceResult,
    SQLComplianceResult,
)
from src.runtime.human_gate import prompt_human_decision
from src.runtime.rag_engine import evaluate_rag_compliance
from src.runtime.sql_engine import evaluate_sql_compliance
from src.utils.logging_config import get_logger
from src.utils.shared import detect_input_type, flag_non_compliant_case


logger = get_logger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
RULES_DB_PATH = PROJECT_ROOT / "database" / "compliance_rules.db"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVALUATIONS_DIR = OUTPUTS_DIR / "evaluations"
AUDIT_DIR = OUTPUTS_DIR / "audit"
FLAGS_DIR = OUTPUTS_DIR / "flags"
COMPANY_DATA_DIR = PROJECT_ROOT / "data" / "company_data"




def run_compliance_pipeline(
    company_input_path: Path,
    evaluation_id: Optional[str] = None,
) -> None:
    """
    Main Phase 2 compliance evaluation pipeline.

    Args:
        company_input_path: Path to company data (SQL database or unstructured file)
        evaluation_id: Optional evaluation ID (generated if not provided)
    """
    logger.info("=" * 80)
    logger.info("Starting Phase 2: Runtime Compliance Evaluation Pipeline")
    logger.info("=" * 80)

    # Ensure directories exist
    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    FLAGS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate evaluation ID
    if not evaluation_id:
        evaluation_id = f"EVAL-{uuid.uuid4().hex[:8].upper()}"

    logger.info("Evaluation ID: %s", evaluation_id)

    # Step 1: Load Company Input & Detect Type
    if not company_input_path.exists():
        logger.error("Company input path does not exist: %s", company_input_path)
        return

    input_type, input_description = detect_input_type(company_input_path)
    logger.info("Detected input type: %s (%s)", input_type, input_description)

    # Step 2: HUMAN DECISION GATE (CRITICAL)
    audit_log_path = AUDIT_DIR / "human_decisions.json"
    selected_analysis = prompt_human_decision(
        evaluation_id=evaluation_id,
        input_type=input_type,
        input_source=str(company_input_path),
        audit_log_path=audit_log_path,
    )

    logger.info("Human selected analysis path: %s", selected_analysis.value)

    # Step 3: Route to Appropriate Engine
    result: SQLComplianceResult | RAGComplianceResult

    if selected_analysis == AnalysisType.SQL:
        # PATH A: SQL Compliance Engine
        logger.info("Routing to SQL Compliance Engine...")
        sql_output_path = EVALUATIONS_DIR / f"sql_results_{evaluation_id}.json"
        result = evaluate_sql_compliance(
            evaluation_id=evaluation_id,
            company_db_path=company_input_path,
            rules_db_path=RULES_DB_PATH,
            output_path=sql_output_path,
        )

    else:  # RAG
        # PATH B: Policy RAG + LLM Semantic Analysis
        logger.info("Routing to Policy RAG + LLM Engine...")
        rag_output_path = EVALUATIONS_DIR / f"rag_results_{evaluation_id}.json"
        result = evaluate_rag_compliance(
            evaluation_id=evaluation_id,
            company_data_path=company_input_path,
            chroma_dir=POLICY_CHROMA_DIR,
            output_path=rag_output_path,
        )

    # Step 4: Compliance Decision Logic & Flagging
    if isinstance(result, RAGComplianceResult):
        is_compliant = result.compliance_score >= 75.0
        if not is_compliant:
            logger.warning(
                "Non-compliant case detected: Score=%.1f%% (< 75%%)",
                result.compliance_score,
            )
            flag_non_compliant_case(result, FLAGS_DIR / "non_compliant_cases.json")
    else:  # SQL
        is_compliant = result.compliant
        if not is_compliant:
            logger.warning("Non-compliant case detected: %d violations", len(result.violations))
            flag_non_compliant_case(result, FLAGS_DIR / "non_compliant_cases.json")

    # Final summary
    logger.info("=" * 80)
    logger.info("Compliance Evaluation Complete")
    logger.info("=" * 80)
    logger.info("Evaluation ID: %s", evaluation_id)
    logger.info("Analysis Type: %s", result.analysis_type.value)
    logger.info("Compliant: %s", "YES" if is_compliant else "NO")
    if isinstance(result, RAGComplianceResult):
        logger.info("Compliance Score: %.1f%%", result.compliance_score)
    logger.info("Confidence: %.2f", result.confidence)
    logger.info("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.runtime.compliance_pipeline <company_input_path>")
        print("\nExample:")
        print("  python -m src.runtime.compliance_pipeline data/company_data/transactions.db")
        print("  python -m src.runtime.compliance_pipeline data/company_data/report.txt")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    run_compliance_pipeline(input_path)
