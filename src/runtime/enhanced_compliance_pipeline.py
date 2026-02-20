from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from src.models.evaluation_schemas import (
    AnalysisType,
    RAGComplianceResult,
    SQLComplianceResult,
    Violation,
)
from src.runtime.citation_validator import enrich_with_citation_metadata
from src.runtime.exception_registry import get_exception_registry
from src.runtime.rag_engine import evaluate_rag_compliance
from src.runtime.scan_strategy import determine_scan_strategy, route_by_scan_strategy, ScanStrategy
from src.runtime.sql_aggregate_engine import evaluate_aggregate_compliance
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
EXCEPTION_REGISTRY_PATH = PROJECT_ROOT / "data" / "exception_registry.json"
COMPANY_DATA_DIR = PROJECT_ROOT / "data" / "company_data"



def run_enhanced_compliance_pipeline(
    company_input_path: Path,
    evaluation_id: Optional[str] = None,
    scan_strategy: Optional[ScanStrategy] = None,
    force_full_audit: bool = False,
) -> None:
    """
    Enhanced Phase 2 compliance pipeline with scan strategy routing and exception registry.

    Args:
        company_input_path: Path to company data
        evaluation_id: Optional evaluation ID
        scan_strategy: Optional scan strategy (auto-determined if None)
        force_full_audit: Force full audit scan
    """
    logger.info("=" * 80)
    logger.info("Starting Enhanced Phase 2: Runtime Compliance Evaluation Pipeline")
    logger.info("=" * 80)

    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    FLAGS_DIR.mkdir(parents=True, exist_ok=True)

    if not evaluation_id:
        evaluation_id = f"EVAL-{uuid.uuid4().hex[:8].upper()}"

    logger.info("Evaluation ID: %s", evaluation_id)

    if not company_input_path.exists():
        logger.error("Company input path does not exist: %s", company_input_path)
        return

    input_type, input_description = detect_input_type(company_input_path)
    logger.info("Detected input type: %s (%s)", input_type, input_description)

    # Step 1: Determine Scan Strategy
    if scan_strategy is None:
        # Load last scan timestamp (simplified - in production would come from state)
        scan_strategy = determine_scan_strategy(
            company_input_path,
            last_scan_timestamp=None,  # Would load from state
            force_full_audit=force_full_audit,
        )

    logger.info("Scan Strategy: %s", scan_strategy.value)

    # Step 2: Route by Scan Strategy
    engine_type = route_by_scan_strategy(scan_strategy)

    # Step 3: Load Exception Registry
    exception_registry = get_exception_registry(EXCEPTION_REGISTRY_PATH)

    # Step 4: Execute Evaluation
    result: SQLComplianceResult | RAGComplianceResult

    if engine_type == "sql_direct":
        # SQL Direct Engine (Incremental Scan)
        logger.info("Executing SQL Direct Engine (incremental scan)...")
        sql_output_path = EVALUATIONS_DIR / f"sql_results_{evaluation_id}.json"
        result = evaluate_sql_compliance(
            evaluation_id=evaluation_id,
            company_db_path=company_input_path,
            rules_db_path=RULES_DB_PATH,
            output_path=sql_output_path,
        )

        # Also run SQL Aggregate Engine for complex checks
        logger.info("Executing SQL Aggregate Engine...")
        aggregate_result = evaluate_aggregate_compliance(
            evaluation_id=f"{evaluation_id}-AGG",
            company_db_path=company_input_path,
            rules_db_path=RULES_DB_PATH,
            output_path=EVALUATIONS_DIR / f"sql_aggregate_{evaluation_id}.json",
        )

        # Merge violations from both engines
        if isinstance(result, SQLComplianceResult) and isinstance(aggregate_result, SQLComplianceResult):
            result.violations.extend(aggregate_result.violations)
            result.policy_reference.extend(aggregate_result.policy_reference)
            result.confidence = min(result.confidence, aggregate_result.confidence)

    else:  # llm_semantic
        # LLM Semantic Analyzer (Full Audit)
        logger.info("Executing LLM Semantic Analyzer (full audit)...")
        rag_output_path = EVALUATIONS_DIR / f"rag_results_{evaluation_id}.json"
        result = evaluate_rag_compliance(
            evaluation_id=evaluation_id,
            company_data_path=company_input_path,
            chroma_dir=POLICY_CHROMA_DIR,
            output_path=rag_output_path,
        )

    # Step 5: Check Exception Registry
    if isinstance(result, SQLComplianceResult):
        violations_dict = [v.model_dump(mode="json", exclude_none=True) for v in result.violations]
        filtered_violations = exception_registry.filter_violations_with_exceptions(violations_dict)
        
        # Update result with filtered violations
        from src.models.evaluation_schemas import Violation
        result.violations = [Violation.model_validate(v) for v in filtered_violations]

    # Step 6: Compliance Decision Logic & Flagging
    if isinstance(result, RAGComplianceResult):
        is_compliant = result.compliance_score >= 75.0
        if not is_compliant:
            logger.warning("Non-compliant case detected: Score=%.1f%%", result.compliance_score)
            flag_non_compliant_case(result, FLAGS_DIR / "non_compliant_cases.json")
    else:  # SQL
        is_compliant = result.compliant
        if not is_compliant:
            logger.warning("Non-compliant case detected: %d violations", len(result.violations))
            flag_non_compliant_case(result, FLAGS_DIR / "non_compliant_cases.json")

    # Final summary
    logger.info("=" * 80)
    logger.info("Enhanced Compliance Evaluation Complete")
    logger.info("=" * 80)
    logger.info("Evaluation ID: %s", evaluation_id)
    logger.info("Scan Strategy: %s", scan_strategy.value)
    logger.info("Engine Type: %s", engine_type)
    logger.info("Analysis Type: %s", result.analysis_type.value)
    logger.info("Compliant: %s", "YES" if is_compliant else "NO")
    if isinstance(result, RAGComplianceResult):
        logger.info("Compliance Score: %.1f%%", result.compliance_score)
    logger.info("Confidence: %.2f", result.confidence)
    logger.info("=" * 80)



if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.runtime.enhanced_compliance_pipeline <company_input_path> [--full-audit]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    force_full = "--full-audit" in sys.argv

    run_enhanced_compliance_pipeline(input_path, force_full_audit=force_full)
