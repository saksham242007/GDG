"""
End-to-End Pipeline Test Script
================================
Tests the full compliance monitoring pipeline:
  Phase 1: Extract rules from AML policy PDF
  Phase 2: Evaluate company transaction data against extracted rules

Usage:
    python run_e2e_test.py
"""

import sys
import json
from pathlib import Path

# Ensure project root is in PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import get_logger

logger = get_logger("e2e_test")


def phase1_extract_rules():
    """Phase 1: Extract rules from the AML policy PDF."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Rule Extraction from Policy PDF")
    logger.info("=" * 60)

    from src.services.pdf_loader import load_and_split_pdf
    from src.services.rule_extractor import extract_rules_from_chunks
    from src.services.rule_classifier import classify_rule
    from src.services.sql_repository import get_rules_engine, ComplianceRule
    from sqlalchemy.orm import Session

    # 1. Load and chunk the policy PDF
    pdf_path = PROJECT_ROOT / "data" / "policies" / "AML_Compliance_Policy_Rules_Extended.pdf"
    if not pdf_path.exists():
        logger.error("Policy PDF not found: %s", pdf_path)
        return False

    logger.info("Loading PDF: %s", pdf_path)
    chunks = load_and_split_pdf(str(pdf_path))
    logger.info("Split PDF into %d chunks", len(chunks))

    # 2. Extract rules via LLM
    logger.info("Extracting rules via LLM (this may take a minute)...")
    rules = extract_rules_from_chunks(chunks)
    logger.info("Extracted %d rules total", len(rules))

    # 3. Classify rules and log summary
    simple_count = 0
    complex_count = 0
    for rule in rules:
        complexity, reason = classify_rule(rule)
        rule.rule_complexity = complexity
        if complexity.value == "simple":
            simple_count += 1
        else:
            complex_count += 1
        logger.info(
            "  [%s] %s: %s (%s)",
            complexity.value.upper(),
            rule.rule_id,
            rule.rule_text[:80],
            reason,
        )

    logger.info("Classification: %d SIMPLE (SQL), %d COMPLEX (RAG)", simple_count, complex_count)

    # 4. Store rules in database
    engine = get_rules_engine()
    from sqlalchemy.orm import sessionmaker
    from src.services.sql_repository import Base

    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    for rule in rules:
        db_rule = ComplianceRule(
            rule_id=rule.rule_id,
            rule_text=rule.rule_text,
            condition=rule.condition,
            threshold=rule.threshold,
            required_action=rule.required_action,
            rule_complexity=rule.rule_complexity.value if rule.rule_complexity else "complex",
            source_pdf=str(pdf_path.name),
        )
        session.merge(db_rule)  # Use merge to handle re-runs

    session.commit()
    session.close()
    logger.info("Stored %d rules in database", len(rules))

    # 5. Save rules as JSON for reference
    rules_json = PROJECT_ROOT / "outputs" / "extracted_rules.json"
    rules_json.parent.mkdir(parents=True, exist_ok=True)
    with open(rules_json, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "rule_id": r.rule_id,
                    "rule_text": r.rule_text,
                    "condition": r.condition,
                    "threshold": r.threshold,
                    "required_action": r.required_action,
                    "complexity": r.rule_complexity.value if r.rule_complexity else "unknown",
                }
                for r in rules
            ],
            f,
            indent=2,
        )
    logger.info("Saved extracted rules to: %s", rules_json)

    return len(rules) > 0


def phase2_evaluate_data():
    """Phase 2: Run compliance evaluation on transaction data."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Compliance Evaluation")
    logger.info("=" * 60)

    csv_path = PROJECT_ROOT / "csv" / "LI-Small_Trans.csv"
    if not csv_path.exists():
        logger.error("Transaction CSV not found: %s", csv_path)
        return False

    logger.info("Transaction data: %s", csv_path)

    # Run the compliance graph
    from src.runtime.compliance_graph import run_compliance_graph

    try:
        final_state = run_compliance_graph(csv_path)

        logger.info("=" * 60)
        logger.info("E2E RESULTS")
        logger.info("=" * 60)
        logger.info("Evaluation ID: %s", final_state.evaluation_id)
        logger.info("Compliant: %s", "YES" if final_state.is_compliant else "NO")
        if final_state.compliance_score is not None:
            logger.info("Score: %.1f%%", final_state.compliance_score)
        if final_state.final_result:
            logger.info("Analysis Type: %s", final_state.final_result.analysis_type.value)
            logger.info("Confidence: %.2f", final_state.final_result.confidence)
        if final_state.error:
            logger.error("Error: %s", final_state.error)

        return True

    except Exception as e:
        logger.error("Phase 2 failed: %s", e, exc_info=True)
        return False


if __name__ == "__main__":
    logger.info("Starting End-to-End Compliance Pipeline Test")
    logger.info("=" * 60)

    # Phase 1: Extract rules
    if not phase1_extract_rules():
        logger.error("Phase 1 failed — cannot proceed")
        sys.exit(1)

    logger.info("")

    # Phase 2: Evaluate
    if not phase2_evaluate_data():
        logger.error("Phase 2 failed")
        sys.exit(1)

    logger.info("")
    logger.info("E2E test completed successfully!")
