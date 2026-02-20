from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.models.schemas import ExtractedRule, RuleComplexity
from src.services.pdf_loader import load_policy_pdfs, split_documents
from src.services.rule_classifier import classify_rule
from src.services.rule_extractor import extract_rules_from_chunks
from src.services.sql_repository import init_db, persist_simple_rules
from src.services.validation_layer import validate_rules_batch, check_rule_consistency
from src.services.vector_store import persist_complex_rules_to_chroma
from src.ingestion.human_verification import verify_rules_batch
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICIES_DIR = PROJECT_ROOT / "data" / "policies"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SQL_DB_PATH = PROJECT_ROOT / "database" / "compliance_rules.db"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
EXTRACTED_RULES_JSON = PROCESSED_DIR / "extracted_rules.json"
VERIFICATION_LOG_PATH = PROCESSED_DIR / "verification_log.json"


def ensure_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SQL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def reclassify_rules(rules: List[ExtractedRule]) -> List[ExtractedRule]:
    """
    Apply deterministic classification heuristics to each rule, overriding the
    rule_complexity field when appropriate.
    """
    for rule in rules:
        complexity, rationale = classify_rule(rule)
        if rule.rule_complexity != complexity:
            logger.info(
                "Overriding rule_complexity for %s from %s to %s (%s)",
                rule.rule_id,
                rule.rule_complexity.value,
                complexity.value,
                rationale,
            )
            rule.rule_complexity = complexity
    return rules


def save_rules_to_json(rules: List[ExtractedRule], output_path: Path) -> None:
    """
    Persist all extracted rules to a JSON file for downstream analysis and audit.
    """
    payload = [rule.model_dump(mode="json") for rule in rules]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d rules to %s", len(rules), output_path)


def run_pipeline() -> None:
    """
    End-to-end Phase 1 ingestion pipeline:
    PDFs -> LLM extraction -> classification -> SQL + Chroma storage.
    """
    logger.info("Starting Policy Ingestion Pipeline (Phase 1).")
    ensure_directories()

    # Step 1: PDF Parsing
    docs = load_policy_pdfs(POLICIES_DIR)
    if not docs:
        logger.warning("No documents loaded; pipeline will exit.")
        return

    chunks = split_documents(docs)

    # Step 2: LLM Rule Extraction
    extracted_rules = extract_rules_from_chunks(chunks)
    if not extracted_rules:
        logger.warning("No rules extracted; pipeline will exit.")
        return

    # Step 3: Validation Layer
    logger.info("Validating extracted rules...")
    valid_rules, invalid_rules = validate_rules_batch(extracted_rules)
    
    if invalid_rules:
        logger.warning("%d rules failed validation", len(invalid_rules))
        for rule, errors in invalid_rules:
            logger.warning("Rule %s errors: %s", rule.rule_id, "; ".join(errors))

    # Check consistency
    consistency_warnings = check_rule_consistency(valid_rules)
    if consistency_warnings:
        logger.warning("Consistency warnings: %s", "; ".join(consistency_warnings))

    # Step 4: Human Verification (CRITICAL)
    logger.info("Human verification required for %d rules", len(valid_rules))
    approved_rules, rejected_rules = verify_rules_batch(
        valid_rules,
        VERIFICATION_LOG_PATH,
        auto_approve=False,  # Set to True for automated testing
    )

    if not approved_rules:
        logger.error("No rules approved after human verification. Pipeline stopped.")
        return

    logger.info("%d rules approved, %d rejected/pending", len(approved_rules), len(rejected_rules))

    # Step 5: Rule Classification Logic
    classified_rules = reclassify_rules(approved_rules)

    # Step 6a: Store simple rules in SQL database
    init_db(SQL_DB_PATH)
    simple_rules = [r for r in classified_rules if r.rule_complexity == RuleComplexity.SIMPLE]
    persist_simple_rules(SQL_DB_PATH, simple_rules)

    # Step 6b: Store complex rules in Chroma vector database
    complex_rules = [r for r in classified_rules if r.rule_complexity == RuleComplexity.COMPLEX]
    persist_complex_rules_to_chroma(CHROMA_DIR, complex_rules)

    # Output: JSON file with all rules
    save_rules_to_json(classified_rules, EXTRACTED_RULES_JSON)

    logger.info("Policy Ingestion Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()

