"""
Full 3-Phase Compliance Monitoring Pipeline
=============================================
Executes all three phases of the architecture end-to-end:

  Phase 1: Policy Ingestion
    PDF → Parse → LLM Extract → Validate → Classify → Store (SQL DB + ChromaDB)

  Phase 2: Runtime Violation Detection
    CSV/DB → Input Detection → SQL Engine → Violations

  Phase 3: Human Review & Reporting
    Violations → Decision Engine → Auto-Log / Review Queue
    → Alerting → Dashboard → Daily Report

Usage:
    python run_full_pipeline.py
"""

import sys
import json
import uuid
import hashlib
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import get_logger

logger = get_logger("full_pipeline")

# ── Directory constants ──────────────────────────────────────────────────────
POLICIES_DIR       = PROJECT_ROOT / "data"  / "policies"
PROCESSED_DIR      = PROJECT_ROOT / "data"  / "processed"
CHROMA_DIR         = PROJECT_ROOT / "data"  / "chroma_db"
FEEDBACK_DIR       = PROJECT_ROOT / "data"  / "feedback"
SQL_DB_PATH        = PROJECT_ROOT / "database" / "compliance_rules.db"
OUTPUTS_DIR        = PROJECT_ROOT / "outputs"
EVALUATIONS_DIR    = OUTPUTS_DIR  / "evaluations"
VIOLATIONS_DIR     = OUTPUTS_DIR  / "violations"
REVIEW_DIR         = OUTPUTS_DIR  / "review"
REPORTS_DIR        = OUTPUTS_DIR  / "reports"
AUDIT_DIR          = OUTPUTS_DIR  / "audit"
FLAGS_DIR          = OUTPUTS_DIR  / "flags"
ALERTS_DIR         = OUTPUTS_DIR  / "audit"

HIGH_CONF_PATH     = VIOLATIONS_DIR / "high_confidence.json"
REVIEW_QUEUE_PATH  = REVIEW_DIR    / "needs_review.json"
REVIEWED_CASES     = REVIEW_DIR    / "reviewed_cases.json"
FEEDBACK_PATH      = FEEDBACK_DIR  / "human_feedback_dataset.json"
ALERT_LOG_PATH     = AUDIT_DIR     / "alert_log.json"
LLM_FEEDBACK_LOG   = REPORTS_DIR   / "llm_feedback_log.json"
EXTRACTED_JSON     = PROCESSED_DIR / "extracted_rules.json"


POLICY_HASH_FILE   = PROCESSED_DIR / ".policy_hash"


def _compute_policies_hash(policies_dir: Path) -> str:
    """Compute a SHA-256 hash of all policy PDF files (name + size + mtime)."""
    h = hashlib.sha256()
    if policies_dir.exists():
        for pdf in sorted(policies_dir.glob("*.pdf")):
            stat = pdf.stat()
            h.update(f"{pdf.name}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return h.hexdigest()


def _policies_changed() -> bool:
    """Check if policy PDFs have changed since last ingestion."""
    current_hash = _compute_policies_hash(POLICIES_DIR)
    if not POLICY_HASH_FILE.exists():
        return True
    stored_hash = POLICY_HASH_FILE.read_text(encoding="utf-8").strip()
    return current_hash != stored_hash


def _save_policy_hash():
    """Save current policy hash after successful ingestion."""
    current_hash = _compute_policies_hash(POLICIES_DIR)
    POLICY_HASH_FILE.write_text(current_hash, encoding="utf-8")


def ensure_dirs():
    for d in [PROCESSED_DIR, CHROMA_DIR, FEEDBACK_DIR, SQL_DB_PATH.parent,
              EVALUATIONS_DIR, VIOLATIONS_DIR, REVIEW_DIR, REPORTS_DIR,
              AUDIT_DIR, FLAGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — POLICY INGESTION
# ═══════════════════════════════════════════════════════════════════════════════
def phase1_policy_ingestion(force: bool = False) -> bool:
    """
    Phase 1: PDF → Parse → LLM Extract → Validate → Classify
             → SQL DB (simple) + ChromaDB (complex)
    
    Skips re-ingestion if policy PDFs haven't changed (hash-based cache).
    Pass force=True to re-ingest regardless.
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: POLICY INGESTION")
    logger.info("=" * 70)

    # ── Cache check: skip if policies haven't changed ─────────────────────────
    if not force and not _policies_changed():
        if EXTRACTED_JSON.exists() and SQL_DB_PATH.exists():
            logger.info("⚡ Policy PDFs unchanged — skipping re-ingestion (using cached rules)")
            logger.info("  To force re-ingestion, delete %s or change policy PDFs", POLICY_HASH_FILE)
            logger.info("✓ PHASE 1 SKIPPED — using cached rules")
            return True

    logger.info("Policy PDFs changed or no cache found — running full ingestion...")

    from src.services.pdf_loader import load_and_split_pdf, load_policy_pdfs, split_documents
    from src.services.rule_extractor import extract_rules_from_chunks
    from src.services.rule_classifier import classify_rule
    from src.services.validation_layer import validate_rules_batch, check_rule_consistency
    from src.services.sql_repository import get_rules_engine, ComplianceRule, Base, init_db, persist_simple_rules
    from src.services.vector_store import persist_complex_rules_to_chroma
    from src.models.schemas import RuleComplexity

    # Step 1: Load & chunk all policy PDFs
    logger.info("Step 1: Loading policy PDFs from %s", POLICIES_DIR)
    docs = load_policy_pdfs(POLICIES_DIR)
    if not docs:
        logger.error("No PDF documents found in %s", POLICIES_DIR)
        return False
    chunks = split_documents(docs)
    logger.info("Loaded %d pages → %d chunks", len(docs), len(chunks))

    # Step 2: LLM rule extraction
    logger.info("Step 2: Extracting rules via LLM...")
    rules = extract_rules_from_chunks(chunks)
    if not rules:
        logger.error("No rules extracted — stopping.")
        return False
    logger.info("Extracted %d rules", len(rules))

    # Step 3: Validation layer
    logger.info("Step 3: Validating extracted rules...")
    valid_rules, invalid_rules = validate_rules_batch(rules)
    if invalid_rules:
        for rule, errors in invalid_rules:
            logger.warning("  INVALID %s: %s", rule.rule_id, "; ".join(errors))
    consistency_warnings = check_rule_consistency(valid_rules)
    if consistency_warnings:
        for w in consistency_warnings:
            logger.warning("  Consistency: %s", w)
    logger.info("Validation: %d valid, %d invalid", len(valid_rules), len(invalid_rules))

    if not valid_rules:
        logger.error("No valid rules after validation — stopping.")
        return False

    # Step 4: Human verification (auto-approve for pipeline run)
    logger.info("Step 4: Human verification — auto-approving %d rules", len(valid_rules))
    approved_rules = valid_rules  # auto_approve=True equivalent

    # Step 5: Classification
    logger.info("Step 5: Classifying rules (simple vs complex)...")
    for rule in approved_rules:
        complexity, reason = classify_rule(rule)
        rule.rule_complexity = complexity
        logger.info("  [%s] %s — %s", complexity.value.upper(), rule.rule_id, reason[:80])

    simple_rules = [r for r in approved_rules if r.rule_complexity == RuleComplexity.SIMPLE]
    complex_rules = [r for r in approved_rules if r.rule_complexity == RuleComplexity.COMPLEX]
    logger.info("Classification: %d SIMPLE (SQL) + %d COMPLEX (RAG)", len(simple_rules), len(complex_rules))

    # Step 6a: Store SIMPLE rules → SQL database
    logger.info("Step 6a: Storing simple rules in SQL database...")
    init_db()
    # Use direct session approach for all rules (including rule_complexity column)
    from sqlalchemy.orm import sessionmaker
    engine = get_rules_engine()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    for rule in approved_rules:
        db_rule = ComplianceRule(
            rule_id=rule.rule_id,
            rule_text=rule.rule_text,
            condition=rule.condition,
            threshold=rule.threshold,
            required_action=rule.required_action,
            rule_complexity=rule.rule_complexity.value if rule.rule_complexity else "complex",
            source_pdf=rule.source_pdf or "AML_Policy",
            category=rule.category,
        )
        session.merge(db_rule)
    session.commit()
    session.close()
    logger.info("Stored %d rules in SQL database", len(approved_rules))

    # Step 6b: Store COMPLEX rules → ChromaDB vector store
    logger.info("Step 6b: Storing complex rules in ChromaDB...")
    n_embedded = persist_complex_rules_to_chroma(CHROMA_DIR, complex_rules)
    logger.info("Embedded %d complex rules in ChromaDB", n_embedded)

    # Step 7: Save all rules as JSON for audit trail
    rules_payload = [
        {
            "rule_id": r.rule_id,
            "rule_text": r.rule_text,
            "condition": r.condition,
            "threshold": r.threshold,
            "required_action": r.required_action,
            "complexity": r.rule_complexity.value if r.rule_complexity else "unknown",
            "source_pdf": r.source_pdf,
            "category": r.category,
        }
        for r in approved_rules
    ]
    with EXTRACTED_JSON.open("w", encoding="utf-8") as f:
        json.dump(rules_payload, f, indent=2)
    logger.info("Saved rules JSON → %s", EXTRACTED_JSON)

    # Save policy hash so we can skip next time if unchanged
    _save_policy_hash()

    logger.info("✓ PHASE 1 COMPLETE — %d rules ingested", len(approved_rules))
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — RUNTIME VIOLATION DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def phase2_violation_detection(company_input: Path) -> dict | None:
    """
    Phase 2: Company Data → Input Detection → SQL or RAG Engine
             → Return evaluation result (SQLComplianceResult or RAGComplianceResult)
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: RUNTIME VIOLATION DETECTION")
    logger.info("=" * 70)

    from src.utils.shared import detect_input_type
    from src.runtime.sql_engine import evaluate_sql_compliance
    from src.runtime.rag_engine import evaluate_rag_compliance
    from src.runtime.exception_registry import get_exception_registry
    from src.models.evaluation_schemas import SQLComplianceResult, RAGComplianceResult, Violation

    if not company_input.exists():
        logger.error("Company input not found: %s", company_input)
        return None

    evaluation_id = f"EVAL-{uuid.uuid4().hex[:8].upper()}"
    logger.info("Evaluation ID: %s", evaluation_id)

    # Step 1: Detect input type
    input_type, description = detect_input_type(company_input)
    logger.info("Step 1: Input detection — %s (%s)", input_type, description)

    # Step 2: Route to appropriate engine (auto-route based on input type)
    if input_type == "structured":
        # SQL path for structured data (CSV, DB)
        logger.info("Step 2: Routing to SQL Compliance Engine (structured data)...")
        sql_output = EVALUATIONS_DIR / f"sql_results_{evaluation_id}.json"
        result = evaluate_sql_compliance(
            evaluation_id=evaluation_id,
            company_db_path=company_input,
            rules_db_path=SQL_DB_PATH,
            output_path=sql_output,
        )
        logger.info("SQL evaluation: compliant=%s, violations=%d, confidence=%.2f",
                     result.compliant, len(result.violations), result.confidence)
    else:
        # RAG path for unstructured data
        logger.info("Step 2: Routing to RAG Semantic Engine (unstructured data)...")
        rag_output = EVALUATIONS_DIR / f"rag_results_{evaluation_id}.json"
        result = evaluate_rag_compliance(
            evaluation_id=evaluation_id,
            company_data_path=company_input,
            chroma_dir=CHROMA_DIR,
            output_path=rag_output,
        )
        logger.info("RAG evaluation: compliant=%s, score=%.1f%%, confidence=%.2f",
                     result.compliant, result.compliance_score, result.confidence)

    # Step 3: Exception registry — filter approved exceptions
    logger.info("Step 3: Checking exception registry...")
    exception_registry = get_exception_registry(
        PROJECT_ROOT / "data" / "exception_registry.json"
    )
    if isinstance(result, SQLComplianceResult) and result.violations:
        violations_dict = [v.model_dump(mode="json", exclude_none=True) for v in result.violations]
        filtered = exception_registry.filter_violations_with_exceptions(violations_dict)
        removed = len(result.violations) - len(filtered)
        if removed > 0:
            logger.info("Exception registry filtered out %d approved exceptions", removed)
        result.violations = [Violation.model_validate(v) for v in filtered]

    logger.info("✓ PHASE 2 COMPLETE — Evaluation %s finished", evaluation_id)
    return {
        "evaluation_id": evaluation_id,
        "result": result,
        "input_type": input_type,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — DECISION ENGINE + REVIEW + REPORTING
# ═══════════════════════════════════════════════════════════════════════════════
def phase3_review_and_reporting(phase2_output: dict) -> bool:
    """
    Phase 3: Evaluation Result → Decision Engine → Auto-Log / Review Queue
             → Alerting → Dashboard → Daily Report → Feedback summary
    """
    logger.info("=" * 70)
    logger.info("PHASE 3: HUMAN REVIEW & REPORTING")
    logger.info("=" * 70)

    from src.decision.decision_engine import (
        create_violation_record,
        make_confidence_decision,
        route_violation,
    )
    from src.models.evaluation_schemas import (
        SQLComplianceResult, RAGComplianceResult, AnalysisType,
    )
    from src.models.review_schemas import DecisionType, ViolationRecord
    from src.reporting.report_generator import generate_daily_report
    from src.reporting.dashboard_component import ComplianceDashboard
    from src.reporting.alerting_service import get_alerting_service
    from src.review.feedback_loop import generate_feedback_summary
    from src.review.llm_feedback import generate_llm_feedback

    result = phase2_output["result"]
    evaluation_id = phase2_output["evaluation_id"]

    # ── Step 1: Confidence Decision Engine ────────────────────────────────────
    logger.info("Step 1: Running Confidence Decision Engine...")
    auto_logged = 0
    human_review = 0
    violation_records: list[ViolationRecord] = []

    if isinstance(result, SQLComplianceResult):
        # For SQL results: create a violation record per violation, or one summary
        if result.violations:
            for i, violation in enumerate(result.violations):
                record_id = f"{evaluation_id}-V{i+1:03d}"
                vr = ViolationRecord(
                    record_id=record_id,
                    analysis_type="SQL",
                    violation=True,
                    confidence=result.confidence,
                    explanation=violation.violation_description,
                    cited_rule_ids=[violation.rule_id],
                    tier="SQL",
                )
                violation_records.append(vr)
                decision = route_violation(vr, HIGH_CONF_PATH, REVIEW_QUEUE_PATH)
                if decision.decision == DecisionType.AUTO_LOG:
                    auto_logged += 1
                else:
                    human_review += 1
                logger.info("  %s → %s (confidence=%.2f)",
                            record_id, decision.decision.value, vr.confidence)
        else:
            # Compliant — create a single compliant record
            record_id = f"{evaluation_id}-CLEAN"
            vr = ViolationRecord(
                record_id=record_id,
                analysis_type="SQL",
                violation=False,
                confidence=result.confidence,
                explanation=result.explanation,
                cited_rule_ids=result.policy_reference,
                tier="SQL",
            )
            violation_records.append(vr)
            decision = make_confidence_decision(vr)
            auto_logged += 1
            logger.info("  %s → COMPLIANT (auto-logged, confidence=%.2f)",
                        record_id, vr.confidence)

    elif isinstance(result, RAGComplianceResult):
        record_id = f"{evaluation_id}-RAG"
        vr = ViolationRecord(
            record_id=record_id,
            analysis_type="RAG",
            violation=not result.compliant,
            confidence=result.confidence,
            explanation=result.explanation,
            cited_rule_ids=result.violated_policies,
            tier="RAG",
        )
        violation_records.append(vr)
        decision = route_violation(vr, HIGH_CONF_PATH, REVIEW_QUEUE_PATH)
        if decision.decision == DecisionType.AUTO_LOG:
            auto_logged += 1
        else:
            human_review += 1
        logger.info("  %s → %s (confidence=%.2f)",
                    record_id, decision.decision.value, vr.confidence)

    logger.info("Decision routing: %d auto-logged, %d → human review", auto_logged, human_review)

    # ── Step 2: Alerting Service ──────────────────────────────────────────────
    logger.info("Step 2: Running Alerting Service...")
    alerting = get_alerting_service(ALERT_LOG_PATH)
    alerts_sent = 0
    for vr in violation_records:
        if vr.violation:
            alerts_sent = alerting.check_and_alert_critical_violations(
                [vr], alert_threshold="high"
            )
    logger.info("Alerts sent: %d", alerts_sent)

    # ── Step 3: Generate Daily Report ─────────────────────────────────────────
    logger.info("Step 3: Generating daily compliance report...")
    today = datetime.now().strftime("%Y-%m-%d")
    report_path = generate_daily_report(
        violations_path=HIGH_CONF_PATH,
        review_path=REVIEW_QUEUE_PATH,
        reports_dir=REPORTS_DIR,
        date=today,
    )
    logger.info("Daily report saved → %s", report_path)

    # ── Step 4: Dashboard Metrics ─────────────────────────────────────────────
    logger.info("Step 4: Generating dashboard metrics...")
    dashboard = ComplianceDashboard(
        violations_path=HIGH_CONF_PATH,
        review_path=REVIEW_QUEUE_PATH,
        reports_dir=REPORTS_DIR,
    )
    metrics = dashboard.get_real_time_metrics()
    logger.info("Dashboard metrics:")
    logger.info("  Total violations:   %d", metrics["total_violations"])
    logger.info("  Pending reviews:    %d", metrics["pending_reviews"])
    logger.info("  SQL violations:     %d", metrics["sql_violations"])
    logger.info("  RAG violations:     %d", metrics["rag_violations"])
    logger.info("  Compliance rate:    %.1f%%", metrics["compliance_rate"])

    # Save dashboard snapshot
    dashboard_snapshot_path = REPORTS_DIR / f"dashboard_snapshot_{today}.json"
    with dashboard_snapshot_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Dashboard snapshot → %s", dashboard_snapshot_path)

    # ── Step 5: LLM Feedback Analysis ─────────────────────────────────────────
    logger.info("Step 5: Generating LLM feedback with confidence...")
    llm_feedback = generate_llm_feedback(
        evaluation_id=evaluation_id,
        result=result,
        violation_records=violation_records,
        rules_json_path=EXTRACTED_JSON,
        output_dir=REPORTS_DIR,
        log_path=LLM_FEEDBACK_LOG,
    )

    # Pretty console summary
    print()
    print("=" * 70)
    print("  LLM COMPLIANCE FEEDBACK REPORT")
    print("=" * 70)
    print(f"  Evaluation ID:      {evaluation_id}")
    ASSESS_ICONS = {"COMPLIANT": "[OK]", "LOW_RISK": "[!]", "MEDIUM_RISK": "[!!]", "HIGH_RISK": "[!!!]"}
    icon = ASSESS_ICONS.get(llm_feedback.overall_assessment, "[?]")
    print(f"  Overall Assessment: {icon} {llm_feedback.overall_assessment}")
    print(f"  LLM Confidence:     {llm_feedback.llm_confidence:.0%}")
    print(f"  Agrees with system: {'Yes' if llm_feedback.agreement_with_system else 'No'}")
    print("-" * 70)
    print("  Risk Narrative:")
    # Word-wrap the narrative for readability
    narrative = llm_feedback.risk_narrative
    while len(narrative) > 64:
        brk = narrative.rfind(" ", 0, 64)
        if brk == -1:
            brk = 64
        print(f"    {narrative[:brk]}")
        narrative = narrative[brk:].lstrip()
    if narrative:
        print(f"    {narrative}")
    if llm_feedback.key_findings:
        print("-" * 70)
        print("  Key Findings:")
        for i, finding in enumerate(llm_feedback.key_findings, 1):
            print(f"    {i}. {finding}")
    if llm_feedback.recommendations:
        print("-" * 70)
        print("  Recommendations:")
        for i, rec in enumerate(llm_feedback.recommendations, 1):
            print(f"    {i}. {rec}")
    if llm_feedback.rule_improvement_suggestions:
        print("-" * 70)
        print("  Rule Improvement Suggestions:")
        for i, sug in enumerate(llm_feedback.rule_improvement_suggestions, 1):
            print(f"    {i}. {sug}")
    print("=" * 70)
    print(f"  Feedback log: {LLM_FEEDBACK_LOG}")
    print("=" * 70)
    print()

    # ── Step 6: Feedback Loop Summary ─────────────────────────────────────────
    logger.info("Step 6: Feedback loop summary...")
    feedback_summary = generate_feedback_summary(FEEDBACK_PATH)
    logger.info("Feedback dataset: %d records (FP=%d, FN=%d, corrections=%d)",
                feedback_summary["total_feedback_records"],
                feedback_summary["false_positives"],
                feedback_summary["false_negatives"],
                feedback_summary["corrections"])

    logger.info("✓ PHASE 3 COMPLETE")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  COMPLIANCE MONITORING SYSTEM — Full 3-Phase Pipeline             ║")
    logger.info("╚" + "═" * 68 + "╝")

    ensure_dirs()

    # Determine CSV input — accept CLI argument or default
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1]).resolve()
    else:
        csv_path = PROJECT_ROOT / "csv" / "LI-Small_Trans.csv"
    if not csv_path.exists():
        logger.error("Transaction CSV not found: %s", csv_path)
        sys.exit(1)
    logger.info("Input CSV: %s", csv_path)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if not phase1_policy_ingestion():
        logger.error("Phase 1 failed — stopping pipeline.")
        sys.exit(1)

    print()

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    phase2_output = phase2_violation_detection(csv_path)
    if phase2_output is None:
        logger.error("Phase 2 failed — stopping pipeline.")
        sys.exit(1)

    print()

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    if not phase3_review_and_reporting(phase2_output):
        logger.error("Phase 3 failed — stopping pipeline.")
        sys.exit(1)

    print()

    # ── Final Summary ─────────────────────────────────────────────────────────
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  ALL 3 PHASES COMPLETED SUCCESSFULLY                              ║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info("Outputs:")
    logger.info("  Rules DB:           %s", SQL_DB_PATH)
    logger.info("  ChromaDB:           %s", CHROMA_DIR)
    logger.info("  Extracted rules:    %s", EXTRACTED_JSON)
    logger.info("  Evaluations:        %s", EVALUATIONS_DIR)
    logger.info("  Violations (auto):  %s", HIGH_CONF_PATH)
    logger.info("  Review queue:       %s", REVIEW_QUEUE_PATH)
    logger.info("  Reports:            %s", REPORTS_DIR)
    logger.info("  Alert log:          %s", ALERT_LOG_PATH)
    logger.info("  LLM Feedback:       %s", REPORTS_DIR)


if __name__ == "__main__":
    main()
