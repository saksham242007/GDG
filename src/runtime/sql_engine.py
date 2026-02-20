from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.models.evaluation_schemas import (
    AnalysisType,
    SQLComplianceResult,
    Violation,
)
from src.services.sql_repository import ComplianceRule, get_rules_engine
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


def load_company_sql_data(db_path: str | Path) -> pd.DataFrame:
    """
    Load company data from SQL database.

    For now, assumes a simple table structure. In production, this would
    accept table name and query parameters.

    Args:
        db_path: Path to company SQLite database or connection string

    Returns:
        DataFrame containing company data
    """
    if isinstance(db_path, Path):
        db_path_str = str(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Company database file not found: {db_path}")
    else:
        db_path_str = db_path

    # Handle CSV files directly with pandas
    if db_path_str.lower().endswith(".csv"):
        df = pd.read_csv(db_path_str)
        logger.info("Loaded %d records from CSV file: %s", len(df), db_path_str)
        return df

    if not db_path_str.startswith("sqlite:///"):
        db_uri = f"sqlite:///{db_path_str}"
    else:
        db_uri = db_path_str

    engine = create_engine(db_uri, echo=False)
    # Example: load from a transactions table (adjust table name as needed)
    # In production, this would be configurable
    try:
        df = pd.read_sql_table("transactions", engine)
    except Exception:
        # Fallback: try a generic query
        logger.warning("Could not load 'transactions' table, attempting generic query.")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            if tables:
                table_name = tables[0]
                logger.info("Using table: %s", table_name)
                df = pd.read_sql_table(table_name, engine)
            else:
                raise ValueError("No tables found in company database")

    logger.info("Loaded %d records from company SQL database", len(df))
    return df


def evaluate_sql_compliance(
    evaluation_id: str,
    company_db_path: str | Path,
    rules_db_path: Path,
    output_path: Path,
) -> SQLComplianceResult:
    """
    Evaluate company structured data against SQL-stored compliance rules.

    Args:
        evaluation_id: Unique evaluation session ID
        company_db_path: Path to company SQL database
        rules_db_path: Path to compliance rules SQLite database
        output_path: Path to save JSON results

    Returns:
        SQLComplianceResult with compliance status and violations
    """
    logger.info("Starting SQL compliance evaluation for %s", evaluation_id)

    # Load compliance rules from Phase 1 database
    rules_engine = get_rules_engine()
    violations: List[Violation] = []
    policy_references: List[str] = []

    from sqlalchemy.orm import Session

    with Session(rules_engine) as session:
        rules = session.query(ComplianceRule).all()

    if not rules:
        logger.warning("No compliance rules found in database")
        return SQLComplianceResult(
            evaluation_id=evaluation_id,
            compliant=True,
            violations=[],
            explanation="No compliance rules found in database. Cannot perform evaluation.",
            confidence=0.0,
            policy_reference=[],
        )

    logger.info("Loaded %d compliance rules", len(rules))

    # Load company data
    try:
        company_df = load_company_sql_data(company_db_path)
    except Exception as e:
        logger.error("Failed to load company data: %s", e)
        return SQLComplianceResult(
            evaluation_id=evaluation_id,
            compliant=False,
            violations=[
                Violation(
                    rule_id="SYSTEM_ERROR",
                    violation_description=f"Failed to load company data: {e}",
                    severity="critical",
                )
            ],
            explanation=f"Error loading company database: {e}",
            confidence=0.0,
            policy_reference=[],
        )

    # Evaluate each rule against company data
    for rule in rules:
        policy_references.append(rule.rule_id)

        try:
            # Build SQL-like condition check
            # Example: if condition is "amount > 10000", check company_df["amount"] > 10000
            condition = rule.condition.lower()

            # Simple threshold-based checks
            if rule.threshold is not None:
                # Try to match condition pattern (e.g., "amount > threshold")
                if ">" in condition or "greater" in condition:
                    # Find numeric column and check threshold
                    numeric_cols = company_df.select_dtypes(include=["number"]).columns
                    for col in numeric_cols:
                        if col.lower() in condition or condition in col.lower():
                            violations_df = company_df[company_df[col] > rule.threshold]
                            if len(violations_df) > 0:
                                violations.append(
                                    Violation(
                                        rule_id=rule.rule_id,
                                        violation_description=(
                                            f"{len(violations_df)} records violate {rule.rule_text}. "
                                            f"Condition: {col} > {rule.threshold}"
                                        ),
                                        severity="high",
                                        affected_records=violations_df.index.astype(str).tolist()[:10],
                                    )
                                )
                                logger.info(
                                    "Rule %s violation: %d records exceed threshold",
                                    rule.rule_id,
                                    len(violations_df),
                                )

                elif "<" in condition or "less" in condition:
                    numeric_cols = company_df.select_dtypes(include=["number"]).columns
                    for col in numeric_cols:
                        if col.lower() in condition or condition in col.lower():
                            violations_df = company_df[company_df[col] < rule.threshold]
                            if len(violations_df) > 0:
                                violations.append(
                                    Violation(
                                        rule_id=rule.rule_id,
                                        violation_description=(
                                            f"{len(violations_df)} records violate {rule.rule_text}. "
                                            f"Condition: {col} < {rule.threshold}"
                                        ),
                                        severity="high",
                                        affected_records=violations_df.index.astype(str).tolist()[:10],
                                    )
                                )

            # Date-based checks (e.g., "invoice date must be within 30 days")
            if "date" in condition and ("within" in condition or "days" in condition):
                date_cols = company_df.select_dtypes(include=["datetime64"]).columns
                for col in date_cols:
                    if "date" in col.lower():
                        # Simple check: flag records older than 30 days (example)
                        from datetime import datetime, timedelta

                        cutoff = datetime.now() - timedelta(days=30)
                        violations_df = company_df[pd.to_datetime(company_df[col]) < cutoff]
                        if len(violations_df) > 0:
                            violations.append(
                                Violation(
                                    rule_id=rule.rule_id,
                                    violation_description=(
                                        f"{len(violations_df)} records violate date constraint: "
                                        f"{rule.rule_text}"
                                    ),
                                    severity="medium",
                                    affected_records=violations_df.index.astype(str).tolist()[:10],
                                )
                            )

        except Exception as e:
            logger.warning("Error evaluating rule %s: %s", rule.rule_id, e)
            continue

    # Determine overall compliance
    compliant = len(violations) == 0
    confidence = 0.95 if compliant else max(0.5, 1.0 - (len(violations) * 0.1))

    explanation = (
        f"Evaluated {len(rules)} compliance rules against {len(company_df)} company records. "
        f"Found {len(violations)} violation(s)."
    )

    result = SQLComplianceResult(
        evaluation_id=evaluation_id,
        analysis_type=AnalysisType.SQL,
        compliant=compliant,
        violations=violations,
        explanation=explanation,
        confidence=confidence,
        policy_reference=policy_references,
    )

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result.model_dump(mode="json", exclude_none=True), f, indent=2, default=str)

    logger.info(
        "SQL compliance evaluation completed: %s (violations: %d)",
        "COMPLIANT" if compliant else "NON-COMPLIANT",
        len(violations),
    )

    return result
