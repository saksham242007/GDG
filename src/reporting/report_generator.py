from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from src.models.review_schemas import DailySummary
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


def load_violations_from_file(file_path: Path) -> List[dict]:
    """
    Load violations from a JSON file.
    """
    if not file_path.exists():
        return []

    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Could not load violations from %s: %s", file_path, e)
        return []


def calculate_severity_breakdown(violations: List[dict]) -> Dict[str, int]:
    """
    Calculate severity breakdown from violations.
    """
    breakdown = {"critical": 0, "high": 0, "medium": 0, "low": 0, "unknown": 0}

    for violation in violations:
        # Try to extract severity from violation data
        severity = violation.get("severity", "unknown")
        if isinstance(severity, str):
            severity = severity.lower()
            if severity in breakdown:
                breakdown[severity] += 1
            else:
                breakdown["unknown"] += 1
        else:
            breakdown["unknown"] += 1

    return breakdown


def load_historical_summaries(reports_dir: Path, days: int = 7) -> List[DailySummary]:
    """
    Load historical daily summaries for trend analysis.
    """
    summaries = []
    cutoff_date = datetime.now() - timedelta(days=days)

    for report_file in reports_dir.glob("daily_summary_*.json"):
        try:
            with report_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                summary = DailySummary.model_validate(data)
                summary_date = datetime.fromisoformat(summary.date)
                if summary_date >= cutoff_date:
                    summaries.append(summary)
        except Exception as e:
            logger.warning("Could not load historical summary %s: %s", report_file, e)

    return sorted(summaries, key=lambda s: s.date)


def generate_daily_summary(
    date: str,
    violations_path: Path,
    review_path: Path,
    reports_dir: Path,
) -> DailySummary:
    """
    Generate daily compliance summary report.

    Args:
        date: Date string (YYYY-MM-DD)
        violations_path: Path to high-confidence violations JSON
        review_path: Path to review queue JSON
        reports_dir: Directory containing historical reports

    Returns:
        DailySummary object
    """
    logger.info("Generating daily summary for %s", date)

    # Load violations
    violations = load_violations_from_file(violations_path)

    # Load review cases
    review_cases = load_violations_from_file(review_path)

    # Count violations by tier
    sql_violations = sum(1 for v in violations if v.get("tier") == "SQL" or v.get("analysis_type") == "SQL")
    rag_violations = sum(1 for v in violations if v.get("tier") == "RAG" or v.get("analysis_type") == "RAG")

    # Count review cases
    human_review_count = len(review_cases)

    # Calculate compliance rate
    total_records = len(violations) + len([c for c in review_cases if not c.get("violation", False)])
    compliant_records = len([v for v in violations if not v.get("violation", False)]) + len([c for c in review_cases if not c.get("violation", False)])
    compliance_rate = (compliant_records / total_records * 100) if total_records > 0 else 100.0

    # Severity breakdown
    severity_breakdown = calculate_severity_breakdown(violations)

    # Trend analysis (compare with previous days)
    historical_summaries = load_historical_summaries(reports_dir, days=7)
    trend_data = None
    if historical_summaries:
        prev_summary = historical_summaries[-1]
        prev_compliance_rate = prev_summary.compliance_rate
        prev_violations = prev_summary.auto_logged_count
        current_violations = len(violations)
        trend_data = {
            "previous_compliance_rate": prev_compliance_rate,
            "compliance_rate_change": compliance_rate - prev_compliance_rate,
            "violation_trend": (
                "increasing" if current_violations > prev_violations
                else "decreasing" if current_violations < prev_violations
                else "stable"
            ),
        }

    summary = DailySummary(
        date=date,
        total_records_processed=total_records,
        sql_violations_count=sql_violations,
        rag_violations_count=rag_violations,
        human_review_count=human_review_count,
        auto_logged_count=len(violations),
        compliance_rate=compliance_rate,
        severity_breakdown=severity_breakdown,
        trend_data=trend_data,
    )

    return summary


def save_daily_summary(summary: DailySummary, reports_dir: Path) -> Path:
    """
    Save daily summary to JSON file.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"daily_summary_{summary.date}.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary.model_dump(mode="json", exclude_none=True), f, indent=2, ensure_ascii=False, default=str)

    logger.info("Daily summary saved to %s", report_path)
    return report_path


def generate_daily_report(
    violations_path: Path,
    review_path: Path,
    reports_dir: Path,
    date: Optional[str] = None,
) -> Path:
    """
    Generate and save daily compliance report.

    Args:
        violations_path: Path to violations JSON
        review_path: Path to review queue JSON
        reports_dir: Directory to save reports
        date: Optional date string (defaults to today)

    Returns:
        Path to saved report file
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    summary = generate_daily_summary(date, violations_path, review_path, reports_dir)
    report_path = save_daily_summary(summary, reports_dir)

    return report_path
