from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from src.models.review_schemas import DailySummary
from src.reporting.report_generator import generate_daily_summary, load_historical_summaries
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class ComplianceDashboard:
    """
    Dashboard component for real-time compliance monitoring.

    Provides:
    - Real-time metrics
    - Trend analysis
    - Compliance scores
    - Violation breakdowns
    """

    def __init__(
        self,
        violations_path: Path,
        review_path: Path,
        reports_dir: Path,
    ):
        self.violations_path = violations_path
        self.review_path = review_path
        self.reports_dir = reports_dir

    def get_real_time_metrics(self) -> Dict:
        """
        Get real-time compliance metrics.

        Returns:
            Dictionary with current metrics
        """
        import json

        # Load violations
        violations = []
        if self.violations_path.exists():
            try:
                with self.violations_path.open("r", encoding="utf-8") as f:
                    violations = json.load(f)
            except Exception:
                pass

        # Load review cases
        review_cases = []
        if self.review_path.exists():
            try:
                with self.review_path.open("r", encoding="utf-8") as f:
                    review_cases = json.load(f)
            except Exception:
                pass

        # Calculate metrics
        total_violations = len(violations)
        pending_reviews = len([c for c in review_cases if c.get("review_status") == "pending"])
        sql_violations = len([v for v in violations if v.get("tier") == "SQL" or v.get("analysis_type") == "SQL"])
        rag_violations = len([v for v in violations if v.get("tier") == "RAG" or v.get("analysis_type") == "RAG"])

        # Calculate compliance rate (simplified)
        total_records = total_violations + len(review_cases)
        compliant_records = len([v for v in violations if not v.get("violation", False)]) + len([c for c in review_cases if not c.get("violation", False)])
        compliance_rate = (compliant_records / total_records * 100) if total_records > 0 else 100.0

        return {
            "timestamp": datetime.now().isoformat(),
            "total_violations": total_violations,
            "pending_reviews": pending_reviews,
            "sql_violations": sql_violations,
            "rag_violations": rag_violations,
            "compliance_rate": round(compliance_rate, 2),
            "auto_logged": total_violations,
            "human_review_queue": pending_reviews,
        }

    def get_trend_analysis(self, days: int = 7) -> Dict:
        """
        Get trend analysis for the last N days.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with trend data
        """
        summaries = load_historical_summaries(self.reports_dir, days=days)

        if not summaries:
            return {
                "trend": "no_data",
                "days_analyzed": 0,
                "compliance_trend": "stable",
            }

        compliance_rates = [s.compliance_rate for s in summaries]
        avg_compliance = sum(compliance_rates) / len(compliance_rates)

        # Determine trend
        if len(compliance_rates) >= 2:
            recent = compliance_rates[-1]
            previous = compliance_rates[-2]
            if recent > previous + 2:
                trend = "improving"
            elif recent < previous - 2:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "days_analyzed": len(summaries),
            "average_compliance_rate": round(avg_compliance, 2),
            "current_compliance_rate": round(compliance_rates[-1], 2) if compliance_rates else None,
            "compliance_trend": trend,
            "violation_trend": "decreasing" if trend == "improving" else "increasing" if trend == "declining" else "stable",
        }

    def get_dashboard_data(self) -> Dict:
        """
        Get complete dashboard data for API/UI consumption.

        Returns:
            Complete dashboard data dictionary
        """
        real_time_metrics = self.get_real_time_metrics()
        trend_analysis = self.get_trend_analysis(days=7)

        # Generate today's summary if not exists
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            today_summary = generate_daily_summary(today, self.violations_path, self.review_path, self.reports_dir)
            daily_summary = today_summary.model_dump(mode="json", exclude_none=True)
        except Exception as e:
            logger.warning("Could not generate today's summary: %s", e)
            daily_summary = None

        return {
            "real_time_metrics": real_time_metrics,
            "trend_analysis": trend_analysis,
            "daily_summary": daily_summary,
            "last_updated": datetime.now().isoformat(),
        }
