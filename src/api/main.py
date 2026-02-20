"""
FastAPI backend for Compliance Monitoring System.

Provides REST API endpoints for:
- Human review interface
- Dashboard data
- Report generation
- Feedback submission
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.models.review_schemas import (
    HumanReviewCase,
    ReviewDecision,
    ReviewStatus,
)
from src.reporting.alerting_service import get_alerting_service
from src.reporting.dashboard_component import ComplianceDashboard
from src.reporting.report_generator import generate_daily_report, generate_daily_summary
from src.review.batch_review import create_review_batches, run_batch_review_workflow
from src.review.feedback_loop import store_human_feedback
from src.review.priority_algorithm import prioritize_review_queue
from src.review.review_interface import load_review_cases, save_review_decision
from src.utils.logging_config import get_logger


logger = get_logger(__name__)

app = FastAPI(
    title="Compliance Monitoring API",
    description="API for human review, reporting, and compliance monitoring",
    version="1.0.0",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_PATH = PROJECT_ROOT / "outputs" / "review" / "needs_review.json"
REVIEWED_PATH = PROJECT_ROOT / "outputs" / "review" / "reviewed_cases.json"
VIOLATIONS_PATH = PROJECT_ROOT / "outputs" / "violations" / "high_confidence.json"
FEEDBACK_PATH = PROJECT_ROOT / "data" / "feedback" / "human_feedback_dataset.json"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
ALERT_LOG_PATH = PROJECT_ROOT / "outputs" / "alerts" / "alert_log.json"

# Initialize services
dashboard = ComplianceDashboard(VIOLATIONS_PATH, REVIEW_PATH, REPORTS_DIR)
alerting_service = get_alerting_service(ALERT_LOG_PATH)


class ReviewDecisionRequest(BaseModel):
    record_id: str
    human_decision: ReviewStatus
    review_comment: Optional[str] = None
    reviewer_id: str
    confidence_override: Optional[float] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Compliance Monitoring API", "version": "1.0.0"}


@app.get("/review/pending", response_model=List[HumanReviewCase])
async def get_pending_reviews():
    """
    Get all pending review cases.
    """
    cases = load_review_cases(REVIEW_PATH)
    return cases


@app.get("/review/{record_id}", response_model=HumanReviewCase)
async def get_review_case(record_id: str):
    """
    Get a specific review case by record ID.
    """
    cases = load_review_cases(REVIEW_PATH)
    case = next((c for c in cases if c.record_id == record_id), None)

    if not case:
        raise HTTPException(status_code=404, detail=f"Review case {record_id} not found")

    return case


@app.post("/review/decide")
async def submit_review_decision(decision: ReviewDecisionRequest):
    """
    Submit a human review decision.
    """
    try:
        # Load the review case
        cases = load_review_cases(REVIEW_PATH)
        case = next((c for c in cases if c.record_id == decision.record_id), None)

        if not case:
            raise HTTPException(status_code=404, detail=f"Review case {decision.record_id} not found")

        # Create review decision
        review_decision = ReviewDecision(
            record_id=decision.record_id,
            human_decision=decision.human_decision,
            review_comment=decision.review_comment,
            reviewer_id=decision.reviewer_id,
            confidence_override=decision.confidence_override,
        )

        # Save decision
        save_review_decision(review_decision, REVIEW_PATH, REVIEWED_PATH)

        # Store feedback for model improvement
        from src.models.review_schemas import ViolationRecord

        violation_record = ViolationRecord(
            record_id=case.record_id,
            analysis_type=case.analysis_type,
            violation=case.violation,
            confidence=case.confidence,
            explanation=case.explanation,
            cited_rule_ids=case.cited_rule_ids,
        )

        store_human_feedback(violation_record, review_decision, FEEDBACK_PATH)

        return {"status": "success", "message": f"Decision saved for {decision.record_id}"}

    except Exception as e:
        logger.error("Error submitting review decision: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/summary")
async def get_dashboard_summary(date: Optional[str] = None):
    """
    Get dashboard summary data (daily compliance summary).
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    try:
        summary = generate_daily_summary(date, VIOLATIONS_PATH, REVIEW_PATH, REPORTS_DIR)
        return summary.model_dump(mode="json", exclude_none=True)
    except Exception as e:
        logger.error("Error generating dashboard summary: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/data")
async def get_dashboard_data():
    """
    Get complete dashboard data including real-time metrics and trends.
    """
    try:
        return dashboard.get_dashboard_data()
    except Exception as e:
        logger.error("Error getting dashboard data: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/review/prioritized")
async def get_prioritized_reviews():
    """
    Get review cases sorted by priority.
    """
    cases = load_review_cases(REVIEW_PATH)
    prioritized = prioritize_review_queue(cases)
    return [case.model_dump(mode="json", exclude_none=True) for case in prioritized]


@app.post("/review/batch")
async def process_batch_review(batch_size: int = 10, reviewer_id: str = "api_reviewer"):
    """
    Process batch review workflow.
    """
    try:
        run_batch_review_workflow(REVIEW_PATH, REVIEWED_PATH, batch_size, reviewer_id)
        return {"status": "success", "message": f"Batch review completed"}
    except Exception as e:
        logger.error("Error in batch review: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts")
async def get_alerts():
    """
    Get alert log.
    """
    return alerting_service.alert_log


@app.get("/reports/daily")
async def generate_report(date: Optional[str] = None):
    """
    Generate and return daily compliance report.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    try:
        report_path = generate_daily_report(VIOLATIONS_PATH, REVIEW_PATH, REPORTS_DIR, date)
        return {"status": "success", "report_path": str(report_path)}
    except Exception as e:
        logger.error("Error generating report: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/violations/high-confidence")
async def get_high_confidence_violations():
    """
    Get all high-confidence auto-logged violations.
    """
    import json

    if not VIOLATIONS_PATH.exists():
        return []

    try:
        with VIOLATIONS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error loading violations: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
