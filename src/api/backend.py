"""
Compliance Monitoring System — Production Backend API
=====================================================
Full REST API serving the compliance pipeline + frontend dashboard.

Endpoints:
  GET  /                         → Serves the dashboard frontend
  GET  /api/health               → Health check
  GET  /api/dashboard            → Real-time dashboard metrics
  GET  /api/evaluations          → All evaluation results
  GET  /api/evaluations/{id}     → Single evaluation detail
  GET  /api/violations           → Auto-logged violations
  GET  /api/reviews              → Human review queue
  GET  /api/feedback             → LLM feedback log
  GET  /api/rules                → Extracted compliance rules
  GET  /api/reports/daily        → Daily compliance report
  GET  /api/history              → Full scan history timeline
  POST /api/scan                 → Upload & scan a file (CSV or PDF)
  POST /api/review/decide        → Submit human review decision
"""

import json
import os
import shutil
import sys
import uuid
import textwrap
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import get_logger

logger = get_logger("api")

# ── Directory constants ──────────────────────────────────────────────────────
POLICIES_DIR      = PROJECT_ROOT / "data" / "policies"
COMPANY_DATA_DIR  = PROJECT_ROOT / "data" / "company_data"
PROCESSED_DIR     = PROJECT_ROOT / "data" / "processed"
CHROMA_DIR        = PROJECT_ROOT / "data" / "chroma_db"
DB_PATH           = PROJECT_ROOT / "database" / "compliance_rules.db"
OUTPUTS_DIR       = PROJECT_ROOT / "outputs"
EVALUATIONS_DIR   = OUTPUTS_DIR / "evaluations"
VIOLATIONS_PATH   = OUTPUTS_DIR / "violations" / "high_confidence.json"
REVIEW_PATH       = OUTPUTS_DIR / "review" / "needs_review.json"
REPORTS_DIR       = OUTPUTS_DIR / "reports"
FEEDBACK_LOG      = REPORTS_DIR / "llm_feedback_log.json"
ALERT_LOG         = OUTPUTS_DIR / "audit" / "alert_log.json"
RULES_JSON        = PROCESSED_DIR / "extracted_rules.json"
FRONTEND_DIR      = PROJECT_ROOT / "frontend"
UPLOAD_DIR        = PROJECT_ROOT / "uploads"
RECYCLE_BIN       = PROJECT_ROOT / "recyclebin"

# Ensure dirs
for d in [EVALUATIONS_DIR, VIOLATIONS_PATH.parent, REVIEW_PATH.parent,
          REPORTS_DIR, ALERT_LOG.parent, UPLOAD_DIR, COMPANY_DATA_DIR, RECYCLE_BIN]:
    d.mkdir(parents=True, exist_ok=True)


# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Compliance Monitoring System",
    description="AI-powered compliance monitoring with LLM feedback",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ────────────────────────────────────────────────
class ScanStatus(BaseModel):
    scan_id: str
    status: str  # "running", "completed", "failed"
    file_name: str
    started_at: str
    completed_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None

class ReviewDecisionRequest(BaseModel):
    record_id: str
    decision: str  # "confirmed_violation", "false_positive", "needs_more_info"
    comment: Optional[str] = None
    reviewer: str = "dashboard_user"


# ── In-memory scan tracker ───────────────────────────────────────────────────
_active_scans: dict[str, ScanStatus] = {}


# ── Helper: safe JSON loader ────────────────────────────────────────────────
def _load_json(path: Path, default=None):
    if default is None:
        default = []
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _load_all_evaluations() -> list:
    """Load all evaluation JSON files."""
    evals = []
    if EVALUATIONS_DIR.exists():
        for f in sorted(EVALUATIONS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                evals.append(json.load(open(f, encoding="utf-8")))
            except Exception:
                pass
    return evals


# ══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "components": {
            "database": DB_PATH.exists(),
            "vector_store": CHROMA_DIR.exists(),
            "policies_loaded": len(list(POLICIES_DIR.glob("*.pdf"))) if POLICIES_DIR.exists() else 0,
        }
    }


@app.get("/api/dashboard")
async def get_dashboard():
    """Main dashboard metrics — the single source of truth for the frontend."""
    violations = _load_json(VIOLATIONS_PATH)
    reviews = _load_json(REVIEW_PATH)
    feedback = _load_json(FEEDBACK_LOG)
    evaluations = _load_all_evaluations()

    # Calculate metrics
    total_scans = len(evaluations)
    compliant_count = sum(1 for e in evaluations if e.get("compliant", False))
    non_compliant_count = total_scans - compliant_count
    compliance_rate = (compliant_count / total_scans * 100) if total_scans > 0 else 0

    # Severity breakdown from ALL non-compliant evaluations
    severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    seen_ids = set()

    # 1. Count from violations file (auto-logged, high-confidence)
    for v in violations:
        rid = v.get("record_id", "")
        if rid:
            seen_ids.add(rid)
        conf = v.get("confidence", 0)
        if conf >= 0.95:
            severity["critical"] += 1
        elif conf >= 0.85:
            severity["high"] += 1
        elif conf >= 0.75:
            severity["medium"] += 1
        else:
            severity["low"] += 1

    # 2. Count from review queue (items pending human review)
    reviews_list = reviews if isinstance(reviews, list) else []
    for r in reviews_list:
        rid = r.get("record_id", "")
        if rid and rid not in seen_ids:
            seen_ids.add(rid)
            conf = r.get("confidence", 0)
            if conf >= 0.95:
                severity["critical"] += 1
            elif conf >= 0.85:
                severity["high"] += 1
            elif conf >= 0.75:
                severity["medium"] += 1
            else:
                severity["low"] += 1

    # 3. Count non-compliant evaluations not already counted via violations/reviews
    for e in evaluations:
        if e.get("compliant", True):
            continue
        eid = e.get("evaluation_id", "")
        # Check if this eval was already counted via violations or reviews
        already_counted = any(eid in sid for sid in seen_ids)
        if not already_counted:
            conf = e.get("confidence", 0)
            if conf >= 0.95:
                severity["critical"] += 1
            elif conf >= 0.85:
                severity["high"] += 1
            elif conf >= 0.75:
                severity["medium"] += 1
            else:
                severity["low"] += 1

    # Recent activity (last 10 evaluations)
    recent = []
    for e in evaluations[:10]:
        recent.append({
            "id": e.get("evaluation_id", ""),
            "type": e.get("analysis_type", ""),
            "compliant": e.get("compliant", False),
            "score": e.get("compliance_score", e.get("confidence", 0) * 100 if e.get("compliant") else 0),
            "confidence": e.get("confidence", 0),
            "timestamp": e.get("timestamp", ""),
            "violated_policies": e.get("violated_policies", e.get("violations", [])),
        })

    # LLM feedback summary — enrich with violated_policies from matching evaluation
    latest_feedback = feedback[-1] if feedback else None
    if latest_feedback:
        fb_eval_id = latest_feedback.get("evaluation_id", "")
        for e in evaluations:
            if e.get("evaluation_id") == fb_eval_id:
                latest_feedback["violated_policies"] = e.get(
                    "violated_policies", e.get("violations", [])
                )
                latest_feedback["compliance_score"] = e.get("compliance_score")
                latest_feedback["analysis_type"] = e.get("analysis_type", "")
                break

    return {
        "metrics": {
            "total_scans": total_scans,
            "compliant": compliant_count,
            "non_compliant": non_compliant_count,
            "compliance_rate": round(compliance_rate, 1),
            "pending_reviews": len(reviews),
            "auto_logged_violations": len(violations),
            "total_rules": len(_load_json(RULES_JSON)),
        },
        "severity": severity,
        "recent_activity": recent,
        "latest_feedback": latest_feedback,
        "active_scans": len([s for s in _active_scans.values() if s.status == "running"]),
    }


@app.get("/api/evaluations")
async def get_evaluations():
    """All evaluation results, newest first."""
    return _load_all_evaluations()


@app.get("/api/evaluations/{eval_id}")
async def get_evaluation(eval_id: str):
    """Get a specific evaluation by ID."""
    for e in _load_all_evaluations():
        if e.get("evaluation_id") == eval_id:
            # Attach matching feedback
            feedback = _load_json(FEEDBACK_LOG)
            fb = next((f for f in feedback if f.get("evaluation_id") == eval_id), None)
            e["llm_feedback"] = fb
            return e
    raise HTTPException(404, f"Evaluation {eval_id} not found")


@app.get("/api/violations")
async def get_violations():
    """All auto-logged violations."""
    return _load_json(VIOLATIONS_PATH)


@app.get("/api/reviews")
async def get_reviews():
    """Human review queue."""
    return _load_json(REVIEW_PATH)


@app.post("/api/review/decide")
async def submit_review(decision: ReviewDecisionRequest):
    """Submit a human review decision."""
    reviews = _load_json(REVIEW_PATH)
    case = next((r for r in reviews if r.get("record_id") == decision.record_id), None)
    if not case:
        raise HTTPException(404, f"Review case {decision.record_id} not found")

    # Remove from queue
    reviews = [r for r in reviews if r.get("record_id") != decision.record_id]
    with open(REVIEW_PATH, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2, default=str)

    # Log the decision
    case["human_decision"] = decision.decision
    case["review_comment"] = decision.comment
    case["reviewer"] = decision.reviewer
    case["reviewed_at"] = datetime.now().isoformat()

    reviewed_path = REVIEW_PATH.parent / "reviewed_cases.json"
    reviewed = _load_json(reviewed_path)
    reviewed.append(case)
    with open(reviewed_path, "w", encoding="utf-8") as f:
        json.dump(reviewed, f, indent=2, default=str)

    return {"status": "success", "remaining_reviews": len(reviews)}


@app.get("/api/feedback")
async def get_feedback():
    """LLM feedback log — all entries."""
    return _load_json(FEEDBACK_LOG)


@app.get("/api/rules")
async def get_rules():
    """Extracted compliance rules."""
    rules = _load_json(RULES_JSON)
    return rules


@app.get("/api/reports/daily")
async def get_daily_report(date: Optional[str] = None):
    """Daily compliance report."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    report_path = REPORTS_DIR / f"daily_summary_{date}.json"
    if report_path.exists():
        return _load_json(report_path, default={})
    return {"date": date, "message": "No report available for this date"}


@app.get("/api/history")
async def get_scan_history():
    """Full scan history from evaluations + feedback."""
    evaluations = _load_all_evaluations()
    feedback_list = _load_json(FEEDBACK_LOG)

    # Merge feedback into evaluations
    fb_map = {f["evaluation_id"]: f for f in feedback_list}
    history = []
    for e in evaluations:
        eid = e.get("evaluation_id", "")
        fb = fb_map.get(eid, {})
        history.append({
            "evaluation_id": eid,
            "analysis_type": e.get("analysis_type", ""),
            "compliant": e.get("compliant", False),
            "compliance_score": e.get("compliance_score", None),
            "confidence": e.get("confidence", 0),
            "violated_policies": e.get("violated_policies", e.get("violations", [])),
            "timestamp": e.get("timestamp", ""),
            "llm_assessment": fb.get("overall_assessment", ""),
            "llm_confidence": fb.get("llm_confidence", None),
            "risk_narrative": fb.get("risk_narrative", ""),
        })
    return history


# ── History Management: Clear, Delete-All (Recycle Bin), PDF Export ────────

@app.post("/api/history/clear")
async def clear_history():
    """Move all evaluation JSONs and feedback log to recyclebin, preserving violations/reviews."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = RECYCLE_BIN / f"archive_{ts}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    # Move evaluation files
    if EVALUATIONS_DIR.exists():
        for f in EVALUATIONS_DIR.glob("*.json"):
            shutil.move(str(f), str(archive_dir / f.name))
            moved += 1

    # Move feedback log
    if FEEDBACK_LOG.exists():
        shutil.move(str(FEEDBACK_LOG), str(archive_dir / FEEDBACK_LOG.name))
        moved += 1

    logger.info("Cleared history: moved %d files to %s", moved, archive_dir)
    return {"status": "success", "moved_files": moved, "archive": str(archive_dir)}


@app.delete("/api/history/delete-all")
async def delete_all_history():
    """Move ALL output files (evaluations, violations, reviews, reports, feedback) to recyclebin."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = RECYCLE_BIN / f"full_archive_{ts}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    # Move evaluations
    if EVALUATIONS_DIR.exists():
        for f in EVALUATIONS_DIR.glob("*.json"):
            shutil.move(str(f), str(archive_dir / f.name))
            moved += 1
    # Move violations
    if VIOLATIONS_PATH.exists():
        shutil.move(str(VIOLATIONS_PATH), str(archive_dir / VIOLATIONS_PATH.name))
        moved += 1
    # Move reviews
    if REVIEW_PATH.exists():
        shutil.move(str(REVIEW_PATH), str(archive_dir / REVIEW_PATH.name))
        moved += 1
    # Move feedback log
    if FEEDBACK_LOG.exists():
        shutil.move(str(FEEDBACK_LOG), str(archive_dir / FEEDBACK_LOG.name))
        moved += 1
    # Move alert log
    if ALERT_LOG.exists():
        shutil.move(str(ALERT_LOG), str(archive_dir / ALERT_LOG.name))
        moved += 1
    # Move daily reports
    for f in REPORTS_DIR.glob("*.json"):
        shutil.move(str(f), str(archive_dir / f.name))
        moved += 1
    # Move reviewed cases
    reviewed_path = REVIEW_PATH.parent / "reviewed_cases.json"
    if reviewed_path.exists():
        shutil.move(str(reviewed_path), str(archive_dir / reviewed_path.name))
        moved += 1

    logger.info("Deleted all history: moved %d files to %s", moved, archive_dir)
    return {"status": "success", "moved_files": moved, "archive": str(archive_dir)}


@app.get("/api/history/{eval_id}/pdf")
async def download_insight_pdf(eval_id: str):
    """Generate and return a PDF report for a specific evaluation's AI insight."""
    # Find evaluation
    evaluation = None
    for e in _load_all_evaluations():
        if e.get("evaluation_id") == eval_id:
            evaluation = e
            break
    if not evaluation:
        raise HTTPException(404, f"Evaluation {eval_id} not found")

    # Find matching feedback
    feedback_list = _load_json(FEEDBACK_LOG)
    fb = next((f for f in feedback_list if f.get("evaluation_id") == eval_id), {})

    # Build PDF in memory using only stdlib
    import io

    buf = io.BytesIO()
    _write_insight_pdf(buf, evaluation, fb)
    buf.seek(0)

    filename = f"AI_Insight_{eval_id}.pdf"
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


def _write_insight_pdf(buf, evaluation: dict, fb: dict):
    """Write a simple PDF (no external libs) to the buffer."""
    # Minimal PDF 1.4 generator
    objects = []
    def obj(content):
        objects.append(content)
        return len(objects)

    def pdf_str(s):
        """Escape special PDF string characters."""
        return s.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)').replace('\r', '').replace('\n', ' ')

    # Collect data
    eval_id = evaluation.get("evaluation_id", "")
    analysis_type = evaluation.get("analysis_type", "")
    compliant = evaluation.get("compliant", False)
    compliance_score = evaluation.get("compliance_score", "N/A")
    confidence = evaluation.get("confidence", 0)
    timestamp = evaluation.get("timestamp", "")
    violated = evaluation.get("violated_policies", evaluation.get("violations", []))
    assessment = fb.get("overall_assessment", "N/A")
    llm_conf = fb.get("llm_confidence", "N/A")
    narrative = fb.get("risk_narrative", "No AI narrative available.")
    findings = fb.get("key_findings", [])
    recommendations = fb.get("recommendations", [])

    # Build text lines
    lines = []
    lines.append("COMPLIANCE AI INSIGHT REPORT")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Evaluation ID:    {eval_id}")
    lines.append(f"Analysis Type:    {analysis_type}")
    lines.append(f"Status:           {'COMPLIANT' if compliant else 'NON-COMPLIANT'}")
    lines.append(f"Compliance Score: {compliance_score}")
    lines.append(f"Confidence:       {round(confidence * 100)}%")
    lines.append(f"Timestamp:        {timestamp}")
    lines.append(f"AI Assessment:    {assessment}")
    if llm_conf != "N/A":
        lines.append(f"AI Confidence:    {round(llm_conf * 100)}%")
    lines.append("")
    if violated:
        lines.append("VIOLATED POLICIES")
        lines.append("-" * 30)
        for v in violated:
            if isinstance(v, dict):
                lines.append(f"  - {v.get('rule_id', v.get('violation_description', str(v)))}")
            else:
                lines.append(f"  - {v}")
        lines.append("")
    lines.append("RISK NARRATIVE")
    lines.append("-" * 30)
    for part in narrative.split(" | "):
        wrapped = textwrap.wrap(part.strip(), width=80)
        lines.extend(wrapped)
        lines.append("")
    if findings:
        lines.append("KEY FINDINGS")
        lines.append("-" * 30)
        for f_item in findings:
            wrapped = textwrap.wrap(f"* {f_item}", width=78)
            lines.extend(wrapped)
        lines.append("")
    if recommendations:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 30)
        for r_item in recommendations:
            wrapped = textwrap.wrap(f"* {r_item}", width=78)
            lines.extend(wrapped)
        lines.append("")
    lines.append("=" * 50)
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("Compliance Shield - AI Compliance Monitoring System")

    # Render PDF pages (simple Courier text layout)
    font_size = 10
    leading = 13
    margin_top = 750
    margin_bottom = 50
    margin_left = 50
    usable_height = margin_top - margin_bottom
    lines_per_page = usable_height // leading

    # Split into pages
    pages_content = []
    for i in range(0, len(lines), lines_per_page):
        pages_content.append(lines[i:i + lines_per_page])
    if not pages_content:
        pages_content = [[""]]

    # Object 1: Catalog
    catalog_id = obj(None)
    # Object 2: Pages
    pages_id = obj(None)
    # Object 3: Font
    font_id = obj(None)

    # Create page objects
    page_ids = []
    stream_ids = []
    for page_lines in pages_content:
        # Build stream
        stream_parts = [f"BT /F1 {font_size} Tf"]
        y = margin_top
        for line in page_lines:
            safe = pdf_str(line)
            stream_parts.append(f"{margin_left} {y} Td ({safe}) Tj")
            y -= leading
            stream_parts.append(f"-{margin_left} 0 Td")
        # Reset positioning workaround: use absolute Td each time
        stream_text = [f"BT /F1 {font_size} Tf"]
        y = margin_top
        for line in page_lines:
            safe = pdf_str(line)
            stream_text.append(f"1 0 0 1 {margin_left} {y} Tm ({safe}) Tj")
            y -= leading
        stream_text.append("ET")
        stream_data = "\n".join(stream_text)

        s_id = obj(stream_data)
        stream_ids.append(s_id)
        p_id = obj(None)
        page_ids.append(p_id)

    # Now write actual PDF bytes
    offsets = {}
    buf.write(b"%PDF-1.4\n")

    def write_obj(obj_id, data):
        offsets[obj_id] = buf.tell()
        buf.write(f"{obj_id} 0 obj\n{data}\nendobj\n".encode())

    # Catalog
    write_obj(catalog_id, f"<< /Type /Catalog /Pages {pages_id} 0 R >>")
    # Pages
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    write_obj(pages_id, f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>")
    # Font
    write_obj(font_id, "<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    # Streams and Pages
    for i, (s_id, p_id) in enumerate(zip(stream_ids, page_ids)):
        stream_data = objects[s_id - 1]  # 0-indexed
        encoded = stream_data.encode("latin-1", errors="replace")
        write_obj(s_id, f"<< /Length {len(encoded)} >>\nstream\n" + stream_data + "\nendstream")
        write_obj(p_id, f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] /Contents {s_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>")

    # xref
    xref_pos = buf.tell()
    num_objs = len(offsets) + 1
    buf.write(b"xref\n")
    buf.write(f"0 {num_objs}\n".encode())
    buf.write(b"0000000000 65535 f \n")
    for obj_id in range(1, num_objs):
        buf.write(f"{offsets[obj_id]:010d} 00000 n \n".encode())

    buf.write(b"trailer\n")
    buf.write(f"<< /Size {num_objs} /Root {catalog_id} 0 R >>\n".encode())
    buf.write(b"startxref\n")
    buf.write(f"{xref_pos}\n".encode())
    buf.write(b"%%EOF\n")


@app.get("/api/scans/{scan_id}")
async def get_scan_status(scan_id: str):
    """Check status of a running scan."""
    if scan_id in _active_scans:
        return _active_scans[scan_id].model_dump()
    raise HTTPException(404, f"Scan {scan_id} not found")


# ── Policy PDF Management ────────────────────────────────────────────────────

POLICIES_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/policies")
async def list_policies():
    """List all uploaded policy PDFs."""
    policies = []
    if POLICIES_DIR.exists():
        for f in sorted(POLICIES_DIR.glob("*.pdf")):
            stat = f.stat()
            policies.append({
                "filename": f.name,
                "size": stat.st_size,
                "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    return policies


@app.post("/api/policies/upload")
async def upload_policies(files: List[UploadFile] = File(...)):
    """Upload one or more policy PDFs. Invalidates the policy hash so Phase 1 re-runs."""
    saved = []
    for upload in files:
        fname = upload.filename or "policy.pdf"
        if not fname.lower().endswith(".pdf"):
            raise HTTPException(400, f"Only PDF files allowed. Got: {fname}")

        dest = POLICIES_DIR / fname
        content = await upload.read()
        with open(dest, "wb") as f:
            f.write(content)
        saved.append(fname)

    # Invalidate policy hash so Phase 1 re-ingests on next scan
    policy_hash_file = PROCESSED_DIR / ".policy_hash"
    if policy_hash_file.exists():
        policy_hash_file.unlink()
        logger.info("Policy hash invalidated — Phase 1 will re-run on next scan")

    logger.info("Uploaded %d policy PDF(s): %s", len(saved), saved)
    return {
        "uploaded": saved,
        "count": len(saved),
        "total_policies": len(list(POLICIES_DIR.glob("*.pdf"))),
    }


@app.delete("/api/policies/{filename}")
async def delete_policy(filename: str):
    """Delete a specific policy PDF. Invalidates the policy hash."""
    target = POLICIES_DIR / filename
    if not target.exists():
        raise HTTPException(404, f"Policy not found: {filename}")

    # Safety: only allow deleting .pdf files within the policies directory
    if not target.suffix.lower() == ".pdf" or target.parent != POLICIES_DIR:
        raise HTTPException(400, "Invalid file")

    target.unlink()

    # Invalidate hash
    policy_hash_file = PROCESSED_DIR / ".policy_hash"
    if policy_hash_file.exists():
        policy_hash_file.unlink()

    remaining = len(list(POLICIES_DIR.glob("*.pdf")))
    logger.info("Deleted policy: %s (%d remaining)", filename, remaining)
    return {"deleted": filename, "remaining": remaining}


@app.delete("/api/policies")
async def delete_all_policies():
    """Delete ALL policy PDFs. Invalidates the policy hash."""
    deleted = []
    for f in POLICIES_DIR.glob("*.pdf"):
        deleted.append(f.name)
        f.unlink()

    policy_hash_file = PROCESSED_DIR / ".policy_hash"
    if policy_hash_file.exists():
        policy_hash_file.unlink()

    logger.info("Deleted all %d policies", len(deleted))
    return {"deleted": deleted, "count": len(deleted)}


# ── File Upload & Scan ───────────────────────────────────────────────────────

def _run_pipeline(scan_id: str, file_path: str):
    """Run the full pipeline in background."""
    try:
        _active_scans[scan_id].status = "running"

        # Import pipeline functions
        from run_full_pipeline import ensure_dirs, phase1_policy_ingestion, phase2_violation_detection, phase3_review_and_reporting

        # Ensure all required directories exist
        ensure_dirs()

        # Phase 1
        phase1_policy_ingestion()

        # Phase 2 — convert string path to Path object (required by phase2)
        phase2_output = phase2_violation_detection(Path(file_path))

        if phase2_output is None:
            raise RuntimeError(f"Phase 2 returned None — file not found or evaluation failed for {file_path}")

        # Phase 3
        phase3_output = phase3_review_and_reporting(phase2_output)

        # Extract result from phase2 output (it's a dict with 'result', 'evaluation_id', 'input_type')
        result_obj = phase2_output.get("result")
        eval_id = phase2_output.get("evaluation_id", "")
        input_type = phase2_output.get("input_type", "")

        # Extract fields from the Pydantic result model
        compliant = getattr(result_obj, "compliant", False)
        confidence = getattr(result_obj, "confidence", 0)
        compliance_score = getattr(result_obj, "compliance_score", None)
        violations = getattr(result_obj, "violations", [])
        violated_policies = [
            v.model_dump(mode="json", exclude_none=True) if hasattr(v, "model_dump") else str(v)
            for v in violations
        ] if violations else []

        _active_scans[scan_id].status = "completed"
        _active_scans[scan_id].completed_at = datetime.now().isoformat()
        _active_scans[scan_id].result = {
            "evaluation_id": eval_id,
            "compliant": compliant,
            "compliance_score": compliance_score,
            "confidence": confidence,
            "analysis_type": input_type,
            "violated_policies": violated_policies,
        }
        logger.info("Scan %s completed successfully", scan_id)

    except Exception as e:
        logger.error("Scan %s failed: %s", scan_id, e)
        _active_scans[scan_id].status = "failed"
        _active_scans[scan_id].completed_at = datetime.now().isoformat()
        _active_scans[scan_id].error = str(e)


@app.post("/api/scan")
async def upload_and_scan(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a file (CSV or PDF) and run the complete compliance pipeline.
    Returns immediately with a scan_id — poll /api/scans/{scan_id} for status.
    """
    # Guard: require at least one policy PDF before scanning
    policy_count = len(list(POLICIES_DIR.glob("*.pdf"))) if POLICIES_DIR.exists() else 0
    if policy_count == 0:
        raise HTTPException(
            400,
            "No policy PDFs uploaded. Please upload at least one policy PDF in the Policies tab before scanning."
        )

    # Validate file type
    filename = file.filename or "unknown"
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ("csv", "pdf"):
        raise HTTPException(400, f"Unsupported file type: .{ext}. Use CSV or PDF.")

    # Save uploaded file
    scan_id = f"SCAN-{uuid.uuid4().hex[:8].upper()}"
    save_path = UPLOAD_DIR / f"{scan_id}_{filename}"
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create scan tracker
    scan = ScanStatus(
        scan_id=scan_id,
        status="queued",
        file_name=filename,
        started_at=datetime.now().isoformat(),
    )
    _active_scans[scan_id] = scan

    # Run pipeline in background
    background_tasks.add_task(_run_pipeline, scan_id, str(save_path))

    return {
        "scan_id": scan_id,
        "status": "queued",
        "file_name": filename,
        "message": f"Scan queued. Poll /api/scans/{scan_id} for status.",
    }


# ── Serve Frontend ──────────────────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    """Serve the dashboard frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse({"message": "Frontend not found. Place index.html in /frontend/"})


if __name__ == "__main__":
    import uvicorn
    print("\n  🚀 Compliance Monitoring System")
    print("  ================================")
    print("  Dashboard:  http://localhost:8000")
    print("  API Docs:   http://localhost:8000/docs")
    print("  Health:     http://localhost:8000/api/health\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
