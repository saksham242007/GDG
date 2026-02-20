"""
LLM-Based Feedback Analyzer
============================
Uses the LLM to review Phase 2 evaluation results and provide:
  - Intelligent analysis of compliance outcomes
  - Confidence assessment of the automated evaluation
  - Recommendations for rule improvements
  - Risk narrative summarizing findings
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from src.config import get_llm
from src.models.evaluation_schemas import SQLComplianceResult, RAGComplianceResult
from src.models.review_schemas import ViolationRecord
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

class LLMFeedbackResult(BaseModel):
    """Structured output from the LLM feedback analysis."""

    evaluation_id: str = Field(..., description="Evaluation being reviewed")
    overall_assessment: str = Field(
        ..., description="HIGH_RISK, MEDIUM_RISK, LOW_RISK, or COMPLIANT"
    )
    llm_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="LLM's own confidence in this assessment (0-1)",
    )
    risk_narrative: str = Field(
        ..., description="Plain-language summary of the compliance status",
    )
    key_findings: List[str] = Field(
        default_factory=list, description="Bullet-point findings",
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Actionable improvement recommendations",
    )
    rule_improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions to improve rule definitions",
    )
    agreement_with_system: bool = Field(
        ..., description="Whether the LLM agrees with the automated result",
    )
    generated_at: datetime = Field(default_factory=datetime.now)


# ── Prompt ────────────────────────────────────────────────────────────────────

LLM_FEEDBACK_PROMPT = """\
You are a senior compliance officer reviewing the output of an automated
compliance monitoring system.  Analyze the evaluation results below and
provide your expert assessment.

=== EVALUATION RESULTS ===
{evaluation_json}

=== VIOLATION RECORDS ===
{violation_records_json}

=== COMPLIANCE RULES USED ===
{rules_json}

=== INSTRUCTIONS ===
Based on the above data, return a single JSON object (no markdown fences)
with these fields:

{{
  "overall_assessment": "COMPLIANT" | "LOW_RISK" | "MEDIUM_RISK" | "HIGH_RISK",
  "llm_confidence": <float 0-1>,
  "risk_narrative": "<2-4 sentences summarizing the compliance picture>",
  "key_findings": ["<finding 1>", "<finding 2>", ...],
  "recommendations": ["<recommendation 1>", ...],
  "rule_improvement_suggestions": ["<suggestion 1>", ...],
  "agreement_with_system": true | false
}}

Be precise.  Base your analysis ONLY on the data provided.
"""


# ── Core function ─────────────────────────────────────────────────────────────

def generate_llm_feedback(
    evaluation_id: str,
    result: SQLComplianceResult | RAGComplianceResult,
    violation_records: List[ViolationRecord],
    rules_json_path: Path,
    output_dir: Path,
    log_path: Optional[Path] = None,
) -> LLMFeedbackResult:
    """
    Ask the LLM to review the evaluation results and produce a feedback report.

    Args:
        evaluation_id: Unique evaluation ID
        result: Phase 2 evaluation result
        violation_records: Violation records from the decision engine
        rules_json_path: Path to extracted_rules.json (for context)
        output_dir: Directory to save the feedback JSON
        log_path: Optional path to a cumulative feedback log JSON.
                  If provided, each feedback entry is appended here
                  instead of creating a separate file per run.

    Returns:
        LLMFeedbackResult with the LLM's analysis
    """
    logger.info("Generating LLM feedback for evaluation %s ...", evaluation_id)

    # ── Build context payloads ────────────────────────────────────────────────
    eval_payload = result.model_dump(mode="json", exclude_none=True)
    # Serialise datetimes
    eval_json = json.dumps(eval_payload, indent=2, default=str)

    vr_payload = [
        vr.model_dump(mode="json", exclude_none=True) for vr in violation_records
    ]
    vr_json = json.dumps(vr_payload, indent=2, default=str)

    rules_text = "[]"
    if rules_json_path.exists():
        try:
            rules_text = rules_json_path.read_text(encoding="utf-8")
        except Exception:
            pass

    prompt = LLM_FEEDBACK_PROMPT.format(
        evaluation_json=eval_json,
        violation_records_json=vr_json,
        rules_json=rules_text,
    )

    # ── Call the LLM ──────────────────────────────────────────────────────────
    llm = get_llm()

    try:
        response = llm.invoke(prompt)
        raw = response.content
    except Exception as exc:
        logger.error("LLM feedback call failed: %s", exc)
        # Return a safe fallback
        return _fallback_feedback(evaluation_id, result)

    # ── Parse JSON from the response ──────────────────────────────────────────
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown fences or surrounding text
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM feedback JSON, using fallback.")
                return _fallback_feedback(evaluation_id, result)
        else:
            logger.warning("No JSON found in LLM feedback, using fallback.")
            return _fallback_feedback(evaluation_id, result)

    feedback = LLMFeedbackResult(
        evaluation_id=evaluation_id,
        overall_assessment=data.get("overall_assessment", "COMPLIANT"),
        llm_confidence=float(data.get("llm_confidence", 0.5)),
        risk_narrative=data.get("risk_narrative", ""),
        key_findings=data.get("key_findings", []),
        recommendations=data.get("recommendations", []),
        rule_improvement_suggestions=data.get("rule_improvement_suggestions", []),
        agreement_with_system=data.get("agreement_with_system", True),
    )

    # ── Persist ───────────────────────────────────────────────────────────────
    entry = feedback.model_dump(mode="json", exclude_none=True)

    if log_path is not None:
        # Append to a single cumulative log file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list = []
        if log_path.exists():
            try:
                with log_path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = []
        existing.append(entry)
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False, default=str)
        logger.info("LLM feedback appended to log → %s (%d entries)", log_path, len(existing))
    else:
        # Legacy: one file per evaluation
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"llm_feedback_{evaluation_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False, default=str)
        logger.info("LLM feedback saved → %s", out_path)

    return feedback


# ── Fallback ──────────────────────────────────────────────────────────────────

def _fallback_feedback(
    evaluation_id: str,
    result: SQLComplianceResult | RAGComplianceResult,
) -> LLMFeedbackResult:
    """Deterministic fallback when the LLM is unavailable."""
    compliant = result.compliant
    return LLMFeedbackResult(
        evaluation_id=evaluation_id,
        overall_assessment="COMPLIANT" if compliant else "MEDIUM_RISK",
        llm_confidence=0.5,
        risk_narrative=(
            "Automated evaluation completed. LLM feedback was unavailable; "
            "this is a deterministic fallback."
        ),
        key_findings=[result.explanation],
        recommendations=["Manual review recommended when LLM feedback is unavailable."],
        rule_improvement_suggestions=[],
        agreement_with_system=True,
    )
