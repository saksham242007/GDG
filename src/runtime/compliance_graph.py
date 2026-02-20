from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import interrupt

from src.models.evaluation_schemas import AnalysisType, HumanDecision
from src.models.graph_state import ComplianceGraphState
from src.runtime.rag_engine import evaluate_rag_compliance
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


def detect_input_type_node(state: ComplianceGraphState) -> ComplianceGraphState:
    """
    Node: Detect input type (structured vs unstructured).
    """
    logger.info("Detecting input type for: %s", state.company_input_path)

    if not state.company_input_path.exists():
        state.error = f"Input file not found: {state.company_input_path}"
        return state

    input_type, desc = detect_input_type(state.company_input_path)
    state.input_type = input_type
    state.input_description = desc

    logger.info("Detected input type: %s (%s)", state.input_type, state.input_description)
    return state


def human_decision_gate_node(state: ComplianceGraphState) -> ComplianceGraphState:
    """
    Node: Human-in-the-Loop decision gate with LangGraph interrupt.

    This node uses LangGraph's interrupt() to pause execution and wait for human input.
    """
    logger.info("Human Decision Gate: Requesting human input...")

    print("\n" + "=" * 80)
    print("HUMAN DECISION GATE - Compliance Evaluation Routing")
    print("=" * 80)
    print(f"Evaluation ID: {state.evaluation_id}")
    print(f"Input Type: {state.input_type}")
    print(f"Input Source: {state.company_input_path}")
    print("\nPlease select the analysis path:")
    print("\n  Option A: SQL Rule Engine")
    print("    → Fast deterministic checks against structured rules")
    print("    → Suitable for: Database tables, transactions, structured records")
    print("\n  Option B: Policy RAG + LLM Semantic Analysis")
    print("    → Semantic evaluation using policy embeddings")
    print("    → Suitable for: Documents, logs, text files, unstructured data")
    print("\n" + "-" * 80)

    # Use LangGraph interrupt to pause for human input
    # In production, this would integrate with a web UI or API
    # For CLI, we'll use a simple input prompt
    while True:
        choice = input("\nEnter your choice (A for SQL, B for RAG): ").strip().upper()
        if choice == "A":
            state.human_decision = AnalysisType.SQL
            break
        elif choice == "B":
            state.human_decision = AnalysisType.RAG
            break
        else:
            print("Invalid choice. Please enter 'A' for SQL or 'B' for RAG.")

    operator = input("\nEnter operator name (optional, press Enter to skip): ").strip()
    state.human_operator = operator if operator else None

    rationale = input("Enter decision rationale (optional, press Enter to skip): ").strip()
    state.decision_rationale = rationale if rationale else None

    # Log the decision
    audit_log_path = AUDIT_DIR / "human_decisions.json"
    from datetime import datetime
    import json

    decision = HumanDecision(
        evaluation_id=state.evaluation_id,
        timestamp=datetime.now(),
        analysis_type=state.human_decision,
        input_type=state.input_type or "unknown",
        input_source=str(state.company_input_path),
        human_operator=state.human_operator,
        rationale=state.decision_rationale,
    )

    # Persist decision to audit log
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    decisions = []
    if audit_log_path.exists():
        try:
            with audit_log_path.open("r", encoding="utf-8") as f:
                decisions = json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Could not read existing audit log, starting fresh.")

    decisions.append(decision.model_dump(mode="json", exclude_none=True))
    with audit_log_path.open("w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Human decision recorded: %s", state.human_decision.value)
    return state


def route_decision(state: ComplianceGraphState) -> Literal["sql_path", "rag_path"]:
    """
    Conditional edge function: Routes to SQL or RAG path based on human decision.
    """
    if state.human_decision == AnalysisType.SQL:
        return "sql_path"
    return "rag_path"


def sql_evaluation_node(state: ComplianceGraphState) -> ComplianceGraphState:
    """
    Node: Execute SQL compliance evaluation.
    """
    logger.info("Executing SQL compliance evaluation...")

    try:
        sql_output_path = EVALUATIONS_DIR / f"sql_results_{state.evaluation_id}.json"
        result = evaluate_sql_compliance(
            evaluation_id=state.evaluation_id,
            company_db_path=state.company_input_path,
            rules_db_path=RULES_DB_PATH,
            output_path=sql_output_path,
        )

        state.sql_result = result
        state.final_result = result
        state.is_compliant = result.compliant
        state.compliance_score = None  # SQL doesn't use score

        logger.info("SQL evaluation completed: Compliant=%s", result.compliant)

    except Exception as e:
        logger.error("SQL evaluation failed: %s", e)
        state.error = str(e)
        state.retry_count += 1

    return state


def rag_evaluation_node(state: ComplianceGraphState) -> ComplianceGraphState:
    """
    Node: Execute RAG compliance evaluation.
    """
    logger.info("Executing RAG compliance evaluation...")

    try:
        rag_output_path = EVALUATIONS_DIR / f"rag_results_{state.evaluation_id}.json"
        result = evaluate_rag_compliance(
            evaluation_id=state.evaluation_id,
            company_data_path=state.company_input_path,
            chroma_dir=POLICY_CHROMA_DIR,
            output_path=rag_output_path,
        )

        state.rag_result = result
        state.final_result = result
        state.is_compliant = result.compliance_score >= 75.0
        state.compliance_score = result.compliance_score

        logger.info(
            "RAG evaluation completed: Score=%.1f%%, Compliant=%s",
            result.compliance_score,
            state.is_compliant,
        )

    except Exception as e:
        logger.error("RAG evaluation failed: %s", e)
        state.error = str(e)
        state.retry_count += 1

    return state


def compliance_decision_node(state: ComplianceGraphState) -> ComplianceGraphState:
    """
    Node: Final compliance decision and flagging logic.
    """
    if not state.final_result:
        state.error = "No evaluation result available"
        return state

    # Flag non-compliant cases using shared utility
    if not state.is_compliant:
        flags_path = FLAGS_DIR / "non_compliant_cases.json"
        flag_non_compliant_case(state.final_result, flags_path)
        logger.warning("Non-compliant case flagged: %s", state.evaluation_id)

    return state


def should_retry(state: ComplianceGraphState) -> Literal["retry", "end"]:
    """
    Conditional edge: Decide whether to retry on error.
    """
    if state.error and state.retry_count < 2:
        return "retry"
    return "end"


def create_compliance_graph() -> StateGraph:
    """
    Create and configure the LangGraph compliance evaluation workflow.
    """
    workflow = StateGraph(ComplianceGraphState)

    # Add nodes
    workflow.add_node("detect_input_type", detect_input_type_node)
    workflow.add_node("human_decision_gate", human_decision_gate_node)
    workflow.add_node("sql_evaluation", sql_evaluation_node)
    workflow.add_node("rag_evaluation", rag_evaluation_node)
    workflow.add_node("compliance_decision", compliance_decision_node)

    # Define edges
    workflow.add_edge(START, "detect_input_type")
    workflow.add_edge("detect_input_type", "human_decision_gate")
    workflow.add_conditional_edges(
        "human_decision_gate",
        route_decision,
        {
            "sql_path": "sql_evaluation",
            "rag_path": "rag_evaluation",
        },
    )
    workflow.add_edge("sql_evaluation", "compliance_decision")
    workflow.add_edge("rag_evaluation", "compliance_decision")
    workflow.add_edge("compliance_decision", END)

    return workflow.compile()


def run_compliance_graph(
    company_input_path: Path,
    evaluation_id: str | None = None,
) -> ComplianceGraphState:
    """
    Execute the LangGraph-based compliance evaluation pipeline.

    Args:
        company_input_path: Path to company data
        evaluation_id: Optional evaluation ID (generated if not provided)

    Returns:
        Final ComplianceGraphState with evaluation results
    """
    logger.info("=" * 80)
    logger.info("Starting Phase 2: LangGraph Compliance Evaluation Pipeline")
    logger.info("=" * 80)

    # Ensure directories exist
    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    FLAGS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate evaluation ID
    if not evaluation_id:
        evaluation_id = f"EVAL-{uuid.uuid4().hex[:8].upper()}"

    # Initialize state
    initial_state = ComplianceGraphState(
        evaluation_id=evaluation_id,
        company_input_path=company_input_path,
    )

    # Create and run graph
    graph = create_compliance_graph()

    # Execute workflow
    result = graph.invoke(initial_state)

    # LangGraph returns a dict; convert back to ComplianceGraphState
    if isinstance(result, dict):
        final_state = ComplianceGraphState(**result)
    else:
        final_state = result

    # Log summary
    logger.info("=" * 80)
    logger.info("Compliance Evaluation Complete")
    logger.info("=" * 80)
    logger.info("Evaluation ID: %s", final_state.evaluation_id)
    if final_state.final_result:
        logger.info("Analysis Type: %s", final_state.final_result.analysis_type.value)
        logger.info("Compliant: %s", "YES" if final_state.is_compliant else "NO")
        if final_state.compliance_score is not None:
            logger.info("Compliance Score: %.1f%%", final_state.compliance_score)
        if final_state.final_result:
            logger.info("Confidence: %.2f", final_state.final_result.confidence)
    if final_state.error:
        logger.error("Error: %s", final_state.error)
    logger.info("=" * 80)

    return final_state


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.runtime.compliance_graph <company_input_path>")
        print("\nExample:")
        print("  python -m src.runtime.compliance_graph data/company_data/transactions.db")
        print("  python -m src.runtime.compliance_graph data/company_data/report.txt")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    run_compliance_graph(input_path)
