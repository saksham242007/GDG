"""
Apache Airflow DAG for Compliance Monitoring System.

This DAG orchestrates:
- Daily SQL scans for high-risk records
- Weekly RAG analysis for medium-risk records
- Monthly full audit
- Human review synchronization
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Default arguments
default_args = {
    "owner": "compliance_team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    "compliance_monitoring",
    default_args=default_args,
    description="Compliance monitoring with SQL and RAG evaluation",
    schedule_interval="@daily",  # Run daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["compliance", "monitoring"],
)


def daily_sql_scan(**context):
    """
    Daily SQL scan for high-risk records.
    """
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from src.orchestration.performance_optimizer import ScanState, incremental_scan, risk_based_prioritization
    from src.runtime.sql_engine import evaluate_sql_compliance
    from src.decision.decision_engine import process_evaluation_results
    import uuid

    # Configuration
    company_db_path = project_root / "data" / "company_data" / "transactions.db"
    rules_db_path = project_root / "database" / "compliance_rules.db"
    scan_state_path = project_root / "data" / "scan_state.json"
    violations_path = project_root / "outputs" / "violations" / "high_confidence.json"
    review_path = project_root / "outputs" / "review" / "needs_review.json"

    # Incremental scan for high-risk records
    scan_state = ScanState(scan_state_path)
    df = incremental_scan(company_db_path, scan_state, table_name="transactions")
    high_risk_df = risk_based_prioritization(df, risk_level="high")

    if high_risk_df.empty:
        print("No high-risk records to process today.")
        return

    # Process each record
    results = []
    for idx, row in high_risk_df.iterrows():
        record_id = f"REC-{uuid.uuid4().hex[:8].upper()}"
        evaluation_id = f"EVAL-{uuid.uuid4().hex[:8].upper()}"

        # Create temporary SQLite DB for single record (simplified)
        # In production, would query directly
        result = evaluate_sql_compliance(
            evaluation_id=evaluation_id,
            company_db_path=company_db_path,
            rules_db_path=rules_db_path,
            output_path=project_root / "outputs" / "evaluations" / f"sql_{evaluation_id}.json",
        )

        results.append((record_id, result))

    # Route based on confidence
    process_evaluation_results(results, violations_path, review_path)

    # Update scan timestamp
    scan_state.update_scan_timestamp()

    print(f"Processed {len(results)} high-risk records.")


def weekly_rag_analysis(**context):
    """
    Weekly RAG analysis for medium-risk records.
    """
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from src.orchestration.performance_optimizer import ScanState, incremental_scan, risk_based_prioritization
    from src.runtime.rag_engine import evaluate_rag_compliance
    from src.decision.decision_engine import process_evaluation_results
    import uuid

    # Configuration
    company_data_dir = project_root / "data" / "company_data"
    chroma_dir = project_root / "data" / "chroma_db"
    violations_path = project_root / "outputs" / "violations" / "high_confidence.json"
    review_path = project_root / "outputs" / "review" / "needs_review.json"

    # Get medium-risk unstructured files
    # In production, would query database for medium-risk records
    unstructured_files = list(company_data_dir.glob("*.txt"))[:10]  # Limit for demo

    results = []
    for file_path in unstructured_files:
        record_id = f"REC-{uuid.uuid4().hex[:8].upper()}"
        evaluation_id = f"EVAL-{uuid.uuid4().hex[:8].upper()}"

        result = evaluate_rag_compliance(
            evaluation_id=evaluation_id,
            company_data_path=file_path,
            chroma_dir=chroma_dir,
            output_path=project_root / "outputs" / "evaluations" / f"rag_{evaluation_id}.json",
        )

        results.append((record_id, result))

    # Route based on confidence
    process_evaluation_results(results, violations_path, review_path)

    print(f"Processed {len(results)} medium-risk records with RAG.")


def monthly_full_audit(**context):
    """
    Monthly full audit of all records.
    """
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from src.orchestration.performance_optimizer import ScanState
    from src.runtime.sql_engine import evaluate_sql_compliance
    import uuid

    scan_state_path = project_root / "data" / "scan_state.json"
    scan_state = ScanState(scan_state_path)

    # Mark full scan timestamp
    scan_state.last_full_scan = datetime.now()
    scan_state.save()

    print("Monthly full audit completed. Scan state updated.")


def human_review_sync(**context):
    """
    Synchronize human review decisions and update feedback dataset.
    """
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from src.review.feedback_loop import load_feedback_dataset
    from src.reporting.report_generator import generate_daily_report

    # Generate daily report
    violations_path = project_root / "outputs" / "violations" / "high_confidence.json"
    review_path = project_root / "outputs" / "review" / "needs_review.json"
    reports_dir = project_root / "outputs" / "reports"

    generate_daily_report(violations_path, review_path, reports_dir)

    print("Human review sync completed. Daily report generated.")


# Define tasks
daily_sql_task = PythonOperator(
    task_id="daily_sql_scan",
    python_callable=daily_sql_scan,
    dag=dag,
)

weekly_rag_task = PythonOperator(
    task_id="weekly_rag_analysis",
    python_callable=weekly_rag_analysis,
    dag=dag,
)

monthly_audit_task = PythonOperator(
    task_id="monthly_full_audit",
    python_callable=monthly_full_audit,
    dag=dag,
)

review_sync_task = PythonOperator(
    task_id="human_review_sync",
    python_callable=human_review_sync,
    dag=dag,
)

# Task dependencies
daily_sql_task >> review_sync_task
weekly_rag_task >> review_sync_task
monthly_audit_task >> review_sync_task
