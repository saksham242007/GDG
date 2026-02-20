# Phase 3: Human Review, Reporting, and Scalable Orchestration

## Overview

Phase 3 implements enterprise-grade features for production compliance monitoring:
- **Confidence-based decision routing** (auto-log vs human review)
- **Human review interface** (CLI + FastAPI)
- **Feedback loop** for model improvement
- **Reporting & dashboard** data generation
- **Performance optimizations** (incremental scanning, risk-based prioritization, batch processing)
- **Apache Airflow orchestration** for scheduled jobs
- **FastAPI backend** for web-based review interface

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 3 Components                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Decision   │───▶│    Review    │───▶│  Feedback    │  │
│  │   Engine     │    │  Interface  │    │    Loop      │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │           │
│         └────────────────────┼────────────────────┘         │
│                              │                                │
│                              ▼                                │
│                    ┌──────────────┐                           │
│                    │  Reporting   │                           │
│                    │   Generator  │                           │
│                    └──────────────┘                           │
│                              │                                │
│                              ▼                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Apache Airflow Orchestration                  │   │
│  │  • Daily SQL scans (high-risk)                        │   │
│  │  • Weekly RAG analysis (medium-risk)                  │   │
│  │  • Monthly full audit                                 │   │
│  │  • Human review sync                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                                │
│                              ▼                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              FastAPI Backend                          │   │
│  │  • REST API for review interface                      │   │
│  │  • Dashboard data endpoints                           │   │
│  │  • Report generation                                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Decision Engine (`src/decision/decision_engine.py`)

**Confidence-Based Routing:**
- Confidence ≥ 0.75 → Auto-log to `outputs/violations/high_confidence.json`
- Confidence < 0.75 → Route to `outputs/review/needs_review.json`

**Usage:**
```python
from src.decision.decision_engine import process_evaluation_results

results = [(record_id, sql_result), ...]
decisions = process_evaluation_results(
    results,
    high_confidence_path=Path("outputs/violations/high_confidence.json"),
    review_path=Path("outputs/review/needs_review.json")
)
```

### 2. Human Review Interface

**CLI Interface:**
```bash
python -m src.review.review_interface
```

**FastAPI Endpoints:**
- `GET /review/pending` - Get pending review cases
- `GET /review/{record_id}` - Get specific case
- `POST /review/decide` - Submit review decision

### 3. Feedback Loop (`src/review/feedback_loop.py`)

Stores human feedback for model improvement:
- False positives (rejected violations)
- False negatives (marked compliant)
- Corrections

**Feedback Dataset:** `data/feedback/human_feedback_dataset.json`

### 4. Reporting (`src/reporting/report_generator.py`)

**Daily Summary Report:**
- Total records processed
- SQL/RAG violation counts
- Human review count
- Compliance rate %
- Severity breakdown
- Trend analysis

**Generate Report:**
```python
from src.reporting.report_generator import generate_daily_report

report_path = generate_daily_report(
    violations_path=Path("outputs/violations/high_confidence.json"),
    review_path=Path("outputs/review/needs_review.json"),
    reports_dir=Path("outputs/reports")
)
```

### 5. Performance Optimizations (`src/orchestration/performance_optimizer.py`)

**Incremental Scanning:**
- Only process records modified since last scan
- Tracks scan state in `data/scan_state.json`

**Risk-Based Prioritization:**
- High-risk: Process daily
- Medium-risk: Process weekly
- Low-risk: Process monthly

**Batch Processing:**
- Process records in chunks (default: 100k)
- Memory-efficient for large datasets

**Smart LLM Sampling:**
- High-risk: Process all ambiguous cases
- Medium-risk: Sample 20%
- Low-risk: Sample 10%

### 6. Apache Airflow Orchestration (`dags/compliance_dag.py`)

**DAG Tasks:**
1. `daily_sql_scan` - Daily high-risk SQL scans
2. `weekly_rag_analysis` - Weekly medium-risk RAG analysis
3. `monthly_full_audit` - Monthly full audit
4. `human_review_sync` - Sync review decisions and generate reports

**Schedule:**
- Daily: SQL scans + review sync
- Weekly: RAG analysis
- Monthly: Full audit

**Deploy:**
```bash
# Copy DAG to Airflow DAGs folder
cp dags/compliance_dag.py $AIRFLOW_HOME/dags/

# Airflow will automatically pick it up
```

### 7. FastAPI Backend (`src/api/main.py`)

**Start Server:**
```bash
python -m src.api.main
# or
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**API Endpoints:**
- `GET /` - API info
- `GET /review/pending` - Pending review cases
- `GET /review/{record_id}` - Get specific case
- `POST /review/decide` - Submit review decision
- `GET /dashboard/summary` - Dashboard summary
- `GET /reports/daily` - Generate daily report
- `GET /violations/high-confidence` - High-confidence violations
- `GET /health` - Health check

## Workflow

### 1. Evaluation → Decision Routing

```
Evaluation Result
    │
    ├─ Confidence ≥ 0.75 ──▶ Auto-log violation
    │
    └─ Confidence < 0.75 ──▶ Human review queue
```

### 2. Human Review Process

```
Pending Review Case
    │
    ├─ Display case details
    │
    ├─ Human decision:
    │   ├─ Approve violation
    │   ├─ Reject violation (false positive)
    │   └─ Mark as compliant
    │
    └─ Store feedback for model improvement
```

### 3. Reporting Cycle

```
Daily Processing
    │
    ├─ Collect violations
    ├─ Collect review cases
    ├─ Calculate metrics
    │
    └─ Generate daily summary report
```

## File Structure

```
outputs/
├── violations/
│   └── high_confidence.json      # Auto-logged violations
├── review/
│   ├── needs_review.json          # Pending review cases
│   └── reviewed_cases.json       # Completed reviews
└── reports/
    └── daily_summary_YYYY-MM-DD.json

data/
├── feedback/
│   └── human_feedback_dataset.json
└── scan_state.json                # Incremental scan state
```

## Configuration

### Confidence Threshold

Default: 0.75 (75%)

Modify in `src/decision/decision_engine.py`:
```python
CONFIDENCE_THRESHOLD = 0.75  # Adjust as needed
```

### Risk Level Thresholds

Modify in `src/orchestration/performance_optimizer.py`:
```python
# High-risk: amount > 10000
# Medium-risk: amount > 5000
# Low-risk: default
```

### Batch Size

Default: 100,000 records per batch

Modify in `src/orchestration/performance_optimizer.py`:
```python
batch_size: int = 100000  # Adjust based on memory
```

## Integration with Phase 1 & 2

Phase 3 integrates seamlessly:

1. **Phase 1 Outputs:**
   - Uses `database/compliance_rules.db` (SQL rules)
   - Uses `data/chroma_db/` (Policy RAG)

2. **Phase 2 Outputs:**
   - Processes evaluation results from `outputs/evaluations/`
   - Routes based on confidence scores

3. **Phase 3 Adds:**
   - Decision routing
   - Human review workflow
   - Feedback collection
   - Reporting
   - Orchestration

## Production Deployment

### 1. Airflow Setup

```bash
# Install Airflow
pip install apache-airflow

# Initialize Airflow
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Copy DAG
cp dags/compliance_dag.py $AIRFLOW_HOME/dags/

# Start Airflow
airflow webserver --port 8080 &
airflow scheduler
```

### 2. FastAPI Deployment

```bash
# Development
uvicorn src.api.main:app --reload

# Production (with Gunicorn)
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### 3. Environment Variables

```bash
export OPENAI_API_KEY="your_key"
export AIRFLOW_HOME="/path/to/airflow"
export DATABASE_URL="postgresql://user:pass@localhost/compliance"
```

## Monitoring & Observability

- **Logs:** Structured logging via `src/utils/logging_config.py`
- **Metrics:** Daily summary reports include compliance rates
- **Audit Trail:** All decisions logged with timestamps
- **Feedback Dataset:** Human corrections stored for model improvement

## Next Steps

1. **Model Fine-tuning:** Use feedback dataset to improve LLM prompts
2. **Dashboard UI:** Build frontend consuming FastAPI endpoints
3. **Alerting:** Add email/Slack notifications for critical violations
4. **ML Models:** Train models on feedback dataset for better accuracy
