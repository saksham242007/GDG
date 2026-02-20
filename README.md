## Compliance Monitoring System

A production-grade **Policy-Driven Compliance Monitoring System** with two phases:

### Phase 1: Policy Ingestion (One-Time Setup)
Parses regulatory PDF policy documents, extracts compliance rules using LLM, and stores them in:
- **Simple / deterministic rules** → SQLite database (via **SQLAlchemy**)
- **Complex / nuanced rules** → Chroma vector database for **RAG**

### Phase 2: Runtime Compliance Evaluation
Evaluates company data (structured or unstructured) against embedded policy guidelines using a **Human-in-the-Loop** decision routing system.

### Tech Stack

- **Language**: Python 3.11+
- **LLM Orchestration**: LangChain
- **LLM**: OpenAI `gpt-4o-mini`
- **Embeddings**: OpenAI `text-embedding-3-large`
- **Vector DB**: ChromaDB (persistent)
- **PDF Parsing**: PyMuPDF
- **Data Processing**: Pandas
- **Validation**: Pydantic
- **ORM**: SQLAlchemy (SQLite, upgradeable to PostgreSQL/MySQL)

### Project Structure

**Phase 1 (Ingestion):**
- `src/ingestion/policy_ingestion_pipeline.py` — main entrypoint for Phase 1
- `src/services/pdf_loader.py` — PDF loading and text chunking
- `src/services/rule_extractor.py` — LLM rule extraction (LangChain + OpenAI)
- `src/services/rule_classifier.py` — simple vs complex classification logic
- `src/services/sql_repository.py` — SQLAlchemy models and persistence
- `src/services/vector_store.py` — ChromaDB vector store persistence
- `src/models/schemas.py` — Pydantic schemas for rules
- `src/utils/logging_config.py` — logging configuration

**Phase 2 (Runtime Evaluation):**
- `src/runtime/compliance_pipeline.py` — main entrypoint for Phase 2
- `src/runtime/human_gate.py` — Human-in-the-Loop decision interface
- `src/runtime/sql_engine.py` — SQL compliance evaluation engine
- `src/runtime/rag_engine.py` — RAG-based semantic compliance analysis
- `src/models/evaluation_schemas.py` — Pydantic schemas for evaluation results

### Installation

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key (and optionally base URL) via environment variables:

```bash
setx OPENAI_API_KEY "your_api_key_here"           # Windows (PowerShell)
# or for a single session:
$env:OPENAI_API_KEY = "your_api_key_here"
```

If you use a custom base URL, set:

```bash
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
```

### Running the Ingestion Pipeline

1. Place regulatory PDF policy documents into:

- `data/policies/`

2. Run the Phase 1 ingestion:

```bash
python -m src.ingestion.policy_ingestion_pipeline
```

### Phase 1 Outputs

- **All extracted rules (simple + complex)**:
  - `data/processed/extracted_rules.json`
- **Simple / deterministic rules (SQL)**:
  - SQLite DB at `database/compliance_rules.db`
  - Designed so the engine can later be upgraded to PostgreSQL/MySQL
- **Complex / nuanced rules (RAG)**:
  - ChromaDB persisted at `data/chroma_db/`

---

## Phase 2: Runtime Compliance Evaluation

### Overview

Phase 2 evaluates company data against policy guidelines using a **Human-in-the-Loop** decision routing system. The system supports two input types:

1. **Structured Data** → SQL Database (tables, transactions, records)
2. **Unstructured Data** → Documents, logs, text files, reports

### Human Decision Gate

Before analysis, a human operator must choose the evaluation path:

- **Option A: SQL Rule Engine** — Fast deterministic checks against structured rules (~75% of cases)
- **Option B: Policy RAG + LLM** — Semantic analysis using policy embeddings

All human decisions are logged to `outputs/audit/human_decisions.json` with timestamps and metadata.

### Running Phase 2

1. Ensure Phase 1 has been completed (policy rules ingested).

2. Place company data in `data/company_data/`:
   - SQL databases: `.db` or `.sqlite` files
   - Unstructured files: `.txt`, `.log`, `.md`, `.pdf`, `.docx`

3. Run the compliance evaluation pipeline:

**Option A: Standard Pipeline (Sequential)**
```bash
python -m src.runtime.compliance_pipeline data/company_data/your_data.db
# or
python -m src.runtime.compliance_pipeline data/company_data/your_report.txt
```

**Option B: LangGraph Pipeline (Recommended for Production)**
```bash
python -m src.runtime.compliance_graph data/company_data/your_data.db
# or
python -m src.runtime.compliance_graph data/company_data/your_report.txt
```

**LangGraph Benefits:**
- ✅ Explicit state management
- ✅ Built-in checkpointing support
- ✅ Visual workflow graph
- ✅ Better error recovery
- ✅ Ready for web UI integration
- ✅ Parallel execution capabilities

See `docs/LANGGRAPH_COMPARISON.md` for detailed comparison.

4. When prompted, select:
   - **A** for SQL Engine (structured data)
   - **B** for RAG Engine (unstructured/semantic data)

### Phase 2 Outputs

- **SQL Evaluation Results**:
  - `outputs/evaluations/sql_results_<EVAL_ID>.json`
- **RAG Evaluation Results**:
  - `outputs/evaluations/rag_results_<EVAL_ID>.json`
- **Human Decision Audit Log**:
  - `outputs/audit/human_decisions.json`
- **Non-Compliant Cases** (flagged for review):
  - `outputs/flags/non_compliant_cases.json`

### Compliance Decision Logic

- **RAG Analysis**: Compliance score ≥ 75% → COMPLIANT
- **SQL Analysis**: No violations → COMPLIANT
- All non-compliant cases are automatically flagged and routed to the Human Review Queue

### Architecture Notes

- **Policy RAG** (`data/chroma_db/`) is the single source of truth for policy guidelines
- The system uses persistent ChromaDB (not in-memory) for reliable policy retrieval
- Full audit logging ensures enterprise compliance traceability
- Modular design allows easy extension and maintenance

---

## Phase 3: Human Review, Reporting, and Scalable Orchestration

### Overview

Phase 3 adds enterprise-grade features for production compliance monitoring:

- **Confidence-based decision routing** (auto-log vs human review)
- **Human review interface** (CLI + FastAPI REST API)
- **Feedback loop** for model improvement
- **Daily reporting** with compliance metrics and trends
- **Performance optimizations** (incremental scanning, risk-based prioritization, batch processing)
- **Apache Airflow orchestration** for scheduled compliance scans
- **FastAPI backend** for web-based review interface

### Key Features

**Decision Engine:**
- Routes violations based on confidence scores
- High confidence (≥75%) → Auto-logged
- Low confidence (<75%) → Human review queue

**Human Review:**
- CLI interface: `python -m src.review.review_interface`
- FastAPI endpoints: `/review/pending`, `/review/decide`
- Stores feedback for model improvement

**Reporting:**
- Daily summary reports with compliance rates
- Severity breakdown and trend analysis
- JSON format for dashboard integration

**Performance:**
- Incremental scanning (only new/modified records)
- Risk-based prioritization (high/medium/low)
- Batch processing for large datasets
- Smart LLM sampling (reduce costs)

**Orchestration:**
- Apache Airflow DAG for scheduled jobs
- Daily SQL scans (high-risk)
- Weekly RAG analysis (medium-risk)
- Monthly full audit

### Quick Start

**Run Human Review Interface (CLI):**
```bash
python -m src.review.review_interface
```

**Start FastAPI Backend:**
```bash
python -m src.api.main
# or
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Generate Daily Report:**
```python
from src.reporting.report_generator import generate_daily_report
from pathlib import Path

generate_daily_report(
    violations_path=Path("outputs/violations/high_confidence.json"),
    review_path=Path("outputs/review/needs_review.json"),
    reports_dir=Path("outputs/reports")
)
```

**Deploy Airflow DAG:**
```bash
cp dags/compliance_dag.py $AIRFLOW_HOME/dags/
```

See `docs/PHASE3_GUIDE.md` for comprehensive documentation.

