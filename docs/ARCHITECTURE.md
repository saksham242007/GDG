# Complete Compliance Monitoring System Architecture

## System Overview

A production-grade **Policy-Driven Compliance Monitoring System** with three integrated phases:

- **Phase 1**: Policy Ingestion (One-time setup)
- **Phase 2**: Runtime Compliance Evaluation (Continuous monitoring)
- **Phase 3**: Human Review, Reporting & Orchestration (Enterprise features)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMPLIANCE MONITORING SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────┐         ┌──────────────────┐                      │
│  │   Phase 1:       │         │   Phase 2:      │                      │
│  │   Policy         │────────▶│   Runtime       │                      │
│  │   Ingestion      │         │   Evaluation    │                      │
│  └──────────────────┘         └──────────────────┘                      │
│         │                              │                                  │
│         │                              ▼                                  │
│         │                    ┌──────────────────┐                         │
│         │                    │   Phase 3:      │                         │
│         └───────────────────▶│   Review &      │                         │
│                              │   Reporting     │                         │
│                              └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Policy Ingestion Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: POLICY INGESTION                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  [PDF Policies]                                                           │
│      │                                                                     │
│      ▼                                                                     │
│  ┌─────────────────┐                                                      │
│  │  PDF Parser     │  (PyMuPDF)                                          │
│  │  - Extract text │                                                      │
│  │  - Preserve     │                                                      │
│  │    structure    │                                                      │
│  └─────────────────┘                                                      │
│      │                                                                     │
│      ▼                                                                     │
│  ┌─────────────────┐                                                      │
│  │  Text Chunker   │  (LangChain)                                        │
│  │  - Split docs   │                                                      │
│  │  - Overlap      │                                                      │
│  └─────────────────┘                                                      │
│      │                                                                     │
│      ▼                                                                     │
│  ┌─────────────────┐                                                      │
│  │ LLM Rule        │  (gpt-4o-mini)                                      │
│  │ Extractor       │  - Extract rules                                    │
│  │                 │  - JSON format                                      │
│  └─────────────────┘                                                      │
│      │                                                                     │
│      ▼                                                                     │
│  ┌─────────────────┐                                                      │
│  │ Validation      │  ✨ NEW                                             │
│  │ Layer           │  - Check completeness                               │
│  │                 │  - Consistency checks                              │
│  └─────────────────┘                                                      │
│      │                                                                     │
│      ▼                                                                     │
│  ┌─────────────────┐                                                      │
│  │ Human           │  ✨ NEW                                             │
│  │ Verification    │  - Approve/Reject                                   │
│  │                 │  - Mandatory step                                    │
│  └─────────────────┘                                                      │
│      │                                                                     │
│      ▼                                                                     │
│  ┌─────────────────┐                                                      │
│  │ Rule            │                                                      │
│  │ Classifier      │  - Simple vs Complex                                │
│  └─────────────────┘                                                      │
│      │                                                                     │
│      ├──────────────────┬──────────────────┐                              │
│      ▼                  ▼                  ▼                              │
│  ┌──────────┐    ┌──────────────┐   ┌──────────────┐                     │
│  │ Simple   │    │ Complex      │   │ All Rules    │                     │
│  │ Rules    │    │ Rules        │   │ (JSON)       │                     │
│  │          │    │              │   │              │                     │
│  │ SQL DB   │    │ ChromaDB     │   │ Processed/   │                     │
│  │          │    │ (RAG)        │   │ extracted_   │                     │
│  │          │    │              │   │ rules.json   │                     │
│  └──────────┘    └──────────────┘   └──────────────┘                     │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Components:**
- `src/services/pdf_loader.py` - PDF parsing
- `src/services/rule_extractor.py` - LLM extraction
- `src/services/validation_layer.py` - Validation ✨
- `src/ingestion/human_verification.py` - Human approval ✨
- `src/services/rule_classifier.py` - Classification
- `src/services/sql_repository.py` - SQL storage
- `src/services/vector_store.py` - Vector storage

**Outputs:**
- `database/compliance_rules.db` - SQL rules
- `data/chroma_db/` - Vector embeddings
- `data/processed/extracted_rules.json` - All rules

---

## Phase 2: Runtime Compliance Evaluation Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   PHASE 2: RUNTIME EVALUATION                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  [Company Data]                                                           │
│      │                                                                     │
│      ├─────────────── Structured (SQL DB)                                 │
│      └─────────────── Unstructured (Text Files)                           │
│      │                                                                     │
│      ▼                                                                     │
│  ┌─────────────────┐                                                      │
│  │ Input Type      │                                                      │
│  │ Detection       │                                                      │
│  └─────────────────┘                                                      │
│      │                                                                     │
│      ▼                                                                     │
│  ┌─────────────────┐                                                      │
│  │ Scan Strategy   │  ✨ NEW                                             │
│  │ Router          │  - Incremental (Daily)                              │
│  │                 │  - Full Audit (Monthly)                              │
│  └─────────────────┘                                                      │
│      │                                                                     │
│      ├──────────────────────┬──────────────────────┐                     │
│      ▼                      ▼                      ▼                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Incremental  │    │ Full Audit   │    │ Human        │               │
│  │ Scan         │    │ Scan         │    │ Decision     │               │
│  │              │    │              │    │ Gate         │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│      │                      │                      │                     │
│      ▼                      ▼                      │                     │
│  ┌──────────────┐    ┌──────────────┐             │                     │
│  │ SQL Direct   │    │ LLM Semantic │◀────────────┘                     │
│  │ Engine       │    │ Analyzer     │                                   │
│  │              │    │              │                                   │
│  │ Fast checks  │    │ + RAG        │                                   │
│  │ ~80% cases   │    │ Retrieval    │                                   │
│  └──────────────┘    │ ~20% cases   │                                   │
│      │               └──────────────┘                                   │
│      │                      │                                             │
│      ▼                      ▼                                             │
│  ┌──────────────┐    ┌──────────────┐                                   │
│  │ SQL          │    │ Citation      │  ✨ NEW                          │
│  │ Aggregate    │    │ Validation    │  - Validate citations            │
│  │ Engine       │    │                │  - Prevent hallucination        │
│  │              │    └──────────────┘                                   │
│  │ Complex SQL  │                                                       │
│  │ Aggregations │                                                       │
│  └──────────────┘                                                       │
│      │                                                                     │
│      ▼                                                                     │
│  ┌─────────────────┐                                                      │
│  │ Exception       │  ✨ NEW                                             │
│  │ Registry        │  - Check approved                                   │
│  │                 │    exceptions                                       │
│  │                 │  - Filter violations                               │
│  └─────────────────┘                                                      │
│      │                                                                     │
│      ▼                                                                     │
│  ┌─────────────────┐                                                      │
│  │ Confidence      │                                                      │
│  │ Decision        │  - ≥75%: Auto-log                                   │
│  │ Engine          │  - <75%: Human Review                               │
│  └─────────────────┘                                                      │
│      │                                                                     │
│      ├──────────────────┬──────────────────┐                             │
│      ▼                  ▼                  ▼                             │
│  ┌──────────┐    ┌──────────────┐   ┌──────────────┐                    │
│  │ Auto-    │    │ Human        │   │ Evaluation   │                    │
│  │ Logged   │    │ Review       │   │ Results      │                    │
│  │          │    │ Queue        │   │ (JSON)       │                    │
│  │ High     │    │              │   │              │                    │
│  │ Conf     │    │ Low Conf     │   │ outputs/     │                    │
│  │          │    │              │   │ evaluations/ │                    │
│  └──────────┘    └──────────────┘   └──────────────┘                    │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Components:**
- `src/runtime/scan_strategy.py` - Scan routing ✨
- `src/runtime/sql_engine.py` - SQL Direct Engine
- `src/runtime/sql_aggregate_engine.py` - SQL Aggregate ✨
- `src/runtime/rag_engine.py` - LLM Semantic Analyzer
- `src/runtime/citation_validator.py` - Citation validation ✨
- `src/runtime/exception_registry.py` - Exception handling ✨
- `src/decision/decision_engine.py` - Confidence routing

**Outputs:**
- `outputs/evaluations/sql_results_*.json`
- `outputs/evaluations/rag_results_*.json`
- `outputs/violations/high_confidence.json`
- `outputs/review/needs_review.json`

---

## Phase 3: Human Review & Reporting Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│              PHASE 3: REVIEW, REPORTING & ORCHESTRATION                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  [Violations & Review Cases]                                              │
│      │                                                                     │
│      ├──────────────────┬──────────────────┐                             │
│      ▼                  ▼                  ▼                             │
│  ┌──────────┐    ┌──────────────┐   ┌──────────────┐                    │
│  │ Auto-    │    │ Human        │   │ Exception    │                    │
│  │ Logged   │    │ Review       │   │ Registry     │                    │
│  │          │    │ Queue        │   │              │                    │
│  └──────────┘    └──────────────┘   └──────────────┘                    │
│      │                  │                                                 │
│      │                  ▼                                                 │
│      │          ┌─────────────────┐                                       │
│      │          │ Priority        │  ✨ NEW                              │
│      │          │ Algorithm       │  - Score by urgency                   │
│      │          │                 │  - Sort by priority                  │
│      │          └─────────────────┘                                       │
│      │                  │                                                 │
│      │                  ▼                                                 │
│      │          ┌─────────────────┐                                       │
│      │          │ Batch Review     │  ✨ NEW                              │
│      │          │                 │  - Group cases                       │
│      │          │                 │  - Efficient processing             │
│      │          └─────────────────┘                                       │
│      │                  │                                                 │
│      │                  ▼                                                 │
│      │          ┌─────────────────┐                                       │
│      │          │ Human Review     │                                       │
│      │          │ Interface        │                                       │
│      │          │ - CLI            │                                       │
│      │          │ - FastAPI        │                                       │
│      │          └─────────────────┘                                       │
│      │                  │                                                 │
│      │                  ▼                                                 │
│      │          ┌─────────────────┐                                       │
│      │          │ Feedback Loop    │                                       │
│      │          │                  │                                       │
│      │          │ - Store feedback │                                       │
│      │          │ - Model improve  │                                       │
│      │          └─────────────────┘                                       │
│      │                                                                     │
│      ▼                                                                     │
│  ┌─────────────────┐                                                       │
│  │ Violations     │                                                       │
│  │ Database       │                                                       │
│  │                │                                                       │
│  │ - All          │                                                       │
│  │   violations   │                                                       │
│  │ - Audit trail  │                                                       │
│  └─────────────────┘                                                       │
│      │                                                                     │
│      ├──────────────────┬──────────────────┐                             │
│      ▼                  ▼                  ▼                             │
│  ┌──────────┐    ┌──────────────┐   ┌──────────────┐                    │
│  │ Alerting │    │ Compliance   │   │ Reporting    │                    │
│  │ Service  │    │ Dashboard    │   │ Generator    │                    │
│  │          │    │              │   │              │                    │
│  │ ✨ NEW   │    │ ✨ NEW       │   │ - Daily      │                    │
│  │          │    │              │   │   summaries  │                    │
│  │ - Email  │    │ - Real-time  │   │ - Trends     │                    │
│  │ - Slack │    │   metrics    │   │ - Severity   │                    │
│  │ - Webhook│    │ - Trends     │   │   breakdown  │                    │
│  └──────────┘    └──────────────┘   └──────────────┘                    │
│                                                                           │
│  ┌──────────────────────────────────────────────────────┐                │
│  │         Apache Airflow Orchestration                 │                │
│  │                                                       │                │
│  │  • Daily SQL scans (high-risk)                       │                │
│  │  • Weekly RAG analysis (medium-risk)                 │                │
│  │  • Monthly full audit                                 │                │
│  │  • Human review sync                                  │                │
│  └──────────────────────────────────────────────────────┘                │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Components:**
- `src/review/priority_algorithm.py` - Priority scoring ✨
- `src/review/batch_review.py` - Batch processing ✨
- `src/review/review_interface.py` - Review UI
- `src/review/feedback_loop.py` - Feedback storage
- `src/reporting/alerting_service.py` - Alerting ✨
- `src/reporting/dashboard_component.py` - Dashboard ✨
- `src/reporting/report_generator.py` - Report generation
- `dags/compliance_dag.py` - Airflow orchestration

**Outputs:**
- `outputs/review/reviewed_cases.json`
- `data/feedback/human_feedback_dataset.json`
- `outputs/reports/daily_summary_*.json`
- `outputs/alerts/alert_log.json`

---

## Data Flow Diagram

```
┌─────────────┐
│ PDF Policies│
└──────┬──────┘
       │
       ▼
┌─────────────────┐      ┌──────────────┐      ┌──────────────┐
│ Phase 1:        │─────▶│ SQL Rules DB │      │ Vector DB    │
│ Ingestion       │      │              │      │ (RAG)        │
└─────────────────┘      └──────┬───────┘      └──────┬───────┘
                                 │                     │
                                 │                     │
┌─────────────┐                 │                     │
│ Company Data│                 │                     │
└──────┬──────┘                 │                     │
       │                         │                     │
       ▼                         │                     │
┌─────────────────┐             │                     │
│ Phase 2:         │─────────────┼─────────────────────┘
│ Evaluation       │             │
└──────┬──────────┘             │
       │                         │
       ▼                         │
┌─────────────────┐             │
│ Violations &    │             │
│ Review Queue    │             │
└──────┬──────────┘             │
       │                         │
       ▼                         │
┌─────────────────┐             │
│ Phase 3:        │─────────────┘
│ Review & Report  │
└─────────────────┘
```

---

## Technology Stack

### Core Technologies
- **Python 3.11+**
- **LangChain** - LLM orchestration
- **OpenAI** - gpt-4o-mini, text-embedding-3-large
- **ChromaDB** - Vector database
- **SQLAlchemy** - SQL ORM
- **Pydantic** - Data validation
- **FastAPI** - REST API
- **Apache Airflow** - Orchestration
- **Pandas** - Data processing

### Storage
- **SQLite** - Compliance rules (upgradeable to PostgreSQL)
- **ChromaDB** - Policy embeddings
- **JSON Files** - Results, logs, feedback

---

## Key Performance Indicators

- **80% SQL Detection** - Fast deterministic checks
- **20% LLM Analysis** - Semantic evaluation
- **<5% Human Review** - Low-confidence cases
- **Confidence Threshold**: 75%
- **Batch Size**: 100k records (configurable)
- **Scan Frequency**: Daily (high-risk), Weekly (medium), Monthly (low)

---

## API Endpoints (FastAPI)

```
GET  /                          - API info
GET  /review/pending            - Pending review cases
GET  /review/{record_id}        - Get specific case
POST /review/decide             - Submit review decision
GET  /review/prioritized        - Prioritized review queue ✨
POST /review/batch              - Batch review workflow ✨
GET  /dashboard/summary         - Daily summary
GET  /dashboard/data            - Complete dashboard data ✨
GET  /reports/daily             - Generate daily report
GET  /violations/high-confidence - High-confidence violations
GET  /alerts                    - Alert log ✨
GET  /health                     - Health check
```

---

## File Structure

```
GDG/
├── src/
│   ├── ingestion/          # Phase 1
│   │   ├── policy_ingestion_pipeline.py
│   │   └── human_verification.py ✨
│   ├── runtime/            # Phase 2
│   │   ├── compliance_pipeline.py
│   │   ├── enhanced_compliance_pipeline.py ✨
│   │   ├── scan_strategy.py ✨
│   │   ├── sql_engine.py
│   │   ├── sql_aggregate_engine.py ✨
│   │   ├── rag_engine.py
│   │   ├── citation_validator.py ✨
│   │   └── exception_registry.py ✨
│   ├── decision/           # Decision routing
│   │   └── decision_engine.py
│   ├── review/             # Phase 3
│   │   ├── review_interface.py
│   │   ├── priority_algorithm.py ✨
│   │   ├── batch_review.py ✨
│   │   └── feedback_loop.py
│   ├── reporting/          # Reporting
│   │   ├── report_generator.py
│   │   ├── alerting_service.py ✨
│   │   └── dashboard_component.py ✨
│   ├── services/           # Shared services
│   │   ├── pdf_loader.py
│   │   ├── rule_extractor.py
│   │   ├── rule_classifier.py
│   │   ├── validation_layer.py ✨
│   │   ├── sql_repository.py
│   │   └── vector_store.py
│   ├── models/             # Schemas
│   │   ├── schemas.py
│   │   ├── evaluation_schemas.py
│   │   ├── review_schemas.py
│   │   └── graph_state.py
│   ├── api/                # FastAPI backend
│   │   └── main.py
│   └── utils/
│       └── logging_config.py
├── dags/                   # Airflow DAGs
│   └── compliance_dag.py
├── data/
│   ├── policies/           # Input PDFs
│   ├── company_data/      # Company data
│   ├── chroma_db/         # Vector DB
│   ├── feedback/          # Feedback dataset
│   └── scan_state.json    # Scan state
├── database/
│   └── compliance_rules.db # SQL rules
├── outputs/
│   ├── violations/        # Auto-logged
│   ├── review/           # Review queue
│   ├── evaluations/      # Evaluation results
│   ├── reports/          # Daily reports
│   └── alerts/           # Alert log
└── docs/
    ├── ARCHITECTURE.md   # This file
    ├── ARCHITECTURE_ALIGNMENT.md
    └── PHASE3_GUIDE.md
```

---

## Next Steps

1. **Review Architecture** ✅ (This document)
2. **Identify Improvements** (Next step)
3. **Implement Improvements**
4. **Testing** (Unit, Integration, E2E)
5. **Documentation**
