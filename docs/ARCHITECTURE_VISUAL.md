# Architecture Visual Guide

## Complete System Flow

```
                    ┌─────────────────────────────────────┐
                    │   COMPLIANCE MONITORING SYSTEM      │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
            ┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
            │  Phase 1:    │ │  Phase 2:   │ │  Phase 3:  │
            │  Ingestion   │ │  Runtime    │ │  Review   │
            └───────┬──────┘ └──────┬──────┘ └─────┬──────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                            ┌───────▼───────┐
                            │   Production  │
                            │   Ready      │
                            └───────────────┘
```

## Phase 1: Detailed Flow

```
PDF Files
    │
    ▼
[PDF Parser] ──▶ Text Chunks
    │
    ▼
[LLM Extractor] ──▶ Raw Rules
    │
    ▼
[Validation] ✨ ──▶ Valid Rules
    │
    ▼
[Human Verify] ✨ ──▶ Approved Rules
    │
    ▼
[Classifier] ──┬──▶ Simple ──▶ SQL DB
               └──▶ Complex ──▶ ChromaDB
```

## Phase 2: Detailed Flow

```
Company Data
    │
    ▼
[Scan Strategy] ✨
    │
    ├──▶ Incremental ──▶ SQL Direct Engine ──▶ SQL Aggregate Engine ✨
    │
    └──▶ Full Audit ──▶ LLM Semantic ──▶ RAG Retrieval ──▶ Citation Validation ✨
    │
    ▼
[Exception Registry] ✨
    │
    ▼
[Confidence Check]
    │
    ├──▶ ≥75% ──▶ Auto-Log
    └──▶ <75% ──▶ Review Queue
```

## Phase 3: Detailed Flow

```
Review Queue
    │
    ▼
[Priority Algorithm] ✨ ──▶ Prioritized Cases
    │
    ▼
[Batch Review] ✨ ──▶ Batched Cases
    │
    ▼
[Human Review] ──▶ Decisions
    │
    ├──▶ Feedback Loop ──▶ Model Improvement
    │
    └──▶ Violations DB ──┬──▶ Alerting Service ✨
                         ├──▶ Dashboard ✨
                         └──▶ Reports
```

## Component Interaction Matrix

| Component | Phase 1 | Phase 2 | Phase 3 | Dependencies |
|-----------|---------|---------|---------|--------------|
| PDF Parser | ✅ | ❌ | ❌ | PyMuPDF |
| LLM Extractor | ✅ | ❌ | ❌ | OpenAI |
| Validation Layer | ✅ | ❌ | ❌ | Pydantic |
| Human Verification | ✅ | ❌ | ❌ | CLI |
| SQL Engine | ❌ | ✅ | ❌ | SQLAlchemy |
| SQL Aggregate | ❌ | ✅ | ❌ | Pandas |
| RAG Engine | ❌ | ✅ | ❌ | ChromaDB |
| Citation Validator | ❌ | ✅ | ❌ | LangChain |
| Exception Registry | ❌ | ✅ | ❌ | JSON |
| Decision Engine | ❌ | ✅ | ✅ | - |
| Priority Algorithm | ❌ | ❌ | ✅ | - |
| Batch Review | ❌ | ❌ | ✅ | - |
| Alerting Service | ❌ | ❌ | ✅ | - |
| Dashboard | ❌ | ❌ | ✅ | FastAPI |

## Data Storage Map

```
┌─────────────────────────────────────────┐
│           DATA STORAGE                   │
├─────────────────────────────────────────┤
│                                         │
│  SQLite:                                │
│  • compliance_rules.db                  │
│    - Simple rules                       │
│    - Deterministic checks               │
│                                         │
│  ChromaDB:                              │
│  • data/chroma_db/                      │
│    - Complex rules (embeddings)        │
│    - Policy RAG                         │
│                                         │
│  JSON Files:                            │
│  • outputs/violations/                  │
│    - high_confidence.json               │
│  • outputs/review/                      │
│    - needs_review.json                  │
│    - reviewed_cases.json                │
│  • outputs/reports/                     │
│    - daily_summary_*.json               │
│  • data/feedback/                       │
│    - human_feedback_dataset.json        │
│  • data/scan_state.json                 │
│  • data/exception_registry.json         │
│                                         │
└─────────────────────────────────────────┘
```

## API Architecture

```
┌─────────────────────────────────────────┐
│         FastAPI Backend                 │
├─────────────────────────────────────────┤
│                                         │
│  /review/*                              │
│  ├── GET /pending                       │
│  ├── GET /{record_id}                   │
│  ├── POST /decide                       │
│  ├── GET /prioritized ✨                │
│  └── POST /batch ✨                     │
│                                         │
│  /dashboard/*                           │
│  ├── GET /summary                       │
│  └── GET /data ✨                       │
│                                         │
│  /reports/*                             │
│  └── GET /daily                         │
│                                         │
│  /violations/*                          │
│  └── GET /high-confidence               │
│                                         │
│  /alerts/* ✨                           │
│  └── GET /                              │
│                                         │
└─────────────────────────────────────────┘
```

## Orchestration (Airflow)

```
┌─────────────────────────────────────────┐
│      Apache Airflow DAG                 │
├─────────────────────────────────────────┤
│                                         │
│  Daily:                                 │
│  ├── SQL Scan (high-risk)              │
│  └── Review Sync                       │
│                                         │
│  Weekly:                                │
│  └── RAG Analysis (medium-risk)         │
│                                         │
│  Monthly:                               │
│  └── Full Audit                        │
│                                         │
└─────────────────────────────────────────┘
```
