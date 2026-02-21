# History Management & Recycle Bin

## Overview

The dashboard provides two history management actions and a per-scan PDF export feature. All deletions are **non-destructive** — files are moved to a `recyclebin/` folder, never permanently deleted.

---

## Actions

### Clear Logs

**Button:** Amber "Clear Logs" in History tab header  
**Endpoint:** `POST /api/history/clear`

Removes scan evaluation logs and feedback while preserving violations and reviews.

| What gets moved         | Affected? |
|-------------------------|-----------|
| Evaluation JSONs        | Yes       |
| Feedback log            | Yes       |
| Violations              | **No**    |
| Reviews                 | **No**    |
| Daily reports           | **No**    |
| Alert log               | **No**    |
| Reviewed cases          | **No**    |

**Use case:** Clean up scan history without losing important compliance findings.

---

### Delete All

**Button:** Red "Delete All" in History tab header  
**Endpoint:** `DELETE /api/history/delete-all`

Full reset — moves all output files to the recycle bin.

| What gets moved         | Affected? |
|-------------------------|-----------|
| Evaluation JSONs        | Yes       |
| Feedback log            | Yes       |
| Violations              | Yes       |
| Reviews                 | Yes       |
| Daily reports           | Yes       |
| Alert log               | Yes       |
| Reviewed cases          | Yes       |

**Use case:** Start fresh. Dashboard metrics reset to zero.

---

### PDF Export (per scan)

**Button:** Blue "PDF" button on each row in the History table  
**Endpoint:** `GET /api/history/{eval_id}/pdf`

Downloads a formatted PDF report for a specific scan's AI Insight containing:

- Evaluation ID, analysis type, status, score, confidence
- AI assessment and confidence
- Violated policies (rule IDs)
- Risk narrative
- Key findings
- Recommendations
- Generation timestamp

---

## Recycle Bin

**Location:** `recyclebin/` (project root)

Every clear/delete action creates a timestamped subfolder:

```
recyclebin/
├── archive_20260221_193500/          ← from "Clear Logs"
│   ├── sql_results_EVAL-XXXX.json
│   ├── rag_results_EVAL-YYYY.json
│   └── llm_feedback_log.json
├── full_archive_20260221_200000/     ← from "Delete All"
│   ├── sql_results_EVAL-XXXX.json
│   ├── high_confidence.json
│   ├── needs_review.json
│   ├── llm_feedback_log.json
│   ├── alert_log.json
│   └── daily_summary_2026-02-21.json
```

### Recovery

To restore files, copy them from the `recyclebin/` subfolder back to their original location:

| File                    | Original Location                        |
|-------------------------|------------------------------------------|
| `sql_results_*.json`    | `outputs/evaluations/`                   |
| `rag_results_*.json`    | `outputs/evaluations/`                   |
| `high_confidence.json`  | `outputs/violations/`                    |
| `needs_review.json`     | `outputs/review/`                        |
| `reviewed_cases.json`   | `outputs/review/`                        |
| `llm_feedback_log.json` | `outputs/reports/`                       |
| `alert_log.json`        | `outputs/audit/`                         |
| `daily_summary_*.json`  | `outputs/reports/`                       |
