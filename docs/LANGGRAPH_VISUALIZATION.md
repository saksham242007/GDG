# LangGraph Workflow Visualization

## Compliance Evaluation Graph Structure

```
                    START
                     |
                     v
            [detect_input_type]
                     |
                     v
          [human_decision_gate] ← Human Input Required
                     |
         +-----------+-----------+
         |                       |
         v                       v
   [sql_evaluation]      [rag_evaluation]
         |                       |
         +-----------+-----------+
                     |
                     v
         [compliance_decision]
                     |
                     v
                    END
```

## Node Descriptions

### 1. **detect_input_type**
- **Purpose**: Auto-detect if input is structured (SQL) or unstructured (text)
- **Input**: `company_input_path`
- **Output**: `input_type`, `input_description`
- **Error Handling**: Sets `error` if file not found

### 2. **human_decision_gate**
- **Purpose**: Pause workflow for human operator to choose SQL or RAG path
- **Input**: `input_type`, `evaluation_id`
- **Output**: `human_decision`, `human_operator`, `decision_rationale`
- **Special**: Uses interrupt() for human input (can integrate with API)

### 3. **sql_evaluation**
- **Purpose**: Execute fast deterministic compliance checks
- **Input**: `company_input_path`, `evaluation_id`
- **Output**: `sql_result`, `is_compliant`
- **Performance**: ~1-2 seconds for typical database

### 4. **rag_evaluation**
- **Purpose**: Semantic compliance analysis using policy RAG
- **Input**: `company_input_path`, `evaluation_id`
- **Output**: `rag_result`, `compliance_score`, `is_compliant`
- **Performance**: ~3-5 seconds (LLM + RAG retrieval)

### 5. **compliance_decision**
- **Purpose**: Final compliance determination and flagging
- **Input**: `final_result` (from SQL or RAG)
- **Output**: Flags non-compliant cases, saves results
- **Logic**: RAG score >= 75% = compliant, SQL no violations = compliant

## Conditional Routing

The graph uses conditional edges to route based on human decision:

```python
def route_decision(state):
    if state.human_decision == AnalysisType.SQL:
        return "sql_path"
    return "rag_path"
```

## State Flow

```
Initial State:
{
    evaluation_id: "EVAL-ABC123",
    company_input_path: Path("data/company_data/transactions.db"),
    input_type: None,
    human_decision: None,
    final_result: None
}

After detect_input_type:
{
    input_type: "structured",
    input_description: "SQL database"
}

After human_decision_gate:
{
    human_decision: AnalysisType.SQL,
    human_operator: "john.doe",
    decision_rationale: "Structured data, use SQL engine"
}

After sql_evaluation:
{
    sql_result: SQLComplianceResult(...),
    final_result: SQLComplianceResult(...),
    is_compliant: True
}

After compliance_decision:
{
    # Non-compliant cases flagged in outputs/flags/
}
```

## Advanced: Parallel Execution (Future Enhancement)

```
                    START
                     |
                     v
            [detect_input_type]
                     |
                     v
          [human_decision_gate]
                     |
         +-----------+-----------+
         |                       |
         v                       v
   [sql_evaluation]      [rag_evaluation]
         |                       |
         +-----------+-----------+
                     |
                     v
         [compare_results] ← Compare SQL vs RAG
                     |
                     v
         [compliance_decision]
                     |
                     v
                    END
```

This parallel approach:
- Runs SQL and RAG simultaneously
- Compares results for higher confidence
- Takes ~3-5 seconds total (vs 4-7 sequential)

## Checkpointing Example

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "eval-123"}}

# Run workflow
state = graph.invoke(initial_state, config)

# If interrupted, can resume:
state = graph.invoke(None, config)  # Resumes from checkpoint
```

## Error Recovery Flow

```
[evaluation_node] → Error occurred
         |
         v
[should_retry?] → retry_count < 2?
         |
    +----+----+
    |         |
   YES       NO
    |         |
    v         v
[retry]    [error_handler]
    |         |
    +----+----+
         |
         v
        END
```
