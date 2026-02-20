from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import create_engine, text

from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class ScanState:
    """
    Tracks incremental scan state for efficient processing.
    """

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.last_scan_timestamp: Optional[datetime] = None
        self.last_full_scan: Optional[datetime] = None
        self.load()

    def load(self) -> None:
        """Load scan state from file."""
        if self.state_path.exists():
            try:
                with self.state_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "last_scan_timestamp" in data:
                        self.last_scan_timestamp = datetime.fromisoformat(data["last_scan_timestamp"])
                    if "last_full_scan" in data:
                        self.last_full_scan = datetime.fromisoformat(data["last_full_scan"])
            except Exception as e:
                logger.warning("Could not load scan state: %s", e)

    def save(self) -> None:
        """Save scan state to file."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_scan_timestamp": self.last_scan_timestamp.isoformat() if self.last_scan_timestamp else None,
            "last_full_scan": self.last_full_scan.isoformat() if self.last_full_scan else None,
        }
        with self.state_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def update_scan_timestamp(self) -> None:
        """Update scan timestamp to current time."""
        self.last_scan_timestamp = datetime.now()
        self.save()


class RiskLevel:
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def determine_risk_level(record: dict) -> str:
    """
    Determine risk level for a record based on heuristics.

    Args:
        record: Record dictionary with fields like amount, type, etc.

    Returns:
        Risk level: "high", "medium", or "low"
    """
    # High-risk indicators
    amount = record.get("amount", 0) or record.get("transaction_amount", 0) or 0
    if isinstance(amount, (int, float)) and amount > 10000:
        return RiskLevel.HIGH

    # Check for suspicious patterns
    record_type = str(record.get("type", "")).lower()
    if any(keyword in record_type for keyword in ["suspicious", "unusual", "flagged"]):
        return RiskLevel.HIGH

    # Medium-risk indicators
    if isinstance(amount, (int, float)) and amount > 5000:
        return RiskLevel.MEDIUM

    # Default to low risk
    return RiskLevel.LOW


def incremental_scan(
    db_path: Path | str,
    scan_state: ScanState,
    table_name: str = "transactions",
    created_at_column: str = "created_at",
    updated_at_column: str = "updated_at",
) -> pd.DataFrame:
    """
    Perform incremental scan: only process records modified since last scan.

    Args:
        db_path: Path to SQLite database or connection string
        scan_state: ScanState object tracking last scan time
        table_name: Name of table to scan
        created_at_column: Column name for creation timestamp
        updated_at_column: Column name for update timestamp

    Returns:
        DataFrame with new/modified records
    """
    if isinstance(db_path, Path):
        db_uri = f"sqlite:///{db_path}"
    else:
        db_uri = db_path if db_path.startswith("sqlite:///") else f"sqlite:///{db_path}"

    engine = create_engine(db_uri, echo=False)

    if scan_state.last_scan_timestamp:
        # Incremental scan: only get records modified since last scan
        query = f"""
        SELECT * FROM {table_name}
        WHERE {created_at_column} > :last_scan
           OR {updated_at_column} > :last_scan
        """
        params = {"last_scan": scan_state.last_scan_timestamp.isoformat()}
    else:
        # First scan: get all records
        query = f"SELECT * FROM {table_name}"
        params = {}

    try:
        df = pd.read_sql(query, engine, params=params)
        logger.info("Incremental scan: Found %d new/modified records", len(df))
        return df
    except Exception as e:
        logger.error("Incremental scan failed: %s", e)
        # Fallback to full scan
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        return df


def risk_based_prioritization(
    records: pd.DataFrame,
    risk_level: str,
) -> pd.DataFrame:
    """
    Filter records based on risk level and scan frequency.

    Args:
        records: DataFrame of records
        risk_level: Risk level filter ("high", "medium", "low")

    Returns:
        Filtered DataFrame
    """
    if records.empty:
        return records

    # Add risk level column if not present
    if "risk_level" not in records.columns:
        records["risk_level"] = records.apply(lambda row: determine_risk_level(row.to_dict()), axis=1)

    # Filter by risk level
    filtered = records[records["risk_level"] == risk_level].copy()

    logger.info("Risk-based filtering: %d records for risk level %s", len(filtered), risk_level)
    return filtered


def batch_process(
    records: List[dict],
    batch_size: int = 100000,
    processor_func: callable = None,
) -> List:
    """
    Process records in batches to manage memory.

    Args:
        records: List of records to process
        batch_size: Number of records per batch
        processor_func: Function to process each batch

    Returns:
        List of processed results
    """
    results = []
    total_batches = (len(records) + batch_size - 1) // batch_size

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        batch_num = (i // batch_size) + 1

        logger.info("Processing batch %d/%d (%d records)", batch_num, total_batches, len(batch))

        if processor_func:
            batch_results = processor_func(batch)
            results.extend(batch_results)
        else:
            results.extend(batch)

    return results


def smart_llm_sampling(
    ambiguous_cases: List[dict],
    risk_level: str,
    sample_rate: float = 0.1,
) -> List[dict]:
    """
    Smart sampling for LLM evaluation based on risk level.

    High-risk: Process all ambiguous cases
    Medium-risk: Sample 20%
    Low-risk: Sample 10%

    Args:
        ambiguous_cases: List of ambiguous cases
        risk_level: Risk level of the cases
        sample_rate: Override sample rate (optional)

    Returns:
        Sampled cases for LLM evaluation
    """
    if risk_level == RiskLevel.HIGH:
        # Process all high-risk ambiguous cases
        return ambiguous_cases
    elif risk_level == RiskLevel.MEDIUM:
        sample_rate = 0.2
    else:  # LOW
        sample_rate = 0.1

    import random

    sample_size = max(1, int(len(ambiguous_cases) * sample_rate))
    sampled = random.sample(ambiguous_cases, min(sample_size, len(ambiguous_cases)))

    logger.info(
        "Smart LLM sampling: Selected %d/%d cases (%.1f%%) for risk level %s",
        len(sampled),
        len(ambiguous_cases),
        sample_rate * 100,
        risk_level,
    )

    return sampled
