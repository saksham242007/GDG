from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class ScanStrategy(str, Enum):
    INCREMENTAL = "incremental"  # Daily fast checks
    FULL_AUDIT = "full_audit"  # Comprehensive monthly checks


def determine_scan_strategy(
    input_path: Path,
    last_scan_timestamp: str | None = None,
    force_full_audit: bool = False,
) -> ScanStrategy:
    """
    Determine scan strategy based on context.

    Args:
        input_path: Path to company data
        last_scan_timestamp: Last scan timestamp (for incremental)
        force_full_audit: Force full audit regardless of timestamp

    Returns:
        ScanStrategy (INCREMENTAL or FULL_AUDIT)
    """
    if force_full_audit:
        logger.info("Full audit forced by parameter")
        return ScanStrategy.FULL_AUDIT

    # Check if this is a scheduled full audit (e.g., monthly)
    # In production, this would check a schedule/calendar
    # For now, if no last_scan_timestamp, do full audit
    if last_scan_timestamp is None:
        logger.info("No previous scan timestamp - performing full audit")
        return ScanStrategy.FULL_AUDIT

    # Incremental scan for daily checks
    logger.info("Incremental scan selected (daily check)")
    return ScanStrategy.INCREMENTAL


def route_by_scan_strategy(
    strategy: ScanStrategy,
) -> Literal["sql_direct", "llm_semantic"]:
    """
    Route to appropriate engine based on scan strategy.

    According to architecture:
    - INCREMENTAL → SQL Direct Engine (fast, ~80% violations)
    - FULL_AUDIT → LLM Semantic Analyzer (comprehensive, ~20% violations)

    Returns:
        Engine type: "sql_direct" or "llm_semantic"
    """
    if strategy == ScanStrategy.INCREMENTAL:
        logger.info("Routing to SQL Direct Engine (incremental scan)")
        return "sql_direct"
    else:  # FULL_AUDIT
        logger.info("Routing to LLM Semantic Analyzer (full audit)")
        return "llm_semantic"
