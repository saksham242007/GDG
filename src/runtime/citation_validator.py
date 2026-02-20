from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from src.utils.logging_config import get_logger


logger = get_logger(__name__)


def validate_citations(
    retrieved_policies: List[Document],
    cited_rule_ids: List[str],
) -> tuple[bool, List[str]]:
    """
    Validate that cited policy rule IDs actually exist in retrieved policies.

    This ensures traceability and prevents hallucination of policy citations.

    Args:
        retrieved_policies: List of policy documents retrieved from RAG
        cited_rule_ids: List of rule IDs cited in the evaluation

    Returns:
        Tuple of (all_valid, list_of_invalid_citations)
    """
    # Extract rule IDs from retrieved policies
    retrieved_rule_ids = set()
    for doc in retrieved_policies:
        rule_id = doc.metadata.get("rule_id")
        if rule_id:
            retrieved_rule_ids.add(rule_id)

    # Check cited rule IDs
    invalid_citations: List[str] = []
    for cited_id in cited_rule_ids:
        if cited_id not in retrieved_rule_ids:
            invalid_citations.append(cited_id)
            logger.warning("Invalid citation: %s not found in retrieved policies", cited_id)

    all_valid = len(invalid_citations) == 0

    if not all_valid:
        logger.warning(
            "Citation validation failed: %d/%d citations invalid",
            len(invalid_citations),
            len(cited_rule_ids),
        )
    else:
        logger.info("Citation validation passed: all %d citations valid", len(cited_rule_ids))

    return all_valid, invalid_citations


def enrich_with_citation_metadata(
    retrieved_policies: List[Document],
    cited_rule_ids: List[str],
) -> dict:
    """
    Enrich citation metadata with validation results and policy details.

    Returns:
        Dictionary with citation validation metadata
    """
    all_valid, invalid_citations = validate_citations(retrieved_policies, cited_rule_ids)

    citation_metadata = {
        "total_citations": len(cited_rule_ids),
        "valid_citations": len(cited_rule_ids) - len(invalid_citations),
        "invalid_citations": invalid_citations,
        "all_valid": all_valid,
        "retrieved_policy_count": len(retrieved_policies),
        "citation_coverage": (len(cited_rule_ids) - len(invalid_citations)) / len(cited_rule_ids) if cited_rule_ids else 0.0,
    }

    return citation_metadata
