from __future__ import annotations

import json
import time
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from src.config import get_llm
from src.models.schemas import ExtractedRule, ExtractedRuleList
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


EXTRACTION_SYSTEM_PROMPT = """
You are a senior compliance analyst. Your task is to read regulatory or policy text
and extract explicit compliance rules in a STRICT JSON format.

Guidelines:
- Focus on rules that can be used for automated compliance monitoring.
- Do NOT invent rules that are not clearly implied.
- Prefer shorter, atomic rules over large, compound ones.
- If no rules are present, return an empty list.

For EACH rule, you MUST output an object with the following fields:
- rule_id: a short, unique, stable identifier (e.g., "RULE-001", "INV-30-DAYS-001").
- rule_text: the exact rule sentence(s) or paragraph excerpt from the document.
- condition: the condition that triggers the rule (e.g., "transaction_amount > 10000").
- threshold: a numeric threshold if applicable (e.g., 10000), otherwise null.
- required_action: what needs to be done when the condition is met.
- rule_complexity: "simple" if the rule is deterministic and easily checkable by SQL or code;
                   "complex" if it is vague, contextual, or requires semantic judgement.

Return ONLY a JSON object of the form:
{
  "rules": [
    {
      "rule_id": "...",
      "rule_text": "...",
      "condition": "...",
      "threshold": 123.45,
      "required_action": "...",
      "rule_complexity": "simple"
    }
  ]
}
"""


def get_extraction_llm():
    """
    Returns a configured LLM for rule extraction (via centralized config).
    """
    return get_llm()


def _call_llm_with_retries(
    llm: ChatOpenAI,
    prompt: str,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> str:
    """
    Call the LLM with basic exponential backoff retry logic.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLM call failed (attempt %d/%d): %s",
                attempt,
                max_retries,
                exc,
            )
            if attempt == max_retries:
                raise
            sleep_for = backoff_seconds * attempt
            time.sleep(sleep_for)
    raise RuntimeError("Unreachable: retries exhausted.")


def extract_rules_from_chunk(
    llm: ChatOpenAI,
    chunk: Document,
) -> List[ExtractedRule]:
    """
    Use the LLM to extract rules from a single document chunk.
    """
    user_prompt = (
        f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
        "Here is the policy text chunk:\n\n"
        f"{chunk.page_content}\n\n"
        "Return the JSON object now."
    )

    raw_output = _call_llm_with_retries(llm, user_prompt)

    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        logger.warning("LLM output was not valid JSON, attempting to repair.")
        # Simple repair heuristic: find first { and last } and parse substring
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start == -1 or end == -1 or start >= end:
            logger.error("Unable to repair LLM JSON output.")
            return []
        repaired = raw_output[start : end + 1]
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            logger.error("Repaired LLM JSON output is still invalid.")
            return []

    try:
        rule_list = ExtractedRuleList.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        logger.error("Validation of LLM output failed: %s", exc)
        return []

    # Attach metadata from the source chunk
    for rule in rule_list.rules:
        rule.source_pdf = chunk.metadata.get("source_pdf")
        # Use section, heading, or similar metadata as category if present
        section = (
            chunk.metadata.get("section")
            or chunk.metadata.get("heading")
            or chunk.metadata.get("title")
        )
        rule.category = section

    return rule_list.rules


def extract_rules_from_chunks(
    chunks: List[Document],
) -> List[ExtractedRule]:
    """
    Extract rules from a list of document chunks using a shared LLM instance.
    """
    if not chunks:
        return []

    llm = get_extraction_llm()
    all_rules: List[ExtractedRule] = []

    for idx, chunk in enumerate(chunks, start=1):
        logger.info("Extracting rules from chunk %d/%d.", idx, len(chunks))
        rules = extract_rules_from_chunk(llm, chunk)
        if not rules:
            continue
        all_rules.extend(rules)

    logger.info("Extracted %d rules from %d chunks.", len(all_rules), len(chunks))
    return all_rules

