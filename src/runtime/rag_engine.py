from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from src.config import get_llm, get_embeddings, CHROMA_DIR
from src.models.evaluation_schemas import (
    AnalysisType,
    RAGComplianceResult,
)
from src.runtime.citation_validator import validate_citations
from src.services.vector_store import build_chroma_vector_store
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


RAG_EVALUATION_PROMPT = """
You are a compliance analyst evaluating company data against regulatory policy guidelines.

You MUST use ONLY the retrieved policy context provided below. Do NOT hallucinate or invent policies.

Your task:
1. Compare the company data against the retrieved policy guidelines
2. Determine if the company data aligns with policy requirements
3. Calculate a compliance score (0-100) based on alignment
4. Identify any violated policies
5. Provide a clear explanation

Policy Context (from Policy RAG):
{policy_context}

Company Data to Evaluate:
{company_data}

Return a JSON object with:
{{
  "compliant": boolean,
  "compliance_score": float (0-100),
  "explanation": string,
  "violated_policies": list of strings,
  "confidence": float (0-1)
}}
"""


def load_unstructured_data(input_path: Path) -> str:
    """
    Load unstructured company data from file (text, log, PDF, document).

    Args:
        input_path: Path to unstructured data file

    Returns:
        Content as string
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix == ".pdf":
        # Use PyMuPDF (fitz) to extract text from PDF
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF parsing. Install with: pip install pymupdf"
            )
        doc = fitz.open(str(input_path))
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        doc.close()
        content = "\n\n".join(pages_text)
        logger.info(
            "Loaded %d characters from PDF file (%d pages)", len(content), len(pages_text)
        )
    else:
        # Plain-text files (.txt, .log, .md, .csv, etc.)
        with input_path.open("r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        logger.info("Loaded %d characters from unstructured data file", len(content))

    return content


def retrieve_relevant_policies(
    chroma_dir: Path,
    company_data_text: str,
    top_k: int = 5,
) -> List[Document]:
    """
    Retrieve top-k relevant policy chunks from ChromaDB using RAG.

    Args:
        chroma_dir: Path to persistent ChromaDB directory
        company_data_text: Company data text to use as query
        top_k: Number of policy chunks to retrieve

    Returns:
        List of relevant policy Documents
    """
    if not chroma_dir.exists():
        raise ValueError(f"ChromaDB directory does not exist: {chroma_dir}")

    vector_store = build_chroma_vector_store(chroma_dir)

    # Use company data as query to retrieve relevant policies
    docs = vector_store.similarity_search(company_data_text, k=top_k)

    logger.info("Retrieved %d relevant policy chunks from RAG", len(docs))
    return docs


def evaluate_rag_compliance(
    evaluation_id: str,
    company_data_path: Path,
    chroma_dir: Path,
    output_path: Path,
) -> RAGComplianceResult:
    """
    Evaluate company unstructured data against policy RAG using semantic analysis.

    Args:
        evaluation_id: Unique evaluation session ID
        company_data_path: Path to unstructured company data file
        chroma_dir: Path to persistent ChromaDB directory (from Phase 1)
        output_path: Path to save JSON results

    Returns:
        RAGComplianceResult with compliance score and analysis
    """
    logger.info("Starting RAG compliance evaluation for %s", evaluation_id)

    # Load company unstructured data
    try:
        company_data = load_unstructured_data(company_data_path)
    except Exception as e:
        logger.error("Failed to load company data: %s", e)
        return RAGComplianceResult(
            evaluation_id=evaluation_id,
            analysis_type=AnalysisType.RAG,
            compliant=False,
            compliance_score=0.0,
            explanation=f"Error loading company data: {e}",
            violated_policies=["SYSTEM_ERROR"],
            confidence=0.0,
        )

    # Retrieve relevant policies from RAG
    try:
        policy_docs = retrieve_relevant_policies(chroma_dir, company_data, top_k=5)
        
        # Citation Validation (CRITICAL)
        if policy_docs:
            # Extract cited rule IDs from company data context (simplified)
            # In production, this would come from LLM response
            cited_rule_ids = [doc.metadata.get("rule_id") for doc in policy_docs if doc.metadata.get("rule_id")]
            
            all_valid, invalid_citations = validate_citations(policy_docs, cited_rule_ids)
            if not all_valid:
                logger.warning("Citation validation failed: %d invalid citations", len(invalid_citations))
    except Exception as e:
        logger.error("Failed to retrieve policies from RAG: %s", e)
        return RAGComplianceResult(
            evaluation_id=evaluation_id,
            analysis_type=AnalysisType.RAG,
            compliant=False,
            compliance_score=0.0,
            explanation=f"Error retrieving policies from RAG: {e}",
            violated_policies=["RAG_ERROR"],
            confidence=0.0,
        )

    if not policy_docs:
        logger.warning("No policies retrieved from RAG")
        return RAGComplianceResult(
            evaluation_id=evaluation_id,
            analysis_type=AnalysisType.RAG,
            compliant=False,
            compliance_score=0.0,
            explanation="No relevant policies found in RAG database.",
            violated_policies=[],
            confidence=0.0,
        )

    # Build policy context string
    policy_context = "\n\n---\n\n".join(
        [
            f"Policy Rule ID: {doc.metadata.get('rule_id', 'UNKNOWN')}\n"
            f"Source: {doc.metadata.get('source_pdf', 'UNKNOWN')}\n"
            f"Category: {doc.metadata.get('category', 'UNKNOWN')}\n"
            f"Policy Text:\n{doc.page_content}"
            for doc in policy_docs
        ]
    )

    # Use LLM to evaluate compliance
    llm = get_llm()

    # Chunk company data instead of truncating (architecture: Unstructured Data Chunking)
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    company_chunks = splitter.split_text(company_data)
    logger.info("Split company data into %d chunks for evaluation", len(company_chunks))

    # Evaluate each chunk and collect results
    all_chunk_results = []
    for chunk_idx, chunk_text in enumerate(company_chunks):
        prompt = RAG_EVALUATION_PROMPT.format(
            policy_context=policy_context,
            company_data=chunk_text,
        )

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = llm.invoke(prompt)
                response_text = response.content

                # Parse JSON from LLM response
                import re

                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                    all_chunk_results.append(result_data)
                    logger.info(
                        "Chunk %d/%d evaluated: compliant=%s, score=%.1f",
                        chunk_idx + 1, len(company_chunks),
                        result_data.get("compliant"),
                        result_data.get("compliance_score", 0),
                    )
                else:
                    logger.warning("No JSON found in LLM response for chunk %d", chunk_idx + 1)
                break  # Success — exit retry loop

            except Exception as e:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "LLM call failed for chunk %d (attempt %d/%d), retrying in %ds: %s",
                        chunk_idx + 1, attempt, max_retries, wait, e,
                    )
                    import time
                    time.sleep(wait)
                else:
                    logger.error(
                        "LLM evaluation failed for chunk %d after %d attempts: %s",
                        chunk_idx + 1, max_retries, e,
                    )

    # Merge results from all chunks (worst-case: any non-compliant chunk → non-compliant)
    if all_chunk_results:
        is_compliant = all(r.get("compliant", False) for r in all_chunk_results)
        avg_score = sum(r.get("compliance_score", 0.0) for r in all_chunk_results) / len(all_chunk_results)
        avg_confidence = sum(r.get("confidence", 0.5) for r in all_chunk_results) / len(all_chunk_results)
        all_violated = []
        all_explanations = []
        for r in all_chunk_results:
            all_violated.extend(r.get("violated_policies", []))
            all_explanations.append(r.get("explanation", ""))
        # Deduplicate violated policies
        unique_violated = list(dict.fromkeys(all_violated))

        result = RAGComplianceResult(
            evaluation_id=evaluation_id,
            analysis_type=AnalysisType.RAG,
            compliant=is_compliant,
            compliance_score=float(avg_score),
            explanation=" | ".join(filter(None, all_explanations)),
            violated_policies=unique_violated,
            confidence=float(avg_confidence),
            retrieved_policy_chunks=[doc.metadata.get("rule_id", "UNKNOWN") for doc in policy_docs],
        )
    else:
        # All chunks failed — fallback result
        result = RAGComplianceResult(
            evaluation_id=evaluation_id,
            analysis_type=AnalysisType.RAG,
            compliant=False,
            compliance_score=0.0,
            explanation="All evaluation chunks failed. Check LLM connectivity and input data.",
            violated_policies=[],
            confidence=0.0,
            retrieved_policy_chunks=[doc.metadata.get("rule_id", "UNKNOWN") for doc in policy_docs],
        )

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result.model_dump(mode="json", exclude_none=True), f, indent=2, default=str)

    logger.info(
        "RAG compliance evaluation completed: Score=%.1f%%, Compliant=%s (%d chunks evaluated)",
        result.compliance_score,
        result.compliant,
        len(all_chunk_results),
    )

    return result
