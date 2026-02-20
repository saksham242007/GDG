from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import get_embeddings
from src.models.schemas import ExtractedRule, RuleComplexity
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


def _get_embeddings():
    """
    Returns the configured Embeddings instance (via centralized config).
    """
    return get_embeddings()


def build_chroma_vector_store(
    persist_directory: Path,
) -> Chroma:
    """
    Initialize (or load existing) Chroma vector store with OpenAI embeddings.
    """
    embeddings = _get_embeddings()
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    return vector_store


def rules_to_documents(rules: Iterable[ExtractedRule]) -> List[Document]:
    """
    Convert complex rules into LangChain Documents with rich metadata.
    """
    docs: List[Document] = []
    for rule in rules:
        if rule.rule_complexity != RuleComplexity.COMPLEX:
            continue

        metadata = {
            "rule_id": rule.rule_id,
            "category": rule.category,
            "source_pdf": rule.source_pdf,
            "complexity": rule.rule_complexity.value,
        }
        docs.append(Document(page_content=rule.rule_text, metadata=metadata))

    return docs


def persist_complex_rules_to_chroma(
    persist_directory: Path,
    rules: Iterable[ExtractedRule],
) -> int:
    """
    Store complex rules in a persistent Chroma vector database.

    Returns:
        Number of complex rules embedded and stored.
    """
    persist_directory.mkdir(parents=True, exist_ok=True)
    vector_store = build_chroma_vector_store(persist_directory)

    docs = rules_to_documents(rules)
    if not docs:
        logger.info("No complex rules to store in Chroma.")
        return 0

    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    vector_store.add_texts(texts=texts, metadatas=metadatas)
    # NOTE: ChromaDB >=0.4 auto-persists when persist_directory is set.
    # Calling .persist() is deprecated and removed in >=0.5.

    logger.info(
        "Persisted %d complex rules into Chroma at %s.",
        len(docs),
        persist_directory,
    )
    return len(docs)

