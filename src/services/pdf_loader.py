from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.utils.logging_config import get_logger


logger = get_logger(__name__)


def load_policy_pdfs(policies_dir: Path) -> List[Document]:
    """
    Load all PDF policy documents from the given directory into LangChain Documents.
    """
    if not policies_dir.exists():
        logger.warning("Policies directory does not exist: %s", policies_dir)
        return []

    pdf_paths = sorted(p for p in policies_dir.glob("*.pdf") if p.is_file())
    if not pdf_paths:
        logger.warning("No PDF files found in %s", policies_dir)
        return []

    all_docs: List[Document] = []
    for pdf_path in pdf_paths:
        logger.info("Loading PDF: %s", pdf_path.name)
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        # Tag each document with the source PDF filename
        for d in docs:
            d.metadata.setdefault("source_pdf", pdf_path.name)
        all_docs.extend(docs)

    logger.info("Loaded %d documents from %d PDFs.", len(all_docs), len(pdf_paths))
    return all_docs


def split_documents(
    docs: List[Document],
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into overlapping chunks, preserving headings and sections as much as possible.
    """
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_documents(docs)
    logger.info("Split %d documents into %d chunks.", len(docs), len(chunks))
    return chunks


def load_and_split_pdf(
    pdf_path: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Convenience function: load a single PDF and split it into chunks.

    Args:
        pdf_path: Path to a single PDF file.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Documents.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Loading PDF: %s", path.name)
    loader = PyMuPDFLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source_pdf", path.name)

    return split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

