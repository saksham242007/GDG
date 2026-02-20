"""
Input Loader & Validator Module.

Architecture component: "Input Loader & Validator"

Validates and prepares company input data before compliance evaluation.
Provides structured validation for both structured (SQL database) and
unstructured (text/document) inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.utils.logging_config import get_logger
from src.utils.shared import detect_input_type


logger = get_logger(__name__)


@dataclass
class InputContext:
    """
    Validated input context ready for compliance evaluation.

    Attributes:
        input_path: Absolute path to the input file
        input_type: "structured" or "unstructured"
        description: Human-readable description of the input type
        is_valid: Whether the input passed validation
        validation_errors: List of validation error messages (empty if valid)
        file_size_bytes: Size of the input file in bytes
        tables: List of table names (for structured data)
    """

    input_path: Path
    input_type: str
    description: str
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    file_size_bytes: int = 0
    tables: List[str] = field(default_factory=list)


# Maximum file sizes (in bytes)
MAX_UNSTRUCTURED_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_DB_SIZE = 500 * 1024 * 1024  # 500 MB


def validate_file_exists(input_path: Path) -> Optional[str]:
    """Check that file exists and is readable."""
    if not input_path.exists():
        return f"Input file not found: {input_path}"
    if not input_path.is_file():
        return f"Path is not a file: {input_path}"
    return None


def validate_file_size(input_path: Path, max_size: int) -> Optional[str]:
    """Check that file size is within acceptable limits."""
    size = input_path.stat().st_size
    if size == 0:
        return f"Input file is empty: {input_path}"
    if size > max_size:
        return (
            f"Input file exceeds size limit: "
            f"{size / (1024 * 1024):.1f} MB > {max_size / (1024 * 1024):.0f} MB"
        )
    return None


def validate_unstructured_encoding(input_path: Path) -> Optional[str]:
    """Check that a text file is readable with UTF-8."""
    try:
        with input_path.open("r", encoding="utf-8") as f:
            f.read(1024)  # Read first 1KB to check encoding
        return None
    except UnicodeDecodeError:
        return f"File encoding issue: {input_path} is not valid UTF-8"
    except IOError as e:
        return f"Cannot read file: {e}"


def validate_structured_db(input_path: Path) -> tuple[Optional[str], List[str]]:
    """
    Validate a SQLite database: check it's openable and has tables.

    Returns:
        Tuple of (error_message, list_of_tables)
    """
    try:
        from sqlalchemy import create_engine, inspect

        engine = create_engine(f"sqlite:///{input_path}", echo=False)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        engine.dispose()

        if not tables:
            return "Database has no tables", []
        return None, tables

    except Exception as e:
        return f"Cannot open database: {e}", []


def load_and_validate_input(input_path: Path) -> InputContext:
    """
    Load and validate company input data.

    This is the main entry point for the Input Loader & Validator component.
    It detects the input type, runs appropriate validations, and returns
    a structured InputContext.

    Args:
        input_path: Path to the company input data file

    Returns:
        InputContext with validation results and metadata
    """
    logger.info("Validating input: %s", input_path)

    # Detect input type
    input_type, description = detect_input_type(input_path)

    context = InputContext(
        input_path=input_path,
        input_type=input_type,
        description=description,
    )

    # Basic validation: file exists
    error = validate_file_exists(input_path)
    if error:
        context.is_valid = False
        context.validation_errors.append(error)
        logger.error("Input validation failed: %s", error)
        return context

    # Get file size
    context.file_size_bytes = input_path.stat().st_size

    if input_type == "structured":
        # Structured data validation (SQL database)
        size_error = validate_file_size(input_path, MAX_DB_SIZE)
        if size_error:
            context.is_valid = False
            context.validation_errors.append(size_error)

        db_error, tables = validate_structured_db(input_path)
        if db_error:
            context.is_valid = False
            context.validation_errors.append(db_error)
        context.tables = tables

    else:
        # Unstructured data validation (text/document)
        size_error = validate_file_size(input_path, MAX_UNSTRUCTURED_SIZE)
        if size_error:
            context.is_valid = False
            context.validation_errors.append(size_error)

        encoding_error = validate_unstructured_encoding(input_path)
        if encoding_error:
            context.is_valid = False
            context.validation_errors.append(encoding_error)

    if context.is_valid:
        logger.info(
            "Input validated: type=%s, size=%d bytes, tables=%s",
            context.input_type,
            context.file_size_bytes,
            context.tables or "N/A",
        )
    else:
        logger.error(
            "Input validation failed with %d error(s): %s",
            len(context.validation_errors),
            "; ".join(context.validation_errors),
        )

    return context
