"""
Unit tests for the Input Loader & Validator module.
"""

import pytest
from pathlib import Path

from src.runtime.input_loader import (
    InputContext,
    load_and_validate_input,
    validate_file_exists,
    validate_file_size,
    validate_unstructured_encoding,
)


class TestValidateFileExists:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert validate_file_exists(f) is None

    def test_missing_file(self, tmp_path):
        f = tmp_path / "nonexistent.txt"
        error = validate_file_exists(f)
        assert error is not None
        assert "not found" in error

    def test_directory_not_file(self, tmp_path):
        error = validate_file_exists(tmp_path)
        assert error is not None
        assert "not a file" in error


class TestValidateFileSize:
    def test_valid_size(self, tmp_path):
        f = tmp_path / "small.txt"
        f.write_text("data" * 100)
        assert validate_file_size(f, max_size=10_000) is None

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.touch()
        error = validate_file_size(f, max_size=10_000)
        assert error is not None
        assert "empty" in error

    def test_oversized_file(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * 1000)
        error = validate_file_size(f, max_size=100)
        assert error is not None
        assert "exceeds" in error


class TestValidateEncoding:
    def test_valid_utf8(self, tmp_path):
        f = tmp_path / "utf8.txt"
        f.write_text("Hello, world! 🌍", encoding="utf-8")
        assert validate_unstructured_encoding(f) is None


class TestLoadAndValidateInput:
    def test_valid_text_file(self, tmp_path):
        f = tmp_path / "report.txt"
        f.write_text("Compliance report content here.")

        ctx = load_and_validate_input(f)
        assert ctx.is_valid is True
        assert ctx.input_type == "unstructured"
        assert ctx.file_size_bytes > 0
        assert len(ctx.validation_errors) == 0

    def test_valid_db_file(self, tmp_path):
        """A valid SQLite database should pass validation."""
        import sqlite3

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO test_table VALUES (1, 'hello')")
        conn.commit()
        conn.close()

        ctx = load_and_validate_input(db_path)
        assert ctx.is_valid is True
        assert ctx.input_type == "structured"
        assert "test_table" in ctx.tables

    def test_missing_file(self, tmp_path):
        f = tmp_path / "missing.txt"
        ctx = load_and_validate_input(f)
        assert ctx.is_valid is False
        assert len(ctx.validation_errors) > 0

    def test_empty_text_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.touch()
        ctx = load_and_validate_input(f)
        assert ctx.is_valid is False
        assert any("empty" in e for e in ctx.validation_errors)

    def test_returns_input_context(self, tmp_path):
        f = tmp_path / "data.log"
        f.write_text("log entry 1\nlog entry 2")
        ctx = load_and_validate_input(f)
        assert isinstance(ctx, InputContext)
        assert ctx.input_path == f
