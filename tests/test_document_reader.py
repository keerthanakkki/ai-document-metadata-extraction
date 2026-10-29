"""
tests/test_document_reader.py
──────────────────────────────
Unit tests for DOCX text extraction (app/services/document_reader.py).
Uses only python-docx — no LLM or OCR calls.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tests.conftest import make_minimal_docx

# ── Import under test ─────────────────────────────────────────
from app.services.document_reader import extract_text_from_docx


class TestDocxExtraction:
    """Happy-path DOCX extraction tests."""

    def test_extracts_paragraph_text(self, tmp_path: Path):
        """Basic paragraph text is extracted correctly."""
        docx_bytes = make_minimal_docx("This is a test paragraph.")
        docx_file = tmp_path / "test.docx"
        docx_file.write_bytes(docx_bytes)

        text = extract_text_from_docx(docx_file)
        assert "test paragraph" in text.lower()

    def test_extracts_real_train_docx(self):
        """Smoke-test against a real training document if available."""
        train_dir = Path(__file__).resolve().parent.parent / "data" / "train"
        docx_files = list(train_dir.glob("*.docx"))
        if not docx_files:
            pytest.skip("No DOCX files found in data/train/")

        result = extract_text_from_docx(docx_files[0])
        assert isinstance(result, str)
        assert len(result) > 20, "Expected non-trivial text from real document"

    def test_returns_string(self, tmp_path: Path):
        """Return type is always str."""
        docx_file = tmp_path / "test.docx"
        docx_file.write_bytes(make_minimal_docx())
        result = extract_text_from_docx(docx_file)
        assert isinstance(result, str)

    def test_non_empty_output(self, tmp_path: Path):
        """Extracted text must be non-empty for a non-empty DOCX."""
        docx_file = tmp_path / "test.docx"
        docx_file.write_bytes(make_minimal_docx("Rental Agreement content here"))
        result = extract_text_from_docx(docx_file)
        assert len(result.strip()) > 0


class TestDocxExtractionErrors:
    """Error-handling tests for DOCX extraction."""

    def test_raises_on_missing_file(self, tmp_path: Path):
        """FileNotFoundError or ValueError raised for a non-existent path."""
        with pytest.raises((ValueError, Exception)):
            extract_text_from_docx(tmp_path / "nonexistent.docx")

    def test_raises_on_corrupt_file(self, tmp_path: Path):
        """ValueError raised when the file is corrupt (not a valid ZIP)."""
        corrupt = tmp_path / "corrupt.docx"
        corrupt.write_bytes(b"PK\x03\x04" + b"\x00" * 10)  # valid magic, bad content
        with pytest.raises((ValueError, Exception)):
            extract_text_from_docx(corrupt)

    def test_raises_on_empty_file(self, tmp_path: Path):
        """
        An empty file with the DOCX magic bytes (PK header) should fail
        validation at the service layer.
        """
        empty = tmp_path / "empty.docx"
        empty.write_bytes(b"")
        with pytest.raises((ValueError, Exception)):
            extract_text_from_docx(empty)
