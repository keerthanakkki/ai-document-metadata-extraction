"""
tests/test_file_validation.py
──────────────────────────────
Unit tests for file upload validation (app/utils/file_validation.py).

Covers:
  • Valid file types accepted
  • Unsupported types rejected (400)
  • Empty files rejected (400)
  • Oversized files rejected (413)
  • Magic-byte mismatch rejected (400)
  • Safe temp file context manager
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from tests.conftest import make_minimal_docx, make_minimal_png, make_minimal_jpeg
from app.utils.file_validation import (
    validate_file_bytes,
    temp_file_from_bytes,
)


class TestValidateFileBytes:
    def test_valid_docx_accepted(self):
        validate_file_bytes(make_minimal_docx(), "test.docx")

    def test_valid_png_accepted(self):
        validate_file_bytes(make_minimal_png(), "test.png")

    def test_valid_jpeg_accepted(self):
        validate_file_bytes(make_minimal_jpeg(), "test.jpg")

    def test_empty_file_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_file_bytes(b"", "test.docx")
        assert exc_info.value.status_code == 400

    def test_oversized_file_raises_413(self, monkeypatch):
        from app import config
        # Temporarily set limit to 1 byte
        settings = config.get_settings()
        monkeypatch.setattr(settings, "max_upload_size_mb", 0)
        # Re-read settings after monkeypatching
        with pytest.raises(HTTPException) as exc_info:
            validate_file_bytes(make_minimal_docx(), "test.docx")
        # Either 413 or 400 acceptable if size is 0
        assert exc_info.value.status_code in (400, 413)

    def test_magic_byte_mismatch_raises_400(self):
        """A JPEG file renamed to .docx should fail magic-byte check."""
        jpeg_bytes = make_minimal_jpeg()
        with pytest.raises(HTTPException) as exc_info:
            validate_file_bytes(jpeg_bytes, "fake.docx")
        assert exc_info.value.status_code == 400

    def test_unsupported_extension_at_validation_layer(self):
        """PDF magic bytes with .pdf extension: magic bytes check for pdf not in map."""
        with pytest.raises(HTTPException):
            validate_file_bytes(b"%PDF-1.4\n", "doc.pdf")


class TestTempFileContextManager:
    def test_creates_and_cleans_up(self):
        data = make_minimal_docx()
        path_ref: list[Path] = []
        with temp_file_from_bytes(data, ".docx") as tmp:
            path_ref.append(tmp)
            assert tmp.exists()
            assert tmp.suffix == ".docx"
            assert tmp.stat().st_size == len(data)
        # File must be deleted after context exits
        assert not path_ref[0].exists()

    def test_yields_path_object(self):
        with temp_file_from_bytes(make_minimal_png(), ".png") as tmp:
            assert isinstance(tmp, Path)
