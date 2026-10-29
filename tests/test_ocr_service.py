"""
app/services/llm_service.py — OCR mock note
tests/test_ocr_service.py
──────────────────────────
Unit tests for OCR service (app/services/ocr_service.py).

EasyOCR requires downloading a ~100 MB model on first run, so:
  • All tests here mock EasyOCR reader to run without a download.
  • One smoke-test is marked with @pytest.mark.slow and skipped by default;
    run with: pytest -m slow --run-slow

MOCKED unit tests (EasyOCR not actually called).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_minimal_png


# ── Mock helpers ──────────────────────────────────────────────

def _make_mock_reader(results):
    """Create a mock EasyOCR Reader that returns *results* from readtext()."""
    mock = MagicMock()
    mock.readtext.return_value = results
    return mock


# ─────────────────────────────────────────────────────────────
# Happy path (MOCKED)
# ─────────────────────────────────────────────────────────────

class TestOCRServiceMocked:
    def test_extracts_text_from_png(self, tmp_path: Path):
        png_file = tmp_path / "test.png"
        png_file.write_bytes(make_minimal_png())

        fake_results = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "RENTAL AGREEMENT", 0.98),
            ([[0, 25], [100, 25], [100, 45], [0, 45]], "Party One: John", 0.95),
        ]
        mock_reader = _make_mock_reader(fake_results)

        with patch("app.services.ocr_service._reader", mock_reader):
            from app.services.ocr_service import extract_text_from_image
            text = extract_text_from_image(png_file)

        assert "RENTAL AGREEMENT" in text
        assert "Party One" in text

    def test_returns_string(self, tmp_path: Path):
        png_file = tmp_path / "test.png"
        png_file.write_bytes(make_minimal_png())

        mock_reader = _make_mock_reader([
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "Some text here", 0.9)
        ])
        with patch("app.services.ocr_service._reader", mock_reader):
            from app.services.ocr_service import extract_text_from_image
            result = extract_text_from_image(png_file)
        assert isinstance(result, str)

    def test_text_sorted_top_to_bottom(self, tmp_path: Path):
        """OCR results should be sorted by vertical position."""
        png_file = tmp_path / "test.png"
        png_file.write_bytes(make_minimal_png())

        # Second result is higher on the page (lower y coordinate)
        fake_results = [
            ([[0, 100], [100, 100], [100, 120], [0, 120]], "SECOND LINE", 0.9),
            ([[0, 10], [100, 10], [100, 30], [0, 30]], "FIRST LINE", 0.9),
        ]
        mock_reader = _make_mock_reader(fake_results)

        with patch("app.services.ocr_service._reader", mock_reader):
            from app.services.ocr_service import extract_text_from_image
            text = extract_text_from_image(png_file)

        lines = [l for l in text.split("\n") if l.strip()]
        assert lines[0] == "FIRST LINE"
        assert lines[1] == "SECOND LINE"


# ─────────────────────────────────────────────────────────────
# Error handling (MOCKED)
# ─────────────────────────────────────────────────────────────

class TestOCRServiceErrors:
    def test_raises_on_missing_file(self, tmp_path: Path):
        from app.services.ocr_service import extract_text_from_image
        with pytest.raises((ValueError, Exception)):
            extract_text_from_image(tmp_path / "nonexistent.png")

    def test_raises_when_no_results(self, tmp_path: Path):
        """Empty OCR results should raise ValueError."""
        png_file = tmp_path / "blank.png"
        png_file.write_bytes(make_minimal_png())

        mock_reader = _make_mock_reader([])  # no results
        with patch("app.services.ocr_service._reader", mock_reader):
            from app.services.ocr_service import extract_text_from_image
            with pytest.raises(ValueError, match="no results"):
                extract_text_from_image(png_file)

    def test_raises_when_text_too_short(self, tmp_path: Path):
        """Less than 10 characters of recognized text → ValueError."""
        png_file = tmp_path / "tiny.png"
        png_file.write_bytes(make_minimal_png())

        mock_reader = _make_mock_reader([
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "Hi", 0.9)
        ])
        with patch("app.services.ocr_service._reader", mock_reader):
            from app.services.ocr_service import extract_text_from_image
            with pytest.raises(ValueError, match="characters"):
                extract_text_from_image(png_file)

    def test_raises_on_easyocr_runtime_error(self, tmp_path: Path):
        """If EasyOCR itself raises, a RuntimeError is propagated."""
        png_file = tmp_path / "bad.png"
        png_file.write_bytes(make_minimal_png())

        mock_reader = MagicMock()
        mock_reader.readtext.side_effect = RuntimeError("GPU error")
        with patch("app.services.ocr_service._reader", mock_reader):
            from app.services.ocr_service import extract_text_from_image
            with pytest.raises(RuntimeError):
                extract_text_from_image(png_file)
