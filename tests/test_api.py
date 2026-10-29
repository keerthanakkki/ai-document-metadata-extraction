"""
tests/test_api.py
──────────────────
Integration tests for the FastAPI endpoints.

The LLM and OCR calls are mocked so tests run without Ollama or EasyOCR.
Clearly labelled as MOCKED unit tests (not end-to-end).
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_minimal_docx, make_minimal_png, make_minimal_jpeg
from app.schemas import ExtractionResult, ExtractionStatus, FieldDetail


# ── Shared mock result ────────────────────────────────────────
def _mock_success_result() -> ExtractionResult:
    return ExtractionResult(
        agreement_value="6500",
        agreement_start_date="20.05.2007",
        agreement_end_date="20.05.2008",
        renewal_notice_days=15,
        party_one="John Doe",
        party_two="Jane Smith",
        field_details={
            k: FieldDetail(value=str(v), confidence_score=0.9, is_missing=False)
            for k, v in {
                "agreement_value": "6500",
                "agreement_start_date": "20.05.2007",
                "agreement_end_date": "20.05.2008",
                "renewal_notice_days": 15,
                "party_one": "John Doe",
                "party_two": "Jane Smith",
            }.items()
        },
        status=ExtractionStatus.SUCCESS,
    )


@pytest.fixture()
def client():
    """TestClient with extraction service fully mocked (MOCKED unit test)."""
    from app.main import app

    with patch(
        "app.services.extraction_service.extract_from_bytes",
        return_value=_mock_success_result(),
    ), patch(
        "app.services.llm_service.check_model_available",
        return_value=(True, None),
    ):
        yield TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model" in data
        assert "model_available" in data

    def test_health_model_available(self, client):
        data = client.get("/health").json()
        assert data["model_available"] is True


# ─────────────────────────────────────────────────────────────
# POST /extract — success cases
# ─────────────────────────────────────────────────────────────

class TestExtractEndpointSuccess:
    def test_extract_docx_returns_200(self, client):
        resp = client.post(
            "/extract",
            files={"file": ("test.docx", make_minimal_docx(), "application/octet-stream")},
        )
        assert resp.status_code == 200

    def test_extract_png_returns_200(self, client):
        resp = client.post(
            "/extract",
            files={"file": ("test.png", make_minimal_png(), "image/png")},
        )
        assert resp.status_code == 200

    def test_extract_jpeg_returns_200(self, client):
        resp = client.post(
            "/extract",
            files={"file": ("test.jpeg", make_minimal_jpeg(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_extract_response_has_six_fields(self, client):
        resp = client.post(
            "/extract",
            files={"file": ("test.docx", make_minimal_docx(), "application/octet-stream")},
        )
        data = resp.json()
        extraction = data["extraction"]
        for field in [
            "agreement_value", "agreement_start_date", "agreement_end_date",
            "renewal_notice_days", "party_one", "party_two",
        ]:
            assert field in extraction

    def test_extract_response_has_filename(self, client):
        resp = client.post(
            "/extract",
            files={"file": ("my_doc.docx", make_minimal_docx(), "application/octet-stream")},
        )
        assert resp.json()["filename"] == "my_doc.docx"

    def test_extract_response_has_field_details(self, client):
        resp = client.post(
            "/extract",
            files={"file": ("test.docx", make_minimal_docx(), "application/octet-stream")},
        )
        assert "field_details" in resp.json()["extraction"]


# ─────────────────────────────────────────────────────────────
# POST /extract — error cases
# ─────────────────────────────────────────────────────────────

class TestExtractEndpointErrors:
    def test_unsupported_format_returns_400(self, client):
        resp = client.post(
            "/extract",
            files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert resp.status_code == 400

    def test_empty_file_returns_400(self):
        """Empty file should fail validation before reaching extraction."""
        from app.main import app
        with patch("app.services.llm_service.check_model_available", return_value=(True, None)):
            tc = TestClient(app, raise_server_exceptions=False)
            resp = tc.post(
                "/extract",
                files={"file": ("empty.docx", b"", "application/octet-stream")},
            )
        assert resp.status_code == 400

    def test_model_unavailable_returns_503(self):
        """When the model is unavailable, expect HTTP 503."""
        from app.main import app
        from app.schemas import ExtractionResult, ExtractionStatus
        unavail = ExtractionResult(
            status=ExtractionStatus.MODEL_UNAVAILABLE,
            error_message="Ollama not running",
        )
        with patch(
            "app.services.extraction_service.extract_from_bytes",
            return_value=unavail,
        ), patch(
            "app.services.llm_service.check_model_available",
            return_value=(False, "Model not found"),
        ):
            tc = TestClient(app, raise_server_exceptions=False)
            resp = tc.post(
                "/extract",
                files={"file": ("test.docx", make_minimal_docx(), "application/octet-stream")},
            )
        assert resp.status_code == 503

    def test_malformed_llm_json_returns_result_with_failed_status(self):
        """
        If the LLM returns un-parseable JSON, the service returns a
        FAILED ExtractionResult (not a 500 crash).
        """
        from app.main import app
        from app.schemas import ExtractionResult, ExtractionStatus
        from app.services.llm_service import ExtractionParseError

        failed = ExtractionResult(
            status=ExtractionStatus.FAILED,
            error_message="Could not parse model output",
        )
        with patch(
            "app.services.extraction_service.extract_from_bytes",
            return_value=failed,
        ), patch(
            "app.services.llm_service.check_model_available",
            return_value=(True, None),
        ):
            tc = TestClient(app, raise_server_exceptions=False)
            resp = tc.post(
                "/extract",
                files={"file": ("test.docx", make_minimal_docx(), "application/octet-stream")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["extraction"]["status"] == "failed"


# ─────────────────────────────────────────────────────────────
# POST /extract/batch
# ─────────────────────────────────────────────────────────────

class TestBatchExtractEndpoint:
    def test_batch_two_files(self, client):
        resp = client.post(
            "/extract/batch",
            files=[
                ("files", ("a.docx", make_minimal_docx(), "application/octet-stream")),
                ("files", ("b.png", make_minimal_png(), "image/png")),
            ],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2

    def test_batch_empty_files_returns_400(self, client):
        resp = client.post("/extract/batch", files=[])
        assert resp.status_code in (400, 422)

    def test_batch_response_has_succeeded_failed_counts(self, client):
        resp = client.post(
            "/extract/batch",
            files=[
                ("files", ("a.docx", make_minimal_docx(), "application/octet-stream")),
            ],
        )
        data = resp.json()
        assert "succeeded" in data
        assert "failed" in data
