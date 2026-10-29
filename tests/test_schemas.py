"""
tests/test_schemas.py
──────────────────────
Unit tests for Pydantic schema validation (app/schemas.py).
No external services required.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    ConfidenceLevel,
    ExtractionResult,
    ExtractionStatus,
    FieldDetail,
    RawLLMOutput,
    ExtractResponse,
    BatchExtractResponse,
    HealthResponse,
)


class TestFieldDetail:
    def test_high_confidence_band(self):
        fd = FieldDetail(value="6500", confidence_score=0.95, is_missing=False)
        assert fd.confidence_level == ConfidenceLevel.HIGH

    def test_medium_confidence_band(self):
        fd = FieldDetail(value="6500", confidence_score=0.6, is_missing=False)
        assert fd.confidence_level == ConfidenceLevel.MEDIUM

    def test_low_confidence_band(self):
        fd = FieldDetail(value="6500", confidence_score=0.35, is_missing=False)
        assert fd.confidence_level == ConfidenceLevel.LOW

    def test_missing_field_sets_band(self):
        fd = FieldDetail(value=None, is_missing=True)
        assert fd.confidence_level == ConfidenceLevel.MISSING

    def test_unknown_when_no_score(self):
        fd = FieldDetail(value="something", confidence_score=None, is_missing=False)
        assert fd.confidence_level == ConfidenceLevel.UNKNOWN

    def test_confidence_score_bounds(self):
        with pytest.raises(ValidationError):
            FieldDetail(value="x", confidence_score=1.5)
        with pytest.raises(ValidationError):
            FieldDetail(value="x", confidence_score=-0.1)


class TestRawLLMOutput:
    def test_all_fields_present(self):
        raw = RawLLMOutput(
            agreement_value="6500",
            agreement_start_date="20.05.2007",
            agreement_end_date="20.05.2008",
            renewal_notice_days=15,
            party_one="John",
            party_two="Jane",
        )
        assert raw.agreement_value == "6500"
        assert raw.renewal_notice_days == 15

    def test_all_fields_none(self):
        """Fully null response is valid (partial extraction)."""
        raw = RawLLMOutput()
        assert raw.agreement_value is None
        assert raw.party_one is None

    def test_extra_fields_ignored(self):
        """Extra keys from the LLM are ignored, not errors."""
        raw = RawLLMOutput.model_validate(
            {"agreement_value": "1000", "unexpected_key": "ignore me"}
        )
        assert raw.agreement_value == "1000"

    def test_renewal_days_as_string(self):
        """renewal_notice_days may arrive as a string from the LLM."""
        raw = RawLLMOutput.model_validate({"renewal_notice_days": "30"})
        assert raw.renewal_notice_days == "30"  # stored as-is; coerced later


class TestExtractionResult:
    def test_success_status_when_all_fields_present(self):
        result = ExtractionResult(
            agreement_value="6500",
            agreement_start_date="20.05.2007",
            agreement_end_date="20.05.2008",
            renewal_notice_days=15,
            party_one="John",
            party_two="Jane",
        )
        assert result.status == ExtractionStatus.SUCCESS

    def test_partial_status_when_some_fields_missing(self):
        result = ExtractionResult(
            agreement_value="6500",
            agreement_start_date="20.05.2007",
            # remaining fields absent
        )
        assert result.status == ExtractionStatus.PARTIAL

    def test_failed_status_when_no_fields(self):
        result = ExtractionResult()
        assert result.status == ExtractionStatus.FAILED

    def test_explicit_failed_status_preserved(self):
        result = ExtractionResult(
            status=ExtractionStatus.FAILED,
            error_message="OCR failed",
        )
        assert result.status == ExtractionStatus.FAILED
        assert result.error_message == "OCR failed"

    def test_to_flat_dict(self):
        result = ExtractionResult(
            agreement_value="9000",
            party_one="Landlord Name",
        )
        flat = result.to_flat_dict()
        assert flat["agreement_value"] == "9000"
        assert flat["party_one"] == "Landlord Name"
        assert "party_two" in flat

    def test_model_unavailable_status(self):
        result = ExtractionResult(
            status=ExtractionStatus.MODEL_UNAVAILABLE,
            error_message="Ollama not reachable",
        )
        assert result.status == ExtractionStatus.MODEL_UNAVAILABLE


class TestHealthResponse:
    def test_valid_health_response(self):
        h = HealthResponse(
            status="ok",
            model="qwen2.5:7b",
            model_available=True,
        )
        assert h.status == "ok"
        assert h.model_available is True

    def test_degraded_with_error(self):
        h = HealthResponse(
            status="degraded",
            model="qwen2.5:7b",
            model_available=False,
            model_error="Model not found",
        )
        assert h.model_error == "Model not found"
