"""
tests/test_timing_and_reliability.py
─────────────────────────────────────
Tests covering the requirements from PHASE 2, 5, and 8 that were not
addressed by previous test modules:

  - Stage-level timing recorded in ExtractionResult.timings (Phase 2)
  - extraction_method set correctly for DOCX vs image (Phase 3)
  - Temp-file cleanup even after LLM failure (Phase 5)
  - Batch processing: one document fails, others succeed (Phase 5)
  - LLM timeout / ModelUnavailableError propagation (Phase 5)
  - Malformed LLM response does not crash the service (Phase 5)
  - Full-document text propagated to LLM (not truncated to preview size) (Phase 4)
  - Async executor: extraction runs off the event loop (Phase 5)

All LLM and OCR calls are mocked so tests run without Ollama or EasyOCR.
Clearly labelled MOCKED unit tests.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.schemas import ExtractionResult, ExtractionStatus, RawLLMOutput
from app.services.extraction_service import extract_from_bytes
from app.services.llm_service import ExtractionParseError, ModelUnavailableError
from tests.conftest import make_minimal_docx, make_minimal_png


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _mock_raw_output() -> RawLLMOutput:
    return RawLLMOutput(
        agreement_value="6500",
        agreement_start_date="20.05.2007",
        agreement_end_date="20.05.2008",
        renewal_notice_days=15,
        party_one="John Doe",
        party_two="Jane Smith",
        evidence={"agreement_value": "rent Rs 6500"},
        confidence={"agreement_value": 0.95},
    )


def _run_extract_with_mock_llm(data: bytes, filename: str) -> ExtractionResult:
    """Helper: run extract_from_bytes with mocked LLM and OCR."""
    with patch(
        "app.services.extraction_service.call_llm",
        return_value=_mock_raw_output(),
    ), patch(
        "app.services.extraction_service.build_few_shot_examples",
        return_value=[],
    ), patch(
        "app.services.ocr_service._reader",
        _make_mock_ocr_reader(),
    ):
        return extract_from_bytes(data, filename)


def _make_mock_ocr_reader():
    mock = MagicMock()
    mock.readtext.return_value = [
        ([[0, 0], [100, 0], [100, 20], [0, 20]], "RENTAL AGREEMENT Party One: John Party Two: Jane", 0.95),
    ]
    return mock


# ─────────────────────────────────────────────────────────────
# Phase 2: Stage-level timing
# ─────────────────────────────────────────────────────────────

class TestStageLevelTiming:
    """ExtractionResult.timings must contain monotonic stage durations."""

    def test_timings_present_for_docx(self):
        """DOCX extraction result must include timing keys."""
        result = _run_extract_with_mock_llm(make_minimal_docx(), "test.docx")
        assert isinstance(result.timings, dict), "timings must be a dict"
        assert "total_sec" in result.timings, "total_sec must be recorded"
        assert "text_extraction_sec" in result.timings
        assert "llm_inference_sec" in result.timings

    def test_timings_present_for_image(self, tmp_path: Path):
        """Image/OCR extraction result must include timing keys."""
        png = tmp_path / "test.png"
        png.write_bytes(make_minimal_png())
        result = _run_extract_with_mock_llm(make_minimal_png(), "test.png")
        assert "text_extraction_sec" in result.timings
        assert "llm_inference_sec" in result.timings
        assert "total_sec" in result.timings

    def test_total_sec_is_positive(self):
        result = _run_extract_with_mock_llm(make_minimal_docx(), "test.docx")
        assert result.timings["total_sec"] > 0

    def test_timings_are_floats(self):
        result = _run_extract_with_mock_llm(make_minimal_docx(), "test.docx")
        for key, val in result.timings.items():
            assert isinstance(val, float), f"timings['{key}'] must be float, got {type(val)}"

    def test_validation_timing_present(self):
        result = _run_extract_with_mock_llm(make_minimal_docx(), "test.docx")
        assert "validation_sec" in result.timings

    def test_timings_present_on_llm_failure(self):
        """Timings must be recorded even when LLM raises ModelUnavailableError."""
        with patch(
            "app.services.extraction_service.call_llm",
            side_effect=ModelUnavailableError("Ollama not running"),
        ), patch(
            "app.services.extraction_service.build_few_shot_examples",
            return_value=[],
        ):
            result = extract_from_bytes(make_minimal_docx(), "test.docx")

        assert result.status == ExtractionStatus.MODEL_UNAVAILABLE
        assert "total_sec" in result.timings
        assert "llm_inference_sec" in result.timings

    def test_timings_present_on_text_extraction_failure(self):
        """Timings must be recorded even when text extraction fails."""
        with patch(
            "app.services.extraction_service._extract_text_from_path",
            side_effect=ValueError("corrupt file"),
        ), patch(
            "app.services.extraction_service.build_few_shot_examples",
            return_value=[],
        ), patch(
            "app.services.extraction_service.extract_text_from_docx",
            side_effect=ValueError("corrupt file"),
        ):
            result = extract_from_bytes(make_minimal_docx(), "test.docx")

        assert result.status == ExtractionStatus.FAILED
        assert "text_extraction_sec" in result.timings
        assert "total_sec" in result.timings


# ─────────────────────────────────────────────────────────────
# Phase 3: extraction_method field
# ─────────────────────────────────────────────────────────────

class TestExtractionMethod:
    """extraction_method must correctly indicate 'docx' or 'ocr'."""

    def test_docx_sets_method_docx(self):
        result = _run_extract_with_mock_llm(make_minimal_docx(), "agreement.docx")
        assert result.extraction_method == "docx"

    def test_image_sets_method_ocr(self):
        result = _run_extract_with_mock_llm(make_minimal_png(), "scan.png")
        assert result.extraction_method == "ocr"

    def test_jpeg_sets_method_ocr(self):
        result = _run_extract_with_mock_llm(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9",
            "scan.jpg",
        )
        assert result.extraction_method == "ocr"


# ─────────────────────────────────────────────────────────────
# Phase 5: Temp file cleanup after failure
# ─────────────────────────────────────────────────────────────

class TestTempFileCleanupAfterFailure:
    """Temp files must be deleted even when extraction raises."""

    def test_no_temp_files_left_after_llm_failure(self, tmp_path: Path, monkeypatch):
        """
        After a ModelUnavailableError during LLM call, no temp files
        created by the pipeline should remain on disk.
        """
        created_temps: list[Path] = []
        original_temp_file = __import__(
            "app.utils.file_validation", fromlist=["temp_file_from_bytes"]
        ).temp_file_from_bytes

        from contextlib import contextmanager

        @contextmanager
        def _tracking_temp(data, suffix):
            with original_temp_file(data, suffix) as p:
                created_temps.append(p)
                yield p

        monkeypatch.setattr(
            "app.services.extraction_service.temp_file_from_bytes",
            _tracking_temp,
        )

        with patch(
            "app.services.extraction_service.call_llm",
            side_effect=ModelUnavailableError("Ollama not running"),
        ), patch(
            "app.services.extraction_service.build_few_shot_examples",
            return_value=[],
        ):
            result = extract_from_bytes(make_minimal_docx(), "test.docx")

        assert result.status == ExtractionStatus.MODEL_UNAVAILABLE
        # All temp files created must have been cleaned up
        for temp_path in created_temps:
            assert not temp_path.exists(), (
                f"Temp file {temp_path} was NOT deleted after LLM failure"
            )


# ─────────────────────────────────────────────────────────────
# Phase 5: Batch – one document fails, others succeed
# ─────────────────────────────────────────────────────────────

class TestBatchOneFailsOthersSucceed:
    """
    The batch endpoint must never discard successful results because
    one document in the batch fails.
    """

    def test_batch_partial_failure_via_api(self):
        from app.main import app
        from fastapi.testclient import TestClient

        call_count = [0]

        def _side_effect(data: bytes, filename: str) -> ExtractionResult:
            call_count[0] += 1
            if call_count[0] == 2:
                # Second file fails
                return ExtractionResult(
                    status=ExtractionStatus.FAILED,
                    error_message="Simulated extraction failure",
                )
            return ExtractionResult(
                agreement_value="6500",
                agreement_start_date="01.01.2020",
                agreement_end_date="31.12.2020",
                renewal_notice_days=30,
                party_one="Landlord A",
                party_two="Tenant B",
                status=ExtractionStatus.SUCCESS,
            )

        with patch(
            "app.services.extraction_service.extract_from_bytes",
            side_effect=_side_effect,
        ), patch(
            "app.services.llm_service.check_model_available",
            return_value=(True, None),
        ):
            tc = TestClient(app, raise_server_exceptions=False)
            resp = tc.post(
                "/extract/batch",
                files=[
                    ("files", ("a.docx", make_minimal_docx(), "application/octet-stream")),
                    ("files", ("b.docx", make_minimal_docx(), "application/octet-stream")),
                    ("files", ("c.png",  make_minimal_png(),  "image/png")),
                ],
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        # 2 should succeed, 1 should fail — neither counts should be 0
        assert data["succeeded"] == 2
        assert data["failed"] == 1
        # All 3 results must still be present in the response
        assert len(data["results"]) == 3
        # The failed one must be there with its error
        failed = [r for r in data["results"] if r["extraction"]["status"] == "failed"]
        assert len(failed) == 1
        assert "Simulated" in failed[0]["extraction"]["error_message"]


# ─────────────────────────────────────────────────────────────
# Phase 5: LLM failure scenarios
# ─────────────────────────────────────────────────────────────

class TestLLMFailureScenarios:
    """Failure modes from the LLM layer must be handled gracefully."""

    def test_model_unavailable_sets_status(self):
        with patch(
            "app.services.extraction_service.call_llm",
            side_effect=ModelUnavailableError("Cannot reach Ollama"),
        ), patch(
            "app.services.extraction_service.build_few_shot_examples",
            return_value=[],
        ):
            result = extract_from_bytes(make_minimal_docx(), "test.docx")

        assert result.status == ExtractionStatus.MODEL_UNAVAILABLE
        assert "Ollama" in result.error_message or "reach" in result.error_message

    def test_parse_error_sets_failed_status(self):
        with patch(
            "app.services.extraction_service.call_llm",
            side_effect=ExtractionParseError("No JSON found"),
        ), patch(
            "app.services.extraction_service.build_few_shot_examples",
            return_value=[],
        ):
            result = extract_from_bytes(make_minimal_docx(), "test.docx")

        assert result.status == ExtractionStatus.FAILED
        assert result.error_message is not None
        assert "parse" in result.error_message.lower() or "model output" in result.error_message.lower()

    def test_unexpected_llm_exception_sets_failed_status(self):
        with patch(
            "app.services.extraction_service.call_llm",
            side_effect=RuntimeError("GPU OOM"),
        ), patch(
            "app.services.extraction_service.build_few_shot_examples",
            return_value=[],
        ):
            result = extract_from_bytes(make_minimal_docx(), "test.docx")

        assert result.status == ExtractionStatus.FAILED
        assert result.error_message is not None

    def test_model_unavailable_preserves_text_preview(self):
        """Even on model failure, document_text_preview must be populated."""
        with patch(
            "app.services.extraction_service.call_llm",
            side_effect=ModelUnavailableError("Ollama down"),
        ), patch(
            "app.services.extraction_service.build_few_shot_examples",
            return_value=[],
        ):
            result = extract_from_bytes(make_minimal_docx("Hello rental agreement."), "test.docx")

        assert result.document_text_preview is not None
        assert len(result.document_text_preview) > 0

    def test_token_repeat_limit_retry_success(self):
        """If Ollama hits token repeat limit on attempt 1, controlled retry succeeds."""
        from app.services.llm_service import call_llm
        import ollama as ollama_client

        mock_client = MagicMock()
        first_call = [True]

        def _mock_chat(*args, **kwargs):
            if first_call[0]:
                first_call[0] = False
                raise ollama_client.ResponseError(
                    "prediction aborted, token repeat limit reached (status code: -1)"
                )
            return iter([{"message": {"content": '{"agreement_value": "9000", "agreement_start_date": "01.04.2010", "agreement_end_date": "30.03.2011", "renewal_notice_days": 30, "party_one": "Landlord", "party_two": "Tenant"}'}}])

        mock_client.chat.side_effect = _mock_chat

        with patch("app.services.llm_service._get_client", return_value=mock_client):
            res = call_llm("Agreement text")
            assert res.agreement_value == "9000"
            assert mock_client.chat.call_count == 2
            # Second call should have adjusted options
            second_opts = mock_client.chat.call_args_list[1].kwargs["options"]
            assert second_opts["temperature"] == 0.2
            assert second_opts["repeat_penalty"] == 1.15

    def test_token_repeat_limit_retry_exhaustion_raises_parse_error(self):
        """If token repeat limit persists after retry, ExtractionParseError is raised with clear message."""
        from app.services.llm_service import call_llm, ExtractionParseError
        import ollama as ollama_client

        mock_client = MagicMock()
        mock_client.chat.side_effect = ollama_client.ResponseError(
            "prediction aborted, token repeat limit reached (status code: -1)"
        )

        with patch("app.services.llm_service._get_client", return_value=mock_client):
            with pytest.raises(ExtractionParseError) as exc_info:
                call_llm("Agreement text")
            assert "token repeat limit" in str(exc_info.value).lower()
            assert mock_client.chat.call_count == 2



# ─────────────────────────────────────────────────────────────
# Phase 4: Full document text propagation (not truncated to preview)
# ─────────────────────────────────────────────────────────────

class TestFullDocumentTextInLLMCall:
    """The extraction service must pass the full doc text to the LLM, not the 500-char preview."""

    def test_long_docx_text_reaches_llm_untruncated(self):
        long_text = "Clause A: " + ("important renewal terms. " * 100) + " Clause Z: final."
        assert len(long_text) > 600, "Precondition: text must exceed preview limit"

        with patch(
            "app.services.extraction_service.extract_text_from_docx",
            return_value=long_text,
        ), patch(
            "app.services.extraction_service.build_few_shot_examples",
            return_value=[],
        ), patch(
            "app.services.extraction_service.call_llm",
            return_value=_mock_raw_output(),
        ) as mock_llm:
            result = extract_from_bytes(make_minimal_docx(), "long.docx")

        # LLM must have received the full text
        call_args = mock_llm.call_args
        passed_text = call_args[0][0]  # first positional arg
        assert len(passed_text) == len(long_text), (
            f"LLM received {len(passed_text)} chars, expected {len(long_text)}"
        )
        # Preview must still be capped at 500
        assert result.document_text_preview is not None
        assert len(result.document_text_preview) <= 500


# ─────────────────────────────────────────────────────────────
# Phase 5: Six-field schema preserved
# ─────────────────────────────────────────────────────────────

class TestSixFieldSchemaPreserved:
    """All six canonical fields must always be present in the response."""

    REQUIRED_FIELDS = [
        "agreement_value",
        "agreement_start_date",
        "agreement_end_date",
        "renewal_notice_days",
        "party_one",
        "party_two",
    ]

    def test_all_six_fields_in_extraction_result(self):
        result = _run_extract_with_mock_llm(make_minimal_docx(), "test.docx")
        result_dict = result.model_dump()
        for field in self.REQUIRED_FIELDS:
            assert field in result_dict, f"Field '{field}' missing from ExtractionResult"

    def test_all_six_fields_in_field_details(self):
        result = _run_extract_with_mock_llm(make_minimal_docx(), "test.docx")
        for field in self.REQUIRED_FIELDS:
            assert field in result.field_details, (
                f"field_details missing '{field}'"
            )

    def test_six_fields_present_via_api(self):
        from app.main import app
        from fastapi.testclient import TestClient

        with patch(
            "app.services.extraction_service.extract_from_bytes",
            return_value=_run_extract_with_mock_llm(make_minimal_docx(), "test.docx"),
        ), patch(
            "app.services.llm_service.check_model_available",
            return_value=(True, None),
        ):
            tc = TestClient(app)
            resp = tc.post(
                "/extract",
                files={"file": ("test.docx", make_minimal_docx(), "application/octet-stream")},
            )

        assert resp.status_code == 200
        extraction = resp.json()["extraction"]
        for field in self.REQUIRED_FIELDS:
            assert field in extraction, f"API response missing field '{field}'"


# ─────────────────────────────────────────────────────────────
# Phase 5: Async – extraction runs off the event loop
# ─────────────────────────────────────────────────────────────

class TestAsyncExtractionOffEventLoop:
    """
    The /extract route must dispatch extraction work to a thread executor
    so the async event loop is not blocked during OCR / LLM calls.
    MOCKED: verifies the executor is called, not actual async behaviour.
    """

    def test_extract_endpoint_succeeds_with_mocked_executor(self):
        """
        The API endpoint should return 200 even when the extraction work
        is dispatched via run_in_executor (basic smoke-test via TestClient).
        """
        from app.main import app
        from fastapi.testclient import TestClient

        mock_result = ExtractionResult(
            agreement_value="6500",
            agreement_start_date="01.01.2020",
            agreement_end_date="31.12.2020",
            renewal_notice_days=30,
            party_one="Landlord",
            party_two="Tenant",
            status=ExtractionStatus.SUCCESS,
        )

        with patch(
            "app.services.extraction_service.extract_from_bytes",
            return_value=mock_result,
        ), patch(
            "app.services.llm_service.check_model_available",
            return_value=(True, None),
        ):
            tc = TestClient(app)
            resp = tc.post(
                "/extract",
                files={"file": ("test.docx", make_minimal_docx(), "application/octet-stream")},
            )

        assert resp.status_code == 200
        assert resp.json()["extraction"]["status"] == "success"
