"""
app/api/routes.py
──────────────────
FastAPI route definitions.

Endpoints
─────────
  GET  /health           – API liveness + model availability check
  POST /extract          – Extract metadata from a single document
  POST /extract/batch    – Extract metadata from multiple documents

Implementation notes
────────────────────
  extract_from_bytes() is a synchronous, CPU-bound function (OCR + LLM).
  To avoid blocking the async FastAPI event loop, it is dispatched via
  asyncio.get_event_loop().run_in_executor() so the event loop can handle
  other requests while one document is being processed.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.config import get_settings
from app.schemas import (
    BatchExtractResponse,
    ExtractResponse,
    ExtractionResult,
    ExtractionStatus,
    HealthResponse,
)
from app.services import extraction_service, llm_service
from app.utils.file_validation import validate_file_bytes, validate_upload

logger = logging.getLogger(__name__)
router = APIRouter()

# Bounded thread-pool for CPU-bound extraction work.
# Limit concurrency so the server doesn't launch unlimited parallel
# OCR / LLM tasks under batch load (each task can be memory-hungry).
_EXTRACT_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="extract-")

ALLOWED_CONTENT_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/png",
    "image/jpeg",
    "application/octet-stream",  # browsers sometimes send this for docx
}


# ─────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health and model readiness check",
    tags=["Monitoring"],
)
async def health_check() -> HealthResponse:
    """
    Returns the operational status of the API and whether the configured
    Ollama model is reachable and available.
    """
    settings = get_settings()
    loop = asyncio.get_event_loop()
    # Run the synchronous Ollama probe off the event loop thread
    model_ok, model_err = await loop.run_in_executor(
        _EXTRACT_EXECUTOR, llm_service.check_model_available
    )

    return HealthResponse(
        status="ok" if model_ok else "degraded",
        model=settings.ollama_model,
        model_available=model_ok,
        model_error=model_err,
    )


# ─────────────────────────────────────────────────────────────
# POST /extract
# ─────────────────────────────────────────────────────────────


@router.post(
    "/extract",
    response_model=ExtractResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract metadata from a single document",
    tags=["Extraction"],
)
async def extract_single(
    file: UploadFile = File(..., description="DOCX, PNG, JPG or JPEG document"),
) -> ExtractResponse:
    """
    Upload a rental agreement document (DOCX or image) and receive
    the six extracted metadata fields in structured JSON.

    Supported formats: `.docx`, `.png`, `.jpg`, `.jpeg`

    Returns structured JSON with:
    - Six metadata fields
    - Per-field evidence and confidence
    - Stage-level timing breakdown (timings field)
    - Extraction method used (docx / ocr)
    - Extraction status
    """
    filename = file.filename or "unknown"
    logger.info("POST /extract — file: '%s'", filename)

    # 1. Extension & metadata validation
    validate_upload(file)

    # 2. Read bytes and perform content validation (size, magic bytes, emptiness)
    data = await file.read()
    validate_file_bytes(data, filename)

    # 3. Perform extraction off the event loop (CPU-bound: OCR + LLM inference)
    loop = asyncio.get_event_loop()
    result: ExtractionResult = await loop.run_in_executor(
        _EXTRACT_EXECUTOR,
        extraction_service.extract_from_bytes,
        data,
        filename,
    )

    # If model is unavailable, return 503 so the client can distinguish
    # from a document-processing failure (400) or server error (500).
    if result.status == ExtractionStatus.MODEL_UNAVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=result.error_message or "LLM model is unavailable.",
        )

    return ExtractResponse(filename=filename, extraction=result)


# ─────────────────────────────────────────────────────────────
# POST /extract/batch
# ─────────────────────────────────────────────────────────────


@router.post(
    "/extract/batch",
    response_model=BatchExtractResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract metadata from multiple documents",
    tags=["Extraction"],
)
async def extract_batch(
    files: List[UploadFile] = File(
        ..., description="One or more DOCX / PNG / JPG / JPEG files"
    ),
) -> BatchExtractResponse:
    """
    Upload multiple rental agreement documents and receive extracted
    metadata for each file individually.

    Files that fail processing are included in the response with their
    error details — successful results are never discarded.

    Returns a list of per-file results with totals.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file must be provided.",
        )

    settings = get_settings()
    if len(files) > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch endpoint accepts at most 20 files per request.",
        )

    logger.info("POST /extract/batch — %d file(s)", len(files))

    loop = asyncio.get_event_loop()
    results: list[ExtractResponse] = []
    succeeded = 0
    failed = 0

    for upload in files:
        fname = upload.filename or "unknown"
        try:
            data = await upload.read()
            # Each file is processed off the event loop thread
            result = await loop.run_in_executor(
                _EXTRACT_EXECUTOR,
                extraction_service.extract_from_bytes,
                data,
                fname,
            )

            if result.status in (ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL):
                succeeded += 1
            else:
                failed += 1

            results.append(ExtractResponse(filename=fname, extraction=result))

        except Exception as exc:
            logger.exception("Batch: unhandled error for '%s'", fname)
            failed += 1
            results.append(
                ExtractResponse(
                    filename=fname,
                    extraction=ExtractionResult(
                        status=ExtractionStatus.FAILED,
                        error_message=f"Unhandled error: {exc}",
                    ),
                )
            )

    return BatchExtractResponse(
        total=len(files),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )
