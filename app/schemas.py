"""
app/schemas.py
──────────────
Pydantic models that define:
  • The canonical six-field extraction result (ExtractionResult)
  • Per-field evidence / confidence (FieldDetail)
  • API request/response envelopes
  • Evaluation output shapes
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────


class ExtractionStatus(str, Enum):
    """Overall status of an extraction attempt."""

    SUCCESS = "success"
    PARTIAL = "partial"          # some fields missing
    FAILED = "failed"            # pipeline error; no usable data
    MODEL_UNAVAILABLE = "model_unavailable"


class ConfidenceLevel(str, Enum):
    """Coarse confidence band returned alongside each field."""

    HIGH = "high"       # ≥ 0.75
    MEDIUM = "medium"   # 0.50 – 0.75
    LOW = "low"         # 0.30 – 0.50
    UNKNOWN = "unknown" # model did not assign confidence
    MISSING = "missing" # field could not be found


def _band(score: Optional[float]) -> ConfidenceLevel:
    if score is None:
        return ConfidenceLevel.UNKNOWN
    if score >= 0.75:
        return ConfidenceLevel.HIGH
    if score >= 0.50:
        return ConfidenceLevel.MEDIUM
    if score >= 0.30:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.LOW


# ─────────────────────────────────────────────────────────────
# Field-level detail
# ─────────────────────────────────────────────────────────────


class FieldDetail(BaseModel):
    """Evidence and confidence for a single extracted field."""

    value: Optional[str] = Field(
        None,
        description="The extracted value, exactly as found in the document.",
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model-estimated confidence [0, 1]. None when unknown.",
    )
    confidence_level: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    evidence: Optional[str] = Field(
        None,
        description="Verbatim supporting text from the document.",
    )
    is_missing: bool = Field(
        False,
        description="True when the field could not be found in the document.",
    )

    @model_validator(mode="after")
    def _set_band(self) -> "FieldDetail":
        if self.is_missing:
            self.confidence_level = ConfidenceLevel.MISSING
        else:
            self.confidence_level = _band(self.confidence_score)
        return self


# ─────────────────────────────────────────────────────────────
# Raw LLM output (internal use)
# ─────────────────────────────────────────────────────────────


class RawLLMOutput(BaseModel):
    """
    Validated shape of the JSON object we expect the LLM to return.
    All fields are optional so that a partial response can still be
    parsed rather than discarded entirely.
    """

    agreement_value: Optional[str] = None
    agreement_start_date: Optional[str] = None
    agreement_end_date: Optional[str] = None
    renewal_notice_days: Optional[Any] = None   # int or str in the wild
    party_one: Optional[str] = None
    party_two: Optional[str] = None

    # Optional evidence block (model may or may not return it)
    evidence: Optional[dict[str, Optional[str]]] = None
    confidence: Optional[dict[str, Optional[float]]] = None


# ─────────────────────────────────────────────────────────────
# Canonical extraction result
# ─────────────────────────────────────────────────────────────


class ExtractionResult(BaseModel):
    """
    The six required metadata fields plus per-field evidence/confidence.
    This is what the API returns.
    """

    # ── Six canonical fields ──────────────────────────────────
    agreement_value: Optional[str] = Field(
        None,
        description="Monthly rental / agreement value as stated in the document.",
    )
    agreement_start_date: Optional[str] = Field(
        None,
        description="Date on which the agreement becomes effective.",
    )
    agreement_end_date: Optional[str] = Field(
        None,
        description="Date on which the agreement expires.",
    )
    renewal_notice_days: Optional[int] = Field(
        None,
        description="Number of days' notice required for renewal/termination.",
    )
    party_one: Optional[str] = Field(
        None,
        description="Landlord / lessor / first party as named in the agreement.",
    )
    party_two: Optional[str] = Field(
        None,
        description="Tenant / lessee / second party as named in the agreement.",
    )

    # ── Supporting metadata ───────────────────────────────────
    field_details: dict[str, FieldDetail] = Field(
        default_factory=dict,
        description="Per-field evidence and confidence information.",
    )

    status: ExtractionStatus = ExtractionStatus.SUCCESS
    error_message: Optional[str] = None
    document_text_preview: Optional[str] = Field(
        None,
        description="First 500 characters of extracted document text (for debugging).",
    )
    timings: dict[str, float] = Field(
        default_factory=dict,
        description="Execution time breakdown: text_extraction_sec, llm_inference_sec, total_sec.",
    )
    extraction_method: Optional[str] = Field(
        None,
        description="Text extraction method used: 'docx' (instant python-docx) or 'ocr' (EasyOCR).",
    )

    @model_validator(mode="after")
    def _derive_status(self) -> "ExtractionResult":
        if self.status in (ExtractionStatus.FAILED, ExtractionStatus.MODEL_UNAVAILABLE):
            return self
        fields_present = [
            f for f in [
                self.agreement_value,
                self.agreement_start_date,
                self.agreement_end_date,
                self.renewal_notice_days,
                self.party_one,
                self.party_two,
            ]
            if f is not None
        ]
        if len(fields_present) == 0:
            self.status = ExtractionStatus.FAILED
        elif len(fields_present) < 6:
            self.status = ExtractionStatus.PARTIAL
        return self

    def to_flat_dict(self) -> dict:
        """Return a flat dict suitable for writing to CSV."""
        return {
            "agreement_value": self.agreement_value,
            "agreement_start_date": self.agreement_start_date,
            "agreement_end_date": self.agreement_end_date,
            "renewal_notice_days": self.renewal_notice_days,
            "party_one": self.party_one,
            "party_two": self.party_two,
        }


# ─────────────────────────────────────────────────────────────
# API response envelopes
# ─────────────────────────────────────────────────────────────


class ExtractResponse(BaseModel):
    """Response for POST /extract."""

    filename: str
    extraction: ExtractionResult


class BatchExtractResponse(BaseModel):
    """Response for POST /extract/batch."""

    total: int
    succeeded: int
    failed: int
    results: list[ExtractResponse]


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    api_version: str = "1.0.0"
    model: str
    model_available: bool
    model_error: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Evaluation shapes
# ─────────────────────────────────────────────────────────────


class FieldRecall(BaseModel):
    """Per-field recall statistics."""

    field: str
    exact_matches: int
    total_ground_truth: int
    recall_pct: float


class EvaluationReport(BaseModel):
    """Full evaluation report written to outputs/evaluation_report.json."""

    micro_averaged_recall: float
    total_exact_matches: int
    total_ground_truth_values: int
    field_recalls: list[FieldRecall]
    overlapping_filenames: list[str]
    overlap_warning: str
    per_document: list[dict]
    comparison_policy: str
    expected_test_documents: int = 4
    successfully_processed: int = 4
    failed_processing: int = 0
    missing_files: list[str] = Field(default_factory=list)
    duplicate_files: list[str] = Field(default_factory=list)
    normalized_micro_averaged_recall: Optional[float] = None
    normalized_field_recalls: Optional[list[FieldRecall]] = None
    processing_time_statistics: Optional[dict[str, float]] = None
