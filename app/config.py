"""
app/config.py
─────────────
Centralised configuration loaded from environment variables / .env file.
All tunable parameters live here so that no magic values are scattered
across the codebase.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root (two levels up from this file: app/ → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings resolved from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Ollama ────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama HTTP API.",
    )
    ollama_model: str = Field(
        default="qwen2.5:7b",
        description="Name of the Ollama model to use for extraction.",
    )
    ollama_timeout: int = Field(
        default=90,
        description="LLM request timeout in seconds.",
    )
    ollama_num_gpu: int = Field(
        default=0,
        description="Number of GPU layers to offload to Ollama (0 = CPU only, prevents CUDA kernel crashes on incompatible laptop drivers).",
    )
    ollama_num_ctx: int = Field(
        default=4096,
        description="Context window size in tokens for Ollama (balanced for CPU speed and document coverage).",
    )
    ollama_num_predict: int = Field(
        default=800,
        description=(
            "Maximum tokens the model may generate. "
            "The full JSON response (6 fields + evidence + confidence) needs ~600-900 tokens; "
            "setting this below that causes truncation and empty-response errors."
        ),
    )
    ollama_keep_alive: str = Field(
        default="30m",
        description="Keep model loaded in RAM to prevent repeated disk-loading cold starts.",
    )
    ollama_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM extraction (small nonzero value prevents token repeat loops).",
    )
    ollama_repeat_penalty: float = Field(
        default=1.1,
        ge=0.0,
        le=2.0,
        description="Repetition penalty for Ollama generation to avoid cyclic token loops.",
    )

    # ── Upload limits ─────────────────────────────────────────
    max_upload_size_mb: int = Field(
        default=20,
        description="Maximum file size in MB per upload.",
    )

    # ── Few-shot ──────────────────────────────────────────────
    few_shot_enabled: bool = Field(
        default=True,
        description="Whether to prepend few-shot examples to the extraction prompt.",
    )
    few_shot_max_examples: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum number of few-shot examples to include (1-2 is optimal for CPU speed).",
    )

    # ── Confidence thresholds ─────────────────────────────────
    low_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence below this level triggers a UI warning.",
    )
    review_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Confidence below this level is flagged for manual review.",
    )

    # ── Data Contract & Extraction Conventions ─────────────────
    date_format: str = Field(
        default="DD.MM.YYYY",
        description="Standard date representation format expected by the dataset.",
    )
    lease_duration_months_default: int = Field(
        default=11,
        description="Standard lease duration in months when duration is not explicitly stated otherwise.",
    )
    calendar_day_counting_rule: str = Field(
        default="month_end_last_day_and_mid_month_day_minus_one",
        description=(
            "Calendar-month convention: leases starting on day 1 (01.MM.YYYY) end on the last calendar day "
            "of the concluding month (e.g. 01.04.2010 -> 31.03.2011); leases starting on day D > 1 end on day D - 1 "
            "(e.g. 15.12.2012 -> 14.11.2013, 07.07.2013 -> 06.06.2014)."
        ),
    )
    notice_period_days_per_month: int = Field(
        default=30,
        description="Conversion ratio of months to days for termination/renewal notice (1 month = 30 days).",
    )
    monetary_conflict_priority: str = Field(
        default="words",
        description=(
            "Policy when numeric amount conflicts with words: 'words' prioritizes amount written in words "
            "per project policy; 'numeric' prioritizes numeric amount. In both cases, conflict is flagged with confidence <= 0.40."
        ),
    )
    party_name_exclude_honorifics: bool = Field(
        default=True,
        description="Whether to strip honorifics (Mr., Mrs., Sri, etc.) to match canonical dataset labels.",
    )
    party_name_exclude_corporate_rep: bool = Field(
        default=True,
        description="Whether to strip corporate representative clauses (e.g. ', represented by Mr. XYZ').",
    )

    # ── Logging ───────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG | INFO | WARNING | ERROR.",
    )

    # ── Dataset & Output Paths ───────────────────────────────
    data_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data",
        description="Path to the directory containing dataset files.",
    )
    train_csv: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "train.csv",
        description="Path to the training metadata CSV file.",
    )
    test_csv: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "test.csv",
        description="Path to the test ground-truth CSV file.",
    )
    train_docs_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "train",
        description="Directory containing training documents.",
    )
    test_docs_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "test",
        description="Directory containing test documents.",
    )
    outputs_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "outputs",
        description="Directory where predictions and evaluation reports are written.",
    )

    # ── Derived helpers ───────────────────────────────────────
    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()
