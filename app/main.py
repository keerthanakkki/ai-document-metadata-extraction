"""
app/main.py
────────────
FastAPI application entry point.

Usage (from project root):
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings

# ── Logging setup ─────────────────────────────────────────────
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load few-shot examples at startup so the first request is fast."""
    logger.info("Starting AI Document Metadata Extraction API…")
    logger.info("Configured model: %s @ %s", settings.ollama_model, settings.ollama_base_url)

    if settings.few_shot_enabled:
        try:
            from app.services.extraction_service import build_few_shot_examples
            examples = build_few_shot_examples()
            logger.info("Few-shot cache: %d example(s) loaded.", len(examples))
        except Exception as exc:
            logger.warning("Could not pre-load few-shot examples: %s", exc)

    yield  # application is running

    logger.info("Shutting down API.")


# ── Application factory ───────────────────────────────────────
app = FastAPI(
    title="AI Document Metadata Extraction API",
    description=(
        "Extracts six metadata fields from rental agreement documents "
        "(DOCX and scanned images) using OCR and an Ollama LLM.\n\n"
        "**Fields extracted:** Agreement Value, Start Date, End Date, "
        "Renewal Notice Days, Party One, Party Two."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────
app.include_router(router, prefix="")

# ── Root redirect ─────────────────────────────────────────────
from fastapi.responses import RedirectResponse


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")
