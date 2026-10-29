"""
tests/conftest.py
──────────────────
Shared pytest fixtures used across all test modules.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Project root on path ──────────────────────────────────────
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────
# Auto-clear lru_cache between tests to prevent OCR/LLM leaks
# ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_few_shot_cache():
    """Clear build_few_shot_examples lru_cache before each test."""
    from app.services.extraction_service import build_few_shot_examples
    build_few_shot_examples.cache_clear()
    yield
    build_few_shot_examples.cache_clear()


# ─────────────────────────────────────────────────────────────
# Minimal valid file helpers
# ─────────────────────────────────────────────────────────────

def make_minimal_docx(text: str = "Hello world rental agreement") -> bytes:
    """Build an in-memory minimal DOCX (ZIP + word/document.xml)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # [Content_Types].xml
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Override PartName="/word/document.xml"'
            ' ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        # _rels/.rels
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
            ' Target="word/document.xml"/>'
            "</Relationships>",
        )
        # word/_rels/document.xml.rels
        zf.writestr(
            "word/_rels/document.xml.rels",
            '<?xml version="1.0"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            "</Relationships>",
        )
        # word/document.xml
        doc_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body>"
            "<w:p><w:r><w:t>" + text + "</w:t></w:r></w:p>"
            "</w:body>"
            "</w:document>"
        )
        zf.writestr("word/document.xml", doc_xml)
    return buf.getvalue()


def make_minimal_png() -> bytes:
    """Return a 1×1 white PNG."""
    # Minimal valid PNG binary
    png = (
        b"\x89PNG\r\n\x1a\n"           # signature
        b"\x00\x00\x00\rIHDR"          # IHDR chunk length + type
        b"\x00\x00\x00\x01"            # width=1
        b"\x00\x00\x00\x01"            # height=1
        b"\x08\x02"                    # bit depth=8, colour type=2 (RGB)
        b"\x00\x00\x00"                # compression, filter, interlace
        b"\x90wS\xde"                  # CRC
        b"\x00\x00\x00\x0cIDATx\x9c"  # IDAT
        b"b\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"  # compressed data
        b"\x00\x00\x00\x00IEND\xaeB`\x82"  # IEND
    )
    return png


def make_minimal_jpeg() -> bytes:
    """Return minimal valid JPEG bytes."""
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xd9"
    )


# ─────────────────────────────────────────────────────────────
# Sample extraction result dicts
# ─────────────────────────────────────────────────────────────

SAMPLE_EXTRACTION = {
    "agreement_value": "6500",
    "agreement_start_date": "20.05.2007",
    "agreement_end_date": "20.05.2008",
    "renewal_notice_days": 15,
    "party_one": "Antonio Levy S. Ingles, Jr.",
    "party_two": "GERALDINE Q. GALINATO",
    "status": "success",
    "error_message": None,
    "document_text_preview": "This agreement is made between…",
    "field_details": {
        "agreement_value": {
            "value": "6500",
            "confidence_score": 0.95,
            "confidence_level": "high",
            "evidence": "Monthly rent: Rs. 6500",
            "is_missing": False,
        },
        "agreement_start_date": {
            "value": "20.05.2007",
            "confidence_score": 0.90,
            "confidence_level": "high",
            "evidence": "commencing from 20.05.2007",
            "is_missing": False,
        },
        "agreement_end_date": {
            "value": "20.05.2008",
            "confidence_score": 0.90,
            "confidence_level": "high",
            "evidence": "expiring on 20.05.2008",
            "is_missing": False,
        },
        "renewal_notice_days": {
            "value": "15",
            "confidence_score": 0.80,
            "confidence_level": "high",
            "evidence": "15 days notice",
            "is_missing": False,
        },
        "party_one": {
            "value": "Antonio Levy S. Ingles, Jr.",
            "confidence_score": 0.95,
            "confidence_level": "high",
            "evidence": "The Lessor: Antonio Levy S. Ingles, Jr.",
            "is_missing": False,
        },
        "party_two": {
            "value": "GERALDINE Q. GALINATO",
            "confidence_score": 0.95,
            "confidence_level": "high",
            "evidence": "The Lessee: GERALDINE Q. GALINATO",
            "is_missing": False,
        },
    },
}

SAMPLE_LLM_JSON = {
    "agreement_value": "6500",
    "agreement_start_date": "20.05.2007",
    "agreement_end_date": "20.05.2008",
    "renewal_notice_days": 15,
    "party_one": "Antonio Levy S. Ingles, Jr.",
    "party_two": "GERALDINE Q. GALINATO",
    "evidence": {
        "agreement_value": "Monthly rent Rs. 6500",
        "agreement_start_date": "commencing from 20.05.2007",
        "agreement_end_date": "expiring 20.05.2008",
        "renewal_notice_days": "15 days prior notice",
        "party_one": "Lessor: Antonio Levy S. Ingles, Jr.",
        "party_two": "Lessee: GERALDINE Q. GALINATO",
    },
    "confidence": {
        "agreement_value": 0.95,
        "agreement_start_date": 0.90,
        "agreement_end_date": 0.90,
        "renewal_notice_days": 0.80,
        "party_one": 0.95,
        "party_two": 0.95,
    },
}


# ─────────────────────────────────────────────────────────────
# FastAPI test client fixture (model mocked out)
# ─────────────────────────────────────────────────────────────

@pytest.fixture()
def api_client():
    """Return a TestClient with the LLM call mocked."""
    from app.main import app
    from app.schemas import ExtractionResult, ExtractionStatus, FieldDetail

    mock_result = ExtractionResult(
        agreement_value="6500",
        agreement_start_date="20.05.2007",
        agreement_end_date="20.05.2008",
        renewal_notice_days=15,
        party_one="Antonio Levy S. Ingles, Jr.",
        party_two="GERALDINE Q. GALINATO",
        field_details={
            k: FieldDetail(value=str(v), confidence_score=0.9, is_missing=False)
            for k, v in {
                "agreement_value": "6500",
                "agreement_start_date": "20.05.2007",
                "agreement_end_date": "20.05.2008",
                "renewal_notice_days": 15,
                "party_one": "Antonio Levy S. Ingles, Jr.",
                "party_two": "GERALDINE Q. GALINATO",
            }.items()
        },
        status=ExtractionStatus.SUCCESS,
    )

    with patch(
        "app.services.extraction_service.call_llm",
        return_value=MagicMock(
            agreement_value="6500",
            agreement_start_date="20.05.2007",
            agreement_end_date="20.05.2008",
            renewal_notice_days=15,
            party_one="Antonio Levy S. Ingles, Jr.",
            party_two="GERALDINE Q. GALINATO",
            evidence={},
            confidence={},
        ),
    ), patch(
        "app.services.extraction_service.build_few_shot_examples",
        return_value=[],
    ):
        client = TestClient(app, raise_server_exceptions=False)
        yield client
