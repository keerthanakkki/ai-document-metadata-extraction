"""
app/services/document_reader.py
────────────────────────────────
Stage B: Text extraction from DOCX files.

Responsibilities
────────────────
• Extract all textual content from a DOCX file using python-docx.
• Preserve logical reading order: paragraphs → tables → headers/footers.
• Return a single clean string for downstream AI extraction.
• Raise clear errors for corrupt or unreadable files.
• Do NOT perform any semantic interpretation here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text_from_docx(file_path: Path) -> str:
    """
    Extract all text from *file_path* (a DOCX file).

    The extraction order is:
      1. Body paragraphs (preserving sequence).
      2. Table cells (row-major order).
      3. Header paragraphs from each section.
      4. Footer paragraphs from each section.

    Parameters
    ----------
    file_path:
        Absolute path to the DOCX file.

    Returns
    -------
    str
        Full document text, sections separated by blank lines.

    Raises
    ------
    ValueError
        If the file cannot be opened or contains no readable text.
    """
    try:
        from docx import Document  # python-docx
        from docx.oxml.ns import qn
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for DOCX extraction. "
            "Install it with: pip install python-docx"
        ) from exc

    file_path = Path(file_path)
    try:
        doc = Document(str(file_path))
    except Exception as exc:
        raise ValueError(
            f"Could not open DOCX file '{file_path.name}': {exc}"
        ) from exc

    parts: list[str] = []

    # ── 1. Body paragraphs ────────────────────────────────────
    body_lines: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            body_lines.append(text)

    if body_lines:
        parts.append("\n".join(body_lines))

    # ── 2. Tables ─────────────────────────────────────────────
    table_parts: list[str] = []
    for table in doc.tables:
        rows: list[str] = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            # Deduplicate merged cells (python-docx repeats them)
            seen: list[str] = []
            for c in cells:
                if not seen or c != seen[-1]:
                    seen.append(c)
            row_text = " | ".join(c for c in seen if c)
            if row_text:
                rows.append(row_text)
        if rows:
            table_parts.append("\n".join(rows))

    if table_parts:
        parts.append("\n\n".join(table_parts))

    # ── 3 & 4. Headers and footers ────────────────────────────
    hf_lines: list[str] = []
    for section in doc.sections:
        for hf in [section.header, section.footer]:
            if hf is None:
                continue
            for para in hf.paragraphs:
                text = para.text.strip()
                if text:
                    hf_lines.append(text)

    if hf_lines:
        parts.append("\n".join(hf_lines))

    full_text = "\n\n".join(parts).strip()

    if not full_text:
        raise ValueError(
            f"No readable text could be extracted from '{file_path.name}'. "
            "The file may be empty or contain only images/embedded objects."
        )

    logger.debug(
        "DOCX '%s': extracted %d characters", file_path.name, len(full_text)
    )
    return full_text
