"""
app/services/ocr_service.py
────────────────────────────
Stage B (image path): OCR text extraction using EasyOCR.

Responsibilities
────────────────
• Initialise EasyOCR once (lazy singleton, GPU optional).
• Accept a Path to a PNG / JPG / JPEG image.
• Return the extracted text with bounding-box details preserved for
  debugging.
• Raise clear errors when the image is unreadable or OCR yields
  insufficient text.
• Keep this stage completely separate from semantic extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── EasyOCR reader singleton ──────────────────────────────────
_reader: Optional[object] = None   # type: easyocr.Reader at runtime


def _get_reader():
    """Return the EasyOCR Reader, initialising it on first call."""
    global _reader
    if _reader is None:
        try:
            import easyocr  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "easyocr is required for image OCR. "
                "Install it with: pip install easyocr"
            ) from exc

        logger.info("Initialising EasyOCR reader (this may take a moment)…")
        # Use CPU to avoid CUDA dependency issues; set gpu=True if available.
        _reader = easyocr.Reader(
            ["en"],
            gpu=False,
            verbose=False,
        )
        logger.info("EasyOCR reader ready.")
    return _reader


def extract_text_from_image(file_path: Path) -> str:
    """
    Run OCR on *file_path* and return the recognised text.

    Parameters
    ----------
    file_path:
        Absolute path to a PNG / JPG / JPEG image.

    Returns
    -------
    str
        All recognised text joined by newlines.

    Raises
    ------
    ValueError
        If the image cannot be read or OCR recognises fewer than 10
        characters (indicating an effectively blank or unreadable scan).
    RuntimeError
        If EasyOCR itself raises an unexpected error.
    """
    if not file_path.exists():
        raise ValueError(f"Image file not found: {file_path}")

    reader = _get_reader()

    try:
        results = reader.readtext(str(file_path), detail=1)
    except Exception as exc:
        raise RuntimeError(
            f"EasyOCR failed on '{file_path.name}': {exc}"
        ) from exc

    if not results:
        raise ValueError(
            f"EasyOCR returned no results for '{file_path.name}'. "
            "The image may be blank, too dark, or rotated beyond recovery."
        )

    # Group detections into lines using 2D spatial reading order (top-to-bottom, left-to-right)
    # Each result is: (bbox, text, confidence)
    boxes = []
    for bbox, text, conf in results:
        text = text.strip()
        if not text:
            continue
        ys = [pt[1] for pt in bbox]
        xs = [pt[0] for pt in bbox]
        h = max(ys) - min(ys)
        boxes.append({
            "bbox": bbox,
            "text": text,
            "conf": conf,
            "y_mid": sum(ys) / 4.0,
            "x_min": min(xs),
            "h": h if h > 5 else 15.0,
        })

    boxes.sort(key=lambda b: b["y_mid"])

    lines: list[str] = []
    current_line: list[dict] = []
    current_y: Optional[float] = None
    current_h: float = 15.0

    for b in boxes:
        threshold = max(current_h, b["h"]) * 0.65
        if current_y is None or abs(b["y_mid"] - current_y) <= threshold:
            current_line.append(b)
            current_y = sum(x["y_mid"] for x in current_line) / len(current_line)
            current_h = sum(x["h"] for x in current_line) / len(current_line)
        else:
            current_line.sort(key=lambda x: x["x_min"])
            lines.append(" ".join(x["text"] for x in current_line))
            current_line = [b]
            current_y = b["y_mid"]
            current_h = b["h"]

    if current_line:
        current_line.sort(key=lambda x: x["x_min"])
        lines.append(" ".join(x["text"] for x in current_line))

    full_text = "\n".join(lines).strip()

    if len(full_text) < 10:
        raise ValueError(
            f"OCR extracted only {len(full_text)} characters from "
            f"'{file_path.name}'. The scan may be too low-quality to process."
        )

    logger.debug(
        "OCR '%s': extracted %d characters from %d lines (%d regions)",
        file_path.name,
        len(full_text),
        len(lines),
        len(results),
    )
    return full_text


def extract_text_from_image_with_details(
    file_path: Path,
) -> dict:
    """
    Like :func:`extract_text_from_image` but also returns per-region
    detail (bounding box, text, confidence score) for debugging.

    Returns
    -------
    dict with keys:
        • ``text``    – full combined text string
        • ``regions`` – list of {bbox, text, confidence} dicts
    """
    if not file_path.exists():
        raise ValueError(f"Image file not found: {file_path}")

    reader = _get_reader()

    try:
        results = reader.readtext(str(file_path), detail=1)
    except Exception as exc:
        raise RuntimeError(
            f"EasyOCR failed on '{file_path.name}': {exc}"
        ) from exc

    regions = []
    lines: list[str] = []
    for bbox, text, conf in sorted(
        results, key=lambda r: min(pt[1] for pt in r[0])
    ):
        text = text.strip()
        if text:
            lines.append(text)
            regions.append({"bbox": bbox, "text": text, "confidence": round(conf, 4)})

    full_text = "\n".join(lines).strip()
    return {"text": full_text, "regions": regions}
