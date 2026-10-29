"""
app/utils/file_validation.py
─────────────────────────────
Safe file-upload validation utilities.

Rules enforced:
  • Only DOCX, PNG, JPG/JPEG extensions are accepted.
  • File size must not exceed the configured limit.
  • The file must be non-empty and the bytes must be readable.
  • Magic-byte checks are performed so a renamed file cannot bypass
    the extension check.
  • No instructions found inside an uploaded document are ever executed.
"""

from __future__ import annotations

import io
import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from fastapi import UploadFile, HTTPException

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Allowed extensions & their magic bytes ────────────────────
ALLOWED_EXTENSIONS = {".docx", ".png", ".jpg", ".jpeg"}

# Magic bytes for supported types
_MAGIC: dict[str, list[bytes]] = {
    ".docx": [b"PK\x03\x04"],                         # ZIP-based
    ".png":  [b"\x89PNG\r\n\x1a\n"],
    ".jpg":  [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
}


def _check_magic(data: bytes, extension: str) -> bool:
    """Return True if *data* starts with a known magic byte sequence."""
    candidates = _MAGIC.get(extension, [])
    return any(data.startswith(magic) for magic in candidates)


def validate_upload(upload: UploadFile) -> None:
    """
    Validate *upload* in-place; raise HTTPException on failure.

    Parameters
    ----------
    upload:
        The FastAPI UploadFile object received from the multipart request.

    Raises
    ------
    HTTPException
        400 if the file is invalid; 413 if too large.
    """
    settings = get_settings()

    # ── Extension check ───────────────────────────────────────
    filename = upload.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )

    # ── Size check ────────────────────────────────────────────
    # FastAPI / Starlette don't guarantee .size is populated before
    # reading; we read the bytes and check length ourselves.
    # (Content-Length from headers can be spoofed, so we ignore it.)
    pass  # size is validated after reading in validate_file_bytes()


def validate_file_bytes(data: bytes, filename: str) -> None:
    """
    Validate raw *data* for size, emptiness and magic bytes.

    Parameters
    ----------
    data:
        The complete file content read from the upload.
    filename:
        The original filename (used for extension lookup and logging).

    Raises
    ------
    HTTPException
        400 / 413 for validation failures.
    """
    settings = get_settings()
    suffix = Path(filename).suffix.lower()

    # ── Empty file ────────────────────────────────────────────
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Size limit ────────────────────────────────────────────
    if len(data) > settings.max_upload_size_bytes:
        limit_mb = settings.max_upload_size_mb
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {limit_mb} MB upload limit.",
        )

    # ── Magic-byte check ──────────────────────────────────────
    if not _check_magic(data, suffix):
        raise HTTPException(
            status_code=400,
            detail=(
                f"File '{filename}' does not appear to be a valid {suffix} file "
                "(magic bytes mismatch). The file may be corrupt or misnamed."
            ),
        )


@contextmanager
def temp_file_from_bytes(
    data: bytes,
    suffix: str,
) -> Generator[Path, None, None]:
    """
    Context manager that writes *data* to a NamedTemporaryFile and yields
    its Path. The file is deleted when the context exits.

    Parameters
    ----------
    data:
        Raw file bytes to write.
    suffix:
        File extension including the leading dot (e.g. '.docx').

    Yields
    ------
    Path
        Path to the temporary file.
    """
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, prefix="docmeta_"
    )
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        yield Path(tmp.name)
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not delete temp file %s: %s", tmp.name, exc)
