"""
app/services/extraction_service.py
────────────────────────────────────
Orchestrates the complete document-processing pipeline:

  Stage A  →  file validation  (app/utils/file_validation.py)
  Stage B  →  text extraction  (document_reader / ocr_service)
  Stage C  →  AI extraction    (llm_service)
  Stage D  →  few-shot context (built from training dataset)
  Stage E  →  structured output (schemas.ExtractionResult)

Entry points
────────────
  extract_from_bytes(data, filename) → ExtractionResult
      Main pipeline; accepts raw file bytes.

  build_few_shot_examples() → list[dict]
      Reads train.csv + training documents to build the few-shot set.
      Called once at startup and cached.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from app.config import get_settings
from app.schemas import ExtractionResult, ExtractionStatus, FieldDetail
from app.services.document_reader import extract_text_from_docx
from app.services.ocr_service import extract_text_from_image
from app.services.llm_service import (
    ModelUnavailableError,
    ExtractionParseError,
    RawLLMOutput,
    call_llm,
)
from app.utils.file_validation import validate_file_bytes, temp_file_from_bytes

logger = logging.getLogger(__name__)

# ── Field definitions used throughout ────────────────────────
FIELD_NAMES = [
    "agreement_value",
    "agreement_start_date",
    "agreement_end_date",
    "renewal_notice_days",
    "party_one",
    "party_two",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DOCX_EXTENSIONS = {".docx"}


# ─────────────────────────────────────────────────────────────
# Stage D: few-shot example builder
# ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def build_few_shot_examples() -> list[dict]:
    """
    Read train.csv and the corresponding training documents to build a
    set of few-shot examples.

    Each example is a dict:
        {
          "document_text": "<extracted text>",
          "expected_output": {<six field dict>}
        }

    Rules
    ─────
    • Uses the TRAINING dataset only. Never touches test documents.
    • Selects up to settings.few_shot_max_examples diverse examples.
    • Silently skips training documents that cannot be read.

    Returns
    -------
    list[dict]
        Ready-to-use few-shot examples, or an empty list if
        the training data cannot be loaded.
    """
    settings = get_settings()
    if not settings.few_shot_enabled:
        return []

    try:
        df = pd.read_csv(settings.train_csv)
    except Exception as exc:
        logger.warning("Could not load train.csv for few-shot: %s", exc)
        return []

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]
    col_map = {
        "File Name": "filename",
        "Aggrement Value": "agreement_value",
        "Aggrement Start Date": "agreement_start_date",
        "Aggrement End Date": "agreement_end_date",
        "Renewal Notice (Days)": "renewal_notice_days",
        "Party One": "party_one",
        "Party Two": "party_two",
    }
    df.rename(columns=col_map, inplace=True)
    df.dropna(subset=["filename"], inplace=True)

    examples: list[dict] = []
    train_dir = settings.train_docs_dir

    overlapping_filenames: set[str] = set()
    try:
        from app.services.evaluation_service import detect_overlap
        overlapping_filenames = set(detect_overlap(str(settings.train_csv), str(settings.test_csv)))
    except Exception as exc:
        logger.debug("Could not determine overlapping filenames: %s", exc)

    for _, row in df.iterrows():
        if len(examples) >= settings.few_shot_max_examples:
            break

        filename: str = str(row["filename"]).strip()

        # Strict isolation: NEVER use overlapping test documents in few-shot training examples
        if filename in overlapping_filenames or "24158401" in filename:
            logger.info("Few-shot: skipping overlapping document '%s' to avoid data leakage", filename)
            continue

        # Find the matching file in the train directory
        found_path: Optional[Path] = None
        for ext in [".docx", ".png", ".jpg", ".jpeg"]:
            candidate = train_dir / (filename + ext)
            if candidate.exists():
                found_path = candidate
                break

        if found_path is None:
            logger.debug("Few-shot: no file found for '%s'", filename)
            continue

        try:
            doc_text = _extract_text_from_path(found_path)
        except Exception as exc:
            logger.warning("Few-shot: could not read '%s': %s", filename, exc)
            continue

        expected = {
            "agreement_value": _str_or_none(row.get("agreement_value")),
            "agreement_start_date": _str_or_none(row.get("agreement_start_date")),
            "agreement_end_date": _str_or_none(row.get("agreement_end_date")),
            "renewal_notice_days": _int_or_none(row.get("renewal_notice_days")),
            "party_one": clean_party_name(_str_or_none(row.get("party_one"))),
            "party_two": clean_party_name(_str_or_none(row.get("party_two"))),
        }

        examples.append(
            {"document_text": doc_text[:3500], "expected_output": expected}
        )
        logger.debug("Few-shot: added example from '%s'", filename)

    logger.info("Built %d few-shot example(s) from training data.", len(examples))
    return examples


# ─────────────────────────────────────────────────────────────
# Internal helpers & parsing utilities
# ─────────────────────────────────────────────────────────────


def _extract_text_from_path(path: Path) -> str:
    """Route extraction to DOCX reader or OCR based on extension."""
    ext = path.suffix.lower()
    if ext in DOCX_EXTENSIONS:
        return extract_text_from_docx(path)
    elif ext in IMAGE_EXTENSIONS:
        return extract_text_from_image(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def _str_or_none(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and __import__("math").isnan(value)):
        return None
    s = str(value).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def _int_or_none(value) -> Optional[int]:
    if value is None:
        return None
    try:
        f = float(str(value))
        if __import__("math").isnan(f):
            return None
        return int(f)
    except (ValueError, TypeError):
        return None


import calendar
import re


def clean_party_name(name: Optional[str], strip_honorifics: bool = False) -> Optional[str]:
    """
    Ensure parental or relational designations (S/O, D/O, W/O, Slo, son of, etc.),
    corporate representation clauses (represented by Mr...), and optionally honorifics
    are excluded from party names.

    Examples:
      'Sri Vishal Bhardwaj S/O Charnel Singh' -> 'Sri Vishal Bhardwaj'
      'V.K.NATARAJ, son of V.KANDASWAMI CHETTIAR' -> 'V.K.NATARAJ'
      'Mr.B.Kishore , S/o. Mr. B. Pampaiah' -> 'Mr.B.Kishore'
      'SRI VYSHNAVI DAIRY SPECIALITIES Private Ltd., represented by Mr. RAJESH CHAVDA' -> 'SRI VYSHNAVI DAIRY SPECIALITIES Private Ltd.'
    """
    if name is None:
        return None
    s = str(name).strip()
    if not s or s.lower() in ("none", "null", "nan"):
        return None

    # Exclude parentage introduced by S/O, D/O, W/O, Slo, Dlo, Wlo, son of, daughter of, wife of, c/o, etc.
    pattern = r"[\s,]+(?:S/O|D/O|W/O|Slo|Dlo|Wlo|S/o|D/o|W/o|s/o|d/o|w/o|S\.O\.|D\.O\.|W\.O\.|S\.L\.|s\.o\.|son\s+of|daughter\s+of|wife\s+of|c/o|C/O)(?:[\s\.\b]|$).*$"
    cleaned = re.sub(pattern, "", s, flags=re.IGNORECASE).strip()

    # Exclude corporate representation clauses (e.g. ", represented by Mr. XYZ")
    cleaned = re.sub(
        r"[\s,]+(?:represented\s+by|rep\.?\s+by|through\s+its|auth(?:\.|orized)?\s+signatory).*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    if strip_honorifics:
        cleaned = re.sub(r"^(?:Mr|Mrs|Ms|Miss|Dr|Sri|Shri|Smt)\b\s*(\.?\s*)", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^SRI\s+", "", cleaned, flags=re.IGNORECASE).strip()

    cleaned = cleaned.rstrip(",;:- ")
    return cleaned if cleaned else s


def resolve_party_one(party_one: Optional[str], doc_text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Ensure Party One is the Lessor/Landlord from contractual role rather than an incidental
    locality name from the property schedule (e.g. 'Ram Murthy' from 'Ram murthy nagar').

    Returns (resolved_name, evidence_snippet).
    """
    cleaned = clean_party_name(party_one)
    is_locality = False
    if cleaned:
        # Check if this name matches an incidental locality/street/colony in the text
        loc_pattern = rf"\b{re.escape(cleaned)}\s*(?:nagar|colony|layout|road|street|extn|extension|palya|halli|puram)\b"
        if re.search(loc_pattern, doc_text, re.IGNORECASE):
            is_locality = True

    if is_locality or not cleaned:
        # Search the opening preamble for the actual Lessor / First Party
        preamble_match = re.search(
            r"(?:between\s*:\s*\n?\s*|\bLESSOR\b.*?between\s*:?\s*)((?:(?:Mr|Mrs|Ms|Sri|Smt)\.?\s+)?[A-Za-z\s\.\-]+?)(?:,|\n|Hno|residing|hereinafter|LESSOR)",
            doc_text,
            re.IGNORECASE,
        )
        if preamble_match:
            candidate = clean_party_name(preamble_match.group(1))
            if candidate and len(candidate) > 2 and candidate.lower() != "the":
                return candidate, f"Extracted Lessor from preamble: {candidate}"

    return cleaned, None


def parse_words_to_number(text: str) -> Optional[int]:
    """
    Convert written English numbers (e.g. 'Nine thousand and two hundred') into integer.
    """
    units = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    }
    scales = {"thousand": 1000, "lakh": 100000, "crore": 10000000}
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    total = 0
    current = 0
    found = False
    for w in words:
        if w in units:
            current += units[w]
            found = True
        elif w == "hundred":
            current = (current if current > 0 else 1) * 100
            found = True
        elif w in scales:
            current = (current if current > 0 else 1) * scales[w]
            total += current
            current = 0
            found = True
    total += current
    return total if found else None


def resolve_agreement_value_and_conflict(
    raw_value: Optional[str],
    raw_evidence: Optional[str],
    raw_confidence: Optional[float],
    doc_text: str,
) -> tuple[Optional[str], Optional[float], Optional[str], bool]:
    """
    Inspect rent text in doc_text and LLM evidence. Retain both numeric and words amounts.
    If they conflict, set confidence <= 0.40 and document the conflict clearly in evidence
    so it is flagged for human review. Never conflate monthly rent with security deposit.
    Supports configurable monetary conflict priority ('words' vs 'numeric').

    Returns: (value_str, confidence_score, evidence_str, has_conflict)
    """
    settings = get_settings()

    # Pre-clean OCR digit artifacts in doc_text (e.g. Rs. 12, OOO -> Rs. 12,000)
    cleaned_doc_text = re.sub(r"(\d+)\s*,\s*([oO0]{3})\b", r"\g<1>000", doc_text)

    # Extract only the rent clause to prevent cross-contamination with security deposit / advance
    m_sec = re.search(
        r"(?:monthly\s+rent|RENT|rent)\b.*?(?=\b(?:ELECTRICITY|DEPOSIT|advance|advance\s+shall|INTERNAL|MAINTENANCE|USER|TERMNNA?TION|\n[A-Z\s]{4,}:)|$)",
        cleaned_doc_text,
        re.IGNORECASE | re.DOTALL,
    )
    rent_sec = m_sec.group(0) if m_sec else cleaned_doc_text

    numeric_val: Optional[int] = None
    words_str: Optional[str] = None
    words_val: Optional[int] = None

    # Try to extract written words in rent clause
    words_match = re.search(
        r"\(\s*([A-Za-z\s]+?\b(?:thousand|hundred|lakh)\b[A-Za-z\s]*?)\s*(?:only|\))",
        rent_sec,
        re.IGNORECASE,
    )
    if words_match:
        words_str = words_match.group(1).strip()
        words_val = parse_words_to_number(words_str)

    # Try to extract numeric amount from rent clause
    num_match = re.search(
        r"(?:Rs\.?|INR|\bRs\b)\s*([0-9]{1,2},[0-9]{3}|[0-9]{4,6})",
        rent_sec,
        re.IGNORECASE,
    )
    if num_match:
        cleaned_num = re.sub(r"[,\s]", "", num_match.group(1))
        numeric_val = int(cleaned_num)
    elif "000" in rent_sec:
        # Check if rent clause has '_ 000' (OCR artifact where leading digit was degraded)
        if words_val and words_val % 1000 != 0:
            numeric_val = (words_val // 1000) * 1000
        elif raw_value and raw_value.isdigit() and int(raw_value) >= 1000:
            numeric_val = int(raw_value)

    has_conflict = False
    if numeric_val is not None and words_val is not None and numeric_val != words_val:
        has_conflict = True
    elif raw_value and words_val and raw_value.isdigit() and int(raw_value) != words_val:
        has_conflict = True
        if numeric_val is None:
            numeric_val = int(raw_value)

    if has_conflict:
        num_repr = f"Rs. {numeric_val:,}" if numeric_val is not None else "Unknown"
        words_repr = words_str if words_str else f"{words_val}"
        ev_str = (
            f"Numeric: {num_repr} | In words: {words_repr} "
            f"[CONFLICT DETECTED: numeric amount ({num_repr}) differs from amount written in words ({words_repr}); flagged for review]"
        )
        priority = getattr(settings, "monetary_conflict_priority", "words").lower()
        if priority == "words" and words_val is not None:
            chosen_val = str(words_val)
        else:
            chosen_val = str(numeric_val) if numeric_val is not None else (raw_value or str(words_val))
        return chosen_val, 0.40, ev_str, True

    # No conflict
    chosen_val = raw_value if raw_value else (str(numeric_val) if numeric_val else None)
    return chosen_val, (raw_confidence if raw_confidence is not None else 0.95), raw_evidence, False


def calculate_calendar_end_date(
    start_date_str: Optional[str],
    raw_end_date: Optional[str] = None,
    duration_months: int = 11,
    doc_text: Optional[str] = None,
) -> Optional[str]:
    """
    Calendar-month calculation for agreement end date based on start date and duration.
    Follows the assignment date-counting convention:
    - Leases starting on day 1 (01.MM.YYYY) end on the LAST CALENDAR DAY of the concluding month
      (e.g. 01.04.2010 -> 31.03.2011 because March has 31 days).
    - Leases starting on day D > 1 end on day D - 1 (e.g. 15.12.2012 -> 14.11.2013).
    """
    if not start_date_str:
        return raw_end_date

    # Parse start date
    m = re.match(r"^(\d{1,2})[\.\/\-](\d{1,2})[\.\/\-](\d{4})$", start_date_str.strip())
    if not m:
        return raw_end_date

    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))

    # Check if document explicitly states a different duration
    if doc_text:
        dur_m = re.search(r"(\d+|eleven|twelve|ten)\s+months", doc_text, re.IGNORECASE)
        if dur_m:
            word_map = {"ten": 10, "eleven": 11, "twelve": 12}
            d_str = dur_m.group(1).lower()
            duration_months = word_map.get(d_str, int(d_str) if d_str.isdigit() else 11)

    # Check if raw_end_date was returned and check for off-by-one errors on month end or mid-month
    if raw_end_date:
        raw_m = re.match(r"^(\d{1,2})[\.\/\-](\d{1,2})[\.\/\-](\d{4})$", raw_end_date.strip())
        if raw_m:
            r_day, r_month, r_year = int(raw_m.group(1)), int(raw_m.group(2)), int(raw_m.group(3))
            last_day_of_raw_month = calendar.monthrange(r_year, r_month)[1]
            # If start date is day 1, and raw end date was day 30 for a 31-day month, fix off-by-one error
            if day == 1 and r_day == (last_day_of_raw_month - 1) and last_day_of_raw_month == 31:
                return f"{last_day_of_raw_month:02d}.{r_month:02d}.{r_year}"
            # If start date is day D > 1, and raw end date has r_day == day (e.g. 15.11.2013 instead of 14.11.2013)
            # and that exact date string does not appear verbatim in the document, fix the day D -> day D - 1 calculation!
            if day > 1 and r_day == day:
                if not doc_text or raw_end_date.strip() not in doc_text:
                    return f"{(day - 1):02d}.{r_month:02d}.{r_year}"
            return raw_end_date


    # Compute calendar end date
    total_months = (month - 1) + duration_months
    end_year = year + (total_months // 12)
    end_month = (total_months % 12) + 1

    if day == 1:
        last_day = calendar.monthrange(end_year, end_month)[1]
        return f"{last_day:02d}.{end_month:02d}.{end_year}"
    else:
        end_day = day - 1
        return f"{end_day:02d}.{end_month:02d}.{end_year}"


def parse_renewal_notice_days(value, doc_text: Optional[str] = None) -> Optional[int]:
    """
    Interpret renewal notice days according to the documented conversion convention:
    - 1 month = 30 days (e.g. 'two months' / '2 months notice' -> 60, '1 month' / 'on month notice' -> 30)
    - Direct integers and day strings (e.g. 15, '15 days', '60')
    """
    if value is not None:
        import math

        if isinstance(value, (int, float)):
            try:
                f = float(value)
                return None if math.isnan(f) else int(f)
            except (ValueError, TypeError):
                pass

        val_str = str(value).strip().lower()
        if val_str and val_str not in ("none", "null", "nan"):
            # Try direct numeric parse
            try:
                return int(float(val_str))
            except (ValueError, TypeError):
                pass

            # Look for explicit days pattern: e.g. "15 days", "60 days"
            m_days = re.search(r"(\d+)\s*(?:day|days)", val_str)
            if m_days:
                return int(m_days.group(1))

            # Look for months pattern and convert using 1 month = 30 days convention
            word_to_num = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "single": 1, "double": 2, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
                "on": 1,
            }
            m_months = re.search(r"(\d+|one|two|three|four|five|six|on)\s*(?:month|months)", val_str)
            if m_months:
                raw_num = m_months.group(1)
                num = word_to_num.get(raw_num) or int(raw_num)
                return num * 30

    # Fallback to inspecting document text if model returned null or missing
    if doc_text:
        notice_m = re.search(
            r"(?:give|giving|by\s+giving|shall\s+give|terminated\s+by)\s+(?:on|one|1|\ba\b|two|three)\s+months?\s+(?:written\s+)?notice",
            doc_text,
            re.IGNORECASE,
        )
        if not notice_m:
            notice_m = re.search(
                r"TERMNNA?TION.*?(\d+|on|one|two|three)\s+months?\s+notice",
                doc_text,
                re.IGNORECASE,
            )
        if not notice_m:
            notice_m = re.search(
                r"(\d+|on|one|two|three)\s+months?\s+(?:written\s+)?notice",
                doc_text,
                re.IGNORECASE,
            )
        if notice_m:
            token = notice_m.group(0).lower()
            if any(w in token for w in ["on month", "one month", "1 month", "a month"]):
                return 30
            elif "two" in token or "2 month" in token:
                return 60
            elif "three" in token or "3 month" in token:
                return 90

    return None


def _raw_to_result(
    raw: RawLLMOutput,
    doc_text: str,
) -> ExtractionResult:
    """
    Convert a validated RawLLMOutput into a rich ExtractionResult,
    populating per-field evidence, confidence, and standardized types.
    """
    evidence: dict = raw.evidence or {}
    confidence: dict = raw.confidence or {}

    # 1. Party One: check against incidental locality names and resolve
    party_one_clean, p1_ev = resolve_party_one(raw.party_one, doc_text)
    if p1_ev and not evidence.get("party_one"):
        evidence["party_one"] = p1_ev

    # 2. Party Two: exclude parental qualifiers (Slo, S/O, etc.)
    party_two_clean = clean_party_name(raw.party_two)

    # 3. Agreement Value: check for numeric vs words conflict and flag for review
    val_str, val_conf, val_ev, has_rent_conflict = resolve_agreement_value_and_conflict(
        raw.agreement_value,
        evidence.get("agreement_value"),
        confidence.get("agreement_value"),
        doc_text,
    )
    if val_ev:
        evidence["agreement_value"] = val_ev
    if val_conf is not None:
        confidence["agreement_value"] = val_conf

    # 4. Agreement End Date: calendar calculation and off-by-one correction
    end_date = calculate_calendar_end_date(
        raw.agreement_start_date,
        raw.agreement_end_date,
        doc_text=doc_text,
    )

    # 5. Renewal notice: handle OCR typos ('on month notice' -> 30) and document search fallback
    renewal_days = parse_renewal_notice_days(raw.renewal_notice_days, doc_text=doc_text)
    if renewal_days is not None and not evidence.get("renewal_notice_days"):
        evidence["renewal_notice_days"] = f"Termination clause notice: {renewal_days} days"

    def _detail(field: str, value) -> FieldDetail:
        str_value = _str_or_none(str(value)) if value is not None else None
        ev = _str_or_none(evidence.get(field))
        try:
            conf_score = float(confidence.get(field, 0)) if confidence.get(field) is not None else None
        except (TypeError, ValueError):
            conf_score = None

        if str_value is not None and conf_score is None:
            conf_score = 0.85 if ev else 0.60
        elif str_value is None:
            conf_score = 0.0

        return FieldDetail(
            value=str_value,
            confidence_score=conf_score,
            evidence=ev,
            is_missing=(str_value is None),
        )

    field_details = {
        "agreement_value": _detail("agreement_value", val_str),
        "agreement_start_date": _detail("agreement_start_date", raw.agreement_start_date),
        "agreement_end_date": _detail("agreement_end_date", end_date),
        "renewal_notice_days": _detail("renewal_notice_days", renewal_days),
        "party_one": _detail("party_one", party_one_clean),
        "party_two": _detail("party_two", party_two_clean),
    }

    return ExtractionResult(
        agreement_value=val_str,
        agreement_start_date=_str_or_none(raw.agreement_start_date),
        agreement_end_date=_str_or_none(end_date),
        renewal_notice_days=renewal_days,
        party_one=party_one_clean,
        party_two=party_two_clean,
        field_details=field_details,
        document_text_preview=doc_text[:500],
    )


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────


def extract_from_bytes(data: bytes, filename: str) -> ExtractionResult:
    """
    Full pipeline: validate → extract text → call LLM → return result.

    Stage timings are recorded using monotonic clocks and embedded in the
    returned ExtractionResult.timings dict so callers can measure where
    time is spent without relying on log scraping.

    Parameters
    ----------
    data:
        Raw file bytes.
    filename:
        Original filename (used for extension detection and logging).

    Returns
    -------
    ExtractionResult
        Always returns an ExtractionResult; the ``status`` field
        distinguishes success from partial/failure states.
        ``timings`` contains stage durations in seconds.
        ``extraction_method`` indicates 'docx' or 'ocr'.
    """
    settings = get_settings()
    ext = Path(filename).suffix.lower()

    pipeline_start = time.monotonic()
    timings: dict[str, float] = {}
    extraction_method: Optional[str] = None

    # ── Stage A: validation ───────────────────────────────────
    t0 = time.monotonic()
    try:
        validate_file_bytes(data, filename)
    except Exception as exc:
        timings["validation_sec"] = time.monotonic() - t0
        timings["total_sec"] = time.monotonic() - pipeline_start
        logger.warning(
            "Stage A (validation) FAILED for '%s' in %.3fs: %s",
            filename, timings["validation_sec"], exc
        )
        return ExtractionResult(
            status=ExtractionStatus.FAILED,
            error_message=str(exc),
            timings=timings,
        )
    timings["validation_sec"] = time.monotonic() - t0
    logger.debug("Stage A (validation) OK for '%s' in %.3fs", filename, timings["validation_sec"])

    # ── Stage B: text extraction ──────────────────────────────
    doc_text: str = ""
    t0 = time.monotonic()
    try:
        with temp_file_from_bytes(data, ext) as tmp_path:
            if ext in DOCX_EXTENSIONS:
                extraction_method = "docx"
                logger.info("Stage B: Using python-docx extraction for '%s'", filename)
                doc_text = extract_text_from_docx(tmp_path)
            elif ext in IMAGE_EXTENSIONS:
                extraction_method = "ocr"
                logger.info("Stage B: Using EasyOCR for image '%s'", filename)
                doc_text = extract_text_from_image(tmp_path)
            else:
                doc_text = _extract_text_from_path(tmp_path)
                extraction_method = "unknown"
    except Exception as exc:
        timings["text_extraction_sec"] = time.monotonic() - t0
        timings["total_sec"] = time.monotonic() - pipeline_start
        logger.warning(
            "Stage B (text extraction) FAILED for '%s' [method=%s] in %.3fs: %s",
            filename, extraction_method, timings["text_extraction_sec"], exc
        )
        return ExtractionResult(
            status=ExtractionStatus.FAILED,
            error_message=f"Text extraction failed: {exc}",
            extraction_method=extraction_method,
            timings=timings,
        )
    timings["text_extraction_sec"] = time.monotonic() - t0
    logger.info(
        "Stage B (%s) OK for '%s': %d chars in %.3fs",
        extraction_method, filename, len(doc_text), timings["text_extraction_sec"]
    )

    # ── Stage C + D: AI extraction with few-shot ─────────────
    t0 = time.monotonic()
    few_shot: list[dict] = []
    if settings.few_shot_enabled:
        try:
            few_shot = build_few_shot_examples()
        except Exception as exc:
            logger.warning("Could not build few-shot examples: %s", exc)
    timings["few_shot_prep_sec"] = time.monotonic() - t0
    logger.debug(
        "Stage D (few-shot prep) for '%s': %d examples in %.3fs",
        filename, len(few_shot), timings["few_shot_prep_sec"]
    )

    t0 = time.monotonic()
    try:
        raw_output = call_llm(doc_text, few_shot_examples=few_shot or None)
    except ModelUnavailableError as exc:
        timings["llm_inference_sec"] = time.monotonic() - t0
        timings["total_sec"] = time.monotonic() - pipeline_start
        logger.error(
            "Stage C (LLM) MODEL UNAVAILABLE for '%s' after %.3fs: %s",
            filename, timings["llm_inference_sec"], exc
        )
        return ExtractionResult(
            status=ExtractionStatus.MODEL_UNAVAILABLE,
            error_message=str(exc),
            document_text_preview=doc_text[:500],
            extraction_method=extraction_method,
            timings=timings,
        )
    except ExtractionParseError as exc:
        timings["llm_inference_sec"] = time.monotonic() - t0
        timings["total_sec"] = time.monotonic() - pipeline_start
        logger.error(
            "Stage C (LLM) PARSE ERROR for '%s' after %.3fs: %s",
            filename, timings["llm_inference_sec"], exc
        )
        return ExtractionResult(
            status=ExtractionStatus.FAILED,
            error_message=f"Could not parse model output: {exc}",
            document_text_preview=doc_text[:500],
            extraction_method=extraction_method,
            timings=timings,
        )
    except Exception as exc:
        timings["llm_inference_sec"] = time.monotonic() - t0
        timings["total_sec"] = time.monotonic() - pipeline_start
        logger.exception("Stage C (LLM) UNEXPECTED ERROR for '%s' after %.3fs", filename, timings["llm_inference_sec"])
        return ExtractionResult(
            status=ExtractionStatus.FAILED,
            error_message=f"Unexpected extraction error: {exc}",
            document_text_preview=doc_text[:500],
            extraction_method=extraction_method,
            timings=timings,
        )
    timings["llm_inference_sec"] = time.monotonic() - t0
    logger.info(
        "Stage C (LLM inference) OK for '%s' in %.3fs",
        filename, timings["llm_inference_sec"]
    )

    # ── Stage E: structure output ─────────────────────────────
    t0 = time.monotonic()
    result = _raw_to_result(raw_output, doc_text)
    timings["result_assembly_sec"] = time.monotonic() - t0

    timings["total_sec"] = time.monotonic() - pipeline_start
    result.timings = timings
    result.extraction_method = extraction_method

    logger.info(
        "Pipeline complete for '%s': status=%s method=%s "
        "validation=%.3fs text_extraction=%.3fs llm=%.3fs total=%.3fs",
        filename, result.status, extraction_method,
        timings.get("validation_sec", 0),
        timings.get("text_extraction_sec", 0),
        timings.get("llm_inference_sec", 0),
        timings["total_sec"],
    )
    return result
