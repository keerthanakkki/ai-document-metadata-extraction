"""
app/services/evaluation_service.py
────────────────────────────────────
Evaluation logic used by scripts/evaluate.py.

Responsibilities
────────────────
• Detect filename overlap between train and test sets.
• Calculate per-field exact-match recall.
• Calculate micro-averaged recall across all non-missing ground-truth values.
• Support a documented comparison policy (strip whitespace, preserve case).
• Never use test labels during prediction.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from app.config import get_settings
from app.schemas import EvaluationReport, FieldRecall

logger = logging.getLogger(__name__)

FIELD_NAMES = [
    "agreement_value",
    "agreement_start_date",
    "agreement_end_date",
    "renewal_notice_days",
    "party_one",
    "party_two",
]

# Map CSV column names → canonical field names
CSV_COL_MAP = {
    "File Name": "filename",
    "Aggrement Value": "agreement_value",
    "Aggrement Start Date": "agreement_start_date",
    "Aggrement End Date": "agreement_end_date",
    "Renewal Notice (Days)": "renewal_notice_days",
    "Party One": "party_one",
    "Party Two": "party_two",
}


def detect_overlap(train_csv: str, test_csv: str) -> list[str]:
    """
    Return a list of filenames that appear in both train.csv and test.csv.
    """
    try:
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        train_df.columns = [c.strip() for c in train_df.columns]
        test_df.columns = [c.strip() for c in test_df.columns]
        train_names = set(train_df["File Name"].dropna().str.strip().tolist())
        test_names = set(test_df["File Name"].dropna().str.strip().tolist())
        return sorted(train_names & test_names)
    except Exception as exc:
        logger.warning("Could not check train/test overlap: %s", exc)
        return []


def _normalize(value: Optional[str]) -> Optional[str]:
    """
    Comparison normalization: strip leading/trailing whitespace only.
    Case, punctuation and date formats are NOT altered so that meaningful
    differences are not silently hidden.
    """
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def _exact_match(pred: Optional[str], gt: Optional[str]) -> bool:
    """Strict exact match after whitespace normalization."""
    return _normalize(pred) == _normalize(gt)


def _secondary_normalize(value: Optional[str]) -> Optional[str]:
    """
    Documented secondary comparison normalization:
    - Strips whitespace
    - Case-insensitive
    - Strips standard honorifics / salutations (Mr., Mrs., Ms., Sri, Shri, Smt., Dr.)
    - Strips corporate representation clauses (', represented by ...')
    - Strips leading dots or punctuation (e.g. '.B.Kishore' -> 'B.Kishore')
    """
    import re
    base = _normalize(value)
    if base is None:
        return None
    s = base.lower()
    # Strip leading honorifics / salutations
    s = re.sub(r"^(?:mr|mrs|ms|miss|dr|sri|shri|smt)\b\s*(\.?\s*)", "", s, flags=re.IGNORECASE)
    # Strip corporate representative clauses
    s = re.sub(
        r"[\s,]+(?:represented\s+by|rep\.?\s+by|through\s+its|auth(?:\.|orized)?\s+signatory).*$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    # Strip leading/trailing punctuation like dot, comma
    s = s.strip(" .,;:-_")
    # Collapse multiple whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None


def _normalized_match(pred: Optional[str], gt: Optional[str]) -> bool:
    """Secondary normalized match comparing cleaned canonical forms."""
    return _secondary_normalize(pred) == _secondary_normalize(gt)


def _clean_filename_stem(name: str) -> str:
    s = str(name).strip()
    for ext in [".pdf.docx", ".docx", ".pdf", ".png", ".jpg", ".jpeg"]:
        if s.lower().endswith(ext):
            return s[:-len(ext)]
    return s


def compute_recall(
    predictions: list[dict],
    ground_truth: list[dict],
) -> EvaluationReport:
    """
    Compute per-field and micro-averaged recall, along with secondary normalized comparison.

    Parameters
    ----------
    predictions:
        List of dicts with keys: filename + the six field names.
        Values are predicted strings (or None/null).
    ground_truth:
        List of dicts with keys: filename + the six field names.
        Values are ground-truth strings from evaluation dataset (or None for missing).

    Returns
    -------
    EvaluationReport
    """
    settings = get_settings()

    # Index predictions by normalized filename stem
    pred_index: dict[str, dict] = {
        _clean_filename_stem(str(p.get("filename", ""))): p for p in predictions
    }

    per_field_matches: dict[str, int] = {f: 0 for f in FIELD_NAMES}
    per_field_norm_matches: dict[str, int] = {f: 0 for f in FIELD_NAMES}
    per_field_total: dict[str, int] = {f: 0 for f in FIELD_NAMES}

    per_doc: list[dict] = []
    failed_processing = 0
    successfully_processed = 0

    for gt_row in ground_truth:
        raw_filename = str(gt_row.get("filename", "")).strip()
        filename = _clean_filename_stem(raw_filename)
        pred_row = pred_index.get(filename, {})

        if pred_row.get("_status") in ("error", "model_unavailable") or not pred_row:
            failed_processing += 1
        else:
            successfully_processed += 1

        doc_entry: dict = {"filename": raw_filename}

        for field in FIELD_NAMES:
            gt_val = _normalize(str(gt_row.get(field, "")) if gt_row.get(field) is not None else None)
            pred_val = _normalize(str(pred_row.get(field, "")) if pred_row.get(field) is not None else None)

            # Skip missing ground-truth values (not treated as failures)
            if gt_val is None:
                doc_entry[field] = {
                    "expected": None,
                    "predicted": pred_val,
                    "match": None,
                    "note": "ground_truth_missing",
                }
                continue

            per_field_total[field] += 1
            exact_m = _exact_match(pred_val, gt_val)
            norm_m = _normalized_match(pred_val, gt_val)

            if exact_m:
                per_field_matches[field] += 1
            if norm_m:
                per_field_norm_matches[field] += 1

            doc_entry[field] = {
                "expected": gt_val,
                "predicted": pred_val,
                "match": exact_m,
                "normalized_match": norm_m,
            }

        per_doc.append(doc_entry)

    # Build field recall list (exact match)
    field_recalls: list[FieldRecall] = []
    norm_field_recalls: list[FieldRecall] = []
    total_matches = 0
    total_norm_matches = 0
    total_gt = 0

    for field in FIELD_NAMES:
        m = per_field_matches[field]
        nm = per_field_norm_matches[field]
        t = per_field_total[field]
        pct = round(100.0 * m / t, 2) if t > 0 else 0.0
        norm_pct = round(100.0 * nm / t, 2) if t > 0 else 0.0

        field_recalls.append(
            FieldRecall(field=field, exact_matches=m, total_ground_truth=t, recall_pct=pct)
        )
        norm_field_recalls.append(
            FieldRecall(field=field, exact_matches=nm, total_ground_truth=t, recall_pct=norm_pct)
        )
        total_matches += m
        total_norm_matches += nm
        total_gt += t

    micro_recall = round(100.0 * total_matches / total_gt, 2) if total_gt > 0 else 0.0
    norm_micro_recall = round(100.0 * total_norm_matches / total_gt, 2) if total_gt > 0 else 0.0

    # Overlap warning
    overlap = detect_overlap(str(settings.train_csv), str(settings.test_csv))
    if overlap:
        overlap_warning = (
            f"WARNING: {len(overlap)} filename(s) appear in both the training "
            f"and test datasets: {overlap}. "
            f"The reported recall score may be optimistic for these documents "
            f"because the model's few-shot examples include their ground-truth "
            f"labels from train.csv. These results should be interpreted with caution."
        )
    else:
        overlap_warning = "No filename overlap detected between training and test sets."

    return EvaluationReport(
        micro_averaged_recall=micro_recall,
        total_exact_matches=total_matches,
        total_ground_truth_values=total_gt,
        field_recalls=field_recalls,
        overlapping_filenames=overlap,
        overlap_warning=overlap_warning,
        per_document=per_doc,
        expected_test_documents=len(ground_truth),
        successfully_processed=successfully_processed,
        failed_processing=failed_processing,
        missing_files=[],
        duplicate_files=[],
        normalized_micro_averaged_recall=norm_micro_recall,
        normalized_field_recalls=norm_field_recalls,
        comparison_policy=(
            "Exact match after stripping leading/trailing whitespace. "
            "Case, punctuation, date formats, and currency are NOT normalised "
            "so that meaningful differences are not hidden. Secondary normalized "
            "metrics strip salutations/honorifics and corporate representation clauses."
        ),
    )



def load_ground_truth(test_csv_path: str) -> list[dict]:
    """Load ground-truth labels from test.csv."""
    df = pd.read_csv(test_csv_path)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns=CSV_COL_MAP, inplace=True)
    df.dropna(subset=["filename"], inplace=True)
    records = []
    for _, row in df.iterrows():
        record = {"filename": str(row["filename"]).strip()}
        for field in FIELD_NAMES:
            val = row.get(field)
            if pd.isna(val) if hasattr(pd, "isna") else val is None:
                record[field] = None
            else:
                record[field] = str(val).strip() if str(val).strip() not in ("nan", "") else None
        records.append(record)
    return records
