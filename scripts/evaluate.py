"""
scripts/evaluate.py
────────────────────
Evaluation pipeline for the AI Document Metadata Extraction system.

Execution order (as required by the assignment):
  1. Read test documents from data/test/
  2. Generate predictions using the REAL extraction pipeline
     (never reads test.csv labels during this phase)
  3. Save predictions to outputs/predictions.csv and outputs/predictions.json
  4. ONLY THEN load ground-truth labels from test.csv
  5. Compare predictions vs ground truth
  6. Calculate per-field exact-match recall and micro-averaged recall
  7. Save evaluation report to outputs/evaluation_report.json
  8. Save field recall table to outputs/field_recall.csv

Usage:
    python scripts/evaluate.py

The script requires:
  • Ollama running with the configured model
  • EasyOCR installed (for image documents)
  • Test documents in data/test/
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Ensure project root is on sys.path ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from app.config import get_settings
from app.schemas import ExtractionStatus
from app.services.extraction_service import extract_from_bytes, FIELD_NAMES
from app.services.evaluation_service import (
    compute_recall,
    detect_overlap,
    load_ground_truth,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def clean_stem(path: Path) -> str:
    """Strip known single or compound extensions to match CSV File Name."""
    name = path.name
    for ext in [".pdf.docx", ".docx", ".pdf", ".png", ".jpg", ".jpeg"]:
        if name.lower().endswith(ext):
            return name[:-len(ext)]
    return path.stem


def main() -> None:
    settings = get_settings()

    # ── Ensure outputs directory exists ───────────────────────
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)

    # ── Report train/test filename overlap ────────────────────
    logger.info("=" * 60)
    logger.info("Checking for train/test filename overlap…")
    overlap = detect_overlap(str(settings.train_csv), str(settings.test_csv))
    if overlap:
        logger.warning(
            "OVERLAP DETECTED: %d filename(s) appear in both datasets: %s",
            len(overlap),
            overlap,
        )
        logger.warning(
            "The recall score for these documents may be optimistic because "
            "the model's few-shot examples include their ground-truth labels."
        )
    else:
        logger.info("No filename overlap detected.")

    # ── Discover test documents ───────────────────────────────
    logger.info("=" * 60)
    test_dir = settings.test_docs_dir
    supported_exts = {".docx", ".pdf", ".png", ".jpg", ".jpeg"}
    test_files = [
        f for f in sorted(test_dir.iterdir())
        if f.is_file() and any(f.name.lower().endswith(ext) for ext in supported_exts)
    ]

    if not test_files:
        logger.error(
            "No test documents found in '%s'. Aborting.", test_dir
        )
        sys.exit(1)

    logger.info("Found %d test document(s) in '%s'.", len(test_files), test_dir)

    # ── Phase 1: Generate predictions (NO test labels read yet) ──
    logger.info("=" * 60)
    logger.info("Phase 1: Generating predictions (test labels NOT loaded yet)…")

    predictions: list[dict] = []
    any_model_unavailable = False

    for doc_path in test_files:
        filename_stem = clean_stem(doc_path)
        logger.info("Processing: %s (stem: %s)", doc_path.name, filename_stem)

        try:
            data = doc_path.read_bytes()
            result = extract_from_bytes(data, doc_path.name)
        except Exception as exc:
            logger.error("  ✗ Unhandled error: %s", exc)
            pred_row = {"filename": filename_stem}
            for f in FIELD_NAMES:
                pred_row[f] = None
            pred_row["_status"] = "error"
            pred_row["_error"] = str(exc)
            predictions.append(pred_row)
            continue

        if result.status == ExtractionStatus.MODEL_UNAVAILABLE:
            logger.error(
                "  ✗ Model unavailable: %s", result.error_message
            )
            any_model_unavailable = True
            pred_row = {"filename": filename_stem}
            for f in FIELD_NAMES:
                pred_row[f] = None
            pred_row["_status"] = "model_unavailable"
            pred_row["_error"] = result.error_message
            predictions.append(pred_row)
            continue

        pred_row = {"filename": filename_stem}
        pred_row["agreement_value"] = result.agreement_value
        pred_row["agreement_start_date"] = result.agreement_start_date
        pred_row["agreement_end_date"] = result.agreement_end_date
        pred_row["renewal_notice_days"] = (
            str(result.renewal_notice_days)
            if result.renewal_notice_days is not None
            else None
        )
        pred_row["party_one"] = result.party_one
        pred_row["party_two"] = result.party_two
        pred_row["_status"] = result.status.value
        pred_row["_error"] = result.error_message

        status_icon = "✓" if result.status == ExtractionStatus.SUCCESS else "~"
        logger.info("  %s Status: %s", status_icon, result.status.value)
        predictions.append(pred_row)

    if any_model_unavailable:
        logger.error(
            "\n⚠️  One or more documents could not be processed because "
            "the LLM model is unavailable.\n"
            "   Start Ollama and ensure the model is downloaded:\n"
            "   ollama serve\n"
            "   ollama pull %s\n"
            "   Then re-run this script.",
            settings.ollama_model,
        )

    # ── Save predictions BEFORE loading ground truth ──────────
    logger.info("=" * 60)
    logger.info("Saving predictions (before loading ground truth)…")

    pred_path_csv = settings.outputs_dir / "predictions.csv"
    pred_path_json = settings.outputs_dir / "predictions.json"

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(pred_path_csv, index=False)
    logger.info("  Saved: %s", pred_path_csv)

    with open(pred_path_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, default=str)
    logger.info("  Saved: %s", pred_path_json)

    # ── Phase 2: Load ground truth and evaluate ───────────────
    logger.info("=" * 60)
    logger.info("Phase 2: Loading ground-truth labels from test.csv…")

    try:
        ground_truth = load_ground_truth(str(settings.test_csv))
    except Exception as exc:
        logger.error("Could not load test.csv: %s", exc)
        sys.exit(1)

    logger.info("  Loaded %d ground-truth record(s).", len(ground_truth))

    # ── Compute recall ────────────────────────────────────────
    logger.info("Computing recall metrics…")
    report = compute_recall(predictions, ground_truth)

    # ── Save evaluation report ────────────────────────────────
    report_path = settings.outputs_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, default=str)
    logger.info("  Saved: %s", report_path)

    # ── Save field recall CSV ─────────────────────────────────
    recall_path = settings.outputs_dir / "field_recall.csv"
    recall_rows = [fr.model_dump() for fr in report.field_recalls]
    pd.DataFrame(recall_rows).to_csv(recall_path, index=False)
    logger.info("  Saved: %s", recall_path)

    # ── Save failures CSV ─────────────────────────────────────
    failures_path = settings.outputs_dir / "failures.csv"
    failures = []
    for doc in report.per_document:
        fn = doc.get("filename")
        for f in FIELD_NAMES:
            entry = doc.get(f, {})
            if isinstance(entry, dict) and entry.get("match") is False:
                failures.append({
                    "filename": fn,
                    "field": f,
                    "expected": entry.get("expected"),
                    "predicted": entry.get("predicted"),
                    "normalized_match": entry.get("normalized_match"),
                })
    pd.DataFrame(failures).to_csv(failures_path, index=False)
    logger.info("  Saved: %s (%d mismatch rows)", failures_path, len(failures))

    # ── Print summary ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    if overlap:
        logger.warning("⚠  OVERLAP WARNING: %s", report.overlap_warning)

    logger.info("Comparison policy: %s", report.comparison_policy)
    logger.info("")
    logger.info("Per-field recall (EXACT MATCH):")
    logger.info("  %-30s  %5s / %-5s  %s", "Field", "Hits", "Total", "Recall")
    logger.info("  " + "-" * 55)
    for fr in report.field_recalls:
        logger.info(
            "  %-30s  %5d / %-5d  %.1f%%",
            fr.field,
            fr.exact_matches,
            fr.total_ground_truth,
            fr.recall_pct,
        )

    logger.info("")
    logger.info(
        "  Exact-match Micro-averaged recall:       %.1f%% (%d / %d)",
        report.micro_averaged_recall,
        report.total_exact_matches,
        report.total_ground_truth_values,
    )

    if report.normalized_micro_averaged_recall is not None:
        logger.info(
            "  Secondary Normalized Micro-avg recall:   %.1f%%",
            report.normalized_micro_averaged_recall,
        )
    logger.info("=" * 60)
    logger.info("Outputs written to: %s", settings.outputs_dir)



if __name__ == "__main__":
    main()
