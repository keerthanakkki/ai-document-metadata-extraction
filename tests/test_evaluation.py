"""
tests/test_evaluation.py
─────────────────────────
Unit tests for evaluation logic (app/services/evaluation_service.py).

Covers:
  • Exact-match recall calculation
  • Null ground-truth values (skipped, not counted as failures)
  • Micro-averaged recall formula
  • Train/test overlap detection
  • Test labels NOT used during prediction (structural guarantee)
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from app.schemas import EvaluationReport, FieldRecall
from app.services.evaluation_service import (
    compute_recall,
    detect_overlap,
    _normalize,
    _exact_match,
    FIELD_NAMES,
)


# ─────────────────────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────────────────────

class TestNormalization:
    def test_strips_whitespace(self):
        assert _normalize("  hello  ") == "hello"

    def test_returns_none_for_none(self):
        assert _normalize(None) is None

    def test_returns_none_for_empty(self):
        assert _normalize("") is None

    def test_returns_none_for_nan(self):
        assert _normalize("nan") is None

    def test_preserves_case(self):
        assert _normalize("GERALDINE Q. GALINATO") == "GERALDINE Q. GALINATO"

    def test_preserves_punctuation(self):
        assert _normalize("Antonio, Jr.") == "Antonio, Jr."


class TestExactMatch:
    def test_identical_strings(self):
        assert _exact_match("6500", "6500") is True

    def test_whitespace_ignored(self):
        assert _exact_match(" 6500 ", "6500") is True

    def test_different_values(self):
        assert _exact_match("6500", "7000") is False

    def test_none_both(self):
        assert _exact_match(None, None) is True

    def test_none_vs_value(self):
        assert _exact_match(None, "6500") is False

    def test_case_sensitive(self):
        """Case is NOT normalised — different cases do not match."""
        assert _exact_match("john doe", "John Doe") is False


# ─────────────────────────────────────────────────────────────
# Recall computation
# ─────────────────────────────────────────────────────────────

class TestComputeRecall:
    def _make_gt(self, **kwargs) -> dict:
        base = {"filename": "doc1"}
        for f in FIELD_NAMES:
            base[f] = None
        base.update(kwargs)
        return base

    def _make_pred(self, **kwargs) -> dict:
        base = {"filename": "doc1"}
        for f in FIELD_NAMES:
            base[f] = None
        base.update(kwargs)
        return base

    def test_perfect_recall(self):
        gt = [self._make_gt(agreement_value="6500", party_one="John")]
        pred = [self._make_pred(agreement_value="6500", party_one="John")]
        report = compute_recall(pred, gt)
        assert report.micro_averaged_recall == 100.0

    def test_zero_recall(self):
        gt = [self._make_gt(agreement_value="6500")]
        pred = [self._make_pred(agreement_value="9999")]
        report = compute_recall(pred, gt)
        assert report.micro_averaged_recall == 0.0

    def test_null_gt_not_counted(self):
        """Null ground-truth values must NOT be counted in the denominator."""
        gt = [self._make_gt(agreement_value=None, party_one="John")]
        pred = [self._make_pred(agreement_value="any", party_one="John")]
        report = compute_recall(pred, gt)
        # Only party_one counts — agreement_value is null in GT
        v_field = next(f for f in report.field_recalls if f.field == "agreement_value")
        assert v_field.total_ground_truth == 0

    def test_partial_recall(self):
        gt = [self._make_gt(agreement_value="6500", party_one="John", party_two="Jane")]
        pred = [self._make_pred(agreement_value="6500", party_one="Wrong", party_two="Jane")]
        report = compute_recall(pred, gt)
        # 2 out of 3 fields match
        assert report.total_exact_matches == 2
        assert report.total_ground_truth_values == 3

    def test_micro_averaged_formula(self):
        """Micro recall = total_matches / total_gt × 100."""
        gt = [self._make_gt(agreement_value="6500", party_one="John")]
        pred = [self._make_pred(agreement_value="6500", party_one="Wrong")]
        report = compute_recall(pred, gt)
        expected = round(100.0 * 1 / 2, 2)
        assert report.micro_averaged_recall == expected

    def test_multiple_documents(self):
        gt = [
            {"filename": "doc1", **{f: None for f in FIELD_NAMES},
             "agreement_value": "6500", "party_one": "John"},
            {"filename": "doc2", **{f: None for f in FIELD_NAMES},
             "agreement_value": "9000", "party_one": "Mary"},
        ]
        pred = [
            {"filename": "doc1", **{f: None for f in FIELD_NAMES},
             "agreement_value": "6500", "party_one": "Wrong"},
            {"filename": "doc2", **{f: None for f in FIELD_NAMES},
             "agreement_value": "9000", "party_one": "Mary"},
        ]
        report = compute_recall(pred, gt)
        assert report.total_exact_matches == 3
        assert report.total_ground_truth_values == 4

    def test_report_contains_all_fields(self):
        gt = [self._make_gt(agreement_value="6500")]
        pred = [self._make_pred(agreement_value="6500")]
        report = compute_recall(pred, gt)
        field_names_in_report = [fr.field for fr in report.field_recalls]
        for f in FIELD_NAMES:
            assert f in field_names_in_report

    def test_per_document_entry_included(self):
        gt = [self._make_gt(agreement_value="6500")]
        pred = [self._make_pred(agreement_value="6500")]
        report = compute_recall(pred, gt)
        assert len(report.per_document) == 1
        assert report.per_document[0]["filename"] == "doc1"


# ─────────────────────────────────────────────────────────────
# Overlap detection
# ─────────────────────────────────────────────────────────────

class TestOverlapDetection:
    def test_detects_overlap(self, tmp_path: Path):
        train = tmp_path / "train.csv"
        test = tmp_path / "test.csv"
        train.write_text("File Name\ndoc_a\ndoc_b\n")
        test.write_text("File Name\ndoc_b\ndoc_c\n")
        overlap = detect_overlap(str(train), str(test))
        assert "doc_b" in overlap

    def test_no_overlap(self, tmp_path: Path):
        train = tmp_path / "train.csv"
        test = tmp_path / "test.csv"
        train.write_text("File Name\ndoc_a\n")
        test.write_text("File Name\ndoc_b\n")
        overlap = detect_overlap(str(train), str(test))
        assert overlap == []

    def test_real_data_overlap(self):
        """Verify the known overlap in the actual dataset is detected."""
        base = Path(__file__).resolve().parent.parent / "data"
        if not (base / "train.csv").exists() or not (base / "test.csv").exists():
            pytest.skip("Dataset CSVs not available")
        overlap = detect_overlap(
            str(base / "train.csv"), str(base / "test.csv")
        )
        # 24158401-Rental-Agreement appears in both
        assert any("24158401" in name for name in overlap), (
            f"Expected known overlap '24158401-Rental-Agreement' but got: {overlap}"
        )


# ─────────────────────────────────────────────────────────────
# Structural guarantee: test labels not used during prediction
# ─────────────────────────────────────────────────────────────

class TestTestLabelsNotUsedDuringPrediction:
    """
    Structural test: verify that evaluation_service.compute_recall
    only reads ground truth AFTER predictions are passed in — it does
    not import or call test.csv loading internally.
    """

    def test_compute_recall_does_not_import_test_csv(self):
        """compute_recall must accept pre-computed predictions, not load them itself."""
        import inspect
        from app.services import evaluation_service
        src = inspect.getsource(evaluation_service.compute_recall)
        # The function should not open or read test.csv by itself
        assert "test.csv" not in src, (
            "compute_recall() should not load test.csv internally. "
            "Ground-truth data must be passed in as a parameter."
        )

    def test_evaluate_script_saves_predictions_before_loading_gt(self):
        """
        Structural check: evaluate.py must save predictions before
        calling load_ground_truth.
        """
        script = Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py"
        if not script.exists():
            pytest.skip("evaluate.py not found")
        src = script.read_text(encoding="utf-8")
        pred_save_pos = src.find("predictions.csv")
        gt_load_pos = src.find("load_ground_truth")
        assert pred_save_pos != -1, "predictions.csv not found in evaluate.py"
        assert gt_load_pos != -1, "load_ground_truth not found in evaluate.py"
        assert pred_save_pos < gt_load_pos, (
            "Predictions must be saved BEFORE ground truth is loaded. "
            "Check the order in evaluate.py."
        )
