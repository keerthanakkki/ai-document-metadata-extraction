"""
tests/test_csv_evaluation.py
────────────────────────────────────────────────────────────────
Validates predictions against ground-truth labels in
data/test.csv and data/train.csv.

Groups:
  1. CSV structure
  2. Overlap detection
  3. Document discovery & clean_stem
  4. Normalization helpers
  5. load_ground_truth (real CSV values)
  6. Prediction file integrity (skipped if predictions not yet generated)
  7. Evaluation report (skipped if evaluate.py not yet run)
  8. compute_recall end-to-end
  9. Data leakage guard
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.services.evaluation_service import (
    detect_overlap,
    load_ground_truth,
    compute_recall,
    FIELD_NAMES,
    _clean_filename_stem,
    _normalize,
    _exact_match,
)

SETTINGS = get_settings()
TRAIN_CSV = SETTINGS.train_csv
TEST_CSV = SETTINGS.test_csv
TRAIN_DIR = SETTINGS.train_docs_dir
TEST_DIR = SETTINGS.test_docs_dir
OUTPUTS_DIR = SETTINGS.outputs_dir

PRED_CSV = OUTPUTS_DIR / "predictions.csv"
PRED_JSON = OUTPUTS_DIR / "predictions.json"
REPORT_JSON = OUTPUTS_DIR / "evaluation_report.json"
FIELD_RECALL_CSV = OUTPUTS_DIR / "field_recall.csv"

_COL_MAP = {
    "File Name": "filename",
    "Aggrement Value": "agreement_value",
    "Aggrement Start Date": "agreement_start_date",
    "Aggrement End Date": "agreement_end_date",
    "Renewal Notice (Days)": "renewal_notice_days",
    "Party One": "party_one",
    "Party Two": "party_two",
}
EXPECTED_FIELDS = ["filename"] + FIELD_NAMES


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns=_COL_MAP, inplace=True)
    df.dropna(subset=["filename"], inplace=True)
    df["filename"] = df["filename"].str.strip()
    return df


PREDICTIONS_EXIST = PRED_CSV.exists() and PRED_JSON.exists()
REPORT_EXISTS = REPORT_JSON.exists()

skip_no_preds = pytest.mark.skipif(
    not PREDICTIONS_EXIST,
    reason="Run `python scripts/evaluate.py` first to generate predictions",
)
skip_no_report = pytest.mark.skipif(
    not REPORT_EXISTS,
    reason="Run `python scripts/evaluate.py` first to generate evaluation_report.json",
)


# ── 1. CSV Structure ──────────────────────────────────────────────────────────

class TestCSVStructure:
    def test_train_csv_exists(self):
        assert TRAIN_CSV.exists(), f"train.csv not found: {TRAIN_CSV}"

    def test_test_csv_exists(self):
        assert TEST_CSV.exists(), f"test.csv not found: {TEST_CSV}"

    def test_train_csv_columns(self):
        df = _load_csv(TRAIN_CSV)
        for col in EXPECTED_FIELDS:
            assert col in df.columns, f"Missing col '{col}' in train.csv"

    def test_test_csv_columns(self):
        df = _load_csv(TEST_CSV)
        for col in EXPECTED_FIELDS:
            assert col in df.columns, f"Missing col '{col}' in test.csv"

    def test_train_csv_row_count(self):
        assert len(_load_csv(TRAIN_CSV)) == 10

    def test_test_csv_row_count(self):
        assert len(_load_csv(TEST_CSV)) == 4

    def test_train_csv_known_filenames(self):
        names = set(_load_csv(TRAIN_CSV)["filename"].tolist())
        assert "6683127-House-Rental-Contract-GERALDINE-GALINATO-v2-Page-1" in names
        assert "24158401-Rental-Agreement" in names

    def test_test_csv_known_filenames(self):
        names = set(_load_csv(TEST_CSV)["filename"].tolist())
        for n in [
            "24158401-Rental-Agreement",
            "95980236-Rental-Agreement",
            "156155545-Rental-Agreement-Kns-Home",
            "228094620-Rental-Agreement",
        ]:
            assert n in names, f"{n!r} missing from test.csv filenames"

    def test_test_csv_no_missing_values(self):
        df = _load_csv(TEST_CSV)
        for f in FIELD_NAMES:
            if f in df.columns:
                missing = df[f].isna().sum()
                assert missing == 0, f"test.csv field '{f}' has {missing} missing value(s)"

    def test_train_csv_one_missing_renewal(self):
        df = _load_csv(TRAIN_CSV)
        missing = df["renewal_notice_days"].isna().sum()
        assert missing == 1, f"Expected 1 missing renewal_notice_days in train.csv, got {missing}"


# ── 2. Overlap Detection ──────────────────────────────────────────────────────

class TestOverlapDetection:
    def test_overlap_detected(self):
        assert len(detect_overlap(str(TRAIN_CSV), str(TEST_CSV))) > 0

    def test_overlap_is_24158401(self):
        assert "24158401-Rental-Agreement" in detect_overlap(str(TRAIN_CSV), str(TEST_CSV))

    def test_overlap_count_exactly_one(self):
        assert len(detect_overlap(str(TRAIN_CSV), str(TEST_CSV))) == 1

    def test_overlap_labels_match_across_splits(self):
        tr = _load_csv(TRAIN_CSV)
        te = _load_csv(TEST_CSV)
        r_tr = tr[tr["filename"] == "24158401-Rental-Agreement"].iloc[0]
        r_te = te[te["filename"] == "24158401-Rental-Agreement"].iloc[0]
        for f in FIELD_NAMES:
            tv = str(r_tr.get(f, "")).strip() if pd.notna(r_tr.get(f)) else ""
            ev = str(r_te.get(f, "")).strip() if pd.notna(r_te.get(f)) else ""
            assert tv == ev, f"Label mismatch for '{f}': train={tv!r} test={ev!r}"

    def test_overlap_file_in_test_dir_only(self):
        assert (TEST_DIR / "24158401-Rental-Agreement.png").exists()
        assert not (TRAIN_DIR / "24158401-Rental-Agreement.png").exists()


# ── 3. Document Discovery ─────────────────────────────────────────────────────

class TestDocumentDiscovery:
    def test_train_dir_exists(self):
        assert TRAIN_DIR.exists()

    def test_test_dir_exists(self):
        assert TEST_DIR.exists()

    def test_train_dir_has_10_docs(self):
        exts = {".docx", ".png", ".jpg", ".jpeg"}
        docs = [f for f in TRAIN_DIR.iterdir() if f.is_file() and f.suffix.lower() in exts]
        assert len(docs) == 10, f"Expected 10 training docs, found {len(docs)}"

    def test_test_dir_has_4_docs(self):
        exts = {".docx", ".pdf", ".png", ".jpg", ".jpeg"}
        docs = [
            f for f in TEST_DIR.iterdir()
            if f.is_file() and any(f.name.lower().endswith(e) for e in exts)
        ]
        assert len(docs) == 4, f"Expected 4 test docs, found {len(docs)}"

    def test_test_dir_has_2_compound_extension_files(self):
        compound = list(TEST_DIR.glob("*.pdf.docx"))
        assert len(compound) == 2, f"Expected 2 .pdf.docx files, found {len(compound)}"

    def test_clean_stem_pdf_docx_156(self):
        assert (
            _clean_filename_stem("156155545-Rental-Agreement-Kns-Home.pdf.docx")
            == "156155545-Rental-Agreement-Kns-Home"
        )

    def test_clean_stem_pdf_docx_228(self):
        assert (
            _clean_filename_stem("228094620-Rental-Agreement.pdf.docx")
            == "228094620-Rental-Agreement"
        )

    def test_clean_stem_docx(self):
        assert _clean_filename_stem("47854715-RENTAL-AGREEMENT.docx") == "47854715-RENTAL-AGREEMENT"

    def test_clean_stem_png(self):
        assert _clean_filename_stem("24158401-Rental-Agreement.png") == "24158401-Rental-Agreement"

    def test_clean_stem_bare(self):
        assert _clean_filename_stem("24158401-Rental-Agreement") == "24158401-Rental-Agreement"


# ── 4. Normalization Helpers ──────────────────────────────────────────────────

class TestNormalizationHelpers:
    def test_normalize_strips_whitespace(self):
        assert _normalize("  12000  ") == "12000"

    def test_normalize_none(self):
        assert _normalize(None) is None

    def test_normalize_nan_string(self):
        assert _normalize("nan") is None

    def test_normalize_empty_string(self):
        assert _normalize("  ") is None

    def test_exact_match_whitespace_insensitive(self):
        assert _exact_match("  12000  ", "12000") is True

    def test_exact_match_case_sensitive(self):
        assert _exact_match("Hanumaiah", "hanumaiah") is False

    def test_exact_match_none_vs_none(self):
        assert _exact_match(None, None) is True

    def test_exact_match_none_vs_value(self):
        assert _exact_match(None, "12000") is False


# ── 5. Ground Truth Loading (real CSV values) ─────────────────────────────────

class TestLoadGroundTruth:
    def test_returns_4_records(self):
        assert len(load_ground_truth(str(TEST_CSV))) == 4

    def test_all_fields_present(self):
        for rec in load_ground_truth(str(TEST_CSV)):
            for f in ["filename"] + FIELD_NAMES:
                assert f in rec, f"Field '{f}' missing in ground-truth record"

    def test_24158401_agreement_value(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "24158401" in r["filename"])
        assert rec["agreement_value"] == "12000"

    def test_24158401_start_date(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "24158401" in r["filename"])
        assert rec["agreement_start_date"] == "01.04.2008"

    def test_24158401_end_date(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "24158401" in r["filename"])
        assert rec["agreement_end_date"] == "31.03.2009"

    def test_24158401_renewal_notice(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "24158401" in r["filename"])
        assert rec["renewal_notice_days"] == "60"

    def test_24158401_party_one(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "24158401" in r["filename"])
        assert "Hanumaiah" in rec["party_one"]

    def test_24158401_party_two(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "24158401" in r["filename"])
        assert "Vishal Bhardwaj" in rec["party_two"]

    def test_95980236_agreement_value(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "95980236" in r["filename"])
        assert rec["agreement_value"] == "9000"

    def test_95980236_start_date(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "95980236" in r["filename"])
        assert rec["agreement_start_date"] == "01.04.2010"

    def test_95980236_end_date(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "95980236" in r["filename"])
        assert rec["agreement_end_date"] == "31.03.2011"

    def test_95980236_renewal_notice(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "95980236" in r["filename"])
        assert rec["renewal_notice_days"] == "30"

    def test_95980236_party_one(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "95980236" in r["filename"])
        assert "Sakunthala" in rec["party_one"]

    def test_95980236_party_two(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "95980236" in r["filename"])
        assert "Ravi Kian" in rec["party_two"]

    def test_156155545_agreement_value(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "156155545" in r["filename"])
        assert rec["agreement_value"] == "12000"

    def test_156155545_start_date(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "156155545" in r["filename"])
        assert rec["agreement_start_date"] == "15.12.2012"

    def test_156155545_end_date(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "156155545" in r["filename"])
        assert rec["agreement_end_date"] == "14.11.2013"

    def test_156155545_renewal_notice(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "156155545" in r["filename"])
        assert rec["renewal_notice_days"] == "30"

    def test_156155545_party_one(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "156155545" in r["filename"])
        assert "NATARAJ" in rec["party_one"]

    def test_156155545_party_two(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "156155545" in r["filename"])
        assert "VYSHNAVI" in rec["party_two"]

    def test_228094620_agreement_value(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "228094620" in r["filename"])
        assert rec["agreement_value"] == "15000"

    def test_228094620_start_date(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "228094620" in r["filename"])
        assert rec["agreement_start_date"] == "07.07.2013"

    def test_228094620_end_date(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "228094620" in r["filename"])
        assert rec["agreement_end_date"] == "06.06.2014"

    def test_228094620_renewal_notice(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "228094620" in r["filename"])
        assert rec["renewal_notice_days"] == "30"

    def test_228094620_party_one(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "228094620" in r["filename"])
        assert "KAPIL" in rec["party_one"]

    def test_228094620_party_two(self):
        rec = next(r for r in load_ground_truth(str(TEST_CSV)) if "228094620" in r["filename"])
        assert "Kishore" in rec["party_two"]


# ── 6. Prediction File Integrity ──────────────────────────────────────────────

@skip_no_preds
class TestPredictionFileIntegrity:
    def test_pred_csv_has_4_rows(self):
        assert len(pd.read_csv(PRED_CSV)) == 4

    def test_pred_csv_has_all_six_fields(self):
        df = pd.read_csv(PRED_CSV)
        for f in FIELD_NAMES:
            assert f in df.columns, f"Missing '{f}' in predictions.csv"

    def test_pred_stems_match_test_csv(self):
        pred_df = pd.read_csv(PRED_CSV)
        gt_names = {
            _clean_filename_stem(r["filename"])
            for r in load_ground_truth(str(TEST_CSV))
        }
        for _, row in pred_df.iterrows():
            stem = _clean_filename_stem(str(row["filename"]))
            assert stem in gt_names, f"Prediction stem {stem!r} not in test.csv filenames"

    def test_pred_json_is_list_of_4(self):
        data = json.load(open(PRED_JSON, encoding="utf-8"))
        assert isinstance(data, list) and len(data) == 4

    def test_no_null_filenames_in_predictions(self):
        assert pd.read_csv(PRED_CSV)["filename"].notna().all()


# ── 7. Evaluation Report ──────────────────────────────────────────────────────

@skip_no_report
class TestEvaluationReport:
    def _load(self) -> dict:
        return json.load(open(REPORT_JSON, encoding="utf-8"))

    def test_micro_recall_in_range(self):
        r = self._load()["micro_averaged_recall"]
        assert 0.0 <= r <= 100.0

    def test_six_field_recalls(self):
        assert len(self._load()["field_recalls"]) == 6

    def test_all_six_fields_covered(self):
        reported = {fr["field"] for fr in self._load()["field_recalls"]}
        assert reported == set(FIELD_NAMES)

    def test_recall_pcts_in_range(self):
        for fr in self._load()["field_recalls"]:
            assert 0.0 <= fr["recall_pct"] <= 100.0, f"recall_pct OOB: {fr}"

    def test_overlap_warning_mentions_24158401(self):
        assert "24158401" in self._load().get("overlap_warning", "")

    def test_overlapping_filenames_list(self):
        assert "24158401-Rental-Agreement" in self._load()["overlapping_filenames"]

    def test_per_document_count(self):
        assert len(self._load()["per_document"]) == 4

    def test_total_gt_is_24(self):
        assert self._load()["total_ground_truth_values"] == 24

    def test_field_recall_csv_exists(self):
        assert FIELD_RECALL_CSV.exists()

    def test_field_recall_csv_has_6_rows(self):
        assert len(pd.read_csv(FIELD_RECALL_CSV)) == 6


# ── 8. compute_recall end-to-end ──────────────────────────────────────────────

@skip_no_preds
class TestComputeRecallEndToEnd:
    def _preds(self):
        return pd.read_csv(PRED_CSV).to_dict(orient="records")

    def _gt(self):
        return load_ground_truth(str(TEST_CSV))

    def test_runs_without_error(self):
        assert compute_recall(self._preds(), self._gt()) is not None

    def test_returns_6_field_recalls(self):
        assert len(compute_recall(self._preds(), self._gt()).field_recalls) == 6

    def test_micro_recall_valid(self):
        r = compute_recall(self._preds(), self._gt()).micro_averaged_recall
        assert 0.0 <= r <= 100.0

    def test_4_docs_evaluated(self):
        assert len(compute_recall(self._preds(), self._gt()).per_document) == 4

    def test_match_is_bool_or_none(self):
        for doc in compute_recall(self._preds(), self._gt()).per_document:
            for f in FIELD_NAMES:
                m = doc[f].get("match")
                assert m is None or isinstance(m, bool), \
                    f"'match' must be bool or None, got {type(m)} for '{f}'"

    def test_total_ground_truth_is_24(self):
        assert compute_recall(self._preds(), self._gt()).total_ground_truth_values == 24

    def test_overlap_reported_in_result(self):
        r = compute_recall(self._preds(), self._gt())
        assert "24158401-Rental-Agreement" in r.overlapping_filenames

    def test_exact_matches_lte_total_gt(self):
        r = compute_recall(self._preds(), self._gt())
        assert r.total_exact_matches <= r.total_ground_truth_values


# ── 9. Data Leakage Guard ─────────────────────────────────────────────────────

class TestDataLeakageGuard:
    def test_few_shot_excludes_overlapping_document(self):
        from app.services.extraction_service import build_few_shot_examples
        build_few_shot_examples.cache_clear()
        examples = build_few_shot_examples()
        for ex in examples:
            expected = ex.get("expected_output", {})
            p1 = expected.get("party_one") or ""
            p2 = expected.get("party_two") or ""
            assert not ("Hanumaiah" in p1 and "Vishal Bhardwaj" in p2), (
                "Overlapping document 24158401-Rental-Agreement was included in "
                "few-shot examples — data leakage!"
            )

    def test_cache_clear_does_not_raise(self):
        from app.services.extraction_service import build_few_shot_examples
        build_few_shot_examples.cache_clear()
        assert True
