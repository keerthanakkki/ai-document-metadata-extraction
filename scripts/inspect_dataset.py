"""
scripts/inspect_dataset.py
───────────────────────────
Dataset inspection utility.

Prints a summary of:
  • Training and test CSV contents
  • Available document files and formats
  • Filename overlap between training and test sets
  • Missing values per field

Usage:
    python scripts/inspect_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from app.config import get_settings


def main() -> None:
    settings = get_settings()

    print("\n" + "=" * 70)
    print("DATASET INSPECTION REPORT")
    print("=" * 70)

    # ── Train CSV ─────────────────────────────────────────────
    print("\n── TRAINING LABELS (train.csv) ──")
    train_df = pd.read_csv(settings.train_csv)
    print(train_df.to_string(index=False))
    print(f"\nShape: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
    print("\nMissing values per column:")
    print(train_df.isnull().sum().to_string())

    # ── Test CSV ──────────────────────────────────────────────
    print("\n── TEST LABELS (test.csv) ──")
    test_df = pd.read_csv(settings.test_csv)
    print(test_df.to_string(index=False))
    print(f"\nShape: {test_df.shape[0]} rows × {test_df.shape[1]} columns")
    print("\nMissing values per column:")
    print(test_df.isnull().sum().to_string())

    # ── Training documents ────────────────────────────────────
    print("\n── TRAINING DOCUMENTS ──")
    train_files = sorted(settings.train_docs_dir.iterdir())
    for f in train_files:
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
    print(f"Total: {len(train_files)} file(s)")

    # ── Test documents ────────────────────────────────────────
    print("\n── TEST DOCUMENTS ──")
    test_files = sorted(settings.test_docs_dir.iterdir())
    for f in test_files:
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
    print(f"Total: {len(test_files)} file(s)")

    # ── Document formats ──────────────────────────────────────
    all_files = train_files + test_files
    ext_counts: dict[str, int] = {}
    for f in all_files:
        ext = f.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    print("\n── DOCUMENT FORMAT BREAKDOWN ──")
    for ext, count in sorted(ext_counts.items()):
        print(f"  {ext}: {count} file(s)")

    # ── Overlap ───────────────────────────────────────────────
    print("\n── FILENAME OVERLAP (train vs test) ──")
    train_stems = {Path(r).stem for r in train_df["File Name"].dropna()}
    test_stems = {Path(r).stem for r in test_df["File Name"].dropna()}
    overlap = sorted(train_stems & test_stems)
    if overlap:
        print(f"  ⚠  {len(overlap)} overlapping filename(s) detected:")
        for name in overlap:
            print(f"     • {name}")
        print(
            "\n  This means the test set shares at least one document with the "
            "training set.\n"
            "  Evaluation results for these documents may be inflated if the "
            "few-shot\n"
            "  examples include that document's ground-truth labels."
        )
    else:
        print("  ✓  No overlap detected.")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
