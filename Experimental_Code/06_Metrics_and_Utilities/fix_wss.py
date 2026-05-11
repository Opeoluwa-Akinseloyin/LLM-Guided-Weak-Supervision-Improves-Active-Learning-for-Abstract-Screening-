"""
Fix wss_at_phase2 in rq4_raw_metrics_with_all_6_methods.csv

The original value was extracted from COMPLETE_REPORT.txt which used
screening_recall instead of total_recall_with_classifier in the WSS formula.

Correct formula:
    total_work = phase2_screened + phase2_tp + phase2_fp
    wss_at_phase2 = (1 - total_work / total_records) - (1 - phase2_total_recall / 100)

This accounts for:
  - predicted positives (TP+FP) that still need manual checking
  - the full recall achieved including the classifier's contributions
"""

import pandas as pd
import numpy as np
import shutil
import os

BASE_DIR   = r"c:\Users\akinseloyo\Documents\Simulation"
CSV_PATH   = os.path.join(BASE_DIR, "RQ4", "rq4_extracted_metrics",
                          "rq4_raw_metrics_with_all_6_methods.csv")
BACKUP_PATH = CSV_PATH.replace(".csv", "_BACKUP_original_wss.csv")
OUTPUT_PATH = CSV_PATH.replace(".csv", "_CORRECTED.csv")


def compute_correct_wss(row):
    """
    Correct WSS formula:
      total_work = screened + TP + FP  (all docs a reviewer must still check)
      WSS = (1 - total_work/N) - (1 - total_recall)
    Returns NaN if required columns are missing.
    """
    try:
        n        = float(row["total_records"])
        screened = float(row["phase2_screened"])
        tp       = float(row["phase2_tp"])
        fp       = float(row["phase2_fp"])
        recall   = float(row["phase2_total_recall"]) / 100.0   # convert % → fraction

        if n == 0 or any(np.isnan(v) for v in [n, screened, tp, fp, recall]):
            return np.nan

        total_work = screened + tp + fp
        wss = (1.0 - total_work / n) - (1.0 - recall)
        return round(wss, 3)
    except (TypeError, ValueError):
        return np.nan


def main():
    print("=" * 70)
    print("FIX wss_at_phase2 — CORRECT FORMULA")
    print("=" * 70)

    # ── Load CSV ──────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded: {len(df)} rows")

    # ── Backup original ───────────────────────────────────────────────────
    shutil.copy2(CSV_PATH, BACKUP_PATH)
    print(f"Backup saved to: {BACKUP_PATH}")

    # ── Compute correct WSS ───────────────────────────────────────────────
    df["wss_corrected"] = df.apply(compute_correct_wss, axis=1)

    # ── Identify affected rows ────────────────────────────────────────────
    # Rows where both old and new values exist, and they differ by > 0.001
    has_both   = df["wss_at_phase2"].notna() & df["wss_corrected"].notna()
    diff       = (df["wss_at_phase2"] - df["wss_corrected"]).abs()
    changed    = has_both & (diff > 0.001)
    unchanged  = has_both & (diff <= 0.001)
    only_new   = df["wss_at_phase2"].isna() & df["wss_corrected"].notna()

    print(f"\n{'='*70}")
    print(f"IMPACT SUMMARY")
    print(f"{'='*70}")
    print(f"  Total rows              : {len(df)}")
    print(f"  Rows unchanged (correct): {unchanged.sum()}")
    print(f"  Rows CORRECTED          : {changed.sum()}")
    print(f"  Rows newly filled       : {only_new.sum()}")
    print(f"  Rows with NaN (skipped) : {df['wss_corrected'].isna().sum()}")

    if changed.sum() > 0:
        print(f"\n{'='*70}")
        print(f"CORRECTED ROWS DETAIL ({changed.sum()} rows):")
        print(f"{'='*70}")
        cols = ["dataset", "method", "ratio",
                "phase2_screened", "phase2_tp", "phase2_fp",
                "phase2_screening_recall", "phase2_total_recall",
                "total_records",
                "wss_at_phase2", "wss_corrected"]
        detail = df.loc[changed, cols].copy()
        detail["diff"] = (detail["wss_corrected"] - detail["wss_at_phase2"]).round(4)
        print(detail.sort_values("diff", ascending=False).to_string(index=False))

        # Per-method breakdown
        print(f"\nBy method:")
        method_counts = df.loc[changed].groupby("method").size().sort_index()
        for m, c in method_counts.items():
            print(f"  {m:<25s} : {c} rows corrected")

    # ── Apply correction ──────────────────────────────────────────────────
    df.loc[changed | only_new, "wss_at_phase2"] = \
        df.loc[changed | only_new, "wss_corrected"]

    df = df.drop(columns=["wss_corrected"])

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n{'='*70}")
    print(f"Corrected CSV saved to: {OUTPUT_PATH}")
    print(f"Original preserved at : {BACKUP_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
