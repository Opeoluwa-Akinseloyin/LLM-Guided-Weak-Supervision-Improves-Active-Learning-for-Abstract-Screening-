================================================================================
  06 — METRICS AND UTILITIES
================================================================================

These are post-processing and helper scripts. They do not run experiments —
they process, fix, extract, or generate data that the experiment scripts
produce or consume. Run these after experiments are complete, or when
setting up configurations before running.

--------------------------------------------------------------------------------
FILES
--------------------------------------------------------------------------------

Asreview csv config.py
  Generates the dataset configuration CSV (dataset_config.csv) that tells
  the multi-dataset launchers where to find each dataset's criteria file,
  ranking file, and LLM prediction file. Edit the paths in this script when
  setting up the project on a new machine or adding new datasets.

generate_random_baseline_csv.py
  Produces a CSV of random baseline metrics by simulating random document
  ordering (no active learning). Used as the comparison baseline in RQ1-RQ3
  to show what performance would look like without any intelligent screening.

phase1_wss_from_predictions.py
  Computes Phase 1 WSS (Work Saved over Sampling) at the 95% and 100% recall
  thresholds directly from prediction score CSV files. Handles both pseudo-
  labeling predictions (with true labels) and random initialisation predictions
  (unlabelled pool). Outputs a consolidated CSV used by RQ1 analysis.

extract_and_merge_hybrid_metrics_v2.py
  Walks the hybrid_results/ directory tree, finds every COMPLETE_REPORT.txt
  and predictions CSV, and extracts all Phase 1–3 metrics using regex parsing.
  Also computes classifier performance on the remaining (unscreened) documents.
  Outputs a merged CSV in the RQ4 format. Run this after hybrid experiments
  complete and before running RQ4 analysis.

extract_random_hybrid_metrics.py
  Same as above but targets hybrid_results_random/ (experiments that used
  random Phase 1 initialisation with hybrid Phase 2 query strategy).

compare_wss.py
  Validation script. Loads a reference CSV of expected WSS values, reads the
  actual WSS from COMPLETE_REPORT.txt files using regex, and compares them.
  Flags mismatches and shows which source file each value came from. Run this
  after extraction to verify the metric pipeline is correct.

fix_wss.py
  Corrects WSS values in a metrics CSV where the calculation was wrong.
  The correct formula is: WSS = (1 - total_work/N) - (1 - total_recall),
  accounting for predicted positives. Creates a backup of the original CSV,
  applies the correction, and prints a summary of how many rows changed.

fix_random_wss.py
  Same correction as fix_wss.py but applied to the random baseline metrics
  CSV rather than the hybrid/pseudo results CSV.

--------------------------------------------------------------------------------
TYPICAL WORKFLOW
--------------------------------------------------------------------------------

  SETUP (before experiments):
    1. Asreview csv config.py          — generate dataset_config.csv
    2. generate_random_baseline_csv.py — create random baseline

  AFTER EXPERIMENTS (before analysis):
    3. phase1_wss_from_predictions.py         — extract Phase 1 WSS
    4. extract_and_merge_hybrid_metrics_v2.py — extract hybrid metrics
    5. extract_random_hybrid_metrics.py       — extract random-init metrics
    6. compare_wss.py                         — validate extracted values
    7. fix_wss.py / fix_random_wss.py         — correct any WSS errors

================================================================================
