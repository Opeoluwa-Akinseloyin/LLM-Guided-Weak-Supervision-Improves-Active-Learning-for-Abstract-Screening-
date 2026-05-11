"""
Phase 1 WSS / recall from prediction CSVs for pseudo and random hybrid experiments.

Metrics (same definitions for both):
  - phase1_wss_at_95_ranking: sort by probability_relevant (desc), stop when true recall >= 95%;
      WSS = (1 - k/N) - (1 - 0.95)
  - phase1_wss_at_100_ranking: stop when all relevant found in the evaluated set; WSS = 1 - k/N
  - phase1_recall_0p5: recall at threshold 0.5
  - phase1_actual_wss_0p5: k = # with prob >= 0.5, R = recall; WSS = (1 - k/N) - (1 - R)

Pseudo: hybrid_file_paths_simple.csv -> phase1/predictions_all_documents.csv (all docs, has true_label).

Random: hybrid_random_file_paths.csv -> phase1/predictions.csv (unlabeled pool only).
Ground truth is merged from dataset_config.csv (PMID, Status) -> true_label.
N and WSS are therefore defined on that Phase-1 prediction set, not the full corpus.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score


def resolve_path(raw_path: str, base: Path) -> Path:
    raw_path = raw_path.replace("/", os.sep).replace("\\", os.sep)
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (base / raw_path).resolve()


def ranking_wss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float,
) -> tuple[float, float, float]:
    """Returns (k, proportion_screened, wss)."""
    n = len(y_true)
    n_rel = int(np.sum(y_true == 1))
    if n_rel == 0 or n == 0:
        return float("nan"), float("nan"), float("nan")

    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    cum_rel = np.cumsum(y_sorted == 1)

    if target_recall >= 1.0 - 1e-12:
        needed = n_rel
    else:
        needed = int(np.ceil(target_recall * n_rel))

    idx = np.searchsorted(cum_rel, needed, side="left")
    if idx >= len(cum_rel):
        return float("nan"), float("nan"), float("nan")

    k = float(idx + 1)
    p_screen = k / n
    wss = (1.0 - p_screen) - (1.0 - target_recall)
    return k, p_screen, wss


def compute_metrics_from_labels_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Core metrics; expects aligned arrays."""
    n = len(y_true)
    n_rel = int(np.sum(y_true == 1))
    out: dict = {
        "n_total": n,
        "n_relevant": n_rel,
    }

    k95, p95, wss95 = ranking_wss(y_true, y_prob, 0.95)
    k100, p100, wss100 = ranking_wss(y_true, y_prob, 1.0)

    out["k_at_95_recall_ranking"] = k95
    out["proportion_screened_at_95_ranking"] = p95
    out["phase1_wss_at_95_ranking"] = wss95

    out["k_at_100_recall_ranking"] = k100
    out["proportion_screened_at_100_ranking"] = p100
    out["phase1_wss_at_100_ranking"] = wss100

    y_pred = (y_prob >= 0.5).astype(int)
    out["phase1_recall_0p5"] = (
        recall_score(y_true, y_pred, zero_division=0) if n_rel > 0 else float("nan")
    )

    k_act = float(np.sum(y_pred == 1))
    r_act = out["phase1_recall_0p5"]
    out["k_predicted_positive_0p5"] = k_act
    if n_rel > 0 and not np.isnan(r_act):
        out["phase1_actual_wss_0p5"] = (1.0 - k_act / n) - (1.0 - r_act)
    else:
        out["phase1_actual_wss_0p5"] = float("nan")

    return out


def compute_row_pseudo(pred_path: Path) -> dict:
    out: dict = {"status": "OK", "error": "", "n_pred_rows": np.nan, "n_merged_rows": np.nan}
    if not pred_path.is_file():
        out["status"] = "FILE_NOT_FOUND"
        out["error"] = str(pred_path)
        return out

    try:
        df = pd.read_csv(
            pred_path,
            usecols=lambda c: c in ("true_label", "probability_relevant"),
        )
    except Exception as e:
        out["status"] = "READ_ERROR"
        out["error"] = str(e)
        return out

    if "true_label" not in df.columns or "probability_relevant" not in df.columns:
        out["status"] = "MISSING_COLUMNS"
        out["error"] = "need true_label and probability_relevant"
        return out

    y_true = df["true_label"].to_numpy()
    y_prob = df["probability_relevant"].to_numpy(dtype=float)

    if np.any(np.isnan(y_prob)):
        out["status"] = "NAN_PROB"
        return out

    out["n_pred_rows"] = len(df)
    out["n_merged_rows"] = len(df)
    out.update(compute_metrics_from_labels_probs(y_true, y_prob))
    return out


_gt_cache: dict[tuple[str, str], pd.DataFrame] = {}


def ground_truth_for_dataset(
    name: str,
    category: str,
    config_df: pd.DataFrame,
    base_dir: Path,
) -> tuple[pd.DataFrame | None, str]:
    """Returns DataFrame with columns record_id (int64), true_label (int), or None."""
    key = (name, category)
    if key in _gt_cache:
        return _gt_cache[key], ""

    m = config_df[
        (config_df["name"] == name) & (config_df["category"] == category)
    ]
    if m.empty:
        m = config_df[config_df["name"] == name]
    if m.empty:
        return None, f"dataset not in config: {name} ({category})"

    rel = str(m.iloc[0]["path"])
    path = resolve_path(rel, base_dir)
    if not path.is_file():
        return None, f"data file not found: {path}"

    raw = pd.read_csv(path, usecols=["PMID", "Status"])
    gt = pd.DataFrame(
        {
            "record_id": raw["PMID"].astype(np.int64),
            "true_label": raw["Status"].astype(int),
        }
    )
    _gt_cache[key] = gt
    return gt, ""


def compute_row_random(
    pred_path: Path,
    name: str,
    category: str,
    config_df: pd.DataFrame,
    base_dir: Path,
) -> dict:
    out: dict = {
        "status": "OK",
        "error": "",
        "n_pred_rows": np.nan,
        "n_merged_rows": np.nan,
    }
    if not pred_path.is_file():
        out["status"] = "FILE_NOT_FOUND"
        out["error"] = str(pred_path)
        return out

    gt, err = ground_truth_for_dataset(name, category, config_df, base_dir)
    if gt is None:
        out["status"] = "NO_GROUND_TRUTH"
        out["error"] = err
        return out

    try:
        pred = pd.read_csv(
            pred_path,
            usecols=["record_id", "probability_relevant"],
        )
    except Exception as e:
        out["status"] = "READ_ERROR"
        out["error"] = str(e)
        return out

    pred["record_id"] = pred["record_id"].astype(np.int64)
    merged = pred.merge(gt, on="record_id", how="inner")
    out["n_pred_rows"] = len(pred)
    out["n_merged_rows"] = len(merged)

    if len(merged) == 0:
        out["status"] = "EMPTY_MERGE"
        out["error"] = "no record_id overlap with dataset PMID"
        return out

    if len(merged) < len(pred):
        out["status"] = "MERGE_PARTIAL"
        out["error"] = f"merged {len(merged)}/{len(pred)} rows"

    y_true = merged["true_label"].to_numpy()
    y_prob = merged["probability_relevant"].to_numpy(dtype=float)
    if np.any(np.isnan(y_prob)):
        out["status"] = "NAN_PROB"
        return out

    out.update(compute_metrics_from_labels_probs(y_true, y_prob))
    return out


def run_pseudo(base_dir: Path, paths_csv: Path, output: Path) -> pd.DataFrame:
    index = pd.read_csv(paths_csv)
    rows = []
    for _, row in index.iterrows():
        abs_pred = resolve_path(str(row["predictions_path"]), base_dir)
        metrics = compute_row_pseudo(abs_pred)
        rows.append(
            {
                "init": "pseudo",
                "dataset": row["name"],
                "category": row["category"],
                "mode": row["mode"],
                "pseudo_pct": float(row["pseudo_pct"]),
                "run": np.nan,
                "predictions_path": row["predictions_path"],
                "predictions_abspath": str(abs_pred),
                **metrics,
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output, index=False)
    return out_df


def run_random(
    base_dir: Path,
    paths_csv: Path,
    dataset_config: Path,
    output: Path,
) -> pd.DataFrame:
    config_df = pd.read_csv(dataset_config)
    index = pd.read_csv(paths_csv)
    rows = []
    for _, row in index.iterrows():
        abs_pred = resolve_path(str(row["predictions_path"]), base_dir)
        metrics = compute_row_random(
            abs_pred,
            str(row["name"]),
            str(row["category"]),
            config_df,
            base_dir,
        )
        rows.append(
            {
                "init": "random",
                "dataset": row["name"],
                "category": row["category"],
                "mode": row["mode"],
                "pseudo_pct": np.nan,
                "run": int(row["run"]),
                "predictions_path": row["predictions_path"],
                "predictions_abspath": str(abs_pred),
                **metrics,
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output, index=False)
    return out_df


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Phase 1 WSS from predictions CSVs.")
    parser.add_argument(
        "--experiment",
        choices=("pseudo", "random", "both"),
        default="both",
        help="Which experiment grid to process.",
    )
    parser.add_argument("--base-dir", type=Path, default=root)
    parser.add_argument(
        "--pseudo-paths-csv",
        type=Path,
        default=root / "hybrid_file_paths_simple.csv",
    )
    parser.add_argument(
        "--random-paths-csv",
        type=Path,
        default=root / "hybrid_random_file_paths.csv",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=root / "dataset_config.csv",
    )
    parser.add_argument(
        "--output-pseudo",
        type=Path,
        default=root / "phase1_wss_predictions_all_pseudo.csv",
    )
    parser.add_argument(
        "--output-random",
        type=Path,
        default=root / "phase1_wss_predictions_all_random.csv",
    )
    args = parser.parse_args()

    if args.experiment in ("pseudo", "both"):
        df = run_pseudo(args.base_dir, args.pseudo_paths_csv, args.output_pseudo)
        ok = (df["status"] == "OK").sum()
        print(f"Pseudo: wrote {len(df)} rows -> {args.output_pseudo} (OK: {ok})")
        if (df["status"] != "OK").any():
            print(df[df["status"] != "OK"]["status"].value_counts().to_string())

    if args.experiment in ("random", "both"):
        df = run_random(
            args.base_dir,
            args.random_paths_csv,
            args.dataset_config,
            args.output_random,
        )
        ok = (df["status"] == "OK").sum()
        print(f"Random: wrote {len(df)} rows -> {args.output_random} (OK: {ok})")
        if (df["status"] != "OK").any():
            print(df[df["status"] != "OK"]["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
