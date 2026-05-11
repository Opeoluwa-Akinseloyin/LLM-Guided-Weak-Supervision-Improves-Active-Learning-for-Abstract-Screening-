"""
Extract Hybrid Metrics Using File Paths CSV

This improved version:
1. Uses hybrid_file_paths_simple.csv to know exactly where files are
2. Extracts all metrics from COMPLETE_REPORT.txt
3. Extracts Phase 1 classifier metrics from predictions_all_documents.csv
4. Creates output matching rq4_raw_metrics_complete.csv format
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def parse_complete_report(report_path):
    """
    Parse COMPLETE_REPORT.txt for all available metrics

    Args:
        report_path: Path to COMPLETE_REPORT.txt file

    Returns:
        Dictionary of extracted metrics
    """
    metrics = {}

    if not os.path.exists(report_path):
        return metrics

    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Dataset information
        match = re.search(r'Total records: (\d+)', content)
        if match:
            metrics['total_records'] = int(match.group(1))

        match = re.search(r'Total relevant: (\d+)', content)
        if match:
            metrics['total_relevant'] = int(match.group(1))

        match = re.search(r'Prevalence: ([\d.]+)%', content)
        if match:
            metrics['prevalence'] = float(match.group(1))

        # Phase 1: Pseudo-labeling
        match = re.search(r'Pseudo-labeled: (\d+) \(([\d.]+)%\)', content)
        if match:
            metrics['pseudo_labeled_count'] = int(match.group(1))
            metrics['pseudo_labeled_pct'] = float(match.group(2))

        match = re.search(r'Pseudo-POSITIVE: (\d+)', content)
        if match:
            metrics['pseudo_positive'] = int(match.group(1))

        match = re.search(r'Pseudo-NEGATIVE: (\d+)', content)
        if match:
            metrics['pseudo_negative'] = int(match.group(1))

        # Phase 2: Active learning
        match = re.search(r'Iterations: (\d+)', content)
        if match:
            metrics['phase2_iterations'] = int(match.group(1))

        match = re.search(r'Records screened: (\d+) \(([\d.]+)%\)', content)
        if match:
            metrics['phase2_screened'] = int(match.group(1))
            metrics['phase2_screened_pct'] = float(match.group(2))

        match = re.search(r'Screening recall: ([\d.]+)%', content)
        if match:
            metrics['phase2_screening_recall'] = float(match.group(1))

        match = re.search(r'Total recall \(with classifier\): ([\d.]+)%', content)
        if match:
            metrics['phase2_total_recall'] = float(match.group(1))

        # Phase 2 Classifier Performance
        match = re.search(r'AUC-ROC: ([\d.]+)', content)
        if match:
            metrics['phase2_auc_roc'] = float(match.group(1))

        match = re.search(r'Confusion: TP=(\d+), TN=(\d+), FP=(\d+), FN=(\d+)', content)
        if match:
            metrics['phase2_tp'] = float(match.group(1))
            metrics['phase2_tn'] = float(match.group(2))
            metrics['phase2_fp'] = float(match.group(3))
            metrics['phase2_fn'] = float(match.group(4))

        # Phase 2 Training Set Composition
        match = re.search(r'Total training samples: (\d+)', content)
        if match:
            metrics['total_training_samples'] = float(match.group(1))

        match = re.search(r'Truly labeled \(queried\): (\d+) \(([\d.]+)%\)', content)
        if match:
            metrics['truly_labeled'] = float(match.group(1))
            metrics['truly_labeled_pct'] = float(match.group(2))

        match = re.search(r'Still pseudo-labeled: (\d+) \(([\d.]+)%\)', content)
        if match:
            metrics['still_pseudo_labeled'] = float(match.group(1))

        # Phase 3: Key milestones
        match = re.search(r'Iteration at 95% recall: (\d+)', content)
        if match:
            metrics['iteration_at_95_recall'] = int(match.group(1))

        match = re.search(r'Iteration at 100% recall: (\d+)', content)
        if match:
            metrics['iteration_at_100_recall'] = int(match.group(1))

        match = re.search(r'Iterations \(95%→100%\): (\d+)', content)
        if match:
            metrics['iterations_95_to_100'] = int(match.group(1))

        # Work savings
        match = re.search(r'At Phase 2: ([\d.]+)% screened, ([\d.]+)% recall, WSS=([\d.]+)', content)
        if match:
            metrics['wss_at_phase2'] = float(match.group(3))

        match = re.search(r'Work saved vs random: ([\d.]+)%', content)
        if match:
            metrics['work_saved_vs_random'] = float(match.group(1))

        # Calculate iterations_95_to_100 if not found
        if 'iterations_95_to_100' not in metrics:
            if 'iteration_at_95_recall' in metrics and 'iteration_at_100_recall' in metrics:
                metrics['iterations_95_to_100'] = (
                    metrics['iteration_at_100_recall'] - metrics['iteration_at_95_recall']
                )

    except Exception as e:
        print(f"      ⚠️  Error parsing {report_path}: {str(e)}")

    return metrics


def calculate_phase1_metrics_from_predictions(predictions_path):
    """
    Calculate Phase 1 classifier metrics from predictions_all_documents.csv

    Args:
        predictions_path: Path to predictions_all_documents.csv

    Returns:
        Dictionary with phase1_precision, phase1_recall, phase1_f1, phase1_auc
    """
    metrics = {}

    if not os.path.exists(predictions_path):
        return metrics

    try:
        pred_df = pd.read_csv(predictions_path)

        # Check for required columns
        if 'true_label' not in pred_df.columns or 'probability_relevant' not in pred_df.columns:
            return metrics

        y_true = pred_df['true_label'].values
        y_prob = pred_df['probability_relevant'].values

        # Binary predictions using 0.5 threshold
        y_pred = (y_prob >= 0.5).astype(int)

        # Calculate metrics
        metrics['phase1_precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['phase1_recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['phase1_f1'] = f1_score(y_true, y_pred, zero_division=0)

        # Calculate AUC if we have both classes
        if len(np.unique(y_true)) > 1:
            try:
                metrics['phase1_auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['phase1_auc'] = np.nan
        else:
            metrics['phase1_auc'] = np.nan

    except Exception as e:
        print(f"      ⚠️  Error calculating Phase 1 metrics from {predictions_path}: {str(e)}")

    return metrics


def map_hybrid_mode_to_method_name(mode_name):
    """
    Map hybrid mode name to method name for consistency with existing data

    Args:
        mode_name: Name from hybrid experiments (e.g., "phase_switch_cert_to_uncert")

    Returns:
        Standardized method name
    """
    mode_mapping = {
        'phase_switch_cert_to_uncert': 'phase_switch_c2u',
        'phase_switch_uncert_to_cert': 'phase_switch_u2c',
        'alternating_cert_start': 'alternating_c_start',
        'alternating_uncert_start': 'alternating_u_start'
    }

    return mode_mapping.get(mode_name, mode_name)


def extract_metrics_from_file_paths(file_paths_csv):
    """
    Extract all metrics using the file paths CSV

    Args:
        file_paths_csv: Path to hybrid_file_paths_simple.csv

    Returns:
        DataFrame with all extracted metrics
    """
    print(f"\n📂 Loading file paths from: {file_paths_csv}")

    paths_df = pd.read_csv(file_paths_csv)
    print(f"   Found {len(paths_df)} experiments to process")
    print(f"   Datasets: {paths_df['name'].nunique()} unique")
    print(f"   Modes: {paths_df['mode'].unique().tolist()}")
    print(f"   Percentages: {sorted(paths_df['pseudo_pct'].unique())}")

    all_metrics = []
    success_count = 0
    missing_report = 0
    missing_predictions = 0

    print(f"\n🔍 Extracting metrics from files...")

    for idx, row in paths_df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"   Progress: {idx + 1}/{len(paths_df)} experiments processed...")

        # Base metadata
        metrics = {
            'dataset': row['name'],
            'method': map_hybrid_mode_to_method_name(row['mode']),
            'init_type': 'pseudo',  # All hybrid experiments use pseudo-labeling
            'ratio': row['pseudo_pct']
        }

        # Extract from COMPLETE_REPORT.txt
        report_path = row['complete_report_path']
        report_metrics = parse_complete_report(report_path)

        if report_metrics:
            metrics.update(report_metrics)
            success_count += 1
        else:
            missing_report += 1
            if missing_report <= 5:
                print(f"      ⚠️  Missing report: {report_path}")

        # Extract Phase 1 metrics from predictions
        predictions_path = row['predictions_path']
        phase1_metrics = calculate_phase1_metrics_from_predictions(predictions_path)

        if phase1_metrics:
            metrics.update(phase1_metrics)
        else:
            missing_predictions += 1

        # Calculate derived metric: wss_at_95_theoretical
        if 'iteration_at_95_recall' in metrics and 'total_records' in metrics:
            if metrics['total_records'] > 0:
                screened_at_95 = metrics['iteration_at_95_recall'] / metrics['total_records']
                metrics['wss_at_95_theoretical'] = max(0, 0.95 - screened_at_95)

        # Add any missing columns with NaN
        for col in ['total_records', 'total_relevant', 'prevalence',
                    'pseudo_labeled_count', 'pseudo_labeled_pct',
                    'pseudo_positive', 'pseudo_negative',
                    'phase2_iterations', 'phase2_screened', 'phase2_screened_pct',
                    'phase2_screening_recall', 'phase2_total_recall',
                    'phase2_tp', 'phase2_tn', 'phase2_fp', 'phase2_fn',
                    'total_training_samples', 'truly_labeled', 'truly_labeled_pct',
                    'still_pseudo_labeled',
                    'iteration_at_95_recall', 'iteration_at_100_recall',
                    'iterations_95_to_100',
                    'wss_at_phase2', 'work_saved_vs_random',
                    'phase1_recall', 'phase1_precision', 'phase1_f1', 'phase1_auc',
                    'wss_at_95_theoretical']:
            if col not in metrics:
                metrics[col] = np.nan

        all_metrics.append(metrics)

    # Create DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # Ensure column order matches rq4_raw_metrics_complete.csv
    column_order = [
        'dataset', 'method', 'init_type', 'ratio',
        'total_records', 'total_relevant', 'prevalence',
        'pseudo_labeled_count', 'pseudo_labeled_pct',
        'pseudo_positive', 'pseudo_negative',
        'phase2_iterations', 'phase2_screened', 'phase2_screened_pct',
        'phase2_screening_recall', 'phase2_total_recall',
        'phase2_tp', 'phase2_tn', 'phase2_fp', 'phase2_fn',
        'total_training_samples', 'truly_labeled', 'truly_labeled_pct',
        'still_pseudo_labeled',
        'iteration_at_95_recall', 'iteration_at_100_recall', 'iterations_95_to_100',
        'wss_at_phase2', 'work_saved_vs_random',
        'phase1_recall', 'phase1_precision', 'phase1_f1', 'phase1_auc',
        'wss_at_95_theoretical'
    ]

    metrics_df = metrics_df[column_order]

    # Summary
    print(f"\n✅ Extraction complete!")
    print(f"   Total experiments: {len(metrics_df)}")
    print(f"   Successfully extracted from COMPLETE_REPORT.txt: {success_count}")
    print(f"   Missing COMPLETE_REPORT.txt: {missing_report}")
    print(f"   Missing predictions (Phase 1 metrics): {missing_predictions}")

    # Check data completeness
    print(f"\n📊 Data Completeness:")
    key_cols = ['phase2_iterations', 'iteration_at_95_recall',
                'iteration_at_100_recall', 'work_saved_vs_random']
    for col in key_cols:
        n_valid = metrics_df[col].notna().sum()
        pct = (n_valid / len(metrics_df)) * 100
        print(f"   {col:30s}: {n_valid:4d}/{len(metrics_df):4d} ({pct:5.1f}%)")

    return metrics_df


def merge_with_existing_csv(existing_csv_path, new_metrics_df, output_path):
    """
    Merge new hybrid metrics with existing CSV

    Args:
        existing_csv_path: Path to existing rq4_raw_metrics_complete.csv
        new_metrics_df: DataFrame with new metrics
        output_path: Path to save merged CSV

    Returns:
        Merged DataFrame
    """
    print(f"\n🔗 Merging with existing CSV...")

    if not os.path.exists(existing_csv_path):
        print(f"   ⚠️  Existing CSV not found: {existing_csv_path}")
        print(f"   Saving new metrics only to: {output_path}")
        new_metrics_df.to_csv(output_path, index=False)
        return new_metrics_df

    existing_df = pd.read_csv(existing_csv_path)
    print(f"   Existing: {len(existing_df)} rows")
    print(f"      Methods: {sorted(existing_df['method'].unique())}")
    print(f"      Datasets: {existing_df['dataset'].nunique()} unique")

    # Ensure columns match
    existing_columns = existing_df.columns.tolist()
    new_metrics_df = new_metrics_df[existing_columns]

    # Concatenate
    merged_df = pd.concat([existing_df, new_metrics_df], ignore_index=True)

    # Sort by dataset, method, ratio
    merged_df = merged_df.sort_values(['dataset', 'method', 'ratio'], ignore_index=True)

    print(f"   New: {len(new_metrics_df)} rows")
    print(f"      Methods: {sorted(new_metrics_df['method'].unique())}")

    print(f"   Merged: {len(merged_df)} rows")
    print(f"      Methods: {sorted(merged_df['method'].unique())}")

    # Save
    merged_df.to_csv(output_path, index=False)
    print(f"   💾 Saved to: {output_path}")

    return merged_df


def validate_results(metrics_df):
    """Print validation statistics"""
    print(f"\n📊 Validation Summary:")
    print(f"=" * 70)

    print(f"\nOverview:")
    print(f"   Total experiments: {len(metrics_df)}")
    print(f"   Unique datasets: {metrics_df['dataset'].nunique()}")
    print(f"   Unique methods: {metrics_df['method'].nunique()}")
    print(f"   Ratios tested: {sorted(metrics_df['ratio'].unique())}")

    print(f"\nExperiments by method:")
    method_counts = metrics_df.groupby('method').size().sort_index()
    for method, count in method_counts.items():
        print(f"   {method:25s}: {count:4d} experiments")

    print(f"\nPhase 1 Metrics Availability:")
    for method in sorted(metrics_df['method'].unique()):
        method_data = metrics_df[metrics_df['method'] == method]
        n_with_phase1 = (~method_data['phase1_recall'].isna()).sum()
        n_total = len(method_data)
        pct = (n_with_phase1 / n_total * 100) if n_total > 0 else 0
        print(f"   {method:25s}: {n_with_phase1:4d}/{n_total:4d} ({pct:5.1f}%)")

    print(f"\nKey Metrics Summary (mean ± std):")
    print(f"   {'Method':<25s} {'Work Saved':<15s} {'Iter@95%':<15s} {'Iter@100%':<15s}")
    print(f"   {'-'*70}")
    for method in sorted(metrics_df['method'].unique()):
        method_data = metrics_df[metrics_df['method'] == method]

        ws_mean = method_data['work_saved_vs_random'].mean()
        ws_std = method_data['work_saved_vs_random'].std()

        i95_mean = method_data['iteration_at_95_recall'].mean()
        i95_std = method_data['iteration_at_95_recall'].std()

        i100_mean = method_data['iteration_at_100_recall'].mean()
        i100_std = method_data['iteration_at_100_recall'].std()

        print(f"   {method:<25s} {ws_mean:5.1f}±{ws_std:4.1f}%     "
              f"{i95_mean:6.0f}±{i95_std:5.0f}    {i100_mean:6.0f}±{i100_std:5.0f}")


def main():
    """Main execution"""
    print("=" * 70)
    print("EXTRACT HYBRID METRICS USING FILE PATHS CSV")
    print("=" * 70)

    # Configuration
    FILE_PATHS_CSV = 'hybrid_file_paths_simple.csv'
    EXISTING_CSV = 'RQ4/rq4_extracted_metrics/rq4_raw_metrics_complete.csv'
    OUTPUT_HYBRID_ONLY = 'RQ4/rq4_extracted_metrics/hybrid_metrics_extracted.csv'
    OUTPUT_MERGED = 'RQ4/rq4_extracted_metrics/rq4_raw_metrics_with_all_6_methods.csv'

    # Step 1: Check if file paths CSV exists
    if not os.path.exists(FILE_PATHS_CSV):
        print(f"\n❌ Error: Cannot find file paths CSV: {FILE_PATHS_CSV}")
        print(f"   Please ensure the file exists in the current directory.")
        return None

    # Step 2: Extract metrics using file paths
    print(f"\n{'='*70}")
    print("STEP 1: EXTRACTING METRICS FROM HYBRID EXPERIMENTS")
    print(f"{'='*70}")

    hybrid_metrics_df = extract_metrics_from_file_paths(FILE_PATHS_CSV)

    if hybrid_metrics_df is None or len(hybrid_metrics_df) == 0:
        print(f"\n❌ Failed to extract metrics.")
        return None

    # Step 3: Save hybrid-only results
    print(f"\n{'='*70}")
    print("STEP 2: SAVING HYBRID METRICS")
    print(f"{'='*70}")

    hybrid_metrics_df.to_csv(OUTPUT_HYBRID_ONLY, index=False)
    print(f"   💾 Saved hybrid-only metrics to: {OUTPUT_HYBRID_ONLY}")

    # Step 4: Merge with existing CSV if available
    print(f"\n{'='*70}")
    print("STEP 3: MERGING WITH EXISTING CSV")
    print(f"{'='*70}")

    merged_df = merge_with_existing_csv(EXISTING_CSV, hybrid_metrics_df, OUTPUT_MERGED)

    # Step 5: Validate
    print(f"\n{'='*70}")
    print("STEP 4: VALIDATION")
    print(f"{'='*70}")

    validate_results(merged_df)

    # Final summary
    print(f"\n{'='*70}")
    print("✅ SUCCESS!")
    print(f"{'='*70}")

    if os.path.exists(EXISTING_CSV):
        existing_count = len(pd.read_csv(EXISTING_CSV))
        print(f"\nOriginal CSV: {EXISTING_CSV}")
        print(f"   - {existing_count} experiments")

    print(f"\nHybrid metrics extracted:")
    print(f"   - {len(hybrid_metrics_df)} experiments")
    print(f"   - 4 methods: {sorted(hybrid_metrics_df['method'].unique())}")
    print(f"   - Saved to: {OUTPUT_HYBRID_ONLY}")

    print(f"\nMerged CSV:")
    print(f"   - {len(merged_df)} total experiments")
    print(f"   - {merged_df['method'].nunique()} methods: {sorted(merged_df['method'].unique())}")
    print(f"   - Saved to: {OUTPUT_MERGED}")

    print(f"\n{'='*70}")

    return merged_df


if __name__ == "__main__":
    merged_df = main()