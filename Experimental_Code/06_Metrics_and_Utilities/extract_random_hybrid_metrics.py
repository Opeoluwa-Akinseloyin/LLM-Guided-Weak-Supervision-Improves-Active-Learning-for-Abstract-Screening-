"""
Extract Random Hybrid Metrics from COMPLETE_REPORT.txt

This script:
1. Uses hybrid_random_file_paths.csv to locate all result files
2. Parses COMPLETE_REPORT.txt for Phase 1, Phase 2, and Phase 3 metrics
   - Phase 1 classifier metrics (Precision, Recall, F1, AUC) from the Phase 1 section
   - Phase 2/3 metrics (iterations, recall, WSS) from the rest of the report
3. Outputs in the same format as random_combined_metrics.csv
4. Optionally merges with existing random_combined_metrics.csv

Output columns match existing random format:
    dataset, method, run, phase1_recall, phase1_precision, phase1_f1, phase1_auc,
    iteration_at_95_recall, iteration_at_100_recall, iterations_95_to_100,
    wss_at_95_theoretical, phase2_total_recall, wss_at_phase2, phase2_screened_pct
"""

import pandas as pd
import numpy as np
import os
import re


def parse_complete_report(report_path):
    """
    Parse COMPLETE_REPORT.txt for all metrics including Phase 1 classifier performance.

    The report now includes a "PHASE 1: PRIOR KNOWLEDGE" section with classifier
    metrics, matching the pattern from asreview_with_artifacts REPORT.txt.

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

        # --- DATASET ---
        match = re.search(r'Total records: (\d+)', content)
        if match:
            metrics['total_records'] = int(match.group(1))

        match = re.search(r'Total relevant: (\d+)', content)
        if match:
            metrics['total_relevant'] = int(match.group(1))

        match = re.search(r'Prevalence: ([\d.]+)%', content)
        if match:
            metrics['prevalence'] = float(match.group(1))

        # --- PHASE 1 CLASSIFIER METRICS ---
        # Extract specifically from the "Phase 1 Classifier Performance" sub-section
        phase1_classifier_block = re.search(
            r'Phase 1 Classifier Performance.*?(?=\n\n|\nPHASE 2)', content, re.DOTALL
        )
        if phase1_classifier_block:
            p1c = phase1_classifier_block.group(0)

            p1_precision = re.search(r'Precision: ([\d.]+)', p1c)
            if p1_precision:
                metrics['phase1_precision'] = float(p1_precision.group(1))

            p1_recall = re.search(r'Recall: ([\d.]+)', p1c)
            if p1_recall:
                metrics['phase1_recall'] = float(p1_recall.group(1))

            p1_f1 = re.search(r'F1-Score: ([\d.]+)', p1c)
            if p1_f1:
                metrics['phase1_f1'] = float(p1_f1.group(1))

            p1_auc = re.search(r'AUC-ROC: ([\d.]+)', p1c)
            if p1_auc:
                metrics['phase1_auc'] = float(p1_auc.group(1))

        # --- PHASE 2 SUMMARY ---
        match = re.search(r'Iterations: (\d+)', content)
        if match:
            metrics['phase2_iterations'] = int(match.group(1))

        match = re.search(r'Records screened: (\d+) \(([\d.]+)%\)', content)
        if match:
            metrics['phase2_screened'] = int(match.group(1))
            metrics['phase2_screened_pct'] = float(match.group(2))

        # "Recall achieved" (random init format)
        match = re.search(r'Recall achieved: ([\d.]+)%', content)
        if match:
            metrics['phase2_recall_achieved'] = float(match.group(1))

        # --- KEY MILESTONES ---
        match = re.search(r'Iteration at 95% recall: (\d+)', content)
        if match:
            metrics['iteration_at_95_recall'] = int(match.group(1))

        match = re.search(r'Iteration at 100% recall: (\d+)', content)
        if match:
            metrics['iteration_at_100_recall'] = int(match.group(1))

        # Iterations 95->100 (handle different arrow encodings)
        match = re.search(r'Iterations \(95%.*?100%\): (\d+)', content)
        if match:
            metrics['iterations_95_to_100'] = int(match.group(1))

        # --- WORK SAVINGS ---
        match = re.search(r'Actual WSS \(Phase 2\): ([\d.]+)', content)
        if match:
            metrics['wss_at_phase2'] = float(match.group(1))

        match = re.search(r'Work saved vs random: ([\d.]+)%', content)
        if match:
            metrics['work_saved_vs_random'] = float(match.group(1))

        # --- DERIVED ---
        if 'iterations_95_to_100' not in metrics:
            if 'iteration_at_95_recall' in metrics and 'iteration_at_100_recall' in metrics:
                metrics['iterations_95_to_100'] = (
                    metrics['iteration_at_100_recall'] - metrics['iteration_at_95_recall']
                )

    except Exception as e:
        print(f"      Warning: Error parsing {report_path}: {str(e)}")

    return metrics


def map_hybrid_mode_to_method_name(mode_name):
    """Map hybrid mode name to short method name."""
    mode_mapping = {
        'phase_switch_cert_to_uncert': 'phase_switch_c2u',
        'phase_switch_uncert_to_cert': 'phase_switch_u2c',
        'alternating_cert_start': 'alternating_c_start',
        'alternating_uncert_start': 'alternating_u_start'
    }
    return mode_mapping.get(mode_name, mode_name)


def extract_random_hybrid_metrics(file_paths_csv):
    """
    Extract all metrics from random hybrid experiments using file paths CSV.

    Phase 1 classifier metrics are now parsed directly from COMPLETE_REPORT.txt
    (the report includes a Phase 1 section with Precision, Recall, F1, AUC-ROC).

    Args:
        file_paths_csv: Path to hybrid_random_file_paths.csv

    Returns:
        DataFrame with extracted metrics matching random_combined_metrics.csv format
    """
    print(f"\n{'='*70}")
    print(f"EXTRACTING RANDOM HYBRID METRICS")
    print(f"{'='*70}")

    print(f"\nLoading file paths from: {file_paths_csv}")
    paths_df = pd.read_csv(file_paths_csv)

    print(f"   Found {len(paths_df)} experiments to process")
    print(f"   Datasets: {paths_df['name'].nunique()} unique")
    print(f"   Modes: {paths_df['mode'].unique().tolist()}")
    print(f"   Runs: {sorted(paths_df['run'].unique())}")

    all_metrics = []
    success_count = 0
    missing_report = 0

    print(f"\nExtracting metrics from COMPLETE_REPORT.txt files...")

    for idx, row in paths_df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"   Progress: {idx + 1}/{len(paths_df)} experiments processed...")

        # Base metadata
        record = {
            'dataset': row['name'],
            'method': map_hybrid_mode_to_method_name(row['mode']),
            'run': row['run'],
        }

        # Parse COMPLETE_REPORT.txt (contains both Phase 1 and Phase 2/3 metrics)
        report_path = row['complete_report_path']
        report_metrics = parse_complete_report(report_path)

        if report_metrics:
            success_count += 1

            # Phase 1 classifier metrics (from Phase 1 section of report)
            record['phase1_recall'] = report_metrics.get('phase1_recall', np.nan)
            record['phase1_precision'] = report_metrics.get('phase1_precision', np.nan)
            record['phase1_f1'] = report_metrics.get('phase1_f1', np.nan)
            record['phase1_auc'] = report_metrics.get('phase1_auc', np.nan)

            # Phase 2/3 metrics
            record['iteration_at_95_recall'] = report_metrics.get('iteration_at_95_recall', np.nan)
            record['iteration_at_100_recall'] = report_metrics.get('iteration_at_100_recall', np.nan)
            record['iterations_95_to_100'] = report_metrics.get('iterations_95_to_100', np.nan)

            # WSS@95 theoretical
            total_records = report_metrics.get('total_records', None)
            iter_95 = report_metrics.get('iteration_at_95_recall', None)
            if total_records and iter_95 and total_records > 0:
                screened_at_95 = iter_95 / total_records
                record['wss_at_95_theoretical'] = round(max(0, 0.95 - screened_at_95), 3)
            else:
                record['wss_at_95_theoretical'] = np.nan

            record['phase2_total_recall'] = report_metrics.get('phase2_recall_achieved', np.nan)
            record['wss_at_phase2'] = report_metrics.get('wss_at_phase2', np.nan)
            record['phase2_screened_pct'] = report_metrics.get('phase2_screened_pct', np.nan)

        else:
            missing_report += 1
            if missing_report <= 5:
                print(f"      Missing report: {report_path}")

            for col in ['phase1_recall', 'phase1_precision', 'phase1_f1', 'phase1_auc',
                         'iteration_at_95_recall', 'iteration_at_100_recall',
                         'iterations_95_to_100', 'wss_at_95_theoretical',
                         'phase2_total_recall', 'wss_at_phase2', 'phase2_screened_pct']:
                record[col] = np.nan

        all_metrics.append(record)

    # Create DataFrame with column order matching random_combined_metrics.csv
    column_order = [
        'dataset', 'method', 'run',
        'phase1_recall', 'phase1_precision', 'phase1_f1', 'phase1_auc',
        'iteration_at_95_recall', 'iteration_at_100_recall', 'iterations_95_to_100',
        'wss_at_95_theoretical',
        'phase2_total_recall', 'wss_at_phase2', 'phase2_screened_pct'
    ]

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df[column_order]

    # Summary
    print(f"\nExtraction complete!")
    print(f"   Total experiments: {len(metrics_df)}")
    print(f"   Successfully extracted: {success_count}")
    print(f"   Missing COMPLETE_REPORT.txt: {missing_report}")

    # Data completeness
    print(f"\nData Completeness:")
    key_cols = ['phase1_auc', 'iteration_at_95_recall', 'iteration_at_100_recall',
                'wss_at_phase2', 'phase2_screened_pct']
    for col in key_cols:
        n_valid = metrics_df[col].notna().sum()
        pct = (n_valid / len(metrics_df)) * 100
        print(f"   {col:30s}: {n_valid:4d}/{len(metrics_df):4d} ({pct:5.1f}%)")

    return metrics_df


def merge_with_existing(existing_csv_path, new_df, output_path):
    """Merge new hybrid metrics with existing random_combined_metrics.csv"""
    print(f"\n{'='*70}")
    print(f"MERGING WITH EXISTING RANDOM METRICS")
    print(f"{'='*70}")

    if not os.path.exists(existing_csv_path):
        print(f"   Existing CSV not found: {existing_csv_path}")
        print(f"   Saving new metrics only.")
        new_df.to_csv(output_path, index=False)
        return new_df

    existing_df = pd.read_csv(existing_csv_path)
    print(f"   Existing: {len(existing_df)} rows")
    print(f"      Methods: {sorted(existing_df['method'].unique())}")
    print(f"      Datasets: {existing_df['dataset'].nunique()}")

    print(f"   New hybrid: {len(new_df)} rows")
    print(f"      Methods: {sorted(new_df['method'].unique())}")

    # Ensure columns match
    new_df = new_df[existing_df.columns.tolist()]

    # Concatenate and sort
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    merged_df = merged_df.sort_values(['dataset', 'method', 'run'], ignore_index=True)

    print(f"   Merged: {len(merged_df)} rows")
    print(f"      Methods: {sorted(merged_df['method'].unique())}")

    merged_df.to_csv(output_path, index=False)
    print(f"   Saved to: {output_path}")

    return merged_df


def validate_results(metrics_df):
    """Print validation statistics"""
    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*70}")

    print(f"\nOverview:")
    print(f"   Total experiments: {len(metrics_df)}")
    print(f"   Unique datasets: {metrics_df['dataset'].nunique()}")
    print(f"   Unique methods: {metrics_df['method'].nunique()}")
    print(f"   Methods: {sorted(metrics_df['method'].unique())}")

    print(f"\nExperiments by method:")
    method_counts = metrics_df.groupby('method').size().sort_index()
    for method, count in method_counts.items():
        print(f"   {method:25s}: {count:4d} experiments")

    print(f"\nPhase 1 Metrics Availability:")
    for method in sorted(metrics_df['method'].unique()):
        method_data = metrics_df[metrics_df['method'] == method]
        n_with_phase1 = method_data['phase1_auc'].notna().sum()
        n_total = len(method_data)
        pct = (n_with_phase1 / n_total * 100) if n_total > 0 else 0
        print(f"   {method:25s}: {n_with_phase1:4d}/{n_total:4d} ({pct:5.1f}%)")

    print(f"\nKey Metrics Summary (mean +/- std):")
    print(f"   {'Method':<25s} {'Screened%':<15s} {'Iter@95%':<15s} {'Iter@100%':<15s} {'WSS@P2':<12s}")
    print(f"   {'-'*75}")
    for method in sorted(metrics_df['method'].unique()):
        md = metrics_df[metrics_df['method'] == method]

        scr_mean = md['phase2_screened_pct'].mean()
        scr_std = md['phase2_screened_pct'].std()

        i95_mean = md['iteration_at_95_recall'].mean()
        i95_std = md['iteration_at_95_recall'].std()

        i100_mean = md['iteration_at_100_recall'].mean()
        i100_std = md['iteration_at_100_recall'].std()

        wss_mean = md['wss_at_phase2'].mean()
        wss_std = md['wss_at_phase2'].std()

        print(f"   {method:<25s} {scr_mean:5.1f}+/-{scr_std:4.1f}%    "
              f"{i95_mean:6.0f}+/-{i95_std:5.0f}    "
              f"{i100_mean:6.0f}+/-{i100_std:5.0f}    "
              f"{wss_mean:.3f}+/-{wss_std:.3f}")


def main():
    """Main execution"""
    print("=" * 70)
    print("EXTRACT RANDOM HYBRID METRICS FROM COMPLETE_REPORT.txt")
    print("=" * 70)

    # ========================================
    # CONFIGURATION - Update these paths
    # ========================================
    FILE_PATHS_CSV = 'hybrid_random_file_paths.csv'
    EXISTING_CSV = 'random_combined_metrics.csv'
    OUTPUT_HYBRID_ONLY = 'hybrid_random_metrics_extracted.csv'
    OUTPUT_MERGED = 'random_combined_metrics_with_hybrid.csv'

    # ========================================
    # STEP 1: Check file paths CSV
    # ========================================
    if not os.path.exists(FILE_PATHS_CSV):
        print(f"\nError: Cannot find file paths CSV: {FILE_PATHS_CSV}")
        print(f"Please ensure the file exists in the current directory.")
        return None

    # ========================================
    # STEP 2: Extract metrics
    # ========================================
    hybrid_df = extract_random_hybrid_metrics(FILE_PATHS_CSV)

    if hybrid_df is None or len(hybrid_df) == 0:
        print(f"\nFailed to extract any metrics.")
        return None

    # ========================================
    # STEP 3: Save hybrid-only results
    # ========================================
    print(f"\n{'='*70}")
    print(f"SAVING HYBRID-ONLY METRICS")
    print(f"{'='*70}")

    hybrid_df.to_csv(OUTPUT_HYBRID_ONLY, index=False)
    print(f"   Saved to: {OUTPUT_HYBRID_ONLY}")

    # ========================================
    # STEP 4: Merge with existing random metrics
    # ========================================
    merged_df = merge_with_existing(EXISTING_CSV, hybrid_df, OUTPUT_MERGED)

    # ========================================
    # STEP 5: Validate
    # ========================================
    validate_results(merged_df)

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print(f"\n{'='*70}")
    print(f"DONE!")
    print(f"{'='*70}")

    if os.path.exists(EXISTING_CSV):
        existing_count = len(pd.read_csv(EXISTING_CSV))
        print(f"\nExisting random metrics: {existing_count} rows")
        print(f"   Methods: certainty, uncertainty")

    print(f"\nNew hybrid metrics extracted: {len(hybrid_df)} rows")
    print(f"   Methods: {sorted(hybrid_df['method'].unique())}")
    print(f"   Saved to: {OUTPUT_HYBRID_ONLY}")

    print(f"\nMerged output: {len(merged_df)} rows")
    print(f"   Methods: {sorted(merged_df['method'].unique())}")
    print(f"   Saved to: {OUTPUT_MERGED}")

    print(f"\n{'='*70}")

    return merged_df


if __name__ == "__main__":
    merged_df = main()
