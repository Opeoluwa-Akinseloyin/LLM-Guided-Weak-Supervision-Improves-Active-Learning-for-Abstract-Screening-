"""
Compare WSS values extracted directly from COMPLETE_REPORT.txt files
against the wss_at_phase2 column in rq4_raw_metrics_with_all_6_methods.csv
"""

import pandas as pd
import re
import os

BASE_DIR = r"c:\Users\akinseloyo\Documents\Simulation"
RQ4_DIR = os.path.join(BASE_DIR, "RQ4")
CSV_PATH = os.path.join(RQ4_DIR, "rq4_extracted_metrics", "rq4_raw_metrics_with_all_6_methods.csv")
OUTPUT_PATH = os.path.join(RQ4_DIR, "rq4_extracted_metrics", "wss_comparison.csv")

METHOD_MAPPING = {
    'phase_switch_cert_to_uncert': 'phase_switch_c2u',
    'phase_switch_uncert_to_cert': 'phase_switch_u2c',
    'alternating_cert_start':      'alternating_c_start',
    'alternating_uncert_start':    'alternating_u_start',
}


def extract_wss_from_report(file_path):
    if not os.path.exists(file_path):
        return None, "FILE_NOT_FOUND"
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        match = re.search(r'At Phase 2:.*?WSS=([\d.]+)', content, re.IGNORECASE)
        if match:
            return float(match.group(1)), "OK"
        return None, "PATTERN_NOT_FOUND"
    except Exception as e:
        return None, f"ERROR: {e}"


def resolve_path(raw_path, base):
    """Resolve a potentially relative path against a base directory."""
    raw_path = raw_path.replace('/', os.sep).replace('\\', os.sep)
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.normpath(os.path.join(base, raw_path))


def main():
    print("=" * 70)
    print("WSS EXTRACTION AND COMPARISON")
    print("=" * 70)

    # ── Load the CSV we want to verify ───────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded CSV: {len(df)} rows, methods: {sorted(df['method'].unique())}")

    records = []

    # ── 1. Certainty & Uncertainty via report_locations_updated.csv ──────
    report_locs_path = os.path.join(RQ4_DIR, "report_locations_updated.csv")
    report_locs = pd.read_csv(report_locs_path)
    print(f"\nProcessing report_locations_updated.csv ({len(report_locs)} rows)...")

    for _, row in report_locs.iterrows():
        abs_path = resolve_path(row['report_path'], RQ4_DIR)
        wss, status = extract_wss_from_report(abs_path)
        records.append({
            'dataset':          row['dataset'],
            'method':           row['method'],
            'ratio':            float(row['ratio']),
            'wss_from_report':  wss,
            'status':           status,
            'source_file':      abs_path,
        })

    # ── 2. Hybrid methods via hybrid_file_paths_simple.csv ───────────────
    hybrid_paths_path = os.path.join(BASE_DIR, "hybrid_file_paths_simple.csv")
    hybrid_paths = pd.read_csv(hybrid_paths_path)
    print(f"Processing hybrid_file_paths_simple.csv ({len(hybrid_paths)} rows)...")

    for _, row in hybrid_paths.iterrows():
        abs_path = resolve_path(row['complete_report_path'], BASE_DIR)
        wss, status = extract_wss_from_report(abs_path)
        records.append({
            'dataset':          row['name'],
            'method':           METHOD_MAPPING.get(row['mode'], row['mode']),
            'ratio':            float(row['pseudo_pct']),
            'wss_from_report':  wss,
            'status':           status,
            'source_file':      abs_path,
        })

    extracted_df = pd.DataFrame(records)

    # ── 3. Merge with original CSV ────────────────────────────────────────
    comparison = pd.merge(
        df[['dataset', 'method', 'ratio', 'wss_at_phase2']],
        extracted_df[['dataset', 'method', 'ratio', 'wss_from_report', 'status', 'source_file']],
        on=['dataset', 'method', 'ratio'],
        how='outer',
        indicator=True,
    )

    # ── 4. Flag matches / mismatches ──────────────────────────────────────
    TOLERANCE = 0.001

    def classify(row):
        csv_val = row['wss_at_phase2']
        rep_val = row['wss_from_report']
        if pd.isna(csv_val) and pd.isna(rep_val):
            return 'BOTH_MISSING'
        if pd.isna(csv_val):
            return 'MISSING_IN_CSV'
        if pd.isna(rep_val):
            return 'MISSING_IN_REPORT'
        diff = abs(float(csv_val) - float(rep_val))
        return 'MATCH' if diff <= TOLERANCE else f'MISMATCH(diff={diff:.6f})'

    comparison['result'] = comparison.apply(classify, axis=1)

    # ── 5. Save output ────────────────────────────────────────────────────
    col_order = ['dataset', 'method', 'ratio', 'wss_at_phase2',
                 'wss_from_report', 'result', 'status', 'source_file', '_merge']
    comparison = comparison[col_order]
    comparison.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved comparison to: {OUTPUT_PATH}")

    # ── 6. Summary ────────────────────────────────────────────────────────
    total          = len(comparison)
    matches        = (comparison['result'] == 'MATCH').sum()
    mismatches     = comparison['result'].str.startswith('MISMATCH').sum()
    missing_report = (comparison['result'] == 'MISSING_IN_REPORT').sum()
    missing_csv    = (comparison['result'] == 'MISSING_IN_CSV').sum()
    not_found      = (comparison['status'] == 'FILE_NOT_FOUND').sum()

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Total rows compared : {total}")
    print(f"  Matches             : {matches}")
    print(f"  Mismatches          : {mismatches}")
    print(f"  Missing in report   : {missing_report}")
    print(f"  Missing in CSV      : {missing_csv}")
    print(f"  Files not found     : {not_found}")

    if mismatches > 0:
        print(f"\n{'='*70}")
        print(f"MISMATCHED ROWS ({mismatches} total):")
        print(f"{'='*70}")
        mismatch_df = comparison[comparison['result'].str.startswith('MISMATCH')]
        print(mismatch_df[['dataset', 'method', 'ratio',
                            'wss_at_phase2', 'wss_from_report', 'result']].to_string(index=False))

    if not_found > 0:
        print(f"\n{'='*70}")
        print(f"FILES NOT FOUND ({not_found} total):")
        print(f"{'='*70}")
        nf = comparison[comparison['status'] == 'FILE_NOT_FOUND']
        for _, r in nf.iterrows():
            print(f"  {r['dataset']} | {r['method']} | ratio={r['ratio']} | {r['source_file']}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")

    return comparison


if __name__ == "__main__":
    main()
