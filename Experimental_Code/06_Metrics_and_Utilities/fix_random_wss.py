"""
Fix random_combined_metrics_with_hybrid.csv

Two issues found in holistic check:

Issue 1 (~132 rows): wss_at_phase2 is NaN because the extraction regex
  `r'Actual WSS (Phase 2): ([\d.]+)'` cannot match negative values.
  Fix: use `r'Actual WSS (Phase 2): (-?[\d.]+)'`

Issue 2 (~18 rows): wss_at_95_theoretical is NaN for CD011977/CD012768
  uncertainty runs. The report shows e.g. "WSS@95%: -0.040" (negative)
  which the original regex missed. Re-compute from total_records and
  iteration_at_95_recall found in the report.
"""

import pandas as pd
import numpy as np
import os
import re

# ── path constants ────────────────────────────────────────────────────────────
BASE_DIR = r'c:\Users\akinseloyo\Documents\Simulation'
CSV_IN   = os.path.join(BASE_DIR, 'random_combined_metrics_with_hybrid.csv')
CSV_BACK = os.path.join(BASE_DIR, 'random_combined_metrics_with_hybrid_BACKUP.csv')
CSV_OUT  = os.path.join(BASE_DIR, 'random_combined_metrics_with_hybrid.csv')

# ── dataset → folder type (INT / DTA) ────────────────────────────────────────
DATASET_TYPES = {
    'CD008874': 'DTA', 'CD009044': 'DTA', 'CD011686': 'DTA', 'CD012080': 'DTA',
    'CD012233': 'DTA', 'CD012567': 'DTA', 'CD012669': 'DTA', 'CD012768': 'DTA',
    'CD000996': 'INT', 'CD001261': 'INT', 'CD004414': 'INT', 'CD006468': 'INT',
    'CD007867': 'INT', 'CD009069': 'INT', 'CD009642': 'INT', 'CD010038': 'INT',
    'CD010239': 'INT', 'CD010558': 'INT', 'CD010753': 'INT', 'CD011140': 'INT',
    'CD011571': 'INT', 'CD011768': 'INT', 'CD011977': 'INT', 'CD012069': 'INT',
    'CD012164': 'INT', 'CD012342': 'INT', 'CD012455': 'INT', 'CD012551': 'INT',
}

# ── hybrid method → folder name ───────────────────────────────────────────────
METHOD_TO_FOLDER = {
    'alternating_c_start': 'alternating_cert_start',
    'alternating_u_start': 'alternating_uncert_start',
    'phase_switch_c2u':    'phase_switch_cert_to_uncert',
    'phase_switch_u2c':    'phase_switch_uncert_to_cert',
}


def get_report_path(dataset: str, method: str, run: int) -> str:
    dtype = DATASET_TYPES[dataset]
    if method == 'certainty':
        return os.path.join(
            BASE_DIR, 'results', 'certainty',
            dtype, dataset,
            f'{dataset}_complete_run_{run}',
            'COMPLETE_REPORT.txt'
        )
    elif method == 'uncertainty':
        return os.path.join(
            BASE_DIR, 'ASReview uncertainty', 'results', 'uncertainty',
            dtype, dataset,
            f'{dataset}_uncertainty_run_{run}',
            'COMPLETE_REPORT.txt'
        )
    else:
        folder = METHOD_TO_FOLDER[method]
        return os.path.join(
            BASE_DIR, 'hybrid_results_random',
            dtype, dataset, folder,
            f'run_{run}',
            'COMPLETE_REPORT.txt'
        )


def read_report(path: str):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def extract_actual_wss(content: str):
    """Extract Actual WSS (Phase 2) — supports negative values."""
    m = re.search(r'Actual WSS \(Phase 2\): (-?[\d.]+)', content)
    return float(m.group(1)) if m else np.nan


def extract_wss95_theoretical(content: str):
    """Re-compute WSS@95% theoretical from total_records + iteration_at_95_recall."""
    m_n = re.search(r'Total records: (\d+)', content)
    m_i = re.search(r'Iteration at 95% recall: (\d+)', content)
    if not (m_n and m_i):
        return np.nan
    n    = int(m_n.group(1))
    i95  = int(m_i.group(1))
    return round(max(0.0, 0.95 - i95 / n), 3)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print('=' * 70)
    print('FIX random_combined_metrics_with_hybrid.csv')
    print('=' * 70)

    df = pd.read_csv(CSV_IN)
    print(f'\nLoaded {len(df)} rows.')
    print(f'  wss_at_phase2 NaN rows       : {df["wss_at_phase2"].isna().sum()}')
    print(f'  wss_at_95_theoretical NaN rows: {df["wss_at_95_theoretical"].isna().sum()}')

    # ── backup ────────────────────────────────────────────────────────────────
    df.to_csv(CSV_BACK, index=False)
    print(f'\nBackup saved -> {CSV_BACK}')

    # ── Issue 1: fix wss_at_phase2 NaN rows ──────────────────────────────────
    wss_nan_mask = df['wss_at_phase2'].isna()
    wss_nan_idx  = df.index[wss_nan_mask].tolist()
    print(f'\n-- Issue 1: wss_at_phase2 NaN ({len(wss_nan_idx)} rows) --')

    fixed_wss   = 0
    missing_wss = 0
    for idx in wss_nan_idx:
        row     = df.loc[idx]
        path    = get_report_path(row['dataset'], row['method'], int(row['run']))
        content = read_report(path)
        if content is None:
            missing_wss += 1
            continue
        val = extract_actual_wss(content)
        if not np.isnan(val):
            df.at[idx, 'wss_at_phase2'] = val
            fixed_wss += 1

    print(f'  Fixed  : {fixed_wss}')
    print(f'  Missing: {missing_wss}  (no COMPLETE_REPORT.txt found)')

    # ── Issue 2: fix wss_at_95_theoretical NaN rows ───────────────────────────
    wss95_nan_mask = df['wss_at_95_theoretical'].isna()
    wss95_nan_idx  = df.index[wss95_nan_mask].tolist()
    print(f'\n-- Issue 2: wss_at_95_theoretical NaN ({len(wss95_nan_idx)} rows) --')

    fixed_95   = 0
    missing_95 = 0
    for idx in wss95_nan_idx:
        row     = df.loc[idx]
        path    = get_report_path(row['dataset'], row['method'], int(row['run']))
        content = read_report(path)
        if content is None:
            missing_95 += 1
            continue
        val = extract_wss95_theoretical(content)
        if not np.isnan(val):
            df.at[idx, 'wss_at_95_theoretical'] = val
            fixed_95 += 1

    print(f'  Fixed  : {fixed_95}')
    print(f'  Missing: {missing_95}  (no COMPLETE_REPORT.txt found)')

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(CSV_OUT, index=False)
    print(f'\nCorrected CSV saved -> {CSV_OUT}')

    # ── Verify ────────────────────────────────────────────────────────────────
    print(f'\n-- Post-fix NaN counts --')
    print(f'  wss_at_phase2 NaN       : {df["wss_at_phase2"].isna().sum()}')
    print(f'  wss_at_95_theoretical NaN: {df["wss_at_95_theoretical"].isna().sum()}')

    print(f'\n-- Sample of corrected wss_at_phase2 values (originally NaN) --')
    originally_nan = df.loc[wss_nan_idx].dropna(subset=['wss_at_phase2'])
    cols = ['dataset', 'method', 'run', 'phase2_total_recall', 'wss_at_phase2']
    print(originally_nan[cols].head(20).to_string(index=False))

    print('\nDone.')


if __name__ == '__main__':
    main()
