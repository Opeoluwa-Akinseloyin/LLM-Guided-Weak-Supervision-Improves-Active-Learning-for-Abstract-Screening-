#!/usr/bin/env python3
"""
Generate Random Baseline Report Locations CSV
Customized for your specific file structure
"""

import pandas as pd
from pathlib import Path

# ============================================================================
# YOUR DATA - EDIT THESE
# ============================================================================

# List all 28 datasets
DATASETS = [
    # DTA datasets (8 total)
    'CD008874', 'CD009044', 'CD011686', 'CD012080',
    'CD012233', 'CD012567', 'CD012669', 'CD012768',
    
    # INT datasets (20 total) - ADD YOUR DATASET NAMES HERE
    "CD000996", "CD001261", "CD004414", "CD006468", "CD007867",
    "CD009069", "CD009642", "CD010038", "CD010239", "CD010558",
    "CD010753", "CD011140", "CD011571", "CD011768", "CD011977",
    "CD012069", "CD012164", "CD012342", "CD012455", "CD012551",
]

# Map each dataset to its type (DTA or INT)
DATASET_TYPES = {
    # DTA
    'CD008874': 'DTA',
    'CD009044': 'DTA',
    'CD011686': 'DTA',
    'CD012080': 'DTA',
    'CD012233': 'DTA',
    'CD012567': 'DTA',
    'CD012669': 'DTA',
    'CD012768': 'DTA',
    
    # INT - ADD YOUR MAPPINGS HERE
    "CD000996": "INT",
    "CD001261": "INT",
    "CD004414": "INT",
    "CD006468": "INT",
    "CD007867": "INT",
    "CD009069": "INT",
    "CD009642": "INT",
    "CD010038": "INT",
    "CD010239": "INT",
    "CD010558": "INT",
    "CD010753": "INT",
    "CD011140": "INT",
    "CD011571": "INT",
    "CD011768": "INT",
    "CD011977": "INT",
    "CD012069": "INT",
    "CD012164": "INT",
    "CD012342": "INT",
    "CD012455": "INT",
    "CD012551": "INT",
}

# Base paths (adjust if needed)
CERTAINTY_BASE = "../results/certainty"
UNCERTAINTY_BASE = "../ASReview uncertainty/results/uncertainty"

# Number of runs
NUM_RUNS = 10

# ============================================================================
# SCRIPT - NO NEED TO EDIT BELOW
# ============================================================================

def generate_path(dataset: str, method: str, run: int) -> str:
    """Generate file path for given dataset, method, and run"""
    
    dtype = DATASET_TYPES[dataset]
    
    if method == 'certainty':
        # Pattern: results/certainty/DTA/CD008874/CD008874_complete_run_1/REPORT.txt
        path = f"{CERTAINTY_BASE}/{dtype}/{dataset}/{dataset}_complete_run_{run}/COMPLETE_REPORT.txt"
    else:  # uncertainty
        # Pattern: ASReview uncertainty/results/uncertainty/DTA/CD008874/CD008874_uncertainty_run_1/REPORT.txt
        path = f"{UNCERTAINTY_BASE}/{dtype}/{dataset}/{dataset}_uncertainty_run_{run}/COMPLETE_REPORT.txt"
    
    return path


def generate_csv():
    """Generate the complete CSV"""
    
    print("="*80)
    print("GENERATING RANDOM BASELINE REPORT LOCATIONS CSV")
    print("="*80)
    print()
    
    rows = []
    
    for dataset in DATASETS:
        for method in ['certainty', 'uncertainty']:
            for run in range(1, NUM_RUNS + 1):
                path = generate_path(dataset, method, run)
                
                rows.append({
                    'dataset': dataset,
                    'method': method,
                    'run': run,
                    'report_path': path
                })
    
    df = pd.DataFrame(rows)
    
    # Save CSV
    output_file = 'RQ2/random_baseline_report_locations.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ CSV created: {output_file}")
    print(f"  Total rows: {len(df)}")
    print(f"  Datasets: {df['dataset'].nunique()}")
    print(f"  Methods: {df['method'].unique().tolist()}")
    print(f"  Runs per dataset: {df.groupby(['dataset', 'method']).size().min()}-{df.groupby(['dataset', 'method']).size().max()}")
    print()
    
    # Show sample
    print("Sample rows:")
    print(df.head(10).to_string(index=False))
    print("...")
    print(df.tail(5).to_string(index=False))
    print()
    
    # Check file existence
    print("Checking if files exist (this may take a moment)...")
    existing = 0
    missing = 0
    missing_files = []
    
    for path in df['report_path']:
        if Path(path).exists():
            existing += 1
        else:
            missing += 1
            if missing <= 10:  # Store first 10 missing
                missing_files.append(path)
    
    print(f"\n✓ Files found: {existing}/{len(df)}")
    
    if missing > 0:
        print(f"❌ Missing files: {missing}/{len(df)}")
        print("\nFirst few missing files:")
        for path in missing_files[:5]:
            print(f"  - {path}")
        print("\nPossible issues:")
        print("  1. Base paths incorrect (check CERTAINTY_BASE, UNCERTAINTY_BASE)")
        print("  2. Files not yet generated")
        print("  3. Different naming pattern")
        print("\nYou can still proceed - the analysis will skip missing files")
    else:
        print("✅ All files found!")
    
    print()
    print(f"✓ Ready to use: {output_file}")
    print()
    print("Next steps:")
    print("  1. Review the CSV (check paths are correct)")
    print("  2. Run: python RQ1_COMPLETE_LAUNCHER.py")
    
    return df


if __name__ == "__main__":
    df = generate_csv()
