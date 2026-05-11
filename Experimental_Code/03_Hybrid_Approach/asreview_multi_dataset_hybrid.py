"""
ASReview with Hybrid Query Methods - Multiple Datasets (Random Initialization)

This script runs ASReview experiments with RANDOM initialization (no pseudo-labeling):
- Multiple datasets configured via CSV
- Support for hybrid query methods ONLY (standard certainty/uncertainty already completed):
  1. PHASE_SWITCHING: Switch methods at mid-point
  2. PERCENTAGE_ALTERNATING: Alternate every 1%
- Phase 1: Random prior knowledge selection
- Phase 2: Active learning with stopping criteria
- Phase 3: Continue to 100% recall

Output: hybrid_results_random/ (separate from pseudo-labeling results)

This is the RANDOM INITIALIZATION counterpart of Pseudo_asreview_multi_dataset_hybrid.py.
"""

import pandas as pd
from asreview_complete_hybrid import (
    ASReviewCompleteHybrid,
    HybridMode,
)
from asreview_safe_simplified import SAFEConfig
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# CONFIGURATION
# ========================================
CONFIG_CSV = 'dataset_config.csv'  # Uses same dataset config (no scores_path needed)

# Hybrid methods to test (standard certainty/uncertainty already completed)
HYBRID_MODES = [
    (HybridMode.PHASE_SWITCHING, "certainty", "phase_switch_cert_to_uncert"),
    (HybridMode.PHASE_SWITCHING, "uncertainty", "phase_switch_uncert_to_cert"),
    (HybridMode.PERCENTAGE_ALTERNATING, "certainty", "alternating_cert_start"),
    (HybridMode.PERCENTAGE_ALTERNATING, "uncertainty", "alternating_uncert_start"),
]

# Fixed parameters
N_RUNS = 10
MIN_SCREENED_PCT = 0.10
CONSECUTIVE_IRRELEVANT = 50

# Output directory (separate from pseudo results)
OUTPUT_BASE = 'hybrid_results_random'


def load_config(config_path):
    """Load dataset configuration from CSV file."""
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return []

    try:
        config_df = pd.read_csv(config_path)

        # Check for required columns - only need path, name, category, enabled
        # (no scores_path needed for random initialization)
        required_cols = ['path', 'name', 'category', 'enabled']

        # Also support the pseudo config format with 'data_path' instead of 'path'
        if 'data_path' in config_df.columns and 'path' not in config_df.columns:
            config_df['path'] = config_df['data_path']

        missing_cols = [col for col in required_cols if col not in config_df.columns]

        if missing_cols:
            logger.error(f"Config file missing required columns: {missing_cols}")
            logger.info(f"Available columns: {config_df.columns.tolist()}")
            logger.info("Required columns: path (or data_path), name, category, enabled")
            return []

        enabled = config_df[config_df['enabled'] == 1]
        logger.info(f"Loaded {len(enabled)}/{len(config_df)} enabled datasets from config")

        return enabled.to_dict('records')

    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        return []


def verify_dataset_files(datasets):
    """Check which dataset files actually exist."""
    valid = []
    missing = []

    for dataset in datasets:
        if os.path.exists(dataset['path']):
            valid.append(dataset)
        else:
            missing.append(dataset['path'])
            logger.warning(f"Dataset file not found: {dataset['path']}")

    return valid, missing


def run_single_experiment(dataset_config, hybrid_mode, initial_query_mode,
                          mode_name, run_number):
    """
    Run a single hybrid experiment with random initialization.

    Args:
        dataset_config: Dataset configuration dict
        hybrid_mode: HybridMode enum
        initial_query_mode: "certainty" or "uncertainty"
        mode_name: Name for output directory
        run_number: Run number (for random seed)

    Returns:
        Dictionary of results
    """
    # Load data
    df = pd.read_csv(dataset_config['path'])

    texts = df['text'].tolist()
    labels = df['Status'].tolist()
    record_ids = df['PMID'].tolist()

    # Configure with SAFEConfig (random initialization, no pseudo-labeling)
    config = SAFEConfig(
        prior_knowledge_percentage=0.01,
        min_prior_relevant=1,
        min_screened_percentage=MIN_SCREENED_PCT,
        consecutive_irrelevant=CONSECUTIVE_IRRELEVANT,
        random_state=run_number,
        verbose=False
    )

    # Create output directory
    output_dir = (f"{OUTPUT_BASE}/{dataset_config['category']}/{dataset_config['name']}/"
                  f"{mode_name}/run_{run_number}")

    # Initialize and run with hybrid mode
    asreview = ASReviewCompleteHybrid(
        config=config,
        output_dir=output_dir,
        hybrid_mode=hybrid_mode,
        initial_query_mode=initial_query_mode
    )

    asreview.load_data(
        texts=texts,
        labels=labels,
        record_ids=record_ids
    )

    # Run - no scores_csv_path needed (random initialization)
    results = asreview.run_safe_phases_1_and_2()

    # Add metadata
    results['dataset_name'] = dataset_config['name']
    results['dataset_category'] = dataset_config['category']
    results['run_number'] = run_number
    results['total_records'] = len(texts)
    results['total_relevant'] = sum(labels)
    results['prevalence'] = sum(labels) / len(labels)
    results['hybrid_mode_name'] = mode_name
    results['initial_query_mode'] = initial_query_mode

    return results


def print_experiment_summary(result, dataset_name, mode_name, run_number):
    """Print results for a single experiment."""
    if not result:
        return

    recall_val = result.get('screening_recall', result.get('final_recall', 0))

    print(f"      ✓ Run {run_number}: "
          f"Recall: {recall_val*100:.1f}% | "
          f"95% @ iter {result['iteration_at_95_recall']} | "
          f"100% @ iter {result['iteration_at_100_recall']}")


def main():
    """Main execution function."""
    logger.info("Starting ASReview Hybrid Multi-Dataset Experiments (Random Initialization)")
    logger.info(f"Output directory: {OUTPUT_BASE}/")

    # Load configuration
    datasets = load_config(CONFIG_CSV)

    if not datasets:
        logger.error("No datasets to process. Check your configuration file.")
        return

    # Verify files exist
    valid_datasets, missing = verify_dataset_files(datasets)

    if missing:
        logger.warning(f"{len(missing)} files not found:")
        for m in missing[:5]:
            logger.warning(f"  - {m}")
        response = input(f"\nContinue with {len(valid_datasets)} valid datasets? (y/n): ")
        if response.lower() != 'y':
            logger.info("Processing cancelled")
            return

    datasets = valid_datasets

    # Calculate total experiments
    total_experiments = len(datasets) * len(HYBRID_MODES) * N_RUNS

    logger.info(f"\n{'=' * 70}")
    logger.info(f"EXPERIMENT CONFIGURATION (Random Initialization)")
    logger.info(f"{'=' * 70}")
    logger.info(f"Datasets: {len(datasets)}")
    logger.info(f"Hybrid modes: {len(HYBRID_MODES)}")
    logger.info(f"Runs per experiment: {N_RUNS}")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Output: {OUTPUT_BASE}/")
    logger.info(f"{'=' * 70}\n")

    all_results = []
    experiment_count = 0

    # Process each dataset
    for dataset_idx, dataset_config in enumerate(datasets, 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"DATASET {dataset_idx}/{len(datasets)}: {dataset_config['name']}")
        logger.info(f"{'=' * 70}")

        # Load basic info
        df = pd.read_csv(dataset_config['path'])
        n_records = len(df)
        n_relevant = df['Status'].sum()
        prevalence = n_relevant / n_records * 100

        logger.info(f"Records: {n_records}, Relevant: {n_relevant} ({prevalence:.1f}%)")

        # Test each hybrid mode
        for mode_idx, (hybrid_mode, initial_query, mode_name) in enumerate(HYBRID_MODES, 1):
            logger.info(f"\n  Mode {mode_idx}/{len(HYBRID_MODES)}: {mode_name}")

            # Run N times with different random seeds
            for run_num in range(1, N_RUNS + 1):
                experiment_count += 1
                logger.info(f"    Run {run_num}/{N_RUNS} "
                           f"(Overall: {experiment_count}/{total_experiments})")

                try:
                    result = run_single_experiment(
                        dataset_config, hybrid_mode, initial_query,
                        mode_name, run_num
                    )
                    all_results.append(result)

                    # Print brief results
                    print_experiment_summary(result, dataset_config['name'],
                                          mode_name, run_num)

                except Exception as e:
                    logger.error(f"      ✗ Run {run_num} failed: {str(e)}")
                    continue

    # ========================================
    # SAVE COMPREHENSIVE RESULTS
    # ========================================
    print("\n\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    if all_results:
        summary_data = []
        for r in all_results:
            iter_95 = r.get('iteration_at_95_recall')
            iter_100 = r.get('iteration_at_100_recall')
            if iter_95 is None:
                iter_95 = iter_100

            row = {
                'dataset': r['dataset_name'],
                'category': r['dataset_category'],
                'run': r['run_number'],
                'hybrid_mode': r['hybrid_mode_name'],
                'initial_query': r['initial_query_mode'],
                'total_records': r['total_records'],
                'total_relevant': r['total_relevant'],
                'prevalence': r['prevalence'] * 100,
                'phase2_iterations': r['phase2_ended_at_iteration'],
                'screened_pct': r['proportion_screened'] * 100,
                'screening_recall': r.get('screening_recall', r.get('final_recall', 0)) * 100,
                'total_recall': r.get('total_recall', r.get('final_recall', 0)) * 100,
                'iter_at_95': iter_95,
                'iter_at_100': iter_100,
                'iter_95_to_100': iter_100 - iter_95,
                'screened_at_100': r['proportion_screened_at_100'] * 100,
                'work_saved': (1 - r['proportion_screened_at_100']) * 100,
            }

            # Add hybrid-specific metrics
            if 'phase2a_ended_at' in r:
                row['phase2a_ended_at'] = r['phase2a_ended_at']
                row['phase2b_started_at'] = r['phase2b_started_at']

            if 'query_mode_switches' in r:
                row['n_switches'] = len(r['query_mode_switches'])
            else:
                row['n_switches'] = 0

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)

        # Save main summary
        os.makedirs(OUTPUT_BASE, exist_ok=True)
        summary_path = f'{OUTPUT_BASE}/hybrid_experiments_random_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"✓ Main summary saved: {summary_path}")

        # Save per-dataset summaries
        for dataset_name in summary_df['dataset'].unique():
            dataset_df = summary_df[summary_df['dataset'] == dataset_name]
            dataset_path = f'{OUTPUT_BASE}/{dataset_name}_hybrid_random_experiments.csv'
            dataset_df.to_csv(dataset_path, index=False)
            logger.info(f"✓ Dataset summary saved: {dataset_path}")

        # Save per-mode summaries
        for mode_name in summary_df['hybrid_mode'].unique():
            mode_df = summary_df[summary_df['hybrid_mode'] == mode_name]
            mode_path = f'{OUTPUT_BASE}/{mode_name}_random_experiments.csv'
            mode_df.to_csv(mode_path, index=False)
            logger.info(f"✓ Mode summary saved: {mode_path}")

        # Generate comparison summary
        print("\n" + "=" * 70)
        print("HYBRID METHOD COMPARISON - RANDOM INITIALIZATION")
        print("(Averaged across datasets & runs)")
        print("=" * 70)

        comparison = summary_df.groupby('hybrid_mode').agg({
            'iter_at_95': 'mean',
            'iter_at_100': 'mean',
            'work_saved': 'mean',
            'screening_recall': 'mean',
            'n_switches': 'mean'
        }).round(1)

        print(comparison.to_string())
        print("=" * 70)

        comparison.to_csv(f'{OUTPUT_BASE}/hybrid_method_comparison_random.csv')
        logger.info(f"✓ Comparison summary saved: {OUTPUT_BASE}/hybrid_method_comparison_random.csv")

        # Per-dataset per-mode summary (mean across runs)
        dataset_mode_summary = summary_df.groupby(['dataset', 'hybrid_mode']).agg({
            'iter_at_95': ['mean', 'std'],
            'iter_at_100': ['mean', 'std'],
            'work_saved': ['mean', 'std'],
            'screening_recall': ['mean', 'std'],
            'phase2_iterations': ['mean', 'std'],
        }).round(2)

        dataset_mode_path = f'{OUTPUT_BASE}/dataset_mode_summary_random.csv'
        dataset_mode_summary.to_csv(dataset_mode_path)
        logger.info(f"✓ Dataset-mode summary saved: {dataset_mode_path}")

    print("=" * 70)
    logger.info(f"\n🎉 All hybrid experiments (random initialization) complete!")
    logger.info(f"Results saved to: {OUTPUT_BASE}/")


if __name__ == "__main__":
    main()
