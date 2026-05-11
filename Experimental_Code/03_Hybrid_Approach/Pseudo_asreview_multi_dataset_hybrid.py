"""
Pseudo-ASReview with Hybrid Query Methods - Multiple Datasets

This script runs Pseudo-ASReview experiments with:
- Multiple datasets configured via CSV
- Multiple pseudo-labeling percentages
- Support for hybrid query methods:
  1. STANDARD: Pure certainty sampling
  2. PHASE_SWITCHING: Switch methods at mid-point
  3. PERCENTAGE_ALTERNATING: Alternate every 1%
- Phase 1: LLM-based pseudo-labeling
- Phase 2: Active learning with stopping criteria
- Phase 3: Continue to 100% recall
"""

import pandas as pd
from pseudo_asreview_complete_hybrid import (
    PseudoASReviewHybrid, 
    HybridMode, 
    PseudoLabelConfig
)
import os
import logging
import sys

# Ensure Windows console can print Unicode (avoid 'charmap' codec errors)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# CONFIGURATION
# ========================================
CONFIG_CSV = 'pseudo_dataset_config.csv'

# Experiment configuration
PSEUDO_LABEL_PERCENTAGES = [27.5, 30.0, 32.5, 35.0]

# Hybrid methods to test (excluding standard since you already have it)
HYBRID_MODES = [
    (HybridMode.PHASE_SWITCHING, "certainty", "phase_switch_cert_to_uncert"),
    (HybridMode.PHASE_SWITCHING, "uncertainty", "phase_switch_uncert_to_cert"),
    (HybridMode.PERCENTAGE_ALTERNATING, "certainty", "alternating_cert_start"),
    (HybridMode.PERCENTAGE_ALTERNATING, "uncertainty", "alternating_uncert_start"),
]

# Fixed parameters
RANDOM_STATE = 1
MIN_SCREENED_PCT = 0.10
CONSECUTIVE_IRRELEVANT = 50


def load_config(config_path):
    """Load dataset configuration from CSV file."""
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return []
    
    try:
        config_df = pd.read_csv(config_path)
        required_cols = ['data_path', 'scores_path', 'name', 'category', 'enabled']
        missing_cols = [col for col in required_cols if col not in config_df.columns]
        
        if missing_cols:
            logger.error(f"Config file missing required columns: {missing_cols}")
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
        data_exists = os.path.exists(dataset['data_path'])
        scores_exists = os.path.exists(dataset['scores_path'])
        
        if data_exists and scores_exists:
            valid.append(dataset)
        else:
            if not data_exists:
                missing.append(f"Data: {dataset['data_path']}")
            if not scores_exists:
                missing.append(f"Scores: {dataset['scores_path']}")
            logger.warning(f"Missing files for {dataset['name']}")
    
    return valid, missing


def run_single_experiment(dataset_config, top_pct, bottom_pct, 
                         hybrid_mode, initial_query_mode, mode_name):
    """
    Run a single pseudo-labeling experiment with hybrid query method.
    
    Args:
        dataset_config: Dataset configuration dict
        top_pct: Top percentage for pseudo-positive labels
        bottom_pct: Bottom percentage for pseudo-negative labels
        hybrid_mode: HybridMode enum
        initial_query_mode: "certainty" or "uncertainty"
        mode_name: Name for output directory
    
    Returns:
        Dictionary of results
    """
    # Load data
    df = pd.read_csv(dataset_config['data_path'])
    
    texts = df['text'].tolist()
    labels = df['Status'].tolist()
    record_ids = df['PMID'].tolist()
    
    # Configure
    config = PseudoLabelConfig(
        top_percentage=top_pct,
        bottom_percentage=bottom_pct,
        llm_score_column='score',
        min_screened_percentage=MIN_SCREENED_PCT,
        consecutive_irrelevant=CONSECUTIVE_IRRELEVANT,
        random_state=RANDOM_STATE,
        verbose=False
    )
    
    # Create output directory
    output_dir = (f"hybrid_results/{dataset_config['category']}/{dataset_config['name']}/"
                  f"{mode_name}/top{top_pct}_bottom{bottom_pct}")
    
    # Initialize and run with hybrid mode
    asreview = PseudoASReviewHybrid(
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
    
    results = asreview.run_llm_pseudo_labeling_phases_1_and_2(
        scores_csv_path=dataset_config['scores_path']
    )
    
    # Add metadata
    results['dataset_name'] = dataset_config['name']
    results['dataset_category'] = dataset_config['category']
    results['top_pct'] = top_pct
    results['bottom_pct'] = bottom_pct
    results['total_records'] = len(texts)
    results['total_relevant'] = sum(labels)
    results['prevalence'] = sum(labels) / len(labels)
    results['hybrid_mode_name'] = mode_name
    results['initial_query_mode'] = initial_query_mode
    
    return results


def print_experiment_summary(result, dataset_name, mode_name):
    """Print results for a single experiment."""
    if not result:
        return
    
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {dataset_name} | {mode_name}")
    print(f"Top/Bottom: {result['top_pct']*100:.1f}%")
    print("=" * 70)
    
    print(f"\nPhase 1: Pseudo-labeled {len(result['pseudo_labels'])} documents")
    print(f"Phase 2: {result['phase2_ended_at_iteration']} iterations, "
          f"{result['proportion_screened']*100:.1f}% screened, "
          f"{result['total_recall']*100:.1f}% recall")
    
    if 'query_mode_switches' in result and len(result['query_mode_switches']) > 0:
        print(f"Switches: {len(result['query_mode_switches'])}")
    
    print(f"Phase 3: 95% at iter {result['iteration_at_95_recall']}, "
          f"100% at iter {result['iteration_at_100_recall']}")
    print(f"Work saved: {(1-result['proportion_screened_at_100'])*100:.1f}%")
    print("=" * 70)


def main():
    """Main execution function."""
    logger.info("Starting Pseudo-ASReview Hybrid Multi-Dataset Experiments")
    
    # Load configuration
    datasets = load_config(CONFIG_CSV)
    
    if not datasets:
        logger.error("No datasets to process. Check your configuration file.")
        return
    
    # Verify files exist
    valid_datasets, missing = verify_dataset_files(datasets)
    
    if missing:
        logger.warning(f"{len(missing)} files not found:")
        for m in missing[:5]:  # Show first 5
            logger.warning(f"  - {m}")
        auto_yes = os.environ.get('PSEUDO_HYBRID_YES', '').lower() in ('1', 'true', 'yes')
        if not valid_datasets:
            logger.error("No valid datasets; fix paths or enable rows in pseudo_dataset_config.csv.")
            return
        if not auto_yes:
            response = input(f"\nContinue with {len(valid_datasets)} valid datasets? (y/n): ")
            if response.lower() != 'y':
                logger.info("Processing cancelled")
                return
        else:
            logger.info("Continuing with %s valid datasets (PSEUDO_HYBRID_YES set).", len(valid_datasets))
    
    datasets = valid_datasets
    
    # Calculate total experiments
    total_experiments = len(datasets) * len(PSEUDO_LABEL_PERCENTAGES) * len(HYBRID_MODES)
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"EXPERIMENT CONFIGURATION")
    logger.info(f"{'=' * 70}")
    logger.info(f"Datasets: {len(datasets)}")
    logger.info(f"Pseudo-labeling percentages: {len(PSEUDO_LABEL_PERCENTAGES)}")
    logger.info(f"Hybrid modes: {len(HYBRID_MODES)}")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"{'=' * 70}\n")
    
    all_results = []
    experiment_count = 0
    
    # Process each dataset
    for dataset_idx, dataset_config in enumerate(datasets, 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"DATASET {dataset_idx}/{len(datasets)}: {dataset_config['name']}")
        logger.info(f"{'=' * 70}")
        
        # Load basic info
        df = pd.read_csv(dataset_config['data_path'])
        n_records = len(df)
        n_relevant = df['Status'].sum()
        prevalence = n_relevant / n_records * 100
        
        logger.info(f"Records: {n_records}, Relevant: {n_relevant} ({prevalence:.1f}%)")
        
        # Test each hybrid mode
        for mode_idx, (hybrid_mode, initial_query, mode_name) in enumerate(HYBRID_MODES, 1):
            logger.info(f"\n  Mode {mode_idx}/{len(HYBRID_MODES)}: {mode_name}")
            
            # Test each pseudo-labeling percentage
            for pct_idx, pct in enumerate(PSEUDO_LABEL_PERCENTAGES, 1):
                experiment_count += 1
                logger.info(f"    Experiment {pct_idx}/{len(PSEUDO_LABEL_PERCENTAGES)}: "
                           f"Top {pct}% / Bottom {pct}% "
                           f"(Overall: {experiment_count}/{total_experiments})")
                
                try:
                    pct_decimal = pct / 100
                    result = run_single_experiment(
                        dataset_config, pct_decimal, pct_decimal,
                        hybrid_mode, initial_query, mode_name
                    )
                    all_results.append(result)
                    
                    # Print brief results
                    print(f"      ✓ Recall: {result['total_recall']*100:.1f}% | "
                          f"95% @ iter {result['iteration_at_95_recall']} | "
                          f"100% @ iter {result['iteration_at_100_recall']}")
                
                except Exception as e:
                    logger.error(f"      ✗ Experiment failed: {str(e)}")
                    continue
    
    # ========================================
    # SAVE COMPREHENSIVE RESULTS
    # ========================================
    print("\n\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    if all_results:
        # Create detailed results DataFrame
        summary_data = []
        for r in all_results:
            # Handle None values safely
            iter_95 = r.get('iteration_at_95_recall')
            iter_100 = r.get('iteration_at_100_recall')
            if iter_95 is None:
                iter_95 = iter_100  # If 95% wasn't tracked, use 100%

            row = {
                'dataset': r['dataset_name'],
                'category': r['dataset_category'],
                'hybrid_mode': r['hybrid_mode_name'],
                'initial_query': r['initial_query_mode'],
                'top_pct': r['top_pct'] * 100,
                'bottom_pct': r['bottom_pct'] * 100,
                'total_records': r['total_records'],
                'total_relevant': r['total_relevant'],
                'prevalence': r['prevalence'] * 100,
                'pseudo_total': len(r['pseudo_labels']),
                'phase2_iterations': r['phase2_ended_at_iteration'],
                'screened_pct': r['proportion_screened'] * 100,
                'screening_recall': r['screening_recall'] * 100,
                'total_recall': r['total_recall'] * 100,
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

        # Save main summary (merge with existing so additional ratio runs keep prior rows)
        os.makedirs('hybrid_results', exist_ok=True)
        summary_path = 'hybrid_results/hybrid_experiments_summary.csv'
        key_cols = ['dataset', 'hybrid_mode', 'initial_query', 'top_pct', 'bottom_pct']
        if os.path.isfile(summary_path):
            existing = pd.read_csv(summary_path)
            merged = pd.concat([existing, summary_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=key_cols, keep='last')
            merged.to_csv(summary_path, index=False)
        else:
            summary_df.to_csv(summary_path, index=False)
        logger.info(f"✓ Main summary saved: {summary_path}")

        # Save per-dataset summaries
        for dataset_name in summary_df['dataset'].unique():
            dataset_df = summary_df[summary_df['dataset'] == dataset_name]
            dataset_path = f'hybrid_results/{dataset_name}_hybrid_experiments.csv'
            if os.path.isfile(dataset_path):
                existing_ds = pd.read_csv(dataset_path)
                merged_ds = pd.concat([existing_ds, dataset_df], ignore_index=True)
                merged_ds = merged_ds.drop_duplicates(subset=key_cols, keep='last')
                merged_ds.to_csv(dataset_path, index=False)
            else:
                dataset_df.to_csv(dataset_path, index=False)
            logger.info(f"✓ Dataset summary saved: {dataset_path}")

        # Save per-mode summaries
        for mode_name in summary_df['hybrid_mode'].unique():
            mode_df = summary_df[summary_df['hybrid_mode'] == mode_name]
            mode_path = f'hybrid_results/{mode_name}_experiments.csv'
            if os.path.isfile(mode_path):
                existing_mode = pd.read_csv(mode_path)
                merged_mode = pd.concat([existing_mode, mode_df], ignore_index=True)
                merged_mode = merged_mode.drop_duplicates(subset=key_cols, keep='last')
                merged_mode.to_csv(mode_path, index=False)
            else:
                mode_df.to_csv(mode_path, index=False)
            logger.info(f"✓ Mode summary saved: {mode_path}")

        # Generate comparison summary
        print("\n" + "=" * 70)
        print("HYBRID METHOD COMPARISON (Averaged across datasets & percentages)")
        print("=" * 70)

        comparison = summary_df.groupby('hybrid_mode').agg({
            'iter_at_95': 'mean',
            'iter_at_100': 'mean',
            'work_saved': 'mean',
            'n_switches': 'mean'
        }).round(1)

        print(comparison.to_string())
        print("=" * 70)

        comparison.to_csv('hybrid_results/hybrid_method_comparison.csv')
        logger.info("✓ Comparison summary saved: hybrid_results/hybrid_method_comparison.csv")

    print("=" * 70)
    logger.info("\n🎉 All hybrid experiments complete!")


if __name__ == "__main__":
    main()