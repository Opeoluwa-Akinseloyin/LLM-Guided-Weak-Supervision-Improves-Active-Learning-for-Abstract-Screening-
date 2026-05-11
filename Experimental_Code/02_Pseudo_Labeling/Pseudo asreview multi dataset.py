"""
Pseudo-ASReview with Complete Screening - Multiple Datasets & Experiments

This script runs Pseudo-ASReview with:
- Multiple datasets configured via CSV
- 10 experiments testing different pseudo-labeling percentages (2.5% to 25%)
- Phase 1: LLM-based pseudo-labeling
- Phase 2: Active learning with stopping criteria
- Phase 3: Continue to 100% recall
- Comprehensive summary statistics per dataset and experiment
"""

import pandas as pd
from pseudo_asreview_complete import PseudoASReviewComplete, PseudoLabelConfig
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
CONFIG_CSV = 'pseudo_dataset_config.csv'

# Experiment configuration: Test different pseudo-labeling percentages
# From 2.5% to 25% in steps of 2.5%
PSEUDO_LABEL_PERCENTAGES = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]

# Fixed parameters (deterministic method - only 1 run needed per percentage)
RANDOM_STATE = 1  # Fixed random state
MIN_SCREENED_PCT = 0.10
CONSECUTIVE_IRRELEVANT = 50


def create_example_config():
    """Create an example configuration CSV for pseudo-labeling datasets."""
    example_config = pd.DataFrame([
        {
            'data_path': 'INT_data/Criteria/CD011768_Criteria.csv',
            'scores_path': 'INT_data/Ranking/CD011768_ranked.csv',
            'name': 'CD011768',
            'category': 'Intervention',
            'enabled': 1
        },
        {
            'data_path': 'DTA_data/Criteria/CD012233_Criteria.csv',
            'scores_path': 'DTA_data/Ranking/CD012233_ranked.csv',
            'name': 'CD012233',
            'category': 'DTA',
            'enabled': 1
        },
        {
            'data_path': 'INT_data/Criteria/CD011686_Criteria.csv',
            'scores_path': 'INT_data/Ranking/CD011686_ranked.csv',
            'name': 'CD011686',
            'category': 'Intervention',
            'enabled': 1
        },
    ])

    example_config.to_csv('pseudo_dataset_config_example.csv', index=False)
    logger.info("Created example configuration file: pseudo_dataset_config_example.csv")
    return example_config


def load_config(config_path):
    """Load dataset configuration from CSV file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Creating example configuration file...")
        create_example_config()
        logger.info("Please edit pseudo_dataset_config_example.csv and rename to pseudo_dataset_config.csv")
        return []

    try:
        config_df = pd.read_csv(config_path)

        # Validate required columns
        required_cols = ['data_path', 'scores_path', 'name', 'category', 'enabled']
        missing_cols = [col for col in required_cols if col not in config_df.columns]

        if missing_cols:
            logger.error(f"Config file missing required columns: {missing_cols}")
            return []

        # Filter enabled datasets
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


def run_single_experiment(dataset_config, top_pct, bottom_pct):
    """
    Run a single pseudo-labeling experiment.

    Args:
        dataset_config: Dataset configuration dict
        top_pct: Top percentage for pseudo-positive labels
        bottom_pct: Bottom percentage for pseudo-negative labels

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
        random_state=RANDOM_STATE,  # Fixed random state
        verbose=False
    )

    # Create output directory
    output_dir = (f"pseudo_results/{dataset_config['category']}/{dataset_config['name']}/"
                  f"top{top_pct}_bottom{bottom_pct}")

    # Initialize and run
    asreview = PseudoASReviewComplete(
        config=config,
        output_dir=output_dir
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

    return results


def print_experiment_summary(result, dataset_name, top_pct, bottom_pct):
    """Print results for a single experiment (deterministic method - 1 run)."""

    if not result:
        return

    print("\n" + "=" * 70)
    print(f"EXPERIMENT RESULTS: {dataset_name} (Top {top_pct}% / Bottom {bottom_pct}%)")
    print("=" * 70)

    print(f"\nPseudo-Labeling (Phase 1):")
    print(f"   Pseudo-labeled: {len(result['pseudo_labels'])} documents")
    print(f"   - Pseudo-POSITIVE: {sum(result['pseudo_labels'])}")
    print(f"   - Pseudo-NEGATIVE: {len(result['pseudo_labels']) - sum(result['pseudo_labels'])}")

    print(f"\nActive Learning (Phase 2):")
    print(f"   Phase 2 iterations: {result['phase2_ended_at_iteration']}")
    print(f"   Screened at stopping: {result['proportion_screened'] * 100:.1f}%")
    print(f"   Screening recall: {result['screening_recall'] * 100:.1f}%")
    print(f"   Total recall (with classifier): {result['total_recall'] * 100:.1f}%")
    print(f"   Stopped early: {result['stopped_early']} ({result['stopping_reason']})")

    print(f"\nMilestones (Phase 3):")
    print(f"   Iteration at 95% recall: {result['iteration_at_95_recall']}")
    print(f"   Iteration at 100% recall: {result['iteration_at_100_recall']}")
    print(f"   Iterations (95%→100%): {result['iteration_at_100_recall'] - result['iteration_at_95_recall']}")

    print(f"\nWork Savings:")
    print(f"   Phase 2 WSS: {result['actual_wss'] * 100:.1f}%")
    print(f"   Screened at 100% recall: {result['proportion_screened_at_100'] * 100:.1f}%")
    print(f"   Work saved vs random: {(1 - result['proportion_screened_at_100']) * 100:.1f}%")

    print("=" * 70)


def print_dataset_summary(all_results, dataset_name):
    """Print overall summary across all experiments for a dataset."""
    dataset_results = [r for r in all_results if r['dataset_name'] == dataset_name]

    if not dataset_results:
        return

    print("\n\n" + "=" * 70)
    print(f"DATASET SUMMARY: {dataset_name}")
    print("=" * 70)

    print(f"Total experiments: {len(dataset_results)}")
    print(f"Dataset info: {dataset_results[0]['total_records']} records, "
          f"{dataset_results[0]['total_relevant']} relevant "
          f"({dataset_results[0]['prevalence'] * 100:.1f}% prevalence)")

    print(f"\nResults by pseudo-labeling percentage:")
    print(f"{'Top %':<8} {'Pseudo':<10} {'Recall':<10} {'Iter@95':<10} {'Iter@100':<10} {'Work Saved':<12}")
    print("-" * 70)

    for result in sorted(dataset_results, key=lambda x: x['top_pct']):
        print(f"{result['top_pct']:<8.1f} "
              f"{len(result['pseudo_labels']):<10} "
              f"{result['total_recall'] * 100:<10.1f} "
              f"{result['iteration_at_95_recall']:<10} "
              f"{result['iteration_at_100_recall']:<10} "
              f"{(1 - result['proportion_screened_at_100']) * 100:<12.1f}")

    print("=" * 70)


def main():
    """Main execution function."""
    logger.info("Starting Pseudo-ASReview Multi-Dataset Experiments")

    # Load configuration
    datasets = load_config(CONFIG_CSV)

    if not datasets:
        logger.error("No datasets to process. Check your configuration file.")
        return

    # Verify files exist
    valid_datasets, missing = verify_dataset_files(datasets)

    if missing:
        logger.warning(f"{len(missing)} files not found:")
        for m in missing:
            logger.warning(f"  - {m}")
        response = input(f"\nContinue with {len(valid_datasets)} valid datasets? (y/n): ")
        if response.lower() != 'y':
            logger.info("Processing cancelled")
            return

    datasets = valid_datasets

    # Calculate total experiments (deterministic - 1 run per percentage)
    total_experiments = len(datasets) * len(PSEUDO_LABEL_PERCENTAGES)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"EXPERIMENT CONFIGURATION")
    logger.info(f"{'=' * 70}")
    logger.info(f"Datasets: {len(datasets)}")
    logger.info(f"Pseudo-labeling percentages: {PSEUDO_LABEL_PERCENTAGES}")
    logger.info(f"Method: Deterministic (1 run per percentage)")
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

        # Test each pseudo-labeling percentage
        for pct_idx, pct in enumerate(PSEUDO_LABEL_PERCENTAGES, 1):
            experiment_count += 1
            logger.info(f"\n  Experiment {pct_idx}/{len(PSEUDO_LABEL_PERCENTAGES)}: "
                        f"Top {pct}% / Bottom {pct}% "
                        f"(Overall: {experiment_count}/{total_experiments})")

            try:
                pct = pct/100
                result = run_single_experiment(
                    dataset_config, pct, pct
                )
                all_results.append(result)

                # Print brief results
                print(f"    ✓ Pseudo: {len(result['pseudo_labels'])} docs | "
                      f"Recall: {result['total_recall'] * 100:.1f}% | "
                      f"Screened: {result['proportion_screened'] * 100:.1f}% | "
                      f"95% at iter {result['iteration_at_95_recall']}")

                # Print detailed summary for this experiment
                print_experiment_summary(result, dataset_config['name'], pct, pct)

            except Exception as e:
                logger.error(f"    ✗ Experiment failed: {str(e)}")
                continue

        # Print overall summary for this dataset
        print_dataset_summary(all_results, dataset_config['name'])

    # ========================================
    # SAVE COMPREHENSIVE RESULTS
    # ========================================
    print("\n\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    if all_results:
        # Create detailed results DataFrame
        summary_df = pd.DataFrame([
            {
                'dataset': r['dataset_name'],
                'category': r['dataset_category'],
                'top_pct': r['top_pct'],
                'bottom_pct': r['bottom_pct'],
                'total_records': r['total_records'],
                'total_relevant': r['total_relevant'],
                'prevalence': r['prevalence'] * 100,
                'pseudo_total': len(r['pseudo_labels']),
                'pseudo_positive': sum(r['pseudo_labels']),
                'pseudo_negative': len(r['pseudo_labels']) - sum(r['pseudo_labels']),
                'phase2_iterations': r['phase2_ended_at_iteration'],
                'screened_pct': r['proportion_screened'] * 100,
                'screening_recall': r['screening_recall'] * 100,
                'total_recall': r['total_recall'] * 100,
                'iter_at_95': r['iteration_at_95_recall'],
                'iter_at_100': r['iteration_at_100_recall'],
                'iter_95_to_100': r['iteration_at_100_recall'] - r['iteration_at_95_recall'],
                'phase2_wss': r['actual_wss'] * 100,
                'screened_at_100': r['proportion_screened_at_100'] * 100,
                'work_saved': (1 - r['proportion_screened_at_100']) * 100,
                'stopped_early': r['stopped_early'],
                'stopping_reason': r['stopping_reason']
            }
            for r in all_results
        ])

        # Save main summary
        os.makedirs('pseudo_results', exist_ok=True)
        summary_path = 'pseudo_results/pseudo_experiments_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"✓ Main summary saved: {summary_path}")

        # Save per-dataset summaries
        for dataset_name in summary_df['dataset'].unique():
            dataset_df = summary_df[summary_df['dataset'] == dataset_name]
            dataset_path = f'pseudo_results/{dataset_name}_experiments.csv'
            dataset_df.to_csv(dataset_path, index=False)
            logger.info(f"✓ Dataset summary saved: {dataset_path}")

    print("=" * 70)
    logger.info("\n🎉 All experiments complete!")


if __name__ == "__main__":
    main()