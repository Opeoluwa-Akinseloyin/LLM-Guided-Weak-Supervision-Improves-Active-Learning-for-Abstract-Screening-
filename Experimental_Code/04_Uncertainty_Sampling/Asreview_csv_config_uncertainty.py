"""
ASReview with Complete Screening - CSV Config Version
QUERY STRATEGY: UNCERTAINTY SAMPLING

This version reads dataset configurations from a CSV file and uses
UNCERTAINTY SAMPLING for all active learning queries.
"""

import pandas as pd
from asreview_complete_uncertainty import ASReviewComplete, SAFEConfig
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# CONFIGURATION FILE
# ========================================
CONFIG_CSV = '../dataset_config.csv'  # Path to your configuration CSV

# Run parameters
N_RUNS = 10
PRIOR_KNOWLEDGE_PCT = 0.01
MIN_PRIOR_RELEVANT = 1
CONSECUTIVE_IRRELEVANT = 50


def create_example_config():
    """
    Create an example configuration CSV if it doesn't exist.

    The CSV should have columns:
    - path: Path to the dataset CSV file
    - name: Short name for the dataset
    - category: Category (e.g., DTA, Intervention)
    - enabled: 1 to process, 0 to skip
    """
    example_config = pd.DataFrame([
        {
            'path': 'DTA_data/Criteria/CD012233_Criteria.csv',
            'name': 'CD012233',
            'category': 'DTA',
            'enabled': 1
        },
        {
            'path': 'DTA_data/Criteria/CD012254_Criteria.csv',
            'name': 'CD012254',
            'category': 'DTA',
            'enabled': 1
        },
        {
            'path': 'Intervention_data/Intervention_001.csv',
            'name': 'Intervention_001',
            'category': 'Intervention',
            'enabled': 1
        },
        # Add more datasets here...
    ])

    example_config.to_csv('dataset_config_example.csv', index=False)
    logger.info("Created example configuration file: dataset_config_example.csv")
    return example_config


def load_config(config_path):
    """
    Load dataset configuration from CSV file.

    Args:
        config_path: Path to configuration CSV

    Returns:
        List of enabled dataset configurations
    """
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Creating example configuration file...")
        create_example_config()
        logger.info("Please edit dataset_config_example.csv and rename to dataset_config.csv")
        return []

    try:
        config_df = pd.read_csv(config_path)

        # Validate required columns
        required_cols = ['path', 'name', 'category', 'enabled']
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
    """
    Check which dataset files actually exist.

    Args:
        datasets: List of dataset configurations

    Returns:
        Tuple of (valid_datasets, missing_paths)
    """
    valid = []
    missing = []

    for dataset in datasets:
        alt_dataset = "../" + dataset['path']
        if os.path.exists(dataset['path']):
            valid.append(dataset)
        else:
            missing.append(dataset['path'])
            logger.warning(f"Dataset file not found: {dataset['path']}")

    return valid, missing


def load_dataset(dataset_config):
    """Load a single dataset from CSV."""
    try:
        df = pd.read_csv(dataset_config['path'])
        logger.info(f"Loaded {len(df)} records from {dataset_config['name']}")

        texts = df['text'].tolist()
        labels = df['Status'].tolist()
        record_ids = df['PMID'].tolist()
        topic = df["Topic"].iloc[0] if 'Topic' in df.columns else dataset_config['name']

        prevalence = sum(labels) / len(labels) * 100
        logger.info(f"  Relevant papers: {sum(labels)} ({prevalence:.1f}%)")

        return texts, labels, record_ids, topic, dataset_config

    except Exception as e:
        logger.error(f"Failed to load {dataset_config['name']}: {str(e)}")
        return None


def run_asreview_on_dataset(texts, labels, record_ids, topic, dataset_info, run_number):
    """Run ASReview with UNCERTAINTY SAMPLING on a single dataset."""
    config = SAFEConfig(
        prior_knowledge_percentage=PRIOR_KNOWLEDGE_PCT,
        min_prior_relevant=MIN_PRIOR_RELEVANT,
        consecutive_irrelevant=CONSECUTIVE_IRRELEVANT,
        random_state=run_number,
        verbose=False
    )

    # Output directory labeled with "uncertainty"
    output_dir = f"results/uncertainty/{dataset_info['category']}/{topic}/{topic}_uncertainty_run_{run_number}"

    asreview = ASReviewComplete(
        config=config,
        output_dir=output_dir
    )

    asreview.load_data(
        texts=texts,
        labels=labels,
        record_ids=record_ids
    )

    results = asreview.run_safe_phases_1_and_2()

    results['dataset_name'] = dataset_info['name']
    results['dataset_category'] = dataset_info['category']
    results['run_number'] = run_number
    results['query_strategy'] = 'uncertainty'  # Track which strategy was used

    return results


def print_dataset_summary(all_results, dataset_name):
    """Print summary statistics for all runs of a dataset."""
    dataset_results = [r for r in all_results if r['dataset_name'] == dataset_name]

    if not dataset_results:
        return

    print("\n" + "=" * 70)
    print(f"SUMMARY FOR: {dataset_name} (UNCERTAINTY SAMPLING)")
    print("=" * 70)

    avg_phase2_iter = sum(r['phase2_ended_at_iteration'] for r in dataset_results) / len(dataset_results)
    avg_screened_pct = sum(r['proportion_screened'] for r in dataset_results) / len(dataset_results) * 100
    avg_recall = sum(r['final_recall'] for r in dataset_results) / len(dataset_results) * 100
    avg_iter_95 = sum(r['iteration_at_95_recall'] for r in dataset_results) / len(dataset_results)
    avg_iter_100 = sum(r['iteration_at_100_recall'] for r in dataset_results) / len(dataset_results)

    print(f"Query Strategy: UNCERTAINTY SAMPLING")
    print(f"Runs completed: {len(dataset_results)}")
    print(f"\nAverage Phase 2 iterations: {avg_phase2_iter:.1f}")
    print(f"Average screened at stopping: {avg_screened_pct:.1f}%")
    print(f"Average final recall: {avg_recall:.1f}%")
    print(f"Average iteration at 95% recall: {avg_iter_95:.1f}")
    print(f"Average iteration at 100% recall: {avg_iter_100:.1f}")
    print("=" * 70)


def main():
    """Main execution function."""
    logger.info("Starting CSV-configured multi-dataset ASReview processing")
    logger.info("QUERY STRATEGY: UNCERTAINTY SAMPLING")

    # Load configuration
    datasets = load_config(CONFIG_CSV)

    if not datasets:
        logger.error("No datasets to process. Check your configuration file.")
        return

    # Verify files exist
    valid_datasets, missing = verify_dataset_files(datasets)

    if missing:
        logger.warning(f"{len(missing)} dataset files not found")
        response = input(f"Continue with {len(valid_datasets)} valid datasets? (y/n): ")
        if response.lower() != 'y':
            logger.info("Processing cancelled")
            return

    datasets = valid_datasets

    logger.info(f"\nProcessing {len(datasets)} datasets with {N_RUNS} runs each")
    logger.info(f"Query Strategy: UNCERTAINTY SAMPLING")
    logger.info(f"Total experiments: {len(datasets) * N_RUNS}\n")

    all_results = []

    # Process each dataset
    for dataset_idx, dataset_config in enumerate(datasets, 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"DATASET {dataset_idx}/{len(datasets)}: {dataset_config['name']}")
        logger.info(f"{'=' * 70}")

        # Load dataset
        dataset_data = load_dataset(dataset_config)
        if dataset_data is None:
            logger.warning(f"Skipping {dataset_config['name']} due to loading error")
            continue

        texts, labels, record_ids, topic, dataset_info = dataset_data

        # Run multiple iterations
        for run_num in range(1, N_RUNS + 1):
            logger.info(f"  Run {run_num}/{N_RUNS} (Uncertainty Sampling)")

            try:
                results = run_asreview_on_dataset(
                    texts, labels, record_ids, topic, dataset_info, run_num
                )
                all_results.append(results)

                print(f"    ✓ Recall: {results['final_recall'] * 100:.1f}% | "
                      f"Screened: {results['proportion_screened'] * 100:.1f}% | "
                      f"95%→100%: {results['iteration_at_100_recall'] - results['iteration_at_95_recall']} iterations")

            except Exception as e:
                logger.error(f"    ✗ Run {run_num} failed: {str(e)}")
                continue

        # Print dataset summary
        print_dataset_summary(all_results, dataset_config['name'])

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n\n" + "=" * 70)
    print("OVERALL SUMMARY (UNCERTAINTY SAMPLING)")
    print("=" * 70)
    print(f"Query Strategy: UNCERTAINTY SAMPLING")
    print(f"Total datasets processed: {len(set(r['dataset_name'] for r in all_results))}")
    print(f"Total runs completed: {len(all_results)}")
    print(f"Failed runs: {len(datasets) * N_RUNS - len(all_results)}")

    # Save summary CSV
    if all_results:
        summary_df = pd.DataFrame([
            {
                'dataset': r['dataset_name'],
                'category': r['dataset_category'],
                'run': r['run_number'],
                'query_strategy': r['query_strategy'],
                'phase2_iterations': r['phase2_ended_at_iteration'],
                'screened_pct': r['proportion_screened'] * 100,
                'final_recall': r['final_recall'] * 100,
                'iter_at_95': r['iteration_at_95_recall'],
                'iter_at_100': r['iteration_at_100_recall'],
                'stopped_early': r['stopped_early'],
                'stopping_reason': r['stopping_reason']
            }
            for r in all_results
        ])

        summary_path = 'results/uncertainty_multi_dataset_summary.csv'
        os.makedirs('results', exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved to: {summary_path}")

        # Also save per-category summaries
        for category in summary_df['category'].unique():
            category_df = summary_df[summary_df['category'] == category]
            category_path = f'results/{category}_uncertainty_summary.csv'
            category_df.to_csv(category_path, index=False)
            logger.info(f"Category summary saved: {category_path}")

    print("=" * 70)
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
