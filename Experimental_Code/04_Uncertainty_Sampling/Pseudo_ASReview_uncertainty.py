"""
Pseudo-ASReview with Complete Screening (Phases 1-3)
QUERY STRATEGY: UNCERTAINTY SAMPLING

This script runs Pseudo-ASReview with:
- Phase 1: LLM-based pseudo-labeling
- Phase 2: Active learning with stopping criteria (UNCERTAINTY SAMPLING)
- Phase 3: Continue to 100% recall
- Tracking of iterations at 95% and 100% recall
"""

import pandas as pd
from pseudo_asreview_complete_uncertainty import PseudoASReviewComplete, PseudoLabelConfig


def run_single_review(data_path: str, scores_path: str, output_dir: str,
                      top_pct: float = 0.5, bottom_pct: float = 0.5,
                      random_state: int = 1):
    """Run complete LLM pseudo-labeling with Phase 3 using UNCERTAINTY SAMPLING"""

    # 1. Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    texts = df['text'].tolist()
    labels = df['Status'].tolist()
    record_ids = df['PMID'].tolist()

    print(f"Loaded {len(texts)} records")
    print(f"  - Relevant: {sum(labels)} ({sum(labels) / len(labels) * 100:.1f}%)")
    print(f"  - Irrelevant: {len(labels) - sum(labels)} ({(len(labels) - sum(labels)) / len(labels) * 100:.1f}%)")

    # 2. Configure
    config = PseudoLabelConfig(
        top_percentage=top_pct,
        bottom_percentage=bottom_pct,
        llm_score_column='score',
        min_screened_percentage=0.10,
        consecutive_irrelevant=50,
        random_state=random_state,
        verbose=True
    )

    # 3. Initialize with COMPLETE version (UNCERTAINTY SAMPLING)
    asreview = PseudoASReviewComplete(
        config=config,
        output_dir=output_dir
    )

    asreview.load_data(
        texts=texts,
        labels=labels,
        record_ids=record_ids
    )

    # 4. Run complete screening (Phases 1, 2, AND 3) with UNCERTAINTY SAMPLING
    results = asreview.run_llm_pseudo_labeling_phases_1_and_2(
        scores_csv_path=scores_path
    )

    # 5. Print summary
    print("\n" + "=" * 70)
    print("COMPLETE SCREENING SUMMARY (UNCERTAINTY SAMPLING)")
    print("=" * 70)
    print(f"Output directory: {results['output_dir']}")

    print(f"\n📋 PHASE 1: LLM Pseudo-Labeling")
    print(f"   Pseudo-labeled: {len(results['pseudo_indices'])} documents")
    print(f"   - Pseudo-POSITIVE: {sum(results['pseudo_labels'])}")
    print(f"   - Pseudo-NEGATIVE: {len(results['pseudo_labels']) - sum(results['pseudo_labels'])}")

    print(f"\n📊 PHASE 2: Active Learning (UNCERTAINTY SAMPLING)")
    print(f"   Query Strategy: UNCERTAINTY SAMPLING")
    print(f"   Iterations: {results['phase2_ended_at_iteration']}")
    print(f"   Records screened: {results['n_screened']}/{len(texts)} ({results['proportion_screened'] * 100:.1f}%)")
    print(f"   Screening recall: {results['screening_recall'] * 100:.1f}%")
    print(f"   Total recall (with classifier): {results['total_recall'] * 100:.1f}%")
    print(f"   Stopped: {results['stopped_early']} ({results['stopping_reason']})")

    # Phase 2 classifier metrics
    if 'classification_metrics' in results.get('phase2_metrics', {}):
        cm = results['phase2_metrics']['classification_metrics']
        print(f"\n   Phase 2 Classifier Performance:")
        print(f"   - Remaining documents: {cm.get('n_remaining', 0)}")
        print(f"   - Relevant in remaining: {cm.get('n_relevant_in_remaining', 0)}")
        print(f"   - Precision: {cm.get('precision', 0):.3f}")
        print(f"   - Recall: {cm.get('recall', 0):.3f}")
        print(f"   - F1-Score: {cm.get('f1_score', 0):.3f}")
        print(f"   - AUC-ROC: {cm.get('auc_roc', 0):.3f}")

    print(f"\n🎯 PHASE 3: Continue to 100% Recall")
    print(f"   Additional iterations: {results['phase3_iterations']}")
    print(f"   Total screened: {results['total_screened_at_100']}/{len(texts)} ({results['proportion_screened_at_100'] * 100:.1f}%)")

    print(f"\n📈 KEY MILESTONES:")
    print(f"   95% recall: iteration {results['iteration_at_95_recall']}")
    print(f"   100% recall: iteration {results['iteration_at_100_recall']}")
    print(f"   Iterations (95%→100%): {results['iteration_at_100_recall'] - results['iteration_at_95_recall']}")

    print(f"\n💰 WORK SAVINGS:")
    print(f"   Phase 2 Actual WSS: {results['actual_wss']:.3f} ({results['actual_wss'] * 100:.1f}%)")
    print(f"   At 100% recall: {results['proportion_screened_at_100'] * 100:.1f}% screened")
    print(f"   Work saved vs random: {(1-results['proportion_screened_at_100']) * 100:.1f}%")

    print(f"\n💾 FILES SAVED:")
    print(f"   - COMPLETE_REPORT.txt (comprehensive stats)")
    print(f"   - recall_curve_complete.png (with milestones)")
    print(f"   - phase2/predictions_final.csv (with FIXED metrics)")
    print(f"   - phase3/all_screened_to_100_recall.csv")
    print("=" * 70)

    return results


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("Pseudo-ASReview Complete Screening with UNCERTAINTY SAMPLING")
    print("="*70)
    print()

    # Run for multiple random states
    DATA_PATH = "INT_data/Criteria/CD011768_Criteria.csv"
    SCORES_PATH = "INT_data/Ranking/CD011768_ranked.csv"

    for run in range(1, 11):
        print(f"\n{'='*70}")
        print(f"RUN {run}/10 (UNCERTAINTY SAMPLING)")
        print(f"{'='*70}\n")

        OUTPUT_DIR = f"results/uncertainty/INT/CD011768/complete_run_{run}/"

        results = run_single_review(
            data_path=DATA_PATH,
            scores_path=SCORES_PATH,
            output_dir=OUTPUT_DIR,
            top_pct=0.1,
            bottom_pct=0.1,
            random_state=run
        )

        print(f"\n✅ Run {run} complete!")
        print(f"   Query Strategy: UNCERTAINTY SAMPLING")
        print(f"   95% recall: {results['iteration_at_95_recall']} iterations")
        print(f"   100% recall: {results['iteration_at_100_recall']} iterations")
