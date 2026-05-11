"""
Wrapper to add Phase 3 (continue to 100% recall) to ASReview

Usage:
    Instead of: from asreview_with_artifacts import ASReviewWithArtifacts
    Use: from asreview_complete import ASReviewComplete
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import matplotlib.pyplot as plt
from asreview_with_artifacts import ASReviewWithArtifacts, SAFEConfig


class ASReviewComplete(ASReviewWithArtifacts):
    """
    Extends ASReviewWithArtifacts to add Phase 3 (continue to 100% recall)
    and track iterations at 95% and 100% recall milestones.
    """

    def __init__(self, config: Optional[SAFEConfig] = None,
                 output_dir: str = "./asreview_output"):
        super().__init__(config, output_dir)

        # Track milestone iterations
        self.iteration_at_95_recall = None
        self.iteration_at_100_recall = None

        # Create Phase 3 directory
        (self.output_dir / "phase3").mkdir(exist_ok=True)

    def run_safe_phases_1_and_2(self, max_iterations: Optional[int] = None):
        """
        Run Phases 1 & 2 as normal, then continue to 100% recall.
        Returns results with milestone tracking.
        """
        # Run original Phase 1 and 2
        results = super().run_safe_phases_1_and_2(max_iterations=max_iterations)

        # Store Phase 2 end state
        phase2_iterations = results['n_iterations']
        phase2_recall = results['final_recall']

        if self.config.verbose:
            print(f"\n{'=' * 70}")
            print("PHASE 2 COMPLETE - STARTING PHASE 3")
            print(f"{'=' * 70}")
            print(f"Phase 2 ended at: {phase2_iterations} iterations")
            print(f"Phase 2 recall: {phase2_recall * 100:.1f}%")
            print(f"\n🎯 PHASE 3: Continuing to 100% recall...")
            print(f"{'=' * 70}\n")

        # Track milestones
        n_total_relevant = np.sum(self.y_labels == 1)
        n_relevant_found = np.sum(self.y_labels[self.labeled_mask] == 1)
        current_recall = n_relevant_found / n_total_relevant

        # Check if 95% already reached
        if current_recall >= 0.95:
            self.iteration_at_95_recall = self.iteration

        # Phase 3: Continue until 100% recall
        n_total = len(self.texts)
        phase3_start_iteration = self.iteration

        while current_recall < 1.0:
            self.iteration += 1

            # Get state
            labeled_indices = np.where(self.labeled_mask)[0]
            unlabeled_indices = np.where(~self.labeled_mask)[0]

            if len(unlabeled_indices) == 0:
                break

            # Train model
            X_train = self.X_features[labeled_indices]
            y_train = self.y_labels[labeled_indices]
            X_unlabeled = self.X_features[unlabeled_indices]

            X_train_resampled, y_train_resampled = self.resampler.resample(
                X_train, y_train, n_total
            )
            self.classifier.fit(X_train_resampled, y_train_resampled)

            # Query next record
            next_idx, confidence = self.query_strategy.query(
                self.classifier, X_unlabeled, unlabeled_indices
            )

            # Get label
            label = int(self.y_labels[next_idx])
            record_id = self.record_ids[next_idx]

            # Update
            self.labeled_mask[next_idx] = True
            if label == 1:
                n_relevant_found += 1

            n_screened = np.sum(self.labeled_mask)
            current_recall = n_relevant_found / n_total_relevant

            # Track 95% milestone
            if current_recall >= 0.95 and self.iteration_at_95_recall is None:
                self.iteration_at_95_recall = self.iteration
                if self.config.verbose:
                    print(f"\n🎯 MILESTONE: 95% recall at iteration {self.iteration}")
                    print(f"   Found {n_relevant_found}/{n_total_relevant} relevant")
                    print(f"   Screened {n_screened}/{n_total} ({n_screened / n_total * 100:.1f}%)\n")

            # Update metrics
            self.metrics.update(
                iteration=self.iteration,
                record_id=record_id,
                label=label,
                confidence=confidence,
                n_relevant_found=n_relevant_found,
                n_total_relevant=n_total_relevant,
                n_screened=n_screened,
                n_total_records=n_total
            )

            # Progress
            if self.config.verbose and (self.iteration % 50 == 0):
                print(f"Iter {self.iteration:4d}: "
                      f"Recall={current_recall:.3f} ({n_relevant_found}/{n_total_relevant}), "
                      f"Screened={n_screened}/{n_total} ({n_screened / n_total * 100:.1f}%), "
                      f"Label={'✓' if label == 1 else '✗'}")

        # Track 100% milestone
        self.iteration_at_100_recall = self.iteration

        if self.config.verbose:
            print(f"\n{'=' * 70}")
            print("✅ 100% RECALL ACHIEVED!")
            print(f"{'=' * 70}")
            print(f"Iteration at 100%: {self.iteration_at_100_recall}")
            print(
                f"Total screened: {np.sum(self.labeled_mask)}/{n_total} ({np.sum(self.labeled_mask) / n_total * 100:.1f}%)")
            print(f"{'=' * 70}\n")

        # Save Phase 3 artifacts
        self._save_phase3_artifacts()

        # Update results
        phase3_iterations = self.iteration - phase2_iterations
        results.update({
            'iteration_at_95_recall': self.iteration_at_95_recall,
            'iteration_at_100_recall': self.iteration_at_100_recall,
            'phase2_ended_at_iteration': phase2_iterations,
            'phase3_iterations': phase3_iterations,
            'final_recall_100': 1.0,
            'total_screened_at_100': int(np.sum(self.labeled_mask)),
            'proportion_screened_at_100': np.sum(self.labeled_mask) / n_total
        })

        # Generate complete report
        self._generate_complete_report(results)

        return results

    def _save_phase3_artifacts(self):
        """Save Phase 3 artifacts"""
        phase3_dir = self.output_dir / "phase3"

        screened_indices = np.where(self.labeled_mask)[0]
        screened_df = pd.DataFrame({
            'record_id': self.record_ids[screened_indices],
            'true_label': self.y_labels[screened_indices],
            'text': self.texts[screened_indices]
        })

        screened_path = phase3_dir / "all_screened_to_100_recall.csv"
        screened_df.to_csv(screened_path, index=False, encoding='utf-8')

        if self.config.verbose:
            print(f"💾 Saved Phase 3 documents: {screened_path}")

    def _generate_complete_report(self, results: Dict):
        """Generate comprehensive report"""
        report_path = self.output_dir / "COMPLETE_REPORT.txt"

        with open(report_path, 'w', encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("ASReview COMPLETE SCREENING REPORT (Phases 1-3)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("DATASET\n" + "-" * 70 + "\n")
            f.write(f"Total records: {len(self.texts)}\n")
            f.write(f"Total relevant: {results['n_total_relevant']}\n")
            f.write(f"Prevalence: {results['n_total_relevant'] / len(self.texts) * 100:.2f}%\n\n")

            f.write("PHASE 2 SUMMARY (with stopping criteria)\n" + "-" * 70 + "\n")
            f.write(f"Iterations: {results['phase2_ended_at_iteration']}\n")
            f.write(f"Records screened: {results['n_screened']} ({results['proportion_screened'] * 100:.1f}%)\n")
            f.write(f"Recall achieved: {results['final_recall'] * 100:.1f}%\n")
            f.write(f"Stopped: {results['stopped_early']} ({results['stopping_reason']})\n")

            # Phase 2 classifier metrics
            if 'classification_metrics' in self.phase2_metrics and self.phase2_metrics['classification_metrics']:
                cm = self.phase2_metrics['classification_metrics']
                f.write(f"\nPhase 2 Classifier Performance (on remaining unlabeled):\n")
                f.write(f"  Precision: {cm['precision']:.3f}\n")
                f.write(f"  Recall: {cm['recall']:.3f}\n")
                f.write(f"  F1-Score: {cm['f1_score']:.3f}\n")
                f.write(f"  AUC-ROC: {cm['auc_roc']:.3f}\n")
                f.write(f"  Accuracy: {cm['accuracy']:.3f}\n")
                f.write(f"  Confusion: TP={cm['true_positives']}, TN={cm['true_negatives']}, ")
                f.write(f"FP={cm['false_positives']}, FN={cm['false_negatives']}\n")
            f.write("\n")

            f.write("PHASE 3 SUMMARY (continue to 100%)\n" + "-" * 70 + "\n")
            f.write(f"Additional iterations: {results['phase3_iterations']}\n")
            f.write(
                f"Total screened: {results['total_screened_at_100']} ({results['proportion_screened_at_100'] * 100:.1f}%)\n\n")

            f.write("KEY MILESTONES\n" + "-" * 70 + "\n")
            f.write(f"📊 Iteration at 95% recall: {results['iteration_at_95_recall']}\n")
            f.write(f"✅ Iteration at 100% recall: {results['iteration_at_100_recall']}\n")
            f.write(
                f"📈 Iterations (95%→100%): {results['iteration_at_100_recall'] - results['iteration_at_95_recall']}\n\n")

            f.write("WORK SAVINGS\n" + "-" * 70 + "\n")
            f.write(
                f"At Phase 2 end: {results['proportion_screened'] * 100:.1f}% screened, {results['final_recall'] * 100:.1f}% recall\n")
            f.write(f"  WSS@95%: {results.get('wss_95', 0):.3f}\n")
            f.write(f"  Actual WSS (Phase 2): {self.phase2_metrics.get('actual_wss', 0):.3f}\n")
            f.write(f"  Total work (screened + predicted positive): {self.phase2_metrics.get('total_work', 0)}\n")
            f.write(f"At 95% recall: iteration {results['iteration_at_95_recall']}\n")
            f.write(
                f"At 100% recall: iteration {results['iteration_at_100_recall']}, {results['proportion_screened_at_100'] * 100:.1f}% screened\n")
            f.write(f"Work saved vs random: {(1 - results['proportion_screened_at_100']) * 100:.1f}%\n\n")

            f.write("=" * 70 + "\n")

        if self.config.verbose:
            print(f"📄 Complete report: {report_path}")

    def plot_recall_curve(self, save_path: Optional[str] = None):
        """Plot with milestone markers"""
        if save_path is None:
            save_path = str(self.output_dir / "recall_curve_complete.png")

        proportions, recalls = self.metrics.get_recall_curve()

        if len(proportions) == 0:
            return

        plt.figure(figsize=(12, 7))
        plt.plot(proportions * 100, recalls * 100, 'b-', linewidth=2, label='Active Learning')
        plt.plot([0, 100], [0, 100], 'r--', linewidth=1, label='Random')

        # 95% marker
        if self.iteration_at_95_recall:
            for p, r in zip(proportions, recalls):
                if r >= 0.95:
                    plt.axvline(p * 100, color='g', linestyle=':', alpha=0.5)
                    plt.axhline(95, color='g', linestyle=':', alpha=0.5)
                    plt.plot(p * 100, 95, 'go', markersize=10,
                             label=f'95% @ {p * 100:.1f}%')
                    break

        # 100% marker
        if self.iteration_at_100_recall:
            final_p = proportions[-1]
            plt.axvline(final_p * 100, color='purple', linestyle=':', alpha=0.5)
            plt.plot(final_p * 100, 100, 'mo', markersize=10,
                     label=f'100% @ {final_p * 100:.1f}%')

        plt.xlabel('Proportion Screened (%)', fontsize=12)
        plt.ylabel('Recall (%)', fontsize=12)
        plt.title('Complete Recall Curve to 100%', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 100])
        plt.ylim([0, 105])

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot: {save_path}")
        plt.close()