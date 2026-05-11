"""
Wrapper to add Phase 3 and fix classifier metrics for Pseudo-ASReview

Usage:
    Instead of: from asreview_llm_pseudo_labeling import ASReviewLLMPseudoLabeling
    Use: from pseudo_asreview_complete import PseudoASReviewComplete
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, confusion_matrix
)
from asreview_llm_pseudo_labeling import (
    ASReviewLLMPseudoLabeling,
    PseudoLabelConfig
)


class PseudoASReviewComplete(ASReviewLLMPseudoLabeling):
    """
    Extends ASReviewLLMPseudoLabeling to:
    1. Add Phase 3 (continue to 100% recall)
    2. Fix Phase 2 classifier metrics display
    3. Track iterations at 95% and 100% recall
    """

    def __init__(self, config: Optional[PseudoLabelConfig] = None,
                 output_dir: str = "./asreview_llm_output"):
        super().__init__(config, output_dir)

        # Track milestones
        self.iteration_at_95_recall = None
        self.iteration_at_100_recall = None

        # Create Phase 3 directory
        (self.output_dir / "phase3").mkdir(exist_ok=True)

    def run_llm_pseudo_labeling_phases_1_and_2(self, scores_csv_path: str,
                                               max_iterations: Optional[int] = None):
        """
        Run Phases 1 & 2, then continue to 100% recall.
        Fixes Phase 2 classifier metrics display.
        """
        # Run original Phase 1 and 2
        results = super().run_llm_pseudo_labeling_phases_1_and_2(
            scores_csv_path=scores_csv_path,
            max_iterations=max_iterations
        )

        # FIX: Recalculate and display Phase 2 classifier metrics properly
        self._fix_phase2_classifier_metrics()

        # Store Phase 2 end state
        phase2_iterations = results['n_iterations']
        phase2_recall = results['total_recall']

        if self.config.verbose:
            print(f"\n{'=' * 70}")
            print("PHASE 2 COMPLETE - STARTING PHASE 3")
            print(f"{'=' * 70}")
            print(f"Phase 2 ended at: {phase2_iterations} iterations")
            print(f"Phase 2 total recall: {phase2_recall * 100:.1f}%")
            print(f"\n🎯 PHASE 3: Continuing to 100% recall...")
            print(f"{'=' * 70}\n")

        # Track milestones
        n_total_relevant = np.sum(self.y_labels == 1)
        n_relevant_found = np.sum(self.y_labels[self.labeled_mask] == 1)
        current_recall = n_relevant_found / n_total_relevant

        # Check if 95% already reached
        if current_recall >= 0.95:
            self.iteration_at_95_recall = self.iteration

        # Phase 3: Continue to 100%
        n_total = len(self.texts)
        phase3_start_iteration = self.iteration

        while current_recall < 1.0:
            self.iteration += 1

            # Build training set (truly labeled + pseudo-labeled)
            training_mask = self.labeled_mask | self.pseudo_labeled_mask
            training_indices = np.where(training_mask)[0]
            unlabeled_indices = np.where(~self.labeled_mask)[0]

            if len(unlabeled_indices) == 0:
                break

            # Get labels for training
            y_train = np.zeros(len(training_indices), dtype=int)
            for i, idx in enumerate(training_indices):
                if self.labeled_mask[idx]:
                    y_train[i] = self.y_labels[idx]  # True label
                else:
                    y_train[i] = self.y_pseudo[idx]  # Pseudo-label

            X_train = self.X_features[training_indices]
            X_unlabeled = self.X_features[unlabeled_indices]

            # Train with resampling
            X_train_resampled, y_train_resampled = self.resampler.resample(
                X_train, y_train, n_total
            )

            from sklearn.naive_bayes import MultinomialNB
            if self.classifier is None:
                self.classifier = MultinomialNB()
            self.classifier.fit(X_train_resampled, y_train_resampled)

            # Query next
            next_idx, confidence = self.query_strategy.query(
                self.classifier, X_unlabeled, unlabeled_indices
            )

            # Get label and update
            label = int(self.y_labels[next_idx])
            record_id = self.record_ids[next_idx]

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

        # Track 100%
        self.iteration_at_100_recall = self.iteration

        if self.config.verbose:
            print(f"\n{'=' * 70}")
            print("✅ 100% RECALL ACHIEVED!")
            print(f"{'=' * 70}")
            print(f"Iteration at 100%: {self.iteration_at_100_recall}")
            print(
                f"Total screened: {np.sum(self.labeled_mask)}/{n_total} ({np.sum(self.labeled_mask) / n_total * 100:.1f}%)")
            print(f"{'=' * 70}\n")

        # Save Phase 3
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
            'proportion_screened_at_100': np.sum(self.labeled_mask) / n_total,
            'phase2_metrics': self.phase2_metrics  # Include all Phase 2 metrics including pseudo_label_stats
        })

        # Generate complete report
        self._generate_complete_report(results)

        return results

    def _fix_phase2_classifier_metrics(self):
        """
        FIX: Recalculate and properly display Phase 2 classifier metrics.

        The original code sometimes shows 0.000 for metrics because:
        1. Classifier might not be trained on final state
        2. Remaining set might be all one class

        This method retrains on final state and calculates metrics properly.
        """
        # Get remaining unlabeled
        unlabeled_indices = np.where(~self.labeled_mask)[0]

        if len(unlabeled_indices) == 0:
            if self.config.verbose:
                print("\nℹ️  No remaining documents - all were screened")
            return

        # Retrain classifier on final Phase 2 state
        training_mask = self.labeled_mask | self.pseudo_labeled_mask
        training_indices = np.where(training_mask)[0]

        y_train = np.zeros(len(training_indices), dtype=int)
        for i, idx in enumerate(training_indices):
            if self.labeled_mask[idx]:
                y_train[i] = self.y_labels[idx]
            else:
                y_train[i] = self.y_pseudo[idx]

        X_train = self.X_features[training_indices]
        X_unlabeled = self.X_features[unlabeled_indices]

        # Retrain
        from sklearn.naive_bayes import MultinomialNB
        classifier = MultinomialNB()
        X_train_resampled, y_train_resampled = self.resampler.resample(
            X_train, y_train, len(self.texts)
        )
        classifier.fit(X_train_resampled, y_train_resampled)

        # Get predictions
        probas = classifier.predict_proba(X_unlabeled)
        if probas.shape[1] == 1:
            probas = np.column_stack([1 - probas[:, 0], probas[:, 0]])

        y_true = self.y_labels[unlabeled_indices]
        y_pred = (probas[:, 1] >= 0.5).astype(int)
        y_pred_proba = probas[:, 1]

        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        try:
            auc = roc_auc_score(y_true, y_pred_proba)
            if np.isnan(auc):
                auc = 0.0
        except:
            auc = 0.0

        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
        except:
            if len(np.unique(y_true)) == 1:
                if np.unique(y_true)[0] == 0:
                    tn = len(y_true)
                    fp = fn = tp = 0
                else:
                    tp = len(y_true)
                    tn = fp = fn = 0
            else:
                tn = fp = fn = tp = 0

        # Calculate pseudo-label statistics in training set
        training_mask = self.labeled_mask | self.pseudo_labeled_mask
        n_total_training = np.sum(training_mask)
        n_truly_labeled = np.sum(self.labeled_mask)
        n_still_pseudo = np.sum(self.pseudo_labeled_mask & ~self.labeled_mask)
        n_corrected_pseudo = np.sum(self.pseudo_labeled_mask & self.labeled_mask)
        pct_pseudo_in_training = (n_still_pseudo / n_total_training * 100) if n_total_training > 0 else 0
        pct_corrected = (n_corrected_pseudo / np.sum(self.pseudo_labeled_mask) * 100) if np.sum(
            self.pseudo_labeled_mask) > 0 else 0

        # Update metrics
        self.phase2_metrics['classification_metrics'] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'auc_roc': float(auc),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'n_relevant_in_remaining': int(np.sum(y_true == 1)),
            'n_remaining': len(unlabeled_indices)
        }

        # Add pseudo-label tracking metrics
        self.phase2_metrics['pseudo_label_stats'] = {
            'n_total_training_set': int(n_total_training),
            'n_truly_labeled': int(n_truly_labeled),
            'n_still_pseudo_labeled': int(n_still_pseudo),
            'n_corrected_pseudo_labels': int(n_corrected_pseudo),
            'pct_pseudo_in_training': float(pct_pseudo_in_training),
            'pct_pseudo_corrected': float(pct_corrected)
        }

        # Display fixed metrics
        if self.config.verbose:
            n_rel_remaining = int(np.sum(y_true == 1))
            print(f"\n{'=' * 70}")
            print(f"Phase 2 Classifier Performance ({len(unlabeled_indices)} remaining unlabeled):")
            print(f"  Relevant in remaining: {n_rel_remaining} ({n_rel_remaining / len(unlabeled_indices) * 100:.1f}%)")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-Score: {f1:.3f}")
            print(f"  AUC-ROC: {auc:.3f}")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

            # Display pseudo-label statistics
            if 'pseudo_label_stats' in self.phase2_metrics:
                ps = self.phase2_metrics['pseudo_label_stats']
                print(f"\nPhase 2 Training Set Composition:")
                print(f"  Total training samples: {ps['n_total_training_set']}")
                print(
                    f"  Truly labeled (queried): {ps['n_truly_labeled']} ({ps['n_truly_labeled'] / ps['n_total_training_set'] * 100:.1f}%)")
                print(f"  Still pseudo-labeled: {ps['n_still_pseudo_labeled']} ({ps['pct_pseudo_in_training']:.1f}%)")
                print(
                    f"  Pseudo-labels corrected: {ps['n_corrected_pseudo_labels']}/{ps['n_still_pseudo_labeled'] + ps['n_corrected_pseudo_labels']} ({ps['pct_pseudo_corrected']:.1f}%)")
            print(f"{'=' * 70}\n")

    def _save_phase3_artifacts(self):
        """Save Phase 3 artifacts"""
        phase3_dir = self.output_dir / "phase3"

        screened_indices = np.where(self.labeled_mask)[0]
        screened_df = pd.DataFrame({
            'record_id': self.record_ids[screened_indices],
            'true_label': self.y_labels[screened_indices],
            'was_pseudo_labeled': self.pseudo_labeled_mask[screened_indices],
            'text': self.texts[screened_indices]
        })

        screened_path = phase3_dir / "all_screened_to_100_recall.csv"
        screened_df.to_csv(screened_path, index=False, encoding='utf-8')

        if self.config.verbose:
            print(f"💾 Saved Phase 3 documents: {screened_path}")

    def _generate_complete_report(self, results: Dict):
        """Generate comprehensive report"""
        report_path = self.output_dir / "COMPLETE_REPORT.txt"

        pseudo_indices = results.get('pseudo_indices', [])
        pseudo_labels = results.get('pseudo_labels', [])

        with open(report_path, 'w', encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("Pseudo-ASReview COMPLETE REPORT (Phases 1-3)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("DATASET\n" + "-" * 70 + "\n")
            f.write(f"Total records: {len(self.texts)}\n")
            f.write(f"Total relevant: {results['n_total_relevant']}\n")
            f.write(f"Prevalence: {results['n_total_relevant'] / len(self.texts) * 100:.2f}%\n\n")

            f.write("PHASE 1: LLM PSEUDO-LABELING\n" + "-" * 70 + "\n")
            f.write(f"Pseudo-labeled: {len(pseudo_indices)} ({len(pseudo_indices) / len(self.texts) * 100:.2f}%)\n")
            f.write(f"  Pseudo-POSITIVE: {np.sum(pseudo_labels == 1)}\n")
            f.write(f"  Pseudo-NEGATIVE: {np.sum(pseudo_labels == 0)}\n\n")

            f.write("PHASE 2: ACTIVE LEARNING (with stopping)\n" + "-" * 70 + "\n")
            f.write(f"Iterations: {results['phase2_ended_at_iteration']}\n")
            f.write(f"Records screened: {results['n_screened']} ({results['proportion_screened'] * 100:.1f}%)\n")
            f.write(f"Screening recall: {results['screening_recall'] * 100:.1f}%\n")
            f.write(f"Total recall (with classifier): {results['total_recall'] * 100:.1f}%\n")
            f.write(f"Stopped: {results['stopped_early']} ({results['stopping_reason']})\n")

            # Phase 2 classifier metrics
            if 'classification_metrics' in self.phase2_metrics and self.phase2_metrics['classification_metrics']:
                cm = self.phase2_metrics['classification_metrics']
                f.write(f"\nPhase 2 Classifier Performance:\n")
                f.write(f"  Remaining documents: {cm['n_remaining']}\n")
                f.write(f"  Relevant in remaining: {cm['n_relevant_in_remaining']}\n")
                f.write(f"  Precision: {cm['precision']:.3f}\n")
                f.write(f"  Recall: {cm['recall']:.3f}\n")
                f.write(f"  F1-Score: {cm['f1_score']:.3f}\n")
                f.write(f"  AUC-ROC: {cm['auc_roc']:.3f}\n")
                f.write(
                    f"  Confusion: TP={cm['true_positives']}, TN={cm['true_negatives']}, FP={cm['false_positives']}, FN={cm['false_negatives']}\n")

            # Phase 2 pseudo-label statistics
            if 'pseudo_label_stats' in self.phase2_metrics:
                ps = self.phase2_metrics['pseudo_label_stats']
                f.write(f"\nPhase 2 Training Set Composition:\n")
                f.write(f"  Total training samples: {ps['n_total_training_set']}\n")
                f.write(
                    f"  Truly labeled (queried): {ps['n_truly_labeled']} ({ps['n_truly_labeled'] / ps['n_total_training_set'] * 100:.1f}%)\n")
                f.write(
                    f"  Still pseudo-labeled: {ps['n_still_pseudo_labeled']} ({ps['pct_pseudo_in_training']:.1f}%)\n")
                f.write(
                    f"  Pseudo-labels corrected: {ps['n_corrected_pseudo_labels']}/{ps['n_still_pseudo_labeled'] + ps['n_corrected_pseudo_labels']} ({ps['pct_pseudo_corrected']:.1f}%)\n")
                f.write(f"  ⚠️  High % of pseudo-labels may indicate overfitting risk\n")
            f.write("\n")

            f.write("PHASE 3: CONTINUE TO 100%\n" + "-" * 70 + "\n")
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
                f"At Phase 2: {results['proportion_screened'] * 100:.1f}% screened, {results['total_recall'] * 100:.1f}% recall, WSS={results['actual_wss']:.3f}\n")
            f.write(f"At 95%: iteration {results['iteration_at_95_recall']}\n")
            f.write(
                f"At 100%: iteration {results['iteration_at_100_recall']}, {results['proportion_screened_at_100'] * 100:.1f}% screened\n")
            f.write(f"Work saved vs random: {(1 - results['proportion_screened_at_100']) * 100:.1f}%\n\n")

            f.write("=" * 70 + "\n")

        if self.config.verbose:
            print(f"📄 Complete report: {report_path}")

    def plot_recall_curve(self, save_path: Optional[str] = None):
        """Plot with milestones"""
        if save_path is None:
            save_path = str(self.output_dir / "recall_curve_complete.png")

        proportions, recalls = self.metrics.get_recall_curve()

        if len(proportions) == 0:
            return

        plt.figure(figsize=(12, 7))
        plt.plot(proportions * 100, recalls * 100, 'b-', linewidth=2,
                 label='LLM Pseudo-Labeling')
        plt.plot([0, 100], [0, 100], 'r--', linewidth=1, label='Random')

        # 95%
        if self.iteration_at_95_recall:
            for p, r in zip(proportions, recalls):
                if r >= 0.95:
                    plt.axvline(p * 100, color='g', linestyle=':', alpha=0.5)
                    plt.axhline(95, color='g', linestyle=':', alpha=0.5)
                    plt.plot(p * 100, 95, 'go', markersize=10,
                             label=f'95% @ {p * 100:.1f}%')
                    break

        # 100%
        if self.iteration_at_100_recall:
            final_p = proportions[-1]
            plt.axvline(final_p * 100, color='purple', linestyle=':', alpha=0.5)
            plt.plot(final_p * 100, 100, 'mo', markersize=10,
                     label=f'100% @ {final_p * 100:.1f}%')

        plt.xlabel('Proportion Screened (%)', fontsize=12)
        plt.ylabel('Recall (%)', fontsize=12)
        plt.title('LLM Pseudo-Labeling: Complete Recall Curve',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 100])
        plt.ylim([0, 105])

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot: {save_path}")
        plt.close()