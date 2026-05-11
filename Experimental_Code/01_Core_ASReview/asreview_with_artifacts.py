"""
ASReview SAFE Implementation with Comprehensive Artifact Saving

Saves:
1. Phase 1 classifier (after initial training)
2. Phase 1 predictions (on all unlabeled data)
3. Phase 2 classifier (final trained model)
4. Phase 2 predictions (final predictions on remaining unlabeled)
5. Documents queried (Phase 1 prior knowledge + Phase 2 screened records)
"""

import numpy as np
import pandas as pd
import pickle
import joblib
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from asreview_safe_simplified import (
    ASReviewSAFE as BaseASReview,
    SAFEConfig,
    TFIDFFeatureExtractor,
    DynamicResampler,
    CertaintyQueryStrategy,
    PerformanceMetrics,
    SimplifiedStoppingCriteria
)


class ASReviewWithArtifacts(BaseASReview):
    """
    Enhanced ASReview that saves all artifacts during screening.

    Creates organized folder structure:
    output_dir/
    ├── phase1/
    │   ├── classifier.pkl
    │   ├── predictions.csv
    │   ├── queried_documents.csv
    │   └── feature_extractor.pkl
    ├── phase2/
    │   ├── classifier_final.pkl
    │   ├── predictions_final.csv
    │   ├── queried_documents.csv
    │   └── all_screened_documents.csv
    ├── config.json
    └── summary.json
    """

    def __init__(self, config: Optional[SAFEConfig] = None,
                 output_dir: str = "./asreview_output"):
        """
        Initialize ASReview with artifact saving.

        Args:
            config: SAFE configuration
            output_dir: Directory to save all artifacts
        """
        super().__init__(config)
        self.output_dir = Path(output_dir)
        self.phase1_classifier = None
        self.phase1_predictions = None
        self.phase1_metrics = {}
        self.phase2_metrics = {}

        # Create output directories
        self._setup_directories()

    def _calculate_classification_metrics(self, y_true: np.ndarray,
                                         y_pred: np.ndarray,
                                         y_pred_proba: np.ndarray) -> Dict:
        """
        Calculate classification metrics for model evaluation.

        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_pred_proba: Predicted probabilities for positive class

        Returns:
            Dictionary with precision, recall, F1, AUC, etc.
        """
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            accuracy_score, roc_auc_score, confusion_matrix
        )

        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        # AUC (handle case where only one class present)
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
            if np.isnan(auc):
                auc = 0.0
        except:
            auc = 0.0

        # Confusion matrix (handle edge cases)
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
        except:
            # Handle case where only one class present
            if len(np.unique(y_true)) == 1:
                if np.unique(y_true)[0] == 0:  # Only negatives
                    tn = len(y_true)
                    fp = 0
                    fn = 0
                    tp = 0
                else:  # Only positives
                    tn = 0
                    fp = 0
                    fn = 0
                    tp = len(y_true)
            else:
                tn = fp = fn = tp = 0

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'auc_roc': float(auc),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        }

    def _setup_directories(self):
        """Create organized directory structure"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "phase1").mkdir(exist_ok=True)
        (self.output_dir / "phase2").mkdir(exist_ok=True)

        if self.config.verbose:
            print(f"\n📁 Output directory: {self.output_dir.absolute()}")

    def _save_config(self):
        """Save configuration to JSON"""
        import json

        config_dict = {
            'prior_knowledge_percentage': self.config.prior_knowledge_percentage,
            'min_prior_relevant': self.config.min_prior_relevant,
            'max_prior_percentage': self.config.max_prior_percentage,
            'min_screened_percentage': self.config.min_screened_percentage,
            'consecutive_irrelevant': self.config.consecutive_irrelevant,
            'expected_relevant_multiplier': self.config.expected_relevant_multiplier,
            'random_state': self.config.random_state,
            'timestamp': datetime.now().isoformat()
        }

        config_path = self.output_dir / "config.json"
        with open(config_path, 'w', encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        if self.config.verbose:
            print(f"💾 Saved configuration to: {config_path}")

    def _save_phase1_artifacts(self, prior_indices: np.ndarray, prior_labels: np.ndarray):
        """
        Save Phase 1 artifacts after initial training.

        Saves:
        - Trained classifier
        - Feature extractor
        - Predictions on all unlabeled data
        - Prior knowledge documents
        """
        phase1_dir = self.output_dir / "phase1"

        # Train Phase 1 classifier on prior knowledge
        X_prior = self.X_features[prior_indices]
        y_prior = prior_labels

        # Apply resampling
        X_resampled, y_resampled = self.resampler.resample(
            X_prior, y_prior, len(self.texts)
        )

        # Train classifier
        self.classifier.fit(X_resampled, y_resampled)
        self.phase1_classifier = self.classifier

        # 1. Save classifier
        classifier_path = phase1_dir / "classifier.pkl"
        joblib.dump(self.classifier, classifier_path)
        if self.config.verbose:
            print(f"💾 Saved Phase 1 classifier to: {classifier_path}")

        # 2. Save feature extractor
        extractor_path = phase1_dir / "feature_extractor.pkl"
        joblib.dump(self.feature_extractor, extractor_path)
        if self.config.verbose:
            print(f"💾 Saved feature extractor to: {extractor_path}")

        # 3. Make predictions on all unlabeled data
        unlabeled_mask = ~self.labeled_mask
        unlabeled_indices = np.where(unlabeled_mask)[0]

        if len(unlabeled_indices) > 0:
            X_unlabeled = self.X_features[unlabeled_indices]
            predictions = self.classifier.predict_proba(X_unlabeled)

            # Handle case where predict_proba returns only one column
            if predictions.shape[1] == 1:
                # If only one column, assume it's the probability of the positive class
                predictions = np.column_stack([1 - predictions[:, 0], predictions[:, 0]])

            # Save predictions
            predictions_df = pd.DataFrame({
                'record_id': self.record_ids[unlabeled_indices],
                'probability_irrelevant': predictions[:, 0],
                'probability_relevant': predictions[:, 1],
                'predicted_label': self.classifier.predict(X_unlabeled)
            })

            # Sort by relevance probability (descending)
            predictions_df = predictions_df.sort_values(
                'probability_relevant',
                ascending=False
            ).reset_index(drop=True)

            predictions_path = phase1_dir / "predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)
            self.phase1_predictions = predictions_df

            if self.config.verbose:
                print(f"💾 Saved Phase 1 predictions ({len(predictions_df)} records) to: {predictions_path}")

        # Calculate Phase 1 classification metrics on unlabeled set
        if len(unlabeled_indices) > 0:
            y_true_unlabeled = self.y_labels[unlabeled_indices]
            y_pred_unlabeled = self.classifier.predict(X_unlabeled)
            y_pred_proba_unlabeled = predictions[:, 1]  # Probability of relevant class

            phase1_classification_metrics = self._calculate_classification_metrics(
                y_true_unlabeled, y_pred_unlabeled, y_pred_proba_unlabeled
            )

            if self.config.verbose:
                print(f"\n📊 Phase 1 Classifier Performance (on unlabeled set):")
                print(f"   Precision: {phase1_classification_metrics['precision']:.3f}")
                print(f"   Recall: {phase1_classification_metrics['recall']:.3f}")
                print(f"   F1-Score: {phase1_classification_metrics['f1_score']:.3f}")
                print(f"   AUC-ROC: {phase1_classification_metrics['auc_roc']:.3f}")
                print(f"   Accuracy: {phase1_classification_metrics['accuracy']:.3f}")
        else:
            phase1_classification_metrics = {}

        # 4. Save queried documents (prior knowledge)
        queried_df = pd.DataFrame({
            'record_id': self.record_ids[prior_indices],
            'label': prior_labels,
            'text': self.texts[prior_indices],
            'phase': 'phase1_prior_knowledge',
            'iteration': 0,
            'true_label': self.y_labels[prior_indices] if self.y_labels is not None else None
        })

        queried_path = phase1_dir / "queried_documents.csv"
        queried_df.to_csv(queried_path, index=False)

        if self.config.verbose:
            print(f"💾 Saved Phase 1 queried documents ({len(queried_df)} records) to: {queried_path}")

        # Calculate Phase 1 metrics (Actual WSS and Recall)
        n_screened = len(prior_indices)
        n_total = len(self.texts)
        n_relevant_found = np.sum(prior_labels == 1)
        n_total_relevant = np.sum(self.y_labels == 1) if self.y_labels is not None else 1

        proportion_screened = n_screened / n_total
        work_saved = 1 - proportion_screened
        screening_recall_phase1 = n_relevant_found / n_total_relevant

        # Calculate total recall (screening + classifier on remaining)
        if len(unlabeled_indices) > 0 and phase1_classification_metrics:
            classifier_recall = phase1_classification_metrics['recall']
            n_relevant_in_remaining = n_total_relevant - n_relevant_found
            n_relevant_found_by_classifier = classifier_recall * n_relevant_in_remaining
            total_recall_phase1 = (n_relevant_found + n_relevant_found_by_classifier) / n_total_relevant

            # Calculate predicted positives on remaining (TP + FP that need screening)
            y_pred_remaining = self.classifier.predict(X_unlabeled)
            n_predicted_positive_remaining = np.sum(y_pred_remaining == 1)
        else:
            classifier_recall = 0.0
            total_recall_phase1 = screening_recall_phase1
            n_predicted_positive_remaining = 0

        # Correct formula: Account for FP that still need screening
        # Total work = screened + predicted positive on remaining
        total_work = n_screened + n_predicted_positive_remaining
        proportion_total_work = total_work / n_total
        actual_wss_phase1 = (1 - proportion_total_work) - (1 - total_recall_phase1)

        if self.config.verbose:
            print(f"\n📊 Phase 1 Metrics:")
            print(f"   Screening Recall: {screening_recall_phase1:.3f} ({screening_recall_phase1*100:.1f}%)")
            if len(unlabeled_indices) > 0:
                print(f"   Classifier Recall (on remaining): {classifier_recall:.3f} ({classifier_recall*100:.1f}%)")
                print(f"   Total Recall: {total_recall_phase1:.3f} ({total_recall_phase1*100:.1f}%)")
                print(f"   Predicted positive on remaining: {n_predicted_positive_remaining} (need manual screening)")
            print(f"   Proportion Screened: {proportion_screened:.3f} ({proportion_screened*100:.1f}%)")
            print(f"   Total Work (screened + predicted positive): {total_work} ({proportion_total_work*100:.1f}%)")
            print(f"   Work Saved: {(1-proportion_total_work):.3f} ({(1-proportion_total_work)*100:.1f}%)")
            print(f"   Actual WSS: {actual_wss_phase1:.3f} ({actual_wss_phase1*100:.1f}%)")

        # Store for summary
        self.phase1_metrics = {
            'screening_recall': float(screening_recall_phase1),
            'classifier_recall': float(classifier_recall) if len(unlabeled_indices) > 0 else 0.0,
            'total_recall': float(total_recall_phase1),
            'n_predicted_positive_remaining': int(n_predicted_positive_remaining),
            'proportion_screened': float(proportion_screened),
            'total_work': int(total_work),
            'proportion_total_work': float(proportion_total_work),
            'work_saved': float(1 - proportion_total_work),
            'actual_wss': float(actual_wss_phase1),
            'classification_metrics': phase1_classification_metrics
        }

    def _save_phase2_artifacts(self):
        """
        Save Phase 2 artifacts after screening completes.

        Saves:
        - Final trained classifier
        - Final predictions on remaining unlabeled data
        - All screened documents (with iteration info)
        - Phase 2 only screened documents
        """
        phase2_dir = self.output_dir / "phase2"

        # Get all labeled data for final training
        labeled_indices = np.where(self.labeled_mask)[0]
        X_labeled = self.X_features[labeled_indices]
        y_labeled = self.y_labels[labeled_indices]

        # Train final classifier
        X_resampled, y_resampled = self.resampler.resample(
            X_labeled, y_labeled, len(self.texts)
        )
        self.classifier.fit(X_resampled, y_resampled)

        # 1. Save final classifier
        classifier_path = phase2_dir / "classifier_final.pkl"
        joblib.dump(self.classifier, classifier_path)
        if self.config.verbose:
            print(f"💾 Saved Phase 2 final classifier to: {classifier_path}")

        # 2. Make predictions on remaining unlabeled data
        unlabeled_mask = ~self.labeled_mask
        unlabeled_indices = np.where(unlabeled_mask)[0]

        if len(unlabeled_indices) > 0:
            X_unlabeled = self.X_features[unlabeled_indices]
            predictions = self.classifier.predict_proba(X_unlabeled)

            # Handle case where predict_proba returns only one column
            if predictions.shape[1] == 1:
                # If only one column, assume it's the probability of the positive class
                predictions = np.column_stack([1 - predictions[:, 0], predictions[:, 0]])

            predictions_df = pd.DataFrame({
                'record_id': self.record_ids[unlabeled_indices],
                'probability_irrelevant': predictions[:, 0],
                'probability_relevant': predictions[:, 1],
                'predicted_label': self.classifier.predict(X_unlabeled),
                'true_label': self.y_labels[unlabeled_indices] if self.y_labels is not None else None
            })

            # Sort by relevance probability
            predictions_df = predictions_df.sort_values(
                'probability_relevant',
                ascending=False
            ).reset_index(drop=True)

            predictions_path = phase2_dir / "predictions_final.csv"
            predictions_df.to_csv(predictions_path, index=False)

            if self.config.verbose:
                print(f"💾 Saved Phase 2 final predictions ({len(predictions_df)} records) to: {predictions_path}")

            # Calculate Phase 2 classification metrics on remaining unlabeled set
            y_true_unlabeled = self.y_labels[unlabeled_indices]
            y_pred_unlabeled = self.classifier.predict(X_unlabeled)
            y_pred_proba_unlabeled = predictions[:, 1]  # Probability of relevant class

            phase2_classification_metrics = self._calculate_classification_metrics(
                y_true_unlabeled, y_pred_unlabeled, y_pred_proba_unlabeled
            )

            if self.config.verbose:
                print(f"\n📊 Phase 2 Classifier Performance (on remaining unlabeled set):")
                print(f"   Precision: {phase2_classification_metrics['precision']:.3f}")
                print(f"   Recall: {phase2_classification_metrics['recall']:.3f}")
                print(f"   F1-Score: {phase2_classification_metrics['f1_score']:.3f}")
                print(f"   AUC-ROC: {phase2_classification_metrics['auc_roc']:.3f}")
                print(f"   Accuracy: {phase2_classification_metrics['accuracy']:.3f}")
        else:
            phase2_classification_metrics = {}

        # 3. Save Phase 2 screened documents (from iteration data)
        if self.metrics.iteration_data:
            phase2_screened = []

            for data in self.metrics.iteration_data:
                phase2_screened.append({
                    'record_id': data['record_id'],
                    'iteration': data['iteration'],
                    'label': data['label'],
                    'confidence': data['confidence'],
                    'phase': 'phase2_screening'
                })

            phase2_df = pd.DataFrame(phase2_screened)

            # Add text and true labels
            record_id_to_idx = {rid: idx for idx, rid in enumerate(self.record_ids)}
            phase2_df['text'] = phase2_df['record_id'].apply(
                lambda rid: self.texts[record_id_to_idx[rid]]
            )
            if self.y_labels is not None:
                phase2_df['true_label'] = phase2_df['record_id'].apply(
                    lambda rid: self.y_labels[record_id_to_idx[rid]]
                )

            queried_path = phase2_dir / "queried_documents.csv"
            phase2_df.to_csv(queried_path, index=False)

            if self.config.verbose:
                print(f"💾 Saved Phase 2 queried documents ({len(phase2_df)} records) to: {queried_path}")

            # 4. Save ALL screened documents (Phase 1 + Phase 2)
            # Load Phase 1 documents
            phase1_path = self.output_dir / "phase1" / "queried_documents.csv"
            phase1_df = pd.read_csv(phase1_path)

            # Combine
            all_screened = pd.concat([phase1_df, phase2_df], ignore_index=True)

            all_path = phase2_dir / "all_screened_documents.csv"
            all_screened.to_csv(all_path, index=False)

            if self.config.verbose:
                print(f"💾 Saved ALL screened documents ({len(all_screened)} records) to: {all_path}")

        # Calculate Phase 2 metrics (Actual WSS and Recall)
        labeled_indices = np.where(self.labeled_mask)[0]
        n_screened = len(labeled_indices)
        n_total = len(self.texts)
        n_relevant_found = np.sum(self.y_labels[labeled_indices] == 1) if self.y_labels is not None else 0
        n_total_relevant = np.sum(self.y_labels == 1) if self.y_labels is not None else 1

        proportion_screened = n_screened / n_total
        work_saved = 1 - proportion_screened
        screening_recall_phase2 = n_relevant_found / n_total_relevant

        # Calculate total recall (screening + classifier on remaining)
        if len(unlabeled_indices) > 0 and phase2_classification_metrics:
            classifier_recall = phase2_classification_metrics['recall']
            n_relevant_in_remaining = n_total_relevant - n_relevant_found
            n_relevant_found_by_classifier = classifier_recall * n_relevant_in_remaining
            total_recall_phase2 = (n_relevant_found + n_relevant_found_by_classifier) / n_total_relevant

            # Calculate predicted positives on remaining (TP + FP that need screening)
            y_pred_remaining = self.classifier.predict(X_unlabeled)
            n_predicted_positive_remaining = np.sum(y_pred_remaining == 1)
        else:
            classifier_recall = 0.0
            total_recall_phase2 = screening_recall_phase2
            n_predicted_positive_remaining = 0

        # Correct formula: Account for FP that still need screening
        # Total work = screened + predicted positive on remaining
        total_work = n_screened + n_predicted_positive_remaining
        proportion_total_work = total_work / n_total
        actual_wss_phase2 = (1 - proportion_total_work) - (1 - total_recall_phase2)


        if self.config.verbose:
            print(f"\n📊 Phase 2 Final Metrics:")
            print(f"   Screening Recall: {screening_recall_phase2:.3f} ({screening_recall_phase2*100:.1f}%)")
            if len(unlabeled_indices) > 0:
                print(f"   Classifier Recall (on remaining): {classifier_recall:.3f} ({classifier_recall*100:.1f}%)")
                print(f"   Total Recall: {total_recall_phase2:.3f} ({total_recall_phase2*100:.1f}%)")
                print(f"   Predicted positive on remaining: {n_predicted_positive_remaining} (need manual screening)")
            print(f"   Proportion Screened: {proportion_screened:.3f} ({proportion_screened*100:.1f}%)")
            print(f"   Total Work (screened + predicted positive): {total_work} ({proportion_total_work*100:.1f}%)")
            print(f"   Work Saved: {(1-proportion_total_work):.3f} ({(1-proportion_total_work)*100:.1f}%)")
            print(f"   Actual WSS: {actual_wss_phase2:.3f} ({actual_wss_phase2*100:.1f}%)")

        # Store for summary
        self.phase2_metrics = {
            'screening_recall': float(screening_recall_phase2),
            'classifier_recall': float(classifier_recall) if len(unlabeled_indices) > 0 else 0.0,
            'total_recall': float(total_recall_phase2),
            'n_predicted_positive_remaining': int(n_predicted_positive_remaining),
            'proportion_screened': float(proportion_screened),
            'total_work': int(total_work),
            'proportion_total_work': float(proportion_total_work),
            'work_saved': float(1 - proportion_total_work),
            'actual_wss': float(actual_wss_phase2),
            'classification_metrics': phase2_classification_metrics
        }

    def _save_summary(self, results: Dict):
        """Save summary statistics"""
        import json
        stopping_criteria = results.get("stopping_criteria") or {}

        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset': {
                'total_records': len(self.texts),
                'total_relevant': int(results['n_total_relevant']),
                'prevalence': float(results['n_total_relevant'] / len(self.texts))
            },
            'phase1': {
                'prior_knowledge_size': int(np.sum(self.labeled_mask) - len(self.metrics.iteration_data)),
                'prior_knowledge_percentage': float((np.sum(self.labeled_mask) - len(self.metrics.iteration_data)) / len(self.texts)),
                'screening_recall': self.phase1_metrics.get('screening_recall', 0.0),
                'classifier_recall': self.phase1_metrics.get('classifier_recall', 0.0),
                'total_recall': self.phase1_metrics.get('total_recall', 0.0),
                'n_predicted_positive_remaining': int(self.phase1_metrics.get('n_predicted_positive_remaining', 0)),
                'total_work': int(self.phase1_metrics.get('total_work', 0)),
                'proportion_total_work': float(self.phase1_metrics.get('proportion_total_work', 0.0)),
                'actual_wss': self.phase1_metrics.get('actual_wss', 0.0),
                'work_saved': self.phase1_metrics.get('work_saved', 0.0)
            },
            'phase2': {
                'iterations': int(results['n_iterations']),
                'stopped_early': bool(results['stopped_early']),
                'stopping_reason': results['stopping_reason'],
                'stopping_criteria': {k: bool(v) for k, v in stopping_criteria.items()},
                'records_screened': int(results['n_screened']),
                'proportion_screened': float(results['proportion_screened']),
                'screening_recall': self.phase2_metrics.get('screening_recall', 0.0),
                'classifier_recall': self.phase2_metrics.get('classifier_recall', 0.0),
                'total_recall': self.phase2_metrics.get('total_recall', 0.0),
                'n_predicted_positive_remaining': int(self.phase2_metrics.get('n_predicted_positive_remaining', 0)),
                'total_work': int(self.phase2_metrics.get('total_work', 0)),
                'proportion_total_work': float(self.phase2_metrics.get('proportion_total_work', 0.0)),
                'actual_wss': self.phase2_metrics.get('actual_wss', 0.0),
                'work_saved': self.phase2_metrics.get('work_saved', 0.0)
            },
            'performance': {
                'relevant_found': int(results['n_relevant_found']),
                'final_recall': float(results['final_recall']),
                'wss_95': float(results['wss_95']),
                'atd': float(results['atd']),
                'work_saved': float(1 - results['proportion_screened'])
            },
            'time': {
                'elapsed_seconds': float(results['elapsed_time'])
            }
        }

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w', encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if self.config.verbose:
            print(f"💾 Saved summary to: {summary_path}")

    def run_safe_phases_1_and_2(self, max_iterations: Optional[int] = None) -> Dict:
        """
        Run complete SAFE Phase 1 and 2 with artifact saving.

        Returns:
            Dictionary with results
        """
        if self.config.verbose:
            print("\n" + "="*70)
            print("ASReview SAFE with Comprehensive Artifact Saving")
            print("="*70)

        # Save configuration
        self._save_config()

        # Phase 1: Adaptive prior knowledge
        prior_indices, prior_labels = self.phase1_adaptive_prior_knowledge()

        # Extract features (needed for predictions)
        if self.config.verbose:
            print("\nExtracting features for all records...")
        self.X_features = self.feature_extractor.fit_transform(self.texts)

        # Save Phase 1 artifacts
        if self.config.verbose:
            print("\n" + "="*70)
            print("Saving Phase 1 Artifacts")
            print("="*70)
        self._save_phase1_artifacts(prior_indices, prior_labels)

        # Phase 2: Screening
        results = self.phase2_screen_with_stopping(max_iterations=max_iterations)

        # Save Phase 2 artifacts
        if self.config.verbose:
            print("\n" + "="*70)
            print("Saving Phase 2 Artifacts")
            print("="*70)
        self._save_phase2_artifacts()

        # Save summary
        self._save_summary(results)

        # Generate report
        self._generate_report(results)

        results['prior_indices'] = prior_indices
        results['prior_labels'] = prior_labels
        results['output_dir'] = str(self.output_dir)

        return results

    def _generate_report(self, results: Dict):
        """Generate human-readable report"""
        report_path = self.output_dir / "REPORT.txt"

        with open(report_path, 'w', encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write("ASReview SAFE Screening Report\n")
            f.write("="*70 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("DATASET\n")
            f.write("-"*70 + "\n")
            f.write(f"Total records: {len(self.texts)}\n")
            f.write(f"Total relevant: {results['n_total_relevant']}\n")
            f.write(f"Prevalence: {results['n_total_relevant']/len(self.texts)*100:.2f}%\n\n")

            f.write("PHASE 1: PRIOR KNOWLEDGE\n")
            f.write("-"*70 + "\n")
            prior_size = np.sum(self.labeled_mask) - len(self.metrics.iteration_data)
            f.write(f"Records used: {prior_size} ({prior_size/len(self.texts)*100:.2f}%)\n")
            f.write(f"Relevant found: {np.sum(self.y_labels[self.labeled_mask][:prior_size] == 1)}\n")
            f.write(f"Screening Recall: {self.phase1_metrics.get('screening_recall', 0)*100:.1f}%\n")
            if self.phase1_metrics.get('classifier_recall', 0) > 0:
                f.write(f"Classifier Recall (on remaining): {self.phase1_metrics.get('classifier_recall', 0)*100:.1f}%\n")
                f.write(f"Total Recall: {self.phase1_metrics.get('total_recall', 0)*100:.1f}%\n")
                f.write(f"Predicted positive on remaining: {self.phase1_metrics.get('n_predicted_positive_remaining', 0)}\n")
            f.write(f"Total Work (screened + predicted positive): {self.phase1_metrics.get('total_work', prior_size)} ({self.phase1_metrics.get('proportion_total_work', 0)*100:.1f}%)\n")
            f.write(f"Actual WSS: {self.phase1_metrics.get('actual_wss', 0):.3f} ({self.phase1_metrics.get('actual_wss', 0)*100:.1f}%)\n\n")

            # Phase 1 classifier metrics
            if 'classification_metrics' in self.phase1_metrics and self.phase1_metrics['classification_metrics']:
                cm = self.phase1_metrics['classification_metrics']
                f.write("Phase 1 Classifier Performance (on unlabeled set):\n")
                f.write(f"  Precision: {cm['precision']:.3f}\n")
                f.write(f"  Recall: {cm['recall']:.3f}\n")
                f.write(f"  F1-Score: {cm['f1_score']:.3f}\n")
                f.write(f"  AUC-ROC: {cm['auc_roc']:.3f}\n")
                f.write(f"  Accuracy: {cm['accuracy']:.3f}\n")
                f.write(f"  Confusion Matrix: TP={cm['true_positives']}, TN={cm['true_negatives']}, ")
                f.write(f"FP={cm['false_positives']}, FN={cm['false_negatives']}\n")
            f.write("\n")

            f.write("PHASE 2: SCREENING\n")
            f.write("-"*70 + "\n")
            f.write(f"Iterations: {results['n_iterations']}\n")
            f.write(f"Records screened: {results['n_screened']} ({results['proportion_screened']*100:.1f}%)\n")
            f.write(f"Stopped early: {results['stopped_early']}\n")
            f.write(f"Stopping reason: {results['stopping_reason']}\n")
            f.write(f"Screening Recall: {self.phase2_metrics.get('screening_recall', 0)*100:.1f}%\n")
            if self.phase2_metrics.get('classifier_recall', 0) > 0:
                f.write(f"Classifier Recall (on remaining): {self.phase2_metrics.get('classifier_recall', 0)*100:.1f}%\n")
                f.write(f"Total Recall: {self.phase2_metrics.get('total_recall', 0)*100:.1f}%\n")
                f.write(f"Predicted positive on remaining: {self.phase2_metrics.get('n_predicted_positive_remaining', 0)}\n")
            f.write(f"Total Work (screened + predicted positive): {self.phase2_metrics.get('total_work', results['n_screened'])} ({self.phase2_metrics.get('proportion_total_work', 0)*100:.1f}%)\n")
            f.write(f"Actual WSS: {self.phase2_metrics.get('actual_wss', 0):.3f} ({self.phase2_metrics.get('actual_wss', 0)*100:.1f}%)\n\n")

            # Phase 2 classifier metrics
            if 'classification_metrics' in self.phase2_metrics and self.phase2_metrics['classification_metrics']:
                cm = self.phase2_metrics['classification_metrics']
                f.write("Phase 2 Final Classifier Performance (on remaining unlabeled):\n")
                f.write(f"  Precision: {cm['precision']:.3f}\n")
                f.write(f"  Recall: {cm['recall']:.3f}\n")
                f.write(f"  F1-Score: {cm['f1_score']:.3f}\n")
                f.write(f"  AUC-ROC: {cm['auc_roc']:.3f}\n")
                f.write(f"  Accuracy: {cm['accuracy']:.3f}\n")
                f.write(f"  Confusion Matrix: TP={cm['true_positives']}, TN={cm['true_negatives']}, ")
                f.write(f"FP={cm['false_positives']}, FN={cm['false_negatives']}\n")
            f.write("\n")

            if results.get('stopping_criteria'):
                f.write("Stopping criteria status:\n")
                for criterion, met in results['stopping_criteria'].items():
                    status = "✓" if met else "✗"
                    f.write(f"  {status} {criterion}\n")
                f.write("\n")

            f.write("PERFORMANCE\n")
            f.write("-"*70 + "\n")
            f.write(f"Relevant found: {results['n_relevant_found']}/{results['n_total_relevant']}\n")
            f.write(f"Final recall: {results['final_recall']*100:.1f}%\n")
            f.write(f"ATD: {results['atd']:.3f} ({results['atd']*100:.1f}% of dataset)\n\n")

            f.write("Work Savings Metrics:\n")
            f.write(f"  WSS@95% (traditional): {results['wss_95']:.3f} ({results['wss_95']*100:.1f}%)\n")
            f.write(f"    → Work saved to reach 95% recall\n")
            f.write(f"  Actual WSS (Phase 2): {self.phase2_metrics.get('actual_wss', 0):.3f} ({self.phase2_metrics.get('actual_wss', 0)*100:.1f}%)\n")
            f.write(f"    → Based on total recall (screening + classifier on remaining)\n")
            f.write(f"    → Accounts for predicted positives (TP+FP) that need manual screening\n")
            f.write(f"    → Formula: (1 - total_work/n_total) - (1 - total_recall)\n")
            f.write(f"    → Total work = screened + predicted_positive_on_remaining\n")
            f.write(f"  Documents can skip: {(1-self.phase2_metrics.get('proportion_total_work', 0))*100:.1f}%\n")
            f.write(f"    → Predicted negative by classifier (can safely skip)\n\n")

            f.write("FILES SAVED\n")
            f.write("-"*70 + "\n")
            f.write("phase1/\n")
            f.write("  - classifier.pkl (Phase 1 trained model)\n")
            f.write("  - feature_extractor.pkl (TF-IDF vectorizer)\n")
            f.write("  - predictions.csv (Predictions on unlabeled data)\n")
            f.write("  - queried_documents.csv (Prior knowledge records)\n\n")

            f.write("phase2/\n")
            f.write("  - classifier_final.pkl (Final trained model)\n")
            f.write("  - predictions_final.csv (Final predictions on remaining)\n")
            f.write("  - queried_documents.csv (Phase 2 screened records)\n")
            f.write("  - all_screened_documents.csv (All screened Phase 1+2)\n\n")

            f.write("Root directory:\n")
            f.write("  - config.json (Configuration parameters)\n")
            f.write("  - summary.json (Summary statistics)\n")
            f.write("  - REPORT.txt (This file)\n")
            f.write("  - recall_curve.png (Performance visualization)\n\n")

            f.write("="*70 + "\n")
            f.write("End of Report\n")
            f.write("="*70 + "\n")

        if self.config.verbose:
            print(f"📄 Generated report: {report_path}")

    def plot_recall_curve(self, save_path: Optional[str] = None):
        """Plot recall curve (save to output directory by default)"""
        if save_path is None:
            save_path = str(self.output_dir / "recall_curve.png")

        super().plot_recall_curve(save_path=save_path)


def main():
    """Demonstration with artifact saving"""
    from asreview_safe_simplified import create_sample_dataset

    print("="*70)
    print("ASReview SAFE with Comprehensive Artifact Saving")
    print("="*70)

    # Create dataset
    print("\nGenerating sample dataset...")
    texts, labels, record_ids = create_sample_dataset(
        n_total=2000,
        n_relevant=40,
        random_state=42
    )

    # Configure output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./asreview_output_{timestamp}"

    # Run ASReview with artifact saving
    config = SAFEConfig(
        prior_knowledge_percentage=0.01,
        min_prior_relevant=1,
        max_prior_percentage=0.05,
        min_screened_percentage=0.10,
        consecutive_irrelevant=50,
        expected_relevant_multiplier=2.0,
        random_state=42,
        verbose=True
    )

    asreview = ASReviewWithArtifacts(config=config, output_dir=output_dir)
    asreview.load_data(texts=texts, labels=labels, record_ids=record_ids)

    results = asreview.run_safe_phases_1_and_2()

    # Plot and save
    asreview.plot_recall_curve()

    print("\n" + "="*70)
    print("SCREENING COMPLETE")
    print("="*70)
    print(f"\n📁 All artifacts saved to: {output_dir}")
    print("\nFiles created:")
    print("  phase1/")
    print("    ├── classifier.pkl")
    print("    ├── feature_extractor.pkl")
    print("    ├── predictions.csv")
    print("    └── queried_documents.csv")
    print("  phase2/")
    print("    ├── classifier_final.pkl")
    print("    ├── predictions_final.csv")
    print("    ├── queried_documents.csv")
    print("    └── all_screened_documents.csv")
    print("  ├── config.json")
    print("  ├── summary.json")
    print("  ├── REPORT.txt")
    print("  └── recall_curve.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()