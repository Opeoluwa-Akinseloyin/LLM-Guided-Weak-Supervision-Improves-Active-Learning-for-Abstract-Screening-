"""
ASReview Complete with Hybrid Query Methods (Random Initialization)

This extends ASReviewComplete with hybrid query strategies:
1. PHASE_SWITCHING: Phase 2a (one method) → Phase 2b (opposite method)
   - Phase 2a stops at 5% screened + 25 consecutive irrelevant
   - Then switches to opposite query method
   - Phase 2b continues until 10% screened + 50 consecutive irrelevant

2. PERCENTAGE_ALTERNATING: Alternates every 1% between certainty and uncertainty

This is the RANDOM INITIALIZATION counterpart of pseudo_asreview_complete_hybrid.py.
Instead of LLM pseudo-labeling, it uses standard random prior knowledge selection.

Usage:
    from asreview_complete_hybrid import ASReviewCompleteHybrid, HybridMode
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
from enum import Enum
from sklearn.naive_bayes import MultinomialNB

from asreview_complete import ASReviewComplete
from asreview_safe_simplified import SAFEConfig


class HybridMode(Enum):
    """Hybrid query method modes"""
    STANDARD = "standard"  # Pure certainty sampling (original)
    PHASE_SWITCHING = "phase_switching"  # Switch at mid-point
    PERCENTAGE_ALTERNATING = "percentage_alternating"  # Switch every 1%


class HybridQueryStrategy:
    """
    Hybrid query strategy that can switch between certainty and uncertainty sampling

    Certainty sampling: Select documents the model is most confident about (highest probability)
    Uncertainty sampling: Select documents the model is most uncertain about (closest to 0.5)
    """

    def __init__(self, initial_mode: str = "certainty"):
        """
        Args:
            initial_mode: "certainty" or "uncertainty"
        """
        self.current_mode = initial_mode

    def set_mode(self, mode: str):
        """Switch between certainty and uncertainty sampling"""
        if mode not in ["certainty", "uncertainty"]:
            raise ValueError(f"Mode must be 'certainty' or 'uncertainty', got {mode}")
        self.current_mode = mode

    def query(self, classifier, X_unlabeled, unlabeled_indices):
        """
        Query next document using current mode

        Args:
            classifier: Trained classifier
            X_unlabeled: Features of unlabeled documents
            unlabeled_indices: Indices of unlabeled documents in original dataset

        Returns:
            next_idx: Index in original dataset
            confidence: Confidence score
        """
        probas = classifier.predict_proba(X_unlabeled)

        if probas.shape[1] == 1:
            probas = np.column_stack([1 - probas[:, 0], probas[:, 0]])

        proba = probas[:, 1]

        if self.current_mode == "certainty":
            query_idx_in_unlabeled = np.argmax(proba)
            confidence = proba[query_idx_in_unlabeled]
        else:  # uncertainty sampling
            uncertainty = np.abs(proba - 0.5)
            query_idx_in_unlabeled = np.argmin(uncertainty)
            confidence = proba[query_idx_in_unlabeled]

        next_idx = unlabeled_indices[query_idx_in_unlabeled]
        return next_idx, confidence


class ModifiedStoppingCriteriaPhase:
    """
    Modified stopping criteria with configurable thresholds for hybrid methods
    """

    def __init__(self, config: SAFEConfig, n_total_records: int,
                 min_screened_pct: float = None, consecutive_irrelevant: int = None):
        self.config = config
        self.n_total_records = n_total_records
        self.consecutive_irrelevant_count = 0

        self.min_screened_percentage = (
            min_screened_pct if min_screened_pct is not None
            else config.min_screened_percentage
        )
        self.consecutive_irrelevant_threshold = (
            consecutive_irrelevant if consecutive_irrelevant is not None
            else config.consecutive_irrelevant
        )

    def update(self, label: int):
        """Update after each labeling decision"""
        if label == 1:
            self.consecutive_irrelevant_count = 0
        else:
            self.consecutive_irrelevant_count += 1

    def check_stopping(self, n_screened: int):
        """Check if stopping criteria are met"""
        min_screened = int(self.n_total_records * self.min_screened_percentage)
        criterion_1 = n_screened >= min_screened
        criterion_2 = self.consecutive_irrelevant_count >= self.consecutive_irrelevant_threshold

        criteria_status = {
            'min_screened_percentage': criterion_1,
            'n_consecutive_irrelevant': criterion_2
        }

        should_stop = all(criteria_status.values())
        return should_stop, criteria_status


class ASReviewCompleteHybrid(ASReviewComplete):
    """
    Extends ASReviewComplete with hybrid query methods using RANDOM initialization.

    Supports three modes:
    1. STANDARD: Pure certainty sampling (original behavior)
    2. PHASE_SWITCHING: Phase 2a (one method) → Phase 2b (opposite method)
       - Phase 2a stops at 5% screened + 25 consecutive irrelevant
       - Then switches to opposite method
       - Phase 2b continues until 10% screened + 50 consecutive irrelevant
    3. PERCENTAGE_ALTERNATING: Alternates every 1% between certainty and uncertainty
    """

    def __init__(self, config: Optional[SAFEConfig] = None,
                 output_dir: str = "./asreview_output",
                 hybrid_mode: HybridMode = HybridMode.STANDARD,
                 initial_query_mode: str = "certainty"):
        """
        Args:
            config: SAFEConfig configuration
            output_dir: Output directory
            hybrid_mode: Which hybrid method to use
            initial_query_mode: Starting query mode ("certainty" or "uncertainty")
        """
        super().__init__(config, output_dir)

        self.hybrid_mode = hybrid_mode
        self.initial_query_mode = initial_query_mode

        # Replace query strategy with hybrid version if not standard
        if hybrid_mode != HybridMode.STANDARD:
            self.query_strategy = HybridQueryStrategy(initial_mode=initial_query_mode)

        # Tracking for hybrid methods
        self.phase2a_ended_at = None
        self.phase2b_started_at = None
        self.query_mode_switches = []
        self.last_switch_iteration = 0

    def run_safe_phases_1_and_2(self, max_iterations: Optional[int] = None):
        """
        Run with hybrid query method.
        Routes to appropriate method based on hybrid_mode.
        """
        if self.hybrid_mode == HybridMode.STANDARD:
            return super().run_safe_phases_1_and_2(max_iterations=max_iterations)

        elif self.hybrid_mode == HybridMode.PHASE_SWITCHING:
            return self._run_phase_switching(max_iterations)

        elif self.hybrid_mode == HybridMode.PERCENTAGE_ALTERNATING:
            return self._run_percentage_alternating(max_iterations)

        else:
            raise ValueError(f"Unknown hybrid mode: {self.hybrid_mode}")

    # ================================================================
    # PHASE SWITCHING
    # ================================================================

    def _run_phase_switching(self, max_iterations: Optional[int] = None):
        """
        Phase-switching hybrid method:
        - Phase 1: Random prior knowledge (standard)
        - Phase 2a: Initial method with relaxed stopping (5% + 25 consecutive)
        - Phase 2b: Switch to opposite method until standard stopping (10% + 50 consecutive)
        - Phase 3: Continue to 100% recall
        """
        if self.config.verbose:
            opposite_mode = 'UNCERTAINTY' if self.initial_query_mode == 'certainty' else 'CERTAINTY'
            print(f"\n{'='*70}")
            print(f"HYBRID MODE: PHASE SWITCHING (Random Initialization)")
            print(f"Phase 1: Random prior knowledge")
            print(f"Phase 2a: {self.initial_query_mode.upper()} sampling (stop at 5% + 25 consecutive)")
            print(f"Phase 2b: {opposite_mode} sampling (stop at 10% + 50 consecutive)")
            print(f"{'='*70}\n")

        # Phase 1: Random prior knowledge (standard ASReview)
        self._run_phase1_random_prior()

        n_total = len(self.texts)
        n_total_relevant = np.sum(self.y_labels == 1)

        # Phase 2a: Initial query method with relaxed stopping criteria
        phase2a_stopping = ModifiedStoppingCriteriaPhase(
            self.config, n_total,
            min_screened_pct=0.05,
            consecutive_irrelevant=25
        )

        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"PHASE 2a: {self.initial_query_mode.upper()} SAMPLING")
            print(f"{'='*70}\n")

        self._run_active_learning_phase(
            phase2a_stopping, n_total, n_total_relevant, phase_name="2a"
        )

        self.phase2a_ended_at = self.iteration
        n_screened_2a = np.sum(self.labeled_mask)
        n_relevant_found_2a = np.sum(self.y_labels[self.labeled_mask] == 1)

        # Record the switch
        new_mode = 'uncertainty' if self.initial_query_mode == 'certainty' else 'certainty'
        self.query_mode_switches.append({
            'iteration': self.iteration,
            'from_mode': self.initial_query_mode,
            'to_mode': new_mode,
            'reason': 'phase2a_complete',
            'screened': int(n_screened_2a),
            'recall': float(n_relevant_found_2a / n_total_relevant)
        })

        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"PHASE 2a COMPLETE - SWITCHING QUERY METHOD")
            print(f"{'='*70}")
            print(f"Phase 2a ended at iteration {self.iteration}")
            print(f"Screened: {n_screened_2a}/{n_total} ({n_screened_2a/n_total*100:.1f}%)")
            print(f"Recall: {n_relevant_found_2a/n_total_relevant*100:.1f}%")
            print(f"\nSwitching from {self.initial_query_mode.upper()} to {new_mode.upper()}")
            print(f"{'='*70}\n")

        # Switch query mode
        self.query_strategy.set_mode(new_mode)
        self.phase2b_started_at = self.iteration + 1

        # Phase 2b: Switched query method with standard stopping criteria
        phase2b_stopping = ModifiedStoppingCriteriaPhase(
            self.config, n_total,
            min_screened_pct=self.config.min_screened_percentage,
            consecutive_irrelevant=self.config.consecutive_irrelevant
        )

        if self.config.verbose:
            print(f"PHASE 2b: {new_mode.upper()} SAMPLING")
            print(f"{'='*70}\n")

        self._run_active_learning_phase(
            phase2b_stopping, n_total, n_total_relevant, phase_name="2b"
        )

        phase2_ended_at = self.iteration

        # Calculate Phase 2 metrics
        results = self._finalize_phase2_results(phase2_ended_at, n_total, n_total_relevant)

        # Save Phase 2 artifacts using parent's method
        # Saves classifier, predictions, screened docs, calculates phase2_metrics
        self._save_phase2_artifacts()

        # Add phase switching info
        results['phase2a_ended_at'] = self.phase2a_ended_at
        results['phase2b_started_at'] = self.phase2b_started_at
        results['query_mode_switches'] = self.query_mode_switches
        results['hybrid_mode'] = 'phase_switching'

        # Phase 3: Continue to 100% recall
        self._run_phase3_to_100_recall(results, n_total, n_total_relevant)

        return results

    # ================================================================
    # PERCENTAGE ALTERNATING
    # ================================================================

    def _run_percentage_alternating(self, max_iterations: Optional[int] = None):
        """
        Percentage-alternating hybrid method:
        Alternates between certainty and uncertainty every 1% of dataset screened
        """
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"HYBRID MODE: PERCENTAGE ALTERNATING (Random Initialization)")
            print(f"Starting with: {self.initial_query_mode.upper()} sampling")
            print(f"Switching every 1% of dataset screened")
            print(f"{'='*70}\n")

        # Phase 1: Random prior knowledge (standard ASReview)
        self._run_phase1_random_prior()

        n_total = len(self.texts)
        n_total_relevant = np.sum(self.y_labels == 1)

        # Phase 2 with alternating query methods
        stopping_criteria = ModifiedStoppingCriteriaPhase(
            self.config, n_total,
            min_screened_pct=self.config.min_screened_percentage,
            consecutive_irrelevant=self.config.consecutive_irrelevant
        )

        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"PHASE 2: PERCENTAGE ALTERNATING")
            print(f"{'='*70}\n")

        self._run_active_learning_with_alternating(
            stopping_criteria, n_total, n_total_relevant
        )

        phase2_ended_at = self.iteration

        # Calculate Phase 2 metrics
        results = self._finalize_phase2_results(phase2_ended_at, n_total, n_total_relevant)

        # Save Phase 2 artifacts using parent's method
        self._save_phase2_artifacts()

        # Add alternating info
        results['query_mode_switches'] = self.query_mode_switches
        results['hybrid_mode'] = 'percentage_alternating'

        # Phase 3: Continue to 100% recall
        self._run_phase3_to_100_recall(results, n_total, n_total_relevant)

        return results

    # ================================================================
    # SHARED HELPER METHODS
    # ================================================================

    def _run_phase1_random_prior(self):
        """Phase 1: Standard random prior knowledge selection + save artifacts (like parent)"""
        if self.config.verbose:
            print("Extracting TF-IDF features...")
        self.X_features = self.feature_extractor.fit_transform(self.texts)

        if self.config.verbose:
            print("Phase 1: Random prior knowledge selection...")

        # Use the parent class Phase 1 method (random sampling)
        prior_indices, prior_labels = self.phase1_adaptive_prior_knowledge()

        # Store prior info
        self._prior_indices = prior_indices
        self._prior_labels = prior_labels

        # Save Phase 1 artifacts using parent's method
        # This trains Phase 1 classifier, saves predictions.csv, queried_documents.csv,
        # calculates Phase 1 classifier metrics and stores in self.phase1_metrics
        self._save_phase1_artifacts(prior_indices, prior_labels)

        if self.config.verbose:
            n_prior = len(prior_indices)
            n_prior_relevant = np.sum(prior_labels == 1)
            print(f"Prior knowledge: {n_prior} documents ({n_prior_relevant} relevant)")

    def _run_active_learning_phase(self, stopping_criteria, n_total, n_total_relevant, phase_name="2"):
        """
        Run active learning with given stopping criteria.
        Uses only truly labeled documents for training (no pseudo-labels).
        """
        n_relevant_found = np.sum(self.y_labels[self.labeled_mask] == 1)

        while True:
            self.iteration += 1

            labeled_indices = np.where(self.labeled_mask)[0]
            unlabeled_indices = np.where(~self.labeled_mask)[0]

            if len(unlabeled_indices) == 0:
                if self.config.verbose:
                    print(f"No more unlabeled documents (Phase {phase_name})")
                break

            # Train on truly labeled documents only
            X_train = self.X_features[labeled_indices]
            y_train = self.y_labels[labeled_indices]
            X_unlabeled = self.X_features[unlabeled_indices]

            X_train_resampled, y_train_resampled = self.resampler.resample(
                X_train, y_train, n_total
            )

            if self.classifier is None:
                self.classifier = MultinomialNB()
            self.classifier.fit(X_train_resampled, y_train_resampled)

            # Query next document using current query strategy
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
                    print(f"\n🎯 MILESTONE: 95% recall at iteration {self.iteration} (Phase {phase_name})")

            # Update stopping criteria
            stopping_criteria.update(label)

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

            # Check stopping
            should_stop, criteria_status = stopping_criteria.check_stopping(n_screened)

            # Progress reporting
            if self.config.verbose and (self.iteration % 50 == 0 or should_stop):
                print(f"[Phase {phase_name}] Iter {self.iteration:4d}: "
                      f"Mode={self.query_strategy.current_mode.upper()[:4]}, "
                      f"Recall={current_recall:.3f}, "
                      f"Screened={n_screened}/{n_total} ({n_screened/n_total*100:.1f}%), "
                      f"Label={'✓' if label == 1 else '✗'}")

                if should_stop:
                    print(f"\nStopping criteria met (Phase {phase_name}):")
                    for criterion, met in criteria_status.items():
                        print(f"  {criterion}: {met}")

            if should_stop:
                break

    def _run_active_learning_with_alternating(self, stopping_criteria, n_total, n_total_relevant):
        """
        Run active learning with percentage-based alternating query methods.
        Switches every 1% of dataset screened.
        Uses only truly labeled documents for training (no pseudo-labels).
        """
        n_relevant_found = np.sum(self.y_labels[self.labeled_mask] == 1)

        one_percent = int(n_total * 0.01)
        if one_percent < 1:
            one_percent = 1
        next_switch_threshold = one_percent

        while True:
            self.iteration += 1

            labeled_indices = np.where(self.labeled_mask)[0]
            unlabeled_indices = np.where(~self.labeled_mask)[0]

            if len(unlabeled_indices) == 0:
                if self.config.verbose:
                    print("No more unlabeled documents")
                break

            # Train on truly labeled documents only
            X_train = self.X_features[labeled_indices]
            y_train = self.y_labels[labeled_indices]
            X_unlabeled = self.X_features[unlabeled_indices]

            X_train_resampled, y_train_resampled = self.resampler.resample(
                X_train, y_train, n_total
            )

            if self.classifier is None:
                self.classifier = MultinomialNB()
            self.classifier.fit(X_train_resampled, y_train_resampled)

            # Query
            next_idx, confidence = self.query_strategy.query(
                self.classifier, X_unlabeled, unlabeled_indices
            )

            # Label
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

            # Check if we should switch query mode (every 1%)
            if n_screened >= next_switch_threshold:
                old_mode = self.query_strategy.current_mode
                new_mode = 'uncertainty' if old_mode == 'certainty' else 'certainty'
                self.query_strategy.set_mode(new_mode)

                self.query_mode_switches.append({
                    'iteration': self.iteration,
                    'from_mode': old_mode,
                    'to_mode': new_mode,
                    'reason': f'screened_{n_screened}_docs',
                    'screened': n_screened,
                    'screened_pct': n_screened / n_total,
                    'recall': current_recall
                })

                if self.config.verbose:
                    print(f"\n>>> SWITCH at {n_screened} docs ({n_screened/n_total*100:.1f}%): "
                          f"{old_mode.upper()} → {new_mode.upper()}\n")

                next_switch_threshold += one_percent

            # Update stopping criteria
            stopping_criteria.update(label)

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

            # Check stopping
            should_stop, criteria_status = stopping_criteria.check_stopping(n_screened)

            # Progress
            if self.config.verbose and (self.iteration % 50 == 0 or should_stop):
                print(f"Iter {self.iteration:4d}: "
                      f"Mode={self.query_strategy.current_mode.upper()[:4]}, "
                      f"Recall={current_recall:.3f}, "
                      f"Screened={n_screened}/{n_total} ({n_screened/n_total*100:.1f}%), "
                      f"Label={'✓' if label == 1 else '✗'}")

                if should_stop:
                    print(f"\nStopping criteria met:")
                    for criterion, met in criteria_status.items():
                        print(f"  {criterion}: {met}")

            if should_stop:
                break

    def _finalize_phase2_results(self, phase2_ended_at, n_total, n_total_relevant):
        """Calculate and return Phase 2 results"""
        n_screened = np.sum(self.labeled_mask)
        n_relevant_found = np.sum(self.y_labels[self.labeled_mask] == 1)

        screening_recall = n_relevant_found / n_total_relevant
        proportion_screened = n_screened / n_total

        # Calculate classifier predictions for remaining documents
        self._fix_phase2_classifier_metrics()

        # Calculate total recall (screening + classifier on remaining)
        unlabeled_indices = np.where(~self.labeled_mask)[0]
        if len(unlabeled_indices) > 0:
            n_relevant_in_remaining = np.sum(self.y_labels[unlabeled_indices] == 1)
            classifier_recall = self.phase2_metrics.get(
                'classification_metrics', {}
            ).get('recall', 0.0)
            n_relevant_found_in_remaining = classifier_recall * n_relevant_in_remaining
            total_recall = (n_relevant_found + n_relevant_found_in_remaining) / n_total_relevant
        else:
            total_recall = screening_recall

        # Calculate WSS
        expected_screened_random = screening_recall
        actual_wss = max(0, expected_screened_random - proportion_screened)

        results = {
            'n_iterations': phase2_ended_at,
            'phase2_ended_at_iteration': phase2_ended_at,
            'n_screened': int(n_screened),
            'n_total_relevant': int(n_total_relevant),
            'n_relevant_found': int(n_relevant_found),
            'screening_recall': float(screening_recall),
            'total_recall': float(total_recall),
            'final_recall': float(screening_recall),
            'proportion_screened': float(proportion_screened),
            'actual_wss': float(actual_wss),
            'stopped_early': True,
            'stopping_reason': 'criteria_met',
            'phase2_metrics': self.phase2_metrics,
            'output_dir': str(self.output_dir)
        }

        return results

    def _fix_phase2_classifier_metrics(self):
        """Calculate classifier metrics on remaining unlabeled documents"""
        unlabeled_indices = np.where(~self.labeled_mask)[0]

        if len(unlabeled_indices) == 0 or self.classifier is None:
            self.phase2_metrics['classification_metrics'] = {}
            return

        X_unlabeled = self.X_features[unlabeled_indices]
        y_true = self.y_labels[unlabeled_indices]

        try:
            y_pred = self.classifier.predict(X_unlabeled)
            y_pred_proba = self.classifier.predict_proba(X_unlabeled)

            if y_pred_proba.shape[1] == 1:
                y_pred_proba = np.column_stack([1 - y_pred_proba[:, 0], y_pred_proba[:, 0]])

            metrics = self._calculate_classification_metrics(y_true, y_pred, y_pred_proba[:, 1])
            self.phase2_metrics['classification_metrics'] = metrics

            # Calculate total work and actual WSS
            n_predicted_positive = int(np.sum(y_pred == 1))
            n_screened = int(np.sum(self.labeled_mask))
            total_work = n_screened + n_predicted_positive
            proportion_total_work = total_work / len(self.texts)

            screening_recall = np.sum(self.y_labels[self.labeled_mask] == 1) / np.sum(self.y_labels == 1)
            classifier_recall_on_remaining = metrics.get('recall', 0.0)
            n_relevant_remaining = np.sum(y_true == 1)
            n_total_relevant = np.sum(self.y_labels == 1)
            total_recall = (np.sum(self.y_labels[self.labeled_mask] == 1) +
                           classifier_recall_on_remaining * n_relevant_remaining) / n_total_relevant

            actual_wss = max(0, total_recall - proportion_total_work - (1 - total_recall))

            self.phase2_metrics['screening_recall'] = float(screening_recall)
            self.phase2_metrics['classifier_recall'] = float(classifier_recall_on_remaining)
            self.phase2_metrics['total_recall'] = float(total_recall)
            self.phase2_metrics['n_predicted_positive_remaining'] = n_predicted_positive
            self.phase2_metrics['total_work'] = total_work
            self.phase2_metrics['proportion_total_work'] = float(proportion_total_work)
            self.phase2_metrics['actual_wss'] = float(actual_wss)

        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not calculate Phase 2 classifier metrics: {e}")
            self.phase2_metrics['classification_metrics'] = {}

    def _generate_complete_report(self, results: Dict):
        """
        Override parent to include Phase 1 classifier metrics in COMPLETE_REPORT.txt
        (matching the pattern from asreview_with_artifacts REPORT.txt)
        """
        from datetime import datetime

        report_path = self.output_dir / "COMPLETE_REPORT.txt"

        with open(report_path, 'w', encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("ASReview COMPLETE SCREENING REPORT (Phases 1-3)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # DATASET
            f.write("DATASET\n" + "-" * 70 + "\n")
            f.write(f"Total records: {len(self.texts)}\n")
            f.write(f"Total relevant: {results['n_total_relevant']}\n")
            f.write(f"Prevalence: {results['n_total_relevant'] / len(self.texts) * 100:.2f}%\n\n")

            # PHASE 1: PRIOR KNOWLEDGE
            f.write("PHASE 1: PRIOR KNOWLEDGE\n" + "-" * 70 + "\n")
            prior_size = len(self._prior_indices) if hasattr(self, '_prior_indices') else 0
            prior_relevant = int(np.sum(self._prior_labels == 1)) if hasattr(self, '_prior_labels') else 0
            f.write(f"Records used: {prior_size} ({prior_size / len(self.texts) * 100:.2f}%)\n")
            f.write(f"Relevant found: {prior_relevant}\n")
            f.write(f"Screening Recall: {self.phase1_metrics.get('screening_recall', 0) * 100:.1f}%\n")
            if self.phase1_metrics.get('classifier_recall', 0) > 0:
                f.write(f"Classifier Recall (on remaining): {self.phase1_metrics.get('classifier_recall', 0) * 100:.1f}%\n")
                f.write(f"Total Recall: {self.phase1_metrics.get('total_recall', 0) * 100:.1f}%\n")
                f.write(f"Predicted positive on remaining: {self.phase1_metrics.get('n_predicted_positive_remaining', 0)}\n")
            f.write(f"Actual WSS: {self.phase1_metrics.get('actual_wss', 0):.3f} ({self.phase1_metrics.get('actual_wss', 0) * 100:.1f}%)\n\n")

            # Phase 1 classifier metrics
            if 'classification_metrics' in self.phase1_metrics and self.phase1_metrics['classification_metrics']:
                cm = self.phase1_metrics['classification_metrics']
                f.write("Phase 1 Classifier Performance (on unlabeled set):\n")
                f.write(f"  Precision: {cm['precision']:.3f}\n")
                f.write(f"  Recall: {cm['recall']:.3f}\n")
                f.write(f"  F1-Score: {cm['f1_score']:.3f}\n")
                f.write(f"  AUC-ROC: {cm['auc_roc']:.3f}\n")
                f.write(f"  Accuracy: {cm['accuracy']:.3f}\n")
                f.write(f"  Confusion: TP={cm['true_positives']}, TN={cm['true_negatives']}, ")
                f.write(f"FP={cm['false_positives']}, FN={cm['false_negatives']}\n")
            f.write("\n")

            # PHASE 2 SUMMARY
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

            # PHASE 3
            f.write("PHASE 3 SUMMARY (continue to 100%)\n" + "-" * 70 + "\n")
            f.write(f"Additional iterations: {results.get('phase3_iterations', 0)}\n")
            f.write(f"Total screened: {results.get('total_screened_at_100', 0)} "
                    f"({results.get('proportion_screened_at_100', 0) * 100:.1f}%)\n\n")

            # KEY MILESTONES
            f.write("KEY MILESTONES\n" + "-" * 70 + "\n")
            f.write(f"Iteration at 95% recall: {results['iteration_at_95_recall']}\n")
            f.write(f"Iteration at 100% recall: {results['iteration_at_100_recall']}\n")
            f.write(f"Iterations (95%->100%): {results['iteration_at_100_recall'] - results['iteration_at_95_recall']}\n\n")

            # WORK SAVINGS
            f.write("WORK SAVINGS\n" + "-" * 70 + "\n")
            f.write(f"At Phase 2 end: {results['proportion_screened'] * 100:.1f}% screened, "
                    f"{results['final_recall'] * 100:.1f}% recall\n")
            f.write(f"  WSS@95%: {results.get('wss_95', 0):.3f}\n")
            f.write(f"  Actual WSS (Phase 2): {self.phase2_metrics.get('actual_wss', 0):.3f}\n")
            f.write(f"  Total work (screened + predicted positive): {self.phase2_metrics.get('total_work', 0)}\n")
            f.write(f"At 95% recall: iteration {results['iteration_at_95_recall']}\n")
            f.write(f"At 100% recall: iteration {results['iteration_at_100_recall']}, "
                    f"{results.get('proportion_screened_at_100', 0) * 100:.1f}% screened\n")
            f.write(f"Work saved vs random: "
                    f"{(1 - results.get('proportion_screened_at_100', 0)) * 100:.1f}%\n\n")

            f.write("=" * 70 + "\n")

        if self.config.verbose:
            print(f"Complete report: {report_path}")

    def _run_phase3_to_100_recall(self, results, n_total, n_total_relevant):
        """Phase 3: Continue to 100% recall (maintains last query mode from Phase 2)"""
        n_relevant_found = np.sum(self.y_labels[self.labeled_mask] == 1)
        current_recall = n_relevant_found / n_total_relevant

        if self.config.verbose:
            print(f"\n{'='*70}")
            print("PHASE 3: CONTINUING TO 100% RECALL")
            print(f"Continuing with: {self.query_strategy.current_mode.upper()} sampling")
            print(f"{'='*70}\n")

        phase3_start_iteration = self.iteration

        while current_recall < 1.0:
            self.iteration += 1

            labeled_indices = np.where(self.labeled_mask)[0]
            unlabeled_indices = np.where(~self.labeled_mask)[0]

            if len(unlabeled_indices) == 0:
                break

            # Train on truly labeled documents only
            X_train = self.X_features[labeled_indices]
            y_train = self.y_labels[labeled_indices]
            X_unlabeled = self.X_features[unlabeled_indices]

            X_train_resampled, y_train_resampled = self.resampler.resample(
                X_train, y_train, n_total
            )

            if self.classifier is None:
                self.classifier = MultinomialNB()
            self.classifier.fit(X_train_resampled, y_train_resampled)

            next_idx, confidence = self.query_strategy.query(
                self.classifier, X_unlabeled, unlabeled_indices
            )

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

            if self.config.verbose and (self.iteration % 50 == 0):
                print(f"[Phase 3] Iter {self.iteration:4d}: Recall={current_recall:.3f}")

        self.iteration_at_100_recall = self.iteration

        # Add Phase 3 info to results
        results['phase3_iterations'] = self.iteration - phase3_start_iteration
        results['iteration_at_95_recall'] = (
            self.iteration_at_95_recall if self.iteration_at_95_recall is not None
            else self.iteration
        )
        results['iteration_at_100_recall'] = self.iteration_at_100_recall
        results['total_screened_at_100'] = int(np.sum(self.labeled_mask))
        results['proportion_screened_at_100'] = float(np.sum(self.labeled_mask) / n_total)

        # Save artifacts
        self._save_phase3_artifacts()
        self._generate_complete_report(results)
        self.plot_recall_curve()

        if self.config.verbose:
            print(f"\n✅ 100% RECALL ACHIEVED at iteration {self.iteration_at_100_recall}")
