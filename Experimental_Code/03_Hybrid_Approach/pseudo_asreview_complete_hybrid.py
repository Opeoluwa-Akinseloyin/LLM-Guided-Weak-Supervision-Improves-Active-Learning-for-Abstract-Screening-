"""
Enhanced Pseudo-ASReview with Hybrid Query Methods

This extends PseudoASReviewComplete with:
1. PHASE_SWITCHING: Phase 2a (one method) → Phase 2b (opposite method)
   - Phase 2a stops at 5% screened + 25 consecutive irrelevant
   - Then switches to opposite query method
   - Phase 2b continues until 10% screened + 50 consecutive irrelevant

2. PERCENTAGE_ALTERNATING: Alternates every 1% between certainty and uncertainty

Usage:
    from pseudo_asreview_complete_hybrid import PseudoASReviewHybrid, HybridMode
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
from enum import Enum
from asreview_llm_pseudo_labeling import PseudoLabelConfig
from pseudo_asreview_complete import PseudoASReviewComplete
from sklearn.naive_bayes import MultinomialNB


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
        # Get predicted probabilities for positive class
        probas = classifier.predict_proba(X_unlabeled)
        
        # Handle case where predict_proba returns only one column
        if probas.shape[1] == 1:
            probas = np.column_stack([1 - probas[:, 0], probas[:, 0]])
        
        proba = probas[:, 1]
        
        if self.current_mode == "certainty":
            # Certainty sampling: Select document with highest confidence
            query_idx_in_unlabeled = np.argmax(proba)
            confidence = proba[query_idx_in_unlabeled]
            
        else:  # uncertainty sampling
            # Uncertainty sampling: Select document closest to 0.5 (most uncertain)
            uncertainty = np.abs(proba - 0.5)
            query_idx_in_unlabeled = np.argmin(uncertainty)
            confidence = proba[query_idx_in_unlabeled]
        
        next_idx = unlabeled_indices[query_idx_in_unlabeled]
        return next_idx, confidence


class ModifiedStoppingCriteriaPhase:
    """
    Modified stopping criteria with configurable thresholds for hybrid methods
    """
    
    def __init__(self, config: PseudoLabelConfig, n_total_records: int, 
                 min_screened_pct: float = None, consecutive_irrelevant: int = None):
        """
        Args:
            config: Base configuration
            n_total_records: Total number of records
            min_screened_pct: Override minimum screened percentage (for phase 2a)
            consecutive_irrelevant: Override consecutive irrelevant threshold (for phase 2a)
        """
        self.config = config
        self.n_total_records = n_total_records
        self.consecutive_irrelevant_count = 0
        
        # Use overrides if provided, otherwise use config values
        self.min_screened_percentage = min_screened_pct if min_screened_pct is not None else config.min_screened_percentage
        self.consecutive_irrelevant_threshold = consecutive_irrelevant if consecutive_irrelevant is not None else config.consecutive_irrelevant
        
    def update(self, label: int):
        """Update after each labeling decision"""
        if label == 1:
            self.consecutive_irrelevant_count = 0
        else:
            self.consecutive_irrelevant_count += 1
            
    def check_stopping(self, n_screened: int):
        """Check if stopping criteria are met"""
        # Criterion 1: Minimum percentage of dataset
        min_screened = int(self.n_total_records * self.min_screened_percentage)
        criterion_1 = n_screened >= min_screened
        
        # Criterion 2: N consecutive irrelevant
        criterion_2 = self.consecutive_irrelevant_count >= self.consecutive_irrelevant_threshold
        
        criteria_status = {
            'min_screened_percentage': criterion_1,
            'n_consecutive_irrelevant': criterion_2
        }
        
        should_stop = all(criteria_status.values())
        return should_stop, criteria_status


class PseudoASReviewHybrid(PseudoASReviewComplete):
    """
    Extends PseudoASReviewComplete with hybrid query methods
    
    Supports three modes:
    1. STANDARD: Pure certainty sampling (original behavior)
    2. PHASE_SWITCHING: Phase 2a (one method) → Phase 2b (opposite method)
       - Phase 2a stops at 5% screened + 25 consecutive irrelevant
       - Then switches to opposite method
       - Phase 2b continues until 10% screened + 50 consecutive irrelevant
    3. PERCENTAGE_ALTERNATING: Alternates every 1% between certainty and uncertainty
    """
    
    def __init__(self, config: Optional[PseudoLabelConfig] = None,
                 output_dir: str = "./asreview_llm_output",
                 hybrid_mode: HybridMode = HybridMode.STANDARD,
                 initial_query_mode: str = "certainty"):
        """
        Args:
            config: Configuration
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
        self.query_mode_switches = []  # Track when and why switches occurred
        self.last_switch_iteration = 0
        
    def run_llm_pseudo_labeling_phases_1_and_2(self, scores_csv_path: str,
                                               max_iterations: Optional[int] = None):
        """
        Run with hybrid query method
        
        Routes to appropriate method based on hybrid_mode
        """
        if self.hybrid_mode == HybridMode.STANDARD:
            # Use parent's standard method
            return super().run_llm_pseudo_labeling_phases_1_and_2(scores_csv_path, max_iterations)
            
        elif self.hybrid_mode == HybridMode.PHASE_SWITCHING:
            return self._run_phase_switching(scores_csv_path, max_iterations)
            
        elif self.hybrid_mode == HybridMode.PERCENTAGE_ALTERNATING:
            return self._run_percentage_alternating(scores_csv_path, max_iterations)
            
        else:
            raise ValueError(f"Unknown hybrid mode: {self.hybrid_mode}")
    
    def _run_phase_switching(self, scores_csv_path: str, max_iterations: Optional[int] = None):
        """
        Phase-switching hybrid method:
        - Phase 2a: Initial method with relaxed stopping (5% + 25 consecutive)
        - Phase 2b: Switch to opposite method until standard stopping (10% + 50 consecutive)
        """
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"HYBRID MODE: PHASE SWITCHING")
            print(f"Phase 2a: {self.initial_query_mode.upper()} sampling (stop at 5% + 25 consecutive)")
            opposite_mode = 'UNCERTAINTY' if self.initial_query_mode == 'certainty' else 'CERTAINTY'
            print(f"Phase 2b: {opposite_mode} sampling (stop at 10% + 50 consecutive)")
            print(f"{'='*70}\n")
        
        # Phase 1: Pseudo-labeling (same as standard)
        self._run_phase1_pseudo_labeling(scores_csv_path)
        
        n_total = len(self.texts)
        n_total_relevant = np.sum(self.y_labels == 1)
        
        # Phase 2a: Initial query method with relaxed stopping criteria
        phase2a_stopping = ModifiedStoppingCriteriaPhase(
            self.config, n_total,
            min_screened_pct=0.05,  # Stop at 5%
            consecutive_irrelevant=25  # Half of standard
        )
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"PHASE 2a: {self.initial_query_mode.upper()} SAMPLING")
            print(f"{'='*70}\n")
        
        # Run Phase 2a
        self._run_active_learning_phase(
            phase2a_stopping,
            n_total,
            n_total_relevant,
            phase_name="2a"
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
            'screened': n_screened_2a,
            'recall': n_relevant_found_2a / n_total_relevant
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
        
        # Run Phase 2b
        self._run_active_learning_phase(
            phase2b_stopping,
            n_total,
            n_total_relevant,
            phase_name="2b"
        )
        
        phase2_ended_at = self.iteration
        
        # Calculate Phase 2 metrics
        results = self._finalize_phase2_results(
            phase2_ended_at,
            n_total,
            n_total_relevant
        )
        
        # Add phase switching info
        results['phase2a_ended_at'] = self.phase2a_ended_at
        results['phase2b_started_at'] = self.phase2b_started_at
        results['query_mode_switches'] = self.query_mode_switches
        results['hybrid_mode'] = 'phase_switching'
        
        # Continue to Phase 3 (100% recall)
        self._run_phase3_to_100_recall(results, n_total, n_total_relevant)
        
        return results
    
    def _run_percentage_alternating(self, scores_csv_path: str, max_iterations: Optional[int] = None):
        """
        Percentage-alternating hybrid method:
        Alternates between certainty and uncertainty every 1% of dataset screened
        """
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"HYBRID MODE: PERCENTAGE ALTERNATING")
            print(f"Starting with: {self.initial_query_mode.upper()} sampling")
            print(f"Switching every 1% of dataset screened")
            print(f"{'='*70}\n")
        
        # Phase 1: Pseudo-labeling (same as standard)
        self._run_phase1_pseudo_labeling(scores_csv_path)
        
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
        
        # Run Phase 2 with percentage-based switching
        self._run_active_learning_with_alternating(
            stopping_criteria,
            n_total,
            n_total_relevant
        )
        
        phase2_ended_at = self.iteration
        
        # Calculate Phase 2 metrics
        results = self._finalize_phase2_results(
            phase2_ended_at,
            n_total,
            n_total_relevant
        )
        
        # Add alternating info
        results['query_mode_switches'] = self.query_mode_switches
        results['hybrid_mode'] = 'percentage_alternating'
        
        # Continue to Phase 3 (100% recall)
        self._run_phase3_to_100_recall(results, n_total, n_total_relevant)
        
        return results
    
    def _run_phase1_pseudo_labeling(self, scores_csv_path: str):
        """Phase 1: LLM-based pseudo-labeling (uses parent class implementation)"""
        # Extract features first
        if self.config.verbose:
            print("Extracting TF-IDF features...")
        self.X_features = self.feature_extractor.fit_transform(self.texts)
        
        # Load LLM scores
        self.load_llm_scores(scores_csv_path)
        
        # Run Phase 1 pseudo-labeling (this is the actual method name in parent)
        pseudo_indices, pseudo_labels, llm_scores_sorted = self.phase1_llm_pseudo_labeling()
        
        # Save Phase 1 artifacts
        self._save_phase1_artifacts(pseudo_indices, pseudo_labels)
    
    def _run_active_learning_phase(self, stopping_criteria, n_total, n_total_relevant, phase_name="2"):
        """
        Run active learning with given stopping criteria
        
        Args:
            stopping_criteria: Stopping criteria object
            n_total: Total number of records
            n_total_relevant: Total number of relevant records
            phase_name: Name for logging (e.g., "2", "2a", "2b")
        """
        n_relevant_found = np.sum(self.y_labels[self.labeled_mask] == 1)
        
        while True:
            self.iteration += 1
            
            # Build training set (truly labeled + pseudo-labeled)
            training_mask = self.labeled_mask | self.pseudo_labeled_mask
            training_indices = np.where(training_mask)[0]
            unlabeled_indices = np.where(~self.labeled_mask)[0]
            
            if len(unlabeled_indices) == 0:
                if self.config.verbose:
                    print(f"No more unlabeled documents (Phase {phase_name})")
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
            
            # Track 95% milestone (can happen in Phase 2)
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
        Run active learning with percentage-based alternating query methods
        Switches every 1% of dataset screened
        """
        n_relevant_found = np.sum(self.y_labels[self.labeled_mask] == 1)

        # Calculate 1% threshold
        one_percent = int(n_total * 0.01)
        if one_percent < 1:
            one_percent = 1  # Ensure at least 1 document per switch
        next_switch_threshold = one_percent

        while True:
            self.iteration += 1

            # Build training set
            training_mask = self.labeled_mask | self.pseudo_labeled_mask
            training_indices = np.where(training_mask)[0]
            unlabeled_indices = np.where(~self.labeled_mask)[0]

            if len(unlabeled_indices) == 0:
                if self.config.verbose:
                    print("No more unlabeled documents")
                break

            # Get labels for training
            y_train = np.zeros(len(training_indices), dtype=int)
            for i, idx in enumerate(training_indices):
                if self.labeled_mask[idx]:
                    y_train[i] = self.y_labels[idx]
                else:
                    y_train[i] = self.y_pseudo[idx]

            X_train = self.X_features[training_indices]
            X_unlabeled = self.X_features[unlabeled_indices]

            # Train
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

            # Track 95% milestone (can happen in Phase 2)
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

        # Get total recall from phase2_metrics (includes screening + classifier on remaining)
        total_recall = self.phase2_metrics.get('classification_metrics', {}).get('recall', screening_recall)

        # Actually, total_recall should be: screening_recall + (classifier_recall * remaining_relevant)
        # Let's calculate it properly
        unlabeled_indices = np.where(~self.labeled_mask)[0]
        if len(unlabeled_indices) > 0:
            n_relevant_in_remaining = np.sum(self.y_labels[unlabeled_indices] == 1)
            classifier_recall = self.phase2_metrics.get('classification_metrics', {}).get('recall', 0.0)
            n_relevant_found_in_remaining = classifier_recall * n_relevant_in_remaining
            total_recall = (n_relevant_found + n_relevant_found_in_remaining) / n_total_relevant
        else:
            total_recall = screening_recall

        # Calculate WSS
        expected_screened_random = screening_recall
        actual_wss = max(0, expected_screened_random - proportion_screened)

        # Get pseudo-labeling info
        pseudo_indices = np.where(self.pseudo_labeled_mask)[0]
        pseudo_labels = self.y_pseudo[pseudo_indices]

        results = {
            'n_iterations': phase2_ended_at,
            'phase2_ended_at_iteration': phase2_ended_at,
            'n_screened': int(n_screened),
            'n_total_relevant': int(n_total_relevant),
            'n_relevant_found': int(n_relevant_found),
            'screening_recall': float(screening_recall),
            'total_recall': float(total_recall),
            'proportion_screened': float(proportion_screened),
            'actual_wss': float(actual_wss),
            'stopped_early': True,
            'stopping_reason': 'criteria_met',
            'pseudo_indices': pseudo_indices.tolist(),
            'pseudo_labels': pseudo_labels.tolist(),
            'phase2_metrics': self.phase2_metrics,
            'output_dir': str(self.output_dir)
        }

        return results

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

        # Continue with the LAST query mode used in Phase 2
        while current_recall < 1.0:
            self.iteration += 1

            training_mask = self.labeled_mask | self.pseudo_labeled_mask
            training_indices = np.where(training_mask)[0]
            unlabeled_indices = np.where(~self.labeled_mask)[0]

            if len(unlabeled_indices) == 0:
                break

            y_train = np.zeros(len(training_indices), dtype=int)
            for i, idx in enumerate(training_indices):
                if self.labeled_mask[idx]:
                    y_train[i] = self.y_labels[idx]
                else:
                    y_train[i] = self.y_pseudo[idx]

            X_train = self.X_features[training_indices]
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
        results['iteration_at_95_recall'] = self.iteration_at_95_recall if self.iteration_at_95_recall is not None else self.iteration
        results['iteration_at_100_recall'] = self.iteration_at_100_recall
        results['total_screened_at_100'] = int(np.sum(self.labeled_mask))
        results['proportion_screened_at_100'] = float(np.sum(self.labeled_mask) / n_total)

        # Save artifacts using parent class methods
        self._save_phase3_artifacts()
        self._save_phase2_artifacts()  # This is a parent class method
        self._generate_complete_report(results)
        self.plot_recall_curve()

        if self.config.verbose:
            print(f"\n✅ 100% RECALL ACHIEVED at iteration {self.iteration_at_100_recall}")