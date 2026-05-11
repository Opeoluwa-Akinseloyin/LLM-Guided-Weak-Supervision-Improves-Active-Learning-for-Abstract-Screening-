"""
ASReview with LLM-Based Pseudo-Labeling

Phase 1: LLM-based pseudo-labeling instead of random prior knowledge
- Load LLM scores from CSV
- Rank documents by scores
- Top T% → pseudo-positive labels
- Bottom B% → pseudo-negative labels
- Pseudo-labeled documents stay in BOTH training and test sets

Phase 2: Active Learning with Pseudo-Label Correction
- When querying a document:
  - If pseudo-labeled → replace pseudo-label with true label in training set
  - If unlabeled → add to training set with true label
  - In both cases → remove from test set (now truly annotated)
- Modified stopping criteria (removed "Screen ≥ 2× expected")
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from asreview_safe_uncertainty import (
    SAFEConfig,
    TFIDFFeatureExtractor,
    DynamicResampler,
    UncertaintyQueryStrategy,
    PerformanceMetrics,
)


class PseudoLabelConfig(SAFEConfig):
    """Extended configuration for pseudo-labeling"""
    # Pseudo-labeling parameters
    top_percentage: float = 0.025  # Top 2.5% as pseudo-positive
    bottom_percentage: float = 0.025  # Bottom 2.5% as pseudo-negative
    llm_score_column: str = 's(d,Q)'  # Column name for LLM scores in CSV

    # Phase 2: Modified stopping criteria (NO "2× expected relevant")
    # Stops when BOTH are met:
    # 1. Screened ≥ min_percentage of dataset
    # 2. N consecutive irrelevant records

    def __init__(self, **kwargs):
        # Extract pseudo-labeling specific parameters
        self.top_percentage = kwargs.pop('top_percentage', 0.025)
        self.bottom_percentage = kwargs.pop('bottom_percentage', 0.025)
        self.llm_score_column = kwargs.pop('llm_score_column', 's(d,Q)')

        # Call parent with remaining kwargs (only SAFEConfig parameters)
        super().__init__(**kwargs)

        # Remove expected_relevant_multiplier since we don't use it
        self.expected_relevant_multiplier = None


class ModifiedStoppingCriteria:
    """
    Modified SAFE Stopping Criteria for Pseudo-Labeling

    Stops when BOTH conditions are met:
    1. Screened ≥ min_percentage of dataset (default 10%)
    2. N consecutive irrelevant records (default 50)

    REMOVED: "Screen ≥ 2× expected relevant" criterion
    """

    def __init__(self, config: PseudoLabelConfig, n_total_records: int):
        self.config = config
        self.n_total_records = n_total_records
        self.consecutive_irrelevant_count = 0

    def update(self, label: int):
        """Update after each labeling decision"""
        if label == 1:
            self.consecutive_irrelevant_count = 0
        else:
            self.consecutive_irrelevant_count += 1

    def check_stopping(self, n_screened: int) -> Tuple[bool, Dict[str, bool]]:
        """Check if both criteria are met"""

        # Criterion 1: Minimum percentage of dataset
        min_screened = int(self.n_total_records * self.config.min_screened_percentage)
        criterion_1 = n_screened >= min_screened

        # Criterion 2: N consecutive irrelevant
        criterion_2 = self.consecutive_irrelevant_count >= self.config.consecutive_irrelevant

        criteria_status = {
            'min_screened_percentage': criterion_1,
            'n_consecutive_irrelevant': criterion_2
        }

        should_stop = all(criteria_status.values())
        return should_stop, criteria_status


class ASReviewLLMPseudoLabeling:
    """
    ASReview with LLM-Based Pseudo-Labeling

    Phase 1: Use LLM scores to create pseudo-labeled training set
    Phase 2: Active learning with progressive pseudo-label correction
    """

    def __init__(self, config: Optional[PseudoLabelConfig] = None,
                 output_dir: str = "./asreview_llm_output"):
        """
        Initialize ASReview with LLM pseudo-labeling.

        Args:
            config: Configuration with pseudo-labeling parameters
            output_dir: Directory to save all artifacts
        """
        self.config = config or PseudoLabelConfig()

        # Initialize components
        self.feature_extractor = TFIDFFeatureExtractor()
        self.classifier = None  # Will use MultinomialNB from sklearn
        self.resampler = DynamicResampler(random_state=self.config.random_state)
        self.query_strategy = UncertaintyQueryStrategy()
        self.metrics = PerformanceMetrics()

        # State
        self.X_features = None
        self.y_labels = None  # True labels (ground truth)
        self.y_pseudo = None  # Pseudo-labels from LLM
        self.texts = None
        self.record_ids = None
        self.llm_scores = None

        # Tracking pseudo vs true labels
        self.pseudo_labeled_mask = None  # Which samples have pseudo-labels
        self.labeled_mask = None  # Which samples have been truly annotated

        self.iteration = 0
        self.rng = np.random.RandomState(self.config.random_state)

        # Output
        self.output_dir = Path(output_dir)
        self.phase1_classifier = None
        self.phase1_predictions = None
        self.phase1_metrics = {}
        self.phase2_metrics = {}

        self._setup_directories()

    def _setup_directories(self):
        """Create organized directory structure"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "phase1").mkdir(exist_ok=True)
        (self.output_dir / "phase2").mkdir(exist_ok=True)

        if self.config.verbose:
            print(f"\n📁 Output directory: {self.output_dir.absolute()}")

    def load_data(self, texts: List[str], labels: Optional[List[int]] = None,
                  record_ids: Optional[List[int]] = None):
        """
        Load dataset.

        Args:
            texts: List of abstracts/documents
            labels: True labels (ground truth) - for evaluation only
            record_ids: Record identifiers
        """
        self.texts = np.array(texts)
        self.y_labels = np.array(labels) if labels is not None else None
        self.record_ids = np.array(record_ids) if record_ids is not None else np.arange(len(texts))

        # Initialize masks
        self.pseudo_labeled_mask = np.zeros(len(texts), dtype=bool)
        self.labeled_mask = np.zeros(len(texts), dtype=bool)
        self.y_pseudo = np.full(len(texts), -1, dtype=int)  # -1 = no pseudo-label

        if self.config.verbose:
            print(f"Loaded {len(texts)} records")
            if labels is not None:
                n_relevant = np.sum(labels)
                print(f"  - {n_relevant} relevant ({n_relevant / len(texts) * 100:.1f}%)")
                print(f"  - {len(texts) - n_relevant} irrelevant ({(len(texts) - n_relevant) / len(texts) * 100:.1f}%)")

    def load_llm_scores(self, scores_csv_path: str):
        """
        Load LLM scores from CSV file.

        Args:
            scores_csv_path: Path to CSV with LLM scores
                           Expected columns: record_id, score_column_name
        """
        if self.config.verbose:
            print(f"\n📥 Loading LLM scores from: {scores_csv_path}")

        scores_df = pd.read_csv(scores_csv_path)

        # Check for required columns
        if self.config.llm_score_column not in scores_df.columns:
            raise ValueError(f"Score column '{self.config.llm_score_column}' not found in CSV. "
                           f"Available columns: {scores_df.columns.tolist()}")

        # Match scores to record_ids
        self.llm_scores = np.zeros(len(self.texts))

        for idx, record_id in enumerate(self.record_ids):
            matching_rows = scores_df[scores_df['PMID'] == record_id]
            if len(matching_rows) > 0:
                self.llm_scores[idx] = matching_rows.iloc[0][self.config.llm_score_column]
            else:
                # If no score found, assign median score
                self.llm_scores[idx] = scores_df[self.config.llm_score_column].median()
                if self.config.verbose and idx < 5:  # Only warn for first few
                    print(f"  ⚠️  No score found for record {record_id}, using median")

        if self.config.verbose:
            print(f"✓ Loaded scores for {len(self.texts)} records")
            print(f"  - Score range: [{self.llm_scores.min():.4f}, {self.llm_scores.max():.4f}]")
            print(f"  - Score mean: {self.llm_scores.mean():.4f}")

    def phase1_llm_pseudo_labeling(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Phase 1: LLM-Based Pseudo-Labeling

        1. Rank documents by LLM scores (descending)
        2. Top T% → pseudo-positive (label=1)
        3. Bottom B% → pseudo-negative (label=0)
        4. Keep pseudo-labeled docs in BOTH training and test sets

        Returns:
            pseudo_indices: Indices of pseudo-labeled documents
            pseudo_labels: Pseudo-labels (0 or 1)
            llm_scores_sorted: Sorted LLM scores for these documents
        """
        if self.llm_scores is None:
            raise ValueError("Must call load_llm_scores() before phase1_llm_pseudo_labeling()")

        if self.config.verbose:
            print(f"\n{'='*70}")
            print("PHASE 1: LLM-Based Pseudo-Labeling")
            print(f"{'='*70}")
            print(f"Top {self.config.top_percentage*100:.1f}% → Pseudo-POSITIVE")
            print(f"Bottom {self.config.bottom_percentage*100:.1f}% → Pseudo-NEGATIVE")

        n_total = len(self.texts)

        # Rank documents by LLM scores (descending)
        ranked_indices = np.argsort(self.llm_scores)[::-1]

        # Select top T% and bottom B%
        n_top = int(n_total * self.config.top_percentage)
        n_bottom = int(n_total * self.config.bottom_percentage)

        top_indices = ranked_indices[:n_top]
        bottom_indices = ranked_indices[-n_bottom:]

        # Create pseudo-labels
        pseudo_indices = np.concatenate([top_indices, bottom_indices])
        pseudo_labels = np.concatenate([
            np.ones(n_top, dtype=int),   # Top → positive
            np.zeros(n_bottom, dtype=int)  # Bottom → negative
        ])

        # Update state
        self.pseudo_labeled_mask[pseudo_indices] = True
        self.y_pseudo[pseudo_indices] = pseudo_labels

        # NOTE: Do NOT set labeled_mask yet - these are pseudo-labels, not true labels
        # They will only be removed from test set when truly annotated in Phase 2

        if self.config.verbose:
            print(f"\n✓ Pseudo-labeling complete:")
            print(f"  - Total pseudo-labeled: {len(pseudo_indices)} ({len(pseudo_indices)/n_total*100:.1f}%)")
            print(f"  - Pseudo-POSITIVE: {n_top} (top {self.config.top_percentage*100:.1f}%)")
            print(f"  - Pseudo-NEGATIVE: {n_bottom} (bottom {self.config.bottom_percentage*100:.1f}%)")

            # Evaluate pseudo-labels against ground truth (if available)
            if self.y_labels is not None:
                true_labels_pseudo = self.y_labels[pseudo_indices]
                accuracy = np.mean(pseudo_labels == true_labels_pseudo)

                # For positives
                pos_mask = pseudo_labels == 1
                if np.sum(pos_mask) > 0:
                    pos_accuracy = np.mean(pseudo_labels[pos_mask] == true_labels_pseudo[pos_mask])
                else:
                    pos_accuracy = 0.0

                # For negatives
                neg_mask = pseudo_labels == 0
                if np.sum(neg_mask) > 0:
                    neg_accuracy = np.mean(pseudo_labels[neg_mask] == true_labels_pseudo[neg_mask])
                else:
                    neg_accuracy = 0.0

                print(f"\n  Pseudo-label Quality (vs ground truth):")
                print(f"    - Overall accuracy: {accuracy*100:.1f}%")
                print(f"    - Pseudo-POSITIVE accuracy: {pos_accuracy*100:.1f}%")
                print(f"    - Pseudo-NEGATIVE accuracy: {neg_accuracy*100:.1f}%")

        return pseudo_indices, pseudo_labels, self.llm_scores[ranked_indices]

    def _save_phase1_artifacts(self, pseudo_indices: np.ndarray, pseudo_labels: np.ndarray):
        """Save Phase 1 pseudo-labeling artifacts"""
        phase1_dir = self.output_dir / "phase1"

        # Extract features
        if self.config.verbose:
            print("\nExtracting TF-IDF features...")
        self.X_features = self.feature_extractor.fit_transform(self.texts)

        # Train initial classifier on pseudo-labels
        from sklearn.naive_bayes import MultinomialNB
        self.classifier = MultinomialNB()

        X_pseudo = self.X_features[pseudo_indices]
        y_pseudo = pseudo_labels

        # Apply resampling
        X_resampled, y_resampled = self.resampler.resample(
            X_pseudo, y_pseudo, len(self.texts)
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

        # 3. Save pseudo-labeled documents
        pseudo_df = pd.DataFrame({
            'record_id': self.record_ids[pseudo_indices],
            'pseudo_label': pseudo_labels,
            'llm_score': self.llm_scores[pseudo_indices],
            'text': self.texts[pseudo_indices]
        })

        if self.y_labels is not None:
            pseudo_df['true_label'] = self.y_labels[pseudo_indices]
            pseudo_df['correct'] = (pseudo_labels == self.y_labels[pseudo_indices])

        pseudo_path = phase1_dir / "pseudo_labeled_documents.csv"
        pseudo_df.to_csv(pseudo_path, index=False)
        if self.config.verbose:
            print(f"💾 Saved pseudo-labeled documents to: {pseudo_path}")

        # 4. Make predictions on ALL documents
        probas = self.classifier.predict_proba(self.X_features)
        if probas.shape[1] == 1:
            probas = np.column_stack([1 - probas[:, 0], probas[:, 0]])

        predictions_df = pd.DataFrame({
            'record_id': self.record_ids,
            'probability_relevant': probas[:, 1],
            'text': self.texts,
            'is_pseudo_labeled': self.pseudo_labeled_mask,
            'pseudo_label': self.y_pseudo
        })

        if self.y_labels is not None:
            predictions_df['true_label'] = self.y_labels
            predictions_df['prediction'] = (probas[:, 1] >= 0.5).astype(int)
            predictions_df['correct'] = (predictions_df['prediction'] == self.y_labels)

        # Sort by probability (descending)
        predictions_df = predictions_df.sort_values('probability_relevant', ascending=False)

        pred_path = phase1_dir / "predictions_all_documents.csv"
        predictions_df.to_csv(pred_path, index=False)
        if self.config.verbose:
            print(f"💾 Saved predictions to: {pred_path}")

        # 5. Calculate Phase 1 metrics
        if self.y_labels is not None:
            from sklearn.metrics import (
                precision_score, recall_score, f1_score,
                accuracy_score, roc_auc_score
            )

            y_pred = (probas[:, 1] >= 0.5).astype(int)

            self.phase1_metrics = {
                'n_pseudo_labeled': len(pseudo_indices),
                'pseudo_positive': np.sum(pseudo_labels == 1),
                'pseudo_negative': np.sum(pseudo_labels == 0),
                'pseudo_label_accuracy': np.mean(pseudo_labels == self.y_labels[pseudo_indices]),

                # Classifier performance on ALL documents
                'classifier_precision': precision_score(self.y_labels, y_pred, zero_division=0),
                'classifier_recall': recall_score(self.y_labels, y_pred, zero_division=0),
                'classifier_f1': f1_score(self.y_labels, y_pred, zero_division=0),
                'classifier_accuracy': accuracy_score(self.y_labels, y_pred),
                'classifier_auc': roc_auc_score(self.y_labels, probas[:, 1])
            }

            if self.config.verbose:
                print(f"\nPhase 1 Classifier Performance:")
                print(f"  Recall: {self.phase1_metrics['classifier_recall']:.3f}")
                print(f"  Precision: {self.phase1_metrics['classifier_precision']:.3f}")
                print(f"  F1: {self.phase1_metrics['classifier_f1']:.3f}")
                print(f"  AUC: {self.phase1_metrics['classifier_auc']:.3f}")

    def phase2_screen_with_pseudo_labels(self, max_iterations: Optional[int] = None) -> Dict:
        """
        Phase 2: Active Learning with Pseudo-Label Correction

        Key differences from standard active learning:
        1. Pseudo-labeled documents start in BOTH training and test sets
        2. When querying:
           - If document has pseudo-label → replace with true label in training
           - If document is unlabeled → add to training with true label
           - In BOTH cases → remove from test set (now truly annotated)
        3. Modified stopping criteria (NO "2× expected relevant")

        Returns:
            Dictionary with screening results
        """
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("PHASE 2: Active Learning with Pseudo-Label Correction")
            print(f"{'='*70}")
            print(f"Model: Naive Bayes + TF-IDF")
            print(f"Query Strategy: Certainty (Maximum)")
            print(f"Balancing: Dynamic Resampling")
            print(f"\nStopping Criteria (BOTH must be met):")
            print(f"  1. Screen ≥ {self.config.min_screened_percentage*100:.0f}% of dataset ({int(len(self.texts) * self.config.min_screened_percentage)} records)")
            print(f"  2. {self.config.consecutive_irrelevant} consecutive irrelevant records")
            print(f"\nNote: 'Screen ≥ 2× expected' criterion REMOVED")

        # Initialize stopping criteria
        stopping = ModifiedStoppingCriteria(self.config, len(self.texts))

        # Setup
        n_total = len(self.texts)
        n_pseudo_labeled = np.sum(self.pseudo_labeled_mask)

        # For Phase 2, unlabeled = documents not truly annotated
        # (includes pseudo-labeled documents that haven't been verified)
        n_unlabeled = n_total - np.sum(self.labeled_mask)

        if max_iterations is None:
            max_iterations = n_unlabeled
        else:
            max_iterations = min(max_iterations, n_unlabeled)

        if self.config.verbose:
            print(f"\nStarting screening...")
            print(f"  - Pseudo-labeled: {n_pseudo_labeled}")
            print(f"  - Truly labeled: {np.sum(self.labeled_mask)}")
            print(f"  - To screen: up to {max_iterations}")
            print()

        # Track metrics
        n_total_relevant = np.sum(self.y_labels == 1)
        n_relevant_found = 0  # Start from 0, count only truly verified relevant
        stopped_early = False
        stopping_reason = None

        # Screening loop
        import time
        start_time = time.time()

        for i in range(max_iterations):
            self.iteration += 1

            # Get current training set: includes pseudo-labeled + truly labeled
            # Training uses pseudo-labels OR true labels (true labels override)
            training_mask = self.pseudo_labeled_mask | self.labeled_mask
            training_indices = np.where(training_mask)[0]

            # Get training labels: use true labels where available, else pseudo-labels
            y_train = np.zeros(len(training_indices), dtype=int)
            for idx_in_train, global_idx in enumerate(training_indices):
                if self.labeled_mask[global_idx]:
                    # True label available
                    y_train[idx_in_train] = self.y_labels[global_idx]
                else:
                    # Use pseudo-label
                    y_train[idx_in_train] = self.y_pseudo[global_idx]

            # Test set: documents not truly annotated
            unlabeled_indices = np.where(~self.labeled_mask)[0]

            if len(unlabeled_indices) == 0:
                stopping_reason = "all_records_labeled"
                break

            # Check stopping criteria (based on truly labeled count)
            n_truly_screened = np.sum(self.labeled_mask)
            should_stop, criteria_status = stopping.check_stopping(n_truly_screened)

            if should_stop:
                stopped_early = True
                stopping_reason = "stopping_criteria_met"

                if self.config.verbose:
                    print(f"\n{'=' * 70}")
                    print("STOPPING CRITERIA MET!")
                    print(f"{'=' * 70}")
                    for criterion, met in criteria_status.items():
                        status = "✓" if met else "✗"
                        print(f"  {status} {criterion}")
                    print(f"{'=' * 70}\n")
                break

            # Train model
            X_train = self.X_features[training_indices]
            X_unlabeled = self.X_features[unlabeled_indices]

            X_train_resampled, y_train_resampled = self.resampler.resample(
                X_train, y_train, n_total
            )
            self.classifier.fit(X_train_resampled, y_train_resampled)

            # Query next record
            next_idx, confidence = self.query_strategy.query(
                self.classifier, X_unlabeled, unlabeled_indices
            )

            # Get TRUE label
            label = int(self.y_labels[next_idx])
            record_id = self.record_ids[next_idx]

            # Update stopping tracking
            stopping.update(label)

            # Update state: mark as truly labeled and remove from test set
            was_pseudo_labeled = self.pseudo_labeled_mask[next_idx]
            self.labeled_mask[next_idx] = True

            if label == 1:
                n_relevant_found += 1

            # Update metrics
            self.metrics.update(
                iteration=self.iteration,
                record_id=record_id,
                label=label,
                confidence=confidence,
                n_relevant_found=n_relevant_found,
                n_total_relevant=n_total_relevant,
                n_screened=np.sum(self.labeled_mask),
                n_total_records=n_total
            )

            # Progress reporting
            if self.config.verbose and (self.iteration % 50 == 0 or self.iteration == 1):
                recall = n_relevant_found / n_total_relevant
                consec = stopping.consecutive_irrelevant_count
                pseudo_indicator = "🔄" if was_pseudo_labeled else "🆕"
                print(f"Iter {self.iteration:4d} {pseudo_indicator}: "
                      f"Recall={recall:.3f} ({n_relevant_found}/{n_total_relevant}), "
                      f"Screened={np.sum(self.labeled_mask)}/{n_total} ({np.sum(self.labeled_mask) / n_total * 100:.1f}%), "
                      f"Consecutive✗={consec}, "
                      f"Label={'✓' if label == 1 else '✗'}")

        elapsed_time = time.time() - start_time

        if not stopped_early and stopping_reason is None:
            stopping_reason = "max_iterations_reached"

        # Calculate metrics
        wss_95 = self.metrics.calculate_wss_at_recall(0.95)
        atd = self.metrics.calculate_atd()

        # Calculate actual WSS and recall
        n_truly_screened = np.sum(self.labeled_mask)
        actual_recall = n_relevant_found / n_total_relevant if n_total_relevant > 0 else 0

        # Calculate Phase 2 final classifier performance on remaining unlabeled
        unlabeled_indices = np.where(~self.labeled_mask)[0]
        classifier_recall_on_remaining = 0.0

        if len(unlabeled_indices) > 0 and self.y_labels is not None:
            from sklearn.metrics import (
                precision_score, recall_score, f1_score,
                accuracy_score, roc_auc_score, confusion_matrix
            )

            X_unlabeled = self.X_features[unlabeled_indices]
            probas = self.classifier.predict_proba(X_unlabeled)
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
            classifier_recall_on_remaining = recall

            # AUC
            try:
                auc = roc_auc_score(y_true, y_pred_proba)
                if np.isnan(auc):
                    auc = 0.0
            except:
                auc = 0.0

            # Confusion matrix
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

            # Store metrics
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

        # Calculate ACTUAL WSS based on classifier performance on remaining unlabeled
        # Total recall = (found during screening) + (classifier can find in remaining)
        n_relevant_in_remaining = np.sum(self.y_labels[unlabeled_indices] == 1) if len(unlabeled_indices) > 0 else 0
        n_relevant_found_in_remaining = classifier_recall_on_remaining * n_relevant_in_remaining
        total_recall = (n_relevant_found + n_relevant_found_in_remaining) / n_total_relevant if n_total_relevant > 0 else 0

        # Calculate predicted positives on remaining (TP + FP that need screening)
        if len(unlabeled_indices) > 0:
            y_pred_remaining = (probas[:, 1] >= 0.5).astype(int)
            n_predicted_positive_remaining = np.sum(y_pred_remaining == 1)
        else:
            n_predicted_positive_remaining = 0

        # Actual WSS = work saved - recall penalty
        # Total work = screened + predicted positive on remaining (TP + FP)
        total_work = n_truly_screened + n_predicted_positive_remaining
        proportion_total_work = total_work / n_total
        actual_wss = (1 - proportion_total_work) - (1 - total_recall)

        if self.config.verbose:
            print(f"\n{'=' * 70}")
            print("SCREENING COMPLETE")
            print(f"{'=' * 70}")
            print(f"Stopping reason: {stopping_reason}")
            print(f"Total iterations: {self.iteration}")
            print(f"\nRecall Breakdown:")
            print(f"  Screening recall: {actual_recall:.1%} ({n_relevant_found}/{n_total_relevant})")
            if len(unlabeled_indices) > 0:
                print(f"  Classifier recall on remaining: {classifier_recall_on_remaining:.1%} ({int(n_relevant_found_in_remaining)}/{n_relevant_in_remaining})")
                print(f"  Total recall (screening + classifier): {total_recall:.1%}")
                print(f"  Predicted positive on remaining: {n_predicted_positive_remaining} (TP+FP need screening)")
            else:
                print(f"  Total recall: {actual_recall:.1%} (all documents screened)")

            print(f"\nWork Metrics:")
            print(f"  Records screened: {n_truly_screened}/{n_total} ({n_truly_screened / n_total * 100:.1f}%)")
            print(f"  Total work (screened + predicted positive): {total_work}/{n_total} ({proportion_total_work * 100:.1f}%)")
            print(f"  Work saved: {(1-proportion_total_work) * 100:.1f}%")
            print(f"  Actual WSS: {actual_wss:.3f} ({actual_wss * 100:.1f}%)")
            print(f"  WSS@95%: {wss_95:.3f} ({wss_95 * 100:.1f}% theoretical)")
            print(f"  ATD: {atd:.3f} ({atd * 100:.1f}% of dataset)")
            print(f"  Time elapsed: {elapsed_time:.2f} seconds")

            # Print Phase 2 classifier metrics if available
            if 'classification_metrics' in self.phase2_metrics and self.phase2_metrics['classification_metrics']:
                cm = self.phase2_metrics['classification_metrics']
                n_remaining = n_total - n_truly_screened
                print(f"\nPhase 2 Classifier Performance ({n_remaining} remaining unlabeled):")
                print(f"  Precision: {cm['precision']:.3f}")
                print(f"  Recall: {cm['recall']:.3f}")
                print(f"  F1-Score: {cm['f1_score']:.3f}")
                print(f"  AUC-ROC: {cm['auc_roc']:.3f}")

            print(f"{'=' * 70}")

        self.phase2_metrics = {
            'screening_recall': actual_recall,
            'classifier_recall_on_remaining': classifier_recall_on_remaining,
            'total_recall': total_recall,
            'n_relevant_found_screening': n_relevant_found,
            'n_relevant_in_remaining': n_relevant_in_remaining,
            'n_relevant_found_by_classifier': float(n_relevant_found_in_remaining),
            'n_predicted_positive_remaining': n_predicted_positive_remaining,
            'total_work': total_work,
            'proportion_total_work': proportion_total_work,
            'actual_wss': actual_wss,
            'n_truly_screened': n_truly_screened
        }

        return {
            'n_iterations': self.iteration,
            'n_relevant_found': n_relevant_found,
            'n_total_relevant': n_total_relevant,
            'screening_recall': actual_recall,
            'classifier_recall_on_remaining': classifier_recall_on_remaining,
            'total_recall': total_recall,
            'final_recall': total_recall,  # For backward compatibility
            'wss_95': wss_95,
            'atd': atd,
            'stopped_early': stopped_early,
            'stopping_reason': stopping_reason,
            'stopping_criteria': criteria_status if stopped_early else None,
            'n_screened': n_truly_screened,
            'proportion_screened': n_truly_screened / n_total,
            'actual_wss': actual_wss,
            'elapsed_time': elapsed_time,
            'metrics': self.metrics
        }

    def _save_phase2_artifacts(self):
        """Save Phase 2 artifacts"""
        phase2_dir = self.output_dir / "phase2"

        # 1. Save final classifier
        classifier_path = phase2_dir / "classifier_final.pkl"
        joblib.dump(self.classifier, classifier_path)
        if self.config.verbose:
            print(f"💾 Saved Phase 2 classifier to: {classifier_path}")

        # 2. Save final predictions on remaining unlabeled
        unlabeled_indices = np.where(~self.labeled_mask)[0]

        if len(unlabeled_indices) > 0:
            X_unlabeled = self.X_features[unlabeled_indices]
            probas = self.classifier.predict_proba(X_unlabeled)
            if probas.shape[1] == 1:
                probas = np.column_stack([1 - probas[:, 0], probas[:, 0]])

            predictions_df = pd.DataFrame({
                'record_id': self.record_ids[unlabeled_indices],
                'probability_relevant': probas[:, 1],
                'text': self.texts[unlabeled_indices]
            })

            if self.y_labels is not None:
                predictions_df['true_label'] = self.y_labels[unlabeled_indices]
                predictions_df['prediction'] = (probas[:, 1] >= 0.5).astype(int)
                predictions_df['correct'] = (predictions_df['prediction'] == predictions_df['true_label'])

            predictions_df = predictions_df.sort_values('probability_relevant', ascending=False)

            pred_path = phase2_dir / "predictions_final.csv"
            predictions_df.to_csv(pred_path, index=False)
            if self.config.verbose:
                print(f"💾 Saved final predictions to: {pred_path}")

                # Print Phase 2 classifier metrics (already calculated)
                if 'classification_metrics' in self.phase2_metrics and self.phase2_metrics['classification_metrics']:
                    cm = self.phase2_metrics['classification_metrics']
                    print(f"\nPhase 2 Final Classifier Performance (on {len(unlabeled_indices)} remaining unlabeled):")
                    print(f"  Precision: {cm['precision']:.3f}")
                    print(f"  Recall: {cm['recall']:.3f}")
                    print(f"  F1-Score: {cm['f1_score']:.3f}")
                    print(f"  AUC-ROC: {cm['auc_roc']:.3f}")
                    print(f"  Accuracy: {cm['accuracy']:.3f}")
                    print(f"  Confusion Matrix: TP={cm['true_positives']}, TN={cm['true_negatives']}, FP={cm['false_positives']}, FN={cm['false_negatives']}")
        else:
            if self.config.verbose:
                print(f"\nℹ️  No remaining unlabeled documents (all were screened)")

        # 3. Save truly queried documents
        labeled_indices = np.where(self.labeled_mask)[0]

        queried_df = pd.DataFrame({
            'record_id': self.record_ids[labeled_indices],
            'true_label': self.y_labels[labeled_indices],
            'was_pseudo_labeled': self.pseudo_labeled_mask[labeled_indices],
            'text': self.texts[labeled_indices]
        })

        # Add pseudo-label if it existed
        pseudo_labels_for_queried = []
        for idx in labeled_indices:
            if self.pseudo_labeled_mask[idx]:
                pseudo_labels_for_queried.append(int(self.y_pseudo[idx]))
            else:
                pseudo_labels_for_queried.append(np.nan)

        queried_df['pseudo_label'] = pseudo_labels_for_queried

        queried_path = phase2_dir / "queried_documents.csv"
        queried_df.to_csv(queried_path, index=False)
        if self.config.verbose:
            print(f"💾 Saved queried documents to: {queried_path}")

    def _save_config(self):
        """Save configuration to JSON"""
        import json

        config_dict = {
            'top_percentage': self.config.top_percentage,
            'bottom_percentage': self.config.bottom_percentage,
            'llm_score_column': self.config.llm_score_column,
            'min_screened_percentage': self.config.min_screened_percentage,
            'consecutive_irrelevant': self.config.consecutive_irrelevant,
            'random_state': self.config.random_state,
            'timestamp': datetime.now().isoformat()
        }

        config_path = self.output_dir / "config.json"
        with open(config_path, 'w', encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        if self.config.verbose:
            print(f"💾 Saved configuration to: {config_path}")

    def _save_summary(self, results: Dict):
        """Save summary statistics"""
        import json

        def convert_to_native(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj

        summary = {
            'dataset': {
                'n_total': int(len(self.texts)),
                'n_relevant': int(np.sum(self.y_labels == 1)),
                'prevalence': float(np.sum(self.y_labels == 1) / len(self.texts))
            },
            'phase1': convert_to_native({
                'n_pseudo_labeled': int(np.sum(self.pseudo_labeled_mask)),
                'n_pseudo_positive': int(np.sum(self.y_pseudo == 1)),
                'n_pseudo_negative': int(np.sum(self.y_pseudo == 0)),
                **self.phase1_metrics
            }),
            'phase2': convert_to_native({
                'n_iterations': results['n_iterations'],
                'n_screened': results['n_screened'],
                'proportion_screened': results['proportion_screened'],
                'n_relevant_found': results['n_relevant_found'],
                'screening_recall': results['screening_recall'],
                'classifier_recall_on_remaining': results['classifier_recall_on_remaining'],
                'total_recall': results['total_recall'],
                'n_predicted_positive_remaining': self.phase2_metrics.get('n_predicted_positive_remaining', 0),
                'total_work': self.phase2_metrics.get('total_work', 0),
                'proportion_total_work': self.phase2_metrics.get('proportion_total_work', 0.0),
                'actual_wss': results['actual_wss'],
                'wss_95': results['wss_95'],
                'atd': results['atd'],
                'stopped_early': results['stopped_early'],
                'stopping_reason': results['stopping_reason'],
                'elapsed_time': results['elapsed_time'],
                'classification_metrics': self.phase2_metrics.get('classification_metrics', {})
            })
        }

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w', encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if self.config.verbose:
            print(f"💾 Saved summary to: {summary_path}")

    def run_llm_pseudo_labeling_phases_1_and_2(self, scores_csv_path: str,
                                               max_iterations: Optional[int] = None) -> Dict:
        """
        Run complete LLM pseudo-labeling workflow.

        Args:
            scores_csv_path: Path to CSV with LLM scores
            max_iterations: Maximum iterations for Phase 2

        Returns:
            Dictionary with results
        """
        # Load LLM scores
        self.load_llm_scores(scores_csv_path)

        # Phase 1: Pseudo-labeling
        pseudo_indices, pseudo_labels, _ = self.phase1_llm_pseudo_labeling()

        # Save Phase 1 artifacts
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("Saving Phase 1 Artifacts")
            print(f"{'='*70}")
        self._save_phase1_artifacts(pseudo_indices, pseudo_labels)

        # Phase 2: Active learning with pseudo-label correction
        results = self.phase2_screen_with_pseudo_labels(max_iterations=max_iterations)

        # Save Phase 2 artifacts
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("Saving Phase 2 Artifacts")
            print(f"{'='*70}")
        self._save_phase2_artifacts()

        # Save config and summary
        self._save_config()
        self._save_summary(results)

        # Generate report
        self._generate_report(results, pseudo_indices, pseudo_labels)

        # Add pseudo-labeling info to results
        results['pseudo_indices'] = pseudo_indices
        results['pseudo_labels'] = pseudo_labels
        results['output_dir'] = str(self.output_dir)

        return results

    def _generate_report(self, results: Dict, pseudo_indices: np.ndarray,
                        pseudo_labels: np.ndarray):
        """Generate human-readable report"""
        report_path = self.output_dir / "REPORT.txt"

        with open(report_path, 'w', encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write("ASReview with LLM-Based Pseudo-Labeling Report\n")
            f.write("="*70 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("DATASET\n")
            f.write("-"*70 + "\n")
            f.write(f"Total records: {len(self.texts)}\n")
            f.write(f"Total relevant: {results['n_total_relevant']}\n")
            f.write(f"Prevalence: {results['n_total_relevant']/len(self.texts)*100:.2f}%\n\n")

            f.write("PHASE 1: LLM-BASED PSEUDO-LABELING\n")
            f.write("-"*70 + "\n")
            f.write(f"Pseudo-labeled: {len(pseudo_indices)} ({len(pseudo_indices)/len(self.texts)*100:.2f}%)\n")
            f.write(f"  - Pseudo-POSITIVE: {np.sum(pseudo_labels==1)}\n")
            f.write(f"  - Pseudo-NEGATIVE: {np.sum(pseudo_labels==0)}\n")

            if 'pseudo_label_accuracy' in self.phase1_metrics:
                f.write(f"Pseudo-label accuracy: {self.phase1_metrics['pseudo_label_accuracy']*100:.1f}%\n")

            if 'classifier_recall' in self.phase1_metrics:
                f.write(f"\nPhase 1 Classifier Performance:\n")
                f.write(f"  Recall: {self.phase1_metrics['classifier_recall']:.3f}\n")
                f.write(f"  Precision: {self.phase1_metrics['classifier_precision']:.3f}\n")
                f.write(f"  F1: {self.phase1_metrics['classifier_f1']:.3f}\n")
                f.write(f"  AUC: {self.phase1_metrics['classifier_auc']:.3f}\n")
            f.write("\n")

            f.write("PHASE 2: ACTIVE LEARNING WITH PSEUDO-LABEL CORRECTION\n")
            f.write("-"*70 + "\n")
            f.write(f"Iterations: {results['n_iterations']}\n")
            f.write(f"Records screened: {results['n_screened']} ({results['proportion_screened']*100:.1f}%)\n")
            f.write(f"Stopped early: {results['stopped_early']}\n")
            f.write(f"Stopping reason: {results['stopping_reason']}\n\n")

            # Phase 2 classifier metrics
            if 'classification_metrics' in self.phase2_metrics and self.phase2_metrics['classification_metrics']:
                cm = self.phase2_metrics['classification_metrics']
                n_remaining = len(self.texts) - results['n_screened']
                f.write(f"Phase 2 Final Classifier Performance (on {n_remaining} remaining unlabeled):\n")
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
            f.write(f"Relevant found during screening: {results['n_relevant_found']}/{results['n_total_relevant']}\n")
            f.write(f"Screening recall: {results['screening_recall']*100:.1f}%\n")

            if 'classification_metrics' in self.phase2_metrics and self.phase2_metrics['classification_metrics']:
                cm = self.phase2_metrics['classification_metrics']
                n_rel_remaining = cm.get('n_relevant_in_remaining', 0)
                if n_rel_remaining > 0:
                    n_found_by_classifier = results['classifier_recall_on_remaining'] * n_rel_remaining
                    f.write(f"Classifier recall on remaining: {results['classifier_recall_on_remaining']*100:.1f}% ({int(n_found_by_classifier)}/{n_rel_remaining})\n")
                    f.write(f"Total recall (screening + classifier): {results['total_recall']*100:.1f}%\n")
                    f.write(f"Predicted positive on remaining: {self.phase2_metrics.get('n_predicted_positive_remaining', 0)} (TP+FP need screening)\n")
            else:
                f.write(f"Total recall: {results['screening_recall']*100:.1f}%\n")

            f.write(f"Total work (screened + predicted positive): {self.phase2_metrics.get('total_work', 0)} ({self.phase2_metrics.get('proportion_total_work', 0)*100:.1f}%)\n")
            f.write(f"ATD: {results['atd']:.3f} ({results['atd']*100:.1f}% of dataset)\n\n")

            f.write("Work Savings Metrics:\n")
            f.write(f"  Actual WSS: {results['actual_wss']:.3f} ({results['actual_wss']*100:.1f}%)\n")
            f.write(f"    → Based on total recall (screening + classifier on remaining)\n")
            f.write(f"    → Accounts for predicted positives (TP+FP) that need manual screening\n")
            f.write(f"    → Formula: (1 - total_work/n_total) - (1 - total_recall)\n")
            f.write(f"    → Total work = screened + predicted_positive_on_remaining\n")
            f.write(f"  WSS@95% (theoretical): {results['wss_95']:.3f} ({results['wss_95']*100:.1f}%)\n")
            f.write(f"    → Assumes 95% recall threshold\n")
            f.write(f"  Documents can skip: {(1-self.phase2_metrics.get('proportion_total_work', 0))*100:.1f}%\n")
            f.write(f"    → Predicted negative by classifier (can safely skip)\n\n")

            f.write("="*70 + "\n")
            f.write("End of Report\n")
            f.write("="*70 + "\n")

        if self.config.verbose:
            print(f"📄 Generated report: {report_path}")

    def plot_recall_curve(self, save_path: Optional[str] = None):
        """Plot recall curve"""
        import matplotlib.pyplot as plt

        if save_path is None:
            save_path = str(self.output_dir / "recall_curve.png")

        proportions, recalls = self.metrics.get_recall_curve()

        if len(proportions) == 0:
            print("No data to plot")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(proportions * 100, recalls * 100, 'b-', linewidth=2, label='Active Learning (LLM Pseudo-Labeling)')
        plt.plot([0, 100], [0, 100], 'r--', linewidth=1, label='Random Screening')

        wss_95 = self.metrics.calculate_wss_at_recall(0.95)
        if wss_95 > 0:
            for i, (p, r) in enumerate(zip(proportions, recalls)):
                if r >= 0.95:
                    plt.axvline(p * 100, color='g', linestyle=':', alpha=0.5)
                    plt.axhline(95, color='g', linestyle=':', alpha=0.5)
                    plt.plot(p * 100, 95, 'go', markersize=10, label=f'95% Recall at {p * 100:.1f}%')
                    break

        plt.xlabel('Proportion of Dataset Screened (%)', fontsize=12)
        plt.ylabel('Recall (%)', fontsize=12)
        plt.title('Active Learning with LLM Pseudo-Labeling: Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 100])
        plt.ylim([0, 105])

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
        plt.close()


def main():
    """Demonstration"""
    print("="*70)
    print("ASReview with LLM-Based Pseudo-Labeling")
    print("="*70)

    # This would be replaced with your actual data and scores
    print("\nNote: This is a template. You need to:")
    print("1. Load your texts, labels, and record_ids")
    print("2. Provide path to your LLM scores CSV")
    print("3. Configure the pseudo-labeling parameters")

    print("\nExample usage:")
    print("""
    # Load your data
    df = pd.read_csv('your_data.csv')
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    record_ids = df['PMID'].tolist()
    
    # Configure
    config = PseudoLabelConfig(
        top_percentage=0.025,  # Top 2.5% as pseudo-positive
        bottom_percentage=0.025,  # Bottom 2.5% as pseudo-negative
        llm_score_column='s(d,Q)',  # Column name in your scores CSV
        min_screened_percentage=0.10,
        consecutive_irrelevant=50,
        random_state=42,
        verbose=True
    )
    
    # Run
    asreview = ASReviewLLMPseudoLabeling(
        config=config,
        output_dir="./output_llm_pseudo"
    )
    asreview.load_data(texts=texts, labels=labels, record_ids=record_ids)
    
    results = asreview.run_llm_pseudo_labeling_phases_1_and_2(
        scores_csv_path='your_llm_scores.csv'
    )
    
    # Plot
    asreview.plot_recall_curve()
    """)


if __name__ == "__main__":
    main()