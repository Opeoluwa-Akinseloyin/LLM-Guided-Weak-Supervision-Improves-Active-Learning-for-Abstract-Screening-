"""
ASReview SAFE Implementation - Simplified Version

Phase 1: Continue random sampling until ≥1 relevant is found
Phase 2: Use 3-fold stopping criteria (no key papers requirement)

Stopping Criteria (ALL must be met):
1. Screened ≥ 2× expected relevant records
2. Screened ≥ 10% of dataset
3. N consecutive irrelevant records (default N=50)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time


@dataclass
class SAFEConfig:
    """Configuration for simplified SAFE procedure"""
    # Phase 1: Prior knowledge
    prior_knowledge_percentage: float = 0.01  # Start with 1%
    min_prior_relevant: int = 1  # Must have at least 1 relevant
    max_prior_percentage: float = 1  # Maximum 5% for prior knowledge

    # Phase 2: Stopping criteria (3-fold)
    min_screened_percentage: float = 0.10  # Minimum 10% of dataset
    consecutive_irrelevant: int = 50  # N consecutive irrelevant
    expected_relevant_multiplier: float = 2.0  # Screen ≥ 2× expected

    # Model settings
    feature_extractor: str = "tfidf"
    classifier: str = "naive_bayes"
    query_strategy: str = "certainty"
    balancing_strategy: str = "dynamic_resampling"

    # General
    random_state: int = 42
    verbose: bool = True


class TFIDFFeatureExtractor:
    """TF-IDF Feature Extraction"""

    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            min_df=1
        )
        self.is_fitted = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        feature_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return feature_matrix.toarray()

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        return self.vectorizer.transform(texts).toarray()


class DynamicResampler:
    """Dynamic Resampling for class imbalance"""

    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)

    def resample(self, X_train: np.ndarray, y_train: np.ndarray,
                 n_total_records: int) -> Tuple[np.ndarray, np.ndarray]:
        relevant_mask = y_train == 1
        irrelevant_mask = y_train == 0

        X_relevant = X_train[relevant_mask]
        X_irrelevant = X_train[irrelevant_mask]

        n_relevant = len(X_relevant)
        n_irrelevant = len(X_irrelevant)
        n_training = len(y_train)

        if n_relevant == 0 or n_irrelevant == 0:
            return X_train, y_train

        relevant_ratio = n_relevant / n_training
        target_relevant_ratio = min(0.5, max(0.1, relevant_ratio * 3))

        target_n_relevant = int(n_training * target_relevant_ratio)
        target_n_irrelevant = n_training - target_n_relevant

        # Oversample relevant
        if n_relevant < target_n_relevant:
            relevant_indices = self.rng.choice(n_relevant, size=target_n_relevant, replace=True)
            X_relevant_resampled = X_relevant[relevant_indices]
        else:
            X_relevant_resampled = X_relevant[:target_n_relevant]

        # Undersample irrelevant
        if n_irrelevant > target_n_irrelevant:
            irrelevant_indices = self.rng.choice(n_irrelevant, size=target_n_irrelevant, replace=False)
            X_irrelevant_resampled = X_irrelevant[irrelevant_indices]
        else:
            X_irrelevant_resampled = X_irrelevant

        X_resampled = np.vstack([X_relevant_resampled, X_irrelevant_resampled])
        y_resampled = np.hstack([
            np.ones(len(X_relevant_resampled)),
            np.zeros(len(X_irrelevant_resampled))
        ])

        shuffle_idx = self.rng.permutation(len(y_resampled))
        return X_resampled[shuffle_idx], y_resampled[shuffle_idx]


class CertaintyQueryStrategy:
    """Certainty-based query strategy"""

    def query(self, model, X_unlabeled: np.ndarray,
              unlabeled_indices: np.ndarray) -> Tuple[int, float]:
        probas = model.predict_proba(X_unlabeled)

        # Handle case where predict_proba returns only one column
        if probas.shape[1] == 1:
            # If only one column, assume it's the probability of the positive class
            probas = np.column_stack([1 - probas[:, 0], probas[:, 0]])

        relevant_probas = probas[:, 1]
        max_idx = np.argmax(relevant_probas)
        return unlabeled_indices[max_idx], relevant_probas[max_idx]


class PerformanceMetrics:
    """Track performance metrics"""

    def __init__(self):
        self.iteration_data = []
        self.td_values = []

    def update(self, iteration: int, record_id: int, label: int, confidence: float,
               n_relevant_found: int, n_total_relevant: int, n_screened: int,
               n_total_records: int):
        recall = n_relevant_found / n_total_relevant if n_total_relevant > 0 else 0
        proportion_screened = n_screened / n_total_records

        self.iteration_data.append({
            'iteration': iteration,
            'record_id': record_id,
            'label': label,
            'confidence': confidence,
            'n_relevant_found': n_relevant_found,
            'recall': recall,
            'proportion_screened': proportion_screened,
            'n_screened': n_screened
        })

        if label == 1:
            self.td_values.append(proportion_screened)

    def calculate_wss_at_recall(self, target_recall: float = 0.95) -> float:
        for data in self.iteration_data:
            if data['recall'] >= target_recall:
                proportion_screened = data['proportion_screened']
                return (1 - proportion_screened) - (1 - target_recall)
        return 0.0

    def calculate_atd(self) -> float:
        return np.mean(self.td_values) if self.td_values else 1.0

    def get_recall_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.iteration_data:
            return np.array([]), np.array([])
        proportions = [d['proportion_screened'] for d in self.iteration_data]
        recalls = [d['recall'] for d in self.iteration_data]
        return np.array(proportions), np.array(recalls)


class SimplifiedStoppingCriteria:
    """
    3-Fold SAFE Stopping Criteria (NO key papers requirement)

    Stops when ALL three conditions are met:
    1. Screened ≥ 2× expected relevant records
    2. Screened ≥ 10% of dataset
    3. N consecutive irrelevant records
    """

    def __init__(self, config: SAFEConfig, n_total_records: int):
        self.config = config
        self.n_total_records = n_total_records
        self.consecutive_irrelevant_count = 0
        self.expected_relevant = None

    def set_expected_relevant(self, n_prior: int, n_relevant_in_prior: int):
        """Calculate expected relevant based on Phase 1 prevalence"""
        observed_prevalence = n_relevant_in_prior / n_prior
        self.expected_relevant = observed_prevalence * self.n_total_records

        if self.config.verbose:
            print(f"\nExpected Relevant Calculation:")
            print(f"  - Prior knowledge size: {n_prior}")
            print(f"  - Relevant in prior: {n_relevant_in_prior}")
            print(f"  - Observed prevalence: {observed_prevalence:.1%}")
            print(f"  - Estimated total relevant: {self.expected_relevant:.1f}")
            print(
                f"  - Target screening: ~{self.expected_relevant * self.config.expected_relevant_multiplier:.0f} records (2× expected)")

    def update(self, label: int):
        """Update after each labeling decision"""
        if label == 1:
            self.consecutive_irrelevant_count = 0
        else:
            self.consecutive_irrelevant_count += 1

    def check_stopping(self, n_screened: int) -> Tuple[bool, Dict[str, bool]]:
        """Check if all 3 criteria are met"""

        # Criterion 1: Screened ≥ 2× expected relevant
        if self.expected_relevant is not None:
            criterion_1 = n_screened >= (self.expected_relevant * self.config.expected_relevant_multiplier)
        else:
            criterion_1 = False

        # Criterion 2: Minimum 10% of dataset
        min_screened = int(self.n_total_records * self.config.min_screened_percentage)
        criterion_2 = n_screened >= min_screened

        # Criterion 3: N consecutive irrelevant
        criterion_3 = self.consecutive_irrelevant_count >= self.config.consecutive_irrelevant

        criteria_status = {
            'screened_2x_expected': criterion_1,
            'min_10_percent_screened': criterion_2,
            'n_consecutive_irrelevant': criterion_3
        }

        should_stop = all(criteria_status.values())
        return should_stop, criteria_status


class ASReviewSAFE:
    """Simplified ASReview SAFE Implementation"""

    def __init__(self, config: Optional[SAFEConfig] = None):
        self.config = config or SAFEConfig()

        # Initialize components
        self.feature_extractor = TFIDFFeatureExtractor()
        self.classifier = MultinomialNB()
        self.resampler = DynamicResampler(random_state=self.config.random_state)
        self.query_strategy = CertaintyQueryStrategy()
        self.metrics = PerformanceMetrics()

        # State
        self.X_features = None
        self.y_labels = None
        self.texts = None
        self.record_ids = None
        self.labeled_mask = None
        self.iteration = 0
        self.rng = np.random.RandomState(self.config.random_state)

    def load_data(self, texts: List[str], labels: Optional[List[int]] = None,
                  record_ids: Optional[List[int]] = None):
        """Load dataset"""
        self.texts = np.array(texts)
        self.y_labels = np.array(labels) if labels is not None else None
        self.record_ids = np.array(record_ids) if record_ids is not None else np.arange(len(texts))
        self.labeled_mask = np.zeros(len(texts), dtype=bool)

        if self.config.verbose:
            print(f"Loaded {len(texts)} records")
            if labels is not None:
                n_relevant = np.sum(labels)
                print(f"  - {n_relevant} relevant ({n_relevant / len(texts) * 100:.1f}%)")
                print(f"  - {len(texts) - n_relevant} irrelevant ({(len(texts) - n_relevant) / len(texts) * 100:.1f}%)")

    def phase1_adaptive_prior_knowledge(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Phase 1: Adaptive prior knowledge selection.

        Start with 1% of dataset, keep adding random samples until
        at least min_prior_relevant records are found (up to 5% max).
        """
        if self.y_labels is None:
            raise ValueError("Ground truth labels required for simulation")

        n_total = len(self.texts)
        initial_n_prior = max(1, int(n_total * self.config.prior_knowledge_percentage))
        max_n_prior = max(1, int(n_total * self.config.max_prior_percentage))

        if self.config.verbose:
            print(f"\n=== SAFE Phase 1: Adaptive Prior Knowledge Selection ===")
            print(
                f"Starting with {initial_n_prior} records ({self.config.prior_knowledge_percentage * 100:.1f}% of dataset)")
            print(f"Requirement: At least {self.config.min_prior_relevant} relevant record(s)")
            print(f"Maximum allowed: {max_n_prior} records ({self.config.max_prior_percentage * 100:.1f}% of dataset)")

        available_indices = np.arange(n_total)
        selected_indices = []
        n_relevant_found = 0

        # Keep selecting until we have enough relevant records
        while n_relevant_found < self.config.min_prior_relevant and len(selected_indices) < max_n_prior:
            # How many more do we need?
            if len(selected_indices) == 0:
                # First batch - select initial_n_prior
                n_to_select = initial_n_prior
            else:
                # Additional batches - add 1% more at a time
                n_to_select = max(1, int(n_total * 0.01))

            # Don't exceed maximum
            n_to_select = min(n_to_select, max_n_prior - len(selected_indices))

            # Select from remaining available indices
            remaining_indices = np.setdiff1d(available_indices, selected_indices)
            if len(remaining_indices) == 0:
                break

            n_to_select = min(n_to_select, len(remaining_indices))
            new_samples = self.rng.choice(remaining_indices, size=n_to_select, replace=False)
            selected_indices.extend(new_samples)

            # Count relevant in current selection
            n_relevant_found = np.sum(self.y_labels[selected_indices] == 1)

            if self.config.verbose and len(selected_indices) > initial_n_prior:
                print(f"  → Sampling more... now have {len(selected_indices)} records with {n_relevant_found} relevant")

        # Convert to array
        prior_indices = np.array(selected_indices)
        prior_labels = self.y_labels[prior_indices]

        # Mark as labeled
        self.labeled_mask[prior_indices] = True

        # CRITICAL: If we still don't have enough relevant, use stratified sampling
        if n_relevant_found < self.config.min_prior_relevant:
            if self.config.verbose:
                print(
                    f"\n⚠ Random sampling reached maximum ({self.config.max_prior_percentage * 100:.0f}%) with only {n_relevant_found} relevant")
                print(f"  → Switching to STRATIFIED sampling to guarantee {self.config.min_prior_relevant} relevant")

            # Get all relevant and irrelevant indices
            all_relevant_indices = np.where(self.y_labels == 1)[0]
            all_irrelevant_indices = np.where(self.y_labels == 0)[0]

            if len(all_relevant_indices) == 0:
                raise ValueError("ERROR: Dataset contains NO relevant records! Cannot initialize Phase 1.")

            # Select required number of relevant (use what we need)
            n_relevant_needed = self.config.min_prior_relevant - n_relevant_found
            n_relevant_needed = min(n_relevant_needed, len(all_relevant_indices))

            # Exclude already selected indices
            available_relevant = np.setdiff1d(all_relevant_indices, prior_indices)
            if len(available_relevant) < n_relevant_needed:
                # Use all available relevant
                selected_relevant = available_relevant
            else:
                selected_relevant = self.rng.choice(available_relevant, size=n_relevant_needed, replace=False)

            # Add these to prior knowledge
            prior_indices = np.concatenate([prior_indices, selected_relevant])
            prior_labels = self.y_labels[prior_indices]
            self.labeled_mask[prior_indices] = True
            n_relevant_found = np.sum(prior_labels == 1)

            if self.config.verbose:
                print(f"  ✓ Added {len(selected_relevant)} relevant record(s) via stratified sampling")

        if self.config.verbose:
            print(f"\n✓ Prior knowledge finalized:")
            print(f"  - Total: {len(prior_indices)} records ({len(prior_indices) / n_total * 100:.2f}%)")
            print(f"  - Relevant: {np.sum(prior_labels == 1)}")
            print(f"  - Irrelevant: {np.sum(prior_labels == 0)}")

        return prior_indices, prior_labels

    def phase2_screen_with_stopping(self, max_iterations: Optional[int] = None) -> Dict:
        """
        Phase 2: Screen with 3-fold stopping criteria.

        Stops when ALL three conditions are met:
        1. Screened ≥ 2× expected relevant
        2. Screened ≥ 10% of dataset
        3. N consecutive irrelevant
        """
        if self.config.verbose:
            print(f"\n=== SAFE Phase 2: Screening with 3-Fold Stopping ===")
            print(f"Model: Naive Bayes + TF-IDF")
            print(f"Query Strategy: Certainty (Maximum)")
            print(f"Balancing: Dynamic Resampling")
            print(f"\nStopping Criteria (ALL must be met):")
            print(f"  1. Screen ≥ 2× expected relevant records")
            print(f"  2. Screen ≥ 10% of dataset ({int(len(self.texts) * 0.1)} records)")
            print(f"  3. {self.config.consecutive_irrelevant} consecutive irrelevant records")

        # Extract features
        if self.config.verbose:
            print("\nExtracting TF-IDF features...")
        self.X_features = self.feature_extractor.fit_transform(self.texts)

        # Initialize stopping criteria
        stopping = SimplifiedStoppingCriteria(self.config, len(self.texts))

        # Set expected relevant from prior knowledge
        n_labeled = np.sum(self.labeled_mask)
        n_relevant_in_prior = np.sum(self.y_labels[self.labeled_mask] == 1)
        stopping.set_expected_relevant(n_labeled, n_relevant_in_prior)

        # Setup
        n_total = len(self.texts)
        n_unlabeled = n_total - n_labeled
        if max_iterations is None:
            max_iterations = n_unlabeled
        else:
            max_iterations = min(max_iterations, n_unlabeled)

        if self.config.verbose:
            print(f"\nStarting screening...")
            print(f"  - Initial labeled: {n_labeled}")
            print(f"  - To screen: up to {max_iterations}")
            print()

        # Track metrics
        n_total_relevant = np.sum(self.y_labels == 1)
        n_relevant_found = n_relevant_in_prior
        stopped_early = False
        stopping_reason = None

        # Screening loop
        start_time = time.time()

        for i in range(max_iterations):
            self.iteration += 1

            # Get current state
            labeled_indices = np.where(self.labeled_mask)[0]
            unlabeled_indices = np.where(~self.labeled_mask)[0]

            if len(unlabeled_indices) == 0:
                stopping_reason = "all_records_labeled"
                break

            # Check stopping criteria
            n_screened = len(labeled_indices)
            should_stop, criteria_status = stopping.check_stopping(n_screened)

            if should_stop:
                stopped_early = True
                stopping_reason = "safe_criteria_met"

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
            X_train = self.X_features[labeled_indices]
            y_train = self.y_labels[labeled_indices]
            X_unlabeled = self.X_features[unlabeled_indices]

            X_train_resampled, y_train_resampled = self.resampler.resample(X_train, y_train, n_total)
            self.classifier.fit(X_train_resampled, y_train_resampled)

            # Query next record
            next_idx, confidence = self.query_strategy.query(self.classifier, X_unlabeled, unlabeled_indices)

            # Get label
            label = int(self.y_labels[next_idx])
            record_id = self.record_ids[next_idx]

            # Update stopping tracking
            stopping.update(label)

            # Update state
            self.labeled_mask[next_idx] = True
            if label == 1:
                n_relevant_found += 1

            n_screened = np.sum(self.labeled_mask)

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

            # Progress reporting
            if self.config.verbose and (self.iteration % 50 == 0 or self.iteration == 1):
                recall = n_relevant_found / n_total_relevant
                consec = stopping.consecutive_irrelevant_count
                print(f"Iter {self.iteration:4d}: "
                      f"Recall={recall:.3f} ({n_relevant_found}/{n_total_relevant}), "
                      f"Screened={n_screened}/{n_total} ({n_screened / n_total * 100:.1f}%), "
                      f"Consecutive✗={consec}, "
                      f"Label={'✓' if label == 1 else '✗'}")

        elapsed_time = time.time() - start_time

        if not stopped_early and stopping_reason is None:
            stopping_reason = "max_iterations_reached"

        # Calculate metrics
        wss_95 = self.metrics.calculate_wss_at_recall(0.95)
        atd = self.metrics.calculate_atd()

        if self.config.verbose:
            print(f"\n{'=' * 70}")
            print("SCREENING COMPLETE")
            print(f"{'=' * 70}")
            print(f"Stopping reason: {stopping_reason}")
            print(f"Total iterations: {self.iteration}")
            print(f"Final recall: {n_relevant_found / n_total_relevant:.1%} ({n_relevant_found}/{n_total_relevant})")
            print(
                f"Records screened: {np.sum(self.labeled_mask)}/{n_total} ({np.sum(self.labeled_mask) / n_total * 100:.1f}%)")
            print(f"WSS@95%: {wss_95:.3f} ({wss_95 * 100:.1f}% work saved)")
            print(f"ATD: {atd:.3f} ({atd * 100:.1f}% of dataset)")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
            print(f"{'=' * 70}")

        return {
            'n_iterations': self.iteration,
            'n_relevant_found': n_relevant_found,
            'n_total_relevant': n_total_relevant,
            'final_recall': n_relevant_found / n_total_relevant,
            'wss_95': wss_95,
            'atd': atd,
            'stopped_early': stopped_early,
            'stopping_reason': stopping_reason,
            'stopping_criteria': criteria_status if stopped_early else None,
            'n_screened': np.sum(self.labeled_mask),
            'proportion_screened': np.sum(self.labeled_mask) / n_total,
            'elapsed_time': elapsed_time,
            'metrics': self.metrics
        }

    def run_safe_phases_1_and_2(self, max_iterations: Optional[int] = None) -> Dict:
        """Run complete SAFE Phase 1 and 2"""
        prior_indices, prior_labels = self.phase1_adaptive_prior_knowledge()
        results = self.phase2_screen_with_stopping(max_iterations=max_iterations)
        results['prior_indices'] = prior_indices
        results['prior_labels'] = prior_labels
        return results

    def plot_recall_curve(self, save_path: Optional[str] = None):
        """Plot recall vs proportion screened"""
        proportions, recalls = self.metrics.get_recall_curve()

        if len(proportions) == 0:
            print("No data to plot")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(proportions * 100, recalls * 100, 'b-', linewidth=2, label='Active Learning')
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
        plt.title('SAFE Active Learning: Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 100])
        plt.ylim([0, 105])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        plt.close()


def create_sample_dataset(n_total: int = 1000, n_relevant: int = 50,
                          random_state: int = 42) -> Tuple[List[str], List[int], List[int]]:
    """Create sample dataset for testing"""
    rng = np.random.RandomState(random_state)

    relevant_words = ['machine learning', 'systematic review', 'meta-analysis',
                      'evidence synthesis', 'screening', 'active learning']
    irrelevant_words = ['unrelated', 'different topic', 'other domain']

    texts = []
    labels = []

    for i in range(n_relevant):
        words = rng.choice(relevant_words, size=rng.randint(20, 50), replace=True)
        texts.append(' '.join(words))
        labels.append(1)

    for i in range(n_total - n_relevant):
        words = rng.choice(irrelevant_words, size=rng.randint(20, 50), replace=True)
        texts.append(' '.join(words))
        labels.append(0)

    indices = rng.permutation(n_total)
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    record_ids = list(range(n_total))

    return texts, labels, record_ids


def main():
    """Demonstration"""
    print("=" * 70)
    print("ASReview SAFE - Simplified Implementation")
    print("3-Fold Stopping Criteria (No Key Papers Requirement)")
    print("=" * 70)

    # Create dataset
    print("\nGenerating sample dataset...")
    texts, labels, record_ids = create_sample_dataset(
        n_total=2000,
        n_relevant=40,  # 2% prevalence
        random_state=42
    )

    # Run ASReview
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

    asreview = ASReviewSAFE(config=config)
    asreview.load_data(texts=texts, labels=labels, record_ids=record_ids)
    results = asreview.run_safe_phases_1_and_2()

    # Plot
    asreview.plot_recall_curve(save_path='/home/claude/recall_curve_simplified.png')

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Stopped early: {results['stopped_early']}")
    print(f"Stopping reason: {results['stopping_reason']}")
    print(f"Work saved: {(1 - results['proportion_screened']) * 100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()