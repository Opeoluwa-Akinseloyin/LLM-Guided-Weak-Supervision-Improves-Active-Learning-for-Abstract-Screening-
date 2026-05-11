"""
ASReview SAFE Implementation - WITH UNCERTAINTY SAMPLING

Phase 1: Continue random sampling until ≥1 relevant is found
Phase 2: Use 3-fold stopping criteria (no key papers requirement)
QUERY STRATEGY: Uncertainty Sampling (instead of Certainty)

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
    query_strategy: str = "uncertainty"  # CHANGED FROM "certainty"
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
    """Certainty-based query strategy (ORIGINAL - for reference)"""

    def query(self, model, X_unlabeled: np.ndarray,
              unlabeled_indices: np.ndarray) -> Tuple[int, float]:
        probas = model.predict_proba(X_unlabeled)

        # Handle case where predict_proba returns only one column
        if probas.shape[1] == 1:
            probas = np.column_stack([1 - probas[:, 0], probas[:, 0]])

        relevant_probas = probas[:, 1]
        max_idx = np.argmax(relevant_probas)  # Select HIGHEST probability
        return unlabeled_indices[max_idx], relevant_probas[max_idx]


class UncertaintyQueryStrategy:
    """
    Uncertainty-based query strategy (NEW)
    
    Selects documents closest to the decision boundary (probability ≈ 0.5)
    This explores the uncertainty in the model's predictions.
    """

    def query(self, model, X_unlabeled: np.ndarray,
              unlabeled_indices: np.ndarray) -> Tuple[int, float]:
        probas = model.predict_proba(X_unlabeled)

        # Handle case where predict_proba returns only one column
        if probas.shape[1] == 1:
            probas = np.column_stack([1 - probas[:, 0], probas[:, 0]])

        relevant_probas = probas[:, 1]
        
        # Calculate uncertainty: distance from 0.5 (decision boundary)
        # Lower values = more uncertain
        uncertainty = np.abs(relevant_probas - 0.5)
        
        # Select the MOST uncertain (closest to 0.5)
        most_uncertain_idx = np.argmin(uncertainty)
        
        return unlabeled_indices[most_uncertain_idx], relevant_probas[most_uncertain_idx]


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
            print(f"  - Observed prevalence: {observed_prevalence:.3f} ({observed_prevalence * 100:.1f}%)")
            print(f"  - Expected relevant in dataset: {self.expected_relevant:.1f}")
            print(f"  - Stopping threshold (2×): {self.expected_relevant * self.config.expected_relevant_multiplier:.1f}")

    def update(self, label: int):
        """Update consecutive irrelevant counter"""
        if label == 0:
            self.consecutive_irrelevant_count += 1
        else:
            self.consecutive_irrelevant_count = 0

    def check_criteria(self, n_screened: int) -> Tuple[bool, Dict[str, bool]]:
        """
        Check if ALL stopping criteria are met.

        Returns:
            (should_stop, criteria_status)
        """
        # Criterion 1: Screened ≥ 2× expected relevant
        if self.expected_relevant is None:
            raise ValueError("Expected relevant not set. Call set_expected_relevant() first.")

        screened_enough_relevant = n_screened >= (self.expected_relevant * self.config.expected_relevant_multiplier)

        # Criterion 2: Screened ≥ 10% of dataset
        proportion_screened = n_screened / self.n_total_records
        screened_min_percentage = proportion_screened >= self.config.min_screened_percentage

        # Criterion 3: N consecutive irrelevant
        consecutive_met = self.consecutive_irrelevant_count >= self.config.consecutive_irrelevant

        # Status dictionary
        criteria_status = {
            'screened_2x_expected': screened_enough_relevant,
            'screened_10_percent': screened_min_percentage,
            'consecutive_irrelevant': consecutive_met
        }

        # ALL must be True
        should_stop = all(criteria_status.values())

        return should_stop, criteria_status


class ASReviewSAFE:
    """
    Simplified SAFE implementation with UNCERTAINTY SAMPLING
    
    Phase 1: Adaptive prior knowledge (≥1 relevant)
    Phase 2: Active learning with 3-fold stopping
    """

    def __init__(self, config: Optional[SAFEConfig] = None):
        self.config = config if config else SAFEConfig()

        # Initialize components
        self.feature_extractor = TFIDFFeatureExtractor()
        self.resampler = DynamicResampler(random_state=self.config.random_state)
        self.query_strategy = UncertaintyQueryStrategy()  # CHANGED FROM CertaintyQueryStrategy()
        self.classifier = MultinomialNB()
        self.metrics = PerformanceMetrics()

        # Data containers
        self.texts = None
        self.y_labels = None
        self.record_ids = None
        self.X_features = None
        self.labeled_mask = None
        self.iteration = 0

        # Random state
        self.rng = np.random.RandomState(self.config.random_state)

    def load_data(self, texts: List[str], labels: List[int], record_ids: List[int]):
        """Load dataset"""
        self.texts = np.array(texts)
        self.y_labels = np.array(labels)
        self.record_ids = np.array(record_ids)
        self.labeled_mask = np.zeros(len(self.texts), dtype=bool)

        # Extract features
        self.X_features = self.feature_extractor.fit_transform(self.texts.tolist())

        if self.config.verbose:
            print(f"\nDataset Loaded:")
            print(f"  Total records: {len(self.texts)}")
            print(f"  Relevant records: {np.sum(self.y_labels == 1)}")
            print(f"  Prevalence: {np.sum(self.y_labels == 1) / len(self.texts) * 100:.2f}%")
            print(f"  Query Strategy: UNCERTAINTY SAMPLING")  # NEW

    def phase1_adaptive_prior_knowledge(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Phase 1: Adaptive prior knowledge sampling

        Continues random sampling until at least 1 relevant is found.
        Maximum: max_prior_percentage of dataset
        """
        if self.config.verbose:
            print(f"\n{'=' * 70}")
            print("PHASE 1: ADAPTIVE PRIOR KNOWLEDGE")
            print(f"{'=' * 70}")

        n_total = len(self.texts)
        initial_n = max(int(n_total * self.config.prior_knowledge_percentage),
                        self.config.min_prior_relevant + 1)

        unlabeled_indices = np.where(~self.labeled_mask)[0]
        prior_indices = self.rng.choice(unlabeled_indices, size=initial_n, replace=False)

        self.labeled_mask[prior_indices] = True
        prior_labels = self.y_labels[prior_indices]

        n_relevant = np.sum(prior_labels == 1)

        if self.config.verbose:
            print(f"Initial random sample: {initial_n} records ({initial_n / n_total * 100:.2f}%)")
            print(f"Relevant found: {n_relevant}")

        # Continue sampling until we have at least min_prior_relevant
        max_prior = int(n_total * self.config.max_prior_percentage)

        while n_relevant < self.config.min_prior_relevant and len(prior_indices) < max_prior:
            unlabeled_indices = np.where(~self.labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            additional = min(initial_n, len(unlabeled_indices))
            new_indices = self.rng.choice(unlabeled_indices, size=additional, replace=False)

            self.labeled_mask[new_indices] = True
            new_labels = self.y_labels[new_indices]

            prior_indices = np.concatenate([prior_indices, new_indices])
            prior_labels = np.concatenate([prior_labels, new_labels])

            n_relevant = np.sum(prior_labels == 1)

            if self.config.verbose:
                print(f"Added {additional} more records. Total: {len(prior_indices)}. Relevant: {n_relevant}")

        if self.config.verbose:
            print(f"\nFinal prior knowledge:")
            print(f"  Size: {len(prior_indices)} ({len(prior_indices) / n_total * 100:.2f}%)")
            print(f"  Relevant: {n_relevant} ({n_relevant / len(prior_indices) * 100:.2f}%)")
            print(f"{'=' * 70}\n")

        return prior_indices, prior_labels

    def phase2_screen_with_stopping(self, max_iterations: Optional[int] = None) -> Dict:
        """
        Phase 2: Active learning screening with 3-fold stopping criteria
        USES UNCERTAINTY SAMPLING
        """
        if self.config.verbose:
            print(f"\n{'=' * 70}")
            print("PHASE 2: SCREENING WITH STOPPING CRITERIA")
            print(f"{'=' * 70}\n")

        n_total = len(self.texts)
        n_total_relevant = np.sum(self.y_labels == 1)
        n_relevant_found = np.sum(self.y_labels[self.labeled_mask] == 1)

        # Initialize stopping criteria
        n_prior = np.sum(self.labeled_mask)
        n_relevant_in_prior = n_relevant_found
        stopping = SimplifiedStoppingCriteria(self.config, n_total)
        stopping.set_expected_relevant(n_prior, n_relevant_in_prior)

        if max_iterations is None:
            max_iterations = n_total

        stopped_early = False
        stopping_reason = None
        criteria_status = None

        start_time = time.time()

        while self.iteration < max_iterations:
            self.iteration += 1

            labeled_indices = np.where(self.labeled_mask)[0]
            unlabeled_indices = np.where(~self.labeled_mask)[0]

            if len(unlabeled_indices) == 0:
                stopping_reason = "all_screened"
                break

            # Check stopping criteria
            should_stop, criteria_status = stopping.check_criteria(np.sum(self.labeled_mask))

            if should_stop:
                stopped_early = True
                stopping_reason = "stopping_criteria_met"

                if self.config.verbose:
                    print(f"\n{'=' * 70}")
                    print("⛔ STOPPING CRITERIA MET!")
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

            # Query next record (USES UNCERTAINTY SAMPLING)
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
                      f"Label={'✓' if label == 1 else '✗'}, "
                      f"Conf={confidence:.3f}")

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
            print(f"Query Strategy: UNCERTAINTY SAMPLING")
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
        plt.plot(proportions * 100, recalls * 100, 'b-', linewidth=2, label='Active Learning (Uncertainty)')
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
        plt.title('SAFE Active Learning: Recall Curve (Uncertainty Sampling)', fontsize=14, fontweight='bold')
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
    """Demonstration with UNCERTAINTY SAMPLING"""
    print("=" * 70)
    print("ASReview SAFE - WITH UNCERTAINTY SAMPLING")
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
    asreview.plot_recall_curve(save_path='/home/claude/recall_curve_uncertainty.png')

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Query Strategy: UNCERTAINTY SAMPLING")
    print(f"Stopped early: {results['stopped_early']}")
    print(f"Stopping reason: {results['stopping_reason']}")
    print(f"Work saved: {(1 - results['proportion_screened']) * 100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
