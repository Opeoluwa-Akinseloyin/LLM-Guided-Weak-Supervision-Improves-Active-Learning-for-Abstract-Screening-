"""
ASReview with Complete Screening (Phases 1-3)
QUERY STRATEGY: UNCERTAINTY SAMPLING

This script runs ASReview with:
- Phase 1: Prior knowledge sampling
- Phase 2: Active learning with stopping criteria (UNCERTAINTY SAMPLING)
- Phase 3: Continue to 100% recall
- Tracking of iterations at 95% and 100% recall
"""

import pandas as pd
from asreview_complete_uncertainty import ASReviewComplete, SAFEConfig

# ========================================
# 1. LOAD YOUR CSV FILE
# ========================================
df = pd.read_csv('INT_data/Criteria/CD012164_Criteria.csv')

print(f"Loaded {len(df)} records")
print(f"Columns: {df.columns.tolist()}")

# ========================================
# 2. PREPARE DATA
# ========================================
texts = df['text'].tolist()
labels = df['Status'].tolist()
record_ids = df['PMID'].tolist()

print(f"\nRelevant papers: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")

topic = df["Topic"].iloc[0]

for i in range(10):
    # ========================================
    # 3. RUN ASREVIEW WITH UNCERTAINTY SAMPLING
    # ========================================

    n = i + 1
    config = SAFEConfig(
        prior_knowledge_percentage=0.01,
        min_prior_relevant=1,
        consecutive_irrelevant=50,
        random_state=n,
        verbose=True
    )

    # ASReviewComplete now uses UNCERTAINTY SAMPLING
    asreview = ASReviewComplete(
        config=config,
        output_dir=f"INT/{topic}/{topic}_uncertainty_run_{n}"
    )

    asreview.load_data(
        texts=texts,
        labels=labels,
        record_ids=record_ids
    )

    # This now runs Phases 1, 2, AND 3 with UNCERTAINTY SAMPLING!
    results = asreview.run_safe_phases_1_and_2()

    # ========================================
    # 4. CHECK RESULTS
    # ========================================

    print("\n" + "="*70)
    print("COMPLETE SCREENING RESULTS (UNCERTAINTY SAMPLING)")
    print("="*70)
    print(f"✅ Output: {results['output_dir']}")

    print(f"\n📊 PHASE 2 (with stopping):")
    print(f"   Query Strategy: UNCERTAINTY SAMPLING")
    print(f"   Iterations: {results['phase2_ended_at_iteration']}")
    print(f"   Screened: {results['n_screened']}/{len(texts)} ({results['proportion_screened']*100:.1f}%)")
    print(f"   Recall: {results['final_recall']*100:.1f}%")
    print(f"   Stopped: {results['stopped_early']} ({results['stopping_reason']})")

    print(f"\n🎯 PHASE 3 (continue to 100%):")
    print(f"   Additional iterations: {results['phase3_iterations']}")
    print(f"   Total screened: {results['total_screened_at_100']}/{len(texts)} ({results['proportion_screened_at_100']*100:.1f}%)")

    print(f"\n📈 KEY MILESTONES:")
    print(f"   95% recall: iteration {results['iteration_at_95_recall']}")
    print(f"   100% recall: iteration {results['iteration_at_100_recall']}")
    print(f"   Iterations (95%→100%): {results['iteration_at_100_recall'] - results['iteration_at_95_recall']}")

    print(f"\n💾 FILES SAVED:")
    print(f"   - COMPLETE_REPORT.txt (comprehensive stats)")
    print(f"   - recall_curve_complete.png (with milestones)")
    print(f"   - phase3/all_screened_to_100_recall.csv")
    print("="*70 + "\n")
