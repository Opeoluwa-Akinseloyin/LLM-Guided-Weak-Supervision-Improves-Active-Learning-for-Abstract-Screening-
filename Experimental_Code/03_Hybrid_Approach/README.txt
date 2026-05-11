================================================================================
  03 — HYBRID APPROACH
================================================================================

These scripts implement hybrid query strategies for Phase 2. Instead of
using only certainty sampling (picking the most confidently relevant documents)
or only uncertainty sampling (picking the most ambiguous documents), the hybrid
approach alternates between both within the same run.

The hypothesis: alternating between certainty and uncertainty sampling could
improve convergence speed or final recall by better exploring the document space.

--------------------------------------------------------------------------------
HYBRID MODES
--------------------------------------------------------------------------------

  Alternating (certainty start):   certainty → uncertainty → certainty → ...
  Alternating (uncertainty start): uncertainty → certainty → uncertainty → ...
  Percentage-based:                N% of each batch uses certainty, the rest
                                   uses uncertainty (configurable split)

--------------------------------------------------------------------------------
FILES
--------------------------------------------------------------------------------

asreview_complete_hybrid.py
  Core hybrid engine. Extends the base ASReview workflow to support the
  hybrid query modes described above. Handles the phase-switching logic and
  percentage-based alternating. Tracks milestones and saves complete reports
  per run. This is the main module imported or run for single-dataset hybrid
  experiments.

asreview_multi_dataset_hybrid.py
  Multi-dataset batch launcher for hybrid experiments with random Phase 1
  initialisation. Reads dataset_config.csv, runs all configured hybrid modes
  across every dataset (10 runs each), and writes summary CSVs comparing
  method performance side by side. Results go to hybrid_results/.

Pseudo_asreview_multi_dataset_hybrid.py
  Same as asreview_multi_dataset_hybrid.py but uses pseudo-labeling for
  Phase 1 initialisation instead of random. Combines the pseudo-labeling
  approach from folder 02 with the hybrid query strategy from this folder.

pseudo_asreview_complete_hybrid.py
  Single-dataset runner combining pseudo-labeling Phase 1 with hybrid Phase 2
  query strategy. The complete version with Phase 3 milestone tracking.

--------------------------------------------------------------------------------
OUTPUTS
--------------------------------------------------------------------------------

  Results are saved to hybrid_results/ (random init) or pseudo_results/
  (pseudo-label init). Each run produces a COMPLETE_REPORT.txt and summary
  metrics used later by the RQ4 analysis scripts.

================================================================================
