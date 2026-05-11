================================================================================
  01 — CORE ASREVIEW
================================================================================

These are the foundational scripts. Every other variant in this project
(pseudo-labeling, hybrid, uncertainty) is built on top of the patterns
established here. Read these first.

--------------------------------------------------------------------------------
FILES
--------------------------------------------------------------------------------

asreview_safe_simplified.py
  The base library module. Defines the core building blocks used by all
  other scripts:
    - TF-IDF feature extraction
    - Logistic Regression classifier
    - Certainty-based query strategy (selects highest-confidence documents)
    - Dynamic resampling during training
    - SAFE stopping criterion for Phase 2
    - Performance metric tracking (WSS, recall, precision, F1, AUC)
  This file is not run directly — it is imported by the other scripts.

ASReview.py
  The top-level runner for a single dataset. Executes 10 independent runs
  of the full 3-phase screening workflow:
    Phase 1 — Random initialisation: a small batch of documents is labelled
              at random to train the first classifier.
    Phase 2 — Active learning with SAFE stopping: the classifier iteratively
              selects the most certain documents until the stopping criterion
              is met (estimated high recall).
    Phase 3 — Continue to 100% recall: screening continues until all relevant
              documents have been found.
  Outputs: recall curves, COMPLETE_REPORT.txt, screened document lists.

asreview_complete.py
  An extended version of ASReview.py that adds detailed Phase 3 milestone
  tracking (iterations to reach 95% and 100% recall). Also generates
  richer summary reports and recall curve visualisations. Use this version
  when you need full performance metrics across all three phases.

asreview_with_artifacts.py
  Adds persistent artifact saving to the complete workflow. At the end of
  each run it saves the trained classifier (.pkl), feature extractor (.pkl),
  and prediction scores (.csv) to disk. Useful when you need to inspect
  the model state or reuse predictions downstream without re-running.

--------------------------------------------------------------------------------
HOW THEY RELATE
--------------------------------------------------------------------------------

  asreview_safe_simplified  <-- imported by all three runners
         |
         +-- ASReview.py               (basic 3-phase run)
         +-- asreview_complete.py      (adds milestone tracking)
         +-- asreview_with_artifacts.py (adds model/prediction saving)

================================================================================
