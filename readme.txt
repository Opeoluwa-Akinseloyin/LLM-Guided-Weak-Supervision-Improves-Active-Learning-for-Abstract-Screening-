================================================================================
  EXPERIMENTAL CODE — ACTIVE LEARNING FOR SYSTEMATIC REVIEW SCREENING
================================================================================

This repository contains the experimental code for an ASReview-style systematic
review screening study. Each folder implements a different combination of
Phase 1 initialisation (random vs. LLM pseudo-labeling) and Phase 2 query
strategy (certainty vs. uncertainty vs. hybrid), plus the metrics tooling
used to extract and validate results across experiments.

All experiments follow the same 3-phase workflow:

    Phase 1 — Initialisation
              A small batch of documents is labelled (randomly, or via LLM
              pseudo-labels) to train the first classifier.
    Phase 2 — Active learning with the SAFE stopping criterion
              The classifier iteratively selects documents using the
              configured query strategy until estimated high recall is met.
    Phase 3 — Continue to 100% recall
              Screening continues until all relevant documents are found.
              Milestones at 95% and 100% recall are tracked.

--------------------------------------------------------------------------------
REPOSITORY LAYOUT
--------------------------------------------------------------------------------

01_Core_ASReview/
  Foundational scripts. Random Phase 1 initialisation + certainty sampling
  in Phase 2. Every other variant builds on the patterns established here.
  Start by reading this folder's README.

02_Pseudo_Labeling/
  Replaces random Phase 1 with LLM-generated confidence scores. Investigates
  whether a better-informed Phase 1 leads to faster convergence and higher
  recall. Includes a multi-dataset batch launcher.

03_Hybrid_Approach/
  Hybrid Phase 2 query strategies that alternate between certainty and
  uncertainty sampling — either by phase-switching or by a percentage split
  within each batch. Covers both random-init and pseudo-label-init variants.

04_Uncertainty_Sampling/
  Mirrors folders 01 and 02 but switches the Phase 2 query strategy to
  uncertainty sampling (documents closest to the decision boundary). Used
  as the comparison condition against certainty sampling.

06_Metrics_and_Utilities/
  Post-processing and setup helpers. Dataset config generation, random
  baseline generation, metric extraction from COMPLETE_REPORT.txt files,
  WSS computation and correction, and validation scripts.

--------------------------------------------------------------------------------
EXPERIMENT MATRIX
--------------------------------------------------------------------------------

                       |  Certainty (Phase 2)  |  Uncertainty (Phase 2)
  ---------------------+-----------------------+------------------------
  Random init          |  Folder 01            |  Folder 04
  Pseudo-label init    |  Folder 02            |  Folder 04
  Hybrid (alternating) |  Folder 03            |  Folder 03
                       |  (random or pseudo)   |  (random or pseudo)

--------------------------------------------------------------------------------
TYPICAL WORKFLOW
--------------------------------------------------------------------------------

  SETUP
    1. 06_Metrics_and_Utilities/Asreview csv config.py
         — generate dataset_config.csv with paths to criteria, ranking,
           and LLM prediction files for each dataset
    2. 06_Metrics_and_Utilities/generate_random_baseline_csv.py
         — create the random baseline used as the comparison condition

  RUN EXPERIMENTS
    3. Use the single-dataset runners (ASReview.py, Pseudo-ASReview.py,
       ASReview_uncertainty.py, ...) for one dataset at a time, or the
       multi-dataset launchers (Pseudo asreview multi dataset.py,
       asreview_multi_dataset_hybrid.py, ...) to batch across datasets.
       Each experiment runs 10 independent repetitions.

  EXTRACT AND VALIDATE METRICS
    4. phase1_wss_from_predictions.py         — Phase 1 WSS at 95% / 100%
    5. extract_and_merge_hybrid_metrics_v2.py — hybrid result metrics
    6. extract_random_hybrid_metrics.py       — random-init hybrid metrics
    7. compare_wss.py                         — validate extracted values
    8. fix_wss.py / fix_random_wss.py         — correct WSS where needed

--------------------------------------------------------------------------------
KEY CONCEPTS
--------------------------------------------------------------------------------

  TF-IDF + Logistic Regression
    The feature extractor and classifier shared across all variants.

  Certainty sampling
    Selects documents the classifier is most confident are relevant. Used
    in folders 01, 02, and as one half of the hybrid strategies.

  Uncertainty sampling
    Selects documents closest to the decision boundary. Used in folder 04
    and as the other half of the hybrid strategies.

  Pseudo-labeling (Phase 1)
    Uses LLM confidence scores to pre-label the top N% most-likely relevant
    and bottom N% most-likely irrelevant documents before active learning
    begins, in place of random initialisation.

  SAFE stopping criterion
    Determines when Phase 2 ends, based on estimated recall.

  WSS (Work Saved over Sampling)
    Primary efficiency metric. Computed at 95% and 100% recall thresholds.
    Formula: WSS = (1 - total_work / N) - (1 - total_recall)

--------------------------------------------------------------------------------
INPUTS REQUIRED
--------------------------------------------------------------------------------

  - Criteria CSV files (DTA_DATA/ and INT_DATA/ datasets)
  - LLM confidence score CSVs (one per dataset, for pseudo-labeling)
  - dataset_config.csv / pseudo_dataset_config.csv (generated by the
    config helper in folder 06)

--------------------------------------------------------------------------------
OUTPUTS
--------------------------------------------------------------------------------

  Each experiment run writes a COMPLETE_REPORT.txt, recall curves, screened
  document lists, and (where applicable) saved classifier .pkl files and
  prediction score CSVs. Multi-dataset launchers also produce summary CSVs
  comparing methods side by side. The folder 06 extraction scripts merge
  these into the consolidated metrics CSVs used for final analysis.

--------------------------------------------------------------------------------
WHERE TO START
--------------------------------------------------------------------------------

  1. Read 01_Core_ASReview/README.txt to understand the base workflow.
  2. Read the README.txt in each subsequent folder for the variant it adds.
  3. Read 06_Metrics_and_Utilities/README.txt last — it depends on outputs
     from the experiment folders.

================================================================================
