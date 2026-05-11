================================================================================
  02 — PSEUDO LABELING
================================================================================

These scripts replace random Phase 1 initialisation with LLM-generated
confidence scores. Instead of picking documents randomly to seed the first
classifier, the system uses a large language model's predictions to pre-label
a set of documents — giving the classifier a smarter starting point.

The key question being investigated: does a better-informed Phase 1 lead to
faster convergence and higher recall in Phase 2?

--------------------------------------------------------------------------------
FILES
--------------------------------------------------------------------------------

asreview_llm_pseudo_labeling.py
  The core pseudo-labeling engine. Reads LLM confidence scores from a CSV
  file and uses them to assign pseudo-labels to documents before active
  learning starts. Supports configurable top/bottom ratio selection — e.g.
  take the top 5% most-likely relevant and bottom 5% most-likely irrelevant
  as the initial labelled set. This module is imported by the other scripts
  in this folder.

Pseudo-ASReview.py
  Single-dataset runner equivalent to ASReview.py (in 01_Core_ASReview)
  but with pseudo-labeling in Phase 1. Runs 10 independent experiments,
  logs Phase 1 pseudo-label statistics alongside Phase 2 and Phase 3
  results, and saves complete reports. Use this to run one dataset at a time.

pseudo_asreview_complete.py
  Extended version of Pseudo-ASReview.py that adds Phase 3 milestone
  tracking and richer metric output. Mirrors the relationship between
  ASReview.py and asreview_complete.py in the core folder.

Pseudo asreview multi dataset.py
  Batch launcher. Reads a dataset configuration CSV (pseudo_dataset_config.csv)
  and runs pseudo-labeling experiments across all listed datasets automatically.
  Results per dataset are saved to the pseudo_results/ folder. Use this when
  you want to run experiments across all DTA and INT datasets in one go.

--------------------------------------------------------------------------------
INPUTS REQUIRED
--------------------------------------------------------------------------------

  - LLM confidence scores CSV (one per dataset, path set in config)
  - pseudo_dataset_config.csv (for the multi-dataset launcher)
  - Criteria CSV files from DTA_DATA/ or INT_DATA/

--------------------------------------------------------------------------------
KEY PARAMETER: top/bottom ratio
--------------------------------------------------------------------------------

  Controls how many pseudo-labels are assigned in Phase 1.
  Example: top=0.05, bottom=0.05 means the top 5% highest-scoring documents
  are labelled "relevant" and the bottom 5% are labelled "irrelevant".
  Multiple ratio configurations are tested to find the optimal setting.

================================================================================
