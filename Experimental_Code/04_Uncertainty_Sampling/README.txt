================================================================================
  04 — UNCERTAINTY SAMPLING
================================================================================

These scripts are direct mirrors of the Core ASReview (folder 01) and
Pseudo-Labeling (folder 02) scripts, but configured to use UNCERTAINTY
sampling as the Phase 2 query strategy instead of certainty sampling.

Uncertainty sampling selects documents where the classifier is least
confident (closest to the decision boundary), rather than documents it
is most confident about. This is a standard alternative in active learning
and is compared against the certainty approach in the research questions.

--------------------------------------------------------------------------------
FILES
--------------------------------------------------------------------------------

asreview_safe_uncertainty.py
  The base library module equivalent to asreview_safe_simplified.py (folder 01)
  but with the query strategy switched to uncertainty sampling. All other
  scripts in this folder import from this module.

ASReview_uncertainty.py
  Single-dataset runner (mirrors ASReview.py). Runs 10 complete 3-phase
  experiments using uncertainty sampling in Phase 2. Outputs recall curves,
  COMPLETE_REPORT.txt, and screened document lists.

asreview_complete_uncertainty.py
  Extended runner with Phase 3 milestone tracking (mirrors asreview_complete.py).
  Produces full performance metrics across all three phases.

asreview_with_artifacts_uncertainty.py
  Adds model and prediction saving to the uncertainty workflow (mirrors
  asreview_with_artifacts.py). Saves trained classifier .pkl files and
  prediction CSVs after each run.

asreview_llm_pseudo_labeling_uncertainty.py
  Core pseudo-labeling engine configured for uncertainty sampling (mirrors
  asreview_llm_pseudo_labeling.py from folder 02).

Pseudo_ASReview_uncertainty.py
  Single-dataset pseudo-labeling runner with uncertainty Phase 2.

pseudo_asreview_complete_uncertainty.py
  Extended pseudo-labeling + uncertainty runner with full milestone tracking.

Pseudo_asreview_multi_dataset_uncertainty.py
  Multi-dataset batch launcher for pseudo-labeling + uncertainty experiments.
  Results go to ASReview uncertainty/pseudo_results_uncertainty/.

Asreview_csv_config_uncertainty.py
  Dataset configuration helper for uncertainty experiments. Generates or
  updates the CSV config used by the multi-dataset launcher.

--------------------------------------------------------------------------------
HOW THIS COMPARES TO FOLDER 01 / 02
--------------------------------------------------------------------------------

  Folder 01 (certainty)  <-->  Folder 04 (uncertainty)
  -------------------------------------------------------
  asreview_safe_simplified    <-->  asreview_safe_uncertainty
  ASReview.py                 <-->  ASReview_uncertainty.py
  asreview_complete.py        <-->  asreview_complete_uncertainty.py
  asreview_with_artifacts.py  <-->  asreview_with_artifacts_uncertainty.py

  Folder 02 (pseudo + certainty)  <-->  Folder 04 (pseudo + uncertainty)
  -----------------------------------------------------------------------
  asreview_llm_pseudo_labeling.py  <-->  asreview_llm_pseudo_labeling_uncertainty.py
  Pseudo-ASReview.py               <-->  Pseudo_ASReview_uncertainty.py
  pseudo_asreview_complete.py      <-->  pseudo_asreview_complete_uncertainty.py
  Pseudo asreview multi dataset.py <-->  Pseudo_asreview_multi_dataset_uncertainty.py

================================================================================
