# Deep learning-based patient stratification for prognostic enrichment of clinical dementia trials


Upon publication, we will publish the code for our article titled "Deep learning-based patient stratification for prognostic enrichment of clinical dementia trials" in this repository.

**All outputs and pointers towards the patient data have been removed. Therefore, the code is not executable in its current form. Its published solely for transparency.**

## VaDER: Clustering patient trajectories
The code for training a deep learning model for clustering the patient trajectories can be found [here](https://github.com/yalchik/VaDER).

## Repository structure

- src/: Hold all the source code
  - clustering/: Code for clustering, cluster validation, and replication.
  - classifier_training/: Nested cross-validation, Bayesian hyperparameter optimization, xgboost classification model
  - cost_analysis/: Code used to run the cost analysis presented in the article.
