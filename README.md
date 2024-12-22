# Graph-Based Game Recommendation System using Steam User-Item Interactions

**Author:** Pablo Deputter

**Email:** pablo.deputter@uantwerp.be

## Overview

This project explores graph-based recommendation techniques, focusing on Personalized PageRank (PPR), for improving game recommendations on Steam. We compare PPR and its variants against traditional algorithms using a dataset of user-game interactions. The evaluation emphasizes strong generalization with unseen users, utilizing metrics like NDCG@20 and Recall@20. This repository contains code, data, and resources to reproduce our findings.

## Key Findings and Results
*   **PPR Performance:** The base PPR model achieved competitive performance (NDCG@20 around 0.318, Recall@20 around 0.379), close to Item-kNN, emphasizing the importance of user-item interactions.
*   **Parameter Importance:** Low alpha values (around 0.023) in PPR indicate a strong reliance on personalization vectors over graph exploration. Similarly, low popularity weights showed the models favor personalized scores over global item popularity.
*   **PPR Variants:** More complex PPR variants (Multi-Alpha PPR and Two-Phase PPR) did not provide meaningful improvements over the base PPR model, often converging to its behavior while increasing computational cost.
*   **Baseline Performance:** Item-kNN performed well, comparable with the base PPR model, highlighting the effectiveness of simple item-similarity in this specific domain.
*   **Complexity vs Performance:** Additional complexity in PPR variants did not improve performance, and the models prioritized direct user preferences rather than higher-order graph information.

## Conclusion
This project found that complex graph-based recommendation methods, while theoretically powerful, offer only marginal gains over simpler collaborative filtering models for game recommendations on Steam.  Emphasis on personalization, with low alpha and low popularity weight, was consistently more impactful than exploration within the user-item graph.  Further research should explore hybrid approaches that incorporate content-based features or investigate the limitations of directly applying GNN approaches to this problem.


## Project Structure

*   **`data/`**: Raw dataset files.
    *   **`data/raw/`**: `.csv` interaction, game, and review data.
*   **`output/`**: Output from experiments.
    *   **`output/optimization/`**: Results from hyperparameter optimization (visualizations, optimal parameters, ...).
    *   **`output/submissions/`**: Submission files for Codabench (with model name and submission ID).
*   **`src/`**: Source code.
    *   **`src/eda.ipynb`**: Exploratory Data Analysis notebook.
    *   **`src/evaluator.py`**: Evaluation framework with sampling, metrics, and submission generation.
    *   **`src/metrics.py`**: Evaluation metrics (NDCG, Recall, ...).
    *   **`src/models/`**: Recommendation model implementations.
        *   **`src/models/itemKNN.py`**: Item-kNN recommender.
        *   **`src/models/popularity.py`**: Popularity recommender.
        *   **`src/models/random.py`**: Random recommender.
        *    **`src/models/userKNN.py`**: User-kNN recommender.
        *   **`src/models/ppr/`**: PPR model implementations.
            *   **`src/models/ppr/mappr.py`**: Multi-Alpha PPR.
            *   **`src/models/ppr/ppr.py`**: Base PPR.
            *   **`src/models/ppr/tppr.py`**: Two-Phase PPR.
    *   **`src/optimization.ipynb`**: Bayesian optimization notebook with findings.
    *   **`src/optimizer.py`**: Bayesian Optimizer implementation.
    *   **`src/results.ipynb`**: Model evaluation and results notebook.
    *   **`src/visualization.ipynb`**: Visualizations for paper and presentation.
*    **`presentation.pdf`**: PDF of the associated presentation.
*    **`report.pdf`**: PDF of the associated paper.
*   **`requirements.txt`**: Python dependencies.
*   **`README.md`**: This file.
*   

## Citation
If you use this code or research, please cite:
```
@misc{deputter2024graph,
title={Graph-Based Game Recommendation System Using Steam User-Item Interactions},
author={Pablo Deputter},
year={2024},
}
```