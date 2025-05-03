# Graph-Based Game Recommendations on Steam
This research project explores graph-based recommendation systems for Steam games, with a focus on Personalized PageRank (PPR). This project compares PPR and its variants (Multi-Alpha PPR, Two-Phase PPR) against traditional algorithms like Item-kNN, using a dataset of user-game interactions. Evaluated under strong generalization for unseen users, the system achieves competitive performance (NDCG@20 ~0.318, Recall@20 ~0.379). This repository contains code, data, and resources to reproduce the findings.

## Key Findings
- PPR Effectiveness: Base PPR performs comparably to Item-kNN, with low alpha (~0.023) emphasizing personalization over graph exploration.
- Variant Limitations: Multi-Alpha and Two-Phase PPR offer no significant improvements, converging to base PPR behavior with higher computational costs.
- Personalization Dominance: Low popularity weights (~0.033) highlight the importance of user-specific preferences over global item popularity.
- Simplicity Wins: Simple collaborative filtering (Item-kNN) matches complex graph-based methods in this domain.

## Project Structure
```
root/
├── data/
│   ├── raw/                    # Raw dataset files (.csv)
│   └── processed/              # Preprocessed data
├── output/
│   ├── optimization/           # Hyperparameter tuning results
│   └── submissions/            # Codabench submission files
├── src/
│   ├── models/
│   │   ├── ppr/                # PPR implementations (base, Multi-Alpha, Two-Phase)
│   │   ├── itemKNN.py          # Item-kNN model
│   │   ├── userKNN.py          # User-kNN model
│   │   ├── popularity.py       # Popularity-based model
│   │   └── random.py           # Random baseline
│   ├── eda.ipynb               # Exploratory Data Analysis
│   ├── optimization.ipynb      # Bayesian optimization
│   ├── results.ipynb           # Model evaluation
│   ├── visualization.ipynb     # Visualizations
│   ├── evaluator.py            # Evaluation framework
│   ├── metrics.py              # NDCG, Recall metrics
│   └── optimizer.py            # Bayesian optimizer
├── report/
│   ├── report.pdf              # Research paper
│   └── source/                 # LaTeX source files
├── presentation.pdf            # Presentation slides
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Citation
If you use this work, please cite:
```
@misc{deputter2025steamgraphrec,
  title={Graph-Based Game Recommendation System Using Steam User-Item Interactions},
  author={Pablo Deputter},
  year={2025},
  url={https://github.com/pabloDeputter/steam-graph-recommendations}
}
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.
