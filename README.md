# Graph-Based Game Recommendations on Steam

This research project explores graph-based recommendation systems for Steam games, with a focus on Personalized PageRank (PPR). This project compares PPR and its variants (Multi-Alpha PPR, Two-Phase PPR) against traditional algorithms like Item-KNN, using a dataset of user-game interactions. Evaluated under strong generalization for unseen users, the system achieves competitive performance (NDCG@20 ~0.318, Recall@20 ~0.379). This repository contains code, data, and resources to reproduce the findings. The report is available [here](report/report.pdf) and the presentation slides [here](presentation.pdf).

## Key Findings

- PPR Effectiveness: Base PPR performs comparably to Item-kNN, with low alpha (~0.023) emphasizing personalization over graph exploration.
- Variant Limitations: Multi-Alpha and Two-Phase PPR offer no significant improvements, converging to base PPR behavior with higher computational costs.
- Personalization Dominance: Low popularity weights (~0.033) highlight the importance of user-specific preferences over global item popularity.
- Simplicity Wins: Simple collaborative filtering (Item-kNN) matches complexer graph-based methods in this domain.

## Project Structure

```
root/
├── data/
│   ├── raw/                    # Raw dataset files (.csv)
├── output/
│   ├── images/                 # Images used in report/presentation
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

## Setup

Install dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

The `src/evaluator.py` script can be run to evaluate different models and generate submission files. Edit the `if __name__ == "__main__"` block to select models and parameters. To run the evaluation:

```bash
python -m src.evaluator
```

The `src/optimizer.py` script uses Bayesian optimization to tune hyperparameters for models (e.g., PPR variants). Again, edit the `if __name__ == "__main__"` block to configure the model and parameter space. To run the optimization:

```bash
python -m src.optimizer
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
