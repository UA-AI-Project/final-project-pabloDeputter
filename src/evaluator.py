import datetime
import logging
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.metrics import get_calibrated_recall, get_ndcg
from src.models import PPR, Recommender


class Evaluator:
    """
    Class to evaluate the performance of a model.

    :param train_path: path to the training data
    :param test_path: path to the test data
    """

    def __init__(
        self,
        train_path: str = "data/raw/train_interactions.csv",
        test_path: str = "data/raw/test_interactions_in.csv",
    ):
        # Use the actual class name when creating the logger
        self.logger = logging.getLogger(self.__class__.__name__)
        # Avoid adding handlers multiple times
        if not self.logger.handlers:
            self._setupLogger()
        self._loadData(train_path, test_path)

    def _setupLogger(self):
        """
        Set up the logger.

        This method configures the logger to output messages to the console.
        """
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _loadData(self, train_path: str, test_path: str):
        """
        Load the training and test data.

        :param train_path: path to the training data
        :param test_path: path to the test data
        """
        # Sanity check: ensure that the paths exist
        assert Path(train_path).exists(), f"Train data not found at {train_path}"
        assert Path(test_path).exists(), f"Test data not found at {test_path}"

        self.train_df = pd.read_csv(train_path)
        self.test_in_df = pd.read_csv(test_path)

        # Calculate global statistics for later use
        self.item_popularity = self.train_df["item_id"].value_counts(normalize=True)
        self.user_activity = self.train_df.groupby("user_id")["item_id"].count()
        self.playtime_stats = self.train_df.groupby("item_id")["playtime"].agg(["mean", "std"])

    @staticmethod
    def _softmax(x: np.array) -> np.array:
        """
        Compute softmax values for each sets of scores in x.

        :param x: input array
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def createValidationSplit(
        self, train_df: pd.DataFrame, val_size: float = 0.1, seed: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a validation split based on test set characteristics.

        :param train_df: training data
        :param val_size: size of the validation set
        :param seed: random seed for reproducibility

        :return: training and validation splits
        """
        np.random.seed(seed)

        # Calculate test set characteristics
        test_user_activity = self.test_in_df.groupby("user_id")["item_id"].count()
        test_item_distr = self.test_in_df["item_id"].value_counts(normalize=True)
        test_playtime_distr = self.test_in_df.groupby("user_id")["playtime"].mean()

        # Score users based on how well they match test set characteristics
        user_scores = {}
        for user, group in train_df.groupby("user_id"):
            # Activity similarity score
            user_activity = len(group)
            activity_diff = abs(user_activity - test_user_activity.mean()) / test_user_activity.std()
            activity_score = 1 / (1 + activity_diff)

            # Item distribution similarity score
            user_items = group["item_id"].value_counts(normalize=True)
            common_items = set(user_items.index) & set(test_item_distr.index)
            if common_items:
                item_score = np.mean(
                    [
                        min(user_items[i], test_item_distr[i]) / max(user_items[i], test_item_distr[i])
                        for i in common_items
                    ]
                )
            else:
                item_score = 0

            # Playtime distribution similarity score
            user_playtime = group["playtime"].mean()
            playtime_diff = abs(user_playtime - test_playtime_distr.mean()) / test_playtime_distr.std()
            playtime_score = 1 / (1 + playtime_diff)

            # Combine scores with weights
            user_scores[user] = 0.4 * activity_score + 0.4 * item_score + 0.2 * playtime_score

        # Select validation users based on scores
        scores = np.array(list(user_scores.values()))
        # Temperature parameter of 3
        probs = self._softmax(scores * 3)

        # Sanity check: ensure that the probabilities sum to 1
        assert np.isclose(probs.sum(), 1), "Probabilities do not sum to 1."

        # Select users for validation set
        n_val_users = int(len(user_scores) * val_size)
        val_users = np.random.choice(list(user_scores.keys()), size=n_val_users, replace=False, p=probs)

        # Create splits
        train_split = train_df[~train_df["user_id"].isin(val_users)].copy()
        val_split = train_df[train_df["user_id"].isin(val_users)].copy()

        return train_split, val_split

    def createFoldInSplit(self, val_df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split validation data to mimic test set characteristics.

        :param val_df: validation data
        :param seed: random seed for reproducibility

        :return: validation fold-in and fold-out splits
        """
        np.random.seed(seed)

        # Calculate test set characteristics
        test_items_per_user = self.test_in_df.groupby("user_id")["item_id"].count()

        val_in = []
        val_out = []

        for _, group in val_df.groupby("user_id"):
            n_items = len(group)
            if n_items < 2:
                continue

            # Target number of items based on test distribution
            target_items = max(
                1,
                min(
                    int(np.random.normal(test_items_per_user.mean(), test_items_per_user.std())),
                    n_items - 1,  # Ensure at least one item for ground truth
                ),
            )

            # Sort items by similarity to test set characteristics
            item_scores = []
            for _, row in group.iterrows():
                # Popularity score
                pop_score = self.item_popularity.get(row["item_id"], 0)

                # Playtime score
                playtime_stats = self.playtime_stats.loc[row["item_id"]]
                playtime_score = 1 / (
                    1 + abs(row["playtime"] - playtime_stats["mean"]) / (playtime_stats["std"] + 1e-6)
                )

                item_scores.append(0.7 * pop_score + 0.3 * playtime_score)

            # Select items for fold-in based on scores
            sorted_indices = np.argsort(item_scores)[::-1]
            fold_in_indices = sorted_indices[:target_items]
            fold_out_indices = sorted_indices[target_items:]

            val_in.append(group.iloc[fold_in_indices])
            val_out.append(group.iloc[fold_out_indices])

        return pd.concat(val_in), pd.concat(val_out)

    def createSampledDataset(self, df: pd.DataFrame, n_users: int, seed: int = 42) -> pd.DataFrame:
        """
        Sample users maintaining activity distribution.

        :param df: input dataframe
        :param n_users: number of users to sample
        :param seed: random seed for reproducibility

        :return: sampled dataframe
        """
        np.random.seed(seed)

        user_counts = df["user_id"].value_counts()
        activity_bins = pd.qcut(user_counts, q=5, labels=["v_low", "low", "med", "high", "v_high"])

        # Sample users maintaining activity distribution
        sampled_users = []
        for bin_label in activity_bins.unique():
            bin_users = user_counts[activity_bins == bin_label].index
            n_bin = int(n_users * len(bin_users) / len(user_counts))
            sampled_users.extend(np.random.choice(bin_users, size=n_bin, replace=False))

        return df[df["user_id"].isin(sampled_users)]

    def _evaluateRecommendations(
        self, recommendations: dict[int, list[int]], ground_truth: dict[int, list[int]], k: int = 20
    ) -> dict[str, float]:
        """
        Evaluate recommendations using NDCG and Recall.

        :param recommendations: dictionary of recommendations
        :param ground_truth: dictionary of ground truth items
        :param k: number of recommendations to consider

        :return: evaluation metrics
        """
        return {
            "ndcg": get_ndcg(recommendations, ground_truth, k),
            "recall": get_calibrated_recall(recommendations, ground_truth, k),
        }

    def runEvaluation(
        self, recommender: Recommender, sample_size: int = None, n_trials: int = 3
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Run local evaluation of a recommender.

        :param recommender: recommender model
        :param sample_size: number of users to sample
        :param n_trials: number of trials to run

        :return: average and standard deviation of evaluation metrics
        """
        if not hasattr(self, "train_df"):
            self.load_data()

        self.logger.info(f"Starting evaluation with {n_trials} trials")
        if sample_size:
            self.logger.info(f"Using sampled dataset with {sample_size} users")

        metrics_list = []
        for trial in tqdm(range(n_trials), desc="Evaluation trials"):
            # Sample if requested
            if sample_size:
                train_data = self.createSampledDataset(self.train_df, sample_size, seed=42 + trial)
            else:
                train_data = self.train_df.copy()
            # Create splits
            train_split, val_split = self.createValidationSplit(train_data, seed=42 + trial)

            val_in, val_out = self.createFoldInSplit(val_split, seed=42 + trial)

            # Train and evaluate
            recommender.fit(train_split)
            if isinstance(recommender, PPR):
                recommendations = recommender.predict(val_in)
            else:
                users = val_in["user_id"].unique()
                recommendations = recommender.predict(users)

            # Convert val_out to ground truth format
            ground_truth = {
                user: val_out[val_out["user_id"] == user]["item_id"].tolist() for user in val_out["user_id"].unique()
            }

            metrics = self._evaluateRecommendations(recommendations, ground_truth)
            metrics_list.append(metrics)

        # Average metrics across trials
        avg_metrics = {metric: np.mean([m[metric] for m in metrics_list]) for metric in metrics_list[0].keys()}
        std_metrics = {metric: np.std([m[metric] for m in metrics_list]) for metric in metrics_list[0].keys()}

        return avg_metrics, std_metrics

    def createSubmission(self, recommender: Recommender, output_dir: str = "output/submissions") -> tuple[str, str]:
        """
        Train on full dataset and generate recommendations for test set.

        :param recommender: recommender model
        :param output_dir: output directory for submission files

        :return: paths to submission files
        """
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load data if not already loaded
        if not hasattr(self, "train_df"):
            self._loadData()

        self.logger.info("Starting submission creation")
        self.logger.info(f"Using recommender: {recommender.__class__.__name__}")

        start = time.time()
        # Train on full dataset
        recommender.fit(self.train_df)

        # Generate recommendations for test set
        if isinstance(recommender, PPR):
            recommendations = recommender.predict(self.test_in_df)
        else:
            users = self.test_in_df["user_id"].unique()
            recommendations = recommender.predict(users)

        execution_time = time.time() - start
        self.logger.info(f"Recommendations generated in {execution_time: .2f} seconds")

        # Create submission dataframe
        submission_df = pd.DataFrame(
            [
                {"user_id": user_id, "item_id": item_id}
                for user_id, items in recommendations.items()
                for item_id in items
            ]
        )

        # Save files
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"{output_dir}/submission_{recommender.__class__.__name__}_{timestamp}.csv"
        zip_path = f"{output_dir}/submission_{recommender.__class__.__name__}_{timestamp}.zip"

        submission_df.to_csv(csv_path, index=False)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(csv_path)

        self.logger.info(f"Saved submission files: \nCSV: {csv_path}\nZIP: {zip_path}")

        return csv_path, zip_path


if __name__ == "__main__":
    evaluator = Evaluator("data/raw/train_interactions.csv", "data/raw/test_interactions_in.csv")

    # Baseline models
    """
    from models import Random, Popularity, UserKNN, ItemKNN
    # Baseline models
    models = [
        # Random(random_state=42),
        Popularity(),
        # UserKNN(n_neighbors=20),
        ItemKNN(use_half_precision=True)
    ]

    for model in models:
        avg_metrics, std_metrics = evaluator.runEvaluation(
            model, sample_size=None, n_trials=1)
        print(f"{model.__class__.__name__}: {avg_metrics} {std_metrics}")
        # evaluator.createSubmission(model)
    """

    # PPR models
    """
    from models import PPR
    params = {
        "alpha": 0.0,
        "popularity_weight": 0.0,
        "num_iterations": 75,
        "interaction_weight_processing": "log"
    }
    evaluator = Evaluator(
        'data/raw/train_interactions.csv', 'data/raw/test_interactions_in.csv')
    avg_metrics, std_metrics = evaluator.runEvaluation(
        PPR(**params), sample_size=7500, n_trials=3)
    print(f"PPR: {avg_metrics} {std_metrics}")
    """

    """
    # Submission creation
    from models import PPR, MultiAlphaPPR, TwoPhasePPR
    params_ppr = {
        "alpha": 0.023158004249229626,
        "popularity_weight": 0.033105547887524185,
        "num_iterations": 93,
        "interaction_weight_processing": "log"
    }
    evaluator.createSubmission(PPR(**params_ppr))
    """
    params_mappr = {
        "alphas": [0.022625538386329365, 0.3015889812327148, 0.0],
        "alpha_weights": [1.0, 0.0, 0.0],
        "popularity_weight": 0.0,
        "num_iterations": 106,
        "interaction_weight_processing": "log",
    }
    from src.models import MultiAlphaPPR

    print(evaluator.runEvaluation(MultiAlphaPPR(**params_mappr), sample_size=5000, n_trials=3))
    """
    params_tppr = {
        "alpha1": 0.04052983557401711,
        "alpha2": 0.0,
        "stage1_k": 148,
        "popularity_weight": 0.0,
        "num_iterations": 271,
        "interaction_weight_processing": "log"
    }
    evaluator.createSubmission(TwoPhasePPR(**params_tppr))
    """
