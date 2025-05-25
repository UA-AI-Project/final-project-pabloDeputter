
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

from src.models.recommender import Recommender


class UserKNN(Recommender):
    """
    Recommends items to users based on the preferences of similar users.

    :param n_neighbors: the number of nearest neighbors to consider
    :param batch_size: the batch size to use for processing users in prediction
    """

    def __init__(self, n_neighbors: int = 20, batch_size: int = 1024):
        """
        Initializes the UserKNNRecommender.
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self._user_item_matrix = None
        self._user_similarity_matrix = None
        self._user_mapping = None
        self._item_mapping = None
        self._raw_user_ids = None
        self._raw_item_ids = None

    def fit(self, df: pd.DataFrame, user_id_column: str = "user_id", item_id_column: str = "item_id") -> None:
        """
        Fits the recommender by calculating user similarity.

        :param df: dataFrame containing user-item interactions
        :param user_id_column: the name of the column containing user IDs
        :param item_id_column: the name of the column containing item IDs
        """
        # Sanity checks
        if user_id_column not in df.columns:
            raise ValueError(f"User ID column '{user_id_column}' not found in input DataFrame.")
        if item_id_column not in df.columns:
            raise ValueError(f"Item ID column '{item_id_column}' not found in input DataFrame.")

        # Create mappings for user and item IDs to internal indices
        unique_users = df[user_id_column].unique()
        unique_items = df[item_id_column].unique()

        self._raw_user_ids = unique_users
        self._raw_item_ids = unique_items

        self._user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self._item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}

        user_indices = df[user_id_column].map(self._user_mapping)
        item_indices = df[item_id_column].map(self._item_mapping)

        self.logger.info("Creating user-item matrix")
        # Create user-item matrix
        self._user_item_matrix = csr_matrix(
            (np.ones(len(df)), (user_indices, item_indices)),
            shape=(len(unique_users), len(unique_items)),
        )

        # Normalize user-item matrix for cosine similarity
        normalized_matrix = normalize(self._user_item_matrix, axis=1)
        self._user_similarity_matrix = normalized_matrix.dot(normalized_matrix.T)

    def predict(self, users: list[int | str], n_items: int = 20) -> dict[int | str, list[int | str]]:
        """
        Generates item recommendations for a list of users.

        :param users: list of user IDs for whom to generate recommendations
        :param n_items: the number of items to recommend for each user
        :return: a dictionary where keys are user IDs and values are lists of recommended item IDs
        """
        assert self._user_item_matrix is not None, "The fit method must be called before predict."

        predictions: dict[int | str, list[int | str]] = {}
        reverse_item_mapping = {v: k for k, v in self._item_mapping.items()}

        for user_id in tqdm(users, total=len(users), desc="Generating recommendations"):
            if user_id not in self._user_mapping:
                # Cold-start user: recommend most popular items
                item_popularity = self._user_item_matrix.sum(axis=0).A1
                top_item_indices = np.argsort(item_popularity)[::-1][:n_items]
                recommended_items = [reverse_item_mapping[idx] for idx in top_item_indices]
                predictions[user_id] = recommended_items
                continue

            user_index = self._user_mapping[user_id]

            # Get similarity scores for the current user
            similarity_scores = self._user_similarity_matrix[user_index].A1

            # Get indices of top N similar users (excluding the user itself)
            similar_user_indices = np.argsort(similarity_scores)[::-1][1 : self.n_neighbors + 1]
            if not similar_user_indices.size:
                # If no similar users, recommend most popular
                item_popularity = self._user_item_matrix.sum(axis=0).A1
                top_item_indices = np.argsort(item_popularity)[::-1][:n_items]
                recommended_items = [reverse_item_mapping[idx] for idx in top_item_indices]
                predictions[user_id] = recommended_items
                continue

            # Get the item interaction vectors of similar users
            neighbor_interactions = self._user_item_matrix[similar_user_indices]

            # Calculate the weighted item scores
            weighted_scores = neighbor_interactions.T.dot(similarity_scores[similar_user_indices])

            # Get the indices of items already interacted with by the user
            interacted_items = self._user_item_matrix[user_index].indices

            # Filter out already interacted items
            weighted_scores[interacted_items] = -1

            # Get top N item indices
            top_item_indices = np.argsort(weighted_scores)[::-1][:n_items]

            # Map back to original item IDs
            recommended_items = [reverse_item_mapping[idx] for idx in top_item_indices]
            predictions[user_id] = recommended_items

        return predictions
