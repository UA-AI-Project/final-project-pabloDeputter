
import pandas as pd
from tqdm.auto import tqdm

from src.models.recommender import Recommender


class Popularity(Recommender):
    """
    Recommends the most popular items based on the training data.
    """

    def __init__(self):
        super().__init__()
        self._item_popularity = None

    def fit(self, df: pd.DataFrame, item_id_column: str = "item_id"):
        """
        Calculates item popularity from the training data.

        :param df: dataFrame containing user-item interactions
        :param item_id_column: the name of the column containing item IDs
        """
        # Sanity check: ensure that required columns exist in the DataFrame
        if item_id_column not in df.columns:
            raise ValueError(f"Item ID column '{item_id_column}' not found in input DataFrame.")

        self._item_popularity = df[item_id_column].value_counts().sort_values(ascending=False)
        assert not self._item_popularity.empty, "No items found in the training data."

    def predict(self, users: list[int | str], n_items: int = 20) -> dict[int | str, list[int | str]]:
        """
        Predicts the top N popular items for a list of user IDs.

        :param users: a list of user IDs for whom to generate recommendations.
        :param n_items: the number of top popular items to recommend for each user.
        :return: a dictionary where keys are user IDs and values are lists of the top N popular item IDs.
        """
        assert self._item_popularity is not None, "The fit method must be called before predict."

        top_items = self._item_popularity.index[:n_items].tolist()
        predictions: dict[int | str, list[int | str]] = {}

        for user_id in tqdm(users, total=len(users), desc="Generating recommendations"):
            predictions[user_id] = top_items

        return predictions
