import numpy as np
import pandas as pd

from typing import List, Dict, Union
from tqdm.auto import tqdm

from .recommender import Recommender


class Random(Recommender):
    """
    Recommender that predicts random items for each user.

    :param random_state: random seed for reproducibility
    """

    def __init__(self, random_state: int = 42):
        """
        Initializes the RandomRecommender.
        """
        super().__init__()
        self.random_state = random_state
        # Use a numpy RandomState for better control over randomness
        self._rng = np.random.RandomState(random_state)
        self._items = None

    def fit(self, df: pd.DataFrame, item_id_column: str = 'item_id'):
        """
        Fits the recommender by learning the set of unique item IDs from the input DataFrame.

        :param df: dataFrame containing user-item interactions
        :param item_id_column: the name of the column containing item IDs
        """
        # Sanity check: ensure that required columns exist in the DataFrame
        if item_id_column not in df.columns:
            raise ValueError(
                f"Item ID column '{item_id_column}' not found in input DataFrame.")

        self._items = df[item_id_column].unique()
        assert len(self._items) > 0, "No items found in the training data."

    def predict(self, users: List[Union[int, str]], n_items: int = 20) -> Dict[Union[int, str], List[Union[int, str]]]:
        """
        Predicts random items for each user in the test DataFrame.

        :param users: a list of user IDs for whom to generate recommendations.
        :param n_items: the number of items to recommend for each user

        :return: a dictionary where keys are user IDs and values are lists of recommended item IDs.
        """
        # Sanity checks
        assert self._items is not None, "The fit method must be called before predict."

        predictions: Dict[Union[int, str], List[Union[int, str]]] = {}

        for user_id in tqdm(users, total=len(users), desc="Generating recommendations"):
            # Randomly sample n_items from the available items
            if len(self._items) >= n_items:
                recommended_items = self._rng.choice(
                    self._items, size=n_items, replace=False).tolist()
            else:
                # If there are fewer items than n_items, return all available items
                recommended_items = self.rng.choice(
                    self._items, size=len(self._items), replace=False).tolist()

            predictions[user_id] = recommended_items

        return predictions
