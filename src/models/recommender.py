import pandas as pd
import logging

from abc import ABC, abstractmethod
from typing import List, Dict, Union


class Recommender(ABC):
    """
    Abstract base class for all recommendation models.

    Provides a common interface for fitting the model and making predictions.
    All concrete recommender models should inherit from this class and implement
    the `fit` and `predict` methods.
    """

    def __init__(self):
        # Use the actual class name when creating the logger
        self.logger = logging.getLogger(self.__class__.__name__)
        # Avoid adding handlers multiple times
        if not self.logger.handlers:
            self._setupLogger()

    def _setupLogger(self):
        """
        Set up the logger for the recommender model.

        This method configures the logger to output messages to the console.
        """
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs) -> None:
        """
        Fits the recommendation model on the provided data.

        This method should be implemented by subclasses to train the model
        using the provided DataFrame.

        :param df: DataFrame containing the training data.
        :param kwargs: Additional keyword arguments specific to the model.
        """
        pass

    @abstractmethod
    def predict(self, users: List[Union[int, str]], n_items: int = 20, **kwargs) -> Dict[Union[int, str], List[Union[int, str]]]:
        """
        Generates item recommendations for the specified users.

        This method should be implemented by subclasses to return a dictionary
        of recommended items for each user.

        :param users: List of user IDs for whom to generate recommendations.
        :param n_items: The number of items to recommend for each user.
        :param kwargs: Additional keyword arguments specific to the model.
        :return: A dictionary where keys are user IDs and values are lists of recommended item IDs.
        """
        pass
