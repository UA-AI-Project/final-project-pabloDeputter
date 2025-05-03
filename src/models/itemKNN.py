import pandas as pd
import numpy as np
import torch
import gc

from torch.amp import autocast
from tqdm import tqdm
from typing import List, Dict, Tuple, Union

from .recommender import Recommender


class ItemKNN(Recommender):
    """
    Recommends items to users based on the similarity between items.
    Uses efficient batched operations and GPU acceleration when available.

    :param batch_size: the number of items to process in each batch for memory efficiency
    :param use_half_precision: whether to use FP16 for computations to reduce memory usage
    """

    def __init__(self, batch_size: int = 1024, use_half_precision: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.use_half_precision = use_half_precision
        self._item_similarity_matrix = None
        self._user_item_matrix = None
        self._popular_items = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def __del__(self):
        """
        Cleanup when object is deleted.
        """
        self._clearGPUMemory()

    def _clearGPUMemory(self):
        """
        Clears GPU memory and performs garbage collection.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _createUserItemMatrix(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Creates the user-item interaction matrix with memory optimization.

        :param df: DataFrame containing user-item interactions
        :return: sparse tensor containing user-item interactions
        """
        self.logger.info("Creating user-item matrix...")

        # Map indices
        user_indices = df['user_id'].map(self._user_mapping).values
        item_indices = df['item_id'].map(self._item_mapping).values

        # Create sparse interaction matrix on GPU
        dtype = torch.float16 if self.use_half_precision else torch.float32
        matrix = torch.zeros(
            (len(self._raw_user_ids), len(self._raw_item_ids)),
            device=self.device,
            dtype=dtype
        )

        # Use binary interactions for simplicity and memory efficiency
        with autocast(device_type='cuda', enabled=self.use_half_precision):
            matrix[user_indices, item_indices] = 1

        if torch.cuda.is_available():
            self.logger.info(f"GPU Memory after matrix creation: {
                             torch.cuda.memory_allocated() / 1e9:.2f} GB")

        return matrix

    def _getSimilarItems(self, item_id: Union[int, str], n: int = 10) -> List[Tuple[Union[int, str], float]]:
        """
        Gets the most similar items for a given item.

        :param item_id: ID of the item to find similar items for
        :param n: number of similar items to return
        :return: list of tuples containing (item_id, similarity_score)
        """
        if item_id not in self._item_mapping:
            return []

        item_idx = self._item_mapping[item_id]

        with autocast(device_type='cuda', enabled=self.use_half_precision):
            similarities = self._item_similarity_matrix[item_idx]
            top_values, top_indices = torch.topk(similarities, k=n+1)

        # Convert to CPU and remove the item itself
        top_values = top_values.cpu().numpy()[1:]
        top_indices = top_indices.cpu().numpy()[1:]

        return [(self._raw_item_ids[idx], sim)
                for idx, sim in zip(top_indices, top_values)]

    def _computeItemSimilarity(self) -> None:
        """
        Computes the item-item similarity matrix using cosine similarity with batched operations.
        """
        self.logger.info("Computing item similarities...")
        n_items = self._user_item_matrix.shape[1]

        # Adjust batch size based on available memory
        batch_size = min(self.batch_size, n_items // 4)
        self.logger.info(f"Using batch size: {batch_size}")

        # Initialize similarity matrix
        dtype = torch.float16 if self.use_half_precision else torch.float32
        self._item_similarity_matrix = torch.zeros(
            (n_items, n_items),
            device=self.device,
            dtype=dtype
        )

        # Compute similarities in batches
        with autocast(device_type='cuda', enabled=self.use_half_precision):
            for i in tqdm(range(0, n_items, batch_size), desc="Computing Item Similarities"):
                end_idx = min(i + batch_size, n_items)
                batch = self._user_item_matrix[:, i:end_idx].T

                # Compute cosine similarity for this batch
                # sim = (A * B) / (||A|| * ||B||)
                norms = torch.norm(batch, dim=1).unsqueeze(1)
                all_norms = torch.norm(self._user_item_matrix, dim=0)

                sim = torch.mm(batch, self._user_item_matrix)
                sim = sim / (norms * all_norms.unsqueeze(0) + 1e-8)

                self._item_similarity_matrix[i:end_idx] = sim

                # Clear cache periodically
                if i % (batch_size * 4) == 0:
                    self._clearGPUMemory()

    def _computePopularItems(self) -> List[Union[int, str]]:
        """
        Computes list of popular items for cold-start recommendations.

        :return: list of item IDs sorted by popularity
        """
        self.logger.info("Computing popular items...")
        with autocast(device_type='cuda', enabled=self.use_half_precision):
            item_counts = torch.sum(self._user_item_matrix, dim=0)
            popular_items_idx = torch.argsort(item_counts, descending=True)

        return [self._raw_item_ids[idx.item()]
                for idx in popular_items_idx.cpu()]

    def fit(self, df: pd.DataFrame, user_id_column: str = 'user_id', item_id_column: str = 'item_id') -> None:
        """
        Fits the recommender by calculating item similarity with memory optimizations.

        :param df: DataFrame containing user-item interactions
        :param user_id_column: the name of the column containing user IDs
        :param item_id_column: the name of the column containing item IDs
        """
        # Sanity checks
        if user_id_column not in df.columns:
            raise ValueError(f"User ID column '{
                             user_id_column}' not found in input DataFrame.")
        if item_id_column not in df.columns:
            raise ValueError(f"Item ID column '{
                             item_id_column}' not found in input DataFrame.")

        # Create mappings for user and item IDs to internal indices
        self._raw_user_ids = df[user_id_column].unique()
        self._raw_item_ids = df[item_id_column].unique()

        self._item_mapping = {item_id: idx for idx,
                              item_id in enumerate(self._raw_item_ids)}
        self._user_mapping = {user_id: idx for idx,
                              user_id in enumerate(self._raw_user_ids)}

        # Create interaction matrix
        self._user_item_matrix = self._createUserItemMatrix(df)

        # Compute item similarities
        self._computeItemSimilarity()

        # Compute popular items for cold-start
        self._popular_items = self._computePopularItems()

        # Clear memory
        self._clearGPUMemory()

    def predict(self, users: List[Union[int, str]], n_items: int = 20) -> Dict[Union[int, str], List[Union[int, str]]]:
        """
        Generates item recommendations for a list of users with efficient batching.

        :param users: list of user IDs for whom to generate recommendations
        :param n_items: the number of items to recommend for each user
        :return: a dictionary where keys are user IDs and values are lists of recommended item IDs
        """
        assert self._user_item_matrix is not None, "The fit method must be called before predict."
        predictions = {}

        # Process users in batches
        batch_size = min(self.batch_size, len(users))

        for i in tqdm(range(0, len(users), batch_size), desc="Generating recommendations"):
            batch_users = users[i:i + batch_size]
            batch_predictions = {}

            with autocast(device_type='cuda', enabled=self.use_half_precision):
                for user_id in batch_users:
                    if user_id not in self._user_mapping:
                        # Cold start: use popular items
                        batch_predictions[user_id] = self._popular_items[:n_items]
                        continue

                    user_idx = self._user_mapping[user_id]
                    user_items = self._user_item_matrix[user_idx].nonzero().squeeze(
                        1)

                    if len(user_items) == 0:
                        # No interactions: use popular items
                        batch_predictions[user_id] = self._popular_items[:n_items]
                        continue

                    # Compute item scores using item similarities
                    item_scores = torch.zeros(
                        len(self._raw_item_ids),
                        device=self.device,
                        dtype=self._item_similarity_matrix.dtype
                    )

                    # Sum similarities from interacted items
                    for item_idx in user_items:
                        item_scores += self._item_similarity_matrix[item_idx]

                    # Zero out items the user has already interacted with
                    item_scores[user_items] = -float('inf')

                    # Get top items
                    _, top_indices = torch.topk(item_scores, k=n_items)
                    top_items = [self._raw_item_ids[idx.item()]
                                 for idx in top_indices.cpu()]

                    batch_predictions[user_id] = top_items

            predictions.update(batch_predictions)

            # Clear cache periodically
            if i % (batch_size * 4) == 0:
                self._clearGPUMemory()

        return predictions
