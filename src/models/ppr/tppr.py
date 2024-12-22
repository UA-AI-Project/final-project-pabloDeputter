import torch

from typing import Optional
from .ppr import PPR


class TwoPhasePPR(PPR):
    """
    Two-Phase Personalized PageRank (PPR) recommendation model.

    This variant performs PPR in two phases:
    1. Broad exploration phase with lower alpha to discover potential candidates
    2. Focused exploration phase with higher alpha on the top candidates from phase 1

    This approach helps balance between exploration and exploitation in the recommendation process.

    :param alpha1: Alpha value for first phase (broad exploration)
    :param alpha2: Alpha value for second phase (focused exploration)
    :param stage1_k: Number of top items to consider from first stage
    :param num_iterations: Number of power iterations for PPR calculation
    :param popularity_weight: Base weight given to item popularity when combining with PPR scores
    :param interaction_weight_processing: Method for processing interaction weights
    """

    def __init__(
            self,
            alpha1: float = 0.3,
            alpha2: float = 0.7,
            stage1_k: int = 100,
            num_iterations: int = 100,
            popularity_weight: float = 0.2,
            interaction_weight_processing: Optional[str] = 'log',
            batch_size: int = 1024):

        # Validate parameters
        if not 0 <= alpha1 <= 1:
            raise ValueError(f"alpha1 must be between 0 and 1, got {alpha1}")
        if not 0 <= alpha2 <= 1:
            raise ValueError(f"alpha2 must be between 0 and 1, got {alpha2}")
        if stage1_k <= 0:
            raise ValueError(f"stage1_k must be positive, got {stage1_k}")

        # Initialize base PPR with dummy alpha (will be overridden in _calculatePPR)
        super().__init__(
            alpha=alpha1,  # Temporary alpha, not actually used
            num_iterations=num_iterations,
            popularity_weight=popularity_weight,
            interaction_weight_processing=interaction_weight_processing,
            batch_size=batch_size
        )

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.stage1_k = stage1_k

    def _calculatePPR(self, personalization_vector: torch.Tensor) -> torch.Tensor:
        """
        Calculate PPR scores using two-phase approach.

        Phase 1: Broad exploration with lower alpha
        Phase 2: Focused exploration on top-k items with higher alpha

        :param personalization_vector: Vector representing user's historical interactions
        :return: Combined PPR scores from both phases
        """
        # Phase 1: Broad exploration with lower alpha
        ppr1 = personalization_vector.clone()

        for _ in range(self.num_iterations):
            ppr1_next = self.alpha1 * (ppr1 @ self._iu_matrix @ self._ui_matrix) + \
                (1 - self.alpha1) * personalization_vector

            if torch.allclose(ppr1, ppr1_next, atol=1e-7):
                self.logger.info(f"Converged after {_ + 1} iterations")
                break

            ppr1 = ppr1_next

        # Get top-k items from first phase
        _, top_k_indices = torch.topk(
            ppr1,
            min(self.stage1_k, ppr1.size(-1)),
            dim=-1
        )

        # Create new personalization vector for phase 2
        personalization2 = torch.zeros_like(personalization_vector)
        personalization2.scatter_(-1, top_k_indices, 1.0 / self.stage1_k)

        # Phase 2: Focused exploration with higher alpha
        ppr2 = personalization2.clone()

        for _ in range(self.num_iterations):
            ppr2_next = self.alpha2 * (ppr2 @ self._iu_matrix @ self._ui_matrix) + \
                (1 - self.alpha2) * personalization2

            if torch.allclose(ppr2, ppr2_next, atol=1e-7):
                self.logger.info(f"Converged after {_ + 1} iterations")
                break

            ppr2 = ppr2_next

        # Combine results from both phases (average)
        final_ppr = (ppr1 + ppr2) / 2

        # Sanity check: ensure output dimensions match input
        assert final_ppr.shape == personalization_vector.shape, \
            f"PPR shape mismatch: expected {
                personalization_vector.shape}, got {final_ppr.shape}"

        return final_ppr
