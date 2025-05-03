import datetime
import json
import logging
import numpy as np
import pandas as pd
import time

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real

from models import Recommender
from evaluator import Evaluator


@dataclass
class OptimizationConfig:
    """
    Configuration class for optimization process.

    :param n_trials: number of optimization trials
    :param n_random_starts: number of random starts
    :param sample_size: size of training data sample
    :param n_cv_folds: number of cross-validation folds
    :param val_size: validation size for each fold
    :param base_seed: base seed for reproducibility
    :param metric_weights: weights for each metric to calculate the final score
    """
    n_trials: int = 100
    n_random_starts: int = 10
    sample_size: Optional[int] = 10000
    n_cv_folds: int = 3
    val_size: float = 0.2
    base_seed: int = 42
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        'ndcg': 0.85,
        'recall': 0.15
    })


class BayesianOptimizer:
    """
    Bayesian optimization class for hyperparameter tuning.

    :param recommender: recommender class to optimize
    :param evaluator: evaluation framework instance
    :param param_space: parameter space for optimization
    :param config: optimization configuration
    """

    def __init__(
        self,
        recommender: Recommender,
        evaluator: Evaluator,
        param_space: Dict[str, Union[Tuple, List]],
        output_dir: Optional[str] = '../output/optimization',
        config: Optional[OptimizationConfig] = None,
    ):
        self.recommender = recommender
        self.evaluator = evaluator
        self.param_space = param_space
        self.output_dir = Path(output_dir)
        self.config = config or OptimizationConfig()

        # Setup experiment tracking
        self.experiment_name = f"{recommender.__name__}_{
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize directories
        self._setupDirectories()
        # Setup logging
        self._setupLogger()
        # Initialize random number generators
        self._setupRandomState()
        # Initialize results tracking
        self._setupResultsTracking()
        # Setup optimization space
        self._setupDimensions()

    def _setupDirectories(self):
        """
        Create directories for storing optimization results.

        - checkpoints: intermediate results
        - visualizations: plots and visualizations
        - logs: log files
        - results: final results
        """
        self.output_dir = self.output_dir / self.experiment_name

        # Create subdirectories for different types of outputs
        self.dirs = {
            'checkpoints': self.output_dir / 'checkpoints',
            'visualizations': self.output_dir / 'visualizations',
            'logs': self.output_dir / 'logs',
            'results': self.output_dir / 'results'
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def _setupLogger(self):
        """
        Set up the logger for the optimization process.

        This method configures the logger to output messages to the console
        """
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(self.dirs['logs'] / 'optimization.log')
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _setupRandomState(self):
        """
        Initialize random number generators with base seed.

        Seeds are generated for sampling, cross-validation, and optimization.
        Each seed is a 32-bit integer.
        """
        self.rng = np.random.RandomState(self.config.base_seed)
        self.seeds = {
            'sampling': self.rng.randint(0, 2**32 - 1),
            'cv': self.rng.randint(0, 2**32 - 1, size=self.config.n_cv_folds * 2),
            'optimization': self.rng.randint(0, 2**32 - 1)
        }

    def _setupResultsTracking(self):
        """
        Initialize results tracking variables.
        """
        self.results = []
        self.best_score = float('-inf')
        self.best_params = None
        self.iteration_times = []

    def _setupDimensions(self):
        """
        Setup optimization dimensions based on parameter space.
        Skopt dimensions are created based on the parameter space dictionary.
        """
        self.dimensions = []

        # Check if we're dealing with MultiAlphaPPR case
        if 'alphas' in self.param_space:
            # Handle MultiAlphaPPR case
            for i, (lower, upper) in enumerate(self.param_space['alphas']):
                self.dimensions.append(Real(lower, upper, name=f"alpha_{i}"))

            # Create dimensions for weights
            n_alphas = len(self.param_space['alphas'])
            for i in range(n_alphas):
                self.dimensions.append(Real(0.0, 1.0, name=f"weight_{i}"))

        # Handle all other parameters
        for name, spec in self.param_space.items():
            if name != 'alphas':  # Skip already processed alphas
                if isinstance(spec, tuple):
                    if isinstance(spec[0], int):
                        self.dimensions.append(
                            Integer(spec[0], spec[1], name=name))
                    else:
                        self.dimensions.append(
                            Real(spec[0], spec[1], name=name))
                elif isinstance(spec, list) and isinstance(spec[0], str):
                    self.dimensions.append(Categorical(spec, name=name))

    def _formatParams(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format parameters for the model to use.

        This method formats the parameters dictionary to ensure that the model
        receives the correct types and values for each parameter.

        :param params: dictionary of parameters

        :return: formatted dictionary of parameters
        """
        formatted = {}

        # Check if we're dealing with MultiAlphaPPR case (looking for alpha_0, alpha_1, etc.)
        alpha_values = [(int(key.split('_')[1]), value) for key, value in params.items()
                        if key.startswith('alpha_') and key[6:].isdigit()]

        weight_values = [(int(key.split('_')[1]), value) for key, value in params.items()
                         if key.startswith('weight_') and key[7:].isdigit()]

        if alpha_values and weight_values:
            # MultiAlphaPPR case
            alpha_values.sort(key=lambda x: x[0])
            weight_values.sort(key=lambda x: x[0])

            formatted['alphas'] = [v[1] for v in alpha_values]
            weights = np.array([v[1] for v in weight_values])
            formatted['alpha_weights'] = list(weights / weights.sum())

        # Handle all other parameters
        for key, value in params.items():
            if not (key.startswith('alpha_') and key[6:].isdigit()) and not (key.startswith('weight_') and key[7:].isdigit()):
                if isinstance(value, list) and len(value) == 1:
                    formatted[key] = value[0]
                else:
                    formatted[key] = value

        return formatted

    def _objective(self, x: List[float]) -> float:
        """
        Objective function to minimize.

        This function is called by the optimizer to evaluate the model with
        the provided hyperparameters. It trains the model and evaluates it
        using the evaluation framework. The final score is a weighted sum a set
        of metrics. Cross-validation is used to evaluate the model.

        :param x: hyperparameters to evaluate
        """
        start_time = time.time()
        iteration = len(self.results)

        try:
            # Convert hyperparameters to dictionary
            params = dict(zip([dim.name for dim in self.dimensions], x))
            # Format parameters
            formatted_params = self._formatParams(params)

            # Sanity check to make sure that all weights sum to 1
            if 'alpha_weights' in formatted_params:
                weight_sum = sum(formatted_params['alpha_weights'])
                assert np.isclose(weight_sum, 1.0), f"Weight sum is not 1: {
                    weight_sum}"

            self.logger.info(f"Starting iteration {
                             iteration} with params: {formatted_params}")

            fold_scores = []
            fold_metrics = []

            # Create sample if using
            if self.config.sample_size:
                # Use sampling seed to ensure fairness across iterations
                train_data = self.evaluator.createSampledDataset(
                    self.evaluator.train_df,
                    self.config.sample_size,
                    seed=self.seeds['sampling']
                )
            else:
                train_data = self.evaluator.train_df.copy()

            # Run cross-validation
            for fold in range(self.config.n_cv_folds):
                # Get seeds for this fold
                split_seed = self.seeds['cv'][fold * 2]
                fold_seed = self.seeds['cv'][fold * 2 + 1]

                # Create splits
                train_split, val_split = self.evaluator.createValidationSplit(
                    train_data,
                    val_size=self.config.val_size,
                    seed=split_seed
                )

                val_in, val_out = self.evaluator.createFoldInSplit(
                    val_split,
                    seed=fold_seed
                )

                self.logger.info(
                    f"Fold {fold + 1}: Training with params {formatted_params}")

                # Train and evaluate
                recommender = self.recommender(**formatted_params)
                recommender.fit(train_split)
                recommendations = recommender.predict(val_in)

                # Evaluate
                metrics = self.evaluator._evaluateRecommendations(
                    recommendations,
                    {
                        user: val_out[val_out['user_id'] == user]['item_id'].tolist() for user in val_out['user_id'].unique()
                    }

                )

                # Calculate combined score for this fold using weights from config
                fold_score = sum(
                    weight * metrics[metric]
                    for metric, weight in self.config.metric_weights.items()
                )

                self.logger.info(
                    f"Fold {fold + 1} score: {fold_score:.4f} with metrics: {metrics}")
                fold_scores.append(fold_score)
                fold_metrics.append(metrics)

            # Calculate average metrics across folds
            avg_score = np.mean(fold_scores)
            score_std = np.std(fold_scores)

            avg_metrics = {
                'ndcg': np.mean([m['ndcg'] for m in fold_metrics]),
                'recall': np.mean([m['recall'] for m in fold_metrics]),
                'ndcg_std': np.std([m['ndcg'] for m in fold_metrics]),
                'recall_std': np.std([m['recall'] for m in fold_metrics])
            }

            # Store results
            iteration_time = time.time() - start_time
            self.iteration_times.append(iteration_time)

            self.logger.info(f"Iteration {iteration} finished in {
                             iteration_time:.2f} seconds with score: {avg_score:.4f} and metrics: {avg_metrics}")

            self.results.append({
                'params': formatted_params,
                'score': avg_score,
                'score_std': score_std,
                'iteration_time': iteration_time,
                **avg_metrics
            })

            # Update best score if this is better
            if avg_score > self.best_score:
                self.best_score = avg_score
                self.best_params = formatted_params
                self.logger.info(f"New best score: {
                                 avg_score:.4f} with params: {formatted_params}")

            return -avg_score  # Minimize negative score

        except Exception as e:
            self.logger.error(f"Error in iteration {iteration}: {str(e)}")
            return float('inf')

    def _convertToNative(self, obj: Any) -> Any:
        """
        Convert numpy types to native Python types.

        :param obj: object to convert

        :return: converted object
        """
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convertToNative(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convertToNative(value) for value in obj]
        return obj

    def _checkpointCallback(self, res: Any):
        """
        Callback function to save intermediate results.

        This function is called by the optimizer after each iteration to save
        the intermediate results to disk.

        :param res: optimization result object
        """

        iteration = len(res.x_iters)
        results_df = pd.DataFrame(self.results)

        # Save results CSV
        results_df.to_csv(
            self.dirs['checkpoints'] / f"results_iter_{iteration}.csv",
            index=False
        )

        # Save optimization state
        with open(self.dirs['checkpoints'] / f"opt_state_{iteration}.json", 'w') as f:
            json.dump({
                'best_score': float(self.best_score),
                'best_params': self._convertToNative(self.best_params),
                'n_iterations': iteration,
                'total_time': sum(self.iteration_times)
            }, f, indent=4)

    def _saveResults(self):
        """
        Save final optimization results.
        """
        # Save all results to CSV
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.dirs['results'] /
                          "final_results.csv", index=False)

        # Save best parameters
        with open(self.output_dir / "best_params.json", 'w') as f:
            json.dump({
                'best_params': self._convertToNative(self.best_params),
                'best_score': float(self.best_score)
            }, f, indent=4)

        self.logger.info(f"Results saved to {self.dirs['results']}")

    def _preprocessResults(self):
        """
        Preprocess results into a DataFrame.

        :return: preprocessed results DataFrame
        """
        results_df = pd.DataFrame(self.results)

        # Extract parameters from dictionary string
        if 'params' in results_df.columns:
            # If params is stored as string, evaluate it
            if isinstance(results_df['params'].iloc[0], str):
                results_df['params'] = results_df['params'].apply(eval)

            # Extract parameters into separate columns
            param_df = pd.json_normalize(results_df['params'])

            # Combine with original results
            results_df = pd.concat(
                [param_df, results_df.drop('params', axis=1)], axis=1)

        return results_df

    def _plotParameterPerformance(self, results_df: pd.DataFrame):
        """
        Create parameter performance plots with enhanced visualization for vector parameters.

        :param results_df: DataFrame of optimization results
        """
        # Get parameter info from param_space
        vector_params = {k: v for k, v in self.param_space.items()
                         if isinstance(v, list) and isinstance(v[0], tuple)}
        scalar_params = {k: v for k, v in self.param_space.items()
                         if not isinstance(v, list) or not isinstance(v[0], tuple)}

        # Process vector parameters
        for param_name, bounds in vector_params.items():
            n_components = len(bounds)

            # Create subplot grid based on number of components
            n_rows = (n_components + 2) // 3  # Ceil division
            fig = plt.figure(figsize=(18, 6 * n_rows))
            gs = plt.GridSpec(n_rows, 3, figure=fig)

            fig.suptitle(f'Parameter Performance Analysis: {
                         param_name}', fontsize=14)

            # Extract component values if they exist in results
            if param_name in results_df.columns:
                component_values = {
                    i: results_df[param_name].apply(lambda x: x[i])
                    for i in range(n_components)
                }

                # Plot each component
                for comp_idx in range(n_components):
                    ax = fig.add_subplot(gs[comp_idx // 3, comp_idx % 3])

                    for metric, style in [
                        ('score', {'color': 'purple', 'alpha': 0.5}),
                        ('ndcg', {'color': 'blue', 'alpha': 0.3}),
                        ('recall', {'color': 'green', 'alpha': 0.3})
                    ]:
                        ax.scatter(
                            component_values[comp_idx],
                            results_df[metric],
                            label=metric.upper(),
                            **style
                        )

                    # Add best parameter line if available
                    if hasattr(self, 'best_params') and param_name in self.best_params:
                        best_value = self.best_params[param_name][comp_idx]
                        # Format best value based on type
                        label_value = f'{best_value:.3f}' if isinstance(
                            best_value, (int, float)) else str(best_value)
                        ax.axvline(x=best_value, color='red', linestyle='--',
                                   label=f'Best: {label_value}')

                    ax.set_xlabel(f'{param_name}[{comp_idx}]')
                    ax.set_ylabel('Metric Value')
                    ax.grid(True, alpha=0.3)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.savefig(
                self.dirs['visualizations'] /
                f"parameter_analysis_{param_name}.png",
                bbox_inches='tight',
                dpi=300
            )
            plt.close()

        # Process scalar parameters
        for param_name in scalar_params:
            if param_name not in results_df.columns:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'Parameter Performance Analysis: {
                         param_name}', fontsize=14)

            for idx, (metric, weight, title) in enumerate([
                ('score', '', 'Weighted Score'),
                ('ndcg', 'ndcg_std', 'NDCG@20'),
                ('recall', 'recall_std', 'Recall@20')
            ]):
                ax = axes[idx]

                if weight and weight in results_df.columns:
                    ax.errorbar(
                        results_df[param_name],
                        results_df[metric],
                        yerr=results_df[weight],
                        fmt='o',
                        alpha=0.5,
                        label='Evaluations'
                    )
                else:
                    ax.scatter(
                        results_df[param_name],
                        results_df[metric],
                        alpha=0.5,
                        label='Evaluations'
                    )

                if hasattr(self, 'best_params') and param_name in self.best_params:
                    best_value = self.best_params[param_name]
                    # Format best value based on type
                    label_value = f'{best_value:.3f}' if isinstance(
                        best_value, (int, float)) else str(best_value)
                    ax.axvline(
                        x=best_value,
                        color='red',
                        linestyle='--',
                        label=f'Best: {label_value}'
                    )

                ax.set_xlabel(param_name)
                ax.set_ylabel(title)
                ax.grid(True, alpha=0.3)
                ax.legend()

            plt.tight_layout()
            plt.savefig(
                self.dirs['visualizations'] /
                f"parameter_analysis_{param_name}.png",
                bbox_inches='tight',
                dpi=300
            )
            plt.close()

    def _plotParameterConvergence(self, results_df: pd.DataFrame):
        """
        Visualize the convergence of all parameters during optimization with scale handling.

        Creates separate subplots for:
        1. Parameters with values < 1 (e.g., weights, alphas)
        2. Parameters with larger values (e.g., num_iterations)
        3. Categorical parameters (e.g., process_playtime)

        :param results_df: DataFrame containing optimization results with parameter values
                        for each iteration
        """
        # Separate vector and scalar parameters
        vector_params = {k: v for k, v in self.param_space.items()
                         if isinstance(v, list) and isinstance(v[0], tuple)}
        scalar_params = {k: v for k, v in self.param_space.items()
                         if not isinstance(v, list) or not isinstance(v[0], tuple)}

        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Function to get parameter type and max value
        def get_param_info(param_name, values):
            if isinstance(values.iloc[0], (str, list)):
                if isinstance(values.iloc[0], list):
                    try:
                        # Try to get max value for numeric lists
                        max_val = max(max(float(x) if isinstance(x, (int, float)) else -1)
                                      for x in values)
                        return 'numeric', max_val
                    except (ValueError, TypeError):
                        return 'categorical', None
                return 'categorical', None
            try:
                return 'numeric', float(max(values))
            except (ValueError, TypeError):
                return 'categorical', None

        # Track parameters for each axis
        small_scale_params = []
        large_scale_params = []
        categorical_params = []

        # Plot vector parameters
        for param_name, bounds in vector_params.items():
            if param_name in results_df.columns:
                for i in range(len(bounds)):
                    values = results_df[param_name].apply(lambda x: x[i])
                    param_type, max_value = get_param_info(param_name, values)

                    if param_type == 'categorical':
                        target_ax = ax3
                        target_list = categorical_params
                    else:
                        target_ax = ax1 if max_value <= 1 else ax2
                        target_list = small_scale_params if max_value <= 1 else large_scale_params

                    target_ax.plot(
                        results_df.index,
                        values,
                        label=f'{param_name}[{i}]',
                        marker='.',
                        markersize=8
                    )
                    target_list.append(f'{param_name}[{i}]')

        # Plot scalar parameters
        for param_name in scalar_params:
            if param_name in results_df.columns:
                param_type, max_value = get_param_info(
                    param_name, results_df[param_name])

                if param_type == 'categorical':
                    target_ax = ax3
                    target_list = categorical_params
                else:
                    target_ax = ax1 if max_value <= 1 else ax2
                    target_list = small_scale_params if max_value <= 1 else large_scale_params

                target_ax.plot(
                    results_df.index,
                    results_df[param_name],
                    label=param_name,
                    marker='.',
                    markersize=8
                )
                target_list.append(param_name)

        # Configure subplots
        if small_scale_params:
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Parameter Value')
            ax1.set_title('Small-scale Parameters (≤ 1)')
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax1.set_visible(False)

        if large_scale_params:
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Parameter Value')
            ax2.set_title('Large-scale Parameters (> 1)')
            ax2.grid(True, alpha=0.3)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax2.set_visible(False)

        if categorical_params:
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Parameter Value')
            ax3.set_title('Categorical Parameters')
            ax3.grid(True, alpha=0.3)
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax3.set_visible(False)

        # Add parameter lists to figure title
        plt.suptitle(
            'Parameter Convergence During Optimization\n', fontsize=16)

        plt.tight_layout()
        plt.savefig(
            self.dirs['visualizations'] / "parameter_convergence.png",
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    def _plotMetricProgression(self, results_df: pd.DataFrame):
        """
        Create metric progression plot with error bars if available.

        :param results_df: DataFrame of optimization results
        """
        plt.figure(figsize=(12, 6))

        # Plot each metric with error bars if available
        metrics = [('ndcg', 'ndcg_std'), ('recall', 'recall_std'),
                   ('score', 'score_std')]
        colors = ['blue', 'green', 'red']

        for (metric, std_col), color in zip(metrics, colors):
            if std_col in results_df.columns:
                plt.errorbar(
                    results_df.index,
                    results_df[metric],
                    yerr=results_df[std_col],
                    label=metric.upper(),
                    color=color,
                    marker='o',
                    alpha=0.7
                )
            else:
                plt.plot(
                    results_df.index,
                    results_df[metric],
                    label=metric.upper(),
                    color=color,
                    marker='o'
                )

        plt.xlabel('Iteration')
        plt.ylabel('Metric Value')
        plt.title('Metric Progression Over Iterations')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            self.dirs['visualizations'] / "metric_progression.png",
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    def _plotScoreDistribution(self, results_df: pd.DataFrame):
        """
        Create score distribution plot.

        :param results_df: DataFrame of optimization results
        """
        plt.figure(figsize=(10, 6))

        plt.hist(
            results_df['score'],
            bins=30,
            density=True,
            alpha=0.7,
            color='blue'
        )

        plt.axvline(
            x=self.best_score,
            color='red',
            linestyle='--',
            label=f'Best Score: {self.best_score:.4f}'
        )

        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.title('Score Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            self.dirs['visualizations'] / "score_distribution.png",
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    def _createVisualizations(self):
        """
        Create visualizations for optimization results.
        """
        results_df = self._preprocessResults()
        # 1. Parameter Performance Plots
        self._plotParameterPerformance(results_df)
        # 2. Parameter Convergence Plot
        self._plotParameterConvergence(results_df)
        # 3. Metric Progression Plot
        self._plotMetricProgression(results_df)
        # 4. Score Distribution Plot
        self._plotScoreDistribution(results_df)

        self.logger.info(f"Visualizations saved to {
                         self.dirs['visualizations']}")

    def optimize(self):
        """
        Run the optimization process.

        This method runs the optimization process using the specified
        configuration and parameter space. It uses Bayesian optimization
        to find the best hyperparameters for the model.

        :return: best parameters and score
        """
        self.logger.info(f"Starting optimization for {
                         self.recommender.__name__}")

        # Run optimization
        self.opt_result = gp_minimize(
            func=self._objective,
            dimensions=self.dimensions,
            n_calls=self.config.n_trials,
            n_random_starts=self.config.n_random_starts,
            noise=0.01,
            # Use seed for reproducibility
            random_state=self.seeds['optimization'],
            callback=[self._checkpointCallback],
            verbose=True
        )

        # Save final results
        self._saveResults()

        self.logger.info(f"Optimization finished. Best score: {
            self.best_score:.4f} with params: {self.best_params}")

        # Create visualizations
        self._createVisualizations()

        return self.best_params, self.best_score


if __name__ == "__main__":
    evaluator = Evaluator(
        'data/raw/train_interactions.csv', 'data/raw/test_interactions_in.csv')

    param_space = {
        'alpha': (0.0, 0.75),
        'popularity_weight': (0.0, 0.20),
        'num_iterations': (75, 600),
        'interaction_weight_processing': ['log']
    }

    from models import PPR
    optimizer = BayesianOptimizer(
        recommender=PPR,
        evaluator=evaluator,
        param_space=param_space,
        output_dir='output/optimization',
        config=OptimizationConfig(
            n_trials=100,
            n_random_starts=10,
            sample_size=10000,
            n_cv_folds=3,
            val_size=0.2,
            base_seed=42,
            metric_weights={'ndcg': 0.85, 'recall': 0.15},
        )
    )
    optimizer.optimize()
