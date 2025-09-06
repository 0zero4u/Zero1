# FILE: Zero1-main/trainer.py (UPDATED FOR FREQUENT LOGGING)

"""
Enhanced Trainer with Transformer Architecture Integration
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.utils import set_random_seed
import os
import multiprocessing as mp
import logging
import time

# Optional dependencies with fallbacks
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available, hyperparameter optimization disabled")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("Weights & Biases not available, experiment tracking disabled")

# Import from local modules
from processor import create_bars_from_trades, EnhancedDataProcessor, generate_stateful_features_for_fitting
from config import SETTINGS, create_config, Environment, ModelArchitectureConfig
from tins import EnhancedHierarchicalAttentionFeatureExtractor # Will use the updated Transformer version
from engine import EnhancedHierarchicalTradingEnvironment
from normalizer import Normalizer

logger = logging.getLogger(__name__)

class RewardHorizonAnalysisCallback(BaseCallback):
    """Callback to analyze reward horizon effectiveness during training."""

    def __init__(self, log_frequency: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.log_frequency = log_frequency
        self.horizon_metrics = []

    def _on_step(self) -> bool:
        """Analyze reward horizon metrics periodically."""
        try:
            if self.n_calls % self.log_frequency == 0:
                if self.locals.get('infos'):
                    info = self.locals['infos'][0] if self.locals['infos'] else {}

                    # Extract reward horizon information
                    reward_horizon_info = info.get('reward_horizon', {})
                    if reward_horizon_info:
                        # Log reward horizon metrics
                        self.logger.record('reward_horizon/steps', reward_horizon_info.get('steps', 1))
                        self.logger.record('reward_horizon/decay_factor', reward_horizon_info.get('decay_factor', 1.0))
                        self.logger.record('reward_horizon/using_multi_step', int(reward_horizon_info.get('using_multi_step', False)))

                        # Log horizon effectiveness if available
                        horizon_info = reward_horizon_info.get('horizon_info', {})
                        if horizon_info:
                            self.logger.record('reward_horizon/immediate_return', horizon_info.get('immediate_return', 0))
                            self.logger.record('reward_horizon/final_return', horizon_info.get('final_return', 0))
                            self.logger.record('reward_horizon/weighted_return', horizon_info.get('weighted_return', 0))

                        # Store for analysis
                        self.horizon_metrics.append({
                            'step': self.n_calls,
                            **horizon_info
                        })

                    # Extract reward components for detailed analysis
                    reward_components = info.get('reward_components', {})
                    if reward_components:
                        for component, value in reward_components.items():
                            if isinstance(value, (int, float)):
                                self.logger.record(f'reward_components/{component}', value)

            return True

        except Exception as e:
            logger.error(f"Error in reward horizon analysis callback: {e}")
            return True

class PerformanceMonitoringCallback(BaseCallback):
    """Enhanced performance monitoring callback with detailed metrics tracking."""

    def __init__(self, performance_threshold: float = -0.15, drawdown_threshold: float = 0.25, verbose: int = 0):
        super().__init__(verbose)
        self.performance_threshold = performance_threshold
        self.drawdown_threshold = drawdown_threshold
        self.portfolio_values = []
        self.episode_returns = []
        self.peak_value = 0

    def _on_step(self) -> bool:
        """Monitor performance metrics during training."""
        try:
            if self.locals.get('infos'):
                info = self.locals['infos'][0] if self.locals['infos'] else {}

                portfolio_value = info.get('portfolio_value', 0)
                if portfolio_value > 0:
                    self.portfolio_values.append(portfolio_value)

                    # Track peak value and drawdown
                    if portfolio_value > self.peak_value:
                        self.peak_value = portfolio_value

                    current_drawdown = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0

                    # Log performance metrics
                    if self.n_calls % 1000 == 0:
                        self.logger.record('performance/portfolio_value', portfolio_value)
                        self.logger.record('performance/drawdown', current_drawdown)
                        self.logger.record('performance/peak_value', self.peak_value)

                        if len(self.portfolio_values) > 100:
                            recent_values = self.portfolio_values[-100:]
                            recent_return = (recent_values[-1] - recent_values[0]) / recent_values[0]
                            self.logger.record('performance/recent_return_100', recent_return)

            return True

        except Exception as e:
            logger.error(f"Error in performance monitoring callback: {e}")
            return True

class WandbCallback(BaseCallback):
    """Weights & Biases integration callback."""

    def __init__(self, project_name: str, experiment_name: str, config: Dict, verbose: int = 0):
        super().__init__(verbose)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.wandb_run = None

    def _on_training_start(self) -> None:
        """Initialize W&B run."""
        if WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project=self.project_name,
                    name=self.experiment_name,
                    config=self.config,
                    reinit=True
                )
                logger.info(f"Started W&B run: {self.wandb_run.name}")
            except Exception as e:
                logger.error(f"Failed to initialize W&B: {e}")

    def _on_step(self) -> bool:
        """Log metrics to W&B."""
        if self.wandb_run and self.n_calls % 1000 == 0:
            try:
                # Log from the logger's current values
                for key, value in self.logger.name_to_value.items():
                    if isinstance(value, (int, float)):
                        wandb.log({key: value}, step=self.n_calls)
            except Exception as e:
                logger.error(f"Error logging to W&B: {e}")

        return True

    def _on_training_end(self) -> None:
        """Finish W&B run."""
        if self.wandb_run:
            wandb.finish()

class OptimizedTrainer:
    """
    FIXED: Trainer with Transformer architecture integration and hyperparameter optimization.
    REMOVED redundant validation logic - trusts Pydantic config as single source of truth.
    """

    def __init__(self, base_config: Optional[Dict] = None, use_wandb: bool = False):
        self.base_config = base_config or SETTINGS.dict()
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.best_trial_results = None
        self.last_saved_model_path = None # --- NEW: To store the latest model path ---

        try:
            # Pre-load data once
            logger.info("🔄 Pre-loading training data...")
            processor = EnhancedDataProcessor(config=SETTINGS)

            # Load bars for environment and features for normalizer
            self.bars_df = processor.create_enhanced_bars_from_trades("in_sample")

            # Validate data sufficiency for reward horizon
            if not SETTINGS.validate_reward_horizon_data(len(self.bars_df)):
                logger.error("Insufficient data for configured reward horizon!")
                raise ValueError("Dataset too small for reward horizon configuration")

            # Generate context features for normalizer fitting (vectorized)
            logger.info("Generating context features for normalizer fitting...")
            self.features_df = generate_stateful_features_for_fitting(
                self.bars_df, SETTINGS.strategy
            )

            # Ensure alignment between bars_df and features_df
            bars_index = self.bars_df.set_index('timestamp').index
            features_index = self.features_df.set_index('timestamp').index

            if not bars_index.equals(features_index):
                logger.warning("Aligning feature_df index with bars_df index.")
                self.features_df = self.features_df.set_index('timestamp').reindex(bars_index).reset_index()
                self.features_df.fillna(0.0, inplace=True)

            # Fit and save the normalizer
            self.normalizer = Normalizer(SETTINGS.strategy)
            self.normalizer.fit(self.bars_df, self.features_df)
            self.normalizer.save(Path(SETTINGS.get_normalizer_path()))

            logger.info(f"✅ Loaded {len(self.bars_df)} bars for training")
            logger.info(f"✅ Reward horizon: {SETTINGS.strategy.reward_horizon_steps} steps")

            # Determine number of parallel workers
            self.num_cpu = min(os.cpu_count(), 8)
            logger.info(f"🚀 Using {self.num_cpu} parallel environments for training.")

        except Exception as e:
            logger.error(f"Failed to initialize trainer: {e}")
            raise

    def _make_env(self, rank: int, seed: int = 0, trial_params: Optional[Dict] = None) -> Callable:
        """Environment factory with reward horizon and Transformer architecture support."""
        def _init():
            try:
                config_overrides = {}

                # Extract tunable parameters
                leverage = trial_params.get('leverage', 10.0) if trial_params else 10.0

                # Extract reward horizon parameters
                reward_horizon_steps = trial_params.get('reward_horizon_steps', 1) if trial_params else 1
                reward_horizon_decay = trial_params.get('reward_horizon_decay', 1.0) if trial_params else 1.0

                # Extract reward weights
                reward_weights = None
                if trial_params:
                    reward_weights = {
                        'base_return': trial_params.get('reward_weight_base_return', 1.0),
                        'risk_adjusted': trial_params.get('reward_weight_risk_adjusted', 0.3),
                        'stability': trial_params.get('reward_weight_stability', 0.2),
                        'transaction_penalty': trial_params.get('reward_weight_transaction_penalty', -0.1),
                        'drawdown_penalty': trial_params.get('reward_weight_drawdown_penalty', -0.4),
                        'position_penalty': trial_params.get('reward_weight_position_penalty', -0.05),
                        'risk_bonus': trial_params.get('reward_weight_risk_bonus', 0.15)
                    }

                # Override configuration for this trial
                if trial_params:
                    config_overrides = {
                        'strategy': {
                            'max_margin_allocation_pct': trial_params.get('max_margin_allocation_pct', 0.02),
                            'reward_horizon_steps': reward_horizon_steps,
                            'reward_horizon_decay': reward_horizon_decay
                        }
                    }

                trial_specific_config = create_config(**config_overrides) if config_overrides else SETTINGS

                # Create environment with enhanced parameters
                env = EnhancedHierarchicalTradingEnvironment(
                    df_base_ohlc=self.bars_df,
                    normalizer=self.normalizer,
                    config=trial_specific_config,
                    leverage=leverage,
                    reward_weights=reward_weights,
                    precomputed_features=self.features_df
                )

                env.reset(seed=seed + rank)
                return env

            except Exception as e:
                logger.error(f"Error creating environment {rank}: {e}")
                raise

        set_random_seed(seed)
        return _init

    def create_model(self, trial_params: Dict, vec_env: VecEnv) -> PPO:
        """
        FIXED: Create PPO model with Transformer architecture parameters.
        Uses validated configuration from Pydantic - no redundant validation.
        """
        try:
            # FIXED: Create and validate architecture config using Pydantic
            # This ensures all validation logic is in one place
            arch_config = ModelArchitectureConfig(
                transformer_d_model=trial_params.get('transformer_d_model', 64),
                transformer_n_heads=trial_params.get('transformer_n_heads', 4),
                transformer_dim_feedforward=trial_params.get('transformer_dim_feedforward', 256),
                transformer_num_layers=trial_params.get('transformer_num_layers', 2),
                expert_output_dim=trial_params.get('expert_output_dim', 32),
                attention_head_features=trial_params.get('attention_features', 64),
                dropout_rate=trial_params.get('dropout_rate', 0.1),
                use_batch_norm=trial_params.get('use_batch_norm', True),
                use_residual_connections=trial_params.get('use_residual_connections', True),
            )

            # Pydantic will automatically validate and adjust parameters if needed
            logger.info(f"Using validated Transformer config: d_model={arch_config.transformer_d_model}, "
                       f"n_heads={arch_config.transformer_n_heads}")

            # Policy configuration using validated architecture
            policy_kwargs = {
                "features_extractor_class": EnhancedHierarchicalAttentionFeatureExtractor,
                "features_extractor_kwargs": {
                    "arch_cfg": arch_config
                },
                "net_arch": {"pi": [256, 128], "vf": [256, 128]}  # Policy and value function networks
            }

            # Learning rate schedule
            learning_rate = trial_params.get('learning_rate', 3e-4)
            if trial_params.get('learning_rate_schedule', 'constant') != 'constant':
                if trial_params['learning_rate_schedule'] == 'linear':
                    learning_rate = lambda progress_remaining: learning_rate * progress_remaining
                elif trial_params['learning_rate_schedule'] == 'cosine':
                    learning_rate = lambda progress_remaining: learning_rate * 0.5 * (1 + np.cos(np.pi * (1 - progress_remaining)))

            # Create model
            model = PPO(
                policy="MultiInputPolicy",
                env=vec_env,
                learning_rate=learning_rate,
                n_steps=trial_params.get('n_steps', 2048),
                batch_size=trial_params.get('batch_size', 64),
                n_epochs=trial_params.get('n_epochs', 10),
                gamma=trial_params.get('gamma', 0.99),
                gae_lambda=trial_params.get('gae_lambda', 0.95),
                clip_range=trial_params.get('clip_range', 0.2),
                ent_coef=trial_params.get('ent_coef', 0.01),
                max_grad_norm=trial_params.get('max_grad_norm', 0.5),
                target_kl=trial_params.get('target_kl', None),
                policy_kwargs=policy_kwargs,
                tensorboard_log=SETTINGS.get_tensorboard_path(),
                device=SETTINGS.device,
                seed=trial_params.get('seed', 42),
                verbose=0
            )

            logger.info("✅ Created PPO model with validated Transformer architecture")
            return model

        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

    def objective(self, trial) -> float:
        """
        FIXED: Optuna objective with Transformer architecture optimization.
        REMOVED redundant validation - trusts Pydantic configuration validation.
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna not available for hyperparameter optimization")

        vec_env = None
        eval_vec_env = None

        try:
            # FIXED: Simplified hyperparameter search space - validation handled by Pydantic
            trial_params = {
                # Core PPO parameters
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'n_epochs': trial.suggest_int('n_epochs', 3, 20),
                'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.999),
                'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
                'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True),
                'max_grad_norm': trial.suggest_float('max_grad_norm', 0.1, 2.0),

                # Transformer architecture hyperparameters (will be validated by Pydantic)
                'transformer_d_model': trial.suggest_categorical('transformer_d_model', [32, 64, 128, 256]),
                'transformer_n_heads': trial.suggest_categorical('transformer_n_heads', [2, 4, 8]),
                'transformer_dim_feedforward': trial.suggest_categorical('transformer_dim_feedforward', [128, 256, 512, 1024]),
                'transformer_num_layers': trial.suggest_int('transformer_num_layers', 1, 4),

                # Expert and attention parameters
                'expert_output_dim': trial.suggest_categorical('expert_output_dim', [16, 32, 64, 128]),
                'attention_features': trial.suggest_categorical('attention_features', [32, 64, 128, 256]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),

                # Training parameters
                'seed': trial.suggest_int('seed', 1, 10000),
                'leverage': trial.suggest_float('leverage', 1.0, 25.0, step=0.5),
                'learning_rate_schedule': trial.suggest_categorical('learning_rate_schedule', ['constant', 'linear', 'cosine']),
                'use_target_kl': trial.suggest_categorical('use_target_kl', [True, False]),
                'max_margin_allocation_pct': trial.suggest_float('max_margin_allocation_pct', 0.01, 0.1, step=0.005),

                # Reward horizon parameters
                'reward_horizon_steps': trial.suggest_int('reward_horizon_steps', 1, min(15, len(self.bars_df) // 1000)),
                'reward_horizon_decay': trial.suggest_float('reward_horizon_decay', 0.7, 1.0, step=0.05),

                # Reward weights
                'reward_weight_base_return': trial.suggest_float('reward_weight_base_return', 0.5, 2.0, step=0.1),
                'reward_weight_risk_adjusted': trial.suggest_float('reward_weight_risk_adjusted', 0.0, 0.8, step=0.05),
                'reward_weight_stability': trial.suggest_float('reward_weight_stability', 0.0, 0.5, step=0.05),
                'reward_weight_transaction_penalty': trial.suggest_float('reward_weight_transaction_penalty', -0.3, 0.0, step=0.02),
                'reward_weight_drawdown_penalty': trial.suggest_float('reward_weight_drawdown_penalty', -1.0, 0.0, step=0.05),
                'reward_weight_position_penalty': trial.suggest_float('reward_weight_position_penalty', -0.2, 0.0, step=0.01),
                'reward_weight_risk_bonus': trial.suggest_float('reward_weight_risk_bonus', 0.0, 0.3, step=0.02),
            }

            # Set target_kl based on choice
            if trial_params['use_target_kl']:
                trial_params['target_kl'] = trial.suggest_float('target_kl', 0.001, 0.1, log=True)
            else:
                trial_params['target_kl'] = None

            # Validate reward horizon against dataset size
            max_allowable_horizon = min(20, len(self.bars_df) // 1000)
            if trial_params['reward_horizon_steps'] > max_allowable_horizon:
                trial_params['reward_horizon_steps'] = max_allowable_horizon
                logger.warning(f"Reduced reward horizon to {max_allowable_horizon} due to dataset size")

            # Create parallel environments
            vec_env = SubprocVecEnv([self._make_env(i, trial.number, trial_params=trial_params) for i in range(self.num_cpu)])
            model = self.create_model(trial_params, vec_env)

            # Setup logging
            experiment_name = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_path = SETTINGS.get_logs_path()
            Path(log_path).mkdir(parents=True, exist_ok=True)
            model.set_logger(configure(log_path, ["stdout", "csv", "tensorboard"]))

            # Setup callbacks
            callbacks = []

            # Evaluation callback
            # --- START OF FIX #1: Capping Evaluation Time ---
            from gymnasium.wrappers import TimeLimit
            eval_env_raw = self._make_env(rank=0, seed=123, trial_params=trial_params)()
            eval_env_limited = TimeLimit(eval_env_raw, max_episode_steps=5000)
            eval_vec_env = DummyVecEnv([lambda: eval_env_limited])
            # --- END OF FIX #1 ---

            eval_callback = EvalCallback(
                eval_vec_env,
                best_model_save_path=str(Path(log_path) / f"best_model_trial_{trial.number}"),
                log_path=log_path,
                eval_freq=max(5000 // self.num_cpu, 500),
                deterministic=True,
                render=False,
                n_eval_episodes=5
            )
            callbacks.append(eval_callback)

            # Reward horizon analysis callback
            horizon_callback = RewardHorizonAnalysisCallback(log_frequency=2000)
            callbacks.append(horizon_callback)

            # Performance monitoring
            perf_callback = PerformanceMonitoringCallback(
                performance_threshold=-0.15,
                drawdown_threshold=0.25
            )
            callbacks.append(perf_callback)

            # W&B logging if enabled
            if self.use_wandb:
                wandb_callback = WandbCallback(
                    project_name="crypto_trading_transformer",
                    experiment_name=experiment_name,
                    config={'trial_number': trial.number, **trial_params}
                )
                callbacks.append(wandb_callback)
                
            # --- START OF FIX #2: Capping Trial Training Time ---
            # Set a fixed, short number of timesteps for each optimization trial
            training_steps = 20480  # Approx. 2-3 minutes per trial
            # --- END OF FIX #2 ---

            # Custom pruning callback
            class PruningCallback(BaseCallback):
                def __init__(self, trial, verbose: int = 0):
                    super().__init__(verbose)
                    self.trial = trial

                def _on_step(self) -> bool:
                    try:
                        if self.n_calls % (10000 // self.training_env.num_envs) == 0 and len(perf_callback.portfolio_values) > 0:
                            recent_values = perf_callback.portfolio_values[-100:]
                            if len(recent_values) > 1:
                                performance = (recent_values[-1] - recent_values[0]) / recent_values[0]
                                self.trial.report(performance, step=self.num_timesteps)

                                if self.trial.should_prune():
                                    raise optuna.TrialPruned()

                        return True

                    except optuna.TrialPruned:
                        raise
                    except Exception as e:
                        logger.error(f"Error in pruning callback: {e}")
                        return True

            if OPTUNA_AVAILABLE:
                pruning_callback = PruningCallback(trial)
                callbacks.append(pruning_callback)

            # Train the model
            model.learn(
                total_timesteps=training_steps,
                callback=callbacks,
                progress_bar=False
            )

            # Calculate final performance metric
            if perf_callback.portfolio_values:
                initial_value = perf_callback.portfolio_values[0]
                final_value = perf_callback.portfolio_values[-1]
                total_return = (final_value - initial_value) / initial_value

                # Calculate time-based metrics
                num_steps = len(perf_callback.portfolio_values)
                base_bar_seconds = SETTINGS.get_timeframe_seconds(SETTINGS.base_bar_timeframe)
                total_seconds = num_steps * base_bar_seconds
                total_days = max(1.0, total_seconds / (24 * 3600))
                annualized_return = total_return * (365 / total_days)

                # Calculate maximum drawdown
                portfolio_values_np = np.array(perf_callback.portfolio_values)
                cumulative_max = np.maximum.accumulate(portfolio_values_np)
                drawdowns = (cumulative_max - portfolio_values_np) / (cumulative_max + 1e-9)
                max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

                # Enhanced performance metric with Transformer and reward horizon adjustment
                leverage = trial_params.get('leverage', 10.0)
                reward_horizon = trial_params.get('reward_horizon_steps', 1)

                if max_drawdown > 0.01:
                    calmar_ratio = annualized_return / max_drawdown

                    # Adjust for both leverage and reward horizon
                    leverage_adjusted_calmar = calmar_ratio * np.sqrt(leverage / 10.0)

                    # Bonus for using longer horizons effectively
                    horizon_bonus = 1.0 + (reward_horizon - 1) * 0.02  # 2% bonus per step beyond immediate

                    # Transformer complexity bonus/penalty
                    transformer_complexity = (trial_params['transformer_d_model'] * 
                                            trial_params['transformer_n_heads'] * 
                                            trial_params['transformer_num_layers'])
                    complexity_factor = 1.0 + min(0.1, transformer_complexity / 10000.0)  # Small bonus for reasonable complexity

                    final_score = leverage_adjusted_calmar * horizon_bonus * complexity_factor
                else:
                    final_score = annualized_return * 10 if annualized_return >= 0 else annualized_return

                # Store trial results
                trial.set_user_attr('total_return', total_return)
                trial.set_user_attr('annualized_return', annualized_return)
                trial.set_user_attr('max_drawdown', max_drawdown)
                trial.set_user_attr('calmar_ratio', calmar_ratio if max_drawdown > 0.01 else 0)
                trial.set_user_attr('final_score', final_score)
                trial.set_user_attr('leverage', leverage)
                trial.set_user_attr('reward_horizon_steps', reward_horizon)
                trial.set_user_attr('reward_horizon_decay', trial_params.get('reward_horizon_decay', 1.0))

                # Store Transformer architecture metrics
                trial.set_user_attr('transformer_d_model', trial_params['transformer_d_model'])
                trial.set_user_attr('transformer_n_heads', trial_params['transformer_n_heads'])
                trial.set_user_attr('transformer_num_layers', trial_params['transformer_num_layers'])
                trial.set_user_attr('transformer_complexity', transformer_complexity)

                # Store reward horizon effectiveness metrics
                if horizon_callback.horizon_metrics:
                    horizon_analysis = self._analyze_horizon_effectiveness(horizon_callback.horizon_metrics)
                    trial.set_user_attr('horizon_effectiveness', horizon_analysis)

                return final_score
            else:
                return -1.0

        except Exception as e:
            if OPTUNA_AVAILABLE and isinstance(e, optuna.TrialPruned):
                raise

            logger.error(f"Trial failed with error: {e}")
            return -10.0

        finally:
            # Cleanup
            try:
                if vec_env:
                    vec_env.close()
                if eval_vec_env:
                    eval_vec_env.close()
            except Exception as e:
                logger.error(f"Error cleaning up environments: {e}")

    def _analyze_horizon_effectiveness(self, horizon_metrics: List[Dict]) -> Dict[str, float]:
        """Analyze the effectiveness of the reward horizon during training."""
        try:
            if not horizon_metrics:
                return {}

            immediate_returns = [h.get('immediate_return', 0) for h in horizon_metrics]
            final_returns = [h.get('final_return', 0) for h in horizon_metrics]
            weighted_returns = [h.get('weighted_return', 0) for h in horizon_metrics]

            analysis = {
                'sample_count': len(horizon_metrics),
                'avg_immediate_return': np.mean(immediate_returns) if immediate_returns else 0,
                'avg_final_return': np.mean(final_returns) if final_returns else 0,
                'avg_weighted_return': np.mean(weighted_returns) if weighted_returns else 0,
                'return_consistency': 1.0 - np.std(weighted_returns) if len(weighted_returns) > 1 else 0,
                'horizon_correlation': np.corrcoef(immediate_returns, final_returns)[0,1] if len(immediate_returns) > 10 else 0
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing horizon effectiveness: {e}")
            return {}

    def optimize(self, n_trials: int = 50, timeout: Optional[int] = None):
        """
        FIXED: Run optimization with Transformer architecture and reward horizon parameters.
        Uses Pydantic validation instead of manual checks.
        """
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna not available, cannot run hyperparameter optimization")
            return None

        try:
            # Create study with advanced pruner
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=10
            )

            study = optuna.create_study(
                direction='maximize',
                pruner=pruner,
                sampler=optuna.samplers.TPESampler(seed=42)
            )

            logger.info(f"🚀 Starting FIXED optimization with Transformer architecture")
            logger.info(f"📊 Dataset size: {len(self.bars_df)} bars")
            logger.info(f"🎯 Max reward horizon: {min(15, len(self.bars_df) // 1000)} steps")
            logger.info("✅ FIXED: Using Pydantic validation, no redundant checks")

            # Optimize
            study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True
            )

            # Store best results
            self.best_trial_results = {
                'best_trial': study.best_trial,
                'best_value': study.best_value,
                'n_trials': len(study.trials)
            }

            # Print results
            logger.info("🎯 FIXED Optimization Results with Transformer Architecture:")
            logger.info(f"Best final score: {study.best_value:.4f}")
            logger.info(f"Best Transformer d_model: {study.best_trial.params.get('transformer_d_model', 'N/A')}")
            logger.info(f"Best Transformer n_heads: {study.best_trial.params.get('transformer_n_heads', 'N/A')}")
            logger.info(f"Best Transformer layers: {study.best_trial.params.get('transformer_num_layers', 'N/A')}")
            logger.info(f"Best reward horizon: {study.best_trial.params.get('reward_horizon_steps', 'N/A')} steps")
            logger.info(f"Best leverage: {study.best_trial.params.get('leverage', 'N/A')}")

            # Analyze architecture patterns across trials
            self._analyze_study_architecture_patterns(study)

            return study

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return None

    def _analyze_study_architecture_patterns(self, study):
        """
        Analyze Transformer architecture patterns across all trials.
        """
        try:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed_trials) < 5:
                return

            # Analyze Transformer architecture patterns
            d_model_performance = {}
            n_heads_performance = {}
            layers_performance = {}

            for trial in completed_trials:
                d_model = trial.params.get('transformer_d_model', 64)
                n_heads = trial.params.get('transformer_n_heads', 4)
                num_layers = trial.params.get('transformer_num_layers', 2)
                final_score = trial.user_attrs.get('final_score', trial.value)

                # Group by architecture parameters
                if d_model not in d_model_performance:
                    d_model_performance[d_model] = []
                d_model_performance[d_model].append(final_score)

                if n_heads not in n_heads_performance:
                    n_heads_performance[n_heads] = []
                n_heads_performance[n_heads].append(final_score)

                if num_layers not in layers_performance:
                    layers_performance[num_layers] = []
                layers_performance[num_layers].append(final_score)

            # Analyze patterns
            logger.info("\n📊 Transformer Architecture Performance Analysis:")

            logger.info("D_Model Performance:")
            for d_model in sorted(d_model_performance.keys()):
                scores = d_model_performance[d_model]
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                logger.info(f"  d_model {d_model}: {avg_score:.4f} ± {std_score:.4f} ({len(scores)} trials)")

            logger.info("N_Heads Performance:")
            for n_heads in sorted(n_heads_performance.keys()):
                scores = n_heads_performance[n_heads]
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                logger.info(f"  {n_heads} heads: {avg_score:.4f} ± {std_score:.4f} ({len(scores)} trials)")

            logger.info("Num_Layers Performance:")
            for num_layers in sorted(layers_performance.keys()):
                scores = layers_performance[num_layers]
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                logger.info(f"  {num_layers} layers: {avg_score:.4f} ± {std_score:.4f} ({len(scores)} trials)")

            # Find optimal architecture
            best_d_model = max(d_model_performance.keys(), key=lambda d: np.mean(d_model_performance[d]))
            best_n_heads = max(n_heads_performance.keys(), key=lambda n: np.mean(n_heads_performance[n]))
            best_layers = max(layers_performance.keys(), key=lambda l: np.mean(layers_performance[l]))

            logger.info(f"🏆 Best performing architecture:")
            logger.info(f"  - D_Model: {best_d_model}")
            logger.info(f"  - N_Heads: {best_n_heads}")
            logger.info(f"  - Layers: {best_layers}")

        except Exception as e:
            logger.error(f"Error analyzing architecture patterns: {e}")

    def train_best_model(self, best_params: Dict, final_training_steps: int) -> PPO:
        """Train the final model with best parameters."""
        try:
            logger.info("🎯 Training final model with best Transformer architecture parameters")

            # Create environment with best parameters
            vec_env = SubprocVecEnv([self._make_env(i, seed=42, trial_params=best_params) for i in range(self.num_cpu)])

            # Create model with best parameters
            model = self.create_model(best_params, vec_env)

            # Setup final logging
            final_log_path = SETTINGS.get_logs_path()
            model.set_logger(configure(final_log_path, ["stdout", "csv", "tensorboard"]))

            # Setup callbacks for final training
            callbacks = []

            # Performance monitoring
            perf_callback = PerformanceMonitoringCallback()
            callbacks.append(perf_callback)

            # Reward horizon analysis
            horizon_callback = RewardHorizonAnalysisCallback(log_frequency=1000)
            callbacks.append(horizon_callback)

            if self.use_wandb:
                wandb_callback = WandbCallback(
                    project_name="crypto_trading_transformer_final",
                    experiment_name=f"final_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=best_params
                )
                callbacks.append(wandb_callback)

            # Train final model
            logger.info(f"Starting final training with {final_training_steps:,} timesteps")
            model.learn(
                total_timesteps=final_training_steps,
                callback=callbacks,
                progress_bar=True
            )

            # Save the model
            model_path_str = SETTINGS.get_model_path()
            model.save(model_path_str)
            logger.info(f"✅ Final model saved to: {model_path_str}")

            # --- UPDATED: Save model and configuration snapshot together ---
            # Save the configuration used to train this specific model
            config_path = model_path_str.replace(".zip", "_config.json")
            with open(config_path, 'w') as f:
                # Use Pydantic's built-in JSON export for a clean, complete snapshot
                f.write(SETTINGS.json(indent=4))
            logger.info(f"✅ Configuration snapshot saved to: {config_path}")
            # --- END OF UPDATE ---
            
            self.last_saved_model_path = model_path_str # --- NEW: Store path ---

            # Clean up
            vec_env.close()

            return model

        except Exception as e:
            logger.error(f"Failed to train final model: {e}")
            raise

# FIXED: Main training interface with Transformer architecture

def train_model_advanced(optimization_trials: int = 20,
                        final_training_steps: int = 500000,
                        use_wandb: bool = False,
                        reward_horizon_steps: int = None) -> str: # --- UPDATED: Returns model path string ---
    """
    FIXED: Advanced training with Transformer architecture and reward horizon system.
    Returns the path to the trained model.
    """
    try:
        logger.info("🎯 Starting FIXED Training with Transformer Architecture")
        logger.info("✅ FIXES APPLIED:")
        logger.info("  - Removed redundant validation logic from trainer")
        logger.info("  - Pydantic config is single source of truth for validation")
        logger.info("  - Cleaner hyperparameter optimization")
        logger.info("  - Enhanced Transformer architecture tuning")

        # Override reward horizon in config if specified
        if reward_horizon_steps is not None:
            SETTINGS.strategy.reward_horizon_steps = reward_horizon_steps
            logger.info(f"🔧 Overriding reward horizon to {reward_horizon_steps} steps")

        print("="*50)

        # Initialize trainer
        trainer = OptimizedTrainer(use_wandb=use_wandb)

        if optimization_trials > 0 and OPTUNA_AVAILABLE:
            # Run hyperparameter optimization
            logger.info(f"Phase 1: FIXED Optimization with Transformer Architecture ({optimization_trials} trials)")
            study = trainer.optimize(n_trials=optimization_trials)

            if study is None:
                logger.error("Optimization failed, using default parameters")
                best_params = {
                    # Core PPO parameters
                    'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64,
                    'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95,
                    'clip_range': 0.2, 'ent_coef': 0.01, 'max_grad_norm': 0.5,

                    # Transformer architecture parameters
                    'transformer_d_model': 64, 'transformer_n_heads': 4,
                    'transformer_dim_feedforward': 256, 'transformer_num_layers': 2,
                    'expert_output_dim': 32, 'attention_features': 64,
                    'dropout_rate': 0.1,

                    # Training parameters
                    'seed': 42, 'leverage': 10.0,
                    'learning_rate_schedule': 'linear', 'target_kl': None,
                    'max_margin_allocation_pct': 0.02,
                    'reward_horizon_steps': reward_horizon_steps or 1,
                    'reward_horizon_decay': 1.0,

                    # Default reward weights
                    'reward_weight_base_return': 1.0,
                    'reward_weight_risk_adjusted': 0.3,
                    'reward_weight_stability': 0.2,
                    'reward_weight_transaction_penalty': -0.1,
                    'reward_weight_drawdown_penalty': -0.4,
                    'reward_weight_position_penalty': -0.05,
                    'reward_weight_risk_bonus': 0.15
                }
            else:
                best_params = study.best_trial.params

            # Train final model
            logger.info(f"Phase 2: Final Training with Optimized Transformer ({final_training_steps:,} steps)")
            model = trainer.train_best_model(best_params, final_training_steps)

        else:
            # Train with default parameters
            logger.info(f"Training with default Transformer parameters ({final_training_steps:,} steps)")
            default_params = {
                # Core PPO parameters
                'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64,
                'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95,
                'clip_range': 0.2, 'ent_coef': 0.01, 'max_grad_norm': 0.5,

                # Transformer architecture parameters
                'transformer_d_model': 64, 'transformer_n_heads': 4,
                'transformer_dim_feedforward': 256, 'transformer_num_layers': 2,
                'expert_output_dim': 32, 'attention_features': 64,
                'dropout_rate': 0.1,

                # Training parameters
                'seed': 42, 'leverage': 10.0,
                'learning_rate_schedule': 'linear', 'target_kl': None,
                'max_margin_allocation_pct': 0.02,
                'reward_horizon_steps': reward_horizon_steps or 1,
                'reward_horizon_decay': 1.0,

                # Default reward weights
                'reward_weight_base_return': 1.0,
                'reward_weight_risk_adjusted': 0.3,
                'reward_weight_stability': 0.2,
                'reward_weight_transaction_penalty': -0.1,
                'reward_weight_drawdown_penalty': -0.4,
                'reward_weight_position_penalty': -0.05,
                'reward_weight_risk_bonus': 0.15
            }

            model = trainer.train_best_model(default_params, final_training_steps)

        logger.info("🎉 FIXED training with Transformer architecture completed!")
        logger.info(f"Model saved to: {trainer.last_saved_model_path}")

        return trainer.last_saved_model_path # --- UPDATED: Return the path ---

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    # Ensure multiprocessing compatibility
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    try:
        # Example: Train with Transformer architecture and 3-minute reward horizon
        model_path = train_model_advanced(
            optimization_trials=10,
            final_training_steps=100000,
            use_wandb=False,
            reward_horizon_steps=9  # 9 steps = 3 minutes
        )

        logger.info(f"✅ FIXED training with Transformer architecture completed successfully!")
        logger.info(f"Final model available at: {model_path}")

    except Exception as e:
        logger.error(f"Training example failed: {e}")
        raise