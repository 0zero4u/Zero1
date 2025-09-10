# enhanced_trainer_verbose.py

"""
ENHANCED FIXED TRAINER: Now with Live Trade Monitoring Support and Hyperparameter Logging

This version passes worker_id to environments to enable verbose mode for Worker #0.
CRITICAL FIX: Added HParamCallback to correctly log hyperparameters to TensorBoard's
HPARAMS tab, as Stable Baselines 3 does not do this automatically.
CRITICAL FIX: Optuna objective now uses raw P&L and penalizes intrinsic reward exploitation.
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from stable_baselines3 import PPO
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

# --- START OF FIX: Import SummaryWriter and TensorBoardOutputFormat ---
try:
    from torch.utils.tensorboard import SummaryWriter
    from stable_baselines3.common.logger import TensorBoardOutputFormat
except ImportError:
    SummaryWriter = None
    TensorBoardOutputFormat = None
    logging.warning("TensorBoard not available, hyperparameter logging will be disabled.")
# --- END OF FIX ---

# Import from local modules
from processor import create_bars_from_trades, EnhancedDataProcessor, generate_stateful_features_for_fitting
from config import SETTINGS, create_config, Environment, ModelArchitectureConfig
from tins import EnhancedHierarchicalAttentionFeatureExtractor

# ENHANCED: Import the new verbose engine
from enhanced_engine_verbose import FixedHierarchicalTradingEnvironment

from normalizer import Normalizer

logger = logging.getLogger(__name__)


# --- START OF FIX: HParamCallback Implementation ---
class HParamCallback(BaseCallback):
    """
    A custom callback to log hyperparameters to TensorBoard's HPARAMS tab.
    This is necessary because Stable Baselines 3 does not do this automatically.
    """
    def __init__(self, trial: "optuna.trial.Trial", metrics_to_log: List[str]):
        """
        :param trial: The Optuna trial object which contains the hyperparameters.
        :param metrics_to_log: A list of metric names that will be used to judge
                               the performance of this hyperparameter set.
                               e.g., ['eval/mean_reward', 'performance/total_return_raw']
        """
        super().__init__()
        self.trial = trial
        self.metrics_to_log = metrics_to_log

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        We use it to log the hyperparameters.
        """
        if SummaryWriter is None or TensorBoardOutputFormat is None:
            return  # Skip if tensorboard is not installed

        # --- START OF CRITICAL FIX ---
        # Manually find the TensorBoard writer from the logger's output formats.
        # This is more robust than relying on the get_writer() helper method, which
        # was causing the "'Logger' object has no attribute 'get_writer'" error.
        writer = None
        for output_format in self.logger.output_formats:
            if isinstance(output_format, TensorBoardOutputFormat):
                writer = output_format.writer
                break
        # --- END OF CRITICAL FIX ---

        if writer is None:
            logger.warning("TensorBoard writer not found, skipping HPARAMS logging.")
            return

        # Sanitize parameters for TensorBoard HPARAMS
        # It only accepts bool, string, float, int, or None
        hparams = {}
        for key, value in self.trial.params.items():
            if isinstance(value, (bool, str, float, int)) or value is None:
                hparams[key] = value
            else:
                hparams[key] = str(value) # Convert other types to string

        # Define the metrics that will be associated with these hyperparameters
        # The values are placeholders (0) because the actual final metrics are not known yet.
        # TensorBoard will populate them later when they are logged by other callbacks (e.g., EvalCallback).
        metric_dict = {metric: 0 for metric in self.metrics_to_log}
        
        try:
            # The core of the solution: log the hyperparameters and metric placeholders
            writer.add_hparams(hparams, metric_dict, run_name="hparams")
            writer.flush()
            logger.info("Logged hyperparameters to TensorBoard HPARAMS tab.")
        except Exception as e:
            logger.error(f"Failed to log HPARAMS to TensorBoard: {e}")

    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.
        For this callback, we only need to log hyperparameters once at the start,
        so this method does nothing but return True to continue training.
        """
        return True

# --- END OF FIX ---


class PerformanceMonitoringCallback(BaseCallback):
    """
    Enhanced performance monitoring callback with detailed metrics tracking.
    FIXED: Now tracks raw (extrinsic) and intrinsic reward components separately.
    """
    
    def __init__(self, performance_threshold: float = -0.15, drawdown_threshold: float = 0.25, verbose: int = 0):
        super().__init__(verbose)
        self.performance_threshold = performance_threshold
        self.drawdown_threshold = drawdown_threshold
        self.portfolio_values = []
        self.episode_returns = []
        self.peak_value = 0
        # --- START OF FIX: Add lists to track separated rewards ---
        self.raw_rewards = []
        self.intrinsic_rewards = []
        # --- END OF FIX ---

    def _on_step(self) -> bool:
        """Monitor performance metrics during training."""
        try:
            # Log every 700 steps for better granularity
            if self.n_calls % 700 == 0 and self.locals.get('infos'):
                info = self.locals['infos'][0] if self.locals['infos'] else {}
                
                # --- START OF FIX: Log separated rewards and their components ---
                raw_reward = info.get('raw_reward', 0.0)
                intrinsic_reward = info.get('intrinsic_reward', 0.0)
                self.raw_rewards.append(raw_reward)
                self.intrinsic_rewards.append(intrinsic_reward)
                self.logger.record('performance/raw_reward', raw_reward)
                self.logger.record('performance/intrinsic_reward', intrinsic_reward)
                
                reward_components = info.get('reward_components', {})
                if reward_components:
                    for component, value in reward_components.items():
                        if isinstance(value, (int, float)):
                            self.logger.record(f'reward_components/{component}', value)
                # --- END OF FIX ---
                
                action_magnitude = info.get('action_magnitude', 0.0)
                self.logger.record('behavior/action_magnitude', action_magnitude)
                
                consecutive_inactive = info.get('consecutive_inactive_steps', 0)
                self.logger.record('behavior/consecutive_inactive_steps', consecutive_inactive)
                
                portfolio_value = info.get('portfolio_value', 0)
                if portfolio_value > 0:
                    self.portfolio_values.append(portfolio_value)
                    
                    if not self.peak_value or portfolio_value > self.peak_value:
                        self.peak_value = portfolio_value
                    
                    current_drawdown = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0
                    
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
        # Log every 700 steps for more frequent updates
        if self.wandb_run and self.n_calls % 700 == 0:
            try:
                log_dict = {}
                for key, value in self.logger.name_to_value.items():
                    if isinstance(value, (int, float)):
                        log_dict[key] = value
                
                if log_dict:
                    wandb.log(log_dict, step=self.n_calls)
                    
            except Exception as e:
                logger.error(f"Error logging to W&B: {e}")
        
        return True

    def _on_training_end(self) -> None:
        """Finish W&B run."""
        if self.wandb_run:
            wandb.finish()

class EnhancedFixedTrainer:
    """
    ENHANCED FIXED: Trainer that prevents turtling, promotes learning, and provides live monitoring.
    """
    
    def __init__(self, base_config: Optional[Dict] = None, use_wandb: bool = False, enable_live_monitoring: bool = True):
        self.base_config = base_config or SETTINGS.dict()
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.enable_live_monitoring = enable_live_monitoring
        self.best_trial_results = None
        self.last_saved_model_path = None
        
        try:
            logger.info("🔄 Pre-loading training data for Enhanced Fixed Reward System...")
            processor = EnhancedDataProcessor(config=SETTINGS)
            self.bars_df = processor.create_enhanced_bars_from_trades("in_sample")
            
            logger.info("Generating stateful (context) features for normalizer fitting...")
            context_features_df = generate_stateful_features_for_fitting(
                self.bars_df, SETTINGS.strategy
            )
            
            # Merge bars with context features
            self.all_features_df = pd.merge(self.bars_df, context_features_df, on='timestamp', how='left')
            self.all_features_df.fillna(method='ffill', inplace=True)
            self.all_features_df.fillna(0.0, inplace=True)
            
            # Ensure alignment between bars_df and features_df
            bars_index = self.bars_df.set_index('timestamp').index
            self.all_features_df = self.all_features_df.set_index('timestamp').reindex(bars_index).reset_index()
            
            self.normalizer = Normalizer(SETTINGS.strategy)
            self.normalizer.fit(self.bars_df, context_features_df)
            self.normalizer.save(Path(SETTINGS.get_normalizer_path()))
            
            logger.info(f"✅ Loaded {len(self.bars_df)} bars for training")
            logger.info(f"✅ Enhanced Fixed Reward System with live monitoring")
            
            self.num_cpu = SETTINGS.num_workers
            logger.info(f"🚀 Using {self.num_cpu} parallel environments for training.")
            
            if self.enable_live_monitoring:
                logger.info(f"📺 LIVE MONITORING ENABLED - Worker #0 will show detailed trades")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced fixed trainer: {e}")
            raise

    def _make_env(self, rank: int, seed: int = 0, trial_params: Optional[Dict] = None) -> Callable:
        """Environment factory for the Enhanced Fixed Reward System with live monitoring."""
        def _init():
            try:
                config_overrides = {}
                leverage = 10.0
                reward_weights = None
                
                if trial_params:
                    leverage = trial_params.get('leverage', 10.0)
                    
                    reward_weights = {
                        'base_return': trial_params.get('reward_weight_base_return', 2.9),
                        'transaction_penalty': trial_params.get('reward_weight_transaction_penalty', -0.08),
                        'drawdown_penalty': trial_params.get('reward_weight_drawdown_penalty', -0.7),
                        'position_penalty': trial_params.get('reward_weight_position_penalty', -0.03),
                        'exploration_bonus': trial_params.get('reward_weight_exploration_bonus', 0.0),
                        'inactivity_penalty': trial_params.get('reward_weight_inactivity_penalty', -0.45),
                        'frequency_penalty': trial_params.get('reward_weight_frequency_penalty', -0.3),
                    }
                    
                    config_overrides = {
                        'strategy': {
                            'max_margin_allocation_pct': trial_params.get('max_margin_allocation_pct', 0.04),
                            'reward_scaling_factor': trial_params.get('reward_scaling_factor', 100.0),
                            'inactivity_grace_period_steps': trial_params.get('inactivity_grace_period', 8),
                            'penalty_ramp_up_steps': trial_params.get('penalty_ramp_up_steps', 20),
                        }
                    }
                
                trial_specific_config = create_config(**config_overrides) if config_overrides else SETTINGS
                
                # ENHANCED: Pass worker_id to enable verbose mode for worker #0
                env = FixedHierarchicalTradingEnvironment(
                    df_base_ohlc=self.bars_df,
                    normalizer=self.normalizer,
                    config=trial_specific_config,
                    leverage=leverage,
                    reward_weights=reward_weights,
                    precomputed_features=self.all_features_df,
                    worker_id=rank  # NEW: Pass worker ID for verbose mode
                )
                
                env.reset(seed=seed + rank)
                return env
                
            except Exception as e:
                logger.error(f"Error creating environment {rank}: {e}")
                raise
        
        set_random_seed(seed)
        return _init

    def create_model(self, trial_params: Dict, vec_env: VecEnv) -> PPO:
        """Create PPO model with fixed hyperparameters."""
        try:
            arch_config = ModelArchitectureConfig(
                transformer_d_model=trial_params.get('transformer_d_model', 64),
                transformer_n_heads=trial_params.get('transformer_n_heads', 4),
                transformer_dim_feedforward=trial_params.get('transformer_dim_feedforward', 256),
                transformer_num_layers=trial_params.get('transformer_num_layers', 2),
                expert_output_dim=trial_params.get('expert_output_dim', 32),
                attention_head_features=trial_params.get('attention_features', 64),
                dropout_rate=trial_params.get('dropout_rate', 0.1),
            )
            
            logger.info(f"Using Transformer config: d_model={arch_config.transformer_d_model}, n_heads={arch_config.transformer_n_heads}")
            
            policy_kwargs = {
                "features_extractor_class": EnhancedHierarchicalAttentionFeatureExtractor,
                "features_extractor_kwargs": {"arch_cfg": arch_config},
                "net_arch": {"pi": [256, 128], "vf": [256, 128]}
            }
            
            learning_rate = trial_params.get('learning_rate', 3e-4)
            
            model = PPO(
                policy="MultiInputPolicy", env=vec_env, learning_rate=learning_rate,
                n_steps=trial_params.get('n_steps', 2048),
                batch_size=trial_params.get('batch_size', 64),
                n_epochs=trial_params.get('n_epochs', 10),
                gamma=trial_params.get('gamma', 0.99),
                gae_lambda=trial_params.get('gae_lambda', 0.95),
                clip_range=trial_params.get('clip_range', 0.2),
                ent_coef=trial_params.get('ent_coef', 0.01),
                max_grad_norm=trial_params.get('max_grad_norm', 0.5),
                policy_kwargs=policy_kwargs,
                tensorboard_log=SETTINGS.get_tensorboard_path(),
                device=SETTINGS.device,
                seed=trial_params.get('seed', 42),
                verbose=0
            )
            
            logger.info("✅ Created PPO model with enhanced fixed hyperparameters")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

    def objective(self, trial) -> float:
        """
        FIXED: Optuna objective that optimizes for raw trading performance and penalizes
        exploitation of intrinsic rewards.
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna not available for hyperparameter optimization")
        
        vec_env, eval_vec_env = None, None
        
        try:
            trial_params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
                'n_epochs': trial.suggest_int('n_epochs', 5, 15),
                'gamma': trial.suggest_float('gamma', 0.95, 0.999),
                'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
                'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
                'ent_coef': trial.suggest_float('ent_coef', 0.005, 0.05, log=True),
                'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 1.0),
                
                # Transformer architecture
                'transformer_d_model': trial.suggest_categorical('transformer_d_model', [64, 128]),
                'transformer_n_heads': trial.suggest_categorical('transformer_n_heads', [4, 8]),
                'transformer_dim_feedforward': trial.suggest_categorical('transformer_dim_feedforward', [256, 512]),
                'transformer_num_layers': trial.suggest_int('transformer_num_layers', 1, 3),
                'expert_output_dim': trial.suggest_categorical('expert_output_dim', [32, 64]),
                'attention_features': trial.suggest_categorical('attention_features', [64, 128]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.35),
                'seed': trial.suggest_int('seed', 1, 10000),
                
                'leverage': trial.suggest_float('leverage', 5.0, 20.0, step=1.0),
                'max_margin_allocation_pct': trial.suggest_float('max_margin_allocation_pct', 0.01, 0.1, step=0.01),
                'reward_scaling_factor': trial.suggest_float('reward_scaling_factor', 50.0, 200.0),
                
                # Reward weights
                'reward_weight_base_return': trial.suggest_float('reward_weight_base_return', 2.8, 7.0),
                'reward_weight_transaction_penalty': trial.suggest_float('reward_weight_transaction_penalty', -0.4, -0.05),
                'reward_weight_drawdown_penalty': trial.suggest_float('reward_weight_drawdown_penalty', -3.5, -0.6),
                'reward_weight_position_penalty': trial.suggest_float('reward_weight_position_penalty', -0.3, -0.03),
                # --- START OF FIX: Constrain exploration bonus search space ---
                'reward_weight_exploration_bonus': trial.suggest_float('reward_weight_exploration_bonus', 0.0, 0.15),
                # --- END OF FIX ---
                'reward_weight_inactivity_penalty': trial.suggest_float('reward_weight_inactivity_penalty', -2.5, -0.35),
                'reward_weight_frequency_penalty': trial.suggest_float('reward_weight_frequency_penalty', -3.7, -0.3),
                
                # Environment parameters
                'inactivity_grace_period': trial.suggest_int('inactivity_grace_period', 5, 20),
                'penalty_ramp_up_steps': trial.suggest_int('penalty_ramp_up_steps', 10, 50),
            }
            
            vec_env = SubprocVecEnv([self._make_env(i, trial.number, trial_params=trial_params) for i in range(self.num_cpu)])
            model = self.create_model(trial_params, vec_env)
            
            experiment_name = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_path = os.path.join(SETTINGS.get_logs_path(), experiment_name)
            model.set_logger(configure(log_path, ["stdout", "csv", "tensorboard"]))
            
            # Callbacks
            from gymnasium.wrappers import TimeLimit
            eval_env_raw = self._make_env(rank=0, seed=123, trial_params=trial_params)()
            eval_env_limited = TimeLimit(eval_env_raw, max_episode_steps=5000)
            eval_vec_env = DummyVecEnv([lambda: eval_env_limited])
            
            eval_callback = EvalCallback(
                eval_vec_env,
                best_model_save_path=str(Path(log_path) / f"best_model_trial_{trial.number}"),
                log_path=log_path,
                eval_freq=max(5000 // self.num_cpu, 500),
                deterministic=True,
                render=False,
                n_eval_episodes=5
            )
            
            perf_callback = PerformanceMonitoringCallback()
            # --- START OF FIX: Add raw return to HParam metrics ---
            hparam_metrics_to_log = ['eval/mean_reward', 'performance/total_return_raw']
            # --- END OF FIX ---
            hparam_callback = HParamCallback(trial, hparam_metrics_to_log)
            callbacks = [eval_callback, perf_callback, hparam_callback]
            
            if self.use_wandb:
                callbacks.append(WandbCallback("crypto_trading_enhanced_fixed", experiment_name, trial_params))
            
            training_steps = 30720
            model.learn(total_timesteps=training_steps, callback=callbacks, progress_bar=False)
            
            # --- START OF FIX: New scoring logic for Optuna objective ---
            if perf_callback.portfolio_values and perf_callback.raw_rewards:
                portfolio_values_np = np.array(perf_callback.portfolio_values)
                
                # 1. Calculate the raw (extrinsic) performance metric
                total_return_raw = (portfolio_values_np[-1] - portfolio_values_np[0]) / portfolio_values_np[0]
                cumulative_max = np.maximum.accumulate(portfolio_values_np)
                drawdowns = (cumulative_max - portfolio_values_np) / (cumulative_max + 1e-9)
                max_drawdown = np.max(drawdowns)
                
                if max_drawdown > 0.01:
                    risk_adjusted_raw_return = total_return_raw / max_drawdown
                else:
                    risk_adjusted_raw_return = total_return_raw * 10 if total_return_raw >= 0 else -10.0
                
                # 2. Calculate the exploitation of intrinsic rewards
                total_raw_reward = np.sum(perf_callback.raw_rewards)
                total_intrinsic_reward = np.sum(perf_callback.intrinsic_rewards)
                
                # Calculate what fraction of total *positive* reward came from intrinsic sources
                total_positive_reward = total_raw_reward + total_intrinsic_reward
                exploit_fraction = 0.0
                if total_positive_reward > 0:
                    exploit_fraction = total_intrinsic_reward / total_positive_reward

                # 3. Apply a penalty for high exploitation
                # No penalty if intrinsic rewards are negative or make up < 20% of positive reward
                exploit_penalty = max(0, exploit_fraction - 0.20) * 5.0
                
                # 4. Final score is raw performance minus the penalty
                final_score = risk_adjusted_raw_return - exploit_penalty
                
                trial.set_user_attr('total_return_raw', total_return_raw)
                trial.set_user_attr('max_drawdown', max_drawdown)
                trial.set_user_attr('risk_adjusted_raw_return', risk_adjusted_raw_return)
                trial.set_user_attr('exploit_fraction', exploit_fraction)
                trial.set_user_attr('exploit_penalty', exploit_penalty)
                trial.set_user_attr('final_score', final_score)
                
                # Log the raw return for TensorBoard HPARAMS
                model.logger.record('performance/total_return_raw', total_return_raw)
                
                return final_score
            else:
                return -10.0 # Return a poor score if training fails to produce metrics
            # --- END OF FIX ---
                
        except Exception as e:
            logger.error(f"Trial failed with error: {e}")
            return -100.0
        
        finally:
            if vec_env: vec_env.close()
            if eval_vec_env: eval_vec_env.close()

    def optimize(self, n_trials: int = 50, timeout: Optional[int] = None):
        """Run optimization with the Enhanced Fixed Reward System."""
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna not available, cannot run hyperparameter optimization")
            return None
        
        try:
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            
            logger.info(f"🚀 Starting optimization for Enhanced Fixed Reward System (Live Monitoring)")
            
            study.optimize(self.objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
            
            self.best_trial_results = {'best_trial': study.best_trial, 'best_value': study.best_value}
            
            logger.info("🎯 Enhanced Fixed System Optimization Results:")
            logger.info(f" Best final score: {study.best_value:.4f}")
            logger.info(f" Best raw return: {study.best_trial.user_attrs.get('total_return_raw', 'N/A'):.2%}")
            logger.info(f" Best exploit fraction: {study.best_trial.user_attrs.get('exploit_fraction', 'N/A'):.2%}")
            logger.info(f" Best learning rate: {study.best_trial.params.get('learning_rate', 'N/A')}")
            
            return study
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return None

    def train_best_model(self, best_params: Dict, final_training_steps: int) -> PPO:
        """Train the final model with best parameters."""
        try:
            logger.info("🎯 Training final model with enhanced fixed parameters...")
            
            vec_env = SubprocVecEnv([self._make_env(i, seed=42, trial_params=best_params) for i in range(self.num_cpu)])
            model = self.create_model(best_params, vec_env)
            
            final_log_path = SETTINGS.get_logs_path()
            model.set_logger(configure(final_log_path, ["stdout", "csv", "tensorboard"]))
            
            callbacks = [PerformanceMonitoringCallback()]
            
            if self.use_wandb:
                callbacks.append(WandbCallback("crypto_trading_final_enhanced", f"final_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}", best_params))
            
            logger.info(f"Starting final training with {final_training_steps:,} timesteps")
            logger.info("📺 Watch Worker #0 for live trade monitoring...")
            
            model.learn(total_timesteps=final_training_steps, callback=callbacks, progress_bar=True)
            
            model_path_str = SETTINGS.get_model_path()
            model.save(model_path_str)
            logger.info(f"✅ Final model saved to: {model_path_str}")
            
            self.last_saved_model_path = model_path_str
            
            vec_env.close()
            return model
            
        except Exception as e:
            logger.error(f"Failed to train final model: {e}")
            raise

def train_model_fixed(optimization_trials: int = 20,
                     final_training_steps: int = 500000,
                     use_wandb: bool = False,
                     enable_live_monitoring: bool = True) -> str:
    """
    ENHANCED FIXED: Advanced training with live monitoring that prevents turtling and encourages learning.
    Returns the path to the trained model.
    """
    try:
        logger.info("🎯 Starting ENHANCED FIXED Training (Live Monitoring System)")
        logger.info("✅ FIXES APPLIED: Dynamic Rewards, Rebalanced Weights, Market Regimes, Action Regulation")
        
        if enable_live_monitoring:
            logger.info("📺 LIVE MONITORING ENABLED - You'll see detailed trades from Worker #0")
        
        trainer = EnhancedFixedTrainer(use_wandb=use_wandb, enable_live_monitoring=enable_live_monitoring)
        
        if optimization_trials > 0 and OPTUNA_AVAILABLE:
            logger.info(f"Phase 1: Optimization ({optimization_trials} trials)")
            study = trainer.optimize(n_trials=optimization_trials)
            best_params = study.best_trial.params if study else {}
        else:
            logger.info("Skipping optimization, using enhanced default parameters.")
            
            best_params = {
                'learning_rate': 5e-4, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10,
                'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.01,
                'max_grad_norm': 0.5, 'transformer_d_model': 64, 'transformer_n_heads': 4,
                'transformer_dim_feedforward': 256, 'transformer_num_layers': 2,
                'expert_output_dim': 32, 'attention_features': 64, 'dropout_rate': 0.1,
                'seed': 42, 'leverage': 10.0, 'max_margin_allocation_pct': 0.04,
                'reward_scaling_factor': 100.0,
                'reward_weight_base_return': 2.9, 'reward_weight_transaction_penalty': -0.08,
                'reward_weight_drawdown_penalty': -0.7, 'reward_weight_position_penalty': -0.03,
                'reward_weight_exploration_bonus': 0.0, 'reward_weight_inactivity_penalty': -0.45,
                'reward_weight_frequency_penalty': -0.8, # Stronger default
                'inactivity_grace_period': 10, 'penalty_ramp_up_steps': 20,
            }
        
        logger.info(f"Phase 2: Final Training ({final_training_steps:,} steps)")
        trainer.train_best_model(best_params, final_training_steps)
        
        logger.info("🎉 ENHANCED FIXED training completed successfully!")
        logger.info(f"Model saved to: {trainer.last_saved_model_path}")
        
        return trainer.last_saved_model_path
        
    except Exception as e:
        logger.error(f"Enhanced fixed training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    
    try:
        model_path = train_model_fixed(
            optimization_trials=10,
            final_training_steps=100000,
            use_wandb=False,
            enable_live_monitoring=True  # Enable live monitoring
        )
        
        logger.info(f"✅ ENHANCED FIXED training completed successfully!")
        logger.info(f"Final model available at: {model_path}")
        
    except Exception as e:
        logger.error(f"Enhanced fixed training example failed: {e}")
        raise