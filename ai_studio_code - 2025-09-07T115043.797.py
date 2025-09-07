# FILE: Zero1-main/trainer.py (FULLY UPDATED FOR IMMEDIATE REWARDS)

"""
Enhanced Trainer with Transformer Architecture Integration and Immediate Rewards.

This version has been simplified by removing the flawed, delayed reward horizon
logic to ensure correct credit assignment for the PPO algorithm. The trainer
now focuses on an immediate, step-by-step reward signal, which is compatible
with Stable Baselines 3's PPO implementation.
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.utils import set_random_seed
import os
import multiprocessing as mp
import logging

# Optional dependencies with fallbacks
try:
    import optuna
    from gymnasium.wrappers import TimeLimit # Optuna-related import
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
from processor import create_bars_from_trades, generate_stateful_features_for_fitting
from config import SETTINGS, create_config, ModelArchitectureConfig
from tins import EnhancedHierarchicalAttentionFeatureExtractor
from engine import EnhancedHierarchicalTradingEnvironment
from normalizer import Normalizer

logger = logging.getLogger(__name__)


class PerformanceMonitoringCallback(BaseCallback):
    """
    Enhanced performance monitoring callback. Logs key metrics like portfolio value,
    drawdown, and reward components at each step.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.portfolio_values = []
        self.peak_value = 0

    def _on_step(self) -> bool:
        """Monitor performance metrics during training."""
        try:
            if self.locals.get('infos'):
                info = self.locals['infos'][0] if self.locals['infos'] else {}
                
                # Log portfolio performance
                portfolio_value = info.get('portfolio_value', 0)
                if portfolio_value > 0:
                    self.portfolio_values.append(portfolio_value)
                    if not self.peak_value: self.peak_value = portfolio_value
                    if portfolio_value > self.peak_value:
                        self.peak_value = portfolio_value
                    
                    current_drawdown = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0
                    
                    if self.n_calls % 1000 == 0:
                        self.logger.record('performance/portfolio_value', portfolio_value)
                        self.logger.record('performance/drawdown', current_drawdown)
                        self.logger.record('performance/peak_value', self.peak_value)
                        
                        if len(self.portfolio_values) > 100:
                            recent_values = self.portfolio_values[-100:]
                            recent_return = (recent_values[-1] - recent_values[0]) / recent_values[0]
                            self.logger.record('performance/recent_return_100', recent_return)
                
                # Log reward components for detailed analysis
                reward_components = info.get('reward_components', {})
                if reward_components:
                    for component, value in reward_components.items():
                        if isinstance(value, (int, float)):
                            self.logger.record(f'reward_components/{component}', value)
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
        if WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(project=self.project_name, name=self.experiment_name, config=self.config, reinit=True)
                logger.info(f"Started W&B run: {self.wandb_run.name}")
            except Exception as e:
                logger.error(f"Failed to initialize W&B: {e}")

    def _on_step(self) -> bool:
        if self.wandb_run and self.n_calls % 1000 == 0:
            try:
                for key, value in self.logger.name_to_value.items():
                    if isinstance(value, (int, float)):
                        wandb.log({key: value}, step=self.n_calls)
            except Exception as e:
                logger.error(f"Error logging to W&B: {e}")
        return True

    def _on_training_end(self) -> None:
        if self.wandb_run:
            wandb.finish()

class OptimizedTrainer:
    """
    Trainer with Transformer architecture integration, hyperparameter optimization,
    and a correct immediate reward structure.
    """
    def __init__(self, use_wandb: bool = False):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.best_trial_results = None
        self.last_saved_model_path = None

        try:
            logger.info("🔄 Pre-loading and preparing training data...")
            self.bars_df = create_bars_from_trades("in_sample")
            
            logger.info("Generating stateful features for normalizer fitting...")
            self.features_df = generate_stateful_features_for_fitting(self.bars_df, SETTINGS.strategy)
            
            self.normalizer = Normalizer(SETTINGS.strategy)
            self.normalizer.fit(self.bars_df, self.features_df)
            self.normalizer.save(Path(SETTINGS.get_normalizer_path()))
            
            logger.info(f"✅ Loaded {len(self.bars_df)} bars for training.")
            self.num_cpu = min(os.cpu_count(), 8)
            logger.info(f"🚀 Using {self.num_cpu} parallel environments for training.")
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {e}", exc_info=True)
            raise

    def _make_env(self, rank: int, seed: int = 0, trial_params: Optional[Dict] = None) -> Callable:
        """Environment factory for creating parallel training environments."""
        def _init():
            try:
                leverage = trial_params.get('leverage', 10.0) if trial_params else 10.0
                
                reward_weights = None
                if trial_params:
                    reward_weights = {k.replace('reward_weight_', ''): v for k, v in trial_params.items() if k.startswith('reward_weight_')}
                
                config_overrides = {}
                if trial_params:
                    config_overrides['strategy'] = {'max_margin_allocation_pct': trial_params.get('max_margin_allocation_pct', 0.04)}
                
                trial_specific_config = create_config(**config_overrides) if config_overrides else SETTINGS

                env = EnhancedHierarchicalTradingEnvironment(
                    df_base_ohlc=self.bars_df, normalizer=self.normalizer, config=trial_specific_config,
                    leverage=leverage, reward_weights=reward_weights, precomputed_features=self.features_df
                )
                env.reset(seed=seed + rank)
                return env
            except Exception as e:
                logger.error(f"Error creating environment {rank}: {e}", exc_info=True)
                raise
        set_random_seed(seed)
        return _init

    def create_model(self, trial_params: Dict, vec_env: VecEnv) -> PPO:
        """Create PPO model with Transformer architecture parameters."""
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
            policy_kwargs = {
                "features_extractor_class": EnhancedHierarchicalAttentionFeatureExtractor,
                "features_extractor_kwargs": {"arch_cfg": arch_config},
                "net_arch": {"pi": [256, 128], "vf": [256, 128]}
            }
            learning_rate = trial_params.get('learning_rate', 3e-4)
            return PPO("MultiInputPolicy", env=vec_env, learning_rate=learning_rate,
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
                       verbose=0)
        except Exception as e:
            logger.error(f"Failed to create model: {e}", exc_info=True)
            raise

    def objective(self, trial) -> float:
        """Optuna objective function for hyperparameter tuning."""
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna not available for hyperparameter optimization")
        vec_env, eval_vec_env = None, None
        try:
            trial_params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
                'n_epochs': trial.suggest_int('n_epochs', 5, 20),
                'gamma': trial.suggest_float('gamma', 0.95, 0.999),
                'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
                'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
                'ent_coef': trial.suggest_float('ent_coef', 1e-7, 1e-2, log=True),
                'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 1.0),
                'transformer_d_model': trial.suggest_categorical('transformer_d_model', [64, 128]),
                'transformer_n_heads': trial.suggest_categorical('transformer_n_heads', [4, 8]),
                'transformer_dim_feedforward': trial.suggest_categorical('transformer_dim_feedforward', [256, 512]),
                'transformer_num_layers': trial.suggest_int('transformer_num_layers', 2, 4),
                'expert_output_dim': trial.suggest_categorical('expert_output_dim', [32, 64]),
                'attention_features': trial.suggest_categorical('attention_features', [64, 128]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.05, 0.3),
                'seed': trial.suggest_int('seed', 1, 10000),
                'leverage': trial.suggest_float('leverage', 5.0, 20.0, step=1.0),
                'max_margin_allocation_pct': trial.suggest_float('max_margin_allocation_pct', 0.02, 0.1, step=0.01),
                'reward_weight_base_return': trial.suggest_float('reward_weight_base_return', 1.0, 2.5),
                'reward_weight_risk_adjusted': trial.suggest_float('reward_weight_risk_adjusted', 0.05, 0.5),
                'reward_weight_stability': trial.suggest_float('reward_weight_stability', 0.05, 0.4),
                'reward_weight_transaction_penalty': trial.suggest_float('reward_weight_transaction_penalty', -0.2, -0.01),
                'reward_weight_drawdown_penalty': trial.suggest_float('reward_weight_drawdown_penalty', -0.8, -0.2),
                'reward_weight_position_penalty': trial.suggest_float('reward_weight_position_penalty', -0.1, -0.01),
                'reward_weight_risk_bonus': trial.suggest_float('reward_weight_risk_bonus', 0.1, 0.4),
            }
            
            vec_env = SubprocVecEnv([self._make_env(i, trial.number, trial_params=trial_params) for i in range(self.num_cpu)])
            model = self.create_model(trial_params, vec_env)
            
            perf_callback = PerformanceMonitoringCallback()
            callbacks = [perf_callback]
            
            if self.use_wandb:
                callbacks.append(WandbCallback("crypto_trading_rl_opt", f"trial_{trial.number}", {'trial_number': trial.number, **trial_params}))

            # Train for a fixed, short duration for each trial
            model.learn(total_timesteps=20480, callback=callbacks, progress_bar=False)

            if perf_callback.portfolio_values:
                portfolio_values_np = np.array(perf_callback.portfolio_values)
                total_return = (portfolio_values_np[-1] - portfolio_values_np[0]) / portfolio_values_np[0]
                
                cumulative_max = np.maximum.accumulate(portfolio_values_np)
                drawdowns = (cumulative_max - portfolio_values_np) / (cumulative_max + 1e-9)
                max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 1.0
                
                # Objective: Calmar-like ratio (Return over Max Drawdown)
                # Penalize trials that don't trade or have zero drawdown.
                final_score = total_return / (max_drawdown + 0.01)
                
                trial.set_user_attr('total_return', total_return)
                trial.set_user_attr('max_drawdown', max_drawdown)
                return final_score
            else:
                return -20.0 # Heavily penalize trials that fail to produce data
        except Exception as e:
            if OPTUNA_AVAILABLE and isinstance(e, optuna.TrialPruned): raise
            logger.error(f"Trial failed with error: {e}", exc_info=True)
            return -30.0
        finally:
            if vec_env: vec_env.close()

    def optimize(self, n_trials: int = 50, timeout: Optional[int] = None):
        """Run hyperparameter optimization using Optuna."""
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna not available, cannot run optimization.")
            return None
        try:
            study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
            logger.info(f"🚀 Starting optimization with {n_trials} trials...")
            study.optimize(self.objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
            self.best_trial_results = {'best_trial': study.best_trial, 'best_value': study.best_value}
            logger.info(f"🎯 Optimization complete. Best score: {study.best_value:.4f}")
            logger.info("Best parameters:")
            for key, value in study.best_trial.params.items():
                logger.info(f"  - {key}: {value}")
            return study
        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            return None

    def train_best_model(self, best_params: Dict, final_training_steps: int) -> PPO:
        """Train the final model with the best found parameters."""
        try:
            logger.info("🎯 Training final model with best parameters...")
            vec_env = SubprocVecEnv([self._make_env(i, seed=42, trial_params=best_params) for i in range(self.num_cpu)])
            model = self.create_model(best_params, vec_env)
            
            callbacks = [PerformanceMonitoringCallback()]
            if self.use_wandb:
                callbacks.append(WandbCallback("crypto_trading_rl_final", f"final_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}", best_params))

            logger.info(f"Starting final training with {final_training_steps:,} timesteps")
            model.learn(total_timesteps=final_training_steps, callback=callbacks, progress_bar=True)
            
            model_path_str = SETTINGS.get_model_path()
            model.save(model_path_str)
            logger.info(f"✅ Final model saved to: {model_path_str}")
            
            config_path = model_path_str.replace(".zip", "_config.json")
            with open(config_path, 'w') as f:
                f.write(SETTINGS.json(indent=4))
            logger.info(f"✅ Configuration snapshot saved to: {config_path}")
            
            self.last_saved_model_path = model_path_str
            vec_env.close()
            return model
        except Exception as e:
            logger.error(f"Failed to train final model: {e}", exc_info=True)
            raise

def train_model_advanced(optimization_trials: int = 20,
                        final_training_steps: int = 500000,
                        use_wandb: bool = False) -> str:
    """Main training function with immediate rewards and optional optimization."""
    try:
        logger.info("🎯 Starting Simplified Training with Immediate Rewards")
        trainer = OptimizedTrainer(use_wandb=use_wandb)
        
        if optimization_trials > 0 and OPTUNA_AVAILABLE:
            study = trainer.optimize(n_trials=optimization_trials)
            if study and study.best_trial:
                best_params = study.best_trial.params
            else:
                logger.warning("Optimization failed or was skipped. Using default parameters.")
                best_params = {} # Will use defaults in create_model
        else:
            logger.info("Skipping optimization, using default parameters.")
            best_params = {}

        # Use default values for any missing keys
        default_params = {
            'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10,
            'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.01,
            'max_grad_norm': 0.5, 'transformer_d_model': 64, 'transformer_n_heads': 4,
            'transformer_dim_feedforward': 256, 'transformer_num_layers': 2,
            'expert_output_dim': 32, 'attention_features': 64, 'dropout_rate': 0.1,
            'seed': 42, 'leverage': 10.0, 'max_margin_allocation_pct': 0.04,
        }
        final_params = {**default_params, **best_params}

        logger.info(f"Phase 2: Final Training ({final_training_steps:,} steps)")
        trainer.train_best_model(final_params, final_training_steps)
        
        logger.info("🎉 Training with immediate rewards completed!")
        return trainer.last_saved_model_path
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    try:
        # Example: Run a short optimization and then a final training run
        model_path = train_model_advanced(
            optimization_trials=10, 
            final_training_steps=100000,
            use_wandb=False
        )
        logger.info(f"✅ Training completed successfully! Model available at: {model_path}")
    except Exception as e:
        logger.error(f"Training example failed: {e}", exc_info=True)
        raise