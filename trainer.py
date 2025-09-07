# FILE: Zero1-main/trainer.py (FINALIZED FOR CENTRALIZED CONFIG AND PERFORMANCE)

"""
FINALIZED: Trainer with Centralized Config, Immediate Reward System, and Performance Fixes.

KEY FEATURES:
1. Reads all default parameters, including reward weights, directly from config.py,
   ensuring a single source of truth.
2. Incorporates the 'inactivity_penalty' into the Optuna search space.
3. Uses SubprocVecEnv for evaluation and robust try/finally blocks to ensure
   proper resource cleanup, combating FPS drop issues.
4. Aligned with the NumPy-optimized engine for high-performance training.
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

from processor import create_bars_from_trades, generate_stateful_features_for_fitting
from config import SETTINGS, create_config, ModelArchitectureConfig
from tins import EnhancedHierarchicalAttentionFeatureExtractor
from engine import FixedHierarchicalTradingEnvironment as EnhancedHierarchicalTradingEnvironment
from normalizer import Normalizer

logger = logging.getLogger(__name__)

class PerformanceMonitoringCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.portfolio_values = []
        self.peak_value = 0.0

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0 and self.locals.get('infos'):
            info = self.locals['infos'][0]
            for component, value in info.get('reward_components', {}).items():
                self.logger.record(f'reward_components/{component}', value)
            portfolio_value = info.get('portfolio_value', 0)
            if portfolio_value > 0:
                self.portfolio_values.append(portfolio_value)
                if portfolio_value > self.peak_value: self.peak_value = portfolio_value
                drawdown = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0
                self.logger.record('performance/portfolio_value', portfolio_value)
                self.logger.record('performance/drawdown', drawdown)
        return True

class WandbCallback(BaseCallback):
    def __init__(self, project_name: str, experiment_name: str, config: Dict, verbose: int = 0):
        super().__init__(verbose)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.wandb_run = None

    def _on_training_start(self) -> None:
        if WANDB_AVAILABLE:
            self.wandb_run = wandb.init(project=self.project_name, name=self.experiment_name, config=self.config, reinit=True)

    def _on_step(self) -> bool:
        if self.wandb_run and self.n_calls % 100 == 0:
            log_dict = {k: v for k, v in self.logger.name_to_value.items() if isinstance(v, (int, float))}
            if log_dict: wandb.log(log_dict, step=self.n_calls)
        return True

    def _on_training_end(self) -> None:
        if self.wandb_run: self.wandb_run.finish()

class OptimizedTrainer:
    def __init__(self, use_wandb: bool = False):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.last_saved_model_path = None

        logger.info("🔄 Pre-loading and preparing all training data...")
        self.bars_df = create_bars_from_trades("in_sample")
        self.features_df = generate_stateful_features_for_fitting(self.bars_df, SETTINGS.strategy)
        
        bars_index = self.bars_df.set_index('timestamp').index
        features_index = self.features_df.set_index('timestamp').index
        if not bars_index.equals(features_index):
            self.features_df = self.features_df.set_index('timestamp').reindex(bars_index).reset_index().fillna(0.0)

        self.normalizer = Normalizer(SETTINGS.strategy)
        self.normalizer.fit(self.bars_df, self.features_df)
        self.normalizer.save(Path(SETTINGS.get_normalizer_path()))
        
        self.num_cpu = min(mp.cpu_count(), SETTINGS.num_workers)
        logger.info(f"✅ Data loaded. Using {self.num_cpu} parallel environments.")

    def _make_env(self, rank: int, seed: int = 0, trial_params: Optional[Dict] = None) -> Callable:
        def _init():
            try:
                leverage, reward_weights = None, None
                config_overrides = {}
                if trial_params:
                    leverage = trial_params.get('leverage')
                    reward_weights = {k: v for k, v in trial_params.items() if k.startswith('reward_weight_')}
                    reward_weights = {k.replace('reward_weight_', ''): v for k, v in reward_weights.items()}
                    config_overrides['strategy'] = {'reward_scaling_factor': trial_params.get('reward_scaling_factor')}
                
                config = create_config(**{k:v for k,v in config_overrides.items() if v is not None})
                
                env = EnhancedHierarchicalTradingEnvironment(
                    self.bars_df, self.normalizer, config, leverage, reward_weights, self.features_df)
                env.reset(seed=seed + rank)
                return env
            except Exception as e:
                logger.error(f"Error creating env rank {rank}: {e}", exc_info=True)
                raise
        set_random_seed(seed)
        return _init

    def create_model(self, trial_params: Dict, vec_env: VecEnv) -> PPO:
        arch_config = ModelArchitectureConfig(**{k:v for k,v in trial_params.items() if k.startswith('transformer_')})
        policy_kwargs = {"features_extractor_class": EnhancedHierarchicalAttentionFeatureExtractor,
                         "features_extractor_kwargs": {"arch_cfg": arch_config}}
        return PPO("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs,
                   learning_rate=trial_params.get('learning_rate', 3e-4),
                   n_steps=trial_params.get('n_steps', 2048),
                   batch_size=trial_params.get('batch_size', 64),
                   n_epochs=trial_params.get('n_epochs', 10),
                   gamma=trial_params.get('gamma', 0.99),
                   ent_coef=trial_params.get('ent_coef', 0.01),
                   tensorboard_log=SETTINGS.get_tensorboard_path(),
                   device=SETTINGS.device, verbose=0)

    def objective(self, trial) -> float:
        vec_env, eval_vec_env = None, None
        try:
            trial_params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                'n_steps': trial.suggest_categorical('n_steps', [2048, 4096]),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
                'gamma': trial.suggest_float('gamma', 0.95, 0.999),
                'ent_coef': trial.suggest_float('ent_coef', 1e-7, 1e-2, log=True),
                'leverage': trial.suggest_float('leverage', 5.0, 20.0, step=1.0),
                'reward_scaling_factor': trial.suggest_float('reward_scaling_factor', 50.0, 500.0),
                'reward_weight_base_return': trial.suggest_float('reward_weight_base_return', 1.0, 3.0),
                'reward_weight_drawdown_penalty': trial.suggest_float('reward_weight_drawdown_penalty', -0.5, 0.0),
                'reward_weight_inactivity_penalty': trial.suggest_float('reward_weight_inactivity_penalty', -0.01, 0.0)
            }
            vec_env = SubprocVecEnv([self._make_env(i, trial.number, trial_params) for i in range(self.num_cpu)])
            model = self.create_model(trial_params, vec_env)
            
            from gymnasium.wrappers import TimeLimit
            eval_env_fn = lambda: TimeLimit(self._make_env(0, 123, trial_params)(), max_episode_steps=5000)
            eval_vec_env = SubprocVecEnv([eval_env_fn])
            
            perf_callback = PerformanceMonitoringCallback()
            eval_callback = EvalCallback(eval_vec_env, best_model_save_path=f"./data/logs/best_model_trial_{trial.number}",
                                         log_path="./data/logs/", eval_freq=max(5000 // self.num_cpu, 500),
                                         n_eval_episodes=3, deterministic=True)
            callbacks = [perf_callback, eval_callback]
            if self.use_wandb:
                callbacks.append(WandbCallback("zero1_crypto_rl", f"trial_{trial.number}", trial_params))

            model.learn(total_timesteps=30720, callback=callbacks, progress_bar=False)
            
            if perf_callback.portfolio_values:
                p_vals = np.array(perf_callback.portfolio_values)
                ret, cmax = (p_vals[-1] - p_vals[0]) / p_vals[0], np.maximum.accumulate(p_vals)
                mdd = np.max((cmax - p_vals) / cmax)
                score = (ret / mdd) if mdd > 0.01 else (ret * 10)
                trial.set_user_attr('total_return', ret); trial.set_user_attr('max_drawdown', mdd)
                return score
            return -10.0
        except Exception as e:
            logger.error(f"Trial failed: {e}", exc_info=True)
            return -100.0
        finally:
            if vec_env: vec_env.close()
            if eval_vec_env: eval_vec_env.close()

    def optimize(self, n_trials: int = 50):
        if not OPTUNA_AVAILABLE: return None
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        logger.info(f"🚀 Starting optimization ({n_trials} trials)")
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        logger.info(f"🎯 Optimization complete. Best score: {study.best_value:.4f}")
        return study

    def train_best_model(self, best_params: Dict, final_training_steps: int):
        vec_env = None
        try:
            logger.info("🎯 Training final model with best parameters...")
            vec_env = SubprocVecEnv([self._make_env(i, 42, best_params) for i in range(self.num_cpu)])
            model = self.create_model(best_params, vec_env)
            model.learn(total_timesteps=final_training_steps, progress_bar=True)
            model_path_str = SETTINGS.get_model_path()
            model.save(model_path_str)
            logger.info(f"✅ Final model saved to: {model_path_str}")
            self.last_saved_model_path = model_path_str
        finally:
            if vec_env: vec_env.close()

def train_model_advanced(optimization_trials: int = 20,
                        final_training_steps: int = 500000,
                        use_wandb: bool = False) -> str:
    try:
        logger.info("🎯 Starting FINALIZED Training with Centralized Config")
        trainer = OptimizedTrainer(use_wandb=use_wandb)
        best_params = {}
        if optimization_trials > 0:
            study = trainer.optimize(n_trials=optimization_trials)
            if study: best_params = study.best_trial.params
        else:
            logger.info("Skipping optimization, using default parameters from config.py.")
        
        trainer.train_best_model(best_params, final_training_steps)
        logger.info("🎉 Training pipeline completed!")
        return trainer.last_saved_model_path
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    try:
        model_path = train_model_advanced(optimization_trials=10, final_training_steps=100000)
        logger.info(f"✅ Training completed successfully! Model at: {model_path}")
    except Exception as e:
        logger.error(f"Training example failed: {e}", exc_info=True)
        raise
