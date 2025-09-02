"""
Enhanced Training System for Crypto Trading RL - FIXED VERSION



"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, StopTrainingOnRewardThreshold,
    StopTrainingOnMaxEpisodes, CheckpointCallback
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
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

# Import from local modules - fixed import paths
from processor import create_bars_from_trades, EnhancedDataProcessor, generate_stateful_features_for_fitting
from config import SETTINGS, create_config, Environment
from tins import EnhancedHierarchicalAttentionFeatureExtractor
from engine import EnhancedHierarchicalTradingEnvironment  # ✅ FIXED: Import from fixed engine
from normalizer import Normalizer

logger = logging.getLogger(__name__)

# --- ADVANCED CALLBACKS FOR MONITORING ---

class WandbCallback(BaseCallback):
    """Weights & Biases integration for experiment tracking."""
    
    def __init__(self, project_name: str, experiment_name: str, config: Dict[str, Any], verbose: int = 0):
        super().__init__(verbose)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.run = None

    def _on_training_start(self) -> None:
        """Initialize W&B run at training start."""
        try:
            if WANDB_AVAILABLE and not self.run:
                self.run = wandb.init(
                    project=self.project_name,
                    name=self.experiment_name,
                    config=self.config,
                    sync_tensorboard=True
                )
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")

    def _on_step(self) -> bool:
        """Log metrics at each step."""
        try:
            if self.run and self.locals.get('infos'):
                # For vectorized environments, infos is a list
                info = self.locals['infos'][0] if self.locals['infos'] else {}
                
                # Log training metrics
                if 'episode' in self.locals:
                    episode_info = self.locals['episode']
                    if episode_info:
                        self.run.log({
                            'episode/reward': episode_info.get('r', 0),
                            'episode/length': episode_info.get('l', 0),
                            'episode/time': episode_info.get('t', 0)
                        })
                
                # Log portfolio metrics if available
                if 'portfolio_value' in info:
                    self.run.log({
                        'portfolio/value': info['portfolio_value'],
                        'portfolio/balance': info.get('balance', 0),
                        'portfolio/asset_held': info.get('asset_held', 0),
                        'portfolio/drawdown': info.get('drawdown', 0),
                        'portfolio/volatility': info.get('volatility', 0),
                        'portfolio/leverage': info.get('leverage', 10.0)  # ✅ FIXED: Log leverage
                    })
                    
            return True
        except Exception as e:
            logger.error(f"Error in W&B callback step: {e}")
            return True

    def _on_training_end(self) -> None:
        """Finish W&B run."""
        try:
            if self.run:
                self.run.finish()
        except Exception as e:
            logger.error(f"Error finishing W&B run: {e}")

class LiveRLMonitoringCallback(BaseCallback):
    """
    Callback for live monitoring of RL training, logging key metrics at a set interval.
    This logs directly to the SB3 logger, which is then picked up by TensorBoard and
    W&B (if sync_tensorboard is enabled).
    """
    
    def __init__(self, log_interval_seconds: int = 30, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval_seconds = log_interval_seconds
        self.last_log_time = 0

    def _on_step(self) -> bool:
        """
        On each step, check if it's time to log, and if so, record metrics.
        """
        current_time = time.time()
        
        # Log only if the interval has passed
        if current_time - self.last_log_time < self.log_interval_seconds:
            return True
            
        self.last_log_time = current_time
        
        if self.locals.get('infos'):
            # For vectorized environments, infos is a list
            info = self.locals['infos'][0] if self.locals['infos'] else {}
            reward = self.locals['rewards'][0]
            action = self.locals['actions'][0]
            
            # Log all requested metrics to the SB3 logger
            self.logger.record('reward/step_reward', reward)
            self.logger.record('unrealized_pnl', info.get('unrealized_pnl', 0))
            self.logger.record('actions/signal', action[0])
            self.logger.record('actions/position_size', action[1])
            self.logger.record('risk/used_margin', info.get('used_margin', 0))
            self.logger.record('behavior/step_win_rate', info.get('step_win_rate', 0))
            self.logger.record('behavior/trade_count', info.get('trade_count', 0))
            self.logger.record('performance_adv/sharpe_ratio_live', info.get('sharpe_ratio_live', 0))
            # ✅ FIXED: Log leverage and reward scaling info
            self.logger.record('config/leverage', info.get('leverage', 10.0))
            self.logger.record('config/reward_scaling_factor', info.get('reward_scaling_factor', 20.0))
            
        return True

class AttentionAnalysisCallback(BaseCallback):
    """Callback to analyze and log attention patterns."""
    
    def __init__(self, log_frequency: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_frequency = log_frequency
        self.attention_history = []

    def _on_step(self) -> bool:
        """Analyze attention patterns periodically."""
        try:
            if self.n_calls % self.log_frequency == 0:
                # Extract attention weights from the model
                if hasattr(self.model.policy.features_extractor, 'get_attention_analysis'):
                    analysis = self.model.policy.features_extractor.get_attention_analysis()
                    if analysis:
                        self.attention_history.append({
                            'step': self.n_calls,
                            'analysis': analysis
                        })
                        
                        # Log to tensorboard
                        if 'expert_weights' in analysis:
                            for expert_name, weights in analysis['expert_weights'].items():
                                self.logger.record(f'attention/{expert_name}_weight', np.mean(weights))
                                
                        if 'attention_entropy' in analysis:
                            self.logger.record('attention/entropy', np.mean(analysis['attention_entropy']))
                            
            return True
        except Exception as e:
            logger.error(f"Error in attention analysis callback: {e}")
            return True

class PerformanceMonitoringCallback(BaseCallback):
    """Advanced performance monitoring and alerting."""
    
    def __init__(self, performance_threshold: float = -0.1,
                 drawdown_threshold: float = 0.2, alert_callback: Optional[Callable] = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.performance_threshold = performance_threshold
        self.drawdown_threshold = drawdown_threshold
        self.alert_callback = alert_callback
        self.episode_returns = []
        self.portfolio_values = []
        self.peak_value = 0

    def _on_step(self) -> bool:
        """Monitor performance metrics."""
        try:
            if self.locals.get('infos'):
                # For vectorized environments, infos is a list
                info = self.locals['infos'][0] if self.locals['infos'] else {}
                
                if 'portfolio_value' in info:
                    portfolio_value = info['portfolio_value']
                    self.portfolio_values.append(portfolio_value)
                    
                    # Update peak value
                    if portfolio_value > self.peak_value:
                        self.peak_value = portfolio_value
                        
                    # Calculate current drawdown
                    current_drawdown = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0
                    
                    # Log performance metrics
                    self.logger.record('performance/portfolio_value', portfolio_value)
                    self.logger.record('performance/drawdown', current_drawdown)
                    self.logger.record('performance/peak_value', self.peak_value)
                    
                    # Check for performance alerts
                    if current_drawdown > self.drawdown_threshold:
                        if self.alert_callback:
                            self.alert_callback(f"High drawdown detected: {current_drawdown:.2%}")
                        if self.verbose > 0:
                            logger.warning(f"Warning: Drawdown of {current_drawdown:.2%} exceeds threshold")
                            
            return True
        except Exception as e:
            logger.error(f"Error in performance monitoring callback: {e}")
            return True

# --- OPTIMIZED TRAINER WITH IMPROVEMENTS ---

class OptimizedTrainer:
    """✅ FIXED: Enhanced trainer with tunable reward weights, leverage, and learning dynamics"""
    
    def __init__(self, base_config: Optional[Dict] = None, use_wandb: bool = False):
        self.base_config = base_config or SETTINGS.dict()
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.best_trial_results = None
        
        try:
            # Pre-load data once
            logger.info("🔄 Pre-loading training data...")
            processor = EnhancedDataProcessor(config=SETTINGS)
            
            # We need bars for the environment and features for fitting the normalizer
            self.bars_df = processor.create_enhanced_bars_from_trades("in_sample")
            
            # Generate context features for normalizer fitting (vectorized)
            logger.info("Generating context features for normalizer fitting (vectorized)...")
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
                
            logger.info(f" -> Generated features_df for normalization with shape: {self.features_df.shape}")
            
            # FIT AND SAVE THE NORMALIZER
            self.normalizer = Normalizer(SETTINGS.strategy)
            self.normalizer.fit(self.bars_df, self.features_df)
            self.normalizer.save(Path(SETTINGS.get_normalizer_path()))
            
            logger.info(f"✅ Loaded {len(self.bars_df)} bars for training and fitted normalizer.")
            
            # Determine number of parallel workers
            self.num_cpu = min(os.cpu_count(), 8)
            logger.info(f"🚀 Using {self.num_cpu} parallel environments for training.")
            
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {e}")
            raise

    def _make_env(self, rank: int, seed: int = 0, trial_params: Optional[Dict] = None) -> Callable:
        """✅ FIXED: Utility function for multiprocessing environments with tunable parameters."""
        def _init():
            try:
                config_overrides = {}
                
                # ✅ FIXED: Extract tunable parameters for environment
                leverage = trial_params.get('leverage', 10.0) if trial_params else 10.0
                
                # ✅ FIXED: Extract reward weights from trial_params
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
                
                if trial_params and 'max_margin_allocation_pct' in trial_params:
                    config_overrides['strategy'] = {'max_margin_allocation_pct': trial_params['max_margin_allocation_pct']}
                
                # Create a config instance specific to this environment process
                trial_specific_config = create_config(**config_overrides) if config_overrides else SETTINGS
                
                # ✅ FIXED: Pass leverage and reward_weights to the fixed environment
                env = EnhancedHierarchicalTradingEnvironment(
                    df_base_ohlc=self.bars_df,
                    normalizer=self.normalizer,
                    config=trial_specific_config,
                    leverage=leverage,
                    reward_weights=reward_weights
                )
                
                env.reset(seed=seed + rank)
                return env
                
            except Exception as e:
                logger.error(f"Error creating environment {rank}: {e}")
                raise
                
        set_random_seed(seed)
        return _init

    def create_model(self, trial_params: Dict, vec_env: VecEnv) -> PPO:
        """✅ FIXED: Create PPO model with tunable hyperparameters including target_kl and learning rate schedule."""
        try:
            # Extract hyperparameters
            learning_rate = trial_params.get('learning_rate', 3e-4)
            n_steps = trial_params.get('n_steps', 2048)
            batch_size = trial_params.get('batch_size', 64)
            n_epochs = trial_params.get('n_epochs', 10)
            gamma = trial_params.get('gamma', 0.99)
            gae_lambda = trial_params.get('gae_lambda', 0.95)
            clip_range = trial_params.get('clip_range', 0.2)
            ent_coef = trial_params.get('ent_coef', 0.01)
            max_grad_norm = trial_params.get('max_grad_norm', 0.5)
            
            # ✅ FIXED: Tunable learning rate schedule
            learning_rate_schedule = trial_params.get('learning_rate_schedule', 'linear')
            
            # ✅ FIXED: Tunable target_kl for policy stability
            target_kl = trial_params.get('target_kl', None)
            
            # Architecture hyperparameters
            lstm_layers = trial_params.get('lstm_layers', 2)
            expert_hidden_size = trial_params.get('expert_hidden_size', 32)
            attention_features = trial_params.get('attention_features', 64)
            dropout_rate = trial_params.get('dropout_rate', 0.1)
            lstm_hidden_size = trial_params.get('lstm_hidden_size', 64)
            
            # Ensure batch_size <= n_steps
            if batch_size > n_steps:
                batch_size = n_steps // 2
                
            # Update architecture config
            arch_config = {
                'lstm_layers': lstm_layers,
                'expert_lstm_hidden_size': expert_hidden_size,
                'attention_head_features': attention_features,
                'dropout_rate': dropout_rate
            }
            
            # Policy configuration
            policy_kwargs = {
                'features_extractor_class': EnhancedHierarchicalAttentionFeatureExtractor,
                'features_extractor_kwargs': {'arch_cfg': type('Config', (), arch_config)()},
                'lstm_hidden_size': lstm_hidden_size,
                'net_arch': {
                    'pi': [expert_hidden_size * 2, expert_hidden_size],
                    'vf': [expert_hidden_size * 2, expert_hidden_size]
                }
            }
            
            # ✅ FIXED: Create model with tunable learning dynamics
            model = PPO(
                "MlpLstmPolicy",
                vec_env,
                policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                max_grad_norm=max_grad_norm,
                target_kl=target_kl,  # ✅ FIXED: Enable target_kl for policy stability
                verbose=0,
                device=SETTINGS.device,
                seed=trial_params.get('seed', 42)
            )
            
            # ✅ FIXED: Set learning rate schedule
            if learning_rate_schedule == 'constant':
                model.lr_schedule = lambda _: learning_rate
            elif learning_rate_schedule == 'cosine':
                def cosine_schedule(progress_remaining):
                    return learning_rate * 0.5 * (1 + np.cos(np.pi * (1 - progress_remaining)))
                model.lr_schedule = cosine_schedule
            # Default 'linear' schedule is already set by stable-baselines3
            
            logger.info(f"Model created with leverage={trial_params.get('leverage', 10.0)}, "
                       f"target_kl={target_kl}, lr_schedule={learning_rate_schedule}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise

    def objective(self, trial) -> float:
        """✅ FIXED: Optuna objective function with tunable reward weights, leverage, and learning dynamics."""
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna not available for hyperparameter optimization")
            
        vec_env = None
        eval_vec_env = None
        
        try:
            # ✅ FIXED: Sample enhanced hyperparameters including reward weights and leverage
            trial_params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'n_epochs': trial.suggest_int('n_epochs', 3, 20),
                'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.999),
                'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
                'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True),
                'max_grad_norm': trial.suggest_float('max_grad_norm', 0.1, 2.0),
                
                # Architecture hyperparameters
                'lstm_layers': trial.suggest_int('lstm_layers', 1, 4),
                'expert_hidden_size': trial.suggest_categorical('expert_hidden_size', [16, 32, 64, 128]),
                'attention_features': trial.suggest_categorical('attention_features', [32, 64, 128, 256]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
                'lstm_hidden_size': trial.suggest_categorical('lstm_hidden_size', [64, 128, 256]),
                'seed': trial.suggest_int('seed', 1, 10000),
                
                # ✅ FIXED: Tunable leverage (1.0 to 25.0x maximum)
                'leverage': trial.suggest_float('leverage', 1.0, 25.0, step=0.5),
                
                # ✅ FIXED: Tunable learning rate schedule
                'learning_rate_schedule': trial.suggest_categorical('learning_rate_schedule', ['constant', 'linear', 'cosine']),
                
                # ✅ FIXED: Tunable target_kl for policy stability
                'use_target_kl': trial.suggest_categorical('use_target_kl', [True, False]),
                
                # Risk management
                'max_margin_allocation_pct': trial.suggest_float('max_margin_allocation_pct', 0.01, 0.1, step=0.005),
                
                # ✅ FIXED: Tunable reward weights - agent "personality"
                'reward_weight_base_return': trial.suggest_float('reward_weight_base_return', 0.5, 2.0, step=0.1),
                'reward_weight_risk_adjusted': trial.suggest_float('reward_weight_risk_adjusted', 0.0, 0.8, step=0.05),
                'reward_weight_stability': trial.suggest_float('reward_weight_stability', 0.0, 0.5, step=0.05),
                'reward_weight_transaction_penalty': trial.suggest_float('reward_weight_transaction_penalty', -0.3, 0.0, step=0.02),
                'reward_weight_drawdown_penalty': trial.suggest_float('reward_weight_drawdown_penalty', -1.0, 0.0, step=0.05),
                'reward_weight_position_penalty': trial.suggest_float('reward_weight_position_penalty', -0.2, 0.0, step=0.01),
                'reward_weight_risk_bonus': trial.suggest_float('reward_weight_risk_bonus', 0.0, 0.3, step=0.02),
            }
            
            # ✅ FIXED: Set target_kl based on use_target_kl choice
            if trial_params['use_target_kl']:
                trial_params['target_kl'] = trial.suggest_float('target_kl', 0.001, 0.1, log=True)
            else:
                trial_params['target_kl'] = None
                
            # Create parallel environments
            vec_env = SubprocVecEnv([self._make_env(i, trial.number, trial_params=trial_params) for i in range(self.num_cpu)])
            model = self.create_model(trial_params, vec_env)
            
            # Setup logging
            experiment_name = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_path = SETTINGS.get_logs_path()
            Path(log_path).mkdir(parents=True, exist_ok=True)
            
            # Configure logger
            model.set_logger(configure(log_path, ["stdout", "csv", "tensorboard"]))
            
            # Setup callbacks
            callbacks = []
            
            # Evaluation callback for early stopping
            eval_vec_env = DummyVecEnv([self._make_env(rank=0, seed=123, trial_params=trial_params)])
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
            
            # Performance monitoring
            perf_callback = PerformanceMonitoringCallback(
                performance_threshold=-0.15,
                drawdown_threshold=0.25
            )
            callbacks.append(perf_callback)
            
            # Live monitoring callback
            live_monitoring_callback = LiveRLMonitoringCallback()
            callbacks.append(live_monitoring_callback)
            
            # Attention analysis
            attention_callback = AttentionAnalysisCallback(log_frequency=2000)
            callbacks.append(attention_callback)
            
            # W&B logging if enabled
            if self.use_wandb:
                wandb_callback = WandbCallback(
                    project_name="crypto_trading_optimization_fixed",
                    experiment_name=experiment_name,
                    config={'trial_number': trial.number, **trial_params}
                )
                callbacks.append(wandb_callback)
                
            # Training with pruning support
            training_steps = trial.suggest_int('total_timesteps', 50000, 200000, step=10000)
            
            # Custom pruning callback
            class PruningCallback(BaseCallback):
                def __init__(self, trial, verbose: int = 0):
                    super().__init__(verbose)
                    self.trial = trial
                    self.evaluation_results = []

                def _on_step(self) -> bool:
                    try:
                        # Report intermediate results every 10k steps
                        if self.n_calls % (10000 // self.training_env.num_envs) == 0 and len(perf_callback.portfolio_values) > 0:
                            recent_values = perf_callback.portfolio_values[-100:]
                            if len(recent_values) > 1:
                                performance = (recent_values[-1] - recent_values[0]) / recent_values[0]
                                self.trial.report(performance, step=self.num_timesteps)
                                
                                # Check if trial should be pruned
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
                
                # Calculate annualized return based on number of steps and bar timeframe
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
                
                # ✅ FIXED: Use Calmar Ratio adjusted for leverage as primary objective
                leverage = trial_params.get('leverage', 10.0)
                if max_drawdown > 0.01:
                    calmar_ratio = annualized_return / max_drawdown
                    # Adjust for leverage - higher leverage should show proportionally better performance
                    leverage_adjusted_calmar = calmar_ratio * np.sqrt(leverage / 10.0)
                else:
                    leverage_adjusted_calmar = annualized_return * 10 if annualized_return >= 0 else annualized_return
                    
                # Store detailed results
                trial.set_user_attr('total_return', total_return)
                trial.set_user_attr('annualized_return', annualized_return)
                trial.set_user_attr('max_drawdown', max_drawdown)
                trial.set_user_attr('calmar_ratio', calmar_ratio)
                trial.set_user_attr('leverage_adjusted_calmar', leverage_adjusted_calmar)
                trial.set_user_attr('final_portfolio_value', final_value)
                trial.set_user_attr('leverage', leverage)
                
                return leverage_adjusted_calmar
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

    def optimize(self, n_trials: int = 50, timeout: Optional[int] = None):
        """Run hyperparameter optimization with enhanced search space."""
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
            
            logger.info(f"🚀 Starting ENHANCED hyperparameter optimization with {n_trials} trials")
            logger.info("✅ TUNING: Reward weights, leverage, learning dynamics, target_kl")
            
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
            logger.info("🎯 ENHANCED Optimization Results:")
            logger.info(f"Best value (leverage-adjusted Calmar): {study.best_value:.4f}")
            logger.info(f"Best leverage: {study.best_trial.params.get('leverage', 'N/A')}")
            logger.info(f"Best target_kl: {study.best_trial.params.get('target_kl', 'N/A')}")
            logger.info(f"Best lr_schedule: {study.best_trial.params.get('learning_rate_schedule', 'N/A')}")
            logger.info(f"Best params: {study.best_trial.params}")
            
            return study
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return None

    def train_best_model(self, study_or_params, total_timesteps: int = 500000) -> PPO:
        """Train final model with best hyperparameters."""
        logger.info("🏋️ Training final model with ENHANCED best hyperparameters...")
        vec_env = None
        
        try:
            # Extract best parameters
            if hasattr(study_or_params, 'best_trial'):
                best_params = study_or_params.best_trial.params
                logger.info(f"Using best trial with leverage-adjusted Calmar: {study_or_params.best_value:.4f}")
                logger.info(f"Best leverage: {best_params.get('leverage', 'N/A')}")
                logger.info(f"Best target_kl: {best_params.get('target_kl', 'N/A')}")
            else:
                best_params = study_or_params
                logger.info("Using provided parameters")
                
            # Create parallel environments
            vec_env = SubprocVecEnv([self._make_env(i, 42, trial_params=best_params) for i in range(self.num_cpu)])
            
            # Create model with best parameters
            model = self.create_model(best_params, vec_env)
            
            # Setup comprehensive logging
            experiment_name = f"final_model_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_path = SETTINGS.get_logs_path()
            model.set_logger(configure(log_path, ["stdout", "csv", "tensorboard"]))
            
            # Setup callbacks
            callbacks = []
            
            # Checkpointing
            checkpoint_callback = CheckpointCallback(
                save_freq=max(10000 // self.num_cpu, 500),
                save_path=str(Path(log_path) / "checkpoints"),
                name_prefix="ppo_crypto_model_enhanced"
            )
            callbacks.append(checkpoint_callback)
            
            # Performance monitoring
            perf_callback = PerformanceMonitoringCallback(verbose=1)
            callbacks.append(perf_callback)
            
            # Live monitoring
            live_monitoring_callback = LiveRLMonitoringCallback()
            callbacks.append(live_monitoring_callback)
            
            # Attention analysis
            attention_callback = AttentionAnalysisCallback(log_frequency=5000)
            callbacks.append(attention_callback)
            
            # W&B logging
            if self.use_wandb:
                wandb_callback = WandbCallback(
                    project_name="crypto_trading_final_enhanced",
                    experiment_name=experiment_name,
                    config={'final_training': True, **best_params}
                )
                callbacks.append(wandb_callback)
                
            # Train the model
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            
            # Save the final model
            model_path = SETTINGS.get_model_path()
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(model_path)
            
            logger.info(f"✅ ENHANCED final model saved to: {model_path}")
            logger.info(f"Model trained with leverage: {best_params.get('leverage', 'N/A')}")
            logger.info(f"Model trained with target_kl: {best_params.get('target_kl', 'N/A')}")
            
            return model
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
            
        finally:
            if vec_env:
                vec_env.close()

# --- MAIN TRAINING INTERFACE ---

def train_model_advanced(optimization_trials: int = 20,
                        final_training_steps: int = 500000,
                        use_wandb: bool = False,
                        use_ensemble: bool = False) -> PPO:
    """✅ FIXED: Advanced training pipeline with all enhancements including tunable reward weights and leverage."""
    try:
        logger.info("🎯 Starting ENHANCED Advanced Training Pipeline")
        logger.info("✅ FIXES IMPLEMENTED:")
        logger.info("  - Dynamic reward scaling (scaling_factor = 200 / leverage)")
        logger.info("  - Tunable reward weights")
        logger.info("  - Leverage as hyperparameter (1.0 to 25.0x)")
        logger.info("  - Tunable learning rate schedule")
        logger.info("  - Enabled target_kl for policy stability")
        print("="*50)
        
        # Check multiprocessing protection
        if __name__ != '__main__':
            logger.warning("Training should be run as main script for multiprocessing safety")
            
        # Initialize trainer
        trainer = OptimizedTrainer(use_wandb=use_wandb)
        
        if optimization_trials > 0 and OPTUNA_AVAILABLE:
            # Run hyperparameter optimization
            logger.info(f"Phase 1: ENHANCED Hyperparameter Optimization ({optimization_trials} trials)")
            study = trainer.optimize(n_trials=optimization_trials)
            
            if study is None:
                logger.error("Optimization failed, using default parameters")
                # Use default parameters
                best_params = {
                    'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64,
                    'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95,
                    'clip_range': 0.2, 'ent_coef': 0.01, 'max_grad_norm': 0.5,
                    'lstm_layers': 2, 'expert_hidden_size': 32,
                    'attention_features': 64, 'dropout_rate': 0.1,
                    'lstm_hidden_size': 64, 'seed': 42,
                    'leverage': 10.0,  # ✅ FIXED: Default leverage
                    'learning_rate_schedule': 'linear',
                    'target_kl': None,
                    'max_margin_allocation_pct': 0.02,
                    # ✅ FIXED: Default reward weights
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
                
            # Train final model with best parameters
            logger.info(f"Phase 2: Final Training with ENHANCED parameters ({final_training_steps:,} steps)")
            model = trainer.train_best_model(best_params, final_training_steps)
            
        else:
            # Train with default parameters
            logger.info(f"Training with ENHANCED default parameters ({final_training_steps:,} steps)")
            default_params = {
                'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64,
                'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95,
                'clip_range': 0.2, 'ent_coef': 0.01, 'max_grad_norm': 0.5,
                'lstm_layers': 2, 'expert_hidden_size': 32,
                'attention_features': 64, 'dropout_rate': 0.1,
                'lstm_hidden_size': 64, 'seed': 42,
                'leverage': 10.0,  # ✅ FIXED: Default leverage
                'learning_rate_schedule': 'linear',
                'target_kl': None,
                'max_margin_allocation_pct': 0.02,
                # ✅ FIXED: Default reward weights
                'reward_weight_base_return': 1.0,
                'reward_weight_risk_adjusted': 0.3,
                'reward_weight_stability': 0.2,
                'reward_weight_transaction_penalty': -0.1,
                'reward_weight_drawdown_penalty': -0.4,
                'reward_weight_position_penalty': -0.05,
                'reward_weight_risk_bonus': 0.15
            }
            
            model = trainer.train_best_model(default_params, final_training_steps)
            
        if use_ensemble:
            logger.info("Phase 3: Ensemble Training")
            logger.info("⚠️ Ensemble training requires multiple model training - implement as needed")
            
        logger.info("🎉 ENHANCED advanced training pipeline completed!")
        logger.info(f"Model saved to: {SETTINGS.get_model_path()}")
        
        return model
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

# Entry point for backwards compatibility
def train_model():
    """Simple training interface for backwards compatibility."""
    return train_model_advanced(optimization_trials=0, final_training_steps=200000)

if __name__ == "__main__":
    # Ensure multiprocessing start method is set for compatibility (especially on macOS/Windows)
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
        
    try:
        model = train_model_advanced(
            optimization_trials=5,
            final_training_steps=100000,
            use_wandb=False,
            use_ensemble=False
        )
        
        logger.info("✅ ENHANCED training completed successfully!")
        
    except Exception as e:
        logger.error(f"ENHANCED training example failed: {e}")
