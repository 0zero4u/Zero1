# Enhanced trainer.py with Reward Horizon Support

"""
Enhanced Trainer with Reward Horizon System Integration

Key enhancements:
1. Hyperparameter optimization includes reward horizon configuration
2. Automatic validation of data sufficiency for reward horizons
3. Detailed reward horizon analysis in training metrics
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
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
from config import SETTINGS, create_config, Environment
from tins import EnhancedHierarchicalAttentionFeatureExtractor
from engine import EnhancedHierarchicalTradingEnvironment
from normalizer import Normalizer

logger = logging.getLogger(__name__)

class RewardHorizonAnalysisCallback(BaseCallback):
    """
    ✅ NEW: Callback to analyze reward horizon effectiveness during training
    """
    
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

class OptimizedTrainer:
    """
    ✅ ENHANCED: Trainer with reward horizon system integration
    """
    
    def __init__(self, base_config: Optional[Dict] = None, use_wandb: bool = False):
        self.base_config = base_config or SETTINGS.dict()
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.best_trial_results = None
        
        try:
            # Pre-load data once
            logger.info("🔄 Pre-loading training data...")
            processor = EnhancedDataProcessor(config=SETTINGS)
            
            # Load bars for environment and features for normalizer
            self.bars_df = processor.create_enhanced_bars_from_trades("in_sample")
            
            # ✅ ENHANCED: Validate data sufficiency for reward horizon
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
        """
        ✅ ENHANCED: Environment factory with reward horizon support
        """
        def _init():
            try:
                config_overrides = {}
                
                # Extract tunable parameters
                leverage = trial_params.get('leverage', 10.0) if trial_params else 10.0
                
                # ✅ NEW: Extract reward horizon parameters
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
                    reward_weights=reward_weights
                )
                
                env.reset(seed=seed + rank)
                return env
                
            except Exception as e:
                logger.error(f"Error creating environment {rank}: {e}")
                raise
        
        set_random_seed(seed)
        return _init
    
    def objective(self, trial) -> float:
        """
        ✅ ENHANCED: Optuna objective with reward horizon optimization
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna not available for hyperparameter optimization")
        
        vec_env = None
        eval_vec_env = None
        
        try:
            # ✅ ENHANCED: Sample reward horizon parameters
            trial_params = {
                # Existing hyperparameters
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
                
                # Trading parameters
                'leverage': trial.suggest_float('leverage', 1.0, 25.0, step=0.5),
                'learning_rate_schedule': trial.suggest_categorical('learning_rate_schedule', ['constant', 'linear', 'cosine']),
                'use_target_kl': trial.suggest_categorical('use_target_kl', [True, False]),
                'max_margin_allocation_pct': trial.suggest_float('max_margin_allocation_pct', 0.01, 0.1, step=0.005),
                
                # ✅ NEW: Reward horizon parameters
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
            
            # ✅ ENHANCED: Validate reward horizon against dataset size
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
            
            # ✅ NEW: Reward horizon analysis callback
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
                    project_name="crypto_trading_reward_horizon",
                    experiment_name=experiment_name,
                    config={'trial_number': trial.number, **trial_params}
                )
                callbacks.append(wandb_callback)
            
            # Train the model
            training_steps = trial.suggest_int('total_timesteps', 50000, 200000, step=10000)
            
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
                
                # ✅ ENHANCED: Reward horizon adjusted performance metric
                leverage = trial_params.get('leverage', 10.0)
                reward_horizon = trial_params.get('reward_horizon_steps', 1)
                
                if max_drawdown > 0.01:
                    calmar_ratio = annualized_return / max_drawdown
                    # Adjust for both leverage and reward horizon
                    leverage_adjusted_calmar = calmar_ratio * np.sqrt(leverage / 10.0)
                    # Bonus for using longer horizons effectively
                    horizon_bonus = 1.0 + (reward_horizon - 1) * 0.02  # 2% bonus per step beyond immediate
                    final_score = leverage_adjusted_calmar * horizon_bonus
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
                
                # ✅ NEW: Store reward horizon effectiveness metrics
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
        """
        ✅ NEW: Analyze the effectiveness of the reward horizon during training
        """
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
        ✅ ENHANCED: Run optimization with reward horizon parameters
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
            
            logger.info(f"🚀 Starting ENHANCED optimization with reward horizon tuning")
            logger.info(f"📊 Dataset size: {len(self.bars_df)} bars")
            logger.info(f"🎯 Max reward horizon: {min(15, len(self.bars_df) // 1000)} steps")
            logger.info("✅ TUNING: Reward horizon, leverage, learning dynamics, reward weights")
            
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
            logger.info("🎯 ENHANCED Optimization Results with Reward Horizon:")
            logger.info(f"Best final score: {study.best_value:.4f}")
            logger.info(f"Best reward horizon: {study.best_trial.params.get('reward_horizon_steps', 'N/A')} steps")
            logger.info(f"Best reward decay: {study.best_trial.params.get('reward_horizon_decay', 'N/A')}")
            logger.info(f"Best leverage: {study.best_trial.params.get('leverage', 'N/A')}")
            logger.info(f"Best parameters: {study.best_trial.params}")
            
            # Analyze horizon effectiveness across trials
            self._analyze_study_horizon_patterns(study)
            
            return study
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return None
    
    def _analyze_study_horizon_patterns(self, study):
        """
        ✅ NEW: Analyze reward horizon patterns across all trials
        """
        try:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            if len(completed_trials) < 5:
                return
            
            horizon_performance = {}
            for trial in completed_trials:
                horizon = trial.params.get('reward_horizon_steps', 1)
                final_score = trial.user_attrs.get('final_score', trial.value)
                
                if horizon not in horizon_performance:
                    horizon_performance[horizon] = []
                horizon_performance[horizon].append(final_score)
            
            # Analyze patterns
            logger.info("\n📊 Reward Horizon Performance Analysis:")
            for horizon in sorted(horizon_performance.keys()):
                scores = horizon_performance[horizon]
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                logger.info(f"  {horizon}-step horizon: {avg_score:.4f} ± {std_score:.4f} ({len(scores)} trials)")
            
            # Find optimal horizon range
            best_horizon = max(horizon_performance.keys(), 
                             key=lambda h: np.mean(horizon_performance[h]))
            logger.info(f"🏆 Best performing horizon: {best_horizon} steps")
            
        except Exception as e:
            logger.error(f"Error analyzing horizon patterns: {e}")

# ✅ ENHANCED: Main training interface with reward horizon
def train_model_advanced(optimization_trials: int = 20,
                        final_training_steps: int = 500000,
                        use_wandb: bool = False,
                        reward_horizon_steps: int = None) -> PPO:
    """
    ✅ ENHANCED: Advanced training with reward horizon system
    """
    try:
        logger.info("🎯 Starting ENHANCED Training with Reward Horizon System")
        logger.info("✅ NEW FEATURES:")
        logger.info(" - Configurable reward horizon (1-20 steps)")
        logger.info(" - Decay factors for multi-step rewards")
        logger.info(" - Horizon effectiveness analysis")
        logger.info(" - Automatic data validation")
        
        # Override reward horizon in config if specified
        if reward_horizon_steps is not None:
            SETTINGS.strategy.reward_horizon_steps = reward_horizon_steps
            logger.info(f"🔧 Overriding reward horizon to {reward_horizon_steps} steps")
        
        print("="*50)
        
        # Initialize trainer
        trainer = OptimizedTrainer(use_wandb=use_wandb)
        
        if optimization_trials > 0 and OPTUNA_AVAILABLE:
            # Run hyperparameter optimization
            logger.info(f"Phase 1: ENHANCED Optimization with Reward Horizon ({optimization_trials} trials)")
            study = trainer.optimize(n_trials=optimization_trials)
            
            if study is None:
                logger.error("Optimization failed, using default parameters")
                best_params = {
                    'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64,
                    'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95,
                    'clip_range': 0.2, 'ent_coef': 0.01, 'max_grad_norm': 0.5,
                    'lstm_layers': 2, 'expert_hidden_size': 32,
                    'attention_features': 64, 'dropout_rate': 0.1,
                    'lstm_hidden_size': 64, 'seed': 42,
                    'leverage': 10.0,
                    'learning_rate_schedule': 'linear',
                    'target_kl': None,
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
            logger.info(f"Phase 2: Final Training with Optimized Reward Horizon ({final_training_steps:,} steps)")
            model = trainer.train_best_model(best_params, final_training_steps)
        else:
            # Train with default parameters
            logger.info(f"Training with default reward horizon parameters ({final_training_steps:,} steps)")
            default_params = {
                'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64,
                'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95,
                'clip_range': 0.2, 'ent_coef': 0.01, 'max_grad_norm': 0.5,
                'lstm_layers': 2, 'expert_hidden_size': 32,
                'attention_features': 64, 'dropout_rate': 0.1,
                'lstm_hidden_size': 64, 'seed': 42,
                'leverage': 10.0,
                'learning_rate_schedule': 'linear',
                'target_kl': None,
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
        
        logger.info("🎉 ENHANCED training with reward horizon completed!")
        logger.info(f"Model saved to: {SETTINGS.get_model_path()}")
        
        return model
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    # Ensure multiprocessing compatibility
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    
    try:
        # Example: Train with 3-minute reward horizon
        model = train_model_advanced(
            optimization_trials=10,
            final_training_steps=100000,
            use_wandb=False,
            reward_horizon_steps=9  # 9 steps = 3 minutes
        )
        
        logger.info("✅ ENHANCED training with reward horizon completed successfully!")
        
    except Exception as e:
        logger.error(f"Training example failed: {e}")
        raise