import torch
import optuna
import wandb
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

from ..processor import create_bars_from_trades
from .config import SETTINGS, create_config, Environment
from .tins import EnhancedHierarchicalAttentionFeatureExtractor
from .engine import HierarchicalTradingEnvironment

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
        if not self.run:
            self.run = wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.config,
                sync_tensorboard=True
            )

    def _on_step(self) -> bool:
        """Log metrics at each step."""
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
                    'portfolio/volatility': info.get('volatility', 0)
                })

        return True

    def _on_training_end(self) -> None:
        """Finish W&B run."""
        if self.run:
            self.run.finish()

class AttentionAnalysisCallback(BaseCallback):
    """Callback to analyze and log attention patterns."""

    def __init__(self, log_frequency: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_frequency = log_frequency
        self.attention_history = []

    def _on_step(self) -> bool:
        """Analyze attention patterns periodically."""
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
                        print(f"⚠️ Warning: Drawdown of {current_drawdown:.2%} exceeds threshold")

        return True

# --- OPTIMIZED TRAINER WITH IMPROVEMENTS ---

class OptimizedTrainer:
    """IMPROVED: Enhanced trainer with pre-loaded data, parallel environments, and better objective metrics."""

    def __init__(self, base_config: Optional[Dict] = None, use_wandb: bool = False):
        self.base_config = base_config or SETTINGS.dict()
        self.use_wandb = use_wandb
        self.best_trial_results = None
        
        # IMPROVEMENT: Pre-load data once
        print("🔄 Pre-loading training data...")
        self.bars_df = create_bars_from_trades("in_sample")
        print(f"✅ Loaded {len(self.bars_df)} bars for training")
        
        # PERFORMANCE: Determine number of parallel workers
        self.num_cpu = min(os.cpu_count(), 8)
        print(f"🚀 Using {self.num_cpu} parallel environments for training.")


    def _make_env(self, rank: int, seed: int = 0) -> Callable:
        """Utility function for multiprocessing environments."""
        def _init():
            env = HierarchicalTradingEnvironment(df_base_ohlc=self.bars_df)
            env.reset(seed=seed + rank)
            return env
        set_random_seed(seed)
        return _init

    def create_model(self, trial: optuna.Trial, vec_env: VecEnv) -> PPO:
        """Create PPO model with trial-suggested hyperparameters."""
        # Sample hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        n_epochs = trial.suggest_int('n_epochs', 3, 20)
        gamma = trial.suggest_float('gamma', 0.9, 0.999)
        gae_lambda = trial.suggest_float('gae_lambda', 0.8, 0.999)
        clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
        ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True)
        max_grad_norm = trial.suggest_float('max_grad_norm', 0.1, 2.0)

        # Architecture hyperparameters
        lstm_layers = trial.suggest_int('lstm_layers', 1, 4)
        expert_hidden_size = trial.suggest_categorical('expert_hidden_size', [16, 32, 64, 128])
        attention_features = trial.suggest_categorical('attention_features', [32, 64, 128, 256])
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)

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
            'net_arch': {
                'pi': [expert_hidden_size * 2, expert_hidden_size],
                'vf': [expert_hidden_size * 2, expert_hidden_size]
            }
        }

        # Create model
        model = PPO(
            "MlpPolicy",
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
            verbose=0,
            device=SETTINGS.device,
            seed=trial.suggest_int('seed', 1, 10000)
        )

        return model

    def objective(self, trial: optuna.Trial) -> float:
        """IMPROVED: Optuna objective function with parallel envs and better metrics."""
        vec_env = None
        eval_vec_env = None
        try:
            # PERFORMANCE: Create parallel environments
            vec_env = SubprocVecEnv([self._make_env(i, trial.number) for i in range(self.num_cpu)])
            model = self.create_model(trial, vec_env)

            # Setup logging
            experiment_name = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_path = SETTINGS.get_logs_path()
            Path(log_path).mkdir(parents=True, exist_ok=True)

            # Configure logger
            model.set_logger(configure(log_path, ["stdout", "csv", "tensorboard"]))

            # Setup callbacks
            callbacks = []

            # Evaluation callback for early stopping
            # Use a single environment for evaluation for consistency
            eval_vec_env = DummyVecEnv([self._make_env(rank=0, seed=123)])
            eval_callback = EvalCallback(
                eval_vec_env,
                best_model_save_path=str(Path(log_path) / f"best_model_trial_{trial.number}"),
                log_path=log_path,
                eval_freq=max(5000 // self.num_cpu, 500),
                deterministic=True,
                render=False,
                n_eval_episodes=5 # Fewer episodes, but consistent
            )
            callbacks.append(eval_callback)

            # Performance monitoring
            perf_callback = PerformanceMonitoringCallback(
                performance_threshold=-0.15,
                drawdown_threshold=0.25
            )
            callbacks.append(perf_callback)

            # Attention analysis
            attention_callback = AttentionAnalysisCallback(log_frequency=2000)
            callbacks.append(attention_callback)

            # W&B logging if enabled
            if self.use_wandb:
                wandb_callback = WandbCallback(
                    project_name="crypto_trading_optimization",
                    experiment_name=experiment_name,
                    config={
                        'trial_number': trial.number,
                        **trial.params
                    }
                )
                callbacks.append(wandb_callback)

            # Training with pruning support
            training_steps = trial.suggest_int('total_timesteps', 50000, 200000, step=10000)

            # Custom pruning callback
            class PruningCallback(BaseCallback):
                def __init__(self, trial: optuna.Trial, verbose: int = 0):
                    super().__init__(verbose)
                    self.trial = trial
                    self.evaluation_results = []

                def _on_step(self) -> bool:
                    # Report intermediate results every 10k steps
                    if self.n_calls % (10000 // self.training_env.num_envs) == 0 and len(perf_callback.portfolio_values) > 0:
                        # Calculate current performance
                        recent_values = perf_callback.portfolio_values[-100:]
                        if len(recent_values) > 1:
                            performance = (recent_values[-1] - recent_values[0]) / recent_values[0]
                            self.trial.report(performance, step=self.num_timesteps)

                            # Check if trial should be pruned
                            if self.trial.should_prune():
                                raise optuna.TrialPruned()

                    return True

            pruning_callback = PruningCallback(trial)
            callbacks.append(pruning_callback)

            # Train the model
            model.learn(
                total_timesteps=training_steps,
                callback=callbacks,
                progress_bar=False
            )

            # IMPROVED: Calculate final performance metric using Calmar Ratio from the first env
            if perf_callback.portfolio_values:
                initial_value = perf_callback.portfolio_values[0]
                final_value = perf_callback.portfolio_values[-1]
                total_return = (final_value - initial_value) / initial_value

                # Calculate annualized return
                days_trading = len(perf_callback.portfolio_values)
                annualized_return = total_return * (252 / days_trading) if days_trading > 0 else 0

                # Calculate maximum drawdown
                cumulative_max = np.maximum.accumulate(perf_callback.portfolio_values)
                drawdowns = (cumulative_max - perf_callback.portfolio_values) / cumulative_max
                max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

                # IMPROVED: Use Calmar Ratio as primary objective
                if max_drawdown > 0.01:  # Minimum 1% drawdown to avoid division issues
                    calmar_ratio = annualized_return / max_drawdown
                else:
                    calmar_ratio = annualized_return * 10  # Bonus for low drawdown

                # Store detailed results
                trial.set_user_attr('total_return', total_return)
                trial.set_user_attr('annualized_return', annualized_return)
                trial.set_user_attr('max_drawdown', max_drawdown)
                trial.set_user_attr('calmar_ratio', calmar_ratio)
                trial.set_user_attr('final_portfolio_value', final_value)

                return calmar_ratio

            else:
                return -1.0  # Poor performance if no portfolio values recorded

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            return -10.0  # Heavy penalty for failed trials
        finally:
            # Cleanup
            if vec_env:
                vec_env.close()
            if eval_vec_env:
                eval_vec_env.close()

    def optimize(self, n_trials: int = 50, timeout: Optional[int] = None) -> optuna.Study:
        """Run hyperparameter optimization."""
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

        print(f"🚀 Starting hyperparameter optimization with {n_trials} trials")

        # Optimize
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        # Store best results
        self.best_trial_results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }

        # Print results
        print("\n🎯 Optimization Results:")
        print(f"Best Calmar Ratio: {study.best_value:.4f}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        if study.best_trial.user_attrs:
            print(f"\nBest trial metrics:")
            for key, value in study.best_trial.user_attrs.items():
                print(f"  {key}: {value:.4f}")

        return study

    def train_best_model(self, study: optuna.Study, total_timesteps: int = 500000) -> PPO:
        """Train final model with best hyperparameters."""
        print("\n🏋️ Training final model with best hyperparameters...")
        vec_env = None
        try:
            # Create parallel environments
            vec_env = SubprocVecEnv([self._make_env(i, 42) for i in range(self.num_cpu)])

            # Create a mock trial with best parameters
            class MockTrial:
                def __init__(self, params):
                    self.params = params
                    # Add a default seed if not found in study
                    if 'seed' not in self.params:
                        self.params['seed'] = 42

                def suggest_float(self, name, low, high, log=False):
                    return self.params[name]

                def suggest_int(self, name, low, high, step=1):
                    return self.params[name]

                def suggest_categorical(self, name, choices):
                    return self.params[name]

            mock_trial = MockTrial(study.best_params)
            model = self.create_model(mock_trial, vec_env)

            # Setup comprehensive logging
            experiment_name = f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_path = SETTINGS.get_logs_path()
            model.set_logger(configure(log_path, ["stdout", "csv", "tensorboard"]))

            # Setup callbacks
            callbacks = []

            # Checkpointing
            checkpoint_callback = CheckpointCallback(
                save_freq=max(10000 // self.num_cpu, 500),
                save_path=str(Path(log_path) / "checkpoints"),
                name_prefix="ppo_crypto_model"
            )
            callbacks.append(checkpoint_callback)

            # Performance monitoring
            perf_callback = PerformanceMonitoringCallback(verbose=1)
            callbacks.append(perf_callback)

            # Attention analysis
            attention_callback = AttentionAnalysisCallback(log_frequency=5000)
            callbacks.append(attention_callback)

            # W&B logging
            if self.use_wandb:
                wandb_callback = WandbCallback(
                    project_name="crypto_trading_final",
                    experiment_name=experiment_name,
                    config={'final_training': True, **study.best_params}
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

            print(f"\n✅ Final model saved to: {model_path}")
            return model
        finally:
            if vec_env:
                vec_env.close()

# --- MULTI-AGENT ENSEMBLE TRAINING ---

class MultiAgentEnsembleTrainer:
    """Train multiple agents for ensemble trading."""

    def __init__(self, n_agents: int = 3, base_config: Optional[Dict] = None):
        self.n_agents = n_agents
        self.base_config = base_config or SETTINGS.dict()
        self.agents = []

    def train_ensemble(self, total_timesteps: int = 200000) -> List[PPO]:
        """Train ensemble of agents with different configurations."""
        print(f"🤖 Training ensemble of {self.n_agents} agents...")

        # Different configurations for diversity
        configs = [
            {'learning_rate': 3e-4, 'n_steps': 2048, 'gamma': 0.99},
            {'learning_rate': 1e-4, 'n_steps': 1024, 'gamma': 0.995},
            {'learning_rate': 1e-3, 'n_steps': 4096, 'gamma': 0.98}
        ]

        for i in range(self.n_agents):
            print(f"\nTraining agent {i+1}/{self.n_agents}")
            env = None
            try:
                # Create environment
                env = self.create_diverse_environment(i)

                # Use different random seeds for diversity
                set_random_seed(42 + i)

                # Create agent with different config
                config = configs[i % len(configs)]
                agent = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=config['learning_rate'],
                    n_steps=config['n_steps'],
                    gamma=config['gamma'],
                    verbose=1,
                    device=SETTINGS.device,
                    seed=42 + i
                )

                # Train agent
                agent.learn(total_timesteps=total_timesteps // self.n_agents)

                # Save agent
                agent_path = SETTINGS.get_model_path().replace('.zip', f'_agent_{i}.zip')
                agent.save(agent_path)
                self.agents.append(agent)

                print(f"Agent {i+1} saved to: {agent_path}")
            finally:
                if env:
                    env.close()

        return self.agents

    def create_diverse_environment(self, agent_id: int) -> HierarchicalTradingEnvironment:
        """Create diverse environments for each agent."""
        # This could involve different data periods, features, or parameters
        bars_df = create_bars_from_trades("in_sample")
        return HierarchicalTradingEnvironment(bars_df)

# --- MAIN TRAINING INTERFACE ---

def train_model_advanced(optimization_trials: int = 20,
                        final_training_steps: int = 500000,
                        use_wandb: bool = False,
                        use_ensemble: bool = False) -> PPO:
    """IMPROVED: Advanced training pipeline with all enhancements."""
    print("🎯 Starting Advanced Training Pipeline")
    print("="*50)

    if __name__ == '__main__': # Protect entry point for multiprocessing
        # Initialize trainer
        trainer = OptimizedTrainer(use_wandb=use_wandb)

        if optimization_trials > 0:
            # Run hyperparameter optimization
            print(f"Phase 1: Hyperparameter Optimization ({optimization_trials} trials)")
            study = trainer.optimize(n_trials=optimization_trials)

            # Train final model with best parameters
            print(f"Phase 2: Final Training ({final_training_steps:,} steps)")
            model = trainer.train_best_model(study, final_training_steps)

        else:
            # Train with default parameters
            print(f"Training with default parameters ({final_training_steps:,} steps)")
            vec_env = SubprocVecEnv([trainer._make_env(i) for i in range(trainer.num_cpu)])
            
            model = PPO(
                "MlpPolicy",
                vec_env,
                policy_kwargs={
                    'features_extractor_class': EnhancedHierarchicalAttentionFeatureExtractor,
                },
                verbose=1,
                device=SETTINGS.device
            )

            model.learn(total_timesteps=final_training_steps)
            model.save(SETTINGS.get_model_path())
            vec_env.close()

        if use_ensemble:
            # Train ensemble
            print("Phase 3: Ensemble Training")
            ensemble_trainer = MultiAgentEnsembleTrainer()
            ensemble_trainer.train_ensemble()

        print("\n🎉 Advanced training pipeline completed!")
        print(f"Model saved to: {SETTINGS.get_model_path()}")
        return model
    else:
        print("Training can only be run as main script due to multiprocessing.")
        return None

# Entry point for backwards compatibility
def train_model():
    """Simple training interface for backwards compatibility."""
    return train_model_advanced(optimization_trials=0, final_training_steps=200000)

if __name__ == "__main__":
    # Example usage
    model = train_model_advanced(
        optimization_trials=10,
        final_training_steps=300000,
        use_wandb=True,
        use_ensemble=False
    )