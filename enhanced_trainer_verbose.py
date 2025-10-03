import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import os
import multiprocessing as mp
import logging
import torch
from collections import defaultdict

# --- MAJOR ARCHITECTURAL CHANGE: Import TQC from SB3-Contrib ---
from sb3_contrib import TQC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.utils import set_random_seed
from gymnasium.wrappers import TimeLimit

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

try:
    from torch.utils.tensorboard import SummaryWriter
    from stable_baselines3.common.logger import TensorBoardOutputFormat
except ImportError:
    SummaryWriter = None
    TensorBoardOutputFormat = None
    logging.warning("TensorBoard not available, hyperparameter logging will be disabled.")

from processor import EnhancedDataProcessor, generate_stateful_features_for_fitting
from config import SETTINGS, ModelArchitectureConfig
from tins import EnhancedHierarchicalAttentionFeatureExtractor
# --- Use the new, refactored environment ---
from enhanced_engine_verbose import FixedHierarchicalTradingEnvironment
from normalizer import Normalizer

logger = logging.getLogger(__name__)


# --- NEW: Lagrangian Constraint Manager ---
# This class lives in the Trainer, as its parameters (lambdas) are part of the learning process.
class ConstraintManager:
    """Manages risk constraints using Lagrangian duality, learning penalty multipliers automatically."""
    def __init__(self, targets: Dict[str, float], device: torch.device, lr: float = 1e-3):
        self.targets = targets
        self.device = device
        # Use log_lambdas to ensure lambdas are always non-negative after exponentiation.
        # These are trainable parameters.
        self.log_lambdas = {key: torch.nn.Parameter(torch.zeros(1, device=device, requires_grad=True)) for key in targets}
        self.lambda_optimizer = torch.optim.Adam(self.log_lambdas.values(), lr=lr)
        logger.info(f"ConstraintManager initialized with targets: {self.targets}")

    @property
    def lambdas(self) -> Dict[str, float]:
        """Returns the current lambda values as a dictionary of floats."""
        return {key: torch.exp(log_lambda).item() for key, log_lambda in self.log_lambdas.items()}

    def update_lambdas(self, costs_batch: torch.Tensor, constraint_keys: List[str]):
        """
        Updates the lambda multipliers based on a batch of costs from the replay buffer.
        This is the core of the dual gradient ascent step.
        """
        # The loss for lambdas is designed to push the expected cost towards the target.
        # We want to maximize: L(λ) = E[-λ * (c(s,a) - d)]
        # So we minimize: -L(λ)
        lambda_loss = 0.0
        for i, key in enumerate(constraint_keys):
            # We detach costs as we don't propagate gradients back to the policy from this update.
            cost_for_lambda = costs_batch[:, i].detach()
            lambda_loss -= self.log_lambdas[key] * cost_for_lambda.mean()

        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()


# --- NEW: Custom TQC class to handle Lagrangian updates ---
class LagrangianTQC(TQC):
    """
    A TQC subclass that integrates the ConstraintManager to update lambdas during training.
    """
    def __init__(self, constraint_manager: ConstraintManager, constraint_keys: List[str], **kwargs):
        super().__init__(**kwargs)
        self.constraint_manager = constraint_manager
        self.constraint_keys = constraint_keys

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        ent_coef_losses = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # --- Core Lagrangian Update Step ---
            # Extract costs from the 'info' dictionary stored in the replay buffer.
            # This is why the environment must add 'costs' to the info dict.
            costs = [info['costs'] for info in replay_data.infos]
            costs_tensor = torch.tensor([[c[key] for key in self.constraint_keys] for c in costs], device=self.device, dtype=torch.float32)

            # Update the lambda parameters
            self.constraint_manager.update_lambdas(costs_tensor, self.constraint_keys)

            # --- Standard TQC Training ---
            # The reward in the replay buffer is already the augmented reward (pnl - penalty)
            # so the TQC learning step implicitly optimizes for the constrained objective.
            loss_results = self._train_on_batch(replay_data)
            actor_losses.append(loss_results["actor_loss"])
            critic_losses.append(loss_results["critic_loss"])
            if "ent_coef_loss" in loss_results:
                ent_coef_losses.append(loss_results["ent_coef_loss"])

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


    def _train_on_batch(self, replay_data: ReplayBuffer) -> Dict[str, float]:
        """Helper method to perform a single gradient update step on a batch of data."""
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
            next_quantiles = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            
            # Truncation of quantiles
            n_target_quantiles = self.critic.n_quantiles * self.critic.n_critics
            top_quantiles_to_drop = self.top_quantiles_to_drop_per_net * self.critic.n_critics
            
            sorted_next_quantiles, _ = torch.sort(next_quantiles, dim=1)
            truncated_next_quantiles = sorted_next_quantiles[:, : n_target_quantiles - top_quantiles_to_drop]
            
            next_q_values = truncated_next_quantiles.mean(dim=1, keepdim=True)
            next_q_values = next_q_values - self.ent_coef * next_log_prob.reshape(-1, 1)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        current_quantiles_list = self.critic(replay_data.observations, replay_data.actions)
        critic_loss = 0.0
        for quantiles in current_quantiles_list:
            # Huber loss
            errors = target_q_values.T - quantiles
            huber_loss = torch.where(errors.abs() <= 1.0, 0.5 * errors.pow(2), errors.abs() - 0.5)
            # Quantile Huber loss
            quantile_huber_loss = (torch.abs(self.critic.quantiles - (errors.detach() < 0).float()) * huber_loss).mean()
            critic_loss += quantile_huber_loss
        
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Freeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = False
        
        actions, log_prob = self.actor.action_log_prob(replay_data.observations)
        q_values_pi = torch.cat(self.critic(replay_data.observations, actions), dim=1).mean(dim=1, keepdim=True)
        actor_loss = (self.ent_coef * log_prob - q_values_pi).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True
        
        ent_coef_loss = None
        if self.ent_coef_optimizer is not None:
            ent_coef_loss = -self.log_ent_coef * (log_prob + self.target_entropy).detach().mean()
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
        
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "ent_coef_loss": ent_coef_loss.item() if ent_coef_loss is not None else 0.0,
        }


class PortfolioValueEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_portfolio_value = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            portfolio_values, episode_rewards = [], []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done, episode_reward = False, 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, rewards, dones, infos = self.eval_env.step(action)
                    done = dones[0]
                    episode_reward += rewards[0]
                if 'portfolio_value' in infos[0]:
                    portfolio_values.append(infos[0]['portfolio_value'])
                episode_rewards.append(episode_reward)

            mean_portfolio_value = np.mean(portfolio_values) if portfolio_values else 0.0
            mean_reward = np.mean(episode_rewards)
            if self.verbose > 0:
                logger.info(f"Eval @ ts {self.num_timesteps}: mean_reward={mean_reward:.2f}, mean_portfolio=${mean_portfolio_value:,.2f}")

            self.logger.record("eval/mean_portfolio_value", mean_portfolio_value)
            self.logger.record("eval/mean_reward", float(mean_reward))

            if mean_portfolio_value > self.best_mean_portfolio_value:
                self.best_mean_portfolio_value = mean_portfolio_value
                if self.verbose > 0:
                    logger.info(f"New best mean portfolio value: ${self.best_mean_portfolio_value:,.2f}")
                if self.best_model_save_path:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            self.logger.dump(self.num_timesteps)
        return True

class HParamCallback(BaseCallback):
    def __init__(self, trial: "optuna.trial.Trial", metrics_to_log: List[str]):
        super().__init__()
        self.trial = trial
        self.metrics_to_log = metrics_to_log
    def _on_training_start(self) -> None:
        if SummaryWriter is None: return
        writer = next((f.writer for f in self.logger.output_formats if isinstance(f, TensorBoardOutputFormat)), None)
        if writer:
            hparams = {k: v for k, v in self.trial.params.items() if isinstance(v, (bool, str, float, int))}
            metric_dict = {metric: 0 for metric in self.metrics_to_log}
            try:
                writer.add_hparams(hparams, metric_dict, run_name="hparams")
                writer.flush()
            except Exception as e:
                logger.error(f"Failed to log HPARAMS to TensorBoard: {e}")
    def _on_step(self) -> bool: return True

class RolloutInfoCallback(BaseCallback):
    def __init__(self, log_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.buffers = defaultdict(list)
        self.keys_to_log = ["drawdown", "realized_pnl", "portfolio_value", "total_cost", "thrashing_ratio"]

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key in self.keys_to_log:
                if key in info: self.buffers[key].append(info[key])
            costs = info.get('costs', {})
            if costs:
                for k, v in costs.items(): self.buffers[f'cost/{k}'].append(v)

        if self.n_calls % self.log_freq == 0:
            for key, values in self.buffers.items():
                if values: self.logger.record(f'rollout/{key}_mean', np.mean(values))
            if isinstance(self.model, LagrangianTQC):
                for key, val in self.model.constraint_manager.lambdas.items():
                    self.logger.record(f'lambda/{key}', val)
            self.buffers.clear()
        return True

class EpisodeStatsCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for i, info in enumerate(infos):
            if "_episode" in info:
                self.logger.record('episode/reward', info['episode']['r'])
                self.logger.record('episode/length', info['episode']['l'])
                if 'portfolio_value' in info:
                    self.logger.record('episode/final_portfolio_value', info['portfolio_value'])
                if 'total_executed_trades' in info:
                     self.logger.record('episode/total_trades', info['total_executed_trades'])
                if self.verbose > 0:
                    logger.info(f"Worker {i}: Episode finished. Reward={info['episode']['r']:.2f}, Length={info['episode']['l']}, Final Portfolio=${info.get('portfolio_value', 0):,.2f}")
        return True

class WandbCallback(BaseCallback):
    def __init__(self, project_name: str, experiment_name: str, config: Dict, verbose: int = 0):
        super().__init__(verbose)
        self.project_name, self.experiment_name, self.config = project_name, experiment_name, config
        self.wandb_run = None
    def _on_training_start(self) -> None:
        if WANDB_AVAILABLE:
            self.wandb_run = wandb.init(project=self.project_name, name=self.experiment_name, config=self.config, reinit=True)
    def _on_step(self) -> bool:
        if self.wandb_run and self.n_calls % 500 == 0:
            log_dict = {k: v for k, v in self.logger.name_to_value.items() if isinstance(v, (int, float))}
            if log_dict: wandb.log(log_dict, step=self.n_calls)
        return True
    def _on_training_end(self) -> None:
        if self.wandb_run: wandb.finish()


class EnhancedFixedTrainer:
    def __init__(self, base_config: Optional[Dict] = None, use_wandb: bool = False, enable_live_monitoring: bool = True):
        self.base_config = base_config or SETTINGS.dict()
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.enable_live_monitoring = enable_live_monitoring
        logger.info("🔄 Pre-loading training data...")
        processor = EnhancedDataProcessor(config=SETTINGS)
        self.bars_df = processor.create_enhanced_bars_from_trades("in_sample")
        context_features_df = generate_stateful_features_for_fitting(self.bars_df, SETTINGS.strategy)
        self.all_features_df = pd.merge(self.bars_df, context_features_df, on='timestamp', how='left').ffill().fillna(0.0)
        self.normalizer = Normalizer(SETTINGS.strategy)
        self.normalizer.fit(self.bars_df, context_features_df)
        self.normalizer.save(Path(SETTINGS.get_normalizer_path()))
        self.num_cpu = SETTINGS.num_workers
        logger.info(f"🚀 Using {self.num_cpu} parallel environments.")
        if self.enable_live_monitoring:
            logger.info("📺 LIVE MONITORING ENABLED - Worker #0 will log state to files.")
        
        self.constraint_manager: Optional[ConstraintManager] = None
        self.constraint_keys: List[str] = ['drawdown', 'trade_cost_pct', 'thrashing_rate']

    def _make_env(self, rank: int, seed: int = 0, trial_params: Optional[Dict] = None, initial_lambdas: Optional[Dict] = None) -> Callable:
        def _init():
            leverage = 10.0
            constraint_targets = {
                'drawdown': 0.15,
                'trade_cost_pct': 0.0005,
                'thrashing_rate': 0.1,
            }
            if trial_params:
                leverage = trial_params.get('leverage', 10.0)
                constraint_targets['drawdown'] = trial_params.get('target_drawdown', 0.15)
                constraint_targets['trade_cost_pct'] = trial_params.get('target_trade_cost_pct', 0.0005)
                constraint_targets['thrashing_rate'] = trial_params.get('target_thrashing_rate', 0.1)

            env = FixedHierarchicalTradingEnvironment(
                df_base_ohlc=self.bars_df, normalizer=self.normalizer, config=SETTINGS,
                leverage=leverage, 
                constraint_targets=constraint_targets,
                initial_lambdas=initial_lambdas or {k: 1.0 for k in self.constraint_keys},
                precomputed_features=self.all_features_df, worker_id=rank
            )
            env.reset(seed=seed + rank)
            return env
        set_random_seed(seed)
        return _init

    def create_lagrangian_model(self, trial_params: Dict, vec_env: VecEnv) -> LagrangianTQC:
        constraint_targets = {
            'drawdown': trial_params.get('target_drawdown', 0.15),
            'trade_cost_pct': trial_params.get('target_trade_cost_pct', 0.0005),
            'thrashing_rate': trial_params.get('target_thrashing_rate', 0.1),
        }
        self.constraint_manager = ConstraintManager(
            targets=constraint_targets, 
            device=torch.device(SETTINGS.device),
            lr=trial_params.get('lambda_lr', 1e-4)
        )

        arch_config = ModelArchitectureConfig(
            transformer_d_model=trial_params.get('transformer_d_model', 64),
            transformer_n_heads=trial_params.get('transformer_n_heads', 4),
            transformer_dim_feedforward=trial_params.get('transformer_dim_feedforward', 256),
            transformer_num_layers=trial_params.get('transformer_num_layers', 2),
            dropout_rate=trial_params.get('dropout_rate', 0.1))
        
        policy_kwargs = {"features_extractor_class": EnhancedHierarchicalAttentionFeatureExtractor,
                         "features_extractor_kwargs": {"arch_cfg": arch_config},
                         "net_arch": {"pi": [256, 128], "qf": [256, 128]},
                         "activation_fn": torch.nn.ReLU, "normalize_images": False}
        
        return LagrangianTQC(
                   constraint_manager=self.constraint_manager,
                   constraint_keys=self.constraint_keys,
                   policy="MultiInputPolicy", env=vec_env,
                   learning_rate=trial_params.get('learning_rate', 3e-4),
                   buffer_size=trial_params.get('buffer_size', 200_000),
                   learning_starts=trial_params.get('learning_starts', 10_000),
                   batch_size=trial_params.get('batch_size', 256),
                   tau=trial_params.get('tau', 0.005), gamma=trial_params.get('gamma', 0.99),
                   train_freq=trial_params.get('train_freq', (1, 'step')),
                   gradient_steps=trial_params.get('gradient_steps', 1),
                   top_quantiles_to_drop_per_net=trial_params.get('top_quantiles_to_drop', 2),
                   ent_coef='auto', target_update_interval=1, use_sde=False,
                   policy_kwargs=policy_kwaargs, tensorboard_log=SETTINGS.get_tensorboard_path(),
                   device=SETTINGS.device, seed=trial_params.get('seed', 42), verbose=0)

    def objective(self, trial) -> float:
        if not OPTUNA_AVAILABLE: raise RuntimeError("Optuna not available.")
        vec_env, eval_env = None, None
        try:
            trial_params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
                'lambda_lr': trial.suggest_float('lambda_lr', 1e-5, 1e-3, log=True),
                'buffer_size': trial.suggest_categorical('buffer_size', [100_000, 200_000]),
                'learning_starts': trial.suggest_categorical('learning_starts', [10000, 20000]),
                'batch_size': trial.suggest_categorical('batch_size', [256, 512]),
                'gamma': trial.suggest_float('gamma', 0.99, 0.999),
                'tau': trial.suggest_float('tau', 0.005, 0.02),
                'top_quantiles_to_drop': trial.suggest_int('top_quantiles_to_drop', 1, 5),
                'leverage': trial.suggest_float('leverage', 5.0, 15.0),
                'target_drawdown': trial.suggest_float('target_drawdown', 0.05, 0.20),
                'target_trade_cost_pct': trial.suggest_float('target_trade_cost_pct', 0.0001, 0.001, log=True),
                'target_thrashing_rate': trial.suggest_float('target_thrashing_rate', 0.05, 0.25),
                'train_freq': (1, 'step'),
                'gradient_steps': 1,
            }
            
            temp_cm = ConstraintManager(targets={k:0 for k in self.constraint_keys}, device=torch.device('cpu'))
            
            vec_env = SubprocVecEnv([self._make_env(i, trial.number, trial_params=trial_params, initial_lambdas=temp_cm.lambdas) for i in range(self.num_cpu)])
            
            model = self.create_lagrangian_model(trial_params, vec_env)
            
            experiment_name = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_path = os.path.join(SETTINGS.get_logs_path(), experiment_name)
            model.set_logger(configure(log_path, ["stdout", "csv", "tensorboard"]))
            
            eval_env_raw = self._make_env(rank=0, seed=123, trial_params=trial_params, initial_lambdas=model.constraint_manager.lambdas)()
            eval_env_limited = TimeLimit(eval_env_raw, max_episode_steps=5000)
            eval_env = DummyVecEnv([lambda: eval_env_limited])
            
            learning_starts_val = trial_params.get('learning_starts', 10_000)
            eval_freq = max(learning_starts_val, 10_000) // self.num_cpu
            eval_callback = PortfolioValueEvalCallback(eval_env, best_model_save_path=str(Path(log_path) / "best_model"),
                                                      log_path=log_path, eval_freq=eval_freq, n_eval_episodes=5,
                                                      deterministic=True, warn=False)
            callbacks = [eval_callback, HParamCallback(trial, ['eval/mean_portfolio_value']),
                         RolloutInfoCallback(log_freq=eval_freq), EpisodeStatsCallback(verbose=0)]
            if self.use_wandb:
                callbacks.append(WandbCallback("crypto_trading_LagrangianTQC", experiment_name, trial_params))
            
            model.learn(total_timesteps=50_000, callback=callbacks, progress_bar=False)
            
            return eval_callback.best_mean_portfolio_value - 1000000.0
        except optuna.exceptions.TrialPruned as e: raise e
        except Exception as e:
            logger.error(f"Trial failed with error: {e}", exc_info=True)
            return -1e9
        finally:
            if vec_env: vec_env.close()
            if eval_env: eval_env.close()

    def optimize(self, n_trials: int = 50, timeout: Optional[int] = None):
        if not OPTUNA_AVAILABLE: return None
        db_path = SETTINGS.get_optuna_db_path()
        study = optuna.create_study(study_name=f"tqc-lagrangian-optimization-{SETTINGS.environment.value}", storage=f"sqlite:///{db_path}",
                                    direction='maximize', sampler=optuna.samplers.TPESampler(), load_if_exists=True)
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        return study

    def train_best_model(self, best_params: Dict, final_training_steps: int) -> LagrangianTQC:
        logger.info("🎯 Training final model with best parameters using Lagrangian TQC...")
        
        temp_vec_env = DummyVecEnv([self._make_env(0, seed=42, trial_params=best_params)])
        model = self.create_lagrangian_model(best_params, temp_vec_env)
        initial_lambdas = model.constraint_manager.lambdas
        temp_vec_env.close()
        
        vec_env = SubprocVecEnv([self._make_env(i, seed=42, trial_params=best_params, initial_lambdas=initial_lambdas) for i in range(self.num_cpu)])
        model.set_env(vec_env)
        
        model.set_logger(configure(SETTINGS.get_logs_path(), ["stdout", "csv", "tensorboard"]))
        callbacks = [RolloutInfoCallback(log_freq=max(5000 // self.num_cpu, 500)),
                     EpisodeStatsCallback(verbose=1 if self.enable_live_monitoring else 0)]
        if self.use_wandb:
            callbacks.append(WandbCallback("crypto_trading_final_LagrangianTQC", f"final_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}", best_params))
        
        model.learn(total_timesteps=final_training_steps, callback=callbacks, progress_bar=True)
        
        model_path_str = SETTINGS.get_model_path()
        model.save(model_path_str)
        logger.info(f"✅ Final Lagrangian TQC model saved to: {model_path_str}")
        self.last_saved_model_path = model_path_str
        vec_env.close()
        return model

def train_model_fixed(optimization_trials: int = 20,
                     final_training_steps: int = 500000,
                     use_wandb: bool = False,
                     enable_live_monitoring: bool = True) -> str:
    try:
        trainer = EnhancedFixedTrainer(use_wandb=use_wandb, enable_live_monitoring=enable_live_monitoring)
        if optimization_trials > 0 and OPTUNA_AVAILABLE:
            study = trainer.optimize(n_trials=optimization_trials)
            best_params = study.best_trial.params if study else {}
        else:
            logger.info("Skipping optimization, using high-quality default TQC + Lagrangian parameters.")
            best_params = {
                'learning_rate': 3e-4, 'lambda_lr': 5e-5, 'buffer_size': 200_000, 
                'learning_starts': 10_000, 'batch_size': 256, 'tau': 0.005, 
                'gamma': 0.99, 'top_quantiles_to_drop': 2, 'train_freq': (1, 'step'), 'gradient_steps': 1,
                'transformer_d_model': 64, 'transformer_n_heads': 4,
                'transformer_num_layers': 2, 'dropout_rate': 0.1, 'seed': 42, 
                'leverage': 10.0,
                'target_drawdown': 0.15,
                'target_trade_cost_pct': 0.0005,
                'target_thrashing_rate': 0.15,
            }
        trainer.train_best_model(best_params, final_training_steps)
        logger.info(f"🎉 Lagrangian TQC training completed! Model saved to: {trainer.last_saved_model_path}")
        return trainer.last_saved_model_path
    except Exception as e:
        logger.exception("FATAL UNHANDLED ERROR in the program will now exit.")
        raise e

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    train_model_fixed(optimization_trials=10, final_training_steps=100000,
                      use_wandb=False, enable_live_monitoring=True)
