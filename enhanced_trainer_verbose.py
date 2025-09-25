import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
# --- MODIFICATION: Swapped PPO for SAC ---
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.utils import set_random_seed
import os
import multiprocessing as mp
import logging
import time
import torch

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
from enhanced_engine_verbose import FixedHierarchicalTradingEnvironment
from normalizer import Normalizer
from gymnasium.wrappers import TimeLimit

logger = logging.getLogger(__name__)


class PortfolioValueEvalCallback(EvalCallback):
    """FIXED: Custom EvalCallback to evaluate based on final portfolio value."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_portfolio_value = -np.inf

    def _on_step(self) -> bool:
        # SAC trains at a different frequency than PPO. We use self.num_timesteps for evaluation frequency.
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            portfolio_values, episode_rewards = [], []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
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
                logger.info(f"Eval at timestep {self.num_timesteps}: mean_reward={mean_reward:.2f}, mean_portfolio_value=${mean_portfolio_value:,.2f}")

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


class ComprehensiveLoggingCallback(BaseCallback):
    """
    A powerful callback to log detailed environment statistics, agent behavior,
    and reward components during training for enhanced observability.
    """
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            infos = self.locals.get("infos", [])
            if not infos:
                return True

            info = infos[0]
            self.logger.record("behavior/action_magnitude", info.get("action_magnitude", 0))
            self.logger.record("behavior/consecutive_inactive_steps", info.get("consecutive_inactive_steps", 0))
            self.logger.record("behavior/thrashing_ratio", info.get("thrashing_ratio", 0))
            self.logger.record("behavior/total_attempted_trades", info.get("total_attempted_trades", 0))
            self.logger.record("behavior/total_executed_trades", info.get("total_executed_trades", 0))
            self.logger.record("behavior/total_insignificant_trades", info.get("total_insignificant_trades", 0))
            self.logger.record("performance/portfolio_value", info.get("portfolio_value", 0))
            self.logger.record("performance/drawdown", info.get("drawdown", 0))
            self.logger.record("performance/peak_value", info.get("peak_value", 0))
            self.logger.record("performance/raw_reward", info.get("raw_reward", 0))
            self.logger.record("performance/intrinsic_reward", info.get("intrinsic_reward", 0))

            raw_components = info.get('raw_reward_components', {})
            if raw_components:
                for k, v in raw_components.items(): self.logger.record(f'reward_raw/{k}', v)

            weighted_rewards = info.get('weighted_rewards', {})
            if weighted_rewards:
                for k, v in weighted_rewards.items(): self.logger.record(f'reward_weighted/{k}', v)

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
            logger.info("📺 LIVE MONITORING ENABLED - Worker #0 will show detailed trades.")

    def _make_env(self, rank: int, seed: int = 0, trial_params: Optional[Dict] = None) -> Callable:
        def _init():
            leverage = 10.0
            reward_weights = None
            if trial_params:
                leverage = trial_params.get('leverage', 10.0)
                reward_weights = {
                    'realized_pnl': trial_params.get('reward_weight_realized_pnl', 3.0),
                    'unrealized_pnl_shaping': trial_params.get('reward_weight_unrealized_pnl_shaping', 0.1),
                    'trade_cost': trial_params.get('reward_weight_trade_cost', 0.75),
                    'drawdown': trial_params.get('reward_weight_drawdown', 1.5),
                    'thrashing': trial_params.get('reward_weight_thrashing', 2.0),
                    'frequency': trial_params.get('reward_weight_frequency', 1.0),
                    'inactivity': trial_params.get('reward_weight_inactivity', 0.7),
                    'action_clarity': trial_params.get('reward_weight_action_clarity', 20.15),
                }

            env = FixedHierarchicalTradingEnvironment(
                df_base_ohlc=self.bars_df, normalizer=self.normalizer, config=SETTINGS,
                leverage=leverage, reward_weights=reward_weights,
                precomputed_features=self.all_features_df, worker_id=rank
            )
            env.reset(seed=seed + rank)
            return env
        set_random_seed(seed)
        return _init

    # --- START OF MODIFICATION: create_model now instantiates SAC with more detailed parameters ---
    def create_model(self, trial_params: Dict, vec_env: VecEnv) -> SAC:
        arch_config = ModelArchitectureConfig(
            transformer_d_model=trial_params.get('transformer_d_model', 64),
            transformer_n_heads=trial_params.get('transformer_n_heads', 4),
            transformer_dim_feedforward=trial_params.get('transformer_dim_feedforward', 256),
            transformer_num_layers=trial_params.get('transformer_num_layers', 2),
            dropout_rate=trial_params.get('dropout_rate', 0.1),
        )
        policy_kwargs = {
            "features_extractor_class": EnhancedHierarchicalAttentionFeatureExtractor,
            "features_extractor_kwargs": {"arch_cfg": arch_config},
            "net_arch": {
                "pi": [256, 128],  # Actor network
                "qf": [256, 128]   # Twin Q-function networks
            },
            "activation_fn": torch.nn.ReLU,
            "normalize_images": False,
        }
        
        return SAC(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=trial_params.get('learning_rate', 3e-4),
            buffer_size=trial_params.get('buffer_size', 200_000),
            learning_starts=trial_params.get('learning_starts', 10_000),
            batch_size=trial_params.get('batch_size', 256),
            tau=trial_params.get('tau', 0.005),
            gamma=trial_params.get('gamma', 0.99),
            train_freq=trial_params.get('train_freq', (1, 'step')),
            gradient_steps=trial_params.get('gradient_steps', 1),
            ent_coef='auto',  # Automatic entropy tuning
            target_update_interval=1,
            use_sde=False,  # State Dependent Exploration
            policy_kwargs=policy_kwargs,
            tensorboard_log=SETTINGS.get_tensorboard_path(),
            device=SETTINGS.device,
            seed=trial_params.get('seed', 42),
            verbose=0
        )
    # --- END OF MODIFICATION ---

    # --- MODIFICATION: Optuna objective now searches for SAC hyperparameters ---
    def objective(self, trial) -> float:
        if not OPTUNA_AVAILABLE: raise RuntimeError("Optuna not available.")
        vec_env, eval_env = None, None
        try:
            trial_params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
                'buffer_size': trial.suggest_categorical('buffer_size', [50_000, 100_000, 200_000]),
                'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
                'gamma': trial.suggest_float('gamma', 0.98, 0.999),
                'tau': trial.suggest_float('tau', 0.005, 0.02),
                'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8]),
                'gradient_steps': trial.suggest_categorical('gradient_steps', [1, 4, 8]),
                'leverage': trial.suggest_float('leverage', 5.0, 15.0),
                # Reward weights remain the same as they are part of the environment logic
                'reward_weight_realized_pnl': trial.suggest_float('reward_weight_realized_pnl', 2.0, 5.0),
                'reward_weight_unrealized_pnl_shaping': trial.suggest_float('reward_weight_unrealized_pnl_shaping', 0.01, 0.2),
                'reward_weight_trade_cost': trial.suggest_float('reward_weight_trade_cost', 0.5, 1.5),
                'reward_weight_drawdown': trial.suggest_float('reward_weight_drawdown', 1.0, 2.5),
                'reward_weight_thrashing': trial.suggest_float('reward_weight_thrashing', 1.0, 3.0),
                'reward_weight_frequency': trial.suggest_float('reward_weight_frequency', 0.4, 1.5),
                'reward_weight_inactivity': trial.suggest_float('reward_weight_inactivity', 0.5, 1.5),
                'reward_weight_action_clarity': trial.suggest_float('reward_weight_action_clarity', 0.05, 0.4),
            }
            # SAC uses tuples for train_freq
            trial_params['train_freq'] = (trial_params['train_freq'], 'step')

            vec_env = SubprocVecEnv([self._make_env(i, trial.number, trial_params=trial_params) for i in range(self.num_cpu)])
            model = self.create_model(trial_params, vec_env)

            experiment_name = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_path = os.path.join(SETTINGS.get_logs_path(), experiment_name)
            model.set_logger(configure(log_path, ["stdout", "csv", "tensorboard"]))

            eval_env_raw = self._make_env(rank=0, seed=123, trial_params=trial_params)()
            eval_env_limited = TimeLimit(eval_env_raw, max_episode_steps=5000)
            eval_env = DummyVecEnv([lambda: eval_env_limited])

            eval_callback = PortfolioValueEvalCallback(
                eval_env, best_model_save_path=str(Path(log_path) / "best_model"),
                log_path=log_path, eval_freq=max(5000 // self.num_cpu, 500),
                n_eval_episodes=5, deterministic=True, warn=False
            )

            callbacks = [
                eval_callback,
                HParamCallback(trial, ['eval/mean_portfolio_value']),
                ComprehensiveLoggingCallback(log_freq=max(5000 // self.num_cpu, 500))
            ]
            if self.use_wandb:
                callbacks.append(WandbCallback("crypto_trading_SAC_v1", experiment_name, trial_params))

            # SAC trains continuously, total_timesteps is the primary driver
            model.learn(total_timesteps=50_000, callback=callbacks, progress_bar=False)

            return eval_callback.best_mean_portfolio_value - 1000000.0
        except optuna.exceptions.TrialPruned as e:
            raise e
        except Exception as e:
            logger.error(f"Trial failed with error: {e}", exc_info=True)
            return -1e9
        finally:
            if vec_env: vec_env.close()
            if eval_env: eval_env.close()

    # --- MODIFICATION: Optuna study name updated ---
    def optimize(self, n_trials: int = 50, timeout: Optional[int] = None):
        if not OPTUNA_AVAILABLE: return None
        db_path = SETTINGS.get_optuna_db_path()
        study = optuna.create_study(
            study_name=f"sac-optimization-{SETTINGS.environment.value}", storage=f"sqlite:///{db_path}",
            direction='maximize', sampler=optuna.samplers.TPESampler(), load_if_exists=True
        )
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        return study

    # --- MODIFICATION: Type hint updated ---
    def train_best_model(self, best_params: Dict, final_training_steps: int) -> SAC:
        logger.info("🎯 Training final model with best parameters using SAC...")
        vec_env = SubprocVecEnv([self._make_env(i, seed=42, trial_params=best_params) for i in range(self.num_cpu)])
        model = self.create_model(best_params, vec_env)
        model.set_logger(configure(SETTINGS.get_logs_path(), ["stdout", "csv", "tensorboard"]))

        callbacks = [ComprehensiveLoggingCallback(log_freq=max(5000 // self.num_cpu, 500))]
        if self.use_wandb:
            callbacks.append(WandbCallback("crypto_trading_final_SAC_v1", f"final_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}", best_params))

        model.learn(total_timesteps=final_training_steps, callback=callbacks, progress_bar=True)

        model_path_str = SETTINGS.get_model_path()
        model.save(model_path_str)
        logger.info(f"✅ Final SAC model saved to: {model_path_str}")
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
            if 'train_freq' in best_params: # Ensure train_freq is a tuple
                best_params['train_freq'] = (best_params['train_freq'], 'step')
        else:
            logger.info("Skipping optimization, using high-quality default SAC parameters.")
            # --- START OF MODIFICATION: Default parameters are now for SAC ---
            best_params = {
                'learning_rate': 3e-4,
                'buffer_size': 200_000,
                'learning_starts': 10_000,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': (1, 'step'),
                'gradient_steps': 1,
                'ent_coef': 'auto',
                'transformer_d_model': 64, 'transformer_n_heads': 4,
                'transformer_num_layers': 2, 'dropout_rate': 0.1, 'seed': 42, 'leverage': 10.0,
                # Redesigned default reward weights (unchanged)
                'reward_weight_realized_pnl': 3.0,
                'reward_weight_unrealized_pnl_shaping': 0.1,
                'reward_weight_trade_cost': 0.35,
                'reward_weight_drawdown': 0.0,
                'reward_weight_thrashing': 0.0,
                'reward_weight_frequency': 0.1,
                'reward_weight_inactivity': 1.0,
                'reward_weight_action_clarity': 20.15,
            }
            # --- END OF MODIFICATION ---

        trainer.train_best_model(best_params, final_training_steps)
        logger.info(f"🎉 SAC training completed! Model saved to: {trainer.last_saved_model_path}")
        return trainer.last_saved_model_path
    except Exception as e:
        logger.exception("FATAL UNHANDLED ERROR in the program will now exit.")
        raise e

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    train_model_fixed(
        optimization_trials=10,
        final_training_steps=100000,
        use_wandb=False,
        enable_live_monitoring=True
)
