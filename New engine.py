
import numpy as np
import pandas as pd
import gymnasium
from gymnasium import spaces
from collections import deque
from typing import Dict, Any, Optional, Tuple
import logging
import json
from datetime import datetime
from scipy import stats

from config import SETTINGS, FeatureKeys
from normalizer import Normalizer

logger = logging.getLogger(__name__)


# --- REWARD ARCHITECTURE REMOVED ---
# The complex, hand-tuned RewardManager and its normalizers are no longer needed.
# The reward logic is now simpler and handled directly in the environment step,
# based on penalties learned via the LagrangianTQC trainer.

class EnhancedRiskManager:
    def __init__(self, config, leverage: float = 10.0):
        self.cfg, self.leverage = config, leverage
        self.volatility_buffer = deque(maxlen=50)

    def update_market_regime(self, returns: np.ndarray, volatility: float) -> str:
        if len(returns) < 20: return "UNCERTAIN"
        recent_returns = returns[-20:]
        vol_percentile = np.percentile(self.volatility_buffer, 80) if len(self.volatility_buffer) > 10 else volatility
        if volatility > vol_percentile * 1.5: return "HIGH_VOLATILITY"
        elif np.mean(recent_returns) > 0.001 and volatility < vol_percentile * 0.8: return "TRENDING_UP"
        elif np.mean(recent_returns) < -0.001 and volatility < vol_percentile * 0.8: return "TRENDING_DOWN"
        elif volatility < vol_percentile * 0.6: return "LOW_VOLATILITY"
        else: return "SIDEWAYS"

    def calculate_dynamic_position_limit(self, volatility: float, portfolio_value: float, market_regime: str) -> float:
        base_limit = self.cfg.strategy.max_position_size
        vol_adjustment = max(0.5, min(2.0, 1.0 / (volatility * 10 + 0.1)))
        leverage_adjustment = min(1.0, 10.0 / self.leverage)
        return base_limit * vol_adjustment * leverage_adjustment

    def get_volatility_percentile(self, current_volatility: float) -> float:
        if len(self.volatility_buffer) < 20: return 0.5
        return stats.percentileofscore(list(self.volatility_buffer), current_volatility) / 100.0


class FixedHierarchicalTradingEnvironment(gymnasium.Env):
    """
    Trading environment refactored for use with a Lagrangian-based RL algorithm.
    - The reward is now a combination of normalized PnL and learned penalties.
    - It accepts constraint targets and lambda multipliers to calculate the augmented reward.
    - It reports raw costs in the `info` dict for the trainer to update the lambdas.
    """
    @property
    def total_asset_held(self) -> float:
        if not self.open_positions: return 0.0
        return sum(qty for qty, price, step in self.open_positions)
    @property
    def average_entry_price(self) -> float:
        total_qty = self.total_asset_held
        if total_qty == 0: return 0.0
        total_value = sum(qty * price for qty, price, step in self.open_positions)
        return total_value / total_qty

    def __init__(self, df_base_ohlc: pd.DataFrame, normalizer: Normalizer, config=None,
                 leverage: float = None, 
                 constraint_targets: Dict[str, float] = None,
                 initial_lambdas: Dict[str, float] = None,
                 precomputed_features: Optional[pd.DataFrame] = None,
                 worker_id: int = 0):
        super().__init__()
        self.worker_id, self.verbose = worker_id, worker_id == 0
        self.cfg, self.strat_cfg, self.normalizer = config or SETTINGS, (config or SETTINGS).strategy, normalizer
        self.leverage = leverage if leverage is not None else self.strat_cfg.leverage
        self.risk_manager = EnhancedRiskManager(self.cfg, leverage=self.leverage)
        
        self.constraint_targets = constraint_targets or {}
        self.lambdas = initial_lambdas or {}
        
        self.position_history_buffer = deque(maxlen=20)
        
        base_df = df_base_ohlc.set_index('timestamp')
        all_required_freqs = set(k.value.split('_')[-1] for k in self.strat_cfg.lookback_periods.keys() if k not in {FeatureKeys.CONTEXT, FeatureKeys.PORTFOLIO_STATE, FeatureKeys.PRECOMPUTED_FEATURES}).union({c.timeframe for c in self.strat_cfg.stateful_calculators})
        self.base_timestamps = base_df.resample(self.cfg.base_bar_timeframe.value).asfreq().index
        self.max_step = len(self.base_timestamps) - 2
        self.timeframes_np: Dict[str, Dict[str, np.ndarray]] = {}
        for freq in all_required_freqs:
            agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'trade_count': 'sum', 'volume_delta': 'sum'}
            valid_agg_rules = {k: v for k, v in agg_rules.items() if k in base_df.columns}
            df_resampled = base_df.resample(freq).agg(valid_agg_rules).ffill().reindex(self.base_timestamps, method='ffill').fillna(0)
            self.timeframes_np[freq] = {col: df_resampled[col].values for col in df_resampled.columns}
        if precomputed_features is not None:
            features_aligned = precomputed_features.set_index('timestamp').reindex(self.base_timestamps, method='ffill').fillna(0.0)
            self.all_features_np = {key: features_aligned[key].values for key in self.strat_cfg.context_feature_keys + self.strat_cfg.precomputed_feature_keys if key in features_aligned.columns}
        else: raise ValueError("`precomputed_features` must be provided.")
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        obs_spaces = {k.value: spaces.Box(low=-np.inf, high=np.inf, shape=((self.strat_cfg.sequence_length, l, 5) if k.value.startswith('ohlcv_') else (self.strat_cfg.sequence_length, l, 4) if k.value.startswith('ohlc_') else (self.strat_cfg.sequence_length, l)), dtype=np.float32) for k, l in self.strat_cfg.lookback_periods.items()}
        self.observation_space = spaces.Dict(obs_spaces)
        self.observation_history = deque(maxlen=self.strat_cfg.sequence_length)
        self.reset()

    def _clear_logs(self):
        if self.worker_id == 0:
            for filename in ["live_env_state.json", "live_closed_trades.jsonl"]:
                try:
                    with open(filename, "w") as f: pass
                except Exception: pass
    
    def _print_episode_start(self):
        if not self.verbose: return
        print(f"\n\033[1m\033[96m🎬 NEW EPISODE STARTING - Worker {self.worker_id}\033[0m\n{'='*80}")
        print(f"💰 Initial Balance: ${self.balance:,.2f}\n⚡ Leverage: {self.leverage}x\n📊 Data Points: {self.max_step:,}\n🎯 Start Step: {self.current_step}\n{'='*80}\n")

    def step(self, action: np.ndarray):
        try:
            self._update_market_regime_and_volatility(self.current_step)
            current_price = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][self.current_step]
            unrealized_pnl = sum(qty * (current_price - price) for qty, price, step in self.open_positions)
            initial_portfolio_value = self.balance + unrealized_pnl
            if initial_portfolio_value > self.episode_peak_value: self.episode_peak_value = initial_portfolio_value
            
            current_drawdown = (self.episode_peak_value - initial_portfolio_value) / max(self.episode_peak_value, 1e-9)
            
            action_signal = np.clip(action[0], -1.0, 1.0)
            action_size = (np.clip(action[1], -1.0, 1.0) + 1.0) / 2.0
            
            target_asset_quantity = 0.0
            if action_size >= 0.01:
                max_position = action_size * self.risk_manager.calculate_dynamic_position_limit(self.volatility_estimate, initial_portfolio_value, self.market_regime)
                target_notional = initial_portfolio_value * np.sign(action_signal) * max_position
                target_asset_quantity = target_notional / max(current_price, 1e-8)
            
            trade_quantity = target_asset_quantity - self.total_asset_held
            total_cost = abs(trade_quantity) * current_price * (self.cfg.transaction_fee_pct + self.cfg.slippage_pct)
            self.balance -= total_cost
            
            realized_pnl = 0.0
            if trade_quantity > 1e-9:
                self.open_positions.append((trade_quantity, current_price, self.current_step))
            elif trade_quantity < -1e-9:
                quantity_to_sell = abs(trade_quantity)
                while quantity_to_sell > 0 and self.open_positions:
                    oldest_qty, oldest_price, entry_step = self.open_positions[0]
                    if oldest_qty <= quantity_to_sell:
                        pnl = oldest_qty * (current_price - oldest_price)
                        realized_pnl += pnl
                        self._log_closed_trade(oldest_qty, oldest_price, entry_step, current_price, pnl)
                        self.open_positions.popleft()
                        quantity_to_sell -= oldest_qty
                    else:
                        pnl = quantity_to_sell * (current_price - oldest_price)
                        realized_pnl += pnl
                        self._log_closed_trade(quantity_to_sell, oldest_price, entry_step, current_price, pnl)
                        self.open_positions[0] = (oldest_qty - quantity_to_sell, oldest_price, entry_step)
                        quantity_to_sell = 0
            
            self.balance += realized_pnl
            
            self.current_step += 1
            next_price = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][self.current_step]
            next_unrealized_pnl = sum(qty * (next_price - price) for qty, price, step in self.open_positions)
            next_portfolio_value = self.balance + next_unrealized_pnl

            is_executed_trade = action_size >= 0.01
            self.total_executed_trades += 1 if is_executed_trade else 0
            if is_executed_trade:
                current_position_sign = np.sign(self.total_asset_held)
                if len(self.position_history_buffer) > 0 and current_position_sign != self.position_history_buffer[-1] and self.position_history_buffer[-1] != 0:
                    self.total_flips += 1
                self.position_history_buffer.append(current_position_sign)
            thrashing_ratio = self.total_flips / self.total_executed_trades if self.total_executed_trades > 5 else 0.0

            costs = {
                'drawdown': current_drawdown - self.constraint_targets.get('drawdown', 1.0),
                'trade_cost_pct': (total_cost / max(initial_portfolio_value, 1e-9)) - self.constraint_targets.get('trade_cost_pct', 1.0),
                'thrashing_rate': thrashing_ratio - self.constraint_targets.get('thrashing_rate', 1.0)
            }

            pnl_reward = realized_pnl / self.initial_balance
            penalty = sum(self.lambdas[key] * max(0, costs[key]) for key in costs)
            reward = pnl_reward - penalty

            terminated = next_portfolio_value <= self.initial_balance * (1.0 - self.strat_cfg.max_drawdown_threshold)
            truncated = self.current_step >= self.max_step
            
            info = {
                'portfolio_value': next_portfolio_value, 
                'costs': costs,
                'realized_pnl': realized_pnl,
                'total_cost': total_cost,
                'drawdown': current_drawdown,
                'thrashing_ratio': thrashing_ratio,
                'total_executed_trades': self.total_executed_trades
            }
            
            self._log_live_state(action_signal, action_size, reward, info, next_portfolio_value)
            self.observation_history.append(self._get_single_step_observation(self.current_step))
            return self._get_observation_sequence(), reward, terminated, truncated, info
        except Exception as e:
            logger.exception(f"FATAL ERROR in Worker {self.worker_id} on step {self.current_step}. This worker will crash.")
            raise e

    def reset(self, seed=None, options=None):
        try:
            super().reset(seed=seed)
            self._clear_logs()
            self.initial_balance = 1000000.0
            self.balance = self.initial_balance
            self.episode_peak_value = self.balance
            self.open_positions = deque()
            self.total_flips = 0
            self.total_executed_trades = 0
            self.position_history_buffer.clear()
            self.portfolio_history = [self.initial_balance]
            self.volatility_estimate = 0.01
            self.market_regime = "UNCERTAIN"
            warmup_period = self.cfg.get_required_warmup_period()
            start_step_range = (warmup_period, self.max_step - 5000)
            if options and 'start_step' in options:
                self.current_step = max(start_step_range[0], min(options['start_step'], start_step_range[1]))
            else: self.current_step = self.np_random.integers(start_step_range[0], start_step_range[1]) if start_step_range[0] < start_step_range[1] else start_step_range[0]
            self.observation_history.clear()
            for i in range(self.strat_cfg.sequence_length):
                step_idx = self.current_step - self.strat_cfg.sequence_length + 1 + i
                self._update_market_regime_and_volatility(step_idx)
                self.observation_history.append(self._get_single_step_observation(step_idx))
            self._print_episode_start()
            return self._get_observation_sequence(), {'portfolio_value': self.balance, 'balance': self.balance, 'asset_held': 0.0}
        except Exception as e:
            logger.exception(f"FATAL ERROR in Worker {getattr(self, 'worker_id', 'N/A')} during environment reset. This worker will crash.")
            raise e
    
    def _get_current_context_features(self, step_index: int) -> np.ndarray:
        return np.array([self.all_features_np[key][step_index] for key in self.strat_cfg.context_feature_keys], dtype=np.float32)

    def _update_market_regime_and_volatility(self, step_index: int):
        if step_index >= 50:
            try:
                prices = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][max(0, step_index - 50) : step_index + 1]
                returns = pd.Series(prices).pct_change().dropna().values
                if len(returns) > 10:
                    base_bar_seconds = self.cfg.get_timeframe_seconds(self.cfg.base_bar_timeframe)
                    bars_per_day = (24 * 3600) / base_bar_seconds
                    annualization_factor = np.sqrt(bars_per_day)
                    self.volatility_estimate = np.std(returns) * annualization_factor
                    self.risk_manager.volatility_buffer.append(self.volatility_estimate)
                    self.market_regime = self.risk_manager.update_market_regime(returns, self.volatility_estimate)
            except Exception as e: 
                logger.warning(f"Could not update market regime at step {step_index}: {e}")

    def _get_single_step_observation(self, step_index) -> dict:
        try:
            raw_obs = {}
            current_price = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][step_index]
            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value
                if key in [FeatureKeys.CONTEXT.value, FeatureKeys.PORTFOLIO_STATE.value, FeatureKeys.PRECOMPUTED_FEATURES.value]: continue
                freq = key.split('_')[-1]
                if key.startswith('ohlc'):
                    cols = ['open','high','low','close','volume'] if 'v' in key else ['open','high','low','close']
                    window_data = np.stack([self.timeframes_np[freq][c][max(0, step_index - lookback + 1) : step_index + 1] for c in cols], axis=1)
                else:
                    window_data = self.timeframes_np[freq]['close'][max(0, step_index - lookback + 1) : step_index + 1]
                if len(window_data) < lookback: window_data = np.pad(window_data, [(lookback - len(window_data), 0)] + [(0,0)]*(window_data.ndim-1), 'edge')
                raw_obs[key] = window_data.astype(np.float32)
            raw_obs[FeatureKeys.CONTEXT.value] = self._get_current_context_features(step_index)
            raw_obs[FeatureKeys.PRECOMPUTED_FEATURES.value] = np.array([self.all_features_np[k][step_index] for k in self.strat_cfg.precomputed_feature_keys], dtype=np.float32)
            unrealized_pnl = sum(qty * (current_price - price) for qty, price, step in self.open_positions)
            portfolio_value = self.balance + unrealized_pnl
            position_value = self.total_asset_held * current_price
            raw_obs[FeatureKeys.PORTFOLIO_STATE.value] = np.array([
                np.clip(position_value / (portfolio_value + 1e-9), -self.leverage, self.leverage),
                np.tanh(unrealized_pnl / (portfolio_value + 1e-9)),
                self.risk_manager.get_volatility_percentile(self.volatility_estimate),
                (self.episode_peak_value - portfolio_value) / max(self.episode_peak_value, 1e-9)], dtype=np.float32)
            return self.normalizer.transform(raw_obs)
        except Exception as e:
            logger.exception(f"FATAL ERROR in Worker {self.worker_id} getting observation at step {step_index}. This worker will crash.")
            raise e

    def _get_observation_sequence(self):
        try:
            return {key: np.stack([obs[key] for obs in self.observation_history]) for key in self.observation_space.spaces.keys()}
        except Exception as e:
            logger.exception(f"FATAL ERROR in Worker {self.worker_id} stacking observation sequence. This worker will crash.")
            raise e

    def _log_closed_trade(self, quantity: float, entry_price: float, entry_step: int, exit_price: float, profit: float):
        if self.worker_id != 0: return
        trade_data = {'entry_step': entry_step, 'exit_step': self.current_step, 'quantity': round(quantity, 6),
                      'entry_price': round(entry_price, 2), 'exit_price': round(exit_price, 2), 'profit': round(profit, 2)}
        try:
            with open("live_closed_trades.jsonl", "a") as f:
                json.dump(trade_data, f)
                f.write('\n')
        except Exception: pass

    def _log_live_state(self, action_signal, action_size, reward, reward_info, next_portfolio_value):
        if self.worker_id != 0: return
        state_data = {
            'timestamp': datetime.now().isoformat(), 'step': self.current_step,
            'price': self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][self.current_step],
            'balance': self.balance, 'portfolio_value': next_portfolio_value, 'peak_value': self.episode_peak_value,
            'drawdown': reward_info.get('drawdown', 0.0),
            'action_signal': float(action_signal), 'action_size': float(action_size), 'final_reward': reward,
            'cost_components': {k: round(v, 5) for k, v in reward_info.get('costs', {}).items()},
            'lambda_penalties': {k: round(self.lambdas.get(k,0), 3) for k in self.constraint_targets.keys()},
            'open_positions': [(round(q, 6), round(p, 2)) for q, p, s in self.open_positions],
            'total_asset_held': self.total_asset_held, 'avg_entry_price': self.average_entry_price,
        }
        try:
            with open("live_env_state.json", "w") as f: json.dump(state_data, f)
        except Exception: pass

    def get_performance_metrics(self) -> Dict[str, Any]:
        try:
            if not hasattr(self, 'portfolio_history') or len(self.portfolio_history) < 2:
                self.portfolio_history = [self.initial_balance]
                return {}
            portfolio_values = np.array(self.portfolio_history)
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            base_bar_seconds = self.cfg.get_timeframe_seconds(self.cfg.base_bar_timeframe)
            bars_per_year = (365 * 24 * 3600) / base_bar_seconds
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            volatility = np.std(returns) * np.sqrt(bars_per_year) if len(returns) > 1 else 0.0
            total_return = (final_value - initial_value) / initial_value
            num_periods = len(returns)
            annualized_return = ((1 + total_return) ** (bars_per_year / num_periods) - 1) if num_periods > 0 else 0.0
            risk_free_rate = 0.02
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0.0
            cumulative_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (cumulative_max - portfolio_values) / cumulative_max
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
            return {'total_return': total_return, 'annualized_return': annualized_return, 'volatility_annualized': volatility,
                    'max_drawdown': max_drawdown, 'sharpe_ratio': sharpe_ratio, 'final_portfolio_value': final_value}
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
