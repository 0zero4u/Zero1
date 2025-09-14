"""
FULLY OVERHAULED & HARDENED TRADING ENGINE (POST-ANALYSIS)

This version represents a complete architectural overhaul based on a comprehensive analysis
of critical bugs and design flaws. All identified issues have been resolved.

--- START OF MAJOR ARCHITECTURAL CHANGES ---
CRITICAL FIX #1: PnL Double-Counting ELIMINATED.
    - The environment now correctly distinguishes between realized and unrealized PnL.
    - The account balance is ONLY updated with realized PnL when a position is
      partially or fully closed, eliminating the "infinite money" bug.

CRITICAL FIX #2: Entry Price Calculation CORRECTED.
    - The flawed `abs()` wrapper has been removed.
    - The logic now correctly calculates the weighted average entry price when adding to
      a position (for both longs and shorts) and maintains the entry price when reducing.
    - Safety checks for division-by-zero are included.

ARCHITECTURAL FIX #3: New Reward System - Per-Component Normalization.
    - The old, unstable reward logic has been replaced with a modular `RewardManager`.
    - EVERY reward component (return, fees, penalties) is now normalized independently
      using a robust `PerComponentNormalizer` class.
    - This eliminates reward signal instability across different market regimes.
    - The new normalizer includes warmup periods, clipping, and squashing for stability.

ARCHITECTURAL FIX #4: Exploit-Resistant Reward Shaping.
    - All raw reward components are first transformed into unitless RATIOS (e.g., fee as a
      percentage of portfolio value) before being passed to the normalizers.
    - This prevents agents from gaming the reward system by manipulating the scale of actions.
    - Complex, opaque penalty calculations (e.g., frequency penalty) have been replaced
      with simple, predictable, and interpretable logic.
--- END OF MAJOR ARCHITECTURAL CHANGES ---
"""

import numpy as np
import pandas as pd
import gymnasium
from gymnasium import spaces
from collections import deque
from typing import Dict, Any, Optional, Tuple
import logging
from scipy import stats

# Import configuration and stateful feature calculators
from config import SETTINGS, FeatureKeys
from normalizer import Normalizer
from features import (
    StatefulBBWPercentRank,
    StatefulPriceDistanceMA,
    StatefulSRDistances,
    StatefulVWAPDistance,
)

logger = logging.getLogger(__name__)

# --- START OF NEW, HARDENED REWARD ARCHITECTURE ---

class PerComponentNormalizer:
    """
    Robust online normalizer for a single reward component.
    Features warmup, raw value clipping, and tanh squashing for stability.
    """
    def __init__(self, warmup: int = 500, clip_raw_abs: float = 10.0, squash_k: float = 2.0, clip_final: float = 5.0, epsilon: float = 1e-8):
        self.warmup = warmup
        self.clip_raw_abs = clip_raw_abs
        self.squash_k = squash_k
        self.clip_final = clip_final
        self.epsilon = epsilon
        self.mean, self.std, self.variance, self.count, self.M2 = 0.0, 1.0, 1.0, 0, 0.0

    def update(self, raw_value: float):
        # Clip raw value before updating stats to handle extreme outliers
        x = np.clip(raw_value, -self.clip_raw_abs, self.clip_raw_abs)
        
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if self.count > 1:
            self.variance = self.M2 / self.count
            self.std = np.sqrt(self.variance)

    def normalize(self, raw_value: float) -> float:
        # During warmup, use a more conservative standard deviation to prevent explosions
        current_std = self.std if self.count > self.warmup else max(self.std, 1.0)
        
        # Normalize (z-score)
        z = (raw_value - self.mean) / (current_std + self.epsilon)
        
        # Squash the normalized value to keep tails compressed and apply final clip
        z_squash = np.tanh(z / self.squash_k) * self.squash_k
        return float(np.clip(z_squash, -self.clip_final, self.clip_final))

class RewardManager:
    """
    Manages the entire reward calculation pipeline using per-component normalization.
    This ensures a stable, balanced, and exploit-resistant reward signal.
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'base_return': 1.0,
            'transaction_penalty': -0.6,
            'drawdown_penalty': -1.0,
            'frequency_penalty': -0.5,
            'inactivity_penalty': -0.2,
            'tiny_action_penalty': -0.1,
        }
        self.normalizers = {
            # Each component gets its own dedicated, tuned normalizer
            'base_return': PerComponentNormalizer(clip_raw_abs=1.0),
            'transaction_penalty': PerComponentNormalizer(clip_raw_abs=0.1),
            'drawdown_penalty': PerComponentNormalizer(clip_raw_abs=1.0),
            'frequency_penalty': PerComponentNormalizer(clip_raw_abs=5.0),
            'inactivity_penalty': PerComponentNormalizer(clip_raw_abs=5.0),
            'tiny_action_penalty': PerComponentNormalizer(clip_raw_abs=1.0),
        }

    def calculate_reward(self, raw_components: Dict[str, float]) -> Tuple[float, Dict, Dict]:
        normed_components = {}
        for k, raw_value in raw_components.items():
            if k in self.normalizers:
                self.normalizers[k].update(raw_value)
                normed_components[k] = self.normalizers[k].normalize(raw_value)
        
        total_reward = sum(self.weights.get(k, 0) * normed_components.get(k, 0) for k in normed_components)
        weighted_normed = {k: self.weights.get(k, 0) * normed_components.get(k, 0) for k in normed_components}

        return float(np.clip(total_reward, -10.0, 10.0)), normed_components, weighted_normed

# --- END OF NEW REWARD ARCHITECTURE ---

class EnhancedRiskManager:
    """Advanced risk management system with dynamic limits and progressive penalties"""
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
    """FULLY FIXED Trading environment with hardened reward architecture and correct PnL accounting."""
    def __init__(self, df_base_ohlc: pd.DataFrame, normalizer: Normalizer, config=None,
                 leverage: float = None, reward_weights: Dict[str, float] = None,
                 precomputed_features: Optional[pd.DataFrame] = None,
                 worker_id: int = 0):
        super().__init__()
        self.worker_id, self.verbose = worker_id, worker_id == 0
        self.COLOR_GREEN, self.COLOR_RED, self.COLOR_YELLOW, self.COLOR_CYAN, self.COLOR_RESET, self.COLOR_BOLD = '\033[92m', '\033[91m', '\033[93m', '\033[96m', '\033[0m', '\033[1m'
        self.cfg, self.strat_cfg, self.normalizer = config or SETTINGS, (config or SETTINGS).strategy, normalizer
        self.leverage = leverage if leverage is not None else self.strat_cfg.leverage
        
        try:
            self.risk_manager = EnhancedRiskManager(self.cfg, leverage=self.leverage)
            self.reward_manager = RewardManager(weights=reward_weights) # Instantiate the new RewardManager
            
            self.action_frequency_buffer = deque(maxlen=100)
            self.inactivity_grace_period = getattr(self.strat_cfg, 'inactivity_grace_period_steps', 10)
            
            base_df = df_base_ohlc.set_index('timestamp')
            all_required_freqs = set(k.value.split('_')[-1].replace('m','T').replace('h','H').replace('s','S').upper() for k in self.strat_cfg.lookback_periods.keys() if k not in {FeatureKeys.CONTEXT, FeatureKeys.PORTFOLIO_STATE, FeatureKeys.PRECOMPUTED_FEATURES}).union({c.timeframe for c in self.strat_cfg.stateful_calculators})
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
            else:
                raise ValueError("`precomputed_features` must be provided.")
            
            self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
            obs_spaces = {k.value: spaces.Box(low=-np.inf, high=np.inf, shape=((self.strat_cfg.sequence_length, l, 5) if k.value.startswith('ohlcv_') else (self.strat_cfg.sequence_length, l, 4) if k.value.startswith('ohlc_') else (self.strat_cfg.sequence_length, l)), dtype=np.float32) for k, l in self.strat_cfg.lookback_periods.items()}
            self.observation_space = spaces.Dict(obs_spaces)
            self.observation_history = deque(maxlen=self.strat_cfg.sequence_length)
            
            self.reset()
        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}", exc_info=True)
            raise

    def _print_trade_info(self, trade_type: str, current_price: float, portfolio_value: float, unrealized_pnl: float, reward: float, weighted_rewards: Dict[str, float]):
        if not self.verbose: return
        self.trade_counter += 1
        pnl_color, reward_color = (self.COLOR_GREEN if x >= 0 else self.COLOR_RED for x in [unrealized_pnl, reward])
        print(f"\n{self.COLOR_BOLD}{'='*80}{self.COLOR_RESET}\n{self.COLOR_CYAN}🚀 TRADE #{self.trade_counter} | Worker {self.worker_id} | Step {self.current_step}{self.COLOR_RESET}")
        print(f"📈 Price: {self.COLOR_YELLOW}${current_price:,.2f}{self.COLOR_RESET} | Entry: ${self.entry_price:,.2f} | Portfolio: ${portfolio_value:,.2f}")
        print(f"💰 PnL: {pnl_color}${unrealized_pnl:+,.2f}{self.COLOR_RESET} | Reward: {reward_color}{reward:+.4f}{self.COLOR_RESET}")
        significant_components = {k: v for k, v in weighted_rewards.items() if abs(v) > 1e-5}
        if significant_components:
            print("🔍 Key Reward Components:" + "".join(f"\n   • {c}: {(self.COLOR_GREEN if v >= 0 else self.COLOR_RED)}{v:+.4f}{self.COLOR_RESET}" for c, v in sorted(significant_components.items(), key=lambda x: abs(x[1]), reverse=True)[:5]))
        print(f"{self.COLOR_BOLD}{'='*80}{self.COLOR_RESET}\n")

    def _print_episode_start(self):
        if not self.verbose: return
        print(f"\n{self.COLOR_BOLD}{self.COLOR_CYAN}🎬 NEW EPISODE STARTING - Worker {self.worker_id}{self.COLOR_RESET}\n{'='*80}")
        print(f"💰 Initial Balance: ${self.balance:,.2f}\n⚡ Leverage: {self.leverage}x\n📊 Data Points: {self.max_step:,}\n🎯 Start Step: {self.current_step}\n{'='*80}\n")
        self.trade_counter = 0

    def step(self, action: np.ndarray):
        try:
            self._update_market_regime_and_volatility(self.current_step)
            current_price = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][self.current_step]
            
            # --- 1. Calculate initial state for PnL and drawdown ---
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            initial_portfolio_value = self.balance + unrealized_pnl
            if initial_portfolio_value > self.episode_peak_value: self.episode_peak_value = initial_portfolio_value
            current_drawdown = (self.episode_peak_value - initial_portfolio_value) / max(self.episode_peak_value, 1e-9)
            
            # --- 2. Determine Action & Target Position ---
            action_signal, action_size = np.clip(action[0], -1.0, 1.0), np.clip(action[1], 0.0, 1.0)
            target_asset_quantity = 0.0
            if action_size >= 0.01:
                max_position = action_size * self.risk_manager.calculate_dynamic_position_limit(self.volatility_estimate, initial_portfolio_value, self.market_regime)
                target_notional = initial_portfolio_value * np.sign(action_signal) * max_position
                target_asset_quantity = target_notional / max(current_price, 1e-8)
            
            trade_quantity = target_asset_quantity - self.asset_held
            previous_asset_held = self.asset_held
            
            # --- 3. CRITICAL FIX: Correctly Calculate Realized PnL ---
            realized_pnl = 0.0
            if abs(previous_asset_held) > 1e-9 and abs(trade_quantity) > 1e-9:
                is_reducing = abs(target_asset_quantity) < abs(previous_asset_held)
                is_flipping = np.sign(target_asset_quantity) * np.sign(previous_asset_held) < 0
                if is_reducing or is_flipping:
                    closed_quantity = abs(previous_asset_held) if is_flipping else abs(previous_asset_held) - abs(target_asset_quantity)
                    realized_pnl = np.sign(previous_asset_held) * closed_quantity * (current_price - self.entry_price)
            
            # --- 4. CRITICAL FIX: Update Balance and Entry Price Correctly ---
            self.balance += realized_pnl # ONLY add realized PnL to balance
            
            if abs(trade_quantity) > 1e-9:
                if previous_asset_held == 0 or np.sign(target_asset_quantity) * np.sign(previous_asset_held) < 0:
                    self.entry_price = current_price
                elif abs(target_asset_quantity) > abs(previous_asset_held):
                    total_cost = (previous_asset_held * self.entry_price) + (trade_quantity * current_price)
                    if abs(target_asset_quantity) > 1e-9:
                        self.entry_price = total_cost / target_asset_quantity
                self.trade_count += 1
            
            self.asset_held = target_asset_quantity
            if abs(self.asset_held) < 1e-9: self.entry_price = 0.0
            
            # --- 5. Calculate Costs & Final State ---
            total_cost = abs(trade_quantity) * current_price * (self.cfg.transaction_fee_pct + self.cfg.slippage_pct)
            self.balance -= total_cost
            
            self.current_step += 1
            next_price = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][self.current_step]
            next_unrealized_pnl = self.asset_held * (next_price - self.entry_price)
            next_portfolio_value = self.balance + next_unrealized_pnl

            # --- 6. ARCHITECTURAL FIX: Calculate Raw Reward Components as Ratios ---
            raw_components = {}
            raw_components['base_return'] = (next_portfolio_value - initial_portfolio_value) / max(initial_portfolio_value, 1e-9)
            raw_components['transaction_penalty'] = - (total_cost / max(initial_portfolio_value, 1e-9))
            if current_drawdown > 0.08:
                raw_components['drawdown_penalty'] = -((current_drawdown - 0.08) ** 1.5)

            self.action_frequency_buffer.append(1 if abs(trade_quantity) > 1e-8 else 0)
            if len(self.action_frequency_buffer) == 100:
                current_frequency = np.mean(list(self.action_frequency_buffer))
                allowed_frequency = 0.3 + np.clip(self.volatility_estimate, 0.1, 0.4)
                if current_frequency > allowed_frequency:
                    raw_components['frequency_penalty'] = -((current_frequency - allowed_frequency) ** 1.5)

            total_action_magnitude = abs(action_signal) * action_size
            if 1e-9 < total_action_magnitude < 0.005:
                raw_components['tiny_action_penalty'] = -1.0

            # --- 7. Delegate to the RewardManager for final reward ---
            reward, normed_components, weighted_rewards = self.reward_manager.calculate_reward(raw_components)

            if self.verbose and abs(trade_quantity) > 1e-8:
                self._print_trade_info("TRADE", current_price, next_portfolio_value, next_unrealized_pnl, reward, weighted_rewards)

            self.observation_history.append(self._get_single_step_observation(self.current_step))
            terminated = next_portfolio_value <= self.initial_balance * (1.0 - self.strat_cfg.max_drawdown_threshold)
            truncated = self.current_step >= self.max_step
            info = {
                'portfolio_value': next_portfolio_value,
                'raw_reward_components': raw_components,
                'normed_reward_components': normed_components,
                'weighted_rewards': weighted_rewards,
                'final_reward': reward
            }
            return self._get_observation_sequence(), reward, terminated, truncated, info
        except Exception as e:
            logger.error(f"Error in environment step: {e}", exc_info=True)
            return self._get_observation_sequence(), -1.0, True, False, {'error': True}

    def reset(self, seed=None, options=None):
        try:
            super().reset(seed=seed)
            self.initial_balance = 1000000.0
            self.balance = self.initial_balance
            self.asset_held, self.entry_price = 0.0, 0.0
            self.episode_peak_value = self.balance
            
            # Initialize state variables to safe defaults
            self.volatility_estimate = 0.01
            self.market_regime = "UNCERTAIN"
            self.portfolio_history = []
            
            self.trade_counter = 0
            self.trade_count = 0
            
            warmup_period = self.cfg.get_required_warmup_period()
            start_step_range = (warmup_period, self.max_step - 5000)
            if options and 'start_step' in options:
                self.current_step = max(start_step_range[0], min(options['start_step'], start_step_range[1]))
            else:
                self.current_step = self.np_random.integers(start_step_range[0], start_step_range[1]) if start_step_range[0] < start_step_range[1] else start_step_range[0]
            
            self.observation_history.clear()
            for i in range(self.strat_cfg.sequence_length):
                step_idx = self.current_step - self.strat_cfg.sequence_length + 1 + i
                self._update_market_regime_and_volatility(step_idx)
                self.observation_history.append(self._get_single_step_observation(step_idx))
            
            self._print_episode_start()
            return self._get_observation_sequence(), {'portfolio_value': self.balance}
        except Exception as e:
            logger.error(f"Error resetting environment: {e}", exc_info=True)
            raise

    def _get_current_context_features(self, step_index: int) -> np.ndarray:
        return np.array([self.all_features_np[key][step_index] for key in self.strat_cfg.context_feature_keys], dtype=np.float32)

    def _update_market_regime_and_volatility(self, step_index: int):
        try:
            if step_index >= 50:
                prices = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][max(0, step_index - 50) : step_index + 1]
                returns = pd.Series(prices).pct_change().dropna().values
                if len(returns) > 10:
                    # Assuming base timeframe is 1 minute for this annualization factor
                    bars_per_day = 24 * 60 
                    annualization_factor = np.sqrt(bars_per_day * 252)
                    self.volatility_estimate = np.std(returns) * annualization_factor
                    self.risk_manager.volatility_buffer.append(self.volatility_estimate)
                    self.market_regime = self.risk_manager.update_market_regime(returns, self.volatility_estimate)
        except Exception as e:
            logger.warning(f"Error updating market regime at step {step_index}: {e}")

    def _get_single_step_observation(self, step_index) -> dict:
        try:
            raw_obs = {}
            current_price = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][step_index]
            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value
                if key in [FeatureKeys.CONTEXT.value, FeatureKeys.PORTFOLIO_STATE.value, FeatureKeys.PRECOMPUTED_FEATURES.value]: continue
                freq = key.split('_')[-1].replace('m','T').replace('h','H').replace('s', 'S').upper()
                
                if key.startswith('ohlc'):
                    cols = ['open','high','low','close','volume'] if 'v' in key else ['open','high','low','close']
                    window_data = np.stack([self.timeframes_np[freq][c][max(0, step_index - lookback + 1) : step_index + 1] for c in cols], axis=1)
                else:
                    col_name = 'volume_delta' if 'volume_delta' in key else 'close'
                    window_data = self.timeframes_np[freq][col_name][max(0, step_index - lookback + 1) : step_index + 1]
                
                if len(window_data) < lookback: window_data = np.pad(window_data, [(lookback - len(window_data), 0)] + [(0,0)]*(window_data.ndim-1), 'edge')
                raw_obs[key] = window_data.astype(np.float32)
            
            raw_obs[FeatureKeys.CONTEXT.value] = self._get_current_context_features(step_index)
            raw_obs[FeatureKeys.PRECOMPUTED_FEATURES.value] = np.array([self.all_features_np[k][step_index] for k in self.strat_cfg.precomputed_feature_keys], dtype=np.float32)
            
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            portfolio_value = self.balance + unrealized_pnl
            position_value = self.asset_held * current_price
            
            raw_obs[FeatureKeys.PORTFOLIO_STATE.value] = np.array([
                np.clip(position_value / (portfolio_value + 1e-9), -self.leverage, self.leverage),
                np.tanh(unrealized_pnl / (portfolio_value + 1e-9)),
                self.risk_manager.get_volatility_percentile(self.volatility_estimate),
                current_drawdown
            ], dtype=np.float32)
            
            return self.normalizer.transform(raw_obs)
        except Exception as e:
            logger.error(f"Error getting observation at step {step_index}: {e}", exc_info=True)
            return {k: np.zeros(s.shape, dtype=np.float32) for k, s in self.observation_space.spaces.items()}

    def _get_observation_sequence(self):
        try:
            return {key: np.stack([obs[key] for obs in self.observation_history]) for key in self.observation_space.spaces.keys()}
        except Exception:
            return {key: np.zeros(space.shape, dtype=np.float32) for key, space in self.observation_space.spaces.items()}


    def get_performance_metrics(self) -> Dict[str, Any]:
        try:
            if len(self.portfolio_history) < 2: return {}
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
            positive_periods = np.sum(returns > 0)
            positive_period_ratio = positive_periods / len(returns) if len(returns) > 0 else 0.0
            cumulative_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (cumulative_max - portfolio_values) / cumulative_max
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility_annualized': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'positive_period_ratio': positive_period_ratio,
                'total_trades': self.trade_count,
                'final_portfolio_value': final_value,
            }
            return metrics
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}