"""
Enhanced Trading Environment for Crypto Trading RL
Gymnasium-compliant environment with isolated margin simulation and enhanced features.
This version uses stateful, incremental feature calculators for massive performance gains.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Dict, List
import logging
from tqdm import tqdm

# Import configuration and the new feature calculators
from config import SETTINGS, FeatureKeys
from normalizer import Normalizer
from features import (
    StatefulBBWPercentRank, StatefulPriceDistanceMA, StatefulSRDistances
)

logger = logging.getLogger(__name__)

class HierarchicalTradingEnvironment(gym.Env):
    """
    Enhanced Gymnasium-compliant RL environment with:
    - CAUSAL and HIGH-PERFORMANCE feature calculation using stateful calculators.
    - Agent state awareness (portfolio state included in observation).
    - Isolated margin simulation and comprehensive risk management.
    """

    def __init__(self, df_base_ohlc: pd.DataFrame, normalizer: Normalizer, config=None):
        super().__init__()

        self.cfg = config or SETTINGS
        self.strat_cfg = self.cfg.strategy
        self.normalizer = normalizer

        logger.info("--- Initializing Enhanced Hierarchical Trading Environment ---")
        logger.info(f" -> Leverage: {self.strat_cfg.leverage}x | Maintenance Margin: {self.strat_cfg.maintenance_margin_rate:.2%}")

        try:
            # Prepare base data
            base_df = df_base_ohlc.set_index('timestamp')

            # Get required timeframes
            model_timeframes = set(k.value for k in self.strat_cfg.lookback_periods.keys())
            feature_timeframes = {'1H', '4H', '3T', '15T'}
            all_required_keys = model_timeframes.union(feature_timeframes)

            self.timeframes = {}
            logger.info("Resampling data for all required timeframes...")

            for key in all_required_keys:
                if key in [FeatureKeys.CONTEXT.value, FeatureKeys.PORTFOLIO_STATE.value]:
                    continue
                freq_str = key.split('_')[-1] if '_' in key else key
                freq = freq_str.replace('m','T').replace('h','H').replace('s', 'S').upper()
                if freq not in self.timeframes:
                    agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'volume_delta': 'sum', 'vwap': 'last'}
                    valid_agg_rules = {k: v for k, v in agg_rules.items() if k in base_df.columns}
                    df_resampled = base_df.resample(freq).agg(valid_agg_rules).ffill()
                    self.timeframes[freq] = df_resampled.dropna()

            self.base_timestamps = self.timeframes[self.cfg.base_bar_timeframe.value].index
            self.max_step = len(self.base_timestamps) - 2

            # Action and Observation spaces (unchanged)
            self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
            obs_spaces = {}
            seq_len = self.strat_cfg.sequence_length
            for key, lookback in self.strat_cfg.lookback_periods.items():
                key_str = key.value
                shape = (seq_len, lookback, 5) if key_str.startswith('ohlcv_') else \
                        (seq_len, lookback, 4) if key_str.startswith('ohlc_') else \
                        (seq_len, lookback)
                obs_spaces[key_str] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
            self.observation_space = spaces.Dict(obs_spaces)
            self.observation_history = deque(maxlen=self.strat_cfg.sequence_length)

            # Portfolio variables
            self.portfolio_history = deque(maxlen=252)
            self.previous_portfolio_value = None
            self.episode_returns = []
            self.previous_action = None

            # --- NEW: Initialize stateful calculators ---
            self.feature_calculators = {}
            self.feature_cache = {}
            self.last_timeframe_indices = {}

            logger.info("Environment initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")
            raise

    def _initialize_and_warmup_calculators(self):
        """
        NEW: Creates and pre-fills the stateful calculators with historical data
        up to the simulation's starting point to ensure they are ready.
        """
        logger.info("Initializing and warming up stateful feature calculators...")
        self.feature_calculators.clear()
        self.last_timeframe_indices.clear()

        sr_window = self.strat_cfg.support_resistance_window

        # 1. Instantiate all calculators
        self.feature_calculators['bbw_1h_pct'] = StatefulBBWPercentRank(period=20, rank_window=250)
        self.feature_calculators['price_dist_ma_4h'] = StatefulPriceDistanceMA(period=50)
        self.feature_calculators['sr_3m'] = StatefulSRDistances(period=sr_window, num_levels=1)
        self.feature_calculators['sr_15m'] = StatefulSRDistances(period=sr_window, num_levels=2)
        self.feature_calculators['sr_1h'] = StatefulSRDistances(period=sr_window, num_levels=2)

        # 2. Warm up with data prior to the simulation start
        start_timestamp = self.base_timestamps[self.current_step]
        
        # Define which calculators to warm up for each timeframe
        warmup_plan = {
            '1H': [('bbw_1h_pct', 'close'), ('sr_1h', 'close')],
            '4H': [('price_dist_ma_4h', 'close')],
            '15T': [('sr_15m', 'close')],
            '3T': [('sr_3m', 'close')],
        }

        for tf, calcs in tqdm(warmup_plan.items(), desc="Warming Up Calculators"):
            df = self.timeframes[tf]
            warmup_data = df.loc[df.index < start_timestamp]
            if not warmup_data.empty:
                for calc_name, col_name in calcs:
                    for value in warmup_data[col_name]:
                        self.feature_calculators[calc_name].update(value)
            # Initialize last seen index
            self.last_timeframe_indices[tf] = df.index.get_loc(start_timestamp, method='ffill')

        # 3. Populate the initial feature cache
        self._update_feature_cache(start_timestamp)
        logger.info("✅ Stateful calculators warmed up and ready.")

    def _update_feature_cache(self, timestamp: pd.Timestamp):
        """
        NEW: Efficiently updates feature values only when a new bar for the
        corresponding timeframe has formed.
        """
        # --- Update 1H Features ---
        df_1h = self.timeframes['1H']
        idx_1h = df_1h.index.get_loc(timestamp, method='ffill')
        if idx_1h > self.last_timeframe_indices.get('1H', -1):
            new_close_1h = df_1h.iloc[idx_1h]['close']
            self.feature_cache['bbw_1h_pct'] = self.feature_calculators['bbw_1h_pct'].update(new_close_1h)
            sr_1h_vals = self.feature_calculators['sr_1h'].update(new_close_1h)
            for k, v in sr_1h_vals.items(): self.feature_cache[f"{k}_1h"] = v
            self.last_timeframe_indices['1H'] = idx_1h

        # --- Update 4H Features ---
        df_4h = self.timeframes['4H']
        idx_4h = df_4h.index.get_loc(timestamp, method='ffill')
        if idx_4h > self.last_timeframe_indices.get('4H', -1):
            new_close_4h = df_4h.iloc[idx_4h]['close']
            self.feature_cache['price_dist_ma_4h'] = self.feature_calculators['price_dist_ma_4h'].update(new_close_4h)
            self.last_timeframe_indices['4H'] = idx_4h

        # --- Update 3M Features ---
        df_3m = self.timeframes['3T']
        idx_3m = df_3m.index.get_loc(timestamp, method='ffill')
        if idx_3m > self.last_timeframe_indices.get('3T', -1):
            row_3m = df_3m.iloc[idx_3m]
            if 'vwap' in row_3m and row_3m['vwap'] > 1e-9:
                self.feature_cache['dist_vwap_3m'] = (row_3m['close'] - row_3m['vwap']) / row_3m['vwap']
            sr_3m_vals = self.feature_calculators['sr_3m'].update(row_3m['close'])
            for k, v in sr_3m_vals.items(): self.feature_cache[f"{k}_3m"] = v
            self.last_timeframe_indices['3T'] = idx_3m

        # --- Update 15M Features ---
        df_15m = self.timeframes['15T']
        idx_15m = df_15m.index.get_loc(timestamp, method='ffill')
        if idx_15m > self.last_timeframe_indices.get('15T', -1):
            new_close_15m = df_15m.iloc[idx_15m]['close']
            sr_15m_vals = self.feature_calculators['sr_15m'].update(new_close_15m)
            for k, v in sr_15m_vals.items(): self.feature_cache[f"{k}_15m"] = v
            self.last_timeframe_indices['15T'] = idx_15m

    def _get_current_context_features(self, step_index: int) -> np.ndarray:
        """
        REVISED: This function is now a high-speed lookup from the feature cache.
        The heavy computation is handled by the stateful `_update_feature_cache`.
        """
        feature_vector = np.array([self.feature_cache.get(key, 0.0) for key in self.strat_cfg.context_feature_keys])
        return feature_vector.astype(np.float32)

    def _get_single_step_observation(self, step_index) -> dict:
        """Construct the observation dictionary for a single point in time."""
        # This function remains largely the same, but now calls the optimized context feature function
        try:
            if self.normalizer is None: return {}
            current_timestamp = self.base_timestamps[step_index]
            raw_obs = {}
            base_df = self.timeframes[self.cfg.base_bar_timeframe.value]
            current_price = base_df['close'].iloc[base_df.index.get_loc(current_timestamp, method='ffill')]

            # --- 1. Get Market Data Features (Unchanged) ---
            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value
                if key in [FeatureKeys.CONTEXT.value, FeatureKeys.PORTFOLIO_STATE.value]: continue
                freq = key.split('_')[-1].replace('m','T').replace('h','H').replace('s', 'S').upper()
                df_tf = self.timeframes[freq]
                end_idx = df_tf.index.get_loc(current_timestamp, method='ffill')
                start_idx = max(0, end_idx - lookback + 1)
                if key.startswith('price_'):
                    window = df_tf.iloc[start_idx : end_idx + 1]['close'].values.astype(np.float32)
                    raw_obs[key] = np.pad(window, (lookback - len(window), 0), 'edge') if len(window) < lookback else window
                elif key.startswith('volume_delta_'):
                    window = df_tf.iloc[start_idx : end_idx + 1]['volume_delta'].values.astype(np.float32)
                    raw_obs[key] = np.pad(window, (lookback - len(window), 0), 'constant') if len(window) < lookback else window
                elif key.startswith('ohlc'):
                    cols = ['open','high','low','close','volume'] if 'v' in key else ['open','high','low','close']
                    window = df_tf.iloc[start_idx : end_idx + 1][cols].values.astype(np.float32)
                    if len(window) < lookback:
                        padding = np.repeat(window[0:1], lookback - len(window), axis=0)
                        window = np.concatenate([padding, window], axis=0)
                    raw_obs[key] = window
            
            # --- 2. Get Causal Context Features (Now Optimized) ---
            raw_obs[FeatureKeys.CONTEXT.value] = self._get_current_context_features(step_index)

            # --- 3. Get Agent Portfolio State (Unchanged) ---
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            portfolio_value = self.balance + unrealized_pnl
            current_notional = self.asset_held * current_price
            normalized_position = np.clip(current_notional / (portfolio_value + 1e-9), -1.0, 1.0)
            pnl_on_margin = unrealized_pnl / (self.used_margin + 1e-9)
            normalized_pnl = np.tanh(pnl_on_margin)
            position_notional = abs(self.asset_held) * current_price
            margin_ratio = np.clip((self.used_margin + unrealized_pnl) / position_notional, 0, 2.0) if position_notional > 0 else 2.0
            raw_obs[FeatureKeys.PORTFOLIO_STATE.value] = np.array([normalized_position, normalized_pnl, margin_ratio], dtype=np.float32)

            return self.normalizer.transform(raw_obs)
        except Exception as e:
            logger.error(f"Error getting observation for step {step_index}: {e}", exc_info=True)
            # Return zero observations as fallback
            obs = {}
            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value
                shape = self.observation_space.spaces[key].shape[1:]
                obs[key] = np.zeros(shape, dtype=np.float32)
            return obs

    def _get_observation_sequence(self):
        # Unchanged
        try:
            return {key: np.stack([obs[key] for obs in self.observation_history]) for key in self.observation_space.spaces.keys()}
        except Exception as e:
            logger.error(f"Error getting observation sequence: {e}")
            return {key: np.zeros(self.observation_space.spaces[key].shape, dtype=np.float32) for key in self.observation_space.spaces.keys()}

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        try:
            super().reset(seed=seed)
            self.balance = 1000000.0
            self.asset_held = 0.0
            self.used_margin = 0.0
            self.entry_price = 0.0
            self.current_step = max(self.cfg.get_required_warmup_period(), self.strat_cfg.sequence_length)
            self.observation_history.clear()
            self.portfolio_history.clear()
            self.episode_returns.clear()
            self.previous_portfolio_value = self.balance
            self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
            
            # --- REVISED: Warm up calculators on reset ---
            self._initialize_and_warmup_calculators()

            # Build initial observation history
            for i in range(self.strat_cfg.sequence_length):
                step_idx = self.current_step - self.strat_cfg.sequence_length + 1 + i
                self.observation_history.append(self._get_single_step_observation(step_idx))

            observation = self._get_observation_sequence()
            info = {'balance': self.balance, 'asset_held': self.asset_held, 'portfolio_value': self.balance}
            return observation, info
        except Exception as e:
            logger.error(f"Error resetting environment: {e}", exc_info=True)
            raise

    def step(self, action: np.ndarray):
        """Execute one environment step with enhanced margin simulation."""
        try:
            current_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step]
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            position_notional = abs(self.asset_held) * current_price
            margin_ratio = float('inf')
            if position_notional > 0:
                margin_health = self.used_margin + unrealized_pnl
                margin_ratio = margin_health / position_notional

            # Liquidation check (unchanged)
            if margin_ratio < self.strat_cfg.maintenance_margin_rate:
                self.balance -= self.used_margin
                self.asset_held = 0.0
                self.used_margin = 0.0
                self.entry_price = 0.0
                self.current_step += 1
                self._update_feature_cache(self.base_timestamps[self.current_step])
                self.observation_history.append(self._get_single_step_observation(self.current_step))
                info = {'portfolio_value': self.balance, 'margin_ratio': margin_ratio, 'liquidation': True}
                return self._get_observation_sequence(), -100, True, self.current_step >= self.max_step, info

            # Trading Logic (unchanged)
            initial_portfolio_value = self.balance + unrealized_pnl
            target_notional = initial_portfolio_value * np.clip(action[0], -1.0, 1.0) * np.clip(action[1], 0.0, 1.0)
            target_asset_quantity = target_notional / (current_price + 1e-9)
            max_allowable_margin = initial_portfolio_value * self.strat_cfg.max_margin_allocation_pct
            required_margin_for_target = (abs(target_asset_quantity) * current_price) / self.strat_cfg.leverage
            if required_margin_for_target > max_allowable_margin:
                capped_notional = max_allowable_margin * self.strat_cfg.leverage
                target_asset_quantity = (capped_notional / (current_price + 1e-9)) * np.sign(target_asset_quantity)
            if (abs(target_asset_quantity) * current_price) / self.strat_cfg.leverage > initial_portfolio_value:
                max_affordable_notional = initial_portfolio_value * self.strat_cfg.leverage
                target_asset_quantity = (max_affordable_notional / (current_price + 1e-9)) * np.sign(target_asset_quantity)
            
            trade_quantity = target_asset_quantity - self.asset_held
            fee = abs(trade_quantity) * current_price * self.cfg.transaction_fee_pct
            self.balance += unrealized_pnl - fee
            self.asset_held = target_asset_quantity
            self.used_margin = (abs(self.asset_held) * current_price) / self.strat_cfg.leverage
            self.entry_price = current_price if trade_quantity != 0 else self.entry_price

            # --- POST-TRADE CALCULATIONS ---
            self.current_step += 1
            
            # --- REVISED: Update feature cache before getting next observation ---
            self._update_feature_cache(self.base_timestamps[self.current_step])

            next_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step]
            next_unrealized_pnl = self.asset_held * (next_price - self.entry_price)
            next_portfolio_value = self.balance + next_unrealized_pnl
            terminated = next_portfolio_value <= 1000
            truncated = self.current_step >= self.max_step
            reward = self._calculate_enhanced_reward(initial_portfolio_value, next_portfolio_value, action)
            self.previous_action = action
            self.portfolio_history.append(next_portfolio_value)
            if self.previous_portfolio_value is not None:
                self.episode_returns.append((next_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value)
            self.previous_portfolio_value = next_portfolio_value
            self.observation_history.append(self._get_single_step_observation(self.current_step))

            info = {
                'balance': self.balance, 'asset_held': self.asset_held,
                'portfolio_value': next_portfolio_value, 'drawdown': self._calculate_current_drawdown(),
                'volatility': self._calculate_recent_volatility(), 'unrealized_pnl': next_unrealized_pnl,
                'margin_ratio': margin_ratio, 'used_margin': self.used_margin
            }
            return self._get_observation_sequence(), reward, terminated, truncated, info
        except Exception as e:
            logger.error(f"Error in environment step: {e}", exc_info=True)
            info = {'portfolio_value': self.balance, 'error': True}
            return self._get_observation_sequence(), -10.0, True, False, info

    def _calculate_enhanced_reward(self, prev_value: float, curr_value: float, action: np.ndarray) -> float:
        # Unchanged
        try:
            period_return = (curr_value - prev_value) / prev_value
            if len(self.episode_returns) < 10: return period_return * 100
            returns_array = np.array(self.episode_returns)
            sharpe_component = (period_return - np.mean(returns_array)) / (np.std(returns_array) + 1e-8)
            reward = period_return * 100 + sharpe_component * 10
            if self.previous_action is not None: reward -= abs(action[0] - self.previous_action[0]) * 0.1
            current_drawdown = self._calculate_current_drawdown()
            if current_drawdown > 0.1 and period_return <= 0: reward -= current_drawdown * 50
            return reward
        except Exception as e:
            logger.error(f"Error calculating reward: {e}"); return 0.0

    def _calculate_current_drawdown(self) -> float:
        # Unchanged
        if not self.portfolio_history: return 0.0
        peak = max(self.portfolio_history)
        return (peak - self.portfolio_history[-1]) / peak if peak > 0 else 0.0

    def _calculate_recent_volatility(self) -> float:
        # Unchanged
        if len(self.episode_returns) < 10: return 0.0
        return np.std(self.episode_returns[-20:]) * np.sqrt(252)

# Convenience functions (unchanged)
def create_bars_from_trades(period: str) -> pd.DataFrame:
    from processor import EnhancedDataProcessor; return EnhancedDataProcessor().create_enhanced_bars_from_trades(period)
def process_trades_for_period(period_name: str, force_reprocess: bool = False):
    from processor import EnhancedDataProcessor; return EnhancedDataProcessor().process_raw_trades_parallel(period_name, force_reprocess)

if __name__ == "__main__":
    logger.info("✅ Engine module loaded successfully with high-performance features!")