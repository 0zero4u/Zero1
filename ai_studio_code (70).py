# --- START OF FILE Zero1-main/engine.py ---

"""
Enhanced Trading Environment for Crypto Trading RL
Gymnasium-compliant environment with isolated margin simulation and enhanced features.
Fixed import issues and removed funding rate dependency.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import scipy.signal
from typing import Dict, List
import logging

# Import configuration - fixed import path
from config import SETTINGS, FeatureKeys
from normalizer import Normalizer

logger = logging.getLogger(__name__)

class HierarchicalTradingEnvironment(gym.Env):
    """
    Enhanced Gymnasium-compliant RL environment with:
    - Causal, on-the-fly feature calculation to prevent lookahead bias.
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
            feature_timeframes = {'1H', '4H', '3T', '15T'}  # Timeframes needed for context calcs
            all_required_keys = model_timeframes.union(feature_timeframes)

            self.timeframes = {}
            logger.info("Resampling data for all required timeframes...")

            for key in all_required_keys:
                # NEW: Skip special keys that are not time-based dataframes
                if key in [FeatureKeys.CONTEXT.value, FeatureKeys.PORTFOLIO_STATE.value]:
                    continue

                # Handle both formats: 'price_3m' and '3T'
                freq_str = key.split('_')[-1] if '_' in key else key
                freq = freq_str.replace('m','T').replace('h','H').replace('s', 'S').upper()


                if freq not in self.timeframes:
                    # UPDATED: Added aggregation rules for new features
                    agg_rules = {
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
                        'volume_delta': 'sum', # Sum volume delta over the period
                        'vwap': 'last'         # Take the last calculated VWAP
                    }
                    
                    # Filter rules for columns that actually exist in base_df
                    valid_agg_rules = {k: v for k, v in agg_rules.items() if k in base_df.columns}

                    df_resampled = base_df.resample(freq).agg(valid_agg_rules)
                    df_resampled = df_resampled.ffill()
                    self.timeframes[freq] = df_resampled.dropna()

            self.base_timestamps = self.timeframes[self.cfg.base_bar_timeframe.value].index

            self.max_step = len(self.base_timestamps) - 2

            # Action space: [position_signal, position_size]
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32
            )

            # UPDATED: Define observation space with new portfolio_state key
            obs_spaces = {}
            seq_len = self.strat_cfg.sequence_length

            for key, lookback in self.strat_cfg.lookback_periods.items():
                key_str = key.value # Use the string value of the enum
                if key_str.startswith('ohlcv_'):
                    shape = (seq_len, lookback, 5)  # O,H,L,C,V
                elif key_str.startswith('ohlc_'):
                    shape = (seq_len, lookback, 4)  # O,H,L,C
                # UPDATED: portfolio_state has a different shape
                elif key == FeatureKeys.PORTFOLIO_STATE:
                    shape = (seq_len, lookback) # lookback is num_features
                elif key_str.startswith('price_') or key_str.startswith('volume_delta_'):
                    shape = (seq_len, lookback)
                else:  # context
                    shape = (seq_len, lookback)

                obs_spaces[key_str] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

            self.observation_space = spaces.Dict(obs_spaces)
            self.observation_history = deque(maxlen=self.strat_cfg.sequence_length)

            # Initialize variables for enhanced reward calculation
            self.portfolio_history = deque(maxlen=252)  # For Sharpe ratio calculation
            self.previous_portfolio_value = None
            self.episode_returns = []
            self.previous_action = None

            logger.info("Environment initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")
            raise

    def _get_current_context_features(self, step_index: int) -> np.ndarray:
        """
        NEW: Calculate all context features CAUSALLY for a given step.
        This function is called at each step to prevent lookahead bias.
        """
        current_timestamp = self.base_timestamps[step_index]
        features = {}

        # 1. BBW 1H % Rank
        df_1h = self.timeframes['1H']
        end_idx_1h = df_1h.index.get_loc(current_timestamp, method='ffill')
        window_1h = df_1h.iloc[max(0, end_idx_1h - 250):end_idx_1h + 1]['close']
        if len(window_1h) >= 20:
            rolling_mean = window_1h.rolling(20).mean().iloc[-1]
            rolling_std = window_1h.rolling(20).std().iloc[-1]
            current_bbw = (4 * rolling_std) / (rolling_mean + 1e-9)
            # Percent rank calculation
            bbw_history = ((4 * window_1h.rolling(20).std()) / (window_1h.rolling(20).mean() + 1e-9)).dropna()
            if not bbw_history.empty:
                rank = (bbw_history < current_bbw).sum() / len(bbw_history)
                features['bbw_1h_pct'] = rank * 100
            else:
                features['bbw_1h_pct'] = 50.0 # Default
        else:
            features['bbw_1h_pct'] = 50.0

        # 2. Price Distance from MA 4H
        df_4h = self.timeframes['4H']
        end_idx_4h = df_4h.index.get_loc(current_timestamp, method='ffill')
        window_4h = df_4h.iloc[max(0, end_idx_4h - 50):end_idx_4h + 1]['close']
        if len(window_4h) >= 50:
            ma_4h = window_4h.mean()
            features['price_dist_ma_4h'] = (window_4h.iloc[-1] / ma_4h) - 1.0
        else:
            features['price_dist_ma_4h'] = 0.0
            
        # 3. Distance from VWAP 3M
        df_3m = self.timeframes['3T']
        if 'vwap' in df_3m.columns:
            end_idx_3m = df_3m.index.get_loc(current_timestamp, method='ffill')
            row_3m = df_3m.iloc[end_idx_3m]
            features['dist_vwap_3m'] = (row_3m['close'] - row_3m['vwap']) / (row_3m['vwap'] + 1e-9)
        else:
            features['dist_vwap_3m'] = 0.0

        # 4. Multi-timeframe S/R distances
        sr_plan = { '3T': {'levels': 1, 'suffix': '3m'}, '15T': {'levels': 2, 'suffix': '15m'}, '1H': {'levels': 2, 'suffix': '1h'} }
        sr_window = self.strat_cfg.support_resistance_window
        for freq, plan in sr_plan.items():
            df_tf = self.timeframes[freq]
            end_idx_tf = df_tf.index.get_loc(current_timestamp, method='ffill')
            price_series_window = df_tf.iloc[max(0, end_idx_tf - sr_window):end_idx_tf + 1]['close']
            sr_distances = self._calculate_multi_level_sr_distances(price_series_window, len(price_series_window), plan['levels'])
            
            # Extract the last calculated distance for each level
            for key, series in sr_distances.items():
                features[f"{key}_{plan['suffix']}"] = series.iloc[-1] if not series.empty else 1.0

        # Assemble feature vector in the correct order
        feature_vector = np.array([features.get(key, 0.0) for key in self.strat_cfg.context_feature_keys])
        return feature_vector.astype(np.float32)

    def _calculate_multi_level_sr_distances(self, price_series: pd.Series, window: int, num_levels: int) -> Dict[str, pd.Series]:
        """Calculate distance to multiple support/resistance levels."""
        try:
            results = {f'dist_s{i+1}': [] for i in range(num_levels)}
            results.update({f'dist_r{i+1}': [] for i in range(num_levels)})

            price_values = price_series.values

            for i in range(len(price_series)):
                if i < window:
                    for level in range(num_levels):
                        results[f'dist_s{level+1}'].append(1.0)
                        results[f'dist_r{level+1}'].append(1.0)
                    continue

                window_vals = price_values[max(0, i - window):i]
                current_price = price_values[i]

                if len(window_vals) < 10:  # Not enough data for reliable peaks
                    for level in range(num_levels):
                        results[f'dist_s{level+1}'].append(1.0)
                        results[f'dist_r{level+1}'].append(1.0)
                    continue

                # Find peaks and troughs
                peaks, _ = scipy.signal.find_peaks(window_vals, distance=5)
                troughs, _ = scipy.signal.find_peaks(-window_vals, distance=5)

                # Support Levels (below current price)
                support_levels = sorted([p for p in window_vals[troughs] if p < current_price], reverse=True)
                for level in range(num_levels):
                    if level < len(support_levels):
                        dist = (current_price - support_levels[level]) / current_price
                        results[f'dist_s{level+1}'].append(dist)
                    else:
                        results[f'dist_s{level+1}'].append(1.0)

                # Resistance Levels (above current price)
                resistance_levels = sorted([p for p in window_vals[peaks] if p > current_price])
                for level in range(num_levels):
                    if level < len(resistance_levels):
                        dist = (resistance_levels[level] - current_price) / current_price
                        results[f'dist_r{level+1}'].append(dist)
                    else:
                        results[f'dist_r{level+1}'].append(1.0)

            return {key: pd.Series(value, index=price_series.index) for key, value in results.items()}

        except Exception as e:
            logger.error(f"Error calculating S/R distances: {e}")
            # Return large distances (1.0) as fallback
            return {f'dist_s{i+1}': pd.Series(np.ones(len(price_series)), index=price_series.index)
                   for i in range(num_levels)} |                    {f'dist_r{i+1}': pd.Series(np.ones(len(price_series)), index=price_series.index)
                   for i in range(num_levels)}

    def _get_single_step_observation(self, step_index) -> dict:
        """Construct the observation dictionary for a single point in time."""
        try:
            # This check allows the dummy environment in the trainer to run without a normalizer
            if self.normalizer is None:
                return {}

            current_timestamp = self.base_timestamps[step_index]
            raw_obs = {}
            
            base_df = self.timeframes[self.cfg.base_bar_timeframe.value]
            current_price = base_df['close'].iloc[base_df.index.get_loc(current_timestamp, method='ffill')]

            # --- 1. Get Market Data Features ---
            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value
                if key in [FeatureKeys.CONTEXT.value, FeatureKeys.PORTFOLIO_STATE.value]:
                    continue # Handled separately

                freq = key.split('_')[-1].replace('m','T').replace('h','H').replace('s', 'S').upper()
                df_tf = self.timeframes[freq]

                end_idx = df_tf.index.get_loc(current_timestamp, method='ffill')
                start_idx = max(0, end_idx - lookback + 1)
                
                if key.startswith('price_'):
                    window = df_tf.iloc[start_idx : end_idx + 1]['close'].values.astype(np.float32)
                    if len(window) < lookback:
                        window = np.pad(window, (lookback - len(window), 0), 'edge')
                    raw_obs[key] = window

                elif key.startswith('volume_delta_'):
                    window = df_tf.iloc[start_idx : end_idx + 1]['volume_delta'].values.astype(np.float32)
                    if len(window) < lookback:
                        window = np.pad(window, (lookback - len(window), 0), 'constant')
                    raw_obs[key] = window

                elif key.startswith('ohlc'):
                    cols = ['open','high','low','close','volume'] if 'v' in key else ['open','high','low','close']
                    window = df_tf.iloc[start_idx : end_idx + 1][cols].values.astype(np.float32)

                    if len(window) < lookback:
                        padding = np.repeat(window[0:1], lookback - len(window), axis=0)
                        window = np.concatenate([padding, window], axis=0)
                    raw_obs[key] = window
            
            # --- 2. Get Causal Context Features ---
            raw_obs[FeatureKeys.CONTEXT.value] = self._get_current_context_features(step_index)

            # --- 3. NEW: Get Agent Portfolio State ---
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            portfolio_value = self.balance + unrealized_pnl
            
            # Normalized Position Size (-1 for full short, 1 for full long)
            current_notional = self.asset_held * current_price
            normalized_position = np.clip(current_notional / (portfolio_value + 1e-9), -1.0, 1.0)
            
            # Return on Used Margin
            pnl_on_margin = unrealized_pnl / (self.used_margin + 1e-9)
            normalized_pnl = np.tanh(pnl_on_margin) # Use tanh to squash extreme values

            # Margin Ratio (0 to 1+)
            margin_ratio = float('inf')
            position_notional = abs(self.asset_held) * current_price
            if position_notional > 0:
                margin_health = self.used_margin + unrealized_pnl
                margin_ratio = np.clip(margin_health / position_notional, 0, 2.0) # Clip for stability
            else:
                margin_ratio = 2.0 # No position, healthy margin

            raw_obs[FeatureKeys.PORTFOLIO_STATE.value] = np.array([
                normalized_position, normalized_pnl, margin_ratio
            ], dtype=np.float32)

            # Delegate normalization to the central Normalizer class
            normalized_obs = self.normalizer.transform(raw_obs)
            return normalized_obs

        except Exception as e:
            logger.error(f"Error getting observation for step {step_index}: {e}")
            # Return zero observations as fallback
            obs = {}
            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value
                if key == FeatureKeys.PORTFOLIO_STATE.value:
                    obs[key] = np.zeros(lookback, dtype=np.float32)
                elif key == 'context':
                    obs[key] = np.zeros(lookback, dtype=np.float32)
                elif key.startswith('ohlcv_'):
                    obs[key] = np.zeros((lookback, 5), dtype=np.float32)
                elif key.startswith('ohlc_'):
                    obs[key] = np.zeros((lookback, 4), dtype=np.float32)
                else:
                    obs[key] = np.zeros(lookback, dtype=np.float32)
            return obs

    def _get_observation_sequence(self):
        """Stack observations from the history deque."""
        try:
            obs_sequence = {
                key: np.stack([obs[key] for obs in self.observation_history])
                for key in self.observation_space.spaces.keys()
            }
            return obs_sequence

        except Exception as e:
            logger.error(f"Error getting observation sequence: {e}")
            # Return zero observations as fallback
            obs_sequence = {}
            for key in self.observation_space.spaces.keys():
                shape = self.observation_space.spaces[key].shape
                obs_sequence[key] = np.zeros(shape, dtype=np.float32)
            return obs_sequence

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        try:
            super().reset(seed=seed)

            # Initialize portfolio state
            self.balance = 1000000.0  # Total account equity
            self.asset_held = 0.0

            # State variables for isolated margin trading
            self.used_margin = 0.0
            self.entry_price = 0.0

            # Use dynamic warm-up period calculation
            warmup_period = self.cfg.get_required_warmup_period()
            self.current_step = max(warmup_period, self.strat_cfg.sequence_length)

            # Clear history
            self.observation_history.clear()
            self.portfolio_history.clear()
            self.episode_returns.clear()

            self.previous_portfolio_value = self.balance
            self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)

            # Build initial observation history
            for i in range(self.strat_cfg.sequence_length):
                step_idx = self.current_step - self.strat_cfg.sequence_length + 1 + i
                self.observation_history.append(self._get_single_step_observation(step_idx))

            observation = self._get_observation_sequence()
            info = {'balance': self.balance, 'asset_held': self.asset_held, 'portfolio_value': self.balance}

            return observation, info

        except Exception as e:
            logger.error(f"Error resetting environment: {e}")
            raise

    def step(self, action: np.ndarray):
        """Execute one environment step with enhanced margin simulation."""
        try:
            current_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step]

            # --- ISOLATED MARGIN & LIQUIDATION LOGIC ---
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            position_notional = abs(self.asset_held) * current_price

            # Calculate margin ratio
            margin_ratio = float('inf')
            if position_notional > 0:
                margin_health = self.used_margin + unrealized_pnl
                margin_ratio = margin_health / position_notional

            # Liquidation check
            if margin_ratio < self.strat_cfg.maintenance_margin_rate:
                liquidation_loss = self.used_margin
                self.balance -= liquidation_loss  # Equity is reduced by the lost margin
                reward = -100  # Severe penalty for liquidation
                terminated = True

                # Reset position state
                self.asset_held = 0.0
                self.used_margin = 0.0
                self.entry_price = 0.0

                # Get next observation and return
                self.current_step += 1
                truncated = self.current_step >= self.max_step
                self.observation_history.append(self._get_single_step_observation(self.current_step))
                observation = self._get_observation_sequence()

                info = {'portfolio_value': self.balance, 'margin_ratio': margin_ratio, 'liquidation': True}
                return observation, reward, terminated, truncated, info

            # --- TRADING LOGIC WITH LEVERAGE ---
            initial_portfolio_value = self.balance + unrealized_pnl

            # Determine target position based on action
            action_signal = np.clip(action[0], -1.0, 1.0)
            action_size = np.clip(action[1], 0.0, 1.0)

            target_exposure_pct = action_signal * action_size
            target_notional = initial_portfolio_value * target_exposure_pct
            target_asset_quantity = target_notional / current_price if current_price > 1e-8 else 0

            # --- NEW: PER-TRADE RISK MANAGEMENT (MARGIN ALLOCATION CAP) ---
            # Enforce a hard cap on the margin allocated to any single position, effectively limiting
            # the capital risked on one idea, as requested.
            max_allowable_margin = initial_portfolio_value * self.strat_cfg.max_margin_allocation_pct
            required_margin_for_target = (abs(target_asset_quantity) * current_price) / self.strat_cfg.leverage

            if required_margin_for_target > max_allowable_margin:
                # The desired position exceeds the risk limit; cap it to the max allowable margin.
                capped_notional = max_allowable_margin * self.strat_cfg.leverage
                capped_asset_quantity = (capped_notional / current_price) if current_price > 1e-8 else 0
                # Apply the cap while preserving the original trade direction (long/short).
                target_asset_quantity = capped_asset_quantity * np.sign(target_asset_quantity)


            # --- PRE-TRADE MARGIN CHECK ---
            # Calculate the margin required for the (potentially risk-capped) desired position
            required_margin = (abs(target_asset_quantity) * current_price) / self.strat_cfg.leverage

            # If required margin exceeds available equity, cap the position size
            if required_margin > initial_portfolio_value:
                max_affordable_notional = initial_portfolio_value * self.strat_cfg.leverage
                capped_asset_quantity = (max_affordable_notional / current_price) if current_price > 1e-8 else 0
                target_asset_quantity = capped_asset_quantity * np.sign(target_asset_quantity)

            # Calculate transaction details
            trade_quantity = target_asset_quantity - self.asset_held
            trade_notional = abs(trade_quantity) * current_price
            fee = trade_notional * self.cfg.transaction_fee_pct

            # Update portfolio state
            self.balance += unrealized_pnl - fee

            # Update position and margin
            self.asset_held = target_asset_quantity
            new_notional_value = abs(self.asset_held) * current_price
            self.used_margin = new_notional_value / self.strat_cfg.leverage
            self.entry_price = current_price if trade_quantity != 0 else self.entry_price


            # --- POST-TRADE CALCULATIONS ---
            self.current_step += 1
            truncated = self.current_step >= self.max_step

            next_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step]
            next_unrealized_pnl = self.asset_held * (next_price - self.entry_price)
            next_portfolio_value = self.balance + next_unrealized_pnl

            terminated = next_portfolio_value <= 1000

            # Enhanced reward function with risk adjustment
            reward = self._calculate_enhanced_reward(initial_portfolio_value, next_portfolio_value, action)

            self.previous_action = action

            # Update portfolio history for risk metrics
            self.portfolio_history.append(next_portfolio_value)
            if self.previous_portfolio_value is not None:
                period_return = (next_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
                self.episode_returns.append(period_return)

            self.previous_portfolio_value = next_portfolio_value
            self.observation_history.append(self._get_single_step_observation(self.current_step))

            observation = self._get_observation_sequence()

            info = {
                'balance': self.balance,
                'asset_held': self.asset_held,
                'portfolio_value': next_portfolio_value,
                'drawdown': self._calculate_current_drawdown(),
                'volatility': self._calculate_recent_volatility(),
                'unrealized_pnl': next_unrealized_pnl,
                'margin_ratio': margin_ratio,
                'used_margin': self.used_margin
            }

            return observation, reward, terminated, truncated, info

        except Exception as e:
            logger.error(f"Error in environment step: {e}")
            # Return safe defaults
            observation = self._get_observation_sequence()
            info = {'portfolio_value': self.balance, 'error': True}
            return observation, -10.0, True, False, info

    def _calculate_enhanced_reward(self, prev_value: float, curr_value: float, action: np.ndarray) -> float:
        """Calculate risk-adjusted reward using differential Sharpe ratio."""
        try:
            # Basic return
            period_return = (curr_value - prev_value) / prev_value

            # If we don't have enough history, return simple return
            if len(self.episode_returns) < 10:
                return period_return * 100  # Scale for better learning

            # Calculate differential Sharpe ratio
            returns_array = np.array(self.episode_returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array) + 1e-8
            sharpe_component = (period_return - mean_return) / std_return

            # Combine return and risk-adjusted components
            reward = period_return * 100 + sharpe_component * 10

            # Add small penalty for large position changes (transaction cost proxy)
            if self.previous_action is not None:
                position_change = abs(action[0] - self.previous_action[0])
                reward -= position_change * 0.1

            # Penalty for high drawdown periods, ONLY if the trade was a loss
            current_drawdown = self._calculate_current_drawdown()
            if current_drawdown > 0.1 and period_return <= 0:
                # Apply drawdown penalty only when losing during a drawdown
                reward -= current_drawdown * 50

            return reward

        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        try:
            if len(self.portfolio_history) < 2:
                return 0.0

            portfolio_values = list(self.portfolio_history)
            peak_value = max(portfolio_values)
            current_value = portfolio_values[-1]

            return (peak_value - current_value) / peak_value if peak_value > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return 0.0

    def _calculate_recent_volatility(self) -> float:
        """Calculate recent portfolio volatility."""
        try:
            if len(self.episode_returns) < 10:
                return 0.0

            recent_returns = self.episode_returns[-20:]  # Last 20 periods
            return np.std(recent_returns) * np.sqrt(252)  # Annualized

        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0

# --- CONVENIENCE FUNCTIONS ---

def create_bars_from_trades(period: str) -> pd.DataFrame:
    """Convenience function to create bars from trades."""
    try:
        from processor import EnhancedDataProcessor
        processor = EnhancedDataProcessor()
        return processor.create_enhanced_bars_from_trades(period)
    except Exception as e:
        logger.error(f"Error creating bars from trades: {e}")
        raise

def process_trades_for_period(period_name: str, force_reprocess: bool = False):
    """Convenience function to process trades for a period."""
    try:
        from processor import EnhancedDataProcessor
        processor = EnhancedDataProcessor()
        return processor.process_raw_trades_parallel(period_name, force_reprocess)
    except Exception as e:
        logger.error(f"Error processing trades for period {period_name}: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        logger.info("Testing trading environment...")

        # This would require actual data files and a fitted normalizer
        # from pathlib import Path
        # bars_df = create_bars_from_trades("in_sample")
        # normalizer = Normalizer(SETTINGS.strategy)
        # normalizer.load(Path(SETTINGS.get_normalizer_path()))
        # env = HierarchicalTradingEnvironment(bars_df, normalizer)
        # logger.info("✅ Environment test completed!")

        logger.info("✅ Engine module loaded successfully!")

    except Exception as e:
        logger.error(f"Engine test failed: {e}")