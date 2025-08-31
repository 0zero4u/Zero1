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
from config import SETTINGS

logger = logging.getLogger(__name__)

class HierarchicalTradingEnvironment(gym.Env):
    """
    Enhanced Gymnasium-compliant RL environment with:
    - Enhanced reward function
    - Better normalization
    - Dynamic warm-up period calculation
    - Isolated margin simulation
    - Comprehensive risk management
    """

    def __init__(self, df_base_ohlc: pd.DataFrame):
        super().__init__()

        self.cfg = SETTINGS
        self.strat_cfg = self.cfg.strategy

        logger.info("--- Initializing Enhanced Hierarchical Trading Environment ---")
        logger.info(f" -> Leverage: {self.strat_cfg.leverage}x | Maintenance Margin: {self.strat_cfg.maintenance_margin_rate:.2%}")

        try:
            # Prepare base data
            base_df = df_base_ohlc.set_index('timestamp')

            # Get required timeframes
            model_timeframes = set(self.strat_cfg.lookback_periods.keys())
            feature_timeframes = {'1H', '4H', '3T', '15T'}
            all_required_keys = model_timeframes.union(feature_timeframes)

            self.timeframes = {}
            logger.info("Resampling data for all required timeframes...")

            for key in all_required_keys:
                if key == 'context':
                    continue

                # Handle both formats: 'price_3m' and '3T'
                freq_str = key.split('_')[-1] if '_' in key else key
                freq = freq_str.replace('m','T').replace('h','H')

                if freq not in self.timeframes:
                    agg_rules = {
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                    }

                    df_resampled = base_df.resample(freq).agg(agg_rules)
                    df_resampled = df_resampled.ffill()  # Fixed deprecated method
                    self.timeframes[freq] = df_resampled.dropna()

            self.base_timestamps = self.timeframes[self.cfg.base_bar_timeframe].index

            # Precompute market features
            self._precompute_market_features()

            self.max_step = len(self.base_timestamps) - 2

            # Action space: [position_signal, position_size]
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32
            )

            # Define observation space
            obs_spaces = {}
            seq_len = self.strat_cfg.sequence_length

            for key, lookback in self.strat_cfg.lookback_periods.items():
                if key.startswith('ohlcv_'):
                    shape = (seq_len, lookback, 5)  # O,H,L,C,V
                elif key.startswith('ohlc_'):
                    shape = (seq_len, lookback, 4)  # O,H,L,C
                elif key.startswith('price_'):
                    shape = (seq_len, lookback)
                else:  # context
                    shape = (seq_len, lookback)

                obs_spaces[key] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

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

    def _precompute_market_features(self):
        """Calculate context features including multi-timeframe support/resistance levels."""
        logger.info("Pre-computing market context features...")

        try:
            # Volatility Feature (1H timeframe)
            df_1h = self.timeframes['1H'].copy()
            rolling_mean = df_1h['close'].rolling(20).mean()
            rolling_std = df_1h['close'].rolling(20).std()
            df_1h['bbw_1h_pct'] = ((4 * rolling_std) / rolling_mean).rolling(250).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
            )

            # Trend Feature (4H timeframe)
            df_4h = self.timeframes['4H'].copy()
            df_4h['price_dist_ma_4h'] = (df_4h['close'] / df_4h['close'].rolling(50).mean()) - 1.0

            # Merge features with base timeframe
            base_df_with_features = self.timeframes[self.cfg.base_bar_timeframe].copy()
            base_df_with_features = pd.merge_asof(
                base_df_with_features, df_1h[['bbw_1h_pct']], left_index=True, right_index=True
            )
            base_df_with_features = pd.merge_asof(
                base_df_with_features, df_4h[['price_dist_ma_4h']], left_index=True, right_index=True
            )

            # Implement multi-timeframe S/R feature calculation
            sr_plan = {
                '3T': {'levels': 1, 'suffix': '3m'},
                '15T': {'levels': 2, 'suffix': '15m'},
                '1H': {'levels': 2, 'suffix': '1h'}
            }

            sr_window = self.strat_cfg.support_resistance_window

            for freq, plan in sr_plan.items():
                logger.info(f" -> Calculating S/R for {freq} timeframe...")
                df_tf = self.timeframes[freq]
                sr_distances = self._calculate_multi_level_sr_distances(df_tf['close'], sr_window, plan['levels'])

                # Rename columns with suffix and create a DataFrame
                renamed_sr = {f"{key}_{plan['suffix']}": value for key, value in sr_distances.items()}
                sr_df = pd.DataFrame(renamed_sr)

                # Merge into base features
                base_df_with_features = pd.merge_asof(base_df_with_features, sr_df, left_index=True, right_index=True)

            # REMOVED: Funding rate integration as it's no longer available

            # Forward fill missing values
            base_df_with_features = base_df_with_features.ffill()

            # UPDATED: Define the final list of 12 context features (removed funding_rate)
            feature_cols = [
                'bbw_1h_pct', 'price_dist_ma_4h',
                'dist_s1_3m', 'dist_r1_3m',
                'dist_s1_15m', 'dist_r1_15m', 'dist_s2_15m', 'dist_r2_15m',
                'dist_s1_1h', 'dist_r1_1h', 'dist_s2_1h', 'dist_r2_1h'
            ]

            self.features_df = base_df_with_features[feature_cols].dropna()
            logger.info("Market context features ready.")

        except Exception as e:
            logger.error(f"Error precomputing market features: {e}")
            # Create dummy features as fallback
            dummy_features = pd.DataFrame(
                np.zeros((len(self.base_timestamps), 12)),
                index=self.base_timestamps,
                columns=[f'feature_{i}' for i in range(12)]
            )
            self.features_df = dummy_features

    def _calculate_multi_level_sr_distances(self, price_series: pd.Series, window: int, num_levels: int) -> Dict[str, pd.Series]:
        """Calculate distance to multiple support/resistance levels."""
        try:
            results = {f'dist_s{i+1}': [] for i in range(num_levels)}
            results.update({f'dist_r{i+1}': [] for i in range(num_levels)})

            price_values = price_series.values

            for i in range(len(price_series)):
                if i < window:
                    for level in range(num_levels):
                        results[f'dist_s{level+1}'].append(0.0)
                        results[f'dist_r{level+1}'].append(0.0)
                    continue

                window_vals = price_values[max(0, i - window):i]
                current_price = price_values[i]

                if len(window_vals) < 10:  # Not enough data for reliable peaks
                    for level in range(num_levels):
                        results[f'dist_s{level+1}'].append(0.0)
                        results[f'dist_r{level+1}'].append(0.0)
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
                        results[f'dist_s{level+1}'].append(0.0)

                # Resistance Levels (above current price)
                resistance_levels = sorted([p for p in window_vals[peaks] if p > current_price])
                for level in range(num_levels):
                    if level < len(resistance_levels):
                        dist = (resistance_levels[level] - current_price) / current_price
                        results[f'dist_r{level+1}'].append(dist)
                    else:
                        results[f'dist_r{level+1}'].append(0.0)

            return {key: pd.Series(value, index=price_series.index) for key, value in results.items()}

        except Exception as e:
            logger.error(f"Error calculating S/R distances: {e}")
            # Return zero distances as fallback
            return {f'dist_s{i+1}': pd.Series(np.zeros(len(price_series)), index=price_series.index) 
                   for i in range(num_levels)} |                    {f'dist_r{i+1}': pd.Series(np.zeros(len(price_series)), index=price_series.index) 
                   for i in range(num_levels)}

    def _get_single_step_observation(self, step_index) -> dict:
        """Construct the observation dictionary for a single point in time."""
        try:
            current_timestamp = self.base_timestamps[step_index]
            obs = {}

            for key, lookback in self.strat_cfg.lookback_periods.items():
                if key == 'context':
                    try:
                        obs[key] = self.features_df.loc[current_timestamp].values.astype(np.float32)
                    except KeyError:
                        obs[key] = np.zeros(lookback, dtype=np.float32)
                    continue

                freq = key.split('_')[-1].replace('m','T').replace('h','H')
                df_tf = self.timeframes[freq]

                end_idx = df_tf.index.get_loc(current_timestamp, method='ffill')
                start_idx = max(0, end_idx - lookback + 1)

                if key.startswith('price_'):
                    window = df_tf.iloc[start_idx : end_idx + 1]['close'].values.astype(np.float32)
                    if len(window) < lookback:
                        window = np.pad(window, (lookback - len(window), 0), 'edge')

                    last_price = window[-1]
                    obs[key] = (window / last_price) - 1.0 if last_price > 1e-6 else np.zeros_like(window)

                elif key.startswith('ohlc'):
                    cols = ['open','high','low','close','volume'] if 'v' in key else ['open','high','low','close']
                    window = df_tf.iloc[start_idx : end_idx + 1][cols].values.astype(np.float32)

                    if len(window) < lookback:
                        padding = np.repeat(window[0:1], lookback - len(window), axis=0)
                        window = np.concatenate([padding, window], axis=0)

                    last_price = window[-1, 3]  # close price

                    # Normalize all features including volume
                    if window.shape[1] == 5:  # OHLCV data
                        # Normalize OHLC by price
                        window[:, :4] = (window[:, :4] / last_price) - 1.0 if last_price > 1e-6 else 0
                        # Normalize volume by recent average volume
                        recent_volume_avg = np.mean(window[-min(20, len(window)):, 4]) + 1e-8
                        window[:, 4] = np.log1p(window[:, 4]) / np.log1p(recent_volume_avg)
                    else:  # OHLC data
                        window = (window / last_price) - 1.0 if last_price > 1e-6 else np.zeros_like(window)

                    obs[key] = window

            return obs

        except Exception as e:
            logger.error(f"Error getting observation for step {step_index}: {e}")
            # Return zero observations as fallback
            obs = {}
            for key, lookback in self.strat_cfg.lookback_periods.items():
                if key == 'context':
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
            self.balance = 10000.0  # Total account equity
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
            current_price = self.timeframes[self.cfg.base_bar_timeframe]['close'].iloc[self.current_step]

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

            # --- PRE-TRADE MARGIN CHECK ---
            # Calculate the margin required for the desired position
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
            self.entry_price = current_price

            # --- POST-TRADE CALCULATIONS ---
            self.current_step += 1
            truncated = self.current_step >= self.max_step

            next_price = self.timeframes[self.cfg.base_bar_timeframe]['close'].iloc[self.current_step]
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
        processor = EnhancedDataProcessor()
        return processor.create_enhanced_bars_from_trades(period)
    except Exception as e:
        logger.error(f"Error creating bars from trades: {e}")
        raise

def process_trades_for_period(period_name: str, force_reprocess: bool = False):
    """Convenience function to process trades for a period."""
    try:
        processor = EnhancedDataProcessor()
        return processor.process_raw_trades_parallel(period_name, force_reprocess)
    except Exception as e:
        logger.error(f"Error processing trades for period {period_name}: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        logger.info("Testing trading environment...")

        # This would require actual data files
        # bars_df = create_bars_from_trades("in_sample")
        # env = HierarchicalTradingEnvironment(bars_df)
        # logger.info("✅ Environment test completed!")

        logger.info("✅ Engine module loaded successfully!")

    except Exception as e:
        logger.error(f"Engine test failed: {e}")