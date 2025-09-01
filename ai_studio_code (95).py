"""
Enhanced Trading Environment for Crypto Trading RL
Gymnasium-compliant environment with isolated margin simulation and enhanced features.
UPDATED: Implemented stateful, incremental feature calculation for massive performance gains
and correct multi-timeframe handling, replacing on-the-fly recalculations.
REFACTORED: Feature calculation logic is now fully decoupled and driven by config.
REWARD_REFACTOR: Reward function redesigned to use log returns, a continuous drawdown penalty,
and a dynamic liquidation penalty to promote more stable, risk-managed behavior.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm

# Import configuration and the new stateful feature calculators
from config import SETTINGS, FeatureKeys
from normalizer import Normalizer
from features import (
    StatefulBBWPercentRank,
    StatefulPriceDistanceMA,
    StatefulSRDistances,
)

# NEW: Map calculator names from config to their classes for dynamic instantiation
STATEFUL_CALCULATOR_MAP = {
    'StatefulBBWPercentRank': StatefulBBWPercentRank,
    'StatefulPriceDistanceMA': StatefulPriceDistanceMA,
    'StatefulSRDistances': StatefulSRDistances,
}

logger = logging.getLogger(__name__)

class HierarchicalTradingEnvironment(gym.Env):
    """
    Enhanced Gymnasium-compliant RL environment with:
    - Stateful, incremental feature calculation to prevent lookahead bias and boost performance.
    - Agent state awareness (portfolio state included in observation).
    - Isolated margin simulation and comprehensive risk management.
    - REVISED: Robust reward function for stable learning.
    """

    def __init__(self, df_base_ohlc: pd.DataFrame, normalizer: Normalizer, config=None, reward_weights: Optional[Dict] = None):
        super().__init__()

        self.cfg = config or SETTINGS
        self.strat_cfg = self.cfg.strategy
        self.normalizer = normalizer

        # --- NEW: Accept optimized reward weights ---
        if reward_weights:
            self.reward_weights = reward_weights
        else:
            # Provide sensible defaults if no weights are passed
            self.reward_weights = {
                'pnl_weight': 100.0,
                'drawdown_penalty_weight': 1.5,
                'churn_penalty_weight': 0.1
            }
        
        logger.info("--- Initializing Enhanced Hierarchical Trading Environment (with Stateful Features) ---")
        logger.info(f" -> Reward Weights: {self.reward_weights}")
        logger.info(f" -> Leverage: {self.strat_cfg.leverage}x | Maintenance Margin: {self.strat_cfg.maintenance_margin_rate:.2%}")

        try:
            base_df = df_base_ohlc.set_index('timestamp')
            # Get all unique timeframes required by stateful calculators and model inputs
            feature_timeframes = {calc.timeframe for calc in self.strat_cfg.stateful_calculators}
            model_timeframes = set(k.value.split('_')[-1].replace('m','T').replace('h','H').replace('s','S').upper() for k in self.strat_cfg.lookback_periods.keys())
            all_required_freqs = model_timeframes.union(feature_timeframes)

            self.timeframes = {}
            logger.info("Resampling data for all required timeframes...")
            for freq in all_required_freqs:
                if freq in ['CONTEXT', 'PORTFOLIO_STATE', 'PRECOMPUTED_FEATURES']: continue
                if freq not in self.timeframes:
                    # Added vwap to aggregation rules
                    agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'volume_delta': 'sum', 'vwap': 'last'}
                    valid_agg_rules = {k: v for k, v in agg_rules.items() if k in base_df.columns}
                    df_resampled = base_df.resample(freq).agg(valid_agg_rules).ffill()
                    self.timeframes[freq] = df_resampled.dropna()

            self.base_timestamps = self.timeframes[self.cfg.base_bar_timeframe.value].index
            self.max_step = len(self.base_timestamps) - 2

            self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

            obs_spaces = {}
            seq_len = self.strat_cfg.sequence_length
            for key, lookback in self.strat_cfg.lookback_periods.items():
                key_str = key.value
                if key_str.startswith('ohlcv_'): shape = (seq_len, lookback, 5)
                elif key_str.startswith('ohlc_'): shape = (seq_len, lookback, 4)
                elif key in [FeatureKeys.PORTFOLIO_STATE, FeatureKeys.CONTEXT, FeatureKeys.PRECOMPUTED_FEATURES]: shape = (seq_len, lookback)
                else: shape = (seq_len, lookback)
                obs_spaces[key_str] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

            self.observation_space = spaces.Dict(obs_spaces)
            self.observation_history = deque(maxlen=self.strat_cfg.sequence_length)

            self._initialize_stateful_features()

            self.portfolio_history = deque(maxlen=252)
            self.previous_portfolio_value = None
            self.episode_returns = []
            self.previous_action = None

            logger.info("Environment initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}", exc_info=True)
            raise

    # --- START OF REWARD REFACTOR ---
    def _calculate_enhanced_reward(self, prev_value: float, curr_value: float, action: np.ndarray) -> float:
        """
        Revised reward function focusing on log returns and risk management.
        This function uses weights that can be optimized as hyperparameters.
        """
        try:
            # 1. Core PnL Component (Logarithmic Return)
            # This tames extreme outliers and treats gains/losses more symmetrically.
            log_return = np.log(curr_value / (prev_value + 1e-9))
            pnl_reward = log_return * self.reward_weights['pnl_weight']

            # 2. Drawdown Penalty (Continuous)
            # Applies a small, continuous penalty for every step the agent is below its peak portfolio value.
            current_drawdown = self._calculate_current_drawdown()
            drawdown_penalty = -current_drawdown * self.reward_weights['drawdown_penalty_weight']

            # 3. Action Churning Penalty
            # Penalizes excessive changes in position to encourage smoother trading.
            churn_penalty = 0.0
            if self.previous_action is not None:
                position_change = abs(action[0] - self.previous_action[0])
                churn_penalty = -position_change * self.reward_weights['churn_penalty_weight']

            # 4. Combine all components
            reward = pnl_reward + drawdown_penalty + churn_penalty
            return reward

        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0
    # --- END OF REWARD REFACTOR ---

    def step(self, action: np.ndarray):
        try:
            self._update_stateful_features(self.current_step)

            current_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step]

            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            position_notional = abs(self.asset_held) * current_price
            margin_ratio = (self.used_margin + unrealized_pnl) / position_notional if position_notional > 0 else float('inf')

            # --- START OF DYNAMIC LIQUIDATION PENALTY ---
            if margin_ratio < self.strat_cfg.maintenance_margin_rate:
                # Penalty is now proportional to the margin that was at risk, plus a large fixed component.
                # This makes the penalty for liquidating a large position much more severe.
                liquidation_penalty = -50 - (self.used_margin * 0.2)
                
                self.balance -= self.used_margin
                reward = liquidation_penalty # Use the new dynamic penalty
                terminated = True
                self.asset_held = 0.0
                self.used_margin = 0.0
                self.entry_price = 0.0
                self.current_step += 1
                truncated = self.current_step >= self.max_step
                self.observation_history.append(self._get_single_step_observation(self.current_step))
                observation = self._get_observation_sequence()
                info = {'portfolio_value': self.balance, 'margin_ratio': margin_ratio, 'liquidation': True}
                return observation, reward, terminated, truncated, info
            # --- END OF DYNAMIC LIQUIDATION PENALTY ---

            initial_portfolio_value = self.balance + unrealized_pnl
            action_signal = np.clip(action[0], -1.0, 1.0)
            action_size = np.clip(action[1], 0.0, 1.0)
            target_notional = initial_portfolio_value * action_signal * action_size
            target_asset_quantity = target_notional / current_price if current_price > 1e-8 else 0

            max_allowable_margin = initial_portfolio_value * self.strat_cfg.max_margin_allocation_pct
            required_margin_for_target = (abs(target_asset_quantity) * current_price) / self.strat_cfg.leverage
            if required_margin_for_target > max_allowable_margin:
                capped_notional = max_allowable_margin * self.strat_cfg.leverage
                target_asset_quantity = (capped_notional / current_price * np.sign(target_asset_quantity)) if current_price > 1e-8 else 0

            required_margin = (abs(target_asset_quantity) * current_price) / self.strat_cfg.leverage
            if required_margin > initial_portfolio_value:
                max_affordable_notional = initial_portfolio_value * self.strat_cfg.leverage
                target_asset_quantity = (max_affordable_notional / current_price * np.sign(target_asset_quantity)) if current_price > 1e-8 else 0

            trade_quantity = target_asset_quantity - self.asset_held
            trade_notional = abs(trade_quantity) * current_price
            fee = trade_notional * self.cfg.transaction_fee_pct

            self.balance += unrealized_pnl - fee
            self.asset_held = target_asset_quantity
            new_notional_value = abs(self.asset_held) * current_price
            self.used_margin = new_notional_value / self.strat_cfg.leverage
            if trade_quantity != 0: self.entry_price = current_price

            self.current_step += 1
            truncated = self.current_step >= self.max_step
            next_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step]
            next_unrealized_pnl = self.asset_held * (next_price - self.entry_price)
            next_portfolio_value = self.balance + next_unrealized_pnl
            terminated = next_portfolio_value <= 1000

            # --- UPDATED REWARD CALL ---
            reward = self._calculate_enhanced_reward(initial_portfolio_value, next_portfolio_value, action)
            
            self.previous_action = action
            self.portfolio_history.append(next_portfolio_value)
            if self.previous_portfolio_value is not None:
                self.episode_returns.append((next_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value)

            self.previous_portfolio_value = next_portfolio_value
            self.observation_history.append(self._get_single_step_observation(self.current_step))
            observation = self._get_observation_sequence()

            info = {
                'balance': self.balance, 'asset_held': self.asset_held, 'portfolio_value': next_portfolio_value,
                'drawdown': self._calculate_current_drawdown(), 'volatility': self._calculate_recent_volatility(),
                'unrealized_pnl': next_unrealized_pnl, 'margin_ratio': margin_ratio, 'used_margin': self.used_margin
            }
            return observation, reward, terminated, truncated, info
        except Exception as e:
            logger.error(f"Error in environment step: {e}", exc_info=True)
            observation = self._get_observation_sequence()
            info = {'portfolio_value': self.balance, 'error': True}
            return observation, -10.0, True, False, info
            
    # --- Other methods remain unchanged ---

    def _initialize_stateful_features(self):
        """Create instances of all stateful calculators and their history deques based on config."""
        logger.info("Initializing stateful feature calculators from config...")
        self.feature_calculators: Dict[str, Any] = {}
        self.feature_histories: Dict[str, deque] = {}
        self.last_update_timestamps: Dict[str, pd.Timestamp] = {}

        for calc_cfg in self.strat_cfg.stateful_calculators:
            if calc_cfg.class_name not in STATEFUL_CALCULATOR_MAP:
                raise ValueError(f"Unknown stateful calculator class: {calc_cfg.class_name}")
            calculator_class = STATEFUL_CALCULATOR_MAP[calc_cfg.class_name]
            self.feature_calculators[calc_cfg.name] = calculator_class(**calc_cfg.params)
            self.last_update_timestamps[calc_cfg.timeframe] = pd.Timestamp(0, tz='UTC')

        for key in self.strat_cfg.context_feature_keys:
            self.feature_histories[key] = deque(maxlen=self.cfg.get_required_warmup_period() + 200)

    def _warmup_features(self, warmup_steps: int):
        logger.info(f"Warming up stateful features for {warmup_steps} steps...")
        for i in tqdm(range(warmup_steps), desc="Warming up features", leave=False, ncols=100):
            self._update_stateful_features(i)

    def _update_stateful_features(self, step_index: int):
        current_timestamp = self.base_timestamps[step_index]
        for calc_cfg in self.strat_cfg.stateful_calculators:
            timeframe = calc_cfg.timeframe
            df_tf = self.timeframes[timeframe]
            try:
                latest_bar_idx = df_tf.index.get_loc(current_timestamp, method='ffill')
                latest_bar_timestamp = df_tf.index[latest_bar_idx]
            except KeyError: continue
            if latest_bar_timestamp > self.last_update_timestamps[timeframe]:
                self.last_update_timestamps[timeframe] = latest_bar_timestamp
                new_data_point = df_tf[calc_cfg.source_col].iloc[latest_bar_idx]
                self.feature_calculators[calc_cfg.name].update(new_data_point)
        for calc_cfg in self.strat_cfg.stateful_calculators:
            calculator = self.feature_calculators[calc_cfg.name]
            values = calculator.get()
            if isinstance(values, dict):
                for key in calc_cfg.output_keys:
                    default_val = 1.0 if 'dist' in key else 0.0
                    self.feature_histories[key].append(values.get(key, default_val))
            else:
                if len(calc_cfg.output_keys) == 1:
                    key = calc_cfg.output_keys[0]
                    self.feature_histories[key].append(values)
                else:
                    logger.warning(f"Mismatch: Calculator '{calc_cfg.name}' returned a single value but has {len(calc_cfg.output_keys)} output keys defined.")

    def _get_current_context_features(self) -> np.ndarray:
        final_vector = [self.feature_histories[key][-1] if self.feature_histories[key] else 0.0
                        for key in self.strat_cfg.context_feature_keys]
        return np.array(final_vector, dtype=np.float32)

    def reset(self, seed=None, options=None):
        try:
            super().reset(seed=seed)
            self.balance = 1000000.0
            self.asset_held = 0.0
            self.used_margin = 0.0
            self.entry_price = 0.0
            warmup_period = self.cfg.get_required_warmup_period()
            self._initialize_stateful_features()
            self._warmup_features(warmup_period)
            self.current_step = warmup_period
            self.observation_history.clear()
            self.portfolio_history.clear()
            self.episode_returns.clear()
            self.previous_portfolio_value = self.balance
            self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
            for i in range(self.strat_cfg.sequence_length):
                step_idx = self.current_step - self.strat_cfg.sequence_length + 1 + i
                self._update_stateful_features(step_idx)
                self.observation_history.append(self._get_single_step_observation(step_idx))
            observation = self._get_observation_sequence()
            info = {'balance': self.balance, 'asset_held': self.asset_held, 'portfolio_value': self.balance}
            return observation, info
        except Exception as e:
            logger.error(f"Error resetting environment: {e}", exc_info=True)
            raise

    def _get_single_step_observation(self, step_index) -> dict:
        try:
            if self.normalizer is None: return {}
            current_timestamp = self.base_timestamps[step_index]
            raw_obs = {}
            base_df = self.timeframes[self.cfg.base_bar_timeframe.value]
            current_price = base_df['close'].iloc[base_df.index.get_loc(current_timestamp, method='ffill')]
            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value
                if key in [FeatureKeys.CONTEXT.value, FeatureKeys.PORTFOLIO_STATE.value, FeatureKeys.PRECOMPUTED_FEATURES.value]: continue
                freq = key.split('_')[-1].replace('m','T').replace('h','H').replace('s', 'S').upper()
                df_tf = self.timeframes[freq]
                end_idx = df_tf.index.get_loc(current_timestamp, method='ffill')
                start_idx = max(0, end_idx - lookback + 1)
                if key.startswith('price_'):
                    window = df_tf.iloc[start_idx : end_idx + 1]['close'].values.astype(np.float32)
                    if len(window) < lookback: window = np.pad(window, (lookback - len(window), 0), 'edge')
                    raw_obs[key] = window
                elif key.startswith('volume_delta_'):
                    window = df_tf.iloc[start_idx : end_idx + 1]['volume_delta'].values.astype(np.float32)
                    if len(window) < lookback: window = np.pad(window, (lookback - len(window), 0), 'constant')
                    raw_obs[key] = window
                elif key.startswith('ohlc'):
                    cols = ['open','high','low','close','volume'] if 'v' in key else ['open','high','low','close']
                    window = df_tf.iloc[start_idx : end_idx + 1][cols].values.astype(np.float32)
                    if len(window) < lookback:
                        padding = np.repeat(window[0:1], lookback - len(window), axis=0)
                        window = np.concatenate([padding, window], axis=0)
                    raw_obs[key] = window
            raw_obs[FeatureKeys.CONTEXT.value] = self._get_current_context_features()
            current_bar_features = base_df.loc[base_df.index.get_loc(current_timestamp, method='ffill')]
            precomputed_vector = current_bar_features[self.strat_cfg.precomputed_feature_keys].fillna(0.0).values.astype(np.float32)
            raw_obs[FeatureKeys.PRECOMPUTED_FEATURES.value] = precomputed_vector
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            portfolio_value = self.balance + unrealized_pnl
            current_notional = self.asset_held * current_price
            normalized_position = np.clip(current_notional / (portfolio_value + 1e-9), -1.0, 1.0)
            pnl_on_margin = unrealized_pnl / (self.used_margin + 1e-9)
            normalized_pnl = np.tanh(pnl_on_margin)
            position_notional = abs(self.asset_held) * current_price
            margin_health = self.used_margin + unrealized_pnl
            margin_ratio = np.clip(margin_health / position_notional, 0, 2.0) if position_notional > 0 else 2.0
            raw_obs[FeatureKeys.PORTFOLIO_STATE.value] = np.array([normalized_position, normalized_pnl, margin_ratio], dtype=np.float32)
            return self.normalizer.transform(raw_obs)
        except Exception as e:
            logger.error(f"Error getting observation for step {step_index}: {e}", exc_info=True)
            obs = {}
            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value
                if key in [FeatureKeys.PORTFOLIO_STATE.value, FeatureKeys.CONTEXT.value, FeatureKeys.PRECOMPUTED_FEATURES.value]:
                    obs[key] = np.zeros(lookback, dtype=np.float32)
                elif key.startswith('ohlcv_'): obs[key] = np.zeros((lookback, 5), dtype=np.float32)
                elif key.startswith('ohlc_'): obs[key] = np.zeros((lookback, 4), dtype=np.float32)
                else: obs[key] = np.zeros(lookback, dtype=np.float32)
            return obs

    def _get_observation_sequence(self):
        try:
            return {key: np.stack([obs[key] for obs in self.observation_history]) for key in self.observation_space.spaces.keys()}
        except Exception as e:
            logger.error(f"Error getting observation sequence: {e}")
            return {key: np.zeros(self.observation_space.spaces[key].shape, dtype=np.float32) for key in self.observation_space.spaces.keys()}

    def _calculate_current_drawdown(self) -> float:
        if len(self.portfolio_history) < 2: return 0.0
        portfolio_values = list(self.portfolio_history)
        peak_value = max(portfolio_values)
        return (peak_value - portfolio_values[-1]) / peak_value if peak_value > 0 else 0.0

    def _calculate_recent_volatility(self) -> float:
        if len(self.episode_returns) < 10: return 0.0
        return np.std(self.episode_returns[-20:]) * np.sqrt(252)

# --- CONVENIENCE FUNCTIONS ---
def create_bars_from_trades(period: str) -> pd.DataFrame:
    try:
        from processor import EnhancedDataProcessor
        processor = EnhancedDataProcessor()
        return processor.create_enhanced_bars_from_trades(period)
    except Exception as e:
        logger.error(f"Error creating bars from trades: {e}")
        raise

def process_trades_for_period(period_name: str, force_reprocess: bool = False):
    try:
        from processor import EnhancedDataProcessor
        processor = EnhancedDataProcessor()
        return processor.process_raw_trades_parallel(period_name, force_reprocess)
    except Exception as e:
        logger.error(f"Error processing trades for period {period_name}: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Testing trading environment...")