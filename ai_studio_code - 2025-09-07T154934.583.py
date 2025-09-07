# FIXED: engine.py with Centralized Config, Immediate Rewards, and NumPy Optimizations

"""
FIXED: Enhanced Trading Environment with Immediate Reward System

CRITICAL FIXES APPLIED:
1. Reads default reward weights from the centralized config.py, establishing a single source of truth.
2. Implemented PPO-compatible immediate reward calculation at every step.
3. Added a sophisticated, progressive inactivity penalty to combat agent paralysis.

PERFORMANCE OPTIMIZATIONS:
1. Converted all critical pandas DataFrames to NumPy arrays during initialization.
2. Replaced slow pandas lookups in the hot loop with direct, high-speed NumPy array indexing.
3. This results in a significant increase in training FPS (Frames Per Second).
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Dict, List, Any, Optional, Tuple
import logging
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

from config import SETTINGS, FeatureKeys
from normalizer import Normalizer
from features import (StatefulBBWPercentRank, StatefulPriceDistanceMA, StatefulSRDistances, StatefulVWAPDistance)

STATEFUL_CALCULATOR_MAP = {
    'StatefulBBWPercentRank': StatefulBBWPercentRank, 'StatefulPriceDistanceMA': StatefulPriceDistanceMA,
    'StatefulSRDistances': StatefulSRDistances, 'StatefulVWAPDistance': StatefulVWAPDistance,
}

logger = logging.getLogger(__name__)

class EnhancedRiskManager:
    def __init__(self, config, leverage: float = 10.0):
        self.cfg = config
        self.leverage = leverage
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
    
    def calculate_dynamic_position_limit(self, volatility: float, market_regime: str) -> float:
        base_limit = self.cfg.strategy.max_position_size
        vol_adjustment = max(0.5, min(2.0, 1.0 / (volatility * 10 + 0.1)))
        regime_multipliers = {"HIGH_VOLATILITY": 0.6, "TRENDING_UP": 1.2, "TRENDING_DOWN": 0.8,
                              "LOW_VOLATILITY": 1.1, "SIDEWAYS": 0.9, "UNCERTAIN": 0.7}
        regime_adjustment = regime_multipliers.get(market_regime, 0.8)
        leverage_adjustment = min(1.0, 10.0 / self.leverage)
        return base_limit * vol_adjustment * regime_adjustment * leverage_adjustment

class FixedRewardCalculator:
    def __init__(self, config, leverage: float = 10.0, reward_weights: Optional[Dict[str, float]] = None):
        self.cfg = config
        self.leverage = leverage
        self.return_buffer = deque(maxlen=500)
        
        # --- CENTRALIZED CONFIG FIX ---
        # The primary source of weights is now the config.
        # The 'reward_weights' param is now ONLY for Optuna overrides.
        default_weights_from_config = self.cfg.strategy.reward_weights.dict()
        self.weights = reward_weights or default_weights_from_config
        
        self.scaling_factor = self.cfg.strategy.reward_scaling_factor / self.leverage
        self.grace_period_steps = 90
        self.penalty_ramp_up_steps = 360
        logger.info(f"FIXED reward calculator initialized with weights: {self.weights}")

    def calculate_immediate_reward(self, prev_value: float, curr_value: float, action: np.ndarray,
                                 portfolio_state: Dict, market_state: Dict, previous_action: np.ndarray,
                                 consecutive_inaction_steps: int) -> Tuple[float, Dict]:
        try:
            components = {}
            period_return = (curr_value - prev_value) / max(prev_value, 1e-6)
            self.return_buffer.append(period_return)
            components['base_return'] = np.tanh(period_return * self.scaling_factor) * self.weights['base_return']
            
            if len(self.return_buffer) >= 30:
                returns_array = np.array(list(self.return_buffer))[-30:]
                mean_return, volatility = np.mean(returns_array), np.std(returns_array) + 1e-8
                risk_adjusted_score = np.clip(np.tanh((mean_return / volatility) * 3), -1.0, 1.0)
                components['risk_adjusted'] = risk_adjusted_score * self.weights['risk_adjusted']
            
            position_change = abs(action[0] - previous_action[0])
            if position_change > 0.01:
                components['exploration_bonus'] = self.weights.get('exploration_bonus', 0.0)
                position_size_factor = (action[1] + previous_action[1]) / 2
                tx_penalty = position_change * position_size_factor * self.cfg.transaction_fee_pct
                components['transaction_penalty'] = tx_penalty * self.weights['transaction_penalty']
            
            current_drawdown = portfolio_state.get('drawdown', 0.0)
            if current_drawdown > 0.05:
                drawdown_severity = min(1.0, (current_drawdown - 0.05) / 0.15)
                components['drawdown_penalty'] = (drawdown_severity ** 2) * self.weights['drawdown_penalty']
            
            margin_ratio = portfolio_state.get('margin_ratio', 2.0)
            if margin_ratio > 0.5:
                components['risk_bonus'] = min(0.2, (margin_ratio - 0.5) * 0.4) * self.weights['risk_bonus']

            max_inactivity_penalty = self.weights.get('inactivity_penalty', 0.0)
            if max_inactivity_penalty < 0 and consecutive_inaction_steps > self.grace_period_steps:
                steps_into_penalty = consecutive_inaction_steps - self.grace_period_steps
                penalty_factor = min(1.0, steps_into_penalty / self.penalty_ramp_up_steps)
                components['inactivity_penalty'] = max_inactivity_penalty * penalty_factor
            
            if abs(action[1]) < 0.01 and components.get('risk_bonus', 0) > 0:
                components['risk_bonus'] = 0.0

            return np.clip(sum(components.values()), -5.0, 5.0), components
        except Exception as e:
            logger.error(f"Error in immediate reward calculation: {e}", exc_info=True)
            return 0.0, {'error': -0.1}

class FixedHierarchicalTradingEnvironment(gym.Env):
    def __init__(self, df_base_ohlc: pd.DataFrame, normalizer: Normalizer, config=None,
                 leverage: float = None, reward_weights: Optional[Dict[str, float]] = None,
                 precomputed_features: Optional[pd.DataFrame] = None):
        super().__init__()
        
        self.cfg = config or SETTINGS
        self.strat_cfg = self.cfg.strategy
        self.normalizer = normalizer
        self.leverage = leverage if leverage is not None else self.strat_cfg.leverage
        self.precomputed_features_df = precomputed_features
        
        logger.info("--- Initializing FIXED Immediate Reward Trading Environment (NumPy OPTIMIZED) ---")
        
        self.risk_manager = EnhancedRiskManager(self.cfg, leverage=self.leverage)
        self.reward_calculator = FixedRewardCalculator(self.cfg, leverage=self.leverage, reward_weights=reward_weights)
        
        base_df = df_base_ohlc.set_index('timestamp')
        all_freqs = {calc.timeframe for calc in self.strat_cfg.stateful_calculators}.union(
            set(k.value.split('_')[-1].replace('m','T').replace('h','H').replace('s','S').upper()
                for k in self.strat_cfg.lookback_periods if k not in {FeatureKeys.CONTEXT, FeatureKeys.PORTFOLIO_STATE, FeatureKeys.PRECOMPUTED_FEATURES})
        )
        self.timeframes = {}
        for freq in all_freqs:
            agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            valid_agg_rules = {k: v for k, v in agg_rules.items() if k in base_df.columns}
            self.timeframes[freq] = base_df.resample(freq).agg(valid_agg_rules).ffill().dropna()
        
        self.base_timestamps = self.timeframes[self.cfg.base_bar_timeframe.value].index
        self.max_step = len(self.base_timestamps) - 2
        
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        obs_spaces = {}
        seq_len = self.strat_cfg.sequence_length
        for key, lookback in self.strat_cfg.lookback_periods.items():
            key_str = key.value
            shape = ((seq_len, lookback, 5) if 'ohlcv' in key_str else
                     (seq_len, lookback, 4) if 'ohlc' in key_str else
                     (seq_len, lookback))
            obs_spaces[key_str] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)
        
        self._initialize_stateful_features()
        self._initialize_numpy_cache()
        self.reset_state_variables()
        logger.info("✅ FIXED environment initialized (NumPy OPTIMIZED).")

    def _initialize_numpy_cache(self):
        logger.info("⚡ Caching data to NumPy arrays for maximum performance...")
        base_df = self.timeframes[self.cfg.base_bar_timeframe.value]
        self.close_prices_np = base_df['close'].to_numpy(dtype=np.float32)
        self.base_timestamps_np = base_df.index.to_numpy()
        self.timeframe_data_np = {}
        for freq, df in self.timeframes.items():
            data = {'index': df.index.to_numpy()}
            if 'close' in df.columns: data['close'] = df['close'].to_numpy(dtype=np.float32)
            ohlc_cols = ['open', 'high', 'low', 'close']
            if all(c in df.columns for c in ohlc_cols):
                data['ohlc'] = df[ohlc_cols].to_numpy(dtype=np.float32)
                if 'volume' in df.columns:
                    data['ohlcv'] = df[ohlc_cols + ['volume']].to_numpy(dtype=np.float32)
            self.timeframe_data_np[freq] = data
        self.precomputed_features_np = base_df[self.strat_cfg.precomputed_feature_keys].fillna(0.0).to_numpy(dtype=np.float32)

    def _initialize_stateful_features(self):
        self.feature_calculators, self.feature_histories, self.last_update_timestamps = {}, {}, {}
        for calc_cfg in self.strat_cfg.stateful_calculators:
            self.feature_calculators[calc_cfg.name] = STATEFUL_CALCULATOR_MAP[calc_cfg.class_name](**calc_cfg.params)
            self.last_update_timestamps[calc_cfg.timeframe] = pd.Timestamp(0, tz='UTC')
        for key in self.strat_cfg.context_feature_keys:
            self.feature_histories[key] = deque(maxlen=self.cfg.get_required_warmup_period() + 200)

    def _update_stateful_features(self, step_index: int):
        current_timestamp = self.base_timestamps[step_index]
        for calc_cfg in self.strat_cfg.stateful_calculators:
            df_tf = self.timeframes[calc_cfg.timeframe]
            try:
                idx = df_tf.index.get_indexer([current_timestamp], method='ffill')[0]
                ts = df_tf.index[idx]
            except (KeyError, IndexError): continue
            if ts > self.last_update_timestamps[calc_cfg.timeframe]:
                self.last_update_timestamps[calc_cfg.timeframe] = ts
                self.feature_calculators[calc_cfg.name].update(df_tf[calc_cfg.source_col].iloc[idx])
        for calc_cfg in self.strat_cfg.stateful_calculators:
            values = self.feature_calculators[calc_cfg.name].get()
            if isinstance(values, dict):
                for key in calc_cfg.output_keys:
                    self.feature_histories[key].append(values.get(key, 1.0))
            else:
                self.feature_histories[calc_cfg.output_keys[0]].append(values)

    def step(self, action: np.ndarray):
        try:
            self._update_stateful_features(self.current_step)
            self._update_market_regime_and_volatility(self.current_step)
            current_price = self.close_prices_np[self.current_step]
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            initial_portfolio_value = self.balance + unrealized_pnl
            if initial_portfolio_value > self.episode_peak_value: self.episode_peak_value = initial_portfolio_value
            current_drawdown = (self.episode_peak_value - initial_portfolio_value) / self.episode_peak_value
            
            action_signal, action_size = np.clip(action[0], -1.0, 1.0), np.clip(action[1], 0.0, 1.0)
            if abs(action_signal) < 0.05 and action_size < 0.05: self.consecutive_inaction_steps += 1
            else: self.consecutive_inaction_steps = 0
            
            dynamic_limit = self.risk_manager.calculate_dynamic_position_limit(self.volatility_estimate, self.market_regime)
            target_notional = initial_portfolio_value * action_signal * (action_size * dynamic_limit)
            target_asset_quantity = target_notional / current_price if current_price > 1e-8 else 0
            trade_quantity = target_asset_quantity - self.asset_held
            total_cost = abs(trade_quantity) * current_price * (self.cfg.transaction_fee_pct + self.cfg.slippage_pct)
            self.balance += unrealized_pnl - total_cost
            self.asset_held = target_asset_quantity
            self.used_margin = (abs(self.asset_held) * current_price) / self.leverage
            if abs(trade_quantity) > 1e-8: self.entry_price = current_price
            
            self.current_step += 1
            next_price = self.close_prices_np[self.current_step]
            next_unrealized_pnl = self.asset_held * (next_price - self.entry_price)
            next_portfolio_value = self.balance + next_unrealized_pnl
            terminated = (self.episode_peak_value - next_portfolio_value) / self.episode_peak_value >= self.strat_cfg.max_drawdown_threshold
            truncated = self.current_step >= self.max_step
            
            portfolio_state = {'drawdown': current_drawdown, 'margin_ratio': (self.used_margin - unrealized_pnl) / (abs(self.asset_held * current_price) + 1e-9)}
            reward, reward_components = self.reward_calculator.calculate_immediate_reward(
                initial_portfolio_value, next_portfolio_value, action, portfolio_state, {}, self.previous_action, self.consecutive_inaction_steps)
            
            self.previous_action = action
            self.portfolio_history.append(next_portfolio_value)
            self.observation_history.append(self._get_single_step_observation(self.current_step))
            observation = self._get_observation_sequence()
            info = {'portfolio_value': next_portfolio_value, 'drawdown': current_drawdown, 'reward_components': reward_components}
            return observation, reward, terminated, truncated, info
        except Exception as e:
            logger.error(f"Error in step: {e}", exc_info=True)
            return self._get_observation_sequence(), -1.0, True, False, {'error': True}

    def reset_state_variables(self):
        self.balance, self.asset_held, self.used_margin, self.entry_price, self.consecutive_losses, self.trade_count, self.winning_trades, self.consecutive_inaction_steps = 1000000.0, 0.0, 0.0, 0.0, 0, 0, 0, 0
        self.episode_peak_value = self.balance
        self.market_regime, self.volatility_estimate = "UNCERTAIN", 0.02
        self.observation_history.clear(); self.portfolio_history.clear()
        self.previous_portfolio_value = self.balance
        self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_state_variables()
        warmup_period = self.cfg.get_required_warmup_period()
        self._initialize_stateful_features()
        self._warmup_features(warmup_period)
        self.current_step = warmup_period
        for i in range(self.strat_cfg.sequence_length):
            step_idx = self.current_step - self.strat_cfg.sequence_length + 1 + i
            self.observation_history.append(self._get_single_step_observation(step_idx))
        return self._get_observation_sequence(), {'portfolio_value': self.balance}

    def _warmup_features(self, warmup_steps: int):
        if self.precomputed_features_df is not None:
            precomputed_indexed = self.precomputed_features_df.set_index('timestamp')
            warmup_timestamps = self.base_timestamps[:warmup_steps]
            for key in self.strat_cfg.context_feature_keys:
                if key in precomputed_indexed.columns:
                    self.feature_histories[key].extend(precomputed_indexed[key].reindex(warmup_timestamps, method='ffill').fillna(0.0).values)
        else:
            for i in tqdm(range(warmup_steps), desc="Warming up features", leave=False, ncols=100): self._update_stateful_features(i)

    def _update_market_regime_and_volatility(self, step_index: int):
        if step_index >= 50:
            recent_prices = self.close_prices_np[max(0, step_index-50):step_index+1]
            returns = (recent_prices[1:] - recent_prices[:-1]) / (recent_prices[:-1] + 1e-9)
            if len(returns) > 10:
                self.volatility_estimate = np.std(returns) * np.sqrt(252)
                self.risk_manager.volatility_buffer.append(self.volatility_estimate)
                self.market_regime = self.risk_manager.update_market_regime(returns, self.volatility_estimate)

    def _get_single_step_observation(self, step_index) -> dict:
        try:
            raw_obs, current_timestamp = {}, self.base_timestamps_np[step_index]
            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value
                if key in [FeatureKeys.CONTEXT.value, FeatureKeys.PORTFOLIO_STATE.value, FeatureKeys.PRECOMPUTED_FEATURES.value]: continue
                freq = key.split('_')[-1].replace('m','T').replace('h','H').replace('s', 'S').upper()
                tf_data = self.timeframe_data_np[freq]
                end_idx = np.searchsorted(tf_data['index'], current_timestamp, side='right') - 1
                start_idx = max(0, end_idx - lookback + 1)
                
                if 'price' in key: window = tf_data['close'][start_idx : end_idx + 1]
                elif 'ohlc' in key: window = tf_data['ohlcv' if 'v' in key else 'ohlc'][start_idx : end_idx + 1]
                
                if len(window) < lookback:
                    pad_width = lookback - len(window)
                    pad_arg = ((pad_width, 0), (0, 0)) if window.ndim == 2 else (pad_width, 0)
                    window = np.pad(window, pad_arg, 'edge')
                raw_obs[key] = window.astype(np.float32)
            
            raw_obs[FeatureKeys.CONTEXT.value] = np.array([self.feature_histories[k][-1] if self.feature_histories[k] else 0.0 for k in self.strat_cfg.context_feature_keys], dtype=np.float32)
            raw_obs[FeatureKeys.PRECOMPUTED_FEATURES.value] = self.precomputed_features_np[step_index]
            
            current_price = self.close_prices_np[step_index]
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            portfolio_value = self.balance + unrealized_pnl
            pos_norm = np.clip((self.asset_held * current_price) / (portfolio_value + 1e-9), -1.0, 1.0)
            pnl_norm = np.tanh(unrealized_pnl / (self.used_margin + 1e-9))
            margin_ratio = np.clip((self.used_margin + unrealized_pnl) / (abs(self.asset_held * current_price) + 1e-9), 0, 2.0)
            regime_map = {"HIGH_VOLATILITY": 1.0, "TRENDING_UP": 0.8, "TRENDING_DOWN": -0.8, "LOW_VOLATILITY": 0.6}
            raw_obs[FeatureKeys.PORTFOLIO_STATE.value] = np.array([pos_norm, pnl_norm, margin_ratio, regime_map.get(self.market_regime, 0.0), np.tanh(self.volatility_estimate * 10)], dtype=np.float32)
            
            return self.normalizer.transform(raw_obs)
        except Exception as e:
            logger.error(f"Error getting observation for step {step_index}: {e}", exc_info=True)
            return {key: np.zeros(self.observation_space.spaces[key].shape[1:], dtype=np.float32) for key in self.observation_space.spaces.keys()}

    def _get_observation_sequence(self):
        return {key: np.stack([obs[key] for obs in self.observation_history]) for key in self.observation_space.spaces.keys()}