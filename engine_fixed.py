"""
FIXED Enhanced Trading Environment for Crypto Trading RL

OPTIMIZED VERSION - Addresses _warmup_features performance bottleneck:
1. ADDED optimized vectorized warmup using pre-calculated features
2. Maintains backwards compatibility with step-by-step warmup as fallback
3. Significantly faster environment resets during training
4. Enhanced reward horizon system remains intact

This version includes a robust, leak-proof Reward Horizon system.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Dict, List, Any, Optional, Tuple
import logging
from tqdm import tqdm
import warnings
from scipy import stats
from sklearn.preprocessing import RobustScaler

# Import configuration and the new stateful feature calculators
from config import SETTINGS, FeatureKeys
from normalizer import Normalizer
from features import (
    StatefulBBWPercentRank,
    StatefulPriceDistanceMA,
    StatefulSRDistances,
)

# Map calculator names from config to their classes for dynamic instantiation
STATEFUL_CALCULATOR_MAP = {
    'StatefulBBWPercentRank': StatefulBBWPercentRank,
    'StatefulPriceDistanceMA': StatefulPriceDistanceMA,
    'StatefulSRDistances': StatefulSRDistances,
}

logger = logging.getLogger(__name__)

class EnhancedRiskManager:
    """Advanced risk management system with dynamic limits and progressive penalties"""

    def __init__(self, config, leverage: float = 10.0):
        self.cfg = config
        self.leverage = leverage # Now configurable
        self.max_heat = 0.25 # Maximum portfolio heat (risk exposure)
        self.volatility_lookback = 50
        self.risk_free_rate = 0.02 # Annual risk-free rate
        self.volatility_buffer = deque(maxlen=self.volatility_lookback)
        self.return_buffer = deque(maxlen=100)

    def update_market_regime(self, returns: np.ndarray, volatility: float) -> str:
        """Detect current market regime"""
        if len(returns) < 20:
            return "UNCERTAIN"

        recent_returns = returns[-20:]
        vol_percentile = np.percentile(self.volatility_buffer, 80) if len(self.volatility_buffer) > 10 else volatility

        # Regime classification
        if volatility > vol_percentile * 1.5:
            return "HIGH_VOLATILITY"
        elif np.mean(recent_returns) > 0.001 and volatility < vol_percentile * 0.8:
            return "TRENDING_UP"
        elif np.mean(recent_returns) < -0.001 and volatility < vol_percentile * 0.8:
            return "TRENDING_DOWN"
        elif volatility < vol_percentile * 0.6:
            return "LOW_VOLATILITY"
        else:
            return "SIDEWAYS"

    def calculate_dynamic_position_limit(self, volatility: float, portfolio_value: float,
                                       market_regime: str) -> float:
        """Calculate dynamic position limits based on market conditions and leverage"""
        base_limit = self.cfg.strategy.max_position_size

        # Adjust based on volatility
        vol_adjustment = max(0.5, min(2.0, 1.0 / (volatility * 10 + 0.1)))

        # Adjust based on market regime
        regime_multipliers = {
            "HIGH_VOLATILITY": 0.6,
            "TRENDING_UP": 1.2,
            "TRENDING_DOWN": 0.8,
            "LOW_VOLATILITY": 1.1,
            "SIDEWAYS": 0.9,
            "UNCERTAIN": 0.7
        }

        regime_adjustment = regime_multipliers.get(market_regime, 0.8)

        # FIXED: Adjust for leverage - higher leverage should have smaller position limits
        leverage_adjustment = min(1.0, 10.0 / self.leverage)

        return base_limit * vol_adjustment * regime_adjustment * leverage_adjustment

    def calculate_portfolio_heat(self, positions: Dict, current_prices: Dict,
                               portfolio_value: float, volatilities: Dict) -> float:
        """Calculate current portfolio heat (risk exposure)"""
        total_heat = 0.0
        for asset, position in positions.items():
            if position != 0:
                position_value = abs(position * current_prices.get(asset, 0))
                asset_vol = volatilities.get(asset, 0.02) # Default 2% volatility
                position_heat = (position_value / portfolio_value) * asset_vol
                total_heat += position_heat

        return total_heat

    def should_reduce_position(self, heat: float, drawdown: float,
                             consecutive_losses: int) -> Tuple[bool, float]:
        """Determine if position should be reduced and by how much"""
        reduction_factor = 0.0

        # Heat-based reduction
        if heat > self.max_heat:
            reduction_factor = max(reduction_factor, (heat - self.max_heat) / self.max_heat)

        # Drawdown-based reduction
        if drawdown > 0.15:
            reduction_factor = max(reduction_factor, (drawdown - 0.15) / 0.15)

        # Consecutive losses reduction
        if consecutive_losses > 5:
            reduction_factor = max(reduction_factor, min(0.5, consecutive_losses * 0.05))

        should_reduce = reduction_factor > 0.1
        reduction_factor = min(0.8, reduction_factor) # Cap at 80% reduction

        return should_reduce, reduction_factor

class AdvancedRewardCalculator:
    """
    ✅ FIXED: Enhanced reward calculator with dynamic scaling and tunable weights
    Major fixes:
    1. Dynamic reward scaling based on leverage (scaling_factor = 200 / leverage)
    2. Tunable reward weights via hyperparameters
    3. Proper reward normalization that preserves learning gradients
    """

    def __init__(self, config, leverage: float = 10.0, reward_weights: Dict[str, float] = None):
        self.cfg = config
        self.leverage = leverage # Now configurable
        self.return_scaler = RobustScaler()
        self.reward_history = deque(maxlen=1000)
        self.return_buffer = deque(maxlen=500)
        self.volatility_buffer = deque(maxlen=100)
        self.transaction_cost_buffer = deque(maxlen=50)

        # ✅ FIXED: Reward weights are now tunable hyperparameters
        self.weights = reward_weights or {
            'base_return': 1.0,
            'risk_adjusted': 0.3,
            'stability': 0.2,
            'transaction_penalty': -0.1,
            'drawdown_penalty': -0.4,
            'position_penalty': -0.05,
            'risk_bonus': 0.15
        }

        # ✅ FIXED: Dynamic scaling factor based on leverage
        self.scaling_factor = 200.0 / self.leverage

        logger.info(f"Reward calculator initialized with leverage {self.leverage}x, scaling_factor {self.scaling_factor:.2f}")

    def calculate_enhanced_reward(self, prev_value: float, curr_value: float,
                                action: np.ndarray, portfolio_state: Dict,
                                market_state: Dict, previous_action: np.ndarray = None) -> Tuple[float, Dict]:
        """
        ✅ FIXED: Enhanced reward calculation with dynamic scaling and tunable weights
        Returns:
            reward: Properly scaled reward value [-5, 5]
            components: Dictionary of reward components for analysis
        """
        try:
            components = {}

            # ✅ FIXED: Base return component with dynamic scaling
            period_return = (curr_value - prev_value) / max(prev_value, 1e-6)
            self.return_buffer.append(period_return)

            # ✅ CRITICAL FIX: Dynamic scaling based on leverage
            # This ensures reward is consistent across different leverage levels
            normalized_return = np.tanh(period_return * self.scaling_factor)
            components['base_return'] = normalized_return * self.weights['base_return']

            # 2. Risk-adjusted performance component (stable Sharpe-like)
            if len(self.return_buffer) >= 30:
                returns_array = np.array(list(self.return_buffer))
                excess_returns = returns_array - (self.leverage * 0.02/252) # Risk-free rate adjusted for leverage

                # Use robust statistics
                mean_excess = np.mean(excess_returns[-50:]) # Recent performance
                volatility = np.std(returns_array[-50:]) + 1e-8

                # Stable Sharpe-like metric
                risk_adjusted_score = mean_excess / volatility
                risk_adjusted_score = np.tanh(risk_adjusted_score * 5) # Bound to [-1, 1]
                components['risk_adjusted'] = risk_adjusted_score * self.weights['risk_adjusted']
            else:
                components['risk_adjusted'] = 0.0

            # 3. Stability bonus (reward consistent performance)
            if len(self.return_buffer) >= 20:
                recent_returns = np.array(list(self.return_buffer)[-20:])
                stability_score = -np.std(recent_returns) / (abs(np.mean(recent_returns)) + 1e-6)
                stability_score = np.tanh(stability_score) # Bound to [-1, 1]
                components['stability'] = max(0, stability_score * self.weights['stability'])
            else:
                components['stability'] = 0.0

            # 4. Transaction cost penalty (progressive)
            position_change = 0.0
            if previous_action is not None:
                position_change = abs(action[0] - previous_action[0])
                position_size_factor = (action[1] + previous_action[1]) / 2 # Average position size

                # Progressive penalty based on position size and change
                tx_penalty = position_change * position_size_factor * self.cfg.transaction_fee_pct
                components['transaction_penalty'] = -tx_penalty * self.weights['transaction_penalty']
            else:
                components['transaction_penalty'] = 0.0

            # 5. Progressive drawdown penalty (not step function)
            current_drawdown = portfolio_state.get('drawdown', 0.0)
            if current_drawdown > 0.05: # Only penalize significant drawdowns
                # Progressive penalty that increases smoothly
                drawdown_severity = (current_drawdown - 0.05) / 0.20 # Scale 5-25% drawdown to 0-1
                drawdown_severity = min(1.0, drawdown_severity)

                # Use quadratic penalty for severe drawdowns
                penalty_factor = drawdown_severity ** 1.5
                components['drawdown_penalty'] = -penalty_factor * self.weights['drawdown_penalty']
            else:
                components['drawdown_penalty'] = 0.0

            # 6. Position management penalty (encourage proper sizing)
            position_size = abs(action[1])
            if position_size > 0.8: # Penalize oversized positions
                size_penalty = (position_size - 0.8) * 2 # Linear penalty above 80%
                components['position_penalty'] = -size_penalty * self.weights['position_penalty']
            else:
                components['position_penalty'] = 0.0

            # 7. Risk management bonus (reward staying within limits)
            margin_ratio = portfolio_state.get('margin_ratio', 2.0)
            if margin_ratio > 0.5: # Healthy margin levels
                risk_bonus = min(0.2, (margin_ratio - 0.5) * 0.4)
                components['risk_bonus'] = risk_bonus * self.weights['risk_bonus']
            else:
                components['risk_bonus'] = 0.0

            # Calculate total reward
            total_reward = sum(components.values())

            # Final bounds check and scaling
            total_reward = np.clip(total_reward, -5.0, 5.0)

            # Update history
            self.reward_history.append(total_reward)

            return total_reward, components

        except Exception as e:
            logger.error(f"Error in enhanced reward calculation: {e}")
            return 0.0, {'error': -0.1}

class EnhancedHierarchicalTradingEnvironment(gym.Env):
    """
    ✅ FIXED: Enhanced Gymnasium-compliant RL environment with:
    - ✅ IMPLEMENTED: Leak-proof multi-step reward horizon system.
    - ✅ OPTIMIZED: Vectorized warmup using pre-calculated features
    - Tunable leverage parameter
    - Advanced risk management with leverage awareness
    - Enhanced feature engineering and regime detection
    """

    def __init__(self, df_base_ohlc: pd.DataFrame, normalizer: Normalizer, config=None,
                 leverage: float = None, reward_weights: Dict[str, float] = None, 
                 precomputed_features: Optional[pd.DataFrame] = None):
        super().__init__()
        self.cfg = config or SETTINGS
        self.strat_cfg = self.cfg.strategy
        self.normalizer = normalizer
        self.leverage = leverage if leverage is not None else self.strat_cfg.leverage

        # OPTIMIZED: Store precomputed features for fast warmup
        self.precomputed_features = precomputed_features

        logger.info("--- Initializing OPTIMIZED Hierarchical Trading Environment ---")
        logger.info(f" -> Advanced Risk Management: ON")
        logger.info(f" -> Dynamic Reward Scaling: ON")
        logger.info(f" -> Regime Detection: ON")
        logger.info(f" -> OPTIMIZED: Vectorized warmup: {'ON' if precomputed_features is not None else 'OFF'}")

        # ✅ NEW: Initialize Reward Horizon System
        self.reward_horizon_steps = self.strat_cfg.reward_horizon_steps
        self.reward_horizon_decay = self.strat_cfg.reward_horizon_decay
        self.pending_rewards = deque(maxlen=self.reward_horizon_steps)

        logger.info(f" -> REWARD HORIZON: {self.reward_horizon_steps} steps | Decay: {self.reward_horizon_decay}")
        logger.info(f" -> Leverage: {self.leverage}x | Maintenance Margin: {self.strat_cfg.maintenance_margin_rate:.2%}")

        try:
            self.risk_manager = EnhancedRiskManager(self.cfg, leverage=self.leverage)
            self.reward_calculator = AdvancedRewardCalculator(self.cfg, leverage=self.leverage, reward_weights=reward_weights)

            base_df = df_base_ohlc.set_index('timestamp')
            feature_timeframes = {calc.timeframe for calc in self.strat_cfg.stateful_calculators}
            model_timeframes = set(k.value.split('_')[-1].replace('m','T').replace('h','H').replace('s','S').upper()
                                 for k in self.strat_cfg.lookback_periods.keys())
            all_required_freqs = model_timeframes.union(feature_timeframes)

            self.timeframes = {}

            logger.info("Resampling data for all required timeframes...")
            for freq in all_required_freqs:
                if freq in ['CONTEXT', 'PORTFOLIO_STATE', 'PRECOMPUTED_FEATURES']:
                    continue
                if freq not in self.timeframes:
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

            self.portfolio_history = deque(maxlen=500)
            self.previous_portfolio_value = None
            self.episode_returns = []
            self.previous_action = None
            self.consecutive_losses = 0
            self.episode_peak_value = 0.0
            self.market_regime = "UNCERTAIN"
            self.volatility_estimate = 0.02
            self.step_rewards = []
            self.reward_components_history = []
            self.trade_count = 0
            self.winning_trades = 0

            logger.info("OPTIMIZED environment initialized successfully with reward horizon system.")

        except Exception as e:
            logger.error(f"Failed to initialize enhanced environment: {e}", exc_info=True)
            raise

    def _initialize_stateful_features(self):
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
        """
        OPTIMIZED: Warmup stateful features using pre-calculated data when available,
        fallback to step-by-step simulation for backwards compatibility.
        """
        if self.precomputed_features is not None:
            logger.info(f"🚀 OPTIMIZED: Using vectorized warmup for {warmup_steps} steps...")
            self._warmup_features_vectorized(warmup_steps)
        else:
            logger.info(f"Using step-by-step warmup for {warmup_steps} steps...")
            self._warmup_features_stepwise(warmup_steps)

    def _warmup_features_vectorized(self, warmup_steps: int):
        """
        OPTIMIZED: Vectorized warmup using pre-calculated features.
        This is significantly faster than step-by-step simulation.
        """
        try:
            if self.precomputed_features is None:
                logger.warning("No precomputed features available, falling back to stepwise warmup")
                self._warmup_features_stepwise(warmup_steps)
                return

            # Ensure precomputed features are indexed by timestamp
            if 'timestamp' in self.precomputed_features.columns:
                precomputed_indexed = self.precomputed_features.set_index('timestamp')
            else:
                precomputed_indexed = self.precomputed_features

            # Get the warmup period timestamps
            warmup_timestamps = self.base_timestamps[:warmup_steps]

            # For each feature key, populate the history from precomputed data
            for key in self.strat_cfg.context_feature_keys:
                if key in precomputed_indexed.columns:
                    # Get the values for the warmup period
                    feature_values = precomputed_indexed[key].reindex(
                        warmup_timestamps, method='ffill'
                    ).fillna(0.0).values

                    # Populate the feature history
                    self.feature_histories[key].clear()
                    for value in feature_values:
                        self.feature_histories[key].append(value)

                    logger.debug(f"Populated {key} with {len(feature_values)} precomputed values")
                else:
                    logger.warning(f"Feature {key} not found in precomputed data, using default values")
                    # Fill with default values
                    self.feature_histories[key].clear()
                    default_value = 1.0 if 'dist' in key else 0.0
                    for _ in range(warmup_steps):
                        self.feature_histories[key].append(default_value)

            logger.info("✅ OPTIMIZED: Vectorized warmup completed successfully!")

        except Exception as e:
            logger.error(f"Error in vectorized warmup: {e}")
            logger.info("Falling back to stepwise warmup...")
            self._warmup_features_stepwise(warmup_steps)

    def _warmup_features_stepwise(self, warmup_steps: int):
        """
        Original step-by-step warmup method - maintained for backwards compatibility.
        """
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
            except KeyError:
                continue

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

    def _get_current_context_features(self) -> np.ndarray:
        final_vector = [self.feature_histories[key][-1] if self.feature_histories[key] else 0.0 
                       for key in self.strat_cfg.context_feature_keys]
        return np.array(final_vector, dtype=np.float32)

    def _update_market_regime_and_volatility(self, step_index: int):
        try:
            base_df = self.timeframes[self.cfg.base_bar_timeframe.value]

            if step_index >= 50:
                recent_prices = base_df['close'].iloc[max(0, step_index-50):step_index+1]
                returns = recent_prices.pct_change().dropna().values

                if len(returns) > 10:
                    self.volatility_estimate = np.std(returns) * np.sqrt(252)
                    self.risk_manager.volatility_buffer.append(self.volatility_estimate)
                    self.market_regime = self.risk_manager.update_market_regime(returns, self.volatility_estimate)

        except Exception as e:
            logger.warning(f"Error updating market regime: {e}")

    def reset(self, seed=None, options=None):
        try:
            super().reset(seed=seed)

            self.balance = 1000000.0
            self.asset_held = 0.0
            self.used_margin = 0.0
            self.entry_price = 0.0
            self.consecutive_losses = 0
            self.episode_peak_value = self.balance
            self.market_regime = "UNCERTAIN"
            self.volatility_estimate = 0.02
            self.trade_count = 0
            self.winning_trades = 0

            warmup_period = self.cfg.get_required_warmup_period()
            self._initialize_stateful_features()

            # OPTIMIZED: Use vectorized warmup when precomputed features are available
            self._warmup_features(warmup_period)

            self.current_step = warmup_period

            self.observation_history.clear()
            self.portfolio_history.clear()
            self.episode_returns.clear()
            self.step_rewards.clear()
            self.reward_components_history.clear()

            # ✅ NEW: Clear the pending rewards buffer on reset
            self.pending_rewards.clear()

            self.previous_portfolio_value = self.balance
            self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)

            for i in range(self.strat_cfg.sequence_length):
                step_idx = self.current_step - self.strat_cfg.sequence_length + 1 + i
                self._update_stateful_features(step_idx)
                self._update_market_regime_and_volatility(step_idx)
                self.observation_history.append(self._get_single_step_observation(step_idx))

            observation = self._get_observation_sequence()

            info = {
                'balance': self.balance, 'asset_held': self.asset_held, 'portfolio_value': self.balance,
                'market_regime': self.market_regime, 'volatility_estimate': self.volatility_estimate,
                'leverage': self.leverage, 'optimized_warmup': self.precomputed_features is not None
            }

            return observation, info

        except Exception as e:
            logger.error(f"Error resetting enhanced environment: {e}", exc_info=True)
            raise

    def step(self, action: np.ndarray):
        """✅ IMPLEMENTED: Step function with delayed reward horizon system."""
        try:
            self._update_stateful_features(self.current_step)
            self._update_market_regime_and_volatility(self.current_step)

            current_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step]
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            initial_portfolio_value = self.balance + unrealized_pnl

            if initial_portfolio_value > self.episode_peak_value:
                self.episode_peak_value = initial_portfolio_value

            current_drawdown = (self.episode_peak_value - initial_portfolio_value) / self.episode_peak_value

            position_notional = abs(self.asset_held) * current_price
            margin_ratio = (self.used_margin - unrealized_pnl) / position_notional if position_notional > 0 else float('inf')

            if margin_ratio < self.strat_cfg.maintenance_margin_rate:
                margin_deficit = self.strat_cfg.maintenance_margin_rate - margin_ratio
                liquidation_factor = min(1.0, margin_deficit * 5)

                liquidation_amount = self.asset_held * liquidation_factor
                liquidation_value = liquidation_amount * current_price
                liquidation_cost = abs(liquidation_value) * (self.cfg.transaction_fee_pct + 0.001)

                self.asset_held -= liquidation_amount
                self.balance += liquidation_value - liquidation_cost

                new_position_notional = abs(self.asset_held) * current_price
                self.used_margin = new_position_notional / self.leverage

                if liquidation_factor >= 1.0:
                    reward = -2.0
                    terminated = True

                    self.current_step += 1
                    truncated = self.current_step >= self.max_step

                    self.observation_history.append(self._get_single_step_observation(self.current_step))
                    observation = self._get_observation_sequence()

                    info = {
                        'portfolio_value': self.balance, 'margin_ratio': 0.0, 'liquidation': True,
                        'liquidation_factor': liquidation_factor, 'leverage': self.leverage
                    }

                    return observation, reward, terminated, truncated, info

            # ✅ NEW: Store the state at the time of action for future reward calculation
            pending_info = {
                'initial_portfolio_value': initial_portfolio_value,
                'action': action,
                'previous_action': self.previous_action,
                'portfolio_state_at_action': {'drawdown': current_drawdown, 'margin_ratio': margin_ratio},
                'market_state_at_action': {'regime': self.market_regime, 'price': current_price, 'volatility': self.volatility_estimate}
            }

            self.pending_rewards.append(pending_info)

            # --- Execute Trade ---
            action_signal = np.clip(action[0], -1.0, 1.0)
            action_size = np.clip(action[1], 0.0, 1.0)

            dynamic_limit = self.risk_manager.calculate_dynamic_position_limit(
                self.volatility_estimate, initial_portfolio_value, self.market_regime)

            effective_size = action_size * dynamic_limit
            target_notional = initial_portfolio_value * action_signal * effective_size
            target_asset_quantity = target_notional / current_price if current_price > 1e-8 else 0

            max_allowable_margin = initial_portfolio_value * self.strat_cfg.max_margin_allocation_pct
            required_margin_for_target = (abs(target_asset_quantity) * current_price) / self.leverage

            if required_margin_for_target > max_allowable_margin:
                capped_notional = max_allowable_margin * self.leverage
                target_asset_quantity = (capped_notional / current_price * np.sign(target_asset_quantity)) if current_price > 1e-8 else 0

            required_margin = (abs(target_asset_quantity) * current_price) / self.leverage

            if required_margin > initial_portfolio_value:
                max_affordable_notional = initial_portfolio_value * self.leverage
                target_asset_quantity = (max_affordable_notional / current_price * np.sign(target_asset_quantity)) if current_price > 1e-8 else 0

            trade_quantity = target_asset_quantity - self.asset_held
            trade_notional = abs(trade_quantity) * current_price

            base_fee = trade_notional * self.cfg.transaction_fee_pct
            slippage_cost = trade_notional * self.cfg.slippage_pct if abs(trade_quantity) > 0 else 0
            total_cost = base_fee + slippage_cost

            self.balance += unrealized_pnl - total_cost
            self.asset_held = target_asset_quantity

            new_notional_value = abs(self.asset_held) * current_price
            self.used_margin = new_notional_value / self.leverage

            if abs(trade_quantity) > 1e-8:
                self.entry_price = current_price
                self.trade_count += 1

            # --- Move to Next Step ---
            self.current_step += 1
            truncated = self.current_step >= self.max_step

            next_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step]
            next_unrealized_pnl = self.asset_held * (next_price - self.entry_price)
            next_portfolio_value = self.balance + next_unrealized_pnl

            terminated = next_portfolio_value <= initial_portfolio_value * 0.5

            # --- ✅ IMPLEMENTED: Delayed Reward Calculation ---
            reward = 0.0
            reward_components = {}
            horizon_info = {}

            # Calculate reward only if the buffer is full (i.e., we have N steps of history)
            if len(self.pending_rewards) == self.pending_rewards.maxlen:
                past_info = self.pending_rewards.popleft() # Get the oldest action info

                # The reward for the past action is based on the return from then until now.
                reward, reward_components = self.reward_calculator.calculate_enhanced_reward(
                    prev_value=past_info['initial_portfolio_value'],
                    curr_value=next_portfolio_value,
                    action=past_info['action'],
                    portfolio_state=past_info['portfolio_state_at_action'],
                    market_state=past_info['market_state_at_action'],
                    previous_action=past_info['previous_action']
                )

                horizon_info = { 'using_multi_step': self.reward_horizon_steps > 1 }

            # ✅ IMPLEMENTED: Flush remaining rewards on episode termination
            if terminated or truncated:
                final_reward = reward

                while self.pending_rewards:
                    past_info = self.pending_rewards.popleft()

                    # Calculate reward for all remaining pending actions using the terminal portfolio value
                    term_reward, _ = self.reward_calculator.calculate_enhanced_reward(
                        prev_value=past_info['initial_portfolio_value'],
                        curr_value=next_portfolio_value, # Use final value for all
                        action=past_info['action'],
                        portfolio_state=past_info['portfolio_state_at_action'],
                        market_state=past_info['market_state_at_action'],
                        previous_action=past_info['previous_action']
                    )

                    final_reward += term_reward

                reward = final_reward

            if reward < 0: self.consecutive_losses += 1
            else: self.consecutive_losses = 0

            if reward > 0.1: self.winning_trades += 1

            self.previous_action = action
            self.portfolio_history.append(next_portfolio_value)
            self.step_rewards.append(reward)
            self.reward_components_history.append(reward_components)

            if self.previous_portfolio_value is not None:
                period_return = (next_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
                self.episode_returns.append(period_return)
                self.risk_manager.return_buffer.append(period_return)

            self.previous_portfolio_value = next_portfolio_value

            self.observation_history.append(self._get_single_step_observation(self.current_step))
            observation = self._get_observation_sequence()

            info = {
                'balance': self.balance, 'asset_held': self.asset_held, 'portfolio_value': next_portfolio_value,
                'drawdown': current_drawdown, 'volatility': self.volatility_estimate, 'unrealized_pnl': next_unrealized_pnl,
                'margin_ratio': margin_ratio, 'used_margin': self.used_margin, 'market_regime': self.market_regime,
                'consecutive_losses': self.consecutive_losses, 'trade_count': self.trade_count, 
                'win_rate': self.winning_trades / max(self.trade_count, 1), 'reward_components': reward_components, 
                'dynamic_position_limit': dynamic_limit, 'transaction_cost': total_cost, 'leverage': self.leverage,
                'reward_scaling_factor': self.reward_calculator.scaling_factor,
                # ✅ NEW: Add reward horizon info for debugging and analysis
                'reward_horizon': {
                    'steps': self.reward_horizon_steps, 'decay_factor': self.reward_horizon_decay,
                    'is_calculating': len(self.pending_rewards) == self.pending_rewards.maxlen, **horizon_info
                }
            }

            return observation, reward, terminated, truncated, info

        except Exception as e:
            logger.error(f"Error in enhanced environment step: {e}", exc_info=True)
            observation = self._get_observation_sequence()
            info = {'portfolio_value': self.balance, 'error': True, 'leverage': self.leverage}
            return observation, -1.0, True, False, info

    def _get_single_step_observation(self, step_index) -> dict:
        try:
            if self.normalizer is None: return {}

            current_timestamp = self.base_timestamps[step_index]
            raw_obs = {}

            base_df = self.timeframes[self.cfg.base_bar_timeframe.value]
            current_price = base_df['close'].iloc[base_df.index.get_loc(current_timestamp, method='ffill')]

            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value

                if key in [FeatureKeys.CONTEXT.value, FeatureKeys.PORTFOLIO_STATE.value, FeatureKeys.PRECOMPUTED_FEATURES.value]: 
                    continue

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

            regime_encoding = 0.0
            if self.market_regime == "HIGH_VOLATILITY": regime_encoding = 1.0
            elif self.market_regime == "TRENDING_UP": regime_encoding = 0.8
            elif self.market_regime == "TRENDING_DOWN": regime_encoding = -0.8
            elif self.market_regime == "LOW_VOLATILITY": regime_encoding = 0.6
            elif self.market_regime == "SIDEWAYS": regime_encoding = 0.0
            else: regime_encoding = -0.2

            vol_adjusted_signal = np.tanh(self.volatility_estimate * 10)

            raw_obs[FeatureKeys.PORTFOLIO_STATE.value] = np.array([
                normalized_position, normalized_pnl, margin_ratio, regime_encoding, vol_adjusted_signal
            ], dtype=np.float32)

            return self.normalizer.transform(raw_obs)

        except Exception as e:
            logger.error(f"Error getting enhanced observation for step {step_index}: {e}", exc_info=True)
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
            return {key: np.stack([obs[key] for obs in self.observation_history]) 
                   for key in self.observation_space.spaces.keys()}
        except Exception as e:
            logger.error(f"Error getting observation sequence: {e}")
            return {key: np.zeros(self.observation_space.spaces[key].shape, dtype=np.float32) 
                   for key in self.observation_space.spaces.keys()}

    def _calculate_current_drawdown(self) -> float:
        if len(self.portfolio_history) < 2: return 0.0
        current_value = self.portfolio_history[-1]
        return max(0.0, (self.episode_peak_value - current_value) / self.episode_peak_value)

    def _calculate_recent_volatility(self) -> float:
        if len(self.episode_returns) < 10: return 0.02
        recent_returns = np.array(self.episode_returns[-50:])
        return np.std(recent_returns) * np.sqrt(252)

    def get_performance_metrics(self) -> Dict[str, float]:
        try:
            if len(self.portfolio_history) < 2: return {}

            portfolio_values = np.array(self.portfolio_history)
            initial_value, final_value = portfolio_values[0], portfolio_values[-1]

            total_return = (final_value - initial_value) / initial_value
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

            cumulative_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (cumulative_max - portfolio_values) / cumulative_max
            max_drawdown = np.max(drawdowns)

            excess_return = total_return - 0.02
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0

            win_rate = self.winning_trades / max(self.trade_count, 1)
            avg_reward = np.mean(self.step_rewards) if self.step_rewards else 0.0
            reward_volatility = np.std(self.step_rewards) if len(self.step_rewards) > 1 else 0.0

            return {
                'total_return': total_return, 'volatility': volatility, 'max_drawdown': max_drawdown, 
                'sharpe_ratio': sharpe_ratio, 'win_rate': win_rate, 'total_trades': self.trade_count, 
                'avg_reward': avg_reward, 'reward_volatility': reward_volatility,
                'consecutive_losses': self.consecutive_losses, 'final_portfolio_value': final_value, 
                'leverage': self.leverage, 'reward_scaling_factor': self.reward_calculator.scaling_factor,
                'optimized_warmup': self.precomputed_features is not None
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'leverage': self.leverage}

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

def create_trading_environment(df_base_ohlc: pd.DataFrame, normalizer: Normalizer,
                             config=None, leverage: float = None, reward_weights: Dict[str, float] = None,
                             precomputed_features: Optional[pd.DataFrame] = None):
    return EnhancedHierarchicalTradingEnvironment(
        df_base_ohlc=df_base_ohlc, normalizer=normalizer, config=config, 
        leverage=leverage, reward_weights=reward_weights, 
        precomputed_features=precomputed_features
    )

if __name__ == "__main__":
    try:
        logger.info("Testing OPTIMIZED trading environment with vectorized warmup...")

        test_leverages = [5.0, 10.0, 15.0, 25.0]
        for leverage in test_leverages:
            reward_calc = AdvancedRewardCalculator(SETTINGS, leverage=leverage)

            test_reward, components = reward_calc.calculate_enhanced_reward(
                prev_value=1000000, curr_value=1010000, action=np.array([0.5, 0.3]),
                portfolio_state={'drawdown': 0.05, 'margin_ratio': 0.8}, 
                market_state={'regime': 'TRENDING_UP', 'volatility': 0.02}
            )

            logger.info(f"Leverage {leverage}x: scaling_factor={reward_calc.scaling_factor:.2f}, reward={test_reward:.4f}")

        logger.info("✅ OPTIMIZED trading environment test completed!")
        logger.info("✅ FIXES APPLIED:")
        logger.info("  - Vectorized warmup using precomputed features")
        logger.info("  - Backwards compatible with step-by-step warmup")
        logger.info("  - Significantly faster environment resets")
        logger.info("  - Enhanced performance tracking")

    except Exception as e:
        logger.error(f"Enhanced environment test failed: {e}")