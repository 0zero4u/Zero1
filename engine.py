"""

FIXED ENGINE: Resolves Turtling and Reward Imbalance Issues

CRITICAL FIXES APPLIED:

1. ✅ BALANCED REWARDS: Fixed reward component scaling to prevent dominance
2. ✅ EXPLORATION INCENTIVE: Lower thresholds and better exploration rewards  
3. ✅ STABLE RISK METRICS: Robust risk-adjusted calculations with proper bounds
4. ✅ GRADUAL PENALTIES: Less harsh drawdown penalties with better progression
5. ✅ ACTION ENCOURAGEMENT: Rewards for taking reasonable actions
6. ✅ IMPROVED SCALING: Better base return scaling for small market movements

KEY CHANGES:
- Reduced drawdown penalty harshness
- Improved exploration bonus calculation
- Fixed risk-adjusted component instability
- Better reward scaling and normalization
- Added position change rewards
- More balanced component weights

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
    StatefulVWAPDistance,
)

# Map calculator names from config to their classes for dynamic instantiation
STATEFUL_CALCULATOR_MAP = {
    'StatefulBBWPercentRank': StatefulBBWPercentRank,
    'StatefulPriceDistanceMA': StatefulPriceDistanceMA,
    'StatefulSRDistances': StatefulSRDistances,
    'StatefulVWAPDistance': StatefulVWAPDistance,
}

logger = logging.getLogger(__name__)

class EnhancedRiskManager:
    """Advanced risk management system with dynamic limits and progressive penalties"""

    def __init__(self, config, leverage: float = 10.0):
        self.cfg = config
        self.leverage = leverage
        self.max_heat = 0.25
        self.volatility_lookback = 50
        self.risk_free_rate = 0.02
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

        # Adjust for leverage - higher leverage should have smaller position limits
        leverage_adjustment = min(1.0, 10.0 / self.leverage)

        return base_limit * vol_adjustment * regime_adjustment * leverage_adjustment


class FixedRewardCalculator:
    """
    FIXED: Enhanced reward calculator that encourages action and prevents turtling.
    
    KEY FIXES:
    1. Balanced reward components to prevent dominance
    2. Lower exploration thresholds to encourage action
    3. Stable risk-adjusted calculations
    4. Gradual penalty progression
    5. Better reward scaling for small returns
    """

    def __init__(self, config, leverage: float = 10.0, reward_weights: Dict[str, float] = None):
        self.cfg = config
        self.leverage = leverage
        self.return_scaler = RobustScaler()
        self.reward_history = deque(maxlen=1000)
        self.return_buffer = deque(maxlen=500)
        self.volatility_buffer = deque(maxlen=100)
        self.transaction_cost_buffer = deque(maxlen=50)

        # FIXED: Better balanced reward weights that prevent turtling
        self.weights = reward_weights or {
            'base_return': 3.0,         # Increased to make returns more significant
            'risk_adjusted': 0.4,       # Reduced from potentially problematic levels
            'stability': 0.2,           # Reward consistency
            'transaction_penalty': -0.1, # Light transaction cost penalty
            'drawdown_penalty': -0.8,   # Reduced harshness
            'position_penalty': -0.05,  # Very light position penalty
            'risk_bonus': 0.3,          # Reward good risk management
            'exploration_bonus': 0.5,   # Strong exploration incentive
            'inactivity_penalty': -0.2, # Moderate inactivity deterrent
            'action_reward': 0.3,       # NEW: Reward taking actions
        }

        # Get inactivity parameters from config
        self.inactivity_grace_period = getattr(self.cfg.strategy, 'inactivity_grace_period_steps', 10)
        self.penalty_ramp_up_steps = getattr(self.cfg.strategy, 'penalty_ramp_up_steps', 20)

        # FIXED: More reasonable scaling factor
        self.scaling_factor = getattr(self.cfg.strategy, 'reward_scaling_factor', 100.0) / self.leverage
        
        logger.info(f"FIXED reward calculator initialized with leverage {self.leverage}x, scaling_factor {self.scaling_factor:.2f}")

    def calculate_immediate_reward(self, prev_value: float, curr_value: float,
                                 action: np.ndarray, portfolio_state: Dict,
                                 market_state: Dict, previous_action: np.ndarray = None,
                                 consecutive_inactive_steps: int = 0) -> Tuple[float, Dict]:
        """
        FIXED: Calculate IMMEDIATE reward that encourages action and prevents turtling.
        """
        
        try:
            components = {}

            # FIXED: Base return component with better scaling
            period_return = (curr_value - prev_value) / max(prev_value, 1e-6)
            self.return_buffer.append(period_return)

            # Better return normalization - less aggressive tanh
            scaled_return = period_return * self.scaling_factor
            normalized_return = np.tanh(scaled_return * 0.5)  # Reduced scaling inside tanh
            components['base_return'] = normalized_return * self.weights['base_return']

            # FIXED: Stable risk-adjusted component
            if len(self.return_buffer) >= 20:  # Reduced from 30 for faster adaptation
                returns_array = np.array(list(self.return_buffer))
                
                # Use more stable calculation
                recent_returns = returns_array[-20:]
                mean_return = np.mean(recent_returns)
                volatility = np.std(recent_returns) + 1e-8  # Prevent division by zero
                
                # Simple Sharpe-like ratio with bounds
                if volatility > 1e-6:
                    sharpe_like = mean_return / volatility
                    # Bound the risk-adjusted score
                    risk_adjusted_score = np.clip(np.tanh(sharpe_like * 2), -0.5, 0.5)
                else:
                    risk_adjusted_score = 0.0
                    
                components['risk_adjusted'] = risk_adjusted_score * self.weights['risk_adjusted']
            else:
                components['risk_adjusted'] = 0.0

            # FIXED: Stability bonus (rewards consistent performance)
            if len(self.return_buffer) >= 10:
                recent_returns = np.array(list(self.return_buffer)[-10:])
                # Reward low volatility relative to returns
                return_vol = np.std(recent_returns)
                avg_return = abs(np.mean(recent_returns))
                
                if return_vol > 1e-8:
                    stability_score = avg_return / (return_vol + 1e-8)
                    stability_score = np.tanh(stability_score * 0.5)
                else:
                    stability_score = 0.1  # Small reward for no volatility
                    
                components['stability'] = max(0, stability_score * self.weights['stability'])
            else:
                components['stability'] = 0.0

            # FIXED: Transaction cost penalty (lighter)
            position_change = 0.0
            if previous_action is not None:
                position_change = abs(action[0] - previous_action[0])
                position_size_factor = (abs(action[1]) + abs(previous_action[1])) / 2
                
                # Lighter transaction penalty
                tx_penalty = position_change * position_size_factor * 0.5  # Reduced penalty
                components['transaction_penalty'] = tx_penalty * self.weights['transaction_penalty']
            else:
                components['transaction_penalty'] = 0.0

            # FIXED: Better exploration bonus (encourages action)
            action_magnitude = abs(action[0]) + abs(action[1])
            
            # Lower threshold for exploration
            if action_magnitude > 0.005:  # Much lower than 0.01
                exploration_reward = min(0.2, action_magnitude * 2)  # Cap the reward
                components['exploration_bonus'] = exploration_reward * self.weights['exploration_bonus']
            else:
                components['exploration_bonus'] = 0.0

            # NEW: Action reward (prevents turtling)
            if action_magnitude > 0.001:  # Very low threshold
                action_reward = min(0.1, action_magnitude)
                components['action_reward'] = action_reward * self.weights.get('action_reward', 0.0)
            else:
                components['action_reward'] = 0.0

            # FIXED: Gradual drawdown penalty (less harsh)
            current_drawdown = portfolio_state.get('drawdown', 0.0)
            if current_drawdown > 0.08:  # Start penalty later (8% instead of 5%)
                # More gradual penalty progression
                drawdown_excess = current_drawdown - 0.08
                penalty_factor = min(1.0, drawdown_excess / 0.12)  # Scale over 12% range
                penalty_factor = penalty_factor ** 0.7  # Less aggressive than sqrt
                components['drawdown_penalty'] = penalty_factor * self.weights['drawdown_penalty']
            else:
                components['drawdown_penalty'] = 0.0

            # FIXED: Position management penalty (much lighter)
            position_size = abs(action[1])
            if position_size > 0.9:  # Higher threshold
                size_penalty = (position_size - 0.9) * 0.5  # Reduced penalty
                components['position_penalty'] = size_penalty * self.weights['position_penalty']
            else:
                components['position_penalty'] = 0.0

            # Risk management bonus (unchanged - was working)
            margin_ratio = portfolio_state.get('margin_ratio', 2.0)
            if margin_ratio > 0.5:
                risk_bonus = min(0.2, (margin_ratio - 0.5) * 0.4)
                components['risk_bonus'] = risk_bonus * self.weights['risk_bonus']
            else:
                components['risk_bonus'] = 0.0

            # FIXED: Inactivity penalty with better progression
            inactivity_penalty_weight = self.weights.get('inactivity_penalty', 0.0)
            if inactivity_penalty_weight < 0 and consecutive_inactive_steps > self.inactivity_grace_period:
                steps_into_penalty = consecutive_inactive_steps - self.inactivity_grace_period
                # More gradual ramp-up
                ramp_progress = min(1.0, steps_into_penalty / self.penalty_ramp_up_steps)
                ramp_progress = ramp_progress ** 0.5  # Square root for gradual increase
                components['inactivity_penalty'] = ramp_progress * inactivity_penalty_weight
            else:
                components['inactivity_penalty'] = 0.0

            # Calculate total immediate reward
            total_reward = sum(components.values())

            # FIXED: Better bounds - allow for more reward range
            total_reward = np.clip(total_reward, -3.0, 3.0)

            # Update history
            self.reward_history.append(total_reward)

            return total_reward, components

        except Exception as e:
            logger.error(f"Error in immediate reward calculation: {e}")
            return 0.0, {'error': -0.1}


class FixedHierarchicalTradingEnvironment(gym.Env):
    """
    FIXED: Trading environment that prevents turtling and encourages learning.
    """

    def __init__(self, df_base_ohlc: pd.DataFrame, normalizer: Normalizer, config=None,
                 leverage: float = None, reward_weights: Dict[str, float] = None,
                 precomputed_features: Optional[pd.DataFrame] = None):
        super().__init__()
        
        self.cfg = config or SETTINGS
        self.strat_cfg = self.cfg.strategy
        self.normalizer = normalizer
        self.leverage = leverage if leverage is not None else self.strat_cfg.leverage

        logger.info("--- Initializing FIXED High-Performance Trading Environment ---")
        logger.info(f" -> FIXED: Prevents turtling and reward imbalance")
        logger.info(f" -> FIXED: Encourages exploration and learning")

        try:
            self.risk_manager = EnhancedRiskManager(self.cfg, leverage=self.leverage)
            self.reward_calculator = FixedRewardCalculator(self.cfg, leverage=self.leverage, reward_weights=reward_weights)

            # --- Data Resampling and Conversion to NumPy ---
            base_df = df_base_ohlc.set_index('timestamp')
            feature_timeframes = {calc.timeframe for calc in self.strat_cfg.stateful_calculators}
            non_market_keys = {FeatureKeys.CONTEXT, FeatureKeys.PORTFOLIO_STATE, FeatureKeys.PRECOMPUTED_FEATURES}
            model_timeframes = set(
                k.value.split('_')[-1].replace('m','T').replace('h','H').replace('s','S').upper()
                for k in self.strat_cfg.lookback_periods.keys()
                if k not in non_market_keys
            )

            all_required_freqs = model_timeframes.union(feature_timeframes)
            self.base_timestamps = base_df.resample(self.cfg.base_bar_timeframe.value).asfreq().index
            self.max_step = len(self.base_timestamps) - 2

            self.timeframes_np: Dict[str, Dict[str, np.ndarray]] = {}
            self.timeframe_indexers: Dict[str, np.ndarray] = {}

            logger.info("Resampling data, creating indexers, and converting to NumPy...")
            
            for freq in all_required_freqs:
                agg_rules = {
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
                    'volume_delta': 'sum', 'vwap': 'last', 'trade_count': 'sum'
                }

                valid_agg_rules = {k: v for k, v in agg_rules.items() if k in base_df.columns}
                df_resampled = base_df.resample(freq).agg(valid_agg_rules).ffill()

                # Reindex to align with the base timeframe for consistent length
                df_resampled = df_resampled.reindex(self.base_timestamps, method='ffill').fillna(0)

                # Create NumPy dictionary for this timeframe
                self.timeframes_np[freq] = {col: df_resampled[col].values for col in df_resampled.columns}

            # Process and store all pre-computed features as NumPy arrays
            if precomputed_features is not None:
                features_indexed = precomputed_features.set_index('timestamp')
                features_aligned = features_indexed.reindex(self.base_timestamps, method='ffill').fillna(0.0)
                self.all_features_np = {
                    key: features_aligned[key].values
                    for key in self.strat_cfg.context_feature_keys + self.strat_cfg.precomputed_feature_keys
                    if key in features_aligned.columns
                }
                logger.info(f"Processed and aligned {len(self.all_features_np)} pre-computed feature columns.")
            else:
                raise ValueError("`precomputed_features` must be provided for the optimized environment.")

            # Action and observation spaces
            self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

            obs_spaces = {}
            seq_len = self.strat_cfg.sequence_length

            for key, lookback in self.strat_cfg.lookback_periods.items():
                key_str = key.value
                if key_str.startswith('ohlcv_'):
                    shape = (seq_len, lookback, 5)
                elif key_str.startswith('ohlc_'):
                    shape = (seq_len, lookback, 4)
                elif key in [FeatureKeys.PORTFOLIO_STATE, FeatureKeys.CONTEXT, FeatureKeys.PRECOMPUTED_FEATURES]:
                    shape = (seq_len, lookback)
                else:
                    shape = (seq_len, lookback)
                obs_spaces[key_str] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

            self.observation_space = spaces.Dict(obs_spaces)
            self.observation_history = deque(maxlen=self.strat_cfg.sequence_length)

            # Initialize environment state
            self.portfolio_history = deque(maxlen=500)
            self.previous_portfolio_value = None
            self.episode_returns = []
            self.previous_action = None
            self.consecutive_losses = 0
            self.consecutive_inactive_steps = 0
            self.episode_peak_value = 0.0
            self.market_regime = "UNCERTAIN"
            self.volatility_estimate = 0.02
            self.step_rewards = []
            self.reward_components_history = []
            self.trade_count = 0
            self.winning_trades = 0

            logger.info("✅ FIXED environment initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize fixed environment: {e}", exc_info=True)
            raise

    def step(self, action: np.ndarray):
        """
        FIXED: Step function with improved reward calculation and action encouragement.
        """
        try:
            self._update_market_regime_and_volatility(self.current_step)

            # Get current price
            current_price = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][self.current_step]
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            initial_portfolio_value = self.balance + unrealized_pnl

            if initial_portfolio_value > self.episode_peak_value:
                self.episode_peak_value = initial_portfolio_value

            current_drawdown = (self.episode_peak_value - initial_portfolio_value) / self.episode_peak_value

            position_notional = abs(self.asset_held) * current_price
            margin_ratio = (self.used_margin - unrealized_pnl) / position_notional if position_notional > 0 else float('inf')

            # Handle liquidation (unchanged - was working)
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

            # Execute the trade
            action_signal = np.clip(action[0], -1.0, 1.0)
            action_size = np.clip(action[1], 0.0, 1.0)

            dynamic_limit = self.risk_manager.calculate_dynamic_position_limit(
                self.volatility_estimate, initial_portfolio_value, self.market_regime)

            effective_size = action_size * dynamic_limit
            target_notional = initial_portfolio_value * action_signal * effective_size
            target_asset_quantity = target_notional / current_price if current_price > 1e-8 else 0

            # Apply position limits
            max_allowable_margin = initial_portfolio_value * self.strat_cfg.max_margin_allocation_pct
            required_margin_for_target = (abs(target_asset_quantity) * current_price) / self.leverage

            if required_margin_for_target > max_allowable_margin:
                capped_notional = max_allowable_margin * self.leverage
                target_asset_quantity = (capped_notional / current_price * np.sign(target_asset_quantity)) if current_price > 1e-8 else 0

            required_margin = (abs(target_asset_quantity) * current_price) / self.leverage
            if required_margin > initial_portfolio_value:
                max_affordable_notional = initial_portfolio_value * self.leverage
                target_asset_quantity = (max_affordable_notional / current_price * np.sign(target_asset_quantity)) if current_price > 1e-8 else 0

            # Calculate trade costs
            trade_quantity = target_asset_quantity - self.asset_held
            trade_notional = abs(trade_quantity) * current_price
            base_fee = trade_notional * self.cfg.transaction_fee_pct
            slippage_cost = trade_notional * self.cfg.slippage_pct if abs(trade_quantity) > 0 else 0
            total_cost = base_fee + slippage_cost

            # FIXED: Better inactivity tracking
            action_magnitude = abs(action[0]) + abs(action[1])
            if action_magnitude > 0.001:  # Lower threshold
                self.consecutive_inactive_steps = 0
            else:
                self.consecutive_inactive_steps += 1

            # Execute trade
            self.balance += unrealized_pnl - total_cost
            self.asset_held = target_asset_quantity
            new_notional_value = abs(self.asset_held) * current_price
            self.used_margin = new_notional_value / self.leverage

            if abs(trade_quantity) > 1e-8:
                self.entry_price = current_price
                self.trade_count += 1

            # Move to next step
            self.current_step += 1
            truncated = self.current_step >= self.max_step

            next_price = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][self.current_step]
            next_unrealized_pnl = self.asset_held * (next_price - self.entry_price)
            next_portfolio_value = self.balance + next_unrealized_pnl

            # Check termination conditions
            next_drawdown_from_peak = (self.episode_peak_value - next_portfolio_value) / self.episode_peak_value
            terminated = next_drawdown_from_peak >= self.strat_cfg.max_drawdown_threshold

            portfolio_state = {
                'drawdown': current_drawdown,
                'margin_ratio': margin_ratio
            }

            market_state = {
                'regime': self.market_regime,
                'price': current_price,
                'volatility': self.volatility_estimate
            }

            # FIXED: Use improved reward calculation
            reward, reward_components = self.reward_calculator.calculate_immediate_reward(
                prev_value=initial_portfolio_value,
                curr_value=next_portfolio_value,
                action=action,
                portfolio_state=portfolio_state,
                market_state=market_state,
                previous_action=self.previous_action,
                consecutive_inactive_steps=self.consecutive_inactive_steps
            )

            if reward < 0: 
                self.consecutive_losses += 1
            else: 
                self.consecutive_losses = 0

            if reward > 0.1: 
                self.winning_trades += 1

            self.previous_action = action
            self.portfolio_history.append(next_portfolio_value)
            self.step_rewards.append(reward)
            self.reward_components_history.append(reward_components)

            if self.previous_portfolio_value is not None:
                period_return = (next_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
                self.episode_returns.append(period_return)
                self.risk_manager.return_buffer.append(period_return)

            self.previous_portfolio_value = next_portfolio_value

            # Update observation
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
                'fixed_reward_system': True,
                'action_magnitude': action_magnitude,
                'consecutive_inactive_steps': self.consecutive_inactive_steps
            }

            return observation, reward, terminated, truncated, info

        except Exception as e:
            logger.error(f"Error in fixed environment step: {e}", exc_info=True)
            observation = self._get_observation_sequence()
            info = {'portfolio_value': self.balance, 'error': True, 'leverage': self.leverage}
            return observation, -1.0, True, False, info

    def reset(self, seed=None, options=None):
        """Reset environment."""
        try:
            super().reset(seed=seed)
            
            self.balance = 1000000.0
            self.asset_held = 0.0
            self.used_margin = 0.0
            self.entry_price = 0.0
            self.consecutive_losses = 0
            self.consecutive_inactive_steps = 0
            self.episode_peak_value = self.balance
            self.market_regime = "UNCERTAIN"
            self.volatility_estimate = 0.02
            self.trade_count = 0
            self.winning_trades = 0

            warmup_period = self.cfg.get_required_warmup_period()
            self.current_step = warmup_period

            self.observation_history.clear()
            self.portfolio_history.clear()
            self.episode_returns.clear()
            self.step_rewards.clear()
            self.reward_components_history.clear()

            self.previous_portfolio_value = self.balance
            self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)

            for i in range(self.strat_cfg.sequence_length):
                step_idx = self.current_step - self.strat_cfg.sequence_length + 1 + i
                self._update_market_regime_and_volatility(step_idx)
                self.observation_history.append(self._get_single_step_observation(step_idx))

            observation = self._get_observation_sequence()

            info = {
                'balance': self.balance, 'asset_held': self.asset_held, 'portfolio_value': self.balance,
                'market_regime': self.market_regime, 'volatility_estimate': self.volatility_estimate,
                'leverage': self.leverage, 'fixed_reward_system': True
            }

            return observation, info

        except Exception as e:
            logger.error(f"Error resetting fixed environment: {e}", exc_info=True)
            raise

    def _get_current_context_features(self, step_index: int) -> np.ndarray:
        """Get context features for a step by direct NumPy array indexing."""
        final_vector = [self.all_features_np[key][step_index]
                       for key in self.strat_cfg.context_feature_keys]
        return np.array(final_vector, dtype=np.float32)

    def _update_market_regime_and_volatility(self, step_index: int):
        """Update market regime using fast NumPy slicing."""
        try:
            if step_index >= 50:
                close_prices_np = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close']
                recent_prices = close_prices_np[max(0, step_index - 50) : step_index + 1]
                
                # Use pandas for pct_change as it handles NaNs correctly
                returns = pd.Series(recent_prices).pct_change().dropna().values

                if len(returns) > 10:
                    self.volatility_estimate = np.std(returns) * np.sqrt(252)
                    self.risk_manager.volatility_buffer.append(self.volatility_estimate)
                    self.market_regime = self.risk_manager.update_market_regime(returns, self.volatility_estimate)

        except Exception as e:
            logger.warning(f"Error updating market regime: {e}")

    def _get_single_step_observation(self, step_index) -> dict:
        """Generate single step observation using NumPy arrays."""
        try:
            if self.normalizer is None: 
                return {}

            raw_obs = {}
            base_freq = self.cfg.base_bar_timeframe.value
            current_price = self.timeframes_np[base_freq]['close'][step_index]

            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value

                if key in [FeatureKeys.CONTEXT.value, FeatureKeys.PORTFOLIO_STATE.value, FeatureKeys.PRECOMPUTED_FEATURES.value]:
                    continue

                freq = key.split('_')[-1].replace('m','T').replace('h','H').replace('s', 'S').upper()

                end_idx = step_index
                start_idx = max(0, end_idx - lookback + 1)

                if key.startswith('price_'):
                    window = self.timeframes_np[freq]['close'][start_idx : end_idx + 1].astype(np.float32)
                    if len(window) < lookback:
                        window = np.pad(window, (lookback - len(window), 0), 'edge')
                    raw_obs[key] = window

                elif key.startswith('volume_delta_'):
                    window = self.timeframes_np[freq]['volume_delta'][start_idx : end_idx + 1].astype(np.float32)
                    if len(window) < lookback:
                        window = np.pad(window, (lookback - len(window), 0), 'constant')
                    raw_obs[key] = window

                elif key.startswith('ohlc'):
                    cols = ['open','high','low','close','volume'] if 'v' in key else ['open','high','low','close']
                    window_arrays = [self.timeframes_np[freq][c][start_idx : end_idx + 1] for c in cols]
                    window = np.stack(window_arrays, axis=1).astype(np.float32)
                    if len(window) < lookback:
                        padding = np.repeat(window[0:1], lookback - len(window), axis=0)
                        window = np.concatenate([padding, window], axis=0)
                    raw_obs[key] = window

            # Get features from pre-computed NumPy arrays
            raw_obs[FeatureKeys.CONTEXT.value] = self._get_current_context_features(step_index)
            
            precomputed_vector = np.array(
                [self.all_features_np[k][step_index] for k in self.strat_cfg.precomputed_feature_keys],
                dtype=np.float32
            )
            raw_obs[FeatureKeys.PRECOMPUTED_FEATURES.value] = precomputed_vector

            # Portfolio state
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
            if self.market_regime == "HIGH_VOLATILITY": 
                regime_encoding = 1.0
            elif self.market_regime == "TRENDING_UP": 
                regime_encoding = 0.8
            elif self.market_regime == "TRENDING_DOWN": 
                regime_encoding = -0.8
            elif self.market_regime == "LOW_VOLATILITY": 
                regime_encoding = 0.6
            elif self.market_regime == "SIDEWAYS": 
                regime_encoding = 0.0
            else: 
                regime_encoding = -0.2

            vol_adjusted_signal = np.tanh(self.volatility_estimate * 10)

            raw_obs[FeatureKeys.PORTFOLIO_STATE.value] = np.array([
                normalized_position, normalized_pnl, margin_ratio, regime_encoding, vol_adjusted_signal
            ], dtype=np.float32)

            return self.normalizer.transform(raw_obs)

        except Exception as e:
            logger.error(f"Error getting observation for step {step_index}: {e}", exc_info=True)
            obs = {}
            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value
                if key in [FeatureKeys.PORTFOLIO_STATE.value, FeatureKeys.CONTEXT.value, FeatureKeys.PRECOMPUTED_FEATURES.value]:
                    obs[key] = np.zeros(lookback, dtype=np.float32)
                elif key.startswith('ohlcv_'):
                    obs[key] = np.zeros((lookback, 5), dtype=np.float32)
                elif key.startswith('ohlc_'):
                    obs[key] = np.zeros((lookback, 4), dtype=np.float32)
                else:
                    obs[key] = np.zeros(lookback, dtype=np.float32)
            return obs

    def _get_observation_sequence(self):
        """Get observation sequence for model input"""
        try:
            return {key: np.stack([obs[key] for obs in self.observation_history])
                   for key in self.observation_space.spaces.keys()}
        except Exception as e:
            logger.error(f"Error getting observation sequence: {e}")
            return {key: np.zeros(self.observation_space.spaces[key].shape, dtype=np.float32)
                   for key in self.observation_space.spaces.keys()}

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics with fixed reward analysis"""
        try:
            if len(self.portfolio_history) < 2:
                return {}

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

            fixed_reward_metrics = {}
            if self.reward_components_history:
                components_df = pd.DataFrame(self.reward_components_history)
                for component in components_df.columns:
                    if component != 'error':
                        fixed_reward_metrics[f'avg_{component}'] = components_df[component].mean()

            base_metrics = {
                'total_return': total_return, 'volatility': volatility, 'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio, 'win_rate': win_rate, 'total_trades': self.trade_count,
                'avg_reward': avg_reward, 'reward_volatility': reward_volatility,
                'consecutive_losses': self.consecutive_losses, 'final_portfolio_value': final_value,
                'leverage': self.leverage, 'reward_scaling_factor': self.reward_calculator.scaling_factor,
                'fixed_reward_system': True, 'prevents_turtling': True,
            }

            base_metrics.update(fixed_reward_metrics)
            return base_metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'leverage': self.leverage, 'fixed_reward_system': True}
