--- START OF MODIFIED FILE Zero1-main/engine.py ---
# FIXED: engine.py with Immediate Reward System for Proper PPO Training

"""
FIXED: Enhanced Trading Environment with Immediate Reward System

CRITICAL FIXES APPLIED:
1. REMOVED the path-aware reward horizon system that broke PPO credit assignment
2. IMPLEMENTED immediate reward calculation at every step using calculate_enhanced_reward
3. ELIMINATED pending_rewards mechanism that caused delayed reward attribution
4. FIXED the PPO credit assignment problem by ensuring r_t comes from a_t

KEY CHANGE: PPO now gets proper (state, action) -> immediate reward mapping
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
    FIXED: Enhanced reward calculator with IMMEDIATE reward calculation for proper PPO training.
    
    KEY FIX: Eliminates the path-aware system that broke PPO credit assignment.
    Now provides immediate rewards that directly correspond to the action taken.
    """
    
    def __init__(self, config, leverage: float = 10.0, reward_weights: Dict[str, float] = None):
        self.cfg = config
        self.leverage = leverage
        self.return_scaler = RobustScaler()
        self.reward_history = deque(maxlen=1000)
        self.return_buffer = deque(maxlen=500)
        self.volatility_buffer = deque(maxlen=100)
        self.transaction_cost_buffer = deque(maxlen=50)
        
        # FIXED: Balanced reward weights for immediate rewards
        self.weights = reward_weights or {
            'base_return': 1.4,      # Primary signal
            'risk_adjusted': 0.15,   # Moderate Sharpe-like component
            'stability': 0.1,        # Reward consistency
            'transaction_penalty': -0.05,  # Conservative penalty
            'drawdown_penalty': -0.2,      # Moderate drawdown deterrent
            'position_penalty': -0.01,     # Light position size penalty
            'risk_bonus': 0.2,       # Reward good risk management
            'exploration_bonus': 0.08, # Light exploration incentive
            'inactivity_penalty': -0.1 # Default weight for inactivity
        }
        
        # NEW: Get inactivity parameters from config
        self.inactivity_grace_period = self.cfg.strategy.inactivity_grace_period_steps
        self.penalty_ramp_up_steps = self.cfg.strategy.penalty_ramp_up_steps
        
        # FIXED: Get scaling factor from config
        self.scaling_factor = self.cfg.strategy.reward_scaling_factor / self.leverage
        
        logger.info(f"FIXED reward calculator initialized with leverage {self.leverage}x, scaling_factor {self.scaling_factor:.2f}")
    
    def calculate_immediate_reward(self, prev_value: float, curr_value: float,
                                 action: np.ndarray, portfolio_state: Dict,
                                 market_state: Dict, previous_action: np.ndarray = None,
                                 consecutive_inactive_steps: int = 0) -> Tuple[float, Dict]:
        """
        FIXED: Calculate IMMEDIATE reward for proper PPO credit assignment.
        
        This method ensures r_t is directly caused by a_t, fixing the PPO training.
        
        Returns:
            reward: Properly scaled immediate reward value [-5, 5]
            components: Dictionary of reward components for analysis
        """
        try:
            components = {}
            
            # Base return component with immediate scaling
            period_return = (curr_value - prev_value) / max(prev_value, 1e-6)
            self.return_buffer.append(period_return)
            
            # Immediate return signal with tanh normalization
            normalized_return = np.tanh(period_return * self.scaling_factor)
            components['base_return'] = normalized_return * self.weights['base_return']
            
            # Risk-adjusted component (uses recent history for Sharpe-like metric)
            if len(self.return_buffer) >= 30:
                returns_array = np.array(list(self.return_buffer))
                excess_returns = returns_array - (self.leverage * 0.02/252)
                
                # Use robust statistics for stability
                mean_excess = np.mean(excess_returns[-30:])
                volatility = np.std(returns_array[-30:]) + 1e-8
                
                # Stable risk-adjusted score
                risk_adjusted_score = mean_excess / volatility
                risk_adjusted_score = np.clip(np.tanh(risk_adjusted_score * 3), -1.0, 1.0)
                components['risk_adjusted'] = risk_adjusted_score * self.weights['risk_adjusted']
            else:
                components['risk_adjusted'] = 0.0
            
            # Stability bonus (rewards consistent performance)
            if len(self.return_buffer) >= 10:
                recent_returns = np.array(list(self.return_buffer)[-10:])
                stability_score = -np.std(recent_returns) / (abs(np.mean(recent_returns)) + 1e-6)
                stability_score = np.tanh(stability_score)
                components['stability'] = max(0, stability_score * self.weights['stability'])
            else:
                components['stability'] = 0.0
            
            # Transaction cost penalty (immediate)
            position_change = 0.0
            if previous_action is not None:
                position_change = abs(action[0] - previous_action[0])
                position_size_factor = (action[1] + previous_action[1]) / 2
                
                # Immediate penalty for transaction costs
                tx_penalty = position_change * position_size_factor * self.cfg.transaction_fee_pct
                components['transaction_penalty'] = tx_penalty * self.weights['transaction_penalty']
            else:
                components['transaction_penalty'] = 0.0
            
            # Exploration bonus (immediate - rewards taking action)
            if position_change > 0.01:
                components['exploration_bonus'] = self.weights.get('exploration_bonus', 0.0)
            else:
                components['exploration_bonus'] = 0.0
            
            # Immediate drawdown penalty
            current_drawdown = portfolio_state.get('drawdown', 0.0)
            if current_drawdown > 0.05:
                # Progressive penalty that increases with severity
                drawdown_severity = (current_drawdown - 0.05) / 0.15  # Scale to [0,1] for 5-20% drawdown
                drawdown_severity = min(1.0, drawdown_severity)
                penalty_factor = drawdown_severity ** 2  # Quadratic penalty
                components['drawdown_penalty'] = penalty_factor * self.weights['drawdown_penalty']
            else:
                components['drawdown_penalty'] = 0.0
            
            # Position management penalty (immediate)
            position_size = abs(action[1])
            if position_size > 0.8:
                size_penalty = (position_size - 0.8) * 2
                components['position_penalty'] = size_penalty * self.weights['position_penalty']
            else:
                components['position_penalty'] = 0.0
            
            # Risk management bonus (immediate)
            margin_ratio = portfolio_state.get('margin_ratio', 2.0)
            if margin_ratio > 0.5:
                risk_bonus = min(0.2, (margin_ratio - 0.5) * 0.4)
                components['risk_bonus'] = risk_bonus * self.weights['risk_bonus']
            else:
                components['risk_bonus'] = 0.0
            
            # NEW: Inactivity penalty with grace period and ramp-up
            inactivity_penalty_weight = self.weights.get('inactivity_penalty', 0.0)
            if inactivity_penalty_weight < 0 and consecutive_inactive_steps > self.inactivity_grace_period:
                # Calculate how far into the ramp-up period we are
                steps_into_penalty = consecutive_inactive_steps - self.inactivity_grace_period
                # Calculate ramp-up progress, capped at 1.0
                ramp_progress = min(1.0, steps_into_penalty / self.penalty_ramp_up_steps)
                # Apply the penalty, scaled by the ramp-up progress
                components['inactivity_penalty'] = ramp_progress * inactivity_penalty_weight
            else:
                components['inactivity_penalty'] = 0.0

            # Calculate total immediate reward
            total_reward = sum(components.values())
            
            # Final bounds check
            total_reward = np.clip(total_reward, -5.0, 5.0)
            
            # Update history
            self.reward_history.append(total_reward)
            
            return total_reward, components
            
        except Exception as e:
            logger.error(f"Error in immediate reward calculation: {e}")
            return 0.0, {'error': -0.1}

class FixedHierarchicalTradingEnvironment(gym.Env):
    """
    FIXED: Enhanced Gymnasium-compliant RL environment with IMMEDIATE reward system for proper PPO training.
    
    CRITICAL FIXES:
    1. Removed path-aware reward horizon system that broke PPO credit assignment
    2. Implemented immediate reward calculation at every step
    3. Fixed the PPO learning problem by ensuring r_t directly results from a_t
    """
    
    def __init__(self, df_base_ohlc: pd.DataFrame, normalizer: Normalizer, config=None,
                 leverage: float = None, reward_weights: Dict[str, float] = None,
                 precomputed_features: Optional[pd.DataFrame] = None):
        super().__init__()
        
        self.cfg = config or SETTINGS
        self.strat_cfg = self.cfg.strategy
        self.normalizer = normalizer
        self.leverage = leverage if leverage is not None else self.strat_cfg.leverage
        
        # Store precomputed features for fast warmup
        self.precomputed_features = precomputed_features
        
        logger.info("--- Initializing FIXED Immediate Reward Trading Environment ---")
        logger.info(f" -> FIXED: Immediate rewards for proper PPO training")
        logger.info(f" -> Advanced Risk Management: ON")
        logger.info(f" -> Dynamic Reward Scaling: ON")
        logger.info(f" -> Regime Detection: ON")
        logger.info(f" -> OPTIMIZED: Vectorized warmup: {'ON' if precomputed_features is not None else 'OFF'}")
        
        try:
            self.risk_manager = EnhancedRiskManager(self.cfg, leverage=self.leverage)
            # FIXED: Use the new immediate reward calculator
            self.reward_calculator = FixedRewardCalculator(self.cfg, leverage=self.leverage, reward_weights=reward_weights)
            
            # Data processing
            base_df = df_base_ohlc.set_index('timestamp')
            feature_timeframes = {calc.timeframe for calc in self.strat_cfg.stateful_calculators}
            non_market_keys = {FeatureKeys.CONTEXT, FeatureKeys.PORTFOLIO_STATE, FeatureKeys.PRECOMPUTED_FEATURES}
            model_timeframes = set(
                k.value.split('_')[-1].replace('m','T').replace('h','H').replace('s','S').upper()
                for k in self.strat_cfg.lookback_periods.keys()
                if k not in non_market_keys
            )
            
            all_required_freqs = model_timeframes.union(feature_timeframes)
            self.timeframes = {}
            
            logger.info("Resampling data for all required timeframes...")
            for freq in all_required_freqs:
                if freq not in self.timeframes:
                    # Expanded aggregation rules to include precomputed features
                    agg_rules = {
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
                        'volume_delta': 'sum', 'vwap': 'last', 'trade_count': 'sum'
                    }
                    
                    for key in self.strat_cfg.precomputed_feature_keys:
                        agg_rules[key] = 'last'
                    
                    valid_agg_rules = {k: v for k, v in agg_rules.items() if k in base_df.columns}
                    df_resampled = base_df.resample(freq).agg(valid_agg_rules).ffill()
                    self.timeframes[freq] = df_resampled.dropna()
            
            self.base_timestamps = self.timeframes[self.cfg.base_bar_timeframe.value].index
            self.max_step = len(self.base_timestamps) - 2
            
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
            
            # Initialize stateful features
            self._initialize_stateful_features()
            
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
            
            logger.info("✅ FIXED environment initialized with immediate reward system for proper PPO training.")
            
        except Exception as e:
            logger.error(f"Failed to initialize fixed environment: {e}", exc_info=True)
            raise
    
    def _initialize_stateful_features(self):
        """Initialize stateful feature calculators from config (including VWAP distance)"""
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
    
    def _update_stateful_features(self, step_index: int):
        """Update stateful features including VWAP distance calculator"""
        current_timestamp = self.base_timestamps[step_index]
        
        for calc_cfg in self.strat_cfg.stateful_calculators:
            timeframe = calc_cfg.timeframe
            df_tf = self.timeframes[timeframe]
            
            try:
                latest_bar_idx = df_tf.index.get_indexer([current_timestamp], method='ffill')[0]
                latest_bar_timestamp = df_tf.index[latest_bar_idx]
            except (KeyError, IndexError):
                continue
            
            if latest_bar_timestamp > self.last_update_timestamps[timeframe]:
                self.last_update_timestamps[timeframe] = latest_bar_timestamp
                
                if calc_cfg.class_name == 'StatefulVWAPDistance':
                    price = df_tf[calc_cfg.source_col].iloc[latest_bar_idx]
                    volume = df_tf.get('volume', pd.Series([1.0] * len(df_tf))).iloc[latest_bar_idx]
                    self.feature_calculators[calc_cfg.name].update_vwap(price, volume)
                else:
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
    
    def step(self, action: np.ndarray):
        """
        FIXED: Step function with IMMEDIATE reward calculation for proper PPO training.
        
        KEY FIX: Now calculates immediate reward r_t from action a_t, fixing PPO credit assignment.
        """
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
            
            # Handle liquidation
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
                    # FIXED: Immediate liquidation penalty
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
            
            # NEW: Update consecutive inactive steps based on position change
            position_change = abs(action[0] - self.previous_action[0]) if self.previous_action is not None else abs(action[0])
            if position_change > 0.01:
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
            
            next_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step]
            next_unrealized_pnl = self.asset_held * (next_price - self.entry_price)
            next_portfolio_value = self.balance + next_unrealized_pnl
            
            # Check termination conditions
            next_drawdown_from_peak = (self.episode_peak_value - next_portfolio_value) / self.episode_peak_value
            terminated = next_drawdown_from_peak >= self.strat_cfg.max_drawdown_threshold
            
            # FIXED: Calculate IMMEDIATE reward for proper PPO training
            portfolio_state = {
                'drawdown': current_drawdown,
                'margin_ratio': margin_ratio
            }
            
            market_state = {
                'regime': self.market_regime,
                'price': current_price,
                'volatility': self.volatility_estimate
            }
            
            # CRITICAL FIX: Use immediate reward calculation
            reward, reward_components = self.reward_calculator.calculate_immediate_reward(
                prev_value=initial_portfolio_value,
                curr_value=next_portfolio_value,
                action=action,
                portfolio_state=portfolio_state,
                market_state=market_state,
                previous_action=self.previous_action,
                consecutive_inactive_steps=self.consecutive_inactive_steps
            )
            
            # Update tracking
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
                'immediate_reward_system': True  # FIXED: Indicator that immediate rewards are being used
            }
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error in fixed environment step: {e}", exc_info=True)
            observation = self._get_observation_sequence()
            info = {'portfolio_value': self.balance, 'error': True, 'leverage': self.leverage}
            return observation, -1.0, True, False, info
    
    def reset(self, seed=None, options=None):
        """Reset environment with immediate reward system"""
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
            
            self._initialize_stateful_features()
            self._warmup_features(warmup_period)
            
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
                self._update_stateful_features(step_idx)
                self._update_market_regime_and_volatility(step_idx)
                self.observation_history.append(self._get_single_step_observation(step_idx))
            
            observation = self._get_observation_sequence()
            
            info = {
                'balance': self.balance, 'asset_held': self.asset_held, 'portfolio_value': self.balance,
                'market_regime': self.market_regime, 'volatility_estimate': self.volatility_estimate,
                'leverage': self.leverage, 'optimized_warmup': self.precomputed_features is not None,
                'immediate_reward_system': True  # FIXED: Indicator that immediate rewards are being used
            }
            
            return observation, info
            
        except Exception as e:
            logger.error(f"Error resetting fixed environment: {e}", exc_info=True)
            raise
    
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
        """OPTIMIZED: Vectorized warmup using pre-calculated features."""
        try:
            if self.precomputed_features is None:
                logger.warning("No precomputed features available, falling back to stepwise warmup")
                self._warmup_features_stepwise(warmup_steps)
                return
            
            if 'timestamp' in self.precomputed_features.columns:
                precomputed_indexed = self.precomputed_features.set_index('timestamp')
            else:
                precomputed_indexed = self.precomputed_features
            
            warmup_timestamps = self.base_timestamps[:warmup_steps]
            
            for key in self.strat_cfg.context_feature_keys:
                if key in precomputed_indexed.columns:
                    feature_values = precomputed_indexed[key].reindex(
                        warmup_timestamps, method='ffill'
                    ).fillna(0.0).values
                    
                    self.feature_histories[key].clear()
                    for value in feature_values:
                        self.feature_histories[key].append(value)
                    
                    logger.debug(f"Populated {key} with {len(feature_values)} precomputed values")
                else:
                    logger.warning(f"Feature {key} not found in precomputed data, using default values")
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
        """Original step-by-step warmup method - maintained for backwards compatibility."""
        logger.info(f"Warming up stateful features for {warmup_steps} steps...")
        for i in tqdm(range(warmup_steps), desc="Warming up features", leave=False, ncols=100):
            self._update_stateful_features(i)
    
    def _get_current_context_features(self) -> np.ndarray:
        """Get current context features including VWAP distance"""
        final_vector = [self.feature_histories[key][-1] if self.feature_histories[key] else 0.0
                       for key in self.strat_cfg.context_feature_keys]
        return np.array(final_vector, dtype=np.float32)
    
    def _update_market_regime_and_volatility(self, step_index: int):
        """Update market regime and volatility estimates"""
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
    
    def _get_single_step_observation(self, step_index) -> dict:
        """Generate single step observation with enhanced features"""
        try:
            if self.normalizer is None: 
                return {}
            
            current_timestamp = self.base_timestamps[step_index]
            raw_obs = {}
            
            base_df = self.timeframes[self.cfg.base_bar_timeframe.value]
            current_price_idx = base_df.index.get_indexer([current_timestamp], method='ffill')[0]
            current_price = base_df['close'].iloc[current_price_idx]
            
            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value
                
                if key in [FeatureKeys.CONTEXT.value, FeatureKeys.PORTFOLIO_STATE.value, FeatureKeys.PRECOMPUTED_FEATURES.value]:
                    continue
                
                freq = key.split('_')[-1].replace('m','T').replace('h','H').replace('s', 'S').upper()
                df_tf = self.timeframes[freq]
                
                end_idx = df_tf.index.get_indexer([current_timestamp], method='ffill')[0]
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
            
            raw_obs[FeatureKeys.CONTEXT.value] = self._get_current_context_features()
            
            current_bar_idx = base_df.index.get_indexer([current_timestamp], method='ffill')[0]
            current_bar_features = base_df.iloc[current_bar_idx]
            
            precomputed_vector = current_bar_features[self.strat_cfg.precomputed_feature_keys].fillna(0.0).values.astype(np.float32)
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
        """Get comprehensive performance metrics with immediate reward analysis"""
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
            
            # Immediate reward system metrics
            immediate_reward_metrics = {}
            if self.reward_components_history:
                components_df = pd.DataFrame(self.reward_components_history)
                for component in components_df.columns:
                    if component != 'error':
                        immediate_reward_metrics[f'avg_{component}'] = components_df[component].mean()
            
            base_metrics = {
                'total_return': total_return, 'volatility': volatility, 'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio, 'win_rate': win_rate, 'total_trades': self.trade_count,
                'avg_reward': avg_reward, 'reward_volatility': reward_volatility,
                'consecutive_losses': self.consecutive_losses, 'final_portfolio_value': final_value,
                'leverage': self.leverage, 'reward_scaling_factor': self.reward_calculator.scaling_factor,
                'optimized_warmup': self.precomputed_features is not None,
                'immediate_reward_system': True,  # FIXED: Always True for this system
            }
            
            base_metrics.update(immediate_reward_metrics)
            
            return base_metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'leverage': self.leverage, 'immediate_reward_system': True}
--- END OF MODIFIED FILE Zero1-main/engine.py ---