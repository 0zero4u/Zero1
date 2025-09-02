"""
✅ ENHANCED Trading Environment with Reward Horizon System
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
        self.leverage = leverage  # Now configurable
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
    ✅ ENHANCED: Reward calculator with configurable reward horizon system.
    """
    
    def __init__(self, config, leverage: float = 10.0, reward_weights: Dict[str, float] = None):
        self.cfg = config
        self.strat_cfg = self.cfg.strategy
        self.leverage = leverage
        
        # ✅ NEW: Reward horizon parameters
        self.reward_horizon_steps = self.strat_cfg.reward_horizon_steps
        self.reward_horizon_decay = self.strat_cfg.reward_horizon_decay
        
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
        if self.reward_horizon_steps > 1:
            logger.info(f"Reward Horizon: {self.reward_horizon_steps} steps with decay {self.reward_horizon_decay}")

    def calculate_enhanced_reward(self, prev_value: float, future_values: List[float],
                                  action: np.ndarray, portfolio_state: Dict,
                                  market_state: Dict, previous_action: Optional[np.ndarray] = None) -> Tuple[float, Dict]:
        """
        ✅ ENHANCED: Multi-step reward calculation with configurable horizon.
        
        Args:
            prev_value: Portfolio value at the time of action.
            future_values: List of portfolio values for the next N steps.
            ... other args ...
            
        Returns:
            reward: Scalar reward value.
            components: Dictionary of reward components for analysis.
        """
        try:
            components = {}
            
            # Use the first future value for immediate return calculations
            curr_value = future_values[0]
            
            # --- 1. Multi-step Return Component (Core of Reward Horizon) ---
            if self.reward_horizon_steps > 1:
                weighted_return = 0.0
                # Calculate returns relative to the value at the time of action
                step_returns = [(val - prev_value) / max(prev_value, 1e-6) for val in future_values]
                
                for i, step_return in enumerate(step_returns):
                    weighted_return += step_return * (self.reward_horizon_decay ** i)
                
                # Average the return over the horizon for stability
                final_weighted_return = weighted_return / len(step_returns)
                
                components['horizon_info'] = {
                    'immediate_return': step_returns[0],
                    'final_return': step_returns[-1],
                    'weighted_return': final_weighted_return,
                    'using_multi_step': True
                }
            else:
                # Standard immediate reward
                final_weighted_return = (curr_value - prev_value) / max(prev_value, 1e-6)
                components['horizon_info'] = {
                    'immediate_return': final_weighted_return,
                    'final_return': final_weighted_return,
                    'weighted_return': final_weighted_return,
                    'using_multi_step': False
                }
            
            self.return_buffer.append(final_weighted_return)
            normalized_return = np.tanh(final_weighted_return * self.scaling_factor)
            components['base_return'] = normalized_return * self.weights['base_return']
            
            # --- 2. Other Reward Components (based on immediate state) ---
            
            # Risk-adjusted performance component (stable Sharpe-like)
            if len(self.return_buffer) >= 30:
                returns_array = np.array(list(self.return_buffer))
                excess_returns = returns_array - (self.leverage * 0.02/252) # Risk-free rate
                mean_excess = np.mean(excess_returns[-50:])
                volatility = np.std(returns_array[-50:]) + 1e-8
                risk_adjusted_score = np.tanh((mean_excess / volatility) * 5)
                components['risk_adjusted'] = risk_adjusted_score * self.weights['risk_adjusted']
            else:
                components['risk_adjusted'] = 0.0
                
            # Stability bonus (reward consistent performance)
            if len(self.return_buffer) >= 20:
                recent_returns = np.array(list(self.return_buffer)[-20:])
                stability_score = -np.std(recent_returns) / (abs(np.mean(recent_returns)) + 1e-6)
                components['stability'] = max(0, np.tanh(stability_score) * self.weights['stability'])
            else:
                components['stability'] = 0.0
                
            # Transaction cost penalty (progressive)
            if previous_action is not None:
                position_change = abs(action[0] - previous_action[0])
                position_size_factor = (action[1] + previous_action[1]) / 2
                tx_penalty = position_change * position_size_factor * self.cfg.transaction_fee_pct
                components['transaction_penalty'] = -tx_penalty * self.weights['transaction_penalty']
            else:
                components['transaction_penalty'] = 0.0
                
            # Progressive drawdown penalty
            current_drawdown = portfolio_state.get('drawdown', 0.0)
            if current_drawdown > 0.05:
                drawdown_severity = min(1.0, (current_drawdown - 0.05) / 0.20)
                penalty_factor = drawdown_severity ** 1.5
                components['drawdown_penalty'] = -penalty_factor * self.weights['drawdown_penalty']
            else:
                components['drawdown_penalty'] = 0.0
                
            # Position management penalty
            position_size = abs(action[1])
            if position_size > 0.8:
                size_penalty = (position_size - 0.8) * 2
                components['position_penalty'] = -size_penalty * self.weights['position_penalty']
            else:
                components['position_penalty'] = 0.0
                
            # Risk management bonus
            margin_ratio = portfolio_state.get('margin_ratio', 2.0)
            if margin_ratio > 0.5:
                risk_bonus = min(0.2, (margin_ratio - 0.5) * 0.4)
                components['risk_bonus'] = risk_bonus * self.weights['risk_bonus']
            else:
                components['risk_bonus'] = 0.0
                
            # Calculate total reward
            total_reward = sum(components.values())
            
            # Final bounds check
            total_reward = np.clip(total_reward, -5.0, 5.0)
            
            self.reward_history.append(total_reward)
            
            return total_reward, components
            
        except Exception as e:
            logger.error(f"Error in enhanced reward calculation: {e}", exc_info=True)
            return 0.0, {'error': -0.1}

class EnhancedHierarchicalTradingEnvironment(gym.Env):
    """
    ✅ ENHANCED Gymnasium-compliant RL environment with Reward Horizon System.
    """
    
    def __init__(self, df_base_ohlc: pd.DataFrame, normalizer: Normalizer, config=None, 
                 leverage: float = None, reward_weights: Dict[str, float] = None):
        super().__init__()
        self.cfg = config or SETTINGS
        self.strat_cfg = self.cfg.strategy
        self.normalizer = normalizer
        
        self.leverage = leverage if leverage is not None else self.strat_cfg.leverage
        
        logger.info("--- Initializing ENHANCED Hierarchical Trading Environment (Reward Horizon) ---")
        logger.info(f" -> Reward Horizon: {self.strat_cfg.reward_horizon_steps} steps")
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
            
            # ✅ NEW: Adjust max_step to leave a buffer for reward horizon calculation
            self.max_step = len(self.base_timestamps) - self.strat_cfg.reward_horizon_steps - 2
            
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
            
            logger.info("Enhanced environment initialized successfully with Reward Horizon.")
            
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
                    self.feature_histories[key].append(values.get(key, 1.0 if 'dist' in key else 0.0))
            else:
                if len(calc_cfg.output_keys) == 1:
                    self.feature_histories[calc_cfg.output_keys[0]].append(values)

    def _get_current_context_features(self) -> np.ndarray:
        final_vector = [self.feature_histories[key][-1] if self.feature_histories[key] else 0.0 for key in self.strat_cfg.context_feature_keys]
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
            info = {'balance': self.balance, 'asset_held': self.asset_held, 'portfolio_value': self.balance, 'market_regime': self.market_regime, 'volatility_estimate': self.volatility_estimate, 'leverage': self.leverage}
            
            return observation, info
            
        except Exception as e:
            logger.error(f"Error resetting enhanced environment: {e}", exc_info=True)
            raise

    def step(self, action: np.ndarray):
        """✅ ENHANCED step function with Reward Horizon logic."""
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
                # Progressive liquidation logic...
                liquidation_factor = min(1.0, (self.strat_cfg.maintenance_margin_rate - margin_ratio) * 5)
                liquidation_amount = self.asset_held * liquidation_factor
                self.asset_held -= liquidation_amount
                self.balance += liquidation_amount * current_price - abs(liquidation_amount * current_price) * (self.cfg.transaction_fee_pct + 0.001)
                self.used_margin = (abs(self.asset_held) * current_price) / self.leverage
                
                if liquidation_factor >= 1.0:
                    reward = -2.0
                    terminated = True
                    self.current_step += 1
                    truncated = self.current_step >= self.max_step
                    self.observation_history.append(self._get_single_step_observation(self.current_step))
                    observation = self._get_observation_sequence()
                    info = {'portfolio_value': self.balance, 'margin_ratio': 0.0, 'liquidation': True, 'liquidation_factor': liquidation_factor, 'leverage': self.leverage}
                    return observation, reward, terminated, truncated, info
            
            action_signal = np.clip(action[0], -1.0, 1.0)
            action_size = np.clip(action[1], 0.0, 1.0)
            dynamic_limit = self.risk_manager.calculate_dynamic_position_limit(self.volatility_estimate, initial_portfolio_value, self.market_regime)
            effective_size = action_size * dynamic_limit
            target_notional = initial_portfolio_value * action_signal * effective_size
            target_asset_quantity = target_notional / current_price if current_price > 1e-8 else 0
            
            max_allowable_margin = initial_portfolio_value * self.strat_cfg.max_margin_allocation_pct
            required_margin_for_target = (abs(target_asset_quantity) * current_price) / self.leverage
            if required_margin_for_target > max_allowable_margin:
                target_asset_quantity = ((max_allowable_margin * self.leverage) / current_price * np.sign(target_asset_quantity)) if current_price > 1e-8 else 0
            
            trade_quantity = target_asset_quantity - self.asset_held
            trade_notional = abs(trade_quantity) * current_price
            total_cost = trade_notional * (self.cfg.transaction_fee_pct + self.cfg.slippage_pct)
            
            self.balance += unrealized_pnl - total_cost
            self.asset_held = target_asset_quantity
            self.used_margin = (abs(self.asset_held) * current_price) / self.leverage
            if abs(trade_quantity) > 1e-8:
                self.entry_price = current_price
                self.trade_count += 1
            
            self.current_step += 1
            
            # ✅ NEW: Calculate future portfolio values for reward horizon
            future_values = []
            for i in range(self.strat_cfg.reward_horizon_steps):
                future_step_index = self.current_step + i
                if future_step_index > self.max_step + self.strat_cfg.reward_horizon_steps:
                    break # Stop if we are at the end of available data
                
                future_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[future_step_index]
                future_unrealized_pnl = self.asset_held * (future_price - self.entry_price)
                future_portfolio_value = self.balance + future_unrealized_pnl
                future_values.append(future_portfolio_value)
            
            # If horizon calculation was cut short, pad with the last value
            if len(future_values) < self.strat_cfg.reward_horizon_steps:
                padding = [future_values[-1]] * (self.strat_cfg.reward_horizon_steps - len(future_values))
                future_values.extend(padding)

            next_portfolio_value = future_values[0] # The value at the next immediate step
            
            terminated = next_portfolio_value <= initial_portfolio_value * 0.5
            truncated = self.current_step >= self.max_step
            
            portfolio_state = {'drawdown': (self.episode_peak_value - next_portfolio_value) / self.episode_peak_value, 'margin_ratio': margin_ratio, 'portfolio_value': next_portfolio_value, 'volatility': self.volatility_estimate}
            market_state = {'regime': self.market_regime, 'price': self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step], 'volatility': self.volatility_estimate}
            
            # ✅ MODIFIED: Use multi-step reward calculation
            reward, reward_components = self.reward_calculator.calculate_enhanced_reward(
                initial_portfolio_value, future_values, action,
                portfolio_state, market_state, self.previous_action
            )
            
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
                'drawdown': portfolio_state['drawdown'], 'volatility': self.volatility_estimate,
                'unrealized_pnl': self.asset_held * (self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step] - self.entry_price),
                'margin_ratio': margin_ratio, 'used_margin': self.used_margin,
                'market_regime': self.market_regime, 'consecutive_losses': self.consecutive_losses,
                'trade_count': self.trade_count, 'win_rate': self.winning_trades / max(self.trade_count, 1),
                'reward_components': reward_components, 'dynamic_position_limit': dynamic_limit,
                'transaction_cost': total_cost, 'leverage': self.leverage,
                'reward_scaling_factor': self.reward_calculator.scaling_factor,
                'reward_horizon': reward_components.get('horizon_info', {}) # ✅ NEW: Add horizon info
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
                    if len(window) < lookback: window = np.concatenate([np.repeat(window[0:1], lookback - len(window), axis=0), window], axis=0)
                    raw_obs[key] = window
            
            raw_obs[FeatureKeys.CONTEXT.value] = self._get_current_context_features()
            
            current_bar_features = base_df.loc[base_df.index.get_loc(current_timestamp, method='ffill')]
            precomputed_vector = current_bar_features[self.strat_cfg.precomputed_feature_keys].fillna(0.0).values.astype(np.float32)
            raw_obs[FeatureKeys.PRECOMPUTED_FEATURES.value] = precomputed_vector
            
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            portfolio_value = self.balance + unrealized_pnl
            current_notional = self.asset_held * current_price
            normalized_position = np.clip(current_notional / (portfolio_value + 1e-9), -1.0, 1.0)
            normalized_pnl = np.tanh(unrealized_pnl / (self.used_margin + 1e-9))
            position_notional = abs(self.asset_held) * current_price
            margin_ratio = np.clip((self.used_margin + unrealized_pnl) / position_notional, 0, 2.0) if position_notional > 0 else 2.0
            
            regime_map = {"HIGH_VOLATILITY": 1.0, "TRENDING_UP": 0.8, "TRENDING_DOWN": -0.8, "LOW_VOLATILITY": 0.6, "SIDEWAYS": 0.0}
            regime_encoding = regime_map.get(self.market_regime, -0.2)
            vol_adjusted_signal = np.tanh(self.volatility_estimate * 10)
            
            raw_obs[FeatureKeys.PORTFOLIO_STATE.value] = np.array([normalized_position, normalized_pnl, margin_ratio, regime_encoding, vol_adjusted_signal], dtype=np.float32)
            
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
            return {key: np.stack([obs[key] for obs in self.observation_history]) for key in self.observation_space.spaces.keys()}
        except Exception as e:
            logger.error(f"Error getting observation sequence: {e}")
            return {key: np.zeros(self.observation_space.spaces[key].shape, dtype=np.float32) for key in self.observation_space.spaces.keys()}

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
            sharpe_ratio = (total_return - 0.02) / volatility if volatility > 0 else 0.0
            win_rate = self.winning_trades / max(self.trade_count, 1)
            
            return {
                'total_return': total_return, 'volatility': volatility, 'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio, 'win_rate': win_rate, 'total_trades': self.trade_count,
                'avg_reward': np.mean(self.step_rewards) if self.step_rewards else 0.0,
                'reward_volatility': np.std(self.step_rewards) if len(self.step_rewards) > 1 else 0.0,
                'final_portfolio_value': final_value, 'leverage': self.leverage,
                'reward_scaling_factor': self.reward_calculator.scaling_factor
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'leverage': self.leverage}

# --- CONVENIENCE FUNCTIONS ---
def create_bars_from_trades(period: str) -> pd.DataFrame:
    try:
        from processor import EnhancedDataProcessor
        return EnhancedDataProcessor().create_enhanced_bars_from_trades(period)
    except Exception as e:
        logger.error(f"Error creating bars from trades: {e}")
        raise

def create_trading_environment(df_base_ohlc: pd.DataFrame, normalizer: Normalizer, 
                              config=None, leverage: float = None, reward_weights: Dict[str, float] = None):
    return EnhancedHierarchicalTradingEnvironment(
        df_base_ohlc=df_base_ohlc, normalizer=normalizer, config=config,
        leverage=leverage, reward_weights=reward_weights
    )

if __name__ == "__main__":
    try:
        logger.info("Testing enhanced trading environment with reward horizon...")
        
        test_leverages = [5.0, 10.0, 25.0]
        for leverage in test_leverages:
            # Create config with 3-step horizon for testing
            test_config = create_config(strategy={'reward_horizon_steps': 3, 'reward_horizon_decay': 0.9})
            reward_calc = AdvancedRewardCalculator(test_config, leverage=leverage)
            
            # Test with future values representing a 1% gain then a 0.5% loss then a 2% gain
            future_vals = [1010000, 1004950, 1025049]
            test_reward, components = reward_calc.calculate_enhanced_reward(
                prev_value=1000000,
                future_values=future_vals,
                action=np.array([0.5, 0.3]),
                portfolio_state={'drawdown': 0.05, 'margin_ratio': 0.8},
                market_state={'regime': 'TRENDING_UP', 'volatility': 0.02}
            )
            
            logger.info(f"Leverage {leverage}x: scaling_factor={reward_calc.scaling_factor:.2f}, reward={test_reward:.4f}")
            logger.info(f" -> Horizon components: {components.get('horizon_info')}")
        
        logger.info("✅ Enhanced trading environment test completed!")
        
    except Exception as e:
        logger.error(f"Enhanced environment test failed: {e}", exc_info=True)
