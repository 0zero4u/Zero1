# REFINED: engine.py with Path-Aware Decayed Reward System

"""
REFINEMENT: Enhanced Trading Environment with Path-Aware Reward Horizon System

Key improvement: The reward system now evaluates the entire trajectory of returns
over the horizon period, not just the final outcome. This provides a more truthful,
risk-aware reward signal that considers the volatility and drawdown of the journey.
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
    StatefulVWAPDistance,  # REFINEMENT: New VWAP distance calculator
)

# Map calculator names from config to their classes for dynamic instantiation
STATEFUL_CALCULATOR_MAP = {
    'StatefulBBWPercentRank': StatefulBBWPercentRank,
    'StatefulPriceDistanceMA': StatefulPriceDistanceMA,
    'StatefulSRDistances': StatefulSRDistances,
    'StatefulVWAPDistance': StatefulVWAPDistance,  # REFINEMENT: Added VWAP calculator
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

class AdvancedRewardCalculator:
    """
    REFINED: Enhanced reward calculator with path-aware evaluation capabilities
    
    Key improvement: Can now process path-aware rewards that consider the entire
    trajectory of returns over the horizon period, providing more stable and
    risk-aware learning signals.
    """
    
    def __init__(self, config, leverage: float = 10.0, reward_weights: Dict[str, float] = None):
        self.cfg = config
        self.leverage = leverage
        self.return_scaler = RobustScaler()
        self.reward_history = deque(maxlen=1000)
        self.return_buffer = deque(maxlen=500)
        self.volatility_buffer = deque(maxlen=100)
        self.transaction_cost_buffer = deque(maxlen=50)
        
        # Reward weights are now tunable hyperparameters
        self.weights = reward_weights or {
            'base_return': 1.0,
            'risk_adjusted': 0.3,
            'stability': 0.2,
            'transaction_penalty': -0.1,
            'drawdown_penalty': -0.4,
            'position_penalty': -0.05,
            'risk_bonus': 0.15
        }
        
        # Dynamic scaling factor based on leverage
        self.scaling_factor = 200.0 / self.leverage
        
        logger.info(f"Reward calculator initialized with leverage {self.leverage}x, scaling_factor {self.scaling_factor:.2f}")

    def calculate_enhanced_reward(self, prev_value: float, curr_value: float,
                                action: np.ndarray, portfolio_state: Dict,
                                market_state: Dict, previous_action: np.ndarray = None) -> Tuple[float, Dict]:
        """
        Enhanced reward calculation with dynamic scaling and tunable weights
        
        Returns:
            reward: Properly scaled reward value [-5, 5]
            components: Dictionary of reward components for analysis
        """
        try:
            components = {}
            
            # Base return component with dynamic scaling
            period_return = (curr_value - prev_value) / max(prev_value, 1e-6)
            self.return_buffer.append(period_return)
            
            # Dynamic scaling based on leverage
            normalized_return = np.tanh(period_return * self.scaling_factor)
            components['base_return'] = normalized_return * self.weights['base_return']
            
            # Risk-adjusted performance component (stable Sharpe-like)
            if len(self.return_buffer) >= 30:
                returns_array = np.array(list(self.return_buffer))
                excess_returns = returns_array - (self.leverage * 0.02/252)
                
                # Use robust statistics
                mean_excess = np.mean(excess_returns[-50:])
                volatility = np.std(returns_array[-50:]) + 1e-8
                
                # Stable Sharpe-like metric
                risk_adjusted_score = mean_excess / volatility
                risk_adjusted_score = np.tanh(risk_adjusted_score * 5)
                components['risk_adjusted'] = risk_adjusted_score * self.weights['risk_adjusted']
            else:
                components['risk_adjusted'] = 0.0
            
            # Stability bonus (reward consistent performance)
            if len(self.return_buffer) >= 20:
                recent_returns = np.array(list(self.return_buffer)[-20:])
                stability_score = -np.std(recent_returns) / (abs(np.mean(recent_returns)) + 1e-6)
                stability_score = np.tanh(stability_score)
                components['stability'] = max(0, stability_score * self.weights['stability'])
            else:
                components['stability'] = 0.0
            
            # Transaction cost penalty (progressive)
            position_change = 0.0
            if previous_action is not None:
                position_change = abs(action[0] - previous_action[0])
                position_size_factor = (action[1] + previous_action[1]) / 2
                
                # Progressive penalty based on position size and change
                tx_penalty = position_change * position_size_factor * self.cfg.transaction_fee_pct
                components['transaction_penalty'] = -tx_penalty * self.weights['transaction_penalty']
            else:
                components['transaction_penalty'] = 0.0
            
            # Progressive drawdown penalty
            current_drawdown = portfolio_state.get('drawdown', 0.0)
            if current_drawdown > 0.05:
                # Progressive penalty that increases smoothly
                drawdown_severity = (current_drawdown - 0.05) / 0.20
                drawdown_severity = min(1.0, drawdown_severity)
                
                # Use quadratic penalty for severe drawdowns
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
            
            # Final bounds check and scaling
            total_reward = np.clip(total_reward, -5.0, 5.0)
            
            # Update history
            self.reward_history.append(total_reward)
            
            return total_reward, components
            
        except Exception as e:
            logger.error(f"Error in enhanced reward calculation: {e}")
            return 0.0, {'error': -0.1}

    def calculate_path_aware_reward(self, portfolio_path: List[float], action: np.ndarray,
                                  portfolio_state: Dict, market_state: Dict,
                                  previous_action: np.ndarray = None,
                                  decay_factor: float = 0.95) -> Tuple[float, Dict]:
        """
        REFINEMENT: Calculate path-aware reward that considers the entire trajectory
        
        This method evaluates not just the final outcome, but the entire journey of
        returns over the horizon period, applying exponential decay to weight
        immediate returns more heavily than distant ones.
        
        Args:
            portfolio_path: List of portfolio values over the horizon period
            action: The action being evaluated
            portfolio_state: Portfolio state information
            market_state: Market state information
            previous_action: Previous action taken
            decay_factor: Decay rate for weighting returns (0.7-1.0)
            
        Returns:
            reward: Path-aware reward value
            components: Detailed breakdown of reward components
        """
        try:
            if len(portfolio_path) < 2:
                # Fallback to standard reward calculation
                return self.calculate_enhanced_reward(
                    portfolio_path[0] if portfolio_path else 1000000.0,
                    portfolio_path[-1] if portfolio_path else 1000000.0,
                    action, portfolio_state, market_state, previous_action
                )
            
            # Calculate step-by-step returns over the path
            path_returns = []
            for i in range(1, len(portfolio_path)):
                period_return = (portfolio_path[i] - portfolio_path[i-1]) / max(portfolio_path[i-1], 1e-6)
                path_returns.append(period_return)
            
            if not path_returns:
                return 0.0, {'error': 'No path returns'}
            
            # Create decay weights (most recent gets weight 1.0, earlier returns get exponentially less)
            num_steps = len(path_returns)
            decay_weights = np.array([decay_factor ** (num_steps - 1 - i) for i in range(num_steps)])
            
            # Normalize weights so they sum to the number of steps (preserves scale)
            decay_weights = decay_weights * (num_steps / np.sum(decay_weights))
            
            # Calculate weighted returns
            weighted_returns = np.array(path_returns) * decay_weights
            
            # Compute equivalent final value that represents the path-aware performance
            # This creates a "summary value" that captures both magnitude and journey quality
            path_return_sum = np.sum(weighted_returns)
            initial_value = portfolio_path[0]
            equivalent_final_value = initial_value * (1 + path_return_sum)
            
            # Calculate path-specific metrics for additional reward components
            path_volatility = np.std(path_returns) if len(path_returns) > 1 else 0.0
            path_consistency = 1.0 / (1.0 + path_volatility * 10)  # Reward smooth paths
            
            # Use the enhanced reward calculator with the equivalent final value
            base_reward, components = self.calculate_enhanced_reward(
                initial_value, equivalent_final_value,
                action, portfolio_state, market_state, previous_action
            )
            
            # Add path-specific bonuses/penalties
            path_smoothness_bonus = path_consistency * 0.1 * self.weights.get('stability', 0.2)
            components['path_smoothness'] = path_smoothness_bonus
            
            # Penalize paths with excessive drawdown during the journey
            if len(portfolio_path) > 2:
                path_array = np.array(portfolio_path)
                path_peak = np.maximum.accumulate(path_array)
                path_drawdowns = (path_peak - path_array) / (path_peak + 1e-9)
                max_interim_drawdown = np.max(path_drawdowns)
                
                if max_interim_drawdown > 0.1:  # More than 10% interim drawdown
                    interim_penalty = -(max_interim_drawdown - 0.1) * 2.0
                    components['interim_drawdown_penalty'] = interim_penalty
                else:
                    components['interim_drawdown_penalty'] = 0.0
            else:
                components['interim_drawdown_penalty'] = 0.0
            
            # Calculate final path-aware reward
            path_aware_reward = base_reward + path_smoothness_bonus + components['interim_drawdown_penalty']
            
            # Ensure bounded output
            path_aware_reward = np.clip(path_aware_reward, -5.0, 5.0)
            
            # Add path analysis to components
            components['path_return_sum'] = path_return_sum
            components['path_volatility'] = path_volatility
            components['path_consistency'] = path_consistency
            components['num_path_steps'] = num_steps
            components['decay_factor_used'] = decay_factor
            
            return path_aware_reward, components
            
        except Exception as e:
            logger.error(f"Error in path-aware reward calculation: {e}")
            # Fall back to standard reward
            return self.calculate_enhanced_reward(
                portfolio_path[0] if portfolio_path else 1000000.0,
                portfolio_path[-1] if portfolio_path else 1000000.0,
                action, portfolio_state, market_state, previous_action
            )

class EnhancedHierarchicalTradingEnvironment(gym.Env):
    """
    REFINED: Enhanced Gymnasium-compliant RL environment with Path-Aware Decayed Reward System
    
    Key improvements:
    - Path-aware reward horizon that evaluates the entire trajectory
    - Proper handling of StatefulVWAPDistance calculator
    - Enhanced reward calculation considering journey quality, not just destination
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
        
        logger.info("--- Initializing REFINED Path-Aware Trading Environment ---")
        logger.info(f" -> Path-Aware Rewards: {'ON' if self.strat_cfg.use_path_aware_rewards else 'OFF'}")
        logger.info(f" -> Advanced Risk Management: ON")
        logger.info(f" -> Dynamic Reward Scaling: ON")
        logger.info(f" -> Regime Detection: ON")
        logger.info(f" -> OPTIMIZED: Vectorized warmup: {'ON' if precomputed_features is not None else 'OFF'}")
        
        # REFINEMENT: Initialize Path-Aware Reward Horizon System
        self.reward_horizon_steps = self.strat_cfg.reward_horizon_steps
        self.reward_horizon_decay = self.strat_cfg.reward_horizon_decay
        self.use_path_aware_rewards = self.strat_cfg.use_path_aware_rewards
        
        # REFINEMENT: Enhanced pending rewards structure for path tracking
        self.pending_rewards = deque(maxlen=self.reward_horizon_steps)
        
        logger.info(f" -> REFINED REWARD HORIZON: {self.reward_horizon_steps} steps | Decay: {self.reward_horizon_decay}")
        logger.info(f" -> Path-Aware Evaluation: {self.use_path_aware_rewards}")
        logger.info(f" -> Leverage: {self.leverage}x | Maintenance Margin: {self.strat_cfg.maintenance_margin_rate:.2%}")
        
        try:
            self.risk_manager = EnhancedRiskManager(self.cfg, leverage=self.leverage)
            self.reward_calculator = AdvancedRewardCalculator(self.cfg, leverage=self.leverage, reward_weights=reward_weights)
            
            # Data processing
            base_df = df_base_ohlc.set_index('timestamp')
            feature_timeframes = {calc.timeframe for calc in self.strat_cfg.stateful_calculators}
            
            # --- THIS IS THE FIX ---
            # Define special keys that are NOT time frequencies
            non_market_keys = {FeatureKeys.CONTEXT, FeatureKeys.PORTFOLIO_STATE, FeatureKeys.PRECOMPUTED_FEATURES}
            
            # Generate model timeframes ONLY from market data keys
            model_timeframes = set(
                k.value.split('_')[-1].replace('m','T').replace('h','H').replace('s','S').upper()
                for k in self.strat_cfg.lookback_periods.keys()
                if k not in non_market_keys # Filter out the non-time-based keys
            )
            # --- END OF FIX ---
            
            all_required_freqs = model_timeframes.union(feature_timeframes)
            
            self.timeframes = {}
            logger.info("Resampling data for all required timeframes...")
            for freq in all_required_freqs:
                if freq not in self.timeframes:
                    agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'volume_delta': 'sum', 'vwap': 'last'}
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
                if key_str.startswith('ohlcv_'): shape = (seq_len, lookback, 5)
                elif key_str.startswith('ohlc_'): shape = (seq_len, lookback, 4)
                elif key in [FeatureKeys.PORTFOLIO_STATE, FeatureKeys.CONTEXT, FeatureKeys.PRECOMPUTED_FEATURES]: shape = (seq_len, lookback)
                else: shape = (seq_len, lookback)
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
            self.episode_peak_value = 0.0
            self.market_regime = "UNCERTAIN"
            self.volatility_estimate = 0.02
            self.step_rewards = []
            self.reward_components_history = []
            self.trade_count = 0
            self.winning_trades = 0
            
            logger.info("REFINED environment initialized successfully with path-aware reward horizon system.")
            
        except Exception as e:
            logger.error(f"Failed to initialize refined environment: {e}", exc_info=True)
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
                latest_bar_idx = df_tf.index.get_loc(current_timestamp, method='ffill')
                latest_bar_timestamp = df_tf.index[latest_bar_idx]
            except KeyError:
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
        REFINEMENT: Step function with Path-Aware Decayed Reward System
        
        The key improvement is in the reward calculation logic, which now considers
        the entire trajectory of returns over the horizon period, providing more
        stable and risk-aware learning signals.
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
            
            pending_info = {
                'initial_portfolio_value': initial_portfolio_value,
                'action': action.copy(),
                'previous_action': self.previous_action.copy() if self.previous_action is not None else None,
                'portfolio_state_at_action': {
                    'drawdown': current_drawdown, 
                    'margin_ratio': margin_ratio
                },
                'market_state_at_action': {
                    'regime': self.market_regime, 
                    'price': current_price, 
                    'volatility': self.volatility_estimate
                },
                'portfolio_path': [initial_portfolio_value]
            }
            
            for pending in self.pending_rewards:
                pending['portfolio_path'].append(initial_portfolio_value)
            
            self.pending_rewards.append(pending_info)
            
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
            
            self.current_step += 1
            truncated = self.current_step >= self.max_step
            
            next_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step]
            next_unrealized_pnl = self.asset_held * (next_price - self.entry_price)
            next_portfolio_value = self.balance + next_unrealized_pnl
            
            terminated = next_portfolio_value <= initial_portfolio_value * 0.5
            
            reward = 0.0
            reward_components = {}
            horizon_info = {}
            
            if len(self.pending_rewards) == self.pending_rewards.maxlen:
                past_info = self.pending_rewards.popleft()
                
                past_info['portfolio_path'].append(next_portfolio_value)
                
                if self.use_path_aware_rewards and len(past_info['portfolio_path']) > 2:
                    reward, reward_components = self.reward_calculator.calculate_path_aware_reward(
                        portfolio_path=past_info['portfolio_path'],
                        action=past_info['action'],
                        portfolio_state=past_info['portfolio_state_at_action'],
                        market_state=past_info['market_state_at_action'],
                        previous_action=past_info['previous_action'],
                        decay_factor=self.reward_horizon_decay
                    )
                    
                    horizon_info = {
                        'using_path_aware': True,
                        'path_length': len(past_info['portfolio_path']),
                        'decay_factor': self.reward_horizon_decay
                    }
                else:
                    reward, reward_components = self.reward_calculator.calculate_enhanced_reward(
                        prev_value=past_info['portfolio_path'][0],
                        curr_value=past_info['portfolio_path'][-1],
                        action=past_info['action'],
                        portfolio_state=past_info['portfolio_state_at_action'],
                        market_state=past_info['market_state_at_action'],
                        previous_action=past_info['previous_action']
                    )
                    
                    horizon_info = {'using_path_aware': False}
            
            if terminated or truncated:
                final_reward = reward
                
                while self.pending_rewards:
                    past_info = self.pending_rewards.popleft()
                    past_info['portfolio_path'].append(next_portfolio_value)
                    
                    if self.use_path_aware_rewards and len(past_info['portfolio_path']) > 2:
                        term_reward, _ = self.reward_calculator.calculate_path_aware_reward(
                            portfolio_path=past_info['portfolio_path'],
                            action=past_info['action'],
                            portfolio_state=past_info['portfolio_state_at_action'],
                            market_state=past_info['market_state_at_action'],
                            previous_action=past_info['previous_action'],
                            decay_factor=self.reward_horizon_decay
                        )
                    else:
                        term_reward, _ = self.reward_calculator.calculate_enhanced_reward(
                            prev_value=past_info['portfolio_path'][0],
                            curr_value=past_info['portfolio_path'][-1],
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
                'reward_horizon': {
                    'steps': self.reward_horizon_steps, 
                    'decay_factor': self.reward_horizon_decay,
                    'use_path_aware': self.use_path_aware_rewards,
                    'is_calculating': len(self.pending_rewards) == self.pending_rewards.maxlen, 
                    **horizon_info
                }
            }
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error in refined environment step: {e}", exc_info=True)
            observation = self._get_observation_sequence()
            info = {'portfolio_value': self.balance, 'error': True, 'leverage': self.leverage}
            return observation, -1.0, True, False, info

    def reset(self, seed=None, options=None):
        """Reset environment with path-aware reward system initialization"""
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
                'leverage': self.leverage, 'optimized_warmup': self.precomputed_features is not None,
                'path_aware_rewards': self.use_path_aware_rewards
            }
            
            return observation, info
            
        except Exception as e:
            logger.error(f"Error resetting refined environment: {e}", exc_info=True)
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
        """Get observation sequence for model input"""
        try:
            return {key: np.stack([obs[key] for obs in self.observation_history])
                   for key in self.observation_space.spaces.keys()}
        except Exception as e:
            logger.error(f"Error getting observation sequence: {e}")
            return {key: np.zeros(self.observation_space.spaces[key].shape, dtype=np.float32)
                   for key in self.observation_space.spaces.keys()}

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics including path-aware reward analysis"""
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
            
            path_aware_metrics = {}
            if self.reward_components_history:
                components_df = pd.DataFrame(self.reward_components_history)
                if 'path_smoothness' in components_df.columns:
                    path_aware_metrics['avg_path_smoothness'] = components_df['path_smoothness'].mean()
                if 'interim_drawdown_penalty' in components_df.columns:
                    path_aware_metrics['avg_interim_drawdown_penalty'] = components_df['interim_drawdown_penalty'].mean()
                if 'path_volatility' in components_df.columns:
                    path_aware_metrics['avg_path_volatility'] = components_df['path_volatility'].mean()
            
            base_metrics = {
                'total_return': total_return, 'volatility': volatility, 'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio, 'win_rate': win_rate, 'total_trades': self.trade_count,
                'avg_reward': avg_reward, 'reward_volatility': reward_volatility,
                'consecutive_losses': self.consecutive_losses, 'final_portfolio_value': final_value,
                'leverage': self.leverage, 'reward_scaling_factor': self.reward_calculator.scaling_factor,
                'optimized_warmup': self.precomputed_features is not None,
                'path_aware_rewards_enabled': self.use_path_aware_rewards,
                'reward_horizon_steps': self.reward_horizon_steps,
                'reward_horizon_decay': self.reward_horizon_decay
            }
            
            base_metrics.update(path_aware_metrics)
            
            return base_metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'leverage': self.leverage}
