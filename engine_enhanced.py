# Enhanced engine.py with Reward Horizon System

"""
Enhanced Trading Environment with Configurable Reward Horizon

Key Enhancement: Multi-step reward calculation that encourages long-term strategic thinking
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

logger = logging.getLogger(__name__)

class EnhancedRewardCalculator:
    """
    ✅ ENHANCED: Reward calculator with configurable reward horizon system
    
    Major enhancements:
    1. Multi-step reward calculation for long-term strategic thinking
    2. Configurable horizon with decay factors
    3. Maintains all existing reward components
    """
    
    def __init__(self, config, leverage: float = 10.0, reward_weights: Dict[str, float] = None):
        self.cfg = config
        self.leverage = leverage
        self.return_scaler = RobustScaler()
        self.reward_history = deque(maxlen=1000)
        self.return_buffer = deque(maxlen=500)
        self.volatility_buffer = deque(maxlen=100)
        self.transaction_cost_buffer = deque(maxlen=50)
        
        # ✅ NEW: Reward horizon configuration
        self.reward_horizon_steps = self.cfg.strategy.reward_horizon_steps
        self.reward_decay_factor = self.cfg.strategy.reward_horizon_decay
        
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
        
        logger.info(f"Reward calculator initialized:")
        logger.info(f" - Leverage: {self.leverage}x")
        logger.info(f" - Reward Horizon: {self.reward_horizon_steps} steps ({self.reward_horizon_steps * 20}s)")
        logger.info(f" - Decay Factor: {self.reward_decay_factor}")
        logger.info(f" - Scaling Factor: {self.scaling_factor:.2f}")
    
    def calculate_multi_step_reward(self, prev_value: float, future_values: List[float], 
                                  action: np.ndarray, portfolio_state: Dict,
                                  market_state: Dict, previous_action: np.ndarray = None) -> Tuple[float, Dict]:
        """
        ✅ NEW: Multi-step reward calculation with configurable horizon
        
        Args:
            prev_value: Portfolio value at current step
            future_values: List of portfolio values for the next N steps (where N = reward_horizon_steps)
            action: Current action taken
            portfolio_state: Current portfolio state
            market_state: Current market conditions
            previous_action: Previous action for transaction cost calculation
            
        Returns:
            reward: Scalar reward value
            components: Dictionary of reward components for analysis
        """
        try:
            components = {}
            
            if not future_values or len(future_values) == 0:
                # Fallback to immediate reward if no future data
                return self.calculate_enhanced_reward(
                    prev_value, prev_value, action, portfolio_state, market_state, previous_action
                )
            
            # ✅ CORE ENHANCEMENT: Multi-step return calculation
            total_weighted_return = 0.0
            total_weight = 0.0
            
            for step, future_value in enumerate(future_values, 1):
                # Calculate return for this step
                step_return = (future_value - prev_value) / max(prev_value, 1e-6)
                
                # Apply decay factor: more recent steps have higher weight
                weight = (self.reward_decay_factor ** (step - 1))
                
                total_weighted_return += step_return * weight
                total_weight += weight
            
            # Normalize by total weight to maintain scale consistency
            if total_weight > 0:
                avg_weighted_return = total_weighted_return / total_weight
            else:
                avg_weighted_return = 0.0
            
            # Store for analysis
            self.return_buffer.append(avg_weighted_return)
            
            # ✅ ENHANCED: Base return component with multi-step horizon
            normalized_return = np.tanh(avg_weighted_return * self.scaling_factor)
            components['base_return'] = normalized_return * self.weights['base_return']
            
            # Store horizon information for analysis
            components['horizon_info'] = {
                'steps': len(future_values),
                'weighted_return': avg_weighted_return,
                'immediate_return': (future_values[0] - prev_value) / max(prev_value, 1e-6) if future_values else 0.0,
                'final_return': (future_values[-1] - prev_value) / max(prev_value, 1e-6) if future_values else 0.0
            }
            
            # 2. Risk-adjusted performance (using multi-step returns)
            if len(self.return_buffer) >= 30:
                returns_array = np.array(list(self.return_buffer))
                excess_returns = returns_array - (self.leverage * 0.02/252)
                mean_excess = np.mean(excess_returns[-50:])
                volatility = np.std(returns_array[-50:]) + 1e-8
                
                risk_adjusted_score = mean_excess / volatility
                risk_adjusted_score = np.tanh(risk_adjusted_score * 5)
                components['risk_adjusted'] = risk_adjusted_score * self.weights['risk_adjusted']
            else:
                components['risk_adjusted'] = 0.0
            
            # 3. Stability bonus (reward consistent performance over horizon)
            if len(self.return_buffer) >= 20:
                recent_returns = np.array(list(self.return_buffer)[-20:])
                stability_score = -np.std(recent_returns) / (abs(np.mean(recent_returns)) + 1e-6)
                stability_score = np.tanh(stability_score)
                components['stability'] = max(0, stability_score * self.weights['stability'])
            else:
                components['stability'] = 0.0
            
            # 4. Transaction cost penalty (unchanged)
            position_change = 0.0
            if previous_action is not None:
                position_change = abs(action[0] - previous_action[0])
                position_size_factor = (action[1] + previous_action[1]) / 2
                tx_penalty = position_change * position_size_factor * self.cfg.transaction_fee_pct
                components['transaction_penalty'] = -tx_penalty * self.weights['transaction_penalty']
            else:
                components['transaction_penalty'] = 0.0
            
            # 5. Progressive drawdown penalty (unchanged)
            current_drawdown = portfolio_state.get('drawdown', 0.0)
            if current_drawdown > 0.05:
                drawdown_severity = (current_drawdown - 0.05) / 0.20
                drawdown_severity = min(1.0, drawdown_severity)
                penalty_factor = drawdown_severity ** 1.5
                components['drawdown_penalty'] = -penalty_factor * self.weights['drawdown_penalty']
            else:
                components['drawdown_penalty'] = 0.0
            
            # 6. Position management penalty (unchanged)
            position_size = abs(action[1])
            if position_size > 0.8:
                size_penalty = (position_size - 0.8) * 2
                components['position_penalty'] = -size_penalty * self.weights['position_penalty']
            else:
                components['position_penalty'] = 0.0
            
            # 7. Risk management bonus (unchanged)
            margin_ratio = portfolio_state.get('margin_ratio', 2.0)
            if margin_ratio > 0.5:
                risk_bonus = min(0.2, (margin_ratio - 0.5) * 0.4)
                components['risk_bonus'] = risk_bonus * self.weights['risk_bonus']
            else:
                components['risk_bonus'] = 0.0
            
            # ✅ NEW: Horizon-specific bonuses
            if self.reward_horizon_steps > 1:
                # Reward consistency over the horizon
                if len(future_values) >= 3:
                    value_changes = np.diff(future_values)
                    consistency_score = 1.0 - np.std(value_changes) / (np.mean(np.abs(value_changes)) + 1e-6)
                    consistency_score = max(0, min(1, consistency_score))
                    components['horizon_consistency'] = consistency_score * 0.1
                else:
                    components['horizon_consistency'] = 0.0
            else:
                components['horizon_consistency'] = 0.0
            
            # Calculate total reward
            total_reward = sum(components.values())
            
            # Final bounds check and scaling
            total_reward = np.clip(total_reward, -5.0, 5.0)
            
            # Update history
            self.reward_history.append(total_reward)
            
            return total_reward, components
            
        except Exception as e:
            logger.error(f"Error in multi-step reward calculation: {e}")
            return 0.0, {'error': -0.1}
    
    def calculate_enhanced_reward(self, prev_value: float, curr_value: float,
                                action: np.ndarray, portfolio_state: Dict,
                                market_state: Dict, previous_action: np.ndarray = None) -> Tuple[float, Dict]:
        """
        Fallback to original single-step reward calculation when horizon data unavailable
        """
        # Use the original implementation for single-step rewards
        return self.calculate_multi_step_reward(
            prev_value, [curr_value], action, portfolio_state, market_state, previous_action
        )

class EnhancedHierarchicalTradingEnvironment(gym.Env):
    """
    ✅ ENHANCED: Trading environment with configurable reward horizon system
    
    Key enhancements:
    1. Multi-step reward calculation using future data
    2. Configurable reward horizon from config
    3. Proper handling of episode boundaries
    4. Maintains all existing functionality
    """
    
    def __init__(self, df_base_ohlc: pd.DataFrame, normalizer: Normalizer, config=None,
                 leverage: float = None, reward_weights: Dict[str, float] = None):
        super().__init__()
        self.cfg = config or SETTINGS
        self.strat_cfg = self.cfg.strategy
        self.normalizer = normalizer
        
        # ✅ ENHANCED: Configurable leverage and reward horizon
        self.leverage = leverage if leverage is not None else self.strat_cfg.leverage
        self.reward_horizon_steps = self.strat_cfg.reward_horizon_steps
        
        logger.info("--- Initializing ENHANCED Trading Environment with Reward Horizon ---")
        logger.info(f" - Leverage: {self.leverage}x")
        logger.info(f" - Reward Horizon: {self.reward_horizon_steps} steps ({self.reward_horizon_steps * 20}s)")
        logger.info(f" - Decay Factor: {self.strat_cfg.reward_horizon_decay}")
        
        try:
            # Initialize enhanced components
            self.risk_manager = EnhancedRiskManager(self.cfg, leverage=self.leverage)
            self.reward_calculator = EnhancedRewardCalculator(self.cfg, leverage=self.leverage, reward_weights=reward_weights)
            
            # Data setup (unchanged from original)
            base_df = df_base_ohlc.set_index('timestamp')
            
            # Get all unique timeframes required
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
                    agg_rules = {
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                        'volume': 'sum', 'volume_delta': 'sum', 'vwap': 'last'
                    }
                    valid_agg_rules = {k: v for k, v in agg_rules.items() if k in base_df.columns}
                    df_resampled = base_df.resample(freq).agg(valid_agg_rules).ffill()
                    self.timeframes[freq] = df_resampled.dropna()
            
            self.base_timestamps = self.timeframes[self.cfg.base_bar_timeframe.value].index
            
            # ✅ ENHANCED: Adjust max_step to account for reward horizon
            self.max_step = len(self.base_timestamps) - max(self.reward_horizon_steps, 2) - 1
            
            logger.info(f"Dataset: {len(self.base_timestamps)} total bars, {self.max_step} usable for training")
            logger.info(f"Reserved {self.reward_horizon_steps} bars at end for reward horizon calculation")
            
            # Validate data sufficiency for reward horizon
            if not self.cfg.validate_reward_horizon_data(len(self.base_timestamps)):
                logger.warning("Dataset may be insufficient for configured reward horizon")
            
            # Action and observation spaces (unchanged)
            self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
            
            # ... [Observation space setup remains the same] ...
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
            
            # Initialize stateful features (unchanged)
            self._initialize_stateful_features()
            
            # Enhanced tracking with reward horizon
            self.portfolio_history = deque(maxlen=500)
            self.previous_portfolio_value = None
            self.episode_returns = []
            self.previous_action = None
            self.consecutive_losses = 0
            self.episode_peak_value = 0.0
            self.market_regime = "UNCERTAIN"
            self.volatility_estimate = 0.02
            
            # ✅ NEW: Reward horizon tracking
            self.future_portfolio_values = deque(maxlen=self.reward_horizon_steps + 1)
            self.reward_horizon_buffer = deque(maxlen=100)  # Store horizon analysis
            
            # Performance tracking
            self.step_rewards = []
            self.reward_components_history = []
            self.trade_count = 0
            self.winning_trades = 0
            
            logger.info("Enhanced environment with reward horizon initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced environment: {e}", exc_info=True)
            raise
    
    def _calculate_future_portfolio_values(self, current_step: int) -> List[float]:
        """
        ✅ NEW: Calculate future portfolio values for reward horizon
        
        This function simulates the portfolio value changes over the next N steps
        without executing any trades, used purely for reward calculation.
        """
        try:
            future_values = []
            
            # Get current portfolio state
            current_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[current_step]
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            current_portfolio_value = self.balance + unrealized_pnl
            
            # Calculate portfolio values for next N steps
            for step_offset in range(1, self.reward_horizon_steps + 1):
                future_step = current_step + step_offset
                
                # Check if we have enough future data
                if future_step >= len(self.base_timestamps):
                    # If we don't have enough future data, use the last available price
                    future_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[-1]
                else:
                    future_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[future_step]
                
                # Calculate future portfolio value with current position
                future_unrealized_pnl = self.asset_held * (future_price - self.entry_price)
                future_portfolio_value = self.balance + future_unrealized_pnl
                
                future_values.append(future_portfolio_value)
            
            return future_values
            
        except Exception as e:
            logger.error(f"Error calculating future portfolio values: {e}")
            return []
    
    def step(self, action: np.ndarray):
        """
        ✅ ENHANCED: Step function with multi-step reward calculation
        
        Key changes:
        1. Calculate future portfolio values for reward horizon
        2. Use multi-step reward calculation
        3. Maintain all existing trading logic
        4. Add reward horizon analysis to info
        """
        try:
            # Update features and market state (unchanged)
            self._update_stateful_features(self.current_step)
            self._update_market_regime_and_volatility(self.current_step)
            
            current_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step]
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            initial_portfolio_value = self.balance + unrealized_pnl
            
            # Update episode peak
            if initial_portfolio_value > self.episode_peak_value:
                self.episode_peak_value = initial_portfolio_value
            
            # Calculate current drawdown from episode peak
            current_drawdown = (self.episode_peak_value - initial_portfolio_value) / self.episode_peak_value
            
            # Enhanced margin call handling (unchanged from original)
            position_notional = abs(self.asset_held) * current_price
            margin_ratio = (self.used_margin - unrealized_pnl) / position_notional if position_notional > 0 else float('inf')
            
            # ... [All existing trading logic remains the same] ...
            
            # [Execute trade logic - keeping all existing code unchanged]
            # ... [Position sizing, risk checks, trade execution] ...
            
            # For brevity, I'm showing the key changes. The full trading logic remains identical.
            # Let me focus on the reward calculation changes:
            
            # Move to next step
            self.current_step += 1
            truncated = self.current_step >= self.max_step
            
            # Calculate next portfolio value (unchanged)
            next_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step]
            next_unrealized_pnl = self.asset_held * (next_price - self.entry_price)
            next_portfolio_value = self.balance + next_unrealized_pnl
            
            # ✅ CORE ENHANCEMENT: Multi-step reward calculation
            if self.reward_horizon_steps > 1 and not truncated:
                # Calculate future portfolio values for reward horizon
                future_values = self._calculate_future_portfolio_values(self.current_step)
                
                if future_values:
                    # Use multi-step reward calculation
                    reward, reward_components = self.reward_calculator.calculate_multi_step_reward(
                        initial_portfolio_value, future_values, action,
                        portfolio_state, market_state, self.previous_action
                    )
                else:
                    # Fallback to single-step if no future data
                    reward, reward_components = self.reward_calculator.calculate_enhanced_reward(
                        initial_portfolio_value, next_portfolio_value, action,
                        portfolio_state, market_state, self.previous_action
                    )
            else:
                # Use single-step reward for immediate horizon or at episode end
                reward, reward_components = self.reward_calculator.calculate_enhanced_reward(
                    initial_portfolio_value, next_portfolio_value, action,
                    portfolio_state, market_state, self.previous_action
                )
            
            # Enhanced termination conditions (unchanged)
            terminated = next_portfolio_value <= initial_portfolio_value * 0.5
            
            # Calculate portfolio state for reward (unchanged)
            portfolio_state = {
                'drawdown': (self.episode_peak_value - next_portfolio_value) / self.episode_peak_value,
                'margin_ratio': margin_ratio,
                'portfolio_value': next_portfolio_value,
                'volatility': self.volatility_estimate
            }
            
            market_state = {
                'regime': self.market_regime,
                'price': next_price,
                'volatility': self.volatility_estimate
            }
            
            # Track consecutive losses (unchanged)
            if reward < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
                
            if reward > 0.1:
                self.winning_trades += 1
            
            # Update tracking (unchanged)
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
            
            # ✅ ENHANCED: Info dictionary with reward horizon analysis
            info = {
                'balance': self.balance,
                'asset_held': self.asset_held,
                'portfolio_value': next_portfolio_value,
                'drawdown': portfolio_state['drawdown'],
                'volatility': self.volatility_estimate,
                'unrealized_pnl': next_unrealized_pnl,
                'margin_ratio': margin_ratio,
                'used_margin': self.used_margin,
                'market_regime': self.market_regime,
                'consecutive_losses': self.consecutive_losses,
                'trade_count': self.trade_count,
                'win_rate': self.winning_trades / max(self.trade_count, 1),
                'reward_components': reward_components,
                'dynamic_position_limit': dynamic_limit,
                'transaction_cost': total_cost,
                'leverage': self.leverage,
                'reward_scaling_factor': self.reward_calculator.scaling_factor,
                # ✅ NEW: Reward horizon information
                'reward_horizon': {
                    'steps': self.reward_horizon_steps,
                    'decay_factor': self.reward_calculator.reward_decay_factor,
                    'horizon_info': reward_components.get('horizon_info', {}),
                    'using_multi_step': len(future_values) > 0 if 'future_values' in locals() else False
                }
            }
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error in enhanced environment step: {e}", exc_info=True)
            observation = self._get_observation_sequence()
            info = {'portfolio_value': self.balance, 'error': True, 'leverage': self.leverage}
            return observation, -1.0, True, False, info
    
    # ... [All other methods remain unchanged from the original implementation] ...
    
    def get_reward_horizon_analysis(self) -> Dict[str, Any]:
        """
        ✅ NEW: Get analysis of reward horizon performance
        """
        try:
            if not self.reward_components_history:
                return {}
            
            # Analyze horizon effectiveness
            horizon_components = [
                comp.get('horizon_info', {}) for comp in self.reward_components_history 
                if 'horizon_info' in comp
            ]
            
            if not horizon_components:
                return {'message': 'No horizon data available'}
            
            immediate_returns = [h.get('immediate_return', 0) for h in horizon_components]
            final_returns = [h.get('final_return', 0) for h in horizon_components]
            weighted_returns = [h.get('weighted_return', 0) for h in horizon_components]
            
            analysis = {
                'total_horizon_steps': len(horizon_components),
                'avg_immediate_return': np.mean(immediate_returns) if immediate_returns else 0,
                'avg_final_return': np.mean(final_returns) if final_returns else 0,
                'avg_weighted_return': np.mean(weighted_returns) if weighted_returns else 0,
                'horizon_consistency': np.std(weighted_returns) if len(weighted_returns) > 1 else 0,
                'horizon_effectiveness': {
                    'immediate_vs_final_correlation': np.corrcoef(immediate_returns, final_returns)[0,1] if len(immediate_returns) > 1 else 0,
                    'weighted_vs_final_correlation': np.corrcoef(weighted_returns, final_returns)[0,1] if len(weighted_returns) > 1 else 0
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in reward horizon analysis: {e}")
            return {'error': str(e)}

# ✅ ENHANCED: Updated convenience function
def create_trading_environment(df_base_ohlc: pd.DataFrame, normalizer: Normalizer,
                              config=None, leverage: float = None, 
                              reward_weights: Dict[str, float] = None,
                              reward_horizon_steps: int = None):
    """
    ✅ ENHANCED: Convenience function with reward horizon configuration
    """
    # Override reward horizon if specified
    if reward_horizon_steps is not None and config is not None:
        config.strategy.reward_horizon_steps = reward_horizon_steps
        logger.info(f"Overriding reward horizon to {reward_horizon_steps} steps")
    
    return EnhancedHierarchicalTradingEnvironment(
        df_base_ohlc=df_base_ohlc,
        normalizer=normalizer,
        config=config,
        leverage=leverage,
        reward_weights=reward_weights
    )

if __name__ == "__main__":
    # Example usage and testing
    try:
        logger.info("Testing enhanced trading environment with reward horizon...")
        
        # Test different reward horizons
        test_horizons = [1, 3, 9]  # 20s, 1min, 3min
        
        for horizon in test_horizons:
            logger.info(f"\n--- Testing {horizon}-step reward horizon ---")
            
            # Test reward calculator
            reward_calc = EnhancedRewardCalculator(SETTINGS, leverage=10.0)
            reward_calc.reward_horizon_steps = horizon
            
            # Simulate multi-step returns
            initial_value = 1000000
            future_values = [
                initial_value * (1 + 0.001 * i) for i in range(1, horizon + 1)
            ]  # Simulate gradual increase
            
            reward, components = reward_calc.calculate_multi_step_reward(
                prev_value=initial_value,
                future_values=future_values,
                action=np.array([0.5, 0.3]),
                portfolio_state={'drawdown': 0.02, 'margin_ratio': 1.5},
                market_state={'regime': 'TRENDING_UP', 'volatility': 0.02}
            )
            
            logger.info(f"Horizon {horizon}: reward={reward:.4f}")
            if 'horizon_info' in components:
                horizon_info = components['horizon_info']
                logger.info(f"  Immediate return: {horizon_info.get('immediate_return', 0):.6f}")
                logger.info(f"  Final return: {horizon_info.get('final_return', 0):.6f}")
                logger.info(f"  Weighted return: {horizon_info.get('weighted_return', 0):.6f}")
        
        logger.info("\n✅ Enhanced trading environment with reward horizon test completed!")
        
    except Exception as e:
        logger.error(f"Enhanced environment test failed: {e}")
        raise