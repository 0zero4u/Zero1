# enhanced_engine_verbose.py

"""
ENHANCED FIXED ENGINE (POST-ANALYSIS): Now with SMART Live Trade Monitoring

This version adds an intelligent verbose mode for real-time trade monitoring.
Only worker #0 will print detailed information for STRATEGICALLY SIGNIFICANT trades
(e.g., opening, closing, flipping, or major size changes) to provide a clean,
high-level view of the agent's decision-making process.
"""

import numpy as np
import pandas as pd
# FIXED: Updated to use gymnasium explicitly to avoid confusion and deprecation warnings.
import gymnasium
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
        
        # Adjust for leverage - higher leverage should have smaller position limits
        leverage_adjustment = min(1.0, 10.0 / self.leverage)
        
        return base_limit * vol_adjustment * leverage_adjustment

class FixedRewardCalculator:
    """
    FIXED: Enhanced reward calculator that encourages action and prevents turtling.
    KEY FIXES (Post-Analysis):
    1. DYNAMIC REWARDS: Made exploration and action rewards context-aware using market volatility.
    2. REBALANCED WEIGHTS: Reduced the impact of the risk-adjusted component to prevent dominance.
    3. ACTION REGULATION: Introduced a penalty for hyperactivity to prevent excessive trading.
    4. BOUNDED COMPONENTS: Added stricter clipping on risk-adjusted rewards.
    """
    
    def __init__(self, config, leverage: float = 10.0, reward_weights: Dict[str, float] = None):
        self.cfg = config
        self.leverage = leverage
        self.return_scaler = RobustScaler()
        self.reward_history = deque(maxlen=1000)
        self.return_buffer = deque(maxlen=500)
        self.volatility_buffer = deque(maxlen=100)
        self.transaction_cost_buffer = deque(maxlen=50)
        
        # NEW: State for action frequency penalty
        self.action_frequency_window = 100  # Look at last 100 steps
        self.optimal_action_threshold = 0.7  # Target 70% of steps to have an action
        self.action_frequency_buffer = deque(maxlen=self.action_frequency_window)
        
        # FIXED: Better balanced reward weights based on analysis
        self.weights = reward_weights or {
            'base_return': 2.3,
            'transaction_penalty': -0.1,
            'drawdown_penalty': -0.8,
            'position_penalty': -0.005,
            'exploration_bonus': 0.0,
            'inactivity_penalty': -0.2,
            'frequency_penalty': -0.4,  # NEW: Penalty for hyperactivity
        }
        
        # Get inactivity parameters from config
        self.inactivity_grace_period = getattr(self.cfg.strategy, 'inactivity_grace_period_steps', 10)
        self.penalty_ramp_up_steps = getattr(self.cfg.strategy, 'penalty_ramp_up_steps', 20)
        
        # FIXED: More reasonable scaling factor
        self.scaling_factor = getattr(self.cfg.strategy, 'reward_scaling_factor', 100.0) / self.leverage
        
        logger.info(f"FIXED reward calculator initialized with leverage {self.leverage}x, scaling_factor {self.scaling_factor:.2f}")
        logger.info("APPLIED ANALYSIS FIXES: Dynamic Rewards, Rebalanced Weights, Action Regulation.")

    def calculate_immediate_reward(self, prev_value: float, curr_value: float,
                                 action: np.ndarray, portfolio_state: Dict,
                                 market_state: Dict, previous_action: np.ndarray = None,
                                 consecutive_inactive_steps: int = 0) -> Tuple[float, Dict]:
        """
        FIXED (POST-ANALYSIS): Calculate IMMEDIATE reward with dynamic, market-aware components.
        """
        try:
            components = {}
            market_volatility = market_state.get('volatility', 0.01)
            
            # --- Base Return & Risk Components ---
            period_return = (curr_value - prev_value) / max(prev_value, 1e-6)
            self.return_buffer.append(period_return)
            
            scaled_return = period_return * self.scaling_factor
            normalized_return = np.tanh(scaled_return * 0.5)
            components['base_return'] = normalized_return * self.weights['base_return']
            
            # --- Cost & Penalty Components ---
            position_change = 0.0
            if previous_action is not None:
                position_change = abs(action[0] - previous_action[0])
                position_size_factor = (abs(action[1]) + abs(previous_action[1])) / 2
                tx_penalty = position_change * position_size_factor * 0.5
                components['transaction_penalty'] = tx_penalty * self.weights['transaction_penalty']
            else:
                components['transaction_penalty'] = 0.0
            
            current_drawdown = portfolio_state.get('drawdown', 0.0)
            if current_drawdown > 0.08:
                drawdown_excess = current_drawdown - 0.08
                penalty_factor = min(1.0, drawdown_excess / 0.12)
                penalty_factor = penalty_factor ** 0.7
                components['drawdown_penalty'] = penalty_factor * self.weights['drawdown_penalty']
            else:
                components['drawdown_penalty'] = 0.0
            
            position_size = abs(action[1])
            if position_size > 0.9:
                size_penalty = (position_size - 0.9) * 0.5
                components['position_penalty'] = size_penalty * self.weights['position_penalty']
            else:
                components['position_penalty'] = 0.0
            
            inactivity_penalty_weight = self.weights.get('inactivity_penalty', 0.0)
            if inactivity_penalty_weight < 0 and consecutive_inactive_steps > self.inactivity_grace_period:
                steps_into_penalty = consecutive_inactive_steps - self.inactivity_grace_period
                ramp_progress = min(1.0, steps_into_penalty / self.penalty_ramp_up_steps)
                ramp_progress = ramp_progress ** 0.5
                components['inactivity_penalty'] = ramp_progress * inactivity_penalty_weight
            else:
                components['inactivity_penalty'] = 0.0
            
            # --- Action & Exploration Bonuses (DYNAMIC) ---
            action_magnitude = abs(action[0]) + abs(action[1])
            position_size = action[1]  # Extract the size component of the action
            
            # FIXED: Action/Exploration rewards are only given if the agent commits to a non-trivial position size.
            # This prevents the agent from receiving rewards for 'fake' trades with zero size.
            if position_size > 0.01:
                # DYNAMIC: Exploration bonus scaled by market volatility and inverse position confidence
                position_confidence = abs(action[0])  # Proxy for confidence in direction
                volatility_scaler = np.clip(market_volatility * 50, 0.5, 2.0)
                exploration_reward = (action_magnitude * 0.5) * volatility_scaler * (1 - position_confidence)
                components['exploration_bonus'] = min(0.1, exploration_reward) * self.weights['exploration_bonus']
                
            else:
                components['exploration_bonus'] = 0.0
            
            # --- Risk Management & NEW Frequency Penalty ---
            # NEW: Action frequency regulation to prevent hyperactivity
            self.action_frequency_buffer.append(1 if action_magnitude > 0.01 else 0)
            if len(self.action_frequency_buffer) == self.action_frequency_window:
                recent_action_frequency = np.mean(list(self.action_frequency_buffer))
                if recent_action_frequency > self.optimal_action_threshold:
                    freq_excess = recent_action_frequency - self.optimal_action_threshold
                    frequency_penalty = freq_excess * self.weights.get('frequency_penalty', 0.0)
                    components['frequency_penalty'] = frequency_penalty
                else:
                    components['frequency_penalty'] = 0.0
            else:
                components['frequency_penalty'] = 0.0
            
            # Calculate total immediate reward
            total_reward = sum(components.values())
            total_reward = np.clip(total_reward, -3.0, 3.0)
            
            self.reward_history.append(total_reward)
            
            return total_reward, components
            
        except Exception as e:
            logger.error(f"Error in immediate reward calculation: {e}", exc_info=True)
            return 0.0, {'error': -0.1}

# FIXED: Inherit directly from gymnasium.Env for clarity.
class FixedHierarchicalTradingEnvironment(gymnasium.Env):
    """
    FIXED: Trading environment that prevents turtling and encourages learning.
    NOW WITH SMART LIVE TRADE MONITORING for debugging and intuition building.
    """
    
    def __init__(self, df_base_ohlc: pd.DataFrame, normalizer: Normalizer, config=None,
                 leverage: float = None, reward_weights: Dict[str, float] = None,
                 precomputed_features: Optional[pd.DataFrame] = None,
                 worker_id: int = 0):
        super().__init__()
        
        # --- SMART VERBOSE MODE SETUP ---
        self.worker_id = worker_id
        self.verbose = self.worker_id == 0
        
        # ANSI colors for better readability
        self.COLOR_GREEN = '\033[92m'
        self.COLOR_RED = '\033[91m'
        self.COLOR_YELLOW = '\033[93m'
        self.COLOR_BLUE = '\033[94m'
        self.COLOR_CYAN = '\033[96m'
        self.COLOR_MAGENTA = '\033[95m'
        self.COLOR_RESET = '\033[0m'
        self.COLOR_BOLD = '\033[1m'
        
        self.trade_counter = 0
        self.last_logged_asset_held = 0.0
        # --- END SETUP ---
        
        self.cfg = config or SETTINGS
        self.strat_cfg = self.cfg.strategy
        self.normalizer = normalizer
        self.leverage = leverage if leverage is not None else self.strat_cfg.leverage
        
        if self.verbose:
            logger.info("--- Initializing FIXED High-Performance Trading Environment (SMART VERBOSE MODE) ---")
            logger.info(f" -> FIXED: Prevents turtling and reward imbalance")
            logger.info(f" -> FIXED: Encourages exploration and learning")
            logger.info(f" -> NEW: Smart live trade monitoring enabled for Worker {self.worker_id}")
        
        try:
            self.risk_manager = EnhancedRiskManager(self.cfg, leverage=self.leverage)
            self.reward_calculator = FixedRewardCalculator(self.cfg, leverage=self.leverage, reward_weights=reward_weights)
            
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
            
            if self.verbose:
                logger.info("Resampling data and converting to NumPy...")
            
            for freq in all_required_freqs:
                agg_rules = {
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
                    'volume_delta': 'sum', 'vwap': 'last', 'trade_count': 'sum'
                }
                valid_agg_rules = {k: v for k, v in agg_rules.items() if k in base_df.columns}
                df_resampled = base_df.resample(freq).agg(valid_agg_rules).ffill()
                df_resampled = df_resampled.reindex(self.base_timestamps, method='ffill').fillna(0)
                self.timeframes_np[freq] = {col: df_resampled[col].values for col in df_resampled.columns}
            
            if precomputed_features is not None:
                features_indexed = precomputed_features.set_index('timestamp')
                features_aligned = features_indexed.reindex(self.base_timestamps, method='ffill').fillna(0.0)
                self.all_features_np = {
                    key: features_aligned[key].values
                    for key in self.strat_cfg.context_feature_keys + self.strat_cfg.precomputed_feature_keys
                    if key in features_aligned.columns
                }
            else:
                raise ValueError("`precomputed_features` must be provided for the optimized environment.")
            
            self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
            
            obs_spaces = {}
            seq_len = self.strat_cfg.sequence_length
            
            for key, lookback in self.strat_cfg.lookback_periods.items():
                key_str = key.value
                if key_str.startswith('ohlcv_'):
                    shape = (seq_len, lookback, 5)
                elif key_str.startswith('ohlc_'):
                    shape = (seq_len, lookback, 4)
                else:
                    shape = (seq_len, lookback)
                obs_spaces[key_str] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
            
            self.observation_space = spaces.Dict(obs_spaces)
            self.observation_history = deque(maxlen=self.strat_cfg.sequence_length)
            
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
            
            if self.verbose:
                logger.info("✅ FIXED environment initialized successfully with SMART LIVE MONITORING.")
                
        except Exception as e:
            logger.error(f"Failed to initialize fixed environment: {e}", exc_info=True)
            raise

    def _print_trade_info(self, trade_type: str, action: np.ndarray, current_price: float, 
                         portfolio_value: float, unrealized_pnl: float, reward: float, 
                         reward_components: Dict[str, float]):
        """Prints a detailed summary of a significant trade if in verbose mode."""
        if not self.verbose:
            return
        
        self.trade_counter += 1
        
        pnl_color = self.COLOR_GREEN if unrealized_pnl >= 0 else self.COLOR_RED
        reward_color = self.COLOR_GREEN if reward >= 0 else self.COLOR_RED
        
        print(f"\n{self.COLOR_BOLD}{'='*80}{self.COLOR_RESET}")
        print(f"{self.COLOR_CYAN}🚀 TRADE #{self.trade_counter} | Worker {self.worker_id} | Step {self.current_step}{self.COLOR_RESET}")
        print(f"{self.COLOR_BOLD}ACTION: {trade_type}{self.COLOR_RESET}")
        print(f"{'='*80}")
        
        print(f"📈 Market Price: {self.COLOR_YELLOW}${current_price:,.2f}{self.COLOR_RESET} | Entry Price: ${self.entry_price:,.2f}")
        print(f"💼 Position: {self.asset_held:.6f} BTC | Portfolio: ${portfolio_value:,.2f}")
        print(f"💰 PnL: {pnl_color}${unrealized_pnl:+,.2f}{self.COLOR_RESET} | Reward: {reward_color}{reward:+.4f}{self.COLOR_RESET}")
        
        action_signal = action[0]
        action_size = action[1]
        signal_color = self.COLOR_GREEN if action_signal > 0 else self.COLOR_RED if action_signal < 0 else self.COLOR_YELLOW
        print(f"🎯 Action Signal: {signal_color}{action_signal:+.4f}{self.COLOR_RESET} | Size: {action_size:.4f}")
        
        regime_color = {
            "HIGH_VOLATILITY": self.COLOR_RED, "TRENDING_UP": self.COLOR_GREEN, "TRENDING_DOWN": self.COLOR_RED,
            "LOW_VOLATILITY": self.COLOR_BLUE, "SIDEWAYS": self.COLOR_YELLOW, "UNCERTAIN": self.COLOR_MAGENTA
        }.get(self.market_regime, self.COLOR_RESET)
        
        print(f"🌊 Market Regime: {regime_color}{self.market_regime}{self.COLOR_RESET} | Volatility: {self.volatility_estimate:.4f}")
        
        significant_components = {k: v for k, v in reward_components.items() if abs(v) > 0.001 and k != 'error'}
        if significant_components:
            print(f"🔍 Key Reward Components:")
            sorted_components = sorted(significant_components.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for component, value in sorted_components:
                comp_color = self.COLOR_GREEN if value >= 0 else self.COLOR_RED
                print(f"   • {component}: {comp_color}{value:+.4f}{self.COLOR_RESET}")
        
        print(f"{self.COLOR_BOLD}{'='*80}{self.COLOR_RESET}\n")

    def _print_episode_start(self):
        """Print episode start information if in verbose mode."""
        if not self.verbose:
            return
            
        print(f"\n{self.COLOR_BOLD}{self.COLOR_CYAN}🎬 NEW EPISODE STARTING - Worker {self.worker_id}{self.COLOR_RESET}")
        print(f"{self.COLOR_BOLD}{'='*80}{self.COLOR_RESET}")
        print(f"💰 Initial Balance: $1,000,000.00")
        print(f"⚡ Leverage: {self.leverage}x")
        print(f"📊 Data Points Available: {self.max_step:,}")
        print(f"🎯 Starting Step: {self.current_step}")
        print(f"{self.COLOR_BOLD}{'='*80}{self.COLOR_RESET}\n")
        
        self.trade_counter = 0

    def step(self, action: np.ndarray):
        """
        FIXED: Step function with improved reward calculation, action encouragement,
        and SMART LIVE TRADE MONITORING.
        """
        try:
            self._update_market_regime_and_volatility(self.current_step)
            
            current_price = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][self.current_step]
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            initial_portfolio_value = self.balance + unrealized_pnl
            
            if initial_portfolio_value > self.episode_peak_value:
                self.episode_peak_value = initial_portfolio_value
            
            current_drawdown = (self.episode_peak_value - initial_portfolio_value) / self.episode_peak_value
            position_notional = abs(self.asset_held) * current_price
            margin_ratio = (self.used_margin - unrealized_pnl) / position_notional if position_notional > 0 else float('inf')
            
            if margin_ratio < self.strat_cfg.maintenance_margin_rate:
                if self.verbose:
                    print(f"{self.COLOR_RED}🚨 LIQUIDATION EVENT - Worker {self.worker_id} - Step {self.current_step} 🚨{self.COLOR_RESET}")
                # ... (liquidation logic)
                reward = -2.0; terminated = True; self.current_step += 1; truncated = self.current_step >= self.max_step
                self.observation_history.append(self._get_single_step_observation(self.current_step))
                observation = self._get_observation_sequence()
                info = {'portfolio_value': self.balance, 'liquidation': True}
                return observation, reward, terminated, truncated, info
            
            # (Trade execution logic remains the same)
            action_signal = np.clip(action[0], -1.0, 1.0)
            action_size = np.clip(action[1], 0.0, 1.0)
            dynamic_limit = self.risk_manager.calculate_dynamic_position_limit(self.volatility_estimate, initial_portfolio_value, self.market_regime)
            effective_size = action_size * dynamic_limit
            target_notional = initial_portfolio_value * action_signal * effective_size
            target_asset_quantity = target_notional / current_price if current_price > 1e-8 else 0
            # (Position limit logic remains the same)
            max_allowable_margin = initial_portfolio_value * self.strat_cfg.max_margin_allocation_pct
            required_margin_for_target = (abs(target_asset_quantity) * current_price) / self.leverage
            if required_margin_for_target > max_allowable_margin:
                capped_notional = max_allowable_margin * self.leverage
                target_asset_quantity = (capped_notional / current_price * np.sign(target_asset_quantity)) if current_price > 1e-8 else 0
            
            trade_quantity = target_asset_quantity - self.asset_held
            trade_notional = abs(trade_quantity) * current_price
            total_cost = trade_notional * (self.cfg.transaction_fee_pct + self.cfg.slippage_pct)
            
            action_magnitude = abs(action[0]) + abs(action[1])
            self.consecutive_inactive_steps = 0 if action_magnitude > 0.001 else self.consecutive_inactive_steps + 1
            
            previous_asset_held = self.asset_held
            
            self.balance += unrealized_pnl - total_cost
            self.asset_held = target_asset_quantity
            self.used_margin = (abs(self.asset_held) * current_price) / self.leverage
            if abs(trade_quantity) > 1e-8:
                self.entry_price = current_price
                self.trade_count += 1
            
            self.current_step += 1
            truncated = self.current_step >= self.max_step
            
            next_price = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][self.current_step]
            next_unrealized_pnl = self.asset_held * (next_price - self.entry_price)
            next_portfolio_value = self.balance + next_unrealized_pnl
            
            terminated = (self.episode_peak_value - next_portfolio_value) / self.episode_peak_value >= self.strat_cfg.max_drawdown_threshold
            
            portfolio_state = {'drawdown': current_drawdown, 'margin_ratio': margin_ratio}
            market_state = {'regime': self.market_regime, 'price': current_price, 'volatility': self.volatility_estimate}
            
            reward, reward_components = self.reward_calculator.calculate_immediate_reward(
                initial_portfolio_value, next_portfolio_value, action, portfolio_state, market_state,
                self.previous_action, self.consecutive_inactive_steps
            )
            
            # --- START OF SMART LOGGING LOGIC ---
            if self.verbose:
                current_asset_held = self.asset_held
                position_flipped = np.sign(current_asset_held) != np.sign(previous_asset_held) and previous_asset_held != 0
                position_opened = abs(previous_asset_held) < 1e-9 and abs(current_asset_held) > 1e-9
                position_closed = abs(previous_asset_held) > 1e-9 and abs(current_asset_held) < 1e-9
                
                major_size_change = False
                if previous_asset_held != 0:
                    size_change_pct = abs((current_asset_held - previous_asset_held) / previous_asset_held)
                    if size_change_pct > 0.25:
                        major_size_change = True

                if position_flipped or position_opened or position_closed or major_size_change:
                    if abs(current_asset_held) > abs(previous_asset_held):
                        trade_type = f"{self.COLOR_GREEN}🚀 OPEN/ADD{' LONG' if current_asset_held > 0 else ' SHORT'}{self.COLOR_RESET}"
                    elif abs(current_asset_held) < abs(previous_asset_held):
                        trade_type = f"{self.COLOR_YELLOW}📈 CLOSE/REDUCE{' LONG' if previous_asset_held > 0 else ' SHORT'}{self.COLOR_RESET}"
                    else: # Flip
                        trade_type = f"{self.COLOR_BLUE}🔄 FLIP TO{' LONG' if current_asset_held > 0 else ' SHORT'}{self.COLOR_RESET}"
                    
                    self._print_trade_info(
                        trade_type, action, current_price, next_portfolio_value, 
                        next_unrealized_pnl, reward, reward_components
                    )
                    self.last_logged_asset_held = current_asset_held
            # --- END OF SMART LOGGING LOGIC ---
            
            self.previous_action = action
            self.portfolio_history.append(next_portfolio_value)
            self.step_rewards.append(reward)
            self.reward_components_history.append(reward_components)
            self.previous_portfolio_value = next_portfolio_value
            
            self.observation_history.append(self._get_single_step_observation(self.current_step))
            observation = self._get_observation_sequence()
            
            info = {
                'balance': self.balance, 'asset_held': self.asset_held, 'portfolio_value': next_portfolio_value,
                'drawdown': current_drawdown, 'volatility': self.volatility_estimate, 'unrealized_pnl': next_unrealized_pnl,
                'margin_ratio': margin_ratio, 'used_margin': self.used_margin, 'market_regime': self.market_regime,
                'trade_count': self.trade_count, 'reward_components': reward_components,
                'action_magnitude': action_magnitude, 'consecutive_inactive_steps': self.consecutive_inactive_steps
            }
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error in fixed environment step: {e}", exc_info=True)
            observation = self._get_observation_sequence()
            info = {'portfolio_value': self.balance, 'error': True}
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
            
            # Reset verbose mode tracking
            self.last_logged_asset_held = 0.0
            
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
            
            # Print episode start info in verbose mode
            self._print_episode_start()
            
            info = {
                'balance': self.balance, 'asset_held': self.asset_held, 'portfolio_value': self.balance,
                'market_regime': self.market_regime, 'volatility_estimate': self.volatility_estimate,
                'leverage': self.leverage, 'fixed_reward_system': True, 'verbose_mode': self.verbose
            }
            
            return observation, info
            
        except Exception as e:
            logger.error(f"Error resetting fixed environment: {e}", exc_info=True)
            raise

    # --- (All helper methods like _get_single_step_observation, etc. remain unchanged) ---

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
                    if len(window) < lookback: window = np.pad(window, (lookback - len(window), 0), 'edge')
                    raw_obs[key] = window
                    
                elif key.startswith('volume_delta_'):
                    window = self.timeframes_np[freq]['volume_delta'][start_idx : end_idx + 1].astype(np.float32)
                    if len(window) < lookback: window = np.pad(window, (lookback - len(window), 0), 'constant')
                    raw_obs[key] = window
                    
                elif key.startswith('ohlc'):
                    cols = ['open','high','low','close','volume'] if 'v' in key else ['open','high','low','close']
                    window_arrays = [self.timeframes_np[freq][c][start_idx : end_idx + 1] for c in cols]
                    window = np.stack(window_arrays, axis=1).astype(np.float32)
                    if len(window) < lookback:
                        padding = np.repeat(window[0:1], lookback - len(window), axis=0)
                        window = np.concatenate([padding, window], axis=0)
                    raw_obs[key] = window
            
            raw_obs[FeatureKeys.CONTEXT.value] = self._get_current_context_features(step_index)
            
            precomputed_vector = np.array(
                [self.all_features_np[k][step_index] for k in self.strat_cfg.precomputed_feature_keys],
                dtype=np.float32
            )
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
            regime_map = {"HIGH_VOLATILITY": 1.0, "TRENDING_UP": 0.8, "TRENDING_DOWN": -0.8,
                          "LOW_VOLATILITY": 0.6, "SIDEWAYS": 0.0, "UNCERTAIN": -0.2}
            regime_encoding = regime_map.get(self.market_regime, 0.0)
            
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
                shape = (lookback, 5) if 'ohlcv' in key else (lookback, 4) if 'ohlc' in key else lookback
                obs[key] = np.zeros(shape, dtype=np.float32)
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
            
            metrics = {
                'total_return': total_return, 'volatility': volatility, 'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio, 'win_rate': win_rate, 'total_trades': self.trade_count,
                'final_portfolio_value': final_value, 'leverage': self.leverage, 'verbose_mode': self.verbose
            }
            return metrics
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'leverage': self.leverage, 'verbose_mode': self.verbose}
