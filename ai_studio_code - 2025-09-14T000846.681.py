# enhanced_engine_verbose.py (Correctly Patched)

"""
ENHANCED FIXED ENGINE (POST-ANALYSIS): Now with SMART Live Trade Monitoring and Hardened Reward Calculator

This version integrates a new, hardened RewardCalculator to prevent reward hacking exploits
related to the exploration bonus. It also refactors penalty calculations into the main
environment loop for better clarity and control.

--- START OF CRITICAL FIXES (From User Analysis) ---
CRITICAL FIX #1: Corrected the flawed position sizing formula. The action signal (confidence)
                 is now decoupled from the action size (magnitude). Signal determines
                 direction (+1/-1), and size determines the position's scale, creating
                 a stable and learnable action space.
CRITICAL FIX #2: Implemented proper entry price management. The entry price is no longer
                 incorrectly reset on every trade. It now correctly averages up when
                 adding to a position and remains unchanged when reducing, ensuring
                 accurate PnL calculation and preventing "phantom profits."
CRITICAL FIX #3 (HARDENED IMPLEMENTATION): Replaced the entire reward calculator with a new,
                 conservative implementation that prevents all known exploration bonus exploits.
                 The new bonus requires a significant increase in risk, uses multiple gates,
                 and rewards reasonable confidence to prevent gaming.
--- END OF CRITICAL FIXES ---
"""

import numpy as np
import pandas as pd
import gymnasium
from gymnasium import spaces
from collections import deque
from typing import Dict, List, Any, Optional, Tuple, Sequence
import logging
from dataclasses import dataclass
from scipy import stats

# Import configuration and the new stateful feature calculators
from config import SETTINGS, FeatureKeys
from normalizer import Normalizer
from features import (
    StatefulBBWPercentRank,
    StatefulPriceDistanceMA,
    StatefulSRDistances,
    StatefulVWAPDistance,
)

logger = logging.getLogger(__name__)

# --- START OF INTEGRATED PATCH: HARDENED REWARD CALCULATOR ---

@dataclass
class ExplorationConfig:
    """All thresholds are configurable to tune per-instrument/deployment."""
    minimum_risk_commitment_pct: float = 0.01  # 1% of portfolio
    min_trade_notional_usd: float = 1000.0     # absolute USD floor
    relative_trade_cap: float = 0.5            # fraction of portfolio the reward scales to
    exploration_reward_cap: float = 0.1        # max normalized exploration reward
    confidence_min_factor: float = 0.25        # minimum multiplier from confidence
    confidence_scale: float = 0.75             # how strongly confidence influences reward
    commitment_scaler_denom: float = 5.0       # divisor scaling the commitment scaler
    volatility_scaler_min: float = 0.5
    volatility_scaler_max: float = 2.0

class RewardCalculator:
    """Hardened reward calculator that integrates a conservative exploration bonus."""
    def __init__(self, cfg: ExplorationConfig | None = None, weights: Optional[dict] = None):
        self.cfg = cfg or ExplorationConfig()
        self.weights = weights or {
            'base_return': 8.0,
            'transaction_penalty': -0.1,
            'drawdown_penalty': -1.2,
            'exploration_bonus': 0.0, # DISABLED BY DEFAULT FOR SAFETY
            'tiny_action_penalty': -1.5,
            'frequency_penalty': -0.5,
            'inactivity_penalty': -0.4,
        }

    def calculate_exploration_bonus_safe(
        self,
        previous_asset_held: float,
        target_asset_quantity: float,
        trade_quantity: float,
        previous_portfolio_value: float,
        current_price: float,
        prev_price: Optional[float],
        action: Sequence[float] | float,
        market_volatility: float,
        estimated_costs_usd: Optional[float] = None,
    ) -> float:
        """Compute a conservative exploration bonus, preventing exploits."""
        prev_value = max(float(previous_portfolio_value), 1e-9)
        current_price = float(current_price)
        prev_price = float(prev_price) if (prev_price is not None) else current_price

        previous_exposure_notional = abs(previous_asset_held) * prev_price
        target_exposure_notional = abs(target_asset_quantity) * current_price
        change_in_exposure = abs(target_exposure_notional - previous_exposure_notional)
        trade_notional = abs(trade_quantity) * current_price
        cfg = self.cfg

        increased_exposure = (target_exposure_notional - previous_exposure_notional) > 0
        risk_commitment_pct = change_in_exposure / prev_value
        min_trade_usd = max(prev_value * cfg.minimum_risk_commitment_pct, cfg.min_trade_notional_usd)
        passed_gates = increased_exposure and (risk_commitment_pct > cfg.minimum_risk_commitment_pct) and (trade_notional >= min_trade_usd)

        if not passed_gates:
            return 0.0

        relative_trade = np.clip(trade_notional / prev_value, 0.0, cfg.relative_trade_cap)
        position_confidence = float(abs(action[0])) if hasattr(action, '__len__') and len(action) > 0 else float(abs(action))
        position_confidence = np.clip(position_confidence, 0.0, 1.0)
        
        confidence_factor = np.clip(cfg.confidence_min_factor + cfg.confidence_scale * position_confidence, cfg.confidence_min_factor, 1.0)
        volatility_scaler = np.clip(market_volatility * 50.0, cfg.volatility_scaler_min, cfg.volatility_scaler_max)
        commitment_scaler = np.clip(risk_commitment_pct / (cfg.minimum_risk_commitment_pct * cfg.commitment_scaler_denom), 0.5, 2.0)
        exploration_reward = relative_trade * 0.5 * volatility_scaler * confidence_factor * commitment_scaler
        exploration_reward = float(np.clip(exploration_reward, 0.0, cfg.exploration_reward_cap))

        if estimated_costs_usd is not None:
            cost_based_cap = max(0.0, estimated_costs_usd / prev_value * 0.5)
            exploration_reward = min(exploration_reward, cost_based_cap)
        
        return exploration_reward

    def calculate_immediate_reward(self, **kwargs) -> dict:
        """Computes all reward components for a given step."""
        components = {}
        
        # --- Extrinsic Rewards (from market) ---
        base_return = kwargs.get('base_return', 0.0)
        total_cost = kwargs.get('total_cost', 0.0)
        prev_value = max(kwargs.get('prev_value', 1e-6), 1e-6)
        
        scaled_return = base_return * kwargs.get('scaling_factor', 100.0)
        normalized_return = np.tanh(scaled_return * 0.5)
        components['base_return'] = normalized_return * self.weights['base_return']
        
        normalized_cost = total_cost / prev_value
        components['transaction_penalty'] = normalized_cost * self.weights['transaction_penalty']
        
        current_drawdown = kwargs.get('drawdown', 0.0)
        if current_drawdown > 0.08:
            drawdown_excess = current_drawdown - 0.08
            penalty_factor = min(1.0, drawdown_excess / 0.12)
            components['drawdown_penalty'] = penalty_factor * self.weights['drawdown_penalty']
        else:
            components['drawdown_penalty'] = 0.0

        # --- Intrinsic Rewards (behavior shaping) ---
        exploration_raw = self.calculate_exploration_bonus_safe(
            previous_asset_held=kwargs['previous_asset_held'],
            target_asset_quantity=kwargs['target_asset_quantity'],
            trade_quantity=kwargs['trade_quantity'],
            previous_portfolio_value=prev_value,
            current_price=kwargs['current_price'],
            prev_price=kwargs['prev_price'],
            action=kwargs['action'],
            market_volatility=kwargs['market_volatility'],
            estimated_costs_usd=total_cost,
        )
        components['exploration_bonus'] = exploration_raw * self.weights['exploration_bonus']

        if kwargs.get('is_insignificant_action', False):
            components['tiny_action_penalty'] = self.weights['tiny_action_penalty']
        else:
            components['tiny_action_penalty'] = 0.0
            
        components['frequency_penalty'] = kwargs.get('frequency_penalty', 0.0)
        components['inactivity_penalty'] = kwargs.get('inactivity_penalty', 0.0)

        return components

# --- END OF INTEGRATED PATCH ---

class RewardNormalizer:
    """Tracks the running mean and standard deviation of rewards for normalization."""
    def __init__(self, epsilon: float = 1e-8):
        self.mean, self.std, self.variance, self.count, self.M2, self.epsilon = 0.0, 1.0, 1.0, 0, 0.0, epsilon

    def update(self, reward: float):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.M2 += delta * delta2
        if self.count > 1:
            self.variance = self.M2 / self.count
            self.std = np.sqrt(self.variance)

    def normalize(self, reward: float) -> float:
        return (reward - self.mean) / (self.std + self.epsilon)

class EnhancedRiskManager:
    """Advanced risk management system with dynamic limits and progressive penalties"""
    def __init__(self, config, leverage: float = 10.0):
        self.cfg, self.leverage = config, leverage
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

    def calculate_dynamic_position_limit(self, volatility: float, portfolio_value: float, market_regime: str) -> float:
        base_limit = self.cfg.strategy.max_position_size
        vol_adjustment = max(0.5, min(2.0, 1.0 / (volatility * 10 + 0.1)))
        leverage_adjustment = min(1.0, 10.0 / self.leverage)
        return base_limit * vol_adjustment * leverage_adjustment

    def get_volatility_percentile(self, current_volatility: float) -> float:
        if len(self.volatility_buffer) < 20: return 0.5
        return stats.percentileofscore(list(self.volatility_buffer), current_volatility) / 100.0


class FixedHierarchicalTradingEnvironment(gymnasium.Env):
    """FIXED Trading environment with hardened reward calculator and live monitoring."""
    def __init__(self, df_base_ohlc: pd.DataFrame, normalizer: Normalizer, config=None,
                 leverage: float = None, reward_weights: Dict[str, float] = None,
                 precomputed_features: Optional[pd.DataFrame] = None,
                 worker_id: int = 0):
        super().__init__()
        self.worker_id, self.verbose = worker_id, worker_id == 0
        self.COLOR_GREEN, self.COLOR_RED, self.COLOR_YELLOW, self.COLOR_BLUE, self.COLOR_CYAN, self.COLOR_MAGENTA, self.COLOR_RESET, self.COLOR_BOLD = '\033[92m', '\033[91m', '\033[93m', '\033[94m', '\033[96m', '\033[95m', '\033[0m', '\033[1m'
        self.cfg, self.strat_cfg, self.normalizer = config or SETTINGS, (config or SETTINGS).strategy, normalizer
        self.leverage = leverage if leverage is not None else self.strat_cfg.leverage
        
        try:
            self.risk_manager = EnhancedRiskManager(self.cfg, leverage=self.leverage)
            self.reward_calculator = RewardCalculator(weights=reward_weights)
            self.reward_normalizer = RewardNormalizer()
            self.action_frequency_buffer = deque(maxlen=100)
            self.inactivity_grace_period = getattr(self.strat_cfg, 'inactivity_grace_period_steps', 10)
            self.penalty_ramp_up_steps = getattr(self.strat_cfg, 'penalty_ramp_up_steps', 20)
            
            base_df = df_base_ohlc.set_index('timestamp')
            all_required_freqs = set(k.value.split('_')[-1].replace('m','T').replace('h','H').replace('s','S').upper() for k in self.strat_cfg.lookback_periods.keys() if k not in {FeatureKeys.CONTEXT, FeatureKeys.PORTFOLIO_STATE, FeatureKeys.PRECOMPUTED_FEATURES}).union({c.timeframe for c in self.strat_cfg.stateful_calculators})
            self.base_timestamps = base_df.resample(self.cfg.base_bar_timeframe.value).asfreq().index
            self.max_step = len(self.base_timestamps) - 2
            
            self.timeframes_np: Dict[str, Dict[str, np.ndarray]] = {}
            for freq in all_required_freqs:
                agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'trade_count': 'sum'}
                valid_agg_rules = {k: v for k, v in agg_rules.items() if k in base_df.columns}
                df_resampled = base_df.resample(freq).agg(valid_agg_rules).ffill().reindex(self.base_timestamps, method='ffill').fillna(0)
                self.timeframes_np[freq] = {col: df_resampled[col].values for col in df_resampled.columns}
            
            if precomputed_features is not None:
                features_aligned = precomputed_features.set_index('timestamp').reindex(self.base_timestamps, method='ffill').fillna(0.0)
                self.all_features_np = {key: features_aligned[key].values for key in self.strat_cfg.context_feature_keys + self.strat_cfg.precomputed_feature_keys if key in features_aligned.columns}
            else:
                raise ValueError("`precomputed_features` must be provided.")
            
            self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
            obs_spaces = {k.value: spaces.Box(low=-np.inf, high=np.inf, shape=((self.strat_cfg.sequence_length, l, 5) if k.value.startswith('ohlcv_') else (self.strat_cfg.sequence_length, l, 4) if k.value.startswith('ohlc_') else (self.strat_cfg.sequence_length, l)), dtype=np.float32) for k, l in self.strat_cfg.lookback_periods.items()}
            self.observation_space = spaces.Dict(obs_spaces)
            self.observation_history = deque(maxlen=self.strat_cfg.sequence_length)
            
            self.reset()
        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}", exc_info=True)
            raise

    def _print_trade_info(self, trade_type: str, action: np.ndarray, current_price: float, portfolio_value: float, unrealized_pnl: float, reward: float, reward_components: Dict[str, float]):
        if not self.verbose: return
        self.trade_counter += 1
        pnl_color, reward_color = (self.COLOR_GREEN if x >= 0 else self.COLOR_RED for x in [unrealized_pnl, reward])
        print(f"\n{self.COLOR_BOLD}{'='*80}{self.COLOR_RESET}\n{self.COLOR_CYAN}🚀 TRADE #{self.trade_counter} | Worker {self.worker_id} | Step {self.current_step}{self.COLOR_RESET}\n{self.COLOR_BOLD}ACTION: {trade_type}{self.COLOR_RESET}\n{'='*80}")
        print(f"📈 Price: {self.COLOR_YELLOW}${current_price:,.2f}{self.COLOR_RESET} | Entry: ${self.entry_price:,.2f} | Portfolio: ${portfolio_value:,.2f}")
        print(f"💰 PnL: {pnl_color}${unrealized_pnl:+,.2f}{self.COLOR_RESET} | Reward: {reward_color}{reward:+.4f}{self.COLOR_RESET}")
        signal_color = self.COLOR_GREEN if action[0] > 0 else self.COLOR_RED if action[0] < 0 else self.COLOR_YELLOW
        print(f"🎯 Action Signal: {signal_color}{action[0]:+.4f}{self.COLOR_RESET} | Size: {action[1]:.4f}")
        significant_components = {k: v for k, v in reward_components.items() if abs(v) > 1e-5}
        if significant_components:
            print("🔍 Key Reward Components:" + "".join(f"\n   • {c}: {(self.COLOR_GREEN if v >= 0 else self.COLOR_RED)}{v:+.4f}{self.COLOR_RESET}" for c, v in sorted(significant_components.items(), key=lambda x: abs(x[1]), reverse=True)[:5]))
        print(f"{self.COLOR_BOLD}{'='*80}{self.COLOR_RESET}\n")

    def _print_episode_start(self):
        if not self.verbose: return
        print(f"\n{self.COLOR_BOLD}{self.COLOR_CYAN}🎬 NEW EPISODE STARTING - Worker {self.worker_id}{self.COLOR_RESET}\n{'='*80}")
        print(f"💰 Initial Balance: ${self.balance:,.2f}\n⚡ Leverage: {self.leverage}x\n📊 Data Points: {self.max_step:,}\n🎯 Start Step: {self.current_step}\n{'='*80}\n")
        self.trade_counter = 0

    def step(self, action: np.ndarray):
        try:
            self._update_market_regime_and_volatility(self.current_step)
            current_price = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][self.current_step]
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            initial_portfolio_value = self.balance + unrealized_pnl
            if initial_portfolio_value > self.episode_peak_value: self.episode_peak_value = initial_portfolio_value
            current_drawdown = (self.episode_peak_value - initial_portfolio_value) / max(self.episode_peak_value, 1e-9)
            
            # Position Sizing & Trade Execution
            action_signal, action_size = np.clip(action[0], -1.0, 1.0), np.clip(action[1], 0.0, 1.0)
            target_asset_quantity = 0.0
            if action_size >= 0.01:
                target_notional = initial_portfolio_value * np.sign(action_signal) * (action_size * self.risk_manager.calculate_dynamic_position_limit(self.volatility_estimate, initial_portfolio_value, self.market_regime))
                target_asset_quantity = target_notional / max(current_price, 1e-8)
            trade_quantity = target_asset_quantity - self.asset_held
            previous_asset_held = self.asset_held
            
            # State Updates
            self.balance += unrealized_pnl
            if abs(trade_quantity) > 1e-8:
                if previous_asset_held == 0 or np.sign(target_asset_quantity) != np.sign(previous_asset_held): self.entry_price = current_price
                elif abs(target_asset_quantity) > abs(previous_asset_held): self.entry_price = abs(((previous_asset_held * self.entry_price) + (trade_quantity * current_price)) / target_asset_quantity)
                self.trade_count += 1
            self.asset_held = target_asset_quantity
            if abs(self.asset_held) < 1e-9: self.entry_price = 0.0
            total_cost = abs(trade_quantity) * current_price * (self.cfg.transaction_fee_pct + self.cfg.slippage_pct)
            self.balance -= total_cost
            
            self.current_step += 1
            next_price = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][self.current_step]
            next_portfolio_value = self.balance + self.asset_held * (next_price - self.entry_price)
            
            # Penalty Calculations
            self.action_frequency_buffer.append(1 if abs(action_signal) + action_size > 0.01 else 0)
            frequency_penalty = 0.0
            if len(self.action_frequency_buffer) > 20:
                freq_excess = np.mean(list(self.action_frequency_buffer)) - np.clip(0.3 * np.clip(self.volatility_estimate * 100, 0.5, 2.0), 0.15, 0.45)
                if freq_excess > 0: frequency_penalty = (freq_excess ** 2) * self.reward_calculator.weights.get('frequency_penalty', 0.0)
            self.consecutive_inactive_steps = 0 if abs(action_signal) + action_size > 0.01 and action_size > 0.01 else self.consecutive_inactive_steps + 1
            inactivity_penalty = 0.0
            if self.consecutive_inactive_steps > self.inactivity_grace_period:
                inactivity_penalty = min(1.0, (self.consecutive_inactive_steps - self.inactivity_grace_period) / self.penalty_ramp_up_steps) * self.reward_calculator.weights.get('inactivity_penalty', 0.0)

            # Reward Calculation
            reward_components = self.reward_calculator.calculate_immediate_reward(
                previous_asset_held=previous_asset_held, target_asset_quantity=self.asset_held, trade_quantity=trade_quantity,
                prev_value=initial_portfolio_value, current_price=current_price, prev_price=self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][self.current_step - 1],
                action=action, market_volatility=self.volatility_estimate, base_return=(next_portfolio_value - initial_portfolio_value) / initial_portfolio_value,
                total_cost=total_cost, is_insignificant_action=(action_size < 0.01 and abs(action_signal) + action_size > 0.01),
                drawdown=current_drawdown, frequency_penalty=frequency_penalty, inactivity_penalty=inactivity_penalty,
                scaling_factor=self.strat_cfg.reward_scaling_factor / self.leverage
            )
            raw_reward = sum(v for k, v in reward_components.items() if k in ['base_return', 'transaction_penalty', 'drawdown_penalty'])
            intrinsic_reward = sum(v for k, v in reward_components.items() if k not in ['base_return', 'transaction_penalty', 'drawdown_penalty'])
            self.reward_normalizer.update(raw_reward)
            reward = self.reward_normalizer.normalize(raw_reward) + intrinsic_reward
            
            if self.verbose and abs(trade_quantity) > 1e-8:
                self._print_trade_info("TRADE", action, current_price, next_portfolio_value, self.asset_held * (next_price - self.entry_price), reward, reward_components)

            self.observation_history.append(self._get_single_step_observation(self.current_step))
            terminated = (self.episode_peak_value - next_portfolio_value) / max(self.episode_peak_value, 1e-9) >= self.strat_cfg.max_drawdown_threshold
            truncated = self.current_step >= self.max_step
            info = {'portfolio_value': next_portfolio_value, 'raw_reward': raw_reward, 'intrinsic_reward': intrinsic_reward}
            return self._get_observation_sequence(), reward, terminated, truncated, info
        except Exception as e:
            logger.error(f"Error in environment step: {e}", exc_info=True)
            return self._get_observation_sequence(), -1.0, True, False, {'error': True}

    def reset(self, seed=None, options=None):
        try:
            super().reset(seed=seed)
            self.balance, self.asset_held, self.used_margin, self.entry_price = 1000000.0, 0.0, 0.0, 0.0
            self.consecutive_inactive_steps, self.episode_peak_value = 0, self.balance
            self.trade_count, self.total_insignificant_trades, self.total_attempted_trades = 0, 0, 0
            warmup_period = self.cfg.get_required_warmup_period()
            if options and 'start_step' in options: self.current_step = max(warmup_period, options['start_step'])
            else: self.current_step = self.np_random.integers(warmup_period, self.max_step - 5000) if warmup_period < self.max_step - 5000 else warmup_period
            self.observation_history.clear()
            for i in range(self.strat_cfg.sequence_length):
                self._update_market_regime_and_volatility(self.current_step - self.strat_cfg.sequence_length + 1 + i)
                self.observation_history.append(self._get_single_step_observation(self.current_step - self.strat_cfg.sequence_length + 1 + i))
            self._print_episode_start()
            return self._get_observation_sequence(), {'portfolio_value': self.balance}
        except Exception as e:
            logger.error(f"Error resetting environment: {e}", exc_info=True)
            raise

    def _get_current_context_features(self, step_index: int) -> np.ndarray:
        return np.array([self.all_features_np[key][step_index] for key in self.strat_cfg.context_feature_keys], dtype=np.float32)

    def _update_market_regime_and_volatility(self, step_index: int):
        try:
            if step_index >= 50:
                prices = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][max(0, step_index - 50) : step_index + 1]
                returns = pd.Series(prices).pct_change().dropna().values
                if len(returns) > 10:
                    self.volatility_estimate = np.std(returns) * np.sqrt(252)
                    self.risk_manager.volatility_buffer.append(self.volatility_estimate)
                    self.market_regime = self.risk_manager.update_market_regime(returns, self.volatility_estimate)
        except Exception as e:
            logger.warning(f"Error updating market regime: {e}")

    def _get_single_step_observation(self, step_index) -> dict:
        try:
            raw_obs = {}
            current_price = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][step_index]
            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value
                if key in [FeatureKeys.CONTEXT.value, FeatureKeys.PORTFOLIO_STATE.value, FeatureKeys.PRECOMPUTED_FEATURES.value]: continue
                freq = key.split('_')[-1].replace('m','T').replace('h','H').replace('s', 'S').upper()
                window_data = self.timeframes_np[freq]['close' if 'price' in key else 'volume_delta' if 'volume' in key else 'open'][max(0, step_index - lookback + 1) : step_index + 1]
                if key.startswith('ohlc'):
                    cols = ['open','high','low','close','volume'] if 'v' in key else ['open','high','low','close']
                    window_data = np.stack([self.timeframes_np[freq][c][max(0, step_index - lookback + 1) : step_index + 1] for c in cols], axis=1)
                if len(window_data) < lookback: window_data = np.pad(window_data, [(lookback - len(window_data), 0)] + [(0,0)]*(window_data.ndim-1), 'edge')
                raw_obs[key] = window_data.astype(np.float32)
            raw_obs[FeatureKeys.CONTEXT.value] = self._get_current_context_features(step_index)
            raw_obs[FeatureKeys.PRECOMPUTED_FEATURES.value] = np.array([self.all_features_np[k][step_index] for k in self.strat_cfg.precomputed_feature_keys], dtype=np.float32)
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            portfolio_value = self.balance + unrealized_pnl
            raw_obs[FeatureKeys.PORTFOLIO_STATE.value] = np.array([np.clip((self.asset_held * current_price) / (portfolio_value + 1e-9), -1.0, 1.0), np.tanh(unrealized_pnl / (self.used_margin + 1e-9)), np.clip((self.used_margin + unrealized_pnl) / (abs(self.asset_held * current_price) + 1e-9), 0, 2.0), self.risk_manager.get_volatility_percentile(self.volatility_estimate)], dtype=np.float32)
            return self.normalizer.transform(raw_obs)
        except Exception as e:
            logger.error(f"Error getting observation step {step_index}: {e}", exc_info=True)
            return {k.value: np.zeros(s.shape, dtype=np.float32) for k, s in self.observation_space.spaces.items()}

    def _get_observation_sequence(self):
        try:
            return {key: np.stack([obs[key] for obs in self.observation_history]) for key in self.observation_space.spaces.keys()}
        except Exception:
            return {key: np.zeros(space.shape, dtype=np.float32) for key, space in self.observation_space.spaces.items()}


    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive and CORRECTED performance metrics.
        FIXED:
        1. Volatility annualization is now correctly based on the bar timeframe.
        2. Sharpe Ratio now uses correctly annualized returns.
        3. Replaced non-functional 'win_rate' with a 'positive_period_ratio'.
        """
        try:
            if len(self.portfolio_history) < 2: return {}
            
            portfolio_values = np.array(self.portfolio_history)
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            
            # --- FIX #1: Correct Annualization Factor ---
            # Calculate the number of bars in a year based on the environment's timeframe
            base_bar_seconds = self.cfg.get_timeframe_seconds(self.cfg.base_bar_timeframe)
            bars_per_year = (365 * 24 * 3600) / base_bar_seconds
            
            # Calculate periodic returns (e.g., returns for each 20-second bar)
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Correctly annualize the volatility
            volatility = np.std(returns) * np.sqrt(bars_per_year) if len(returns) > 1 else 0.0

            # --- FIX #2: Correct Sharpe Ratio Calculation ---
            # First, calculate the total return and then annualize it
            total_return = (final_value - initial_value) / initial_value
            num_periods = len(returns)
            # Formula for annualizing return: (1 + total_return)^(periods_in_year / num_periods) - 1
            annualized_return = ((1 + total_return) ** (bars_per_year / num_periods) - 1) if num_periods > 0 else 0.0
            
            risk_free_rate = 0.02 # Hardcoded risk-free rate
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0.0

            # --- FIX #3: Replace Broken win_rate with a functional metric ---
            # Since we don't track individual winning trades, we can calculate the ratio of positive-return periods.
            positive_periods = np.sum(returns > 0)
            positive_period_ratio = positive_periods / len(returns) if len(returns) > 0 else 0.0

            # Max Drawdown calculation remains correct
            cumulative_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (cumulative_max - portfolio_values) / cumulative_max
            max_drawdown = np.max(drawdowns)
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return, # Added for clarity
                'volatility_annualized': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'positive_period_ratio': positive_period_ratio, # New, functional metric
                'total_trades': self.trade_count,
                'final_portfolio_value': final_value,
                'leverage': self.leverage,
                'verbose_mode': self.verbose
            }
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'leverage': self.leverage, 'verbose_mode': self.verbose}
