"""
FULLY CORRECTED AND HARDENED TRADING ENGINE

This definitive version incorporates all emergency and high-priority fixes identified
in the reward system and bug analysis reports. It is the stable, reliable foundation
for all future training.

--- CRITICAL FIXES APPLIED ---
1.  **Unrealized PnL Double-Counting ELIMINATED:** The showstopper bug that added unrealized PnL
    to the balance every step has been removed. PnL is now correctly realized only when a
    position is reduced or flipped, preventing phantom profits and ensuring valid training.

2.  **Entry Price Calculation CORRECTED:** The flawed entry price logic with the `abs()` wrapper has
    been replaced with a robust, sign-aware weighted average calculation. PnL for both long and
    short positions is now accurate.

3.  **Reward Normalization REBALANCED:** Replaced the unstable normalization strategy with a new
    `MultiComponentNormalizer`. It separately normalizes returns, penalties, and bonuses, then
    combines them with stable weights, preventing reward signal corruption across different
    market regimes.

4.  **Penalty & Bonus Logic REFINED:**
    *   **Frequency Penalty:** Simplified to a clear, predictable formula.
    *   **Exploration Bonus:** Restored full logic with gates to prevent exploitation and
      fixed the cost-based cap to be a proportional reduction.
    *   **Tiny Action Penalty:** Logic corrected to trigger reliably.

5.  **Initialization HARDENED:** Critical state variables like `volatility_estimate` are now
    safely initialized to prevent errors at the start of an episode.
"""

import numpy as np
import pandas as pd
import gymnasium
from gymnasium import spaces
from collections import deque
from typing import Dict, Any, Optional, Tuple, Sequence
import logging
from dataclasses import dataclass
from scipy import stats

# Import configuration and stateful feature calculators
from config import SETTINGS, FeatureKeys
from normalizer import Normalizer
from features import (
    StatefulBBWPercentRank,
    StatefulPriceDistanceMA,
    StatefulSRDistances,
    StatefulVWAPDistance,
)

logger = logging.getLogger(__name__)


# --- START OF NEW, STABLE REWARD ARCHITECTURE ---

class RewardComponentNormalizer:
    """Tracks running mean/std for a single reward component for stable normalization."""
    def __init__(self, epsilon: float = 1e-8):
        self.mean, self.std, self.variance, self.count, self.M2 = 0.0, 1.0, 1.0, 0, 0.0
        self.epsilon = epsilon

    def update(self, value: float):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
        if self.count > 1:
            self.variance = self.M2 / (self.count - 1) # Use sample variance
            self.std = np.sqrt(self.variance)

    def normalize(self, value: float) -> float:
        return (value - self.mean) / (self.std + self.epsilon)

class MultiComponentNormalizer:
    """
    Orchestrates normalization across different reward types (returns, penalties, bonuses)
    to ensure a balanced and stable final reward signal.
    """
    def __init__(self):
        self.return_normalizer = RewardComponentNormalizer()
        self.penalty_normalizer = RewardComponentNormalizer()
        self.bonus_normalizer = RewardComponentNormalizer()

    def normalize_reward(self, reward_components: Dict[str, float]) -> float:
        base_return = reward_components.get('base_return', 0.0)
        
        # Separate penalties and bonuses from the components
        penalties = sum(v for k, v in reward_components.items() if v < 0 and k != 'base_return')
        bonuses = sum(v for k, v in reward_components.items() if v >= 0 and k != 'base_return')

        # Update the respective online normalizers
        self.return_normalizer.update(base_return)
        if abs(penalties) > 1e-9:
            self.penalty_normalizer.update(penalties)
        if abs(bonuses) > 1e-9:
            self.bonus_normalizer.update(bonuses)

        # Normalize each component
        norm_return = self.return_normalizer.normalize(base_return)
        norm_penalty = self.penalty_normalizer.normalize(penalties) if abs(penalties) > 1e-9 else 0.0
        norm_bonus = self.bonus_normalizer.normalize(bonuses) if abs(bonuses) > 1e-9 else 0.0
        
        # Combine with stable weights to form the final reward
        # These weights prevent any single component from dominating the learning signal.
        final_reward = (0.7 * norm_return + 0.2 * norm_penalty + 0.1 * norm_bonus)
        
        return float(np.clip(final_reward, -10.0, 10.0))

@dataclass
class ExplorationConfig:
    """Configurable thresholds to tune the exploration bonus per instrument/deployment."""
    minimum_risk_commitment_pct: float = 0.01  # Must increase risk by at least 1% of portfolio
    min_trade_notional_usd: float = 1000.0     # Absolute USD floor for a trade to count
    relative_trade_cap: float = 0.5            # Max fraction of portfolio the reward scales to
    exploration_reward_cap: float = 0.1        # Max normalized exploration reward
    confidence_min_factor: float = 0.25        # Minimum multiplier from agent's confidence
    confidence_scale: float = 0.75             # How strongly confidence influences reward
    commitment_scaler_denom: float = 5.0       # Divisor scaling the commitment factor
    volatility_scaler_min: float = 0.5
    volatility_scaler_max: float = 2.0

class RewardCalculator:
    """
    Calculates all raw reward components before normalization.
    This version includes the full, sophisticated exploration bonus logic.
    """
    def __init__(self, weights: Optional[dict] = None, exploration_cfg: Optional[ExplorationConfig] = None):
        # Recommended balanced weights from the analysis
        self.weights = weights or {
            'base_return': 3.0,
            'transaction_penalty': -0.3,
            'drawdown_penalty': -2.0,
            'frequency_penalty': -1.0,
            'inactivity_penalty': -0.4,
            'tiny_action_penalty': -0.5,
            'exploration_bonus': 0.05,
        }
        self.exploration_cfg = exploration_cfg or ExplorationConfig()

    def calculate_exploration_bonus_safe(
        self,
        previous_asset_held: float,
        target_asset_quantity: float,
        trade_quantity: float,
        previous_portfolio_value: float,
        current_price: float,
        action: Sequence[float],
        market_volatility: float,
        estimated_costs: float,
    ) -> float:
        """
        Computes a conservative exploration bonus, preventing exploits by requiring
        the agent to take meaningful, new risk.
        """
        # --- 1. Gating Conditions: Does this trade even qualify for a bonus? ---
        prev_value = max(float(previous_portfolio_value), 1e-9)
        
        previous_exposure_notional = abs(previous_asset_held) * current_price # Use current price for apple-to-apples
        target_exposure_notional = abs(target_asset_quantity) * current_price
        trade_notional = abs(trade_quantity) * current_price
        
        # Gate 1: Did the agent actually increase its risk exposure?
        increased_exposure = (target_exposure_notional - previous_exposure_notional) > 1e-6
        
        # Gate 2: Was the increase in risk significant relative to the portfolio?
        risk_commitment_pct = (target_exposure_notional - previous_exposure_notional) / prev_value
        passed_risk_gate = risk_commitment_pct > self.exploration_cfg.minimum_risk_commitment_pct

        # Gate 3: Was the trade of a meaningful absolute size?
        passed_notional_gate = trade_notional >= self.exploration_cfg.min_trade_notional_usd
        
        if not (increased_exposure and passed_risk_gate and passed_notional_gate):
            return 0.0 # Trade does not qualify, bonus is zero.

        # --- 2. Calculate Bonus Magnitude for Qualified Trades ---
        cfg = self.exploration_cfg
        
        # Scale reward by the size of the trade relative to the portfolio
        relative_trade = np.clip(trade_notional / prev_value, 0.0, cfg.relative_trade_cap)
        
        # Scale reward by the agent's confidence (the action signal)
        position_confidence = float(abs(action[0]))
        confidence_factor = np.clip(cfg.confidence_min_factor + cfg.confidence_scale * position_confidence, cfg.confidence_min_factor, 1.0)
        
        # Scale reward by market volatility (more reward for taking risk in riskier conditions)
        volatility_scaler = np.clip(market_volatility * 50.0, cfg.volatility_scaler_min, cfg.volatility_scaler_max)
        
        # Scale reward by how much the agent exceeded the minimum risk commitment
        commitment_scaler = np.clip(risk_commitment_pct / (cfg.minimum_risk_commitment_pct * cfg.commitment_scaler_denom), 0.5, 2.0)
        
        # Combine all factors
        exploration_reward = relative_trade * 0.5 * volatility_scaler * confidence_factor * commitment_scaler
        exploration_reward = float(np.clip(exploration_reward, 0.0, cfg.exploration_reward_cap))

        # --- 3. FIXED: Apply Proportional Cost Reduction (Issue #2) ---
        cost_ratio = estimated_costs / prev_value
        cost_penalty = np.clip(cost_ratio * 2.0, 0.0, 0.8)  # Max 80% reduction
        exploration_reward *= (1.0 - cost_penalty)
        
        return exploration_reward

    def calculate_raw_components(self, **kwargs) -> Dict[str, float]:
        """Computes all raw, unnormalized reward components for a given step."""
        components = {}
        prev_value = max(kwargs.get('prev_value', 1e-6), 1e-6)

        # Base market return (extrinsic)
        components['base_return'] = kwargs.get('base_return', 0.0) * self.weights['base_return']
        
        # Penalties (intrinsic)
        normalized_cost = kwargs.get('total_cost', 0.0) / prev_value
        components['transaction_penalty'] = normalized_cost * self.weights['transaction_penalty']
        
        current_drawdown = kwargs.get('drawdown', 0.0)
        if current_drawdown > 0.08:
            drawdown_excess = current_drawdown - 0.08
            penalty_factor = min(1.0, drawdown_excess / 0.12)
            components['drawdown_penalty'] = penalty_factor * self.weights['drawdown_penalty']

        if kwargs.get('is_insignificant_action', False):
            components['tiny_action_penalty'] = self.weights['tiny_action_penalty']
        
        components['frequency_penalty'] = kwargs.get('frequency_penalty', 0.0)
        components['inactivity_penalty'] = kwargs.get('inactivity_penalty', 0.0)
        
        # Exploration Bonus (intrinsic)
        bonus = self.calculate_exploration_bonus_safe(
            previous_asset_held=kwargs['previous_asset_held'],
            target_asset_quantity=kwargs['target_asset_quantity'],
            trade_quantity=kwargs['trade_quantity'],
            previous_portfolio_value=prev_value,
            current_price=kwargs['current_price'],
            action=kwargs['action'],
            market_volatility=kwargs['market_volatility'],
            estimated_costs=kwargs.get('total_cost', 0.0)
        )
        components['exploration_bonus'] = bonus * self.weights['exploration_bonus']
        
        return components

# --- END OF NEW, STABLE REWARD ARCHITECTURE ---


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
    """FULLY CORRECTED Trading environment with hardened PnL, entry price, and reward logic."""
    def __init__(self, df_base_ohlc: pd.DataFrame, normalizer: Normalizer, config=None,
                 leverage: float = None, reward_weights: Dict[str, float] = None,
                 precomputed_features: Optional[pd.DataFrame] = None,
                 worker_id: int = 0):
        super().__init__()
        self.worker_id, self.verbose = worker_id, worker_id == 0
        self.COLOR_GREEN, self.COLOR_RED, self.COLOR_YELLOW, self.COLOR_CYAN, self.COLOR_RESET, self.COLOR_BOLD = '\033[92m', '\033[91m', '\033[93m', '\033[96m', '\033[0m', '\033[1m'
        self.cfg, self.strat_cfg, self.normalizer = config or SETTINGS, (config or SETTINGS).strategy, normalizer
        self.leverage = leverage if leverage is not None else self.strat_cfg.leverage
        
        try:
            self.risk_manager = EnhancedRiskManager(self.cfg, leverage=self.leverage)
            self.reward_calculator = RewardCalculator(weights=reward_weights)
            self.multi_component_normalizer = MultiComponentNormalizer()
            
            self.action_frequency_buffer = deque(maxlen=100)
            self.inactivity_grace_period = getattr(self.strat_cfg, 'inactivity_grace_period_steps', 10)
            
            base_df = df_base_ohlc.set_index('timestamp')
            all_required_freqs = set(k.value.split('_')[-1].replace('m','T').replace('h','H').replace('s','S').upper() for k in self.strat_cfg.lookback_periods.keys() if k not in {FeatureKeys.CONTEXT, FeatureKeys.PORTFOLIO_STATE, FeatureKeys.PRECOMPUTED_FEATURES}).union({c.timeframe for c in self.strat_cfg.stateful_calculators})
            self.base_timestamps = base_df.resample(self.cfg.base_bar_timeframe.value).asfreq().index
            self.max_step = len(self.base_timestamps) - 2
            
            self.timeframes_np: Dict[str, Dict[str, np.ndarray]] = {}
            for freq in all_required_freqs:
                agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'trade_count': 'sum', 'volume_delta': 'sum'}
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

    def _print_trade_info(self, trade_type: str, current_price: float, portfolio_value: float, unrealized_pnl: float, reward: float, raw_components: Dict[str, float]):
        if not self.verbose: return
        self.trade_counter += 1
        pnl_color, reward_color = (self.COLOR_GREEN if x >= 0 else self.COLOR_RED for x in [unrealized_pnl, reward])
        print(f"\n{self.COLOR_BOLD}{'='*80}{self.COLOR_RESET}\n{self.COLOR_CYAN}🚀 TRADE #{self.trade_counter} | Worker {self.worker_id} | Step {self.current_step}{self.COLOR_RESET}")
        print(f"📈 Price: {self.COLOR_YELLOW}${current_price:,.2f}{self.COLOR_RESET} | Entry: ${self.entry_price:,.2f} | Portfolio: ${portfolio_value:,.2f}")
        print(f"💰 PnL: {pnl_color}${unrealized_pnl:+,.2f}{self.COLOR_RESET} | Reward: {reward_color}{reward:+.4f}{self.COLOR_RESET}")
        significant_components = {k: v for k, v in raw_components.items() if abs(v) > 1e-5}
        if significant_components:
            print("🔍 Key Raw Reward Components:" + "".join(f"\n   • {c}: {(self.COLOR_GREEN if v >= 0 else self.COLOR_RED)}{v:+.4f}{self.COLOR_RESET}" for c, v in sorted(significant_components.items(), key=lambda x: abs(x[1]), reverse=True)[:5]))
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
            
            # --- 1. Calculate initial state for PnL and drawdown ---
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            initial_portfolio_value = self.balance + unrealized_pnl
            if initial_portfolio_value > self.episode_peak_value: self.episode_peak_value = initial_portfolio_value
            current_drawdown = (self.episode_peak_value - initial_portfolio_value) / max(self.episode_peak_value, 1e-9)
            
            # --- 2. Determine Action & Target Position ---
            action_signal, action_size = np.clip(action[0], -1.0, 1.0), np.clip(action[1], 0.0, 1.0)
            target_asset_quantity = 0.0
            if action_size >= 0.01:
                max_position = action_size * self.risk_manager.calculate_dynamic_position_limit(self.volatility_estimate, initial_portfolio_value, self.market_regime)
                target_notional = initial_portfolio_value * np.sign(action_signal) * max_position
                target_asset_quantity = target_notional / max(current_price, 1e-8)
            
            trade_quantity = target_asset_quantity - self.asset_held
            previous_asset_held = self.asset_held
            
            # --- 3. EMERGENCY FIX #1: Correctly Calculate Realized PnL ---
            realized_pnl = 0.0
            if abs(previous_asset_held) > 1e-9 and abs(trade_quantity) > 1e-9:
                is_reducing = abs(target_asset_quantity) < abs(previous_asset_held)
                is_flipping = np.sign(target_asset_quantity) * np.sign(previous_asset_held) < 0
                if is_reducing or is_flipping:
                    closed_quantity = abs(previous_asset_held) if is_flipping else abs(previous_asset_held) - abs(target_asset_quantity)
                    realized_pnl = np.sign(previous_asset_held) * closed_quantity * (current_price - self.entry_price)
            
            # --- 4. EMERGENCY FIX #1 (cont.): Update Balance ONLY with Realized PnL ---
            self.balance += realized_pnl # DO NOT add unrealized PnL here.
            
            # --- 5. EMERGENCY FIX #2: Correctly Update Entry Price ---
            if abs(target_asset_quantity) < 1e-9:
                self.entry_price = 0.0
            elif previous_asset_held == 0 or np.sign(target_asset_quantity) * np.sign(previous_asset_held) < 0:
                self.entry_price = current_price # New position or flip
            elif abs(target_asset_quantity) > abs(previous_asset_held):
                # Adding to position - weighted average WITHOUT abs()
                total_cost = (previous_asset_held * self.entry_price) + (trade_quantity * current_price)
                if abs(target_asset_quantity) > 1e-9:
                    self.entry_price = total_cost / target_asset_quantity
            # On position reduction, entry price remains the same.
            
            self.asset_held = target_asset_quantity
            if abs(self.asset_held) < 1e-9: self.entry_price = 0.0
            
            # --- 6. Calculate Costs & Final State ---
            total_cost = abs(trade_quantity) * current_price * (self.cfg.transaction_fee_pct + self.cfg.slippage_pct)
            self.balance -= total_cost
            
            self.current_step += 1
            next_price = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][self.current_step]
            next_unrealized_pnl = self.asset_held * (next_price - self.entry_price)
            next_portfolio_value = self.balance + next_unrealized_pnl

            # --- 7. Calculate Penalties with Refined Logic ---
            # Simplified Frequency Penalty (Issue #3)
            self.action_frequency_buffer.append(1 if abs(trade_quantity) > 1e-8 else 0)
            frequency_penalty = 0.0
            if len(self.action_frequency_buffer) == 100:
                base_freq_threshold = 0.3
                annualized_vol = self.volatility_estimate * np.sqrt(252)
                vol_adjustment = np.clip(annualized_vol, 0.1, 0.4)
                allowed_frequency = base_freq_threshold + vol_adjustment
                current_frequency = np.mean(list(self.action_frequency_buffer))
                if current_frequency > allowed_frequency:
                    freq_excess = current_frequency - allowed_frequency
                    frequency_penalty = (freq_excess ** 1.5) * self.reward_calculator.weights.get('frequency_penalty', 0.0)

            # Corrected Tiny Action Penalty (Issue #4)
            total_action_magnitude = abs(action_signal) * action_size
            is_insignificant = 0.0 < total_action_magnitude < 0.005

            # Inactivity Penalty
            self.consecutive_inactive_steps = 0 if abs(trade_quantity) > 1e-8 else self.consecutive_inactive_steps + 1
            inactivity_penalty = 0.0
            if self.consecutive_inactive_steps > self.inactivity_grace_period:
                inactivity_penalty = self.reward_calculator.weights.get('inactivity_penalty', 0.0)

            # --- 8. Calculate Final Reward with New Balanced Normalization (Issue #1) ---
            raw_components = self.reward_calculator.calculate_raw_components(
                base_return=(next_portfolio_value - initial_portfolio_value) / max(initial_portfolio_value, 1e-9),
                total_cost=total_cost,
                prev_value=initial_portfolio_value,
                drawdown=current_drawdown,
                is_insignificant_action=is_insignificant,
                frequency_penalty=frequency_penalty,
                inactivity_penalty=inactivity_penalty,
                # Pass necessary args for the full exploration bonus
                previous_asset_held=previous_asset_held,
                target_asset_quantity=self.asset_held,
                trade_quantity=trade_quantity,
                current_price=current_price,
                action=action,
                market_volatility=self.volatility_estimate,
            )
            reward = self.multi_component_normalizer.normalize_reward(raw_components)

            if self.verbose and abs(trade_quantity) > 1e-8:
                self._print_trade_info("TRADE", current_price, next_portfolio_value, next_unrealized_pnl, reward, raw_components)

            self.observation_history.append(self._get_single_step_observation(self.current_step))
            terminated = next_portfolio_value <= self.initial_balance * (1.0 - self.strat_cfg.max_drawdown_threshold)
            truncated = self.current_step >= self.max_step
            info = {
                'portfolio_value': next_portfolio_value,
                'raw_reward_components': raw_components,
                'final_reward': reward
            }
            return self._get_observation_sequence(), reward, terminated, truncated, info
        except Exception as e:
            logger.error(f"Error in environment step: {e}", exc_info=True)
            return self._get_observation_sequence(), -1.0, True, False, {'error': True}

    def reset(self, seed=None, options=None):
        try:
            super().reset(seed=seed)
            self.initial_balance = 1000000.0
            self.balance = self.initial_balance
            self.asset_held, self.entry_price = 0.0, 0.0
            self.episode_peak_value = self.balance
            self.consecutive_inactive_steps = 0
            
            # EMERGENCY FIX #3: Initialize critical variables
            self.volatility_estimate = 0.01
            self.market_regime = "UNCERTAIN"
            self.portfolio_history = []
            
            self.trade_counter = 0
            
            warmup_period = self.cfg.get_required_warmup_period()
            start_step_range = (warmup_period, self.max_step - 5000)
            if options and 'start_step' in options:
                self.current_step = max(start_step_range[0], min(options['start_step'], start_step_range[1]))
            else:
                self.current_step = self.np_random.integers(start_step_range[0], start_step_range[1]) if start_step_range[0] < start_step_range[1] else start_step_range[0]
            
            self.observation_history.clear()
            for i in range(self.strat_cfg.sequence_length):
                step_idx = self.current_step - self.strat_cfg.sequence_length + 1 + i
                self._update_market_regime_and_volatility(step_idx)
                self.observation_history.append(self._get_single_step_observation(step_idx))
            
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
                    base_bar_seconds = self.cfg.get_timeframe_seconds(self.cfg.base_bar_timeframe)
                    bars_per_day = (24 * 3600) / base_bar_seconds
                    annualization_factor = np.sqrt(bars_per_day) # Daily vol
                    self.volatility_estimate = np.std(returns) * annualization_factor
                    self.risk_manager.volatility_buffer.append(self.volatility_estimate)
                    self.market_regime = self.risk_manager.update_market_regime(returns, self.volatility_estimate)
        except Exception as e:
            logger.warning(f"Error updating market regime at step {step_index}: {e}")

    def _get_single_step_observation(self, step_index) -> dict:
        try:
            raw_obs = {}
            current_price = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][step_index]
            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value
                if key in [FeatureKeys.CONTEXT.value, FeatureKeys.PORTFOLIO_STATE.value, FeatureKeys.PRECOMPUTED_FEATURES.value]: continue
                freq = key.split('_')[-1].replace('m','T').replace('h','H').replace('s', 'S').upper()
                
                if key.startswith('ohlc'):
                    cols = ['open','high','low','close','volume'] if 'v' in key else ['open','high','low','close']
                    window_data = np.stack([self.timeframes_np[freq][c][max(0, step_index - lookback + 1) : step_index + 1] for c in cols], axis=1)
                else:
                    col_name = 'volume_delta' if 'volume_delta' in key else 'close'
                    window_data = self.timeframes_np[freq][col_name][max(0, step_index - lookback + 1) : step_index + 1]
                
                if len(window_data) < lookback: window_data = np.pad(window_data, [(lookback - len(window_data), 0)] + [(0,0)]*(window_data.ndim-1), 'edge')
                raw_obs[key] = window_data.astype(np.float32)
            
            raw_obs[FeatureKeys.CONTEXT.value] = self._get_current_context_features(step_index)
            raw_obs[FeatureKeys.PRECOMPUTED_FEATURES.value] = np.array([self.all_features_np[k][step_index] for k in self.strat_cfg.precomputed_feature_keys], dtype=np.float32)
            
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            portfolio_value = self.balance + unrealized_pnl
            position_value = self.asset_held * current_price
            
            raw_obs[FeatureKeys.PORTFOLIO_STATE.value] = np.array([
                np.clip(position_value / (portfolio_value + 1e-9), -self.leverage, self.leverage),
                np.tanh(unrealized_pnl / (portfolio_value + 1e-9)),
                self.risk_manager.get_volatility_percentile(self.volatility_estimate),
                (self.episode_peak_value - portfolio_value) / max(self.episode_peak_value, 1e-9)
            ], dtype=np.float32)
            
            return self.normalizer.transform(raw_obs)
        except Exception as e:
            logger.error(f"Error getting observation at step {step_index}: {e}", exc_info=True)
            return {k: np.zeros(s.shape, dtype=np.float32) for k, s in self.observation_space.spaces.items()}

    def _get_observation_sequence(self):
        try:
            return {key: np.stack([obs[key] for obs in self.observation_history]) for key in self.observation_space.spaces.keys()}
        except Exception:
            return {key: np.zeros(space.shape, dtype=np.float32) for key, space in self.observation_space.spaces.items()}


    def get_performance_metrics(self) -> Dict[str, Any]:
        try:
            if len(self.portfolio_history) < 2: return {}
            portfolio_values = np.array(self.portfolio_history)
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            base_bar_seconds = self.cfg.get_timeframe_seconds(self.cfg.base_bar_timeframe)
            bars_per_year = (365 * 24 * 3600) / base_bar_seconds
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            volatility = np.std(returns) * np.sqrt(bars_per_year) if len(returns) > 1 else 0.0
            total_return = (final_value - initial_value) / initial_value
            num_periods = len(returns)
            annualized_return = ((1 + total_return) ** (bars_per_year / num_periods) - 1) if num_periods > 0 else 0.0
            risk_free_rate = 0.02
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0.0
            positive_periods = np.sum(returns > 0)
            positive_period_ratio = positive_periods / len(returns) if len(returns) > 0 else 0.0
            cumulative_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (cumulative_max - portfolio_values) / cumulative_max
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility_annualized': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'positive_period_ratio': positive_period_ratio,
                'final_portfolio_value': final_value,
            }
            return metrics
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
