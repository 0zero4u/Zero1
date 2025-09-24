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


# --- REDESIGNED, ROBUST REWARD ARCHITECTURE ---

class RobustComponentNormalizer:
    """
    Handles the full normalization pipeline for a single reward component.
    - Clips raw values to prevent outlier distortion.
    - Uses a warmup period before normalizing.
    - Calculates a running Z-score.
    - Applies a sign-preserving tanh squash to bound the output.
    """
    def __init__(self, warmup_steps: int = 250, clip_range: Tuple[float, float] = (-10.0, 10.0), epsilon: float = 1e-8):
        self.warmup_steps = warmup_steps
        self.clip_range = clip_range
        self.epsilon = epsilon
        self.mean, self.std, self.variance, self.count, self.M2 = 0.0, 1.0, 1.0, 0, 0.0

    def update(self, value: float):
        """Update running mean and std using Welford's algorithm."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
        if self.count > 1:
            self.variance = self.M2 / (self.count - 1)
            self.std = np.sqrt(self.variance)

    def normalize(self, raw_value: float) -> float:
        """Full normalization pipeline for a given raw component value."""
        clipped_value = np.clip(raw_value, self.clip_range[0], self.clip_range[1])
        self.update(clipped_value)
        if self.count < self.warmup_steps:
            return clipped_value
        z_score = (clipped_value - self.mean) / (self.std + self.epsilon)
        return np.tanh(z_score)

class RewardManager:
    """
    Orchestrates the entire reward calculation process:
    1.  Takes raw environment data from a step.
    2.  Calculates raw, untransformed reward components.
    3.  Pre-transforms raw values into meaningful, unitless ratios.
    4.  Normalizes each transformed component independently.
    5.  Applies final weights and sums to get the final reward.
    """
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        self.normalizers = {key: RobustComponentNormalizer() for key in weights.keys()}
        # --- START OF HESITATION PENALTY FIX (1/3) ---
        # Added 'hesitation' to the set of penalties.
        self.penalty_components = {'trade_cost', 'drawdown', 'frequency', 'tiny_action', 'inactivity', 'thrashing', 'hesitation'}
        # --- END OF HESITATION PENALTY FIX (1/3) ---
        self.REWARD_SCALE_FACTOR = 10000.0

    def _calculate_raw_components(self, **kwargs) -> Dict[str, float]:
        """Calculates raw, physical values from the environment step."""
        components = {}
        initial_value = max(kwargs.get('initial_portfolio_value', 1e-9), 1e-9)
        
        components['unrealized_pnl_shaping'] = (kwargs['next_portfolio_value'] - initial_value)
        components['realized_pnl'] = kwargs.get('realized_pnl', 0.0)
        
        components['trade_cost'] = kwargs.get('total_cost', 0.0)
        components['drawdown'] = kwargs.get('drawdown', 0.0)
        components['inactivity_steps'] = kwargs.get('consecutive_inactive_steps', 0)
        
        trade_frequency = np.mean(list(kwargs.get('action_frequency_buffer', [0])))
        if trade_frequency > 0.4:
            components['frequency'] = (trade_frequency - 0.4) ** 2
        else:
            components['frequency'] = 0.0

        # --- START OF HESITATION PENALTY FIX (2/3) ---
        # This revised logic correctly distinguishes between a "tiny_action" and a "hesitation" action.
        action_signal, action_size = kwargs['action'][0], kwargs['action'][1]
        
        # Initialize penalties to zero
        components['tiny_action'] = 0.0
        components['hesitation'] = 0.0

        # Check if the action size is in the non-executable "dead zone" (0 < size < 0.01)
        if 0.0 < action_size < 0.01:
            total_action_magnitude = abs(action_signal) * action_size
            if total_action_magnitude < 0.005:
                # This is a truly insignificant action in both size and signal.
                components['tiny_action'] = 1.0
            else:
                # This is the "glove tap" loophole: size is small, but the intended magnitude was not tiny.
                components['hesitation'] = 1.0
        # --- END OF HESITATION PENALTY FIX (2/3) ---

        # Action clarity bonus - rewards decisive actions to break turtling behavior
        components['action_clarity'] = 0.1 if action_size > 0.25 else 0.0

        thrashing_ratio = kwargs.get('UNCERTAIN_ratio', 0.0)
        if thrashing_ratio > 0.4:  # Penalize when more than 40% of trades are flips
            components['thrashing'] = (thrashing_ratio - 0.4)**2
        else:
            components['thrashing'] = 0.0

        return components

    def _transform_to_ratios(self, raw_components: Dict[str, float], **kwargs) -> Dict[str, float]:
        """
        CRITICAL STEP: Transforms raw values into unitless ratios before normalization.
        """
        ratios = {}
        initial_value = max(kwargs.get('initial_portfolio_value', 1e-9), 1e-9)

        ratios['unrealized_pnl_shaping'] = (raw_components['unrealized_pnl_shaping'] / initial_value) * self.REWARD_SCALE_FACTOR
        ratios['realized_pnl'] = (raw_components['realized_pnl'] / initial_value) * self.REWARD_SCALE_FACTOR
        
        ratios['trade_cost'] = (-raw_components['trade_cost'] / (initial_value + self.normalizers['trade_cost'].epsilon)) * self.REWARD_SCALE_FACTOR
        
        ratios['drawdown'] = -raw_components['drawdown'] * self.REWARD_SCALE_FACTOR
        ratios['frequency'] = -raw_components['frequency'] * self.REWARD_SCALE_FACTOR
        ratios['tiny_action'] = -raw_components['tiny_action'] * self.REWARD_SCALE_FACTOR
        ratios['thrashing'] = -raw_components['thrashing'] * self.REWARD_SCALE_FACTOR
        
        # --- START OF HESITATION PENALTY FIX (3/3) ---
        # Add the transformation for the new penalty.
        if 'hesitation' in raw_components:
            ratios['hesitation'] = -raw_components['hesitation'] * self.REWARD_SCALE_FACTOR
        # --- END OF HESITATION PENALTY FIX (3/3) ---
        
        inactivity_penalty = 0.0
        if raw_components['inactivity_steps'] > 10:
            inactivity_penalty = (raw_components['inactivity_steps'] - 10) / 100.0
        ratios['inactivity'] = -inactivity_penalty * self.REWARD_SCALE_FACTOR
        
        return ratios

    def calculate_final_reward(self, **kwargs) -> Tuple[float, Dict[str, Any]]:
        """The main method called by the environment at each step."""
        raw_components = self._calculate_raw_components(**kwargs)
        transformed_ratios = self._transform_to_ratios(raw_components, **kwargs)

        final_reward = 0.0
        weighted_rewards = {}
        normalized_rewards = {}

        for key, weight in self.weights.items():
            if key in transformed_ratios and transformed_ratios[key] != 0:
                norm_value = self.normalizers[key].normalize(transformed_ratios[key])

                if key in self.penalty_components:
                    norm_value = min(0.0, norm_value)
                elif key == 'unrealized_pnl_shaping' and transformed_ratios['unrealized_pnl_shaping'] < 0:
                    norm_value = min(0.0, norm_value)

                weighted_value = weight * norm_value
                final_reward += weighted_value
                normalized_rewards[key] = norm_value
                weighted_rewards[key] = weighted_value

        final_reward = float(np.clip(final_reward, -10.0, 10.0))

        info = {
            'raw_reward_components': raw_components,
            'transformed_ratios': transformed_ratios,
            'normalized_rewards': normalized_rewards,
            'weighted_rewards': weighted_rewards,
            'final_reward': final_reward
        }
        return final_reward, info

# --- END OF REDESIGNED REWARD ARCHITECTURE ---

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
        # --- START OF FIX ---
        # Removed the stray 'sel' at the end of this line, which was a syntax error.
        return stats.percentileofscore(list(self.volatility_buffer), current_volatility) / 100.0
        # --- END OF FIX ---

class FixedHierarchicalTradingEnvironment(gymnasium.Env):
    """FULLY CORRECTED Trading environment with hardened PnL, entry price, and REDESIGNED reward logic."""
    def __init__(self, df_base_ohlc: pd.DataFrame, normalizer: Normalizer, config=None,
                  leverage: float = None, reward_weights: Dict[str, float] = None,
                 precomputed_features: Optional[pd.DataFrame] = None,
                 worker_id: int = 0):
        super().__init__()
        
        try:
            self.worker_id, self.verbose = worker_id, worker_id == 0
            self.COLOR_GREEN, self.COLOR_RED, self.COLOR_YELLOW, self.COLOR_CYAN, self.COLOR_RESET, self.COLOR_BOLD = '\033[92m', '\033[91m', '\033[93m', '\033[96m', '\033[0m', '\033[1m'
            self.cfg, self.strat_cfg, self.normalizer = config or SETTINGS, (config or SETTINGS).strategy, normalizer
            self.leverage = leverage if leverage is not None else self.strat_cfg.leverage
            
            self.risk_manager = EnhancedRiskManager(self.cfg, leverage=self.leverage)
            
            # --- REWARD SYSTEM INTEGRATION ---
            default_reward_weights = {
                'realized_pnl': 3.0,
                'unrealized_pnl_shaping': 0.1,
                'trade_cost': 0.75,
                'drawdown': 1.5,
                'thrashing': 2.0,
                'frequency': 1.0,
                'inactivity': 1.2,
                'tiny_action': 1.3,
                'action_clarity': 0.15,
                # The trainer will add 'hesitation' here if it's in the hyperparameters
            }
            final_reward_weights = reward_weights if reward_weights is not None else default_reward_weights
            self.reward_manager = RewardManager(weights=final_reward_weights)
            
            self.action_frequency_buffer = deque(maxlen=100)
            self.position_history_buffer = deque(maxlen=20)
            
            base_df = df_base_ohlc.set_index('timestamp')
            
            all_required_freqs = set(k.value.split('_')[-1] for k in self.strat_cfg.lookback_periods.keys() if k not in {FeatureKeys.CONTEXT, FeatureKeys.PORTFOLIO_STATE, FeatureKeys.PRECOMPUTED_FEATURES}).union({c.timeframe for c in self.strat_cfg.stateful_calculators})
            
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
            
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32), 
                high=np.array([1.0, 1.0], dtype=np.float32), 
                dtype=np.float32
            )

            obs_spaces = {k.value: spaces.Box(low=-np.inf, high=np.inf, shape=((self.strat_cfg.sequence_length, l, 5) if k.value.startswith('ohlcv_') else (self.strat_cfg.sequence_length, l, 4) if k.value.startswith('ohlc_') else (self.strat_cfg.sequence_length, l)), dtype=np.float32) for k, l in self.strat_cfg.lookback_periods.items()}
            self.observation_space = spaces.Dict(obs_spaces)
            self.observation_history = deque(maxlen=self.strat_cfg.sequence_length)
            
            self.reset()
        except Exception as e:
            logger.exception(f"FATAL ERROR in Worker {worker_id} during environment initialization. This worker will crash.")
            raise e

    def _print_trade_info(self, trade_type: str, current_price: float, portfolio_value: float, unrealized_pnl: float, reward: float, raw_components: Dict[str, float]):
        if not self.verbose: return
        self.trade_counter += 1
        pnl_color, reward_color = (self.COLOR_GREEN if x >= 0 else self.COLOR_RED for x in [unrealized_pnl, reward])
        print(f"\n{self.COLOR_BOLD}{'='*80}{self.COLOR_RESET}\n{self.COLOR_CYAN}🚀 TRADE #{self.trade_counter} | Worker {self.worker_id} | Step {self.current_step}{self.COLOR_RESET}")
        print(f"📈 Price: {self.COLOR_YELLOW}${current_price:,.2f}{self.COLOR_RESET} | Entry: ${self.entry_price:,.2f} | Portfolio: ${portfolio_value:,.2f}")
        print(f"💰 PnL: {pnl_color}${unrealized_pnl:+,.2f}{self.COLOR_RESET} | Reward: {reward_color}{reward:+.4f}{self.COLOR_RESET}")
        significant_components = {k: v for k, v in raw_components.items() if abs(v) > 1e-5}
        if significant_components:
            print("🔍 Key Weighted Reward Components:" + "".join(f"\n   • {c}: {(self.COLOR_GREEN if v >= 0 else self.COLOR_RED)}{v:+.4f}{self.COLOR_RESET}" for c, v in sorted(significant_components.items(), key=lambda x: abs(x[1]), reverse=True)[:5]))
        print(f"{self.COLOR_BOLD}{'='*80}{self.COLOR_RESET}\n")

    def _print_hold_info(self, step: int, price: float, portfolio_value: float, unrealized_pnl: float, action: np.ndarray):
        """Prints a log when the agent holds its position or takes an insignificant action."""
        if not self.verbose: return
        pnl_color = self.COLOR_GREEN if unrealized_pnl >= 0 else self.COLOR_RED
        action_signal, action_size = action[0], action[1]
        
        log_message = (
            f"{self.COLOR_YELLOW}⏳ HOLD | Step {step:<7} | "
            f"Price: ${price:,.2f} | "
            f"Portfolio: ${portfolio_value:,.2f} | "
            f"PnL: {pnl_color}${unrealized_pnl:+.2f}{self.COLOR_RESET} | "
            f"Action (Sig/Size): {action_signal:+.2f}/{action_size:.2f}"
        )
        print(log_message)

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
            
            action_signal, action_size = np.clip(action[0], -1.0, 1.0), np.clip(action[1], 0.0, 1.0)
            
            action_magnitude = abs(action_signal) * action_size
            if action_size >= 0.01:
                self.total_attempted_trades += 1
            elif 0 < action_size < 0.01:
                self.total_insignificant_trades += 1

            target_asset_quantity = 0.0
            if action_size >= 0.01:
                max_position = action_size * self.risk_manager.calculate_dynamic_position_limit(self.volatility_estimate, initial_portfolio_value, self.market_regime)
                target_notional = initial_portfolio_value * np.sign(action_signal) * max_position
                target_asset_quantity = target_notional / max(current_price, 1e-8)
            
            trade_quantity = target_asset_quantity - self.asset_held
            previous_asset_held = self.asset_held
            
            realized_pnl = 0.0
            if abs(previous_asset_held) > 1e-9 and abs(trade_quantity) > 1e-9:
                is_reducing = abs(target_asset_quantity) < abs(previous_asset_held)
                is_flipping = np.sign(target_asset_quantity) * np.sign(previous_asset_held) < 0
                if is_reducing or is_flipping:
                    closed_quantity = abs(previous_asset_held) if is_flipping else abs(previous_asset_held) - abs(target_asset_quantity)
                    realized_pnl = np.sign(previous_asset_held) * closed_quantity * (current_price - self.entry_price)
            
            self.balance += realized_pnl
            
            if abs(target_asset_quantity) < 1e-9:
                self.entry_price = 0.0
            elif previous_asset_held == 0 or np.sign(target_asset_quantity) * np.sign(previous_asset_held) < 0:
                self.entry_price = current_price
            elif abs(target_asset_quantity) > abs(previous_asset_held):
                total_cost = (previous_asset_held * self.entry_price) + (trade_quantity * current_price)
                if abs(target_asset_quantity) > 1e-9:
                    self.entry_price = total_cost / target_asset_quantity
            
            self.asset_held = target_asset_quantity
            if abs(self.asset_held) < 1e-9: self.entry_price = 0.0
            
            total_cost = abs(trade_quantity) * current_price * (self.cfg.transaction_fee_pct + self.cfg.slippage_pct)
            self.balance -= total_cost
            
            self.current_step += 1
            next_price = self.timeframes_np[self.cfg.base_bar_timeframe.value]['close'][self.current_step]
            next_unrealized_pnl = self.asset_held * (next_price - self.entry_price)
            next_portfolio_value = self.balance + next_unrealized_pnl

            self.portfolio_history.append(next_portfolio_value)

            is_executed_trade = abs(trade_quantity) > 1e-8
            if is_executed_trade:
                self.total_executed_trades += 1
                current_position_sign = np.sign(self.asset_held)
                
                if len(self.position_history_buffer) > 0:
                    last_position_sign = self.position_history_buffer[-1]
                    if current_position_sign != last_position_sign and last_position_sign != 0:
                        self.total_flips += 1
                
                self.position_history_buffer.append(current_position_sign)
            
            thrashing_ratio = self.total_flips / self.total_executed_trades if self.total_executed_trades > 0 else 0.0

            self.action_frequency_buffer.append(1 if is_executed_trade else 0)
            self.consecutive_inactive_steps = 0 if is_executed_trade else self.consecutive_inactive_steps + 1

            reward, reward_info = self.reward_manager.calculate_final_reward(
                initial_portfolio_value=initial_portfolio_value,
                next_portfolio_value=next_portfolio_value,
                total_cost=total_cost,
                trade_quantity=trade_quantity,
                current_price=current_price,
                drawdown=current_drawdown,
                action=action,
                action_frequency_buffer=self.action_frequency_buffer,
                consecutive_inactive_steps=self.consecutive_inactive_steps,
                realized_pnl=realized_pnl,
                thrashing_ratio=thrashing_ratio
            )

            raw_components_for_print = reward_info.get('weighted_rewards', {})
            if self.verbose and is_executed_trade:
                self._print_trade_info("TRADE", current_price, next_portfolio_value, next_unrealized_pnl, reward, raw_components_for_print)
            elif self.verbose and (self.consecutive_inactive_steps == 1 or self.consecutive_inactive_steps % 50 == 0):
                self._print_hold_info(self.current_step, next_price, next_portfolio_value, next_unrealized_pnl, action)

            self.observation_history.append(self._get_single_step_observation(self.current_step))
            terminated = next_portfolio_value <= self.initial_balance * (1.0 - self.strat_cfg.max_drawdown_threshold)
            truncated = self.current_step >= self.max_step
            
            info = {
                'portfolio_value': next_portfolio_value,
                **reward_info,
                'action_magnitude': action_magnitude,
                'consecutive_inactive_steps': self.consecutive_inactive_steps,
                'thrashing_ratio': thrashing_ratio,
                'total_attempted_trades': self.total_attempted_trades,
                'total_executed_trades': self.total_executed_trades,
                'total_insignificant_trades': self.total_insignificant_trades,
                'drawdown': current_drawdown,
                'peak_value': self.episode_peak_value,
                'raw_reward': reward,
                'intrinsic_reward': reward_info.get('transformed_ratios', {}).get('realized_pnl', 0.0)
            }
            return self._get_observation_sequence(), reward, terminated, truncated, info
        except Exception as e:
            logger.exception(f"FATAL ERROR in Worker {self.worker_id} on step {self.current_step}. This worker will crash.")
            raise e

    def reset(self, seed=None, options=None):
        try:
            super().reset(seed=seed)
            self.initial_balance = 1000000.0
            self.balance = self.initial_balance
            self.asset_held, self.entry_price = 0.0, 0.0
            self.episode_peak_value = self.balance
            
            self.consecutive_inactive_steps = 0
            self.total_attempted_trades = 0
            self.total_executed_trades = 0
            self.total_insignificant_trades = 0
            self.total_flips = 0
            self.position_history_buffer.clear()

            self.volatility_estimate = 0.01
            self.market_regime = "UNCERTAIN"
            
            self.portfolio_history = [self.initial_balance]
            
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
            return self._get_observation_sequence(), {
                'portfolio_value': self.balance,
                'balance': self.balance,
                'asset_held': self.asset_held
            }
        except Exception as e:
            logger.exception(f"FATAL ERROR in Worker {getattr(self, 'worker_id', 'N/A')} during environment reset. This worker will crash.")
            raise e

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
                    annualization_factor = np.sqrt(bars_per_day)
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
                
                freq = key.split('_')[-1]
                
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
            logger.exception(f"FATAL ERROR in Worker {self.worker_id} getting observation at step {step_index}. This worker will crash.")
            raise e

    def _get_observation_sequence(self):
        try:
            return {key: np.stack([obs[key] for obs in self.observation_history]) for key in self.observation_space.spaces.keys()}
        except Exception as e:
            logger.exception(f"FATAL ERROR in Worker {self.worker_id} stacking observation sequence. This is a primary suspect for pipe errors if observation is too large.")
            raise e

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
