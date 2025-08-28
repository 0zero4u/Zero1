# rl-main/engine.py

import numpy as np
import pandas as pd
import torch
from .config import SETTINGS

class HierarchicalTradingEnvironment:
    """
    An RL environment that provides a multi-timeframe state for the HierarchicalTIN.
    It manages multiple resampled dataframes to construct the state at each step.
    """
    def __init__(self, df_base: pd.DataFrame):
        self.cfg = SETTINGS
        self.strat_cfg = self.cfg.strategy

        print("--- Initializing Hierarchical Environment ---")
        base_df = df_base.set_index('timestamp')
        
        # Create and store a dataframe for each required timeframe
        self.timeframes = {tf: None for tf in self.strat_cfg.LOOKBACK_PERIODS.keys()}
        print("Resampling data for all timeframes...")
        for tf_str in self.timeframes.keys():
            resample_freq = tf_str.replace('S', 'S').replace('M', 'T') # Pandas freq format
            df_resampled = base_df['close'].resample(resample_freq).last().to_frame()
            df_resampled['close'].fillna(method='ffill', inplace=True)
            self.timeframes[tf_str] = df_resampled.dropna()
        
        self.base_timestamps = self.timeframes[self.cfg.BASE_BAR_TIMEFRAME].index
        self.max_step = len(self.base_timestamps) - 1
        print("Environment initialized.")

        # --- State Variables ---
        self.current_step = 0
        self.balance = 10000.0
        self.asset_held = 0.0
        self.done = False

    def _get_state(self) -> dict[str, torch.Tensor]:
        """
        Constructs a dictionary of states, one for each specialist TIN,
        for the current time step.
        """
        state_dict = {}
        current_timestamp = self.base_timestamps[self.current_step]

        for tf, lookback in self.strat_cfg.LOOKBACK_PERIODS.items():
            df_tf = self.timeframes[tf]
            
            # Find the index of the latest bar on this timeframe
            end_idx = df_tf.index.get_loc(current_timestamp, method='ffill')
            start_idx = max(0, end_idx - lookback + 1)
            
            window_prices = df_tf.iloc[start_idx : end_idx + 1]['close'].values.astype(np.float32)

            # Pad with the first price if not enough history
            if len(window_prices) < lookback:
                padding = np.full(lookback - len(window_prices), window_prices[0])
                window_prices = np.concatenate([padding, window_prices])
            
            # Normalize window by its last price to focus on shape
            last_price = window_prices[-1]
            if last_price > 1e-6:
                normalized_window = (window_prices / last_price) - 1.0
            else:
                normalized_window = np.zeros_like(window_prices)

            state_dict[tf] = torch.from_numpy(normalized_window).to(self.cfg.DEVICE)
            
        return state_dict

    def reset(self) -> dict[str, torch.Tensor]:
        """Resets the environment."""
        self.current_step = 0
        self.balance = 10000.0
        self.asset_held = 0.0
        self.done = False
        # Find the first step with enough history for all lookbacks
        min_step = 0
        for tf, lookback in self.strat_cfg.LOOKBACK_PERIODS.items():
            # A rough estimate for safety
            if len(self.timeframes[tf]) > lookback:
                 min_step = max(min_step, lookback)
        self.current_step = min_step
        return self._get_state()

    def step(self, action: int) -> tuple[dict, float, bool]:
        """Executes one time step within the environment."""
        current_price = self.timeframes[self.cfg.BASE_BAR_TIMEFRAME]['close'].iloc[self.current_step]
        initial_portfolio_value = self.balance + self.asset_held * current_price

        # --- Execute Action ---
        if action == 1: # Buy
            if self.balance > 10:
                self.asset_held += (self.balance * 0.999) / current_price # Simple fee
                self.balance = 0.0
        elif action == 2: # Sell
            if self.asset_held > 0:
                self.balance += (self.asset_held * current_price) * 0.999 # Simple fee
                self.asset_held = 0.0

        # --- Move to next step ---
        self.current_step += 1
        if self.current_step >= self.max_step:
            self.done = True

        # --- Calculate Reward ---
        next_price = self.timeframes[self.cfg.BASE_BAR_TIMEFRAME]['close'].iloc[self.current_step]
        next_portfolio_value = self.balance + self.asset_held * next_price
        reward = next_portfolio_value - initial_portfolio_value
        
        next_state = self._get_state() if not self.done else None

        return next_state, reward, self.done
