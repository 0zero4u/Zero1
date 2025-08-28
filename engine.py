# rl-main/engine.py

import numpy as np
import pandas as pd
import torch
from .config import SETTINGS

class TradingEnvironment:
    """
    A Reinforcement Learning environment for financial trading.
    It simulates market interactions for the TIN agent.
    """
    def __init__(self, df_bars: pd.DataFrame):
        self.df = df_bars.dropna().reset_index(drop=True)
        self.cfg = SETTINGS
        self.strat_cfg = self.cfg.strategy

        # --- State Variables ---
        self.current_step = 0
        self.balance = 10000.0  # Initial portfolio balance in USD
        self.asset_held = 0.0   # Initial asset quantity
        self.done = False

    def _get_state(self) -> torch.Tensor:
        """
        Returns the current state for the agent.
        The state is a normalized window of past closing prices.
        """
        if self.current_step < self.strat_cfg.LOOKBACK_WINDOW - 1:
            # Not enough data for a full window, return zeros
            return torch.zeros(self.strat_cfg.LOOKBACK_WINDOW, device=self.cfg.DEVICE)
            
        start = self.current_step - self.strat_cfg.LOOKBACK_WINDOW + 1
        end = self.current_step + 1
        
        window_prices = self.df['close'].iloc[start:end].values.astype(np.float32)
        
        # Normalize the window to be % change from the start of the window
        normalized_window = (window_prices / window_prices[0]) - 1.0
        
        return torch.from_numpy(normalized_window).to(self.cfg.DEVICE)

    def reset(self) -> torch.Tensor:
        """Resets the environment to its initial state."""
        self.current_step = 0
        self.balance = 10000.0
        self.asset_held = 0.0
        self.done = False
        return self._get_state()

    def step(self, action: int) -> tuple[torch.Tensor, float, bool]:
        """
        Executes one time step within the environment.
        action: 0=Hold, 1=Buy, 2=Sell
        """
        initial_portfolio_value = self.balance + self.asset_held * self.df['close'].iloc[self.current_step]

        # --- Execute Action ---
        current_price = self.df['close'].iloc[self.current_step]
        if action == 1: # Buy
            if self.balance > 10: # Minimum trade
                self.asset_held += self.balance / current_price
                self.balance = 0.0
        elif action == 2: # Sell
            if self.asset_held > 0:
                self.balance += self.asset_held * current_price
                self.asset_held = 0.0
        # action 0 is Hold, so do nothing

        # --- Move to next step ---
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        # --- Calculate Reward ---
        next_portfolio_value = self.balance + self.asset_held * self.df['close'].iloc[self.current_step]
        reward = next_portfolio_value - initial_portfolio_value

        next_state = self._get_state()

        return next_state, reward, self.done
