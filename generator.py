# rl-main/generator.py

import pandas as pd
import torch
import numpy as np

from ..data.processor import create_bars_from_trades # Adapt this as needed
from .config import SETTINGS
from .tins import DQN_TIN
from .engine import TradingEnvironment

def run_backtest():
    """
    Loads a trained TIN model and evaluates its performance on out-of-sample data.
    """
    cfg = SETTINGS
    print(f"--- Starting Out-of-Sample Backtest ---")
    
    # 1. Load Model
    model_path = cfg.get_model_path()
    model = DQN_TIN().to(cfg.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    model.eval()
    print(f"Loaded trained model from: {model_path}")

    # 2. Prepare Out-of-Sample Data and Environment
    # This should get data for the "out_of_sample" period
    bars_df = create_bars_from_trades("out_of_sample")
    env = TradingEnvironment(bars_df)
    
    state = env.reset()
    done = False
    
    portfolio_values = []
    initial_value = env.balance

    print("Running simulation...")
    while not done:
        with torch.no_grad():
            # Choose the best action, no exploration
            action = model(state.unsqueeze(0)).max(1)[1].view(1, 1)
        
        state, _, done = env.step(action.item())
        
        current_price = env.df['close'].iloc[env.current_step]
        current_value = env.balance + env.asset_held * current_price
        portfolio_values.append(current_value)

    # 3. Calculate and Print Performance Metrics
    final_value = portfolio_values[-1]
    total_return_pct = (final_value / initial_value - 1) * 100
    
    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * (24*60/int(cfg.BAR_TIMEFRAME[:-1]))) # Annualized

    print("\n--- Backtest Results ---")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print("✅ Backtest complete.")
