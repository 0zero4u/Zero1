# rl-main/generator.py

import pandas as pd
import torch
import numpy as np
from typing import Dict

from .processor import create_bars_from_trades
from .config import SETTINGS
from .tins import HierarchicalTIN
from .engine import HierarchicalTradingEnvironment

def run_backtest():
    """
    Loads a trained TIN model and evaluates its performance on out-of-sample data.
    """
    cfg = SETTINGS
    print(f"--- Starting Out-of-Sample Backtest ---")
    
    # 1. Load Model
    model_path = cfg.get_model_path()
    model = HierarchicalTIN().to(cfg.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    model.eval()
    print(f"Loaded trained model from: {model_path}")

    # 2. Prepare Out-of-Sample Data and Environment
    bars_df = create_bars_from_trades("out_of_sample") 
    env = HierarchicalTradingEnvironment(bars_df)
    
    state = env.reset()
    done = False
    
    portfolio_values = []
    initial_value = env.balance

    print("Running simulation...")
    while not done:
        with torch.no_grad():
            batched_state = {
                'specialists': {
                    tf: tensor.unsqueeze(0)
                    for tf, tensor in state['specialists'].items()
                },
                'context': state['context'].unsqueeze(0)
            }
            
            # Choose the best action, no exploration
            action = model(batched_state).max(1)[1].view(1, 1)
        
        state, _, done = env.step(action.item())
        
        if done:
            break

        current_price = env.timeframes[cfg.BASE_BAR_TIMEFRAME]['close'].iloc[env.current_step]
        current_value = env.balance + env.asset_held * current_price
        portfolio_values.append(current_value)

    # 3. Calculate and Print Performance Metrics
    if not portfolio_values:
        print("\n--- Backtest Warning ---")
        print("No trades were made or simulation ended immediately. Cannot calculate performance.")
        return

    final_value = portfolio_values[-1]
    total_return_pct = (final_value / initial_value - 1) * 100
    
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    if returns.empty or returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        base_seconds = int(''.join(filter(str.isdigit, cfg.BASE_BAR_TIMEFRAME)))
        if 'M' in cfg.BASE_BAR_TIMEFRAME.upper():
            base_seconds *= 60
        elif 'H' in cfg.BASE_BAR_TIMEFRAME.upper():
            base_seconds *= 3600
        
        bars_per_day = (24 * 60 * 60) / base_seconds
        annualization_factor = np.sqrt(252 * bars_per_day) # Using 252 trading days
        sharpe_ratio = (returns.mean() / returns.std()) * annualization_factor

    print("\n--- Backtest Results ---")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print("✅ Backtest complete.")
