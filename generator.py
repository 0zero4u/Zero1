# rl-main/generator.py

import pandas as pd
import torch
import numpy as np
from typing import Dict

# FIX: create_bars_from_trades is in processor.py, not data.processor
from .processor import create_bars_from_trades
from .config import SETTINGS
# FIX 1: Corrected model import from DQN_TIN to HierarchicalTIN
from .tins import HierarchicalTIN
# FIX 5: Corrected environment class import
from .engine import HierarchicalTradingEnvironment

def run_backtest():
    """
    Loads a trained TIN model and evaluates its performance on out-of-sample data.
    """
    cfg = SETTINGS
    print(f"--- Starting Out-of-Sample Backtest ---")
    
    # 1. Load Model
    model_path = cfg.get_model_path()
    # FIX 1: Use the correct model class HierarchicalTIN
    model = HierarchicalTIN().to(cfg.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    model.eval()
    print(f"Loaded trained model from: {model_path}")

    # 2. Prepare Out-of-Sample Data and Environment
    # This should get data for the "out_of_sample" period
    bars_df = create_bars_from_trades() # Assuming this function is adapted to select periods
    # FIX 5: Use the correct environment class name
    env = HierarchicalTradingEnvironment(bars_df)
    
    state = env.reset()
    done = False
    
    portfolio_values = []
    initial_value = env.balance

    print("Running simulation...")
    while not done:
        with torch.no_grad():
            # FIX 2: Correctly handle the dictionary state.
            # The model expects tensors with a batch dimension. We must add this
            # dimension to each tensor within the state dictionary.
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
        
        # Stop if the simulation is done and there's no next state
        if done:
            break

        # FIX 3: Access the correct environment attribute for price data
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
        # FIX 4: Use the correct config attribute BASE_BAR_TIMEFRAME
        # Note: Annualization factor depends on the bar frequency.
        # This assumes BASE_BAR_TIMEFRAME is like '1S', '60S', '1T', etc.
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
