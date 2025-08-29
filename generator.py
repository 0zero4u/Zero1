# Zero1-main/generator.py

import pandas as pd
import torch
import numpy as np

from ..processor import create_feature_dataframe
from .config import SETTINGS
from .tins import HybridMonolithicTIN
from .engine import TradingEnvironment

def run_backtest():
    cfg = SETTINGS
    print(f"--- Starting Out-of-Sample Backtest for Hybrid Monolithic TIN ---")
    
    # 1. Load Model
    model_path = cfg.get_model_path()
    model = HybridMonolithicTIN().to(cfg.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    model.eval()
    print(f"Loaded trained model from: {model_path}")

    # 2. Prepare Out-of-Sample Data and Environment
    feature_df = create_feature_dataframe("out_of_sample")
    env = TradingEnvironment(feature_df)
    
    state = env.reset(); done = False
    portfolio_values = []; initial_value = env.balance

    print("Running simulation...")
    while not done:
        with torch.no_grad():
            batched_state = {key: val.unsqueeze(0) for key, val in state.items()}
            action = model(batched_state).max(1)[1].view(1, 1)
        
        state, _, done = env.step(action.item())
        if done: break

        current_price = env.features_df['price_1m'].iloc[env.current_step]
        current_value = env.balance + env.asset_held * current_price
        portfolio_values.append(current_value)

    # 3. Calculate and Print Performance Metrics
    if not portfolio_values:
        print("\n--- Backtest Warning: No simulation steps were taken. ---")
        return

    final_value = portfolio_values[-1]; total_return_pct = (final_value / initial_value - 1) * 100
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    sharpe_ratio = 0.0
    if not returns.empty and returns.std() != 0:
        bars_per_day = 24 * 60
        annualization_factor = np.sqrt(252 * bars_per_day)
        sharpe_ratio = (returns.mean() / returns.std()) * annualization_factor

    print("\n--- Backtest Results ---")
    print(f"Final Portfolio Value: ${final_value:,.2f}"); print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Total Return: {total_return_pct:.2f}%"); print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print("✅ Backtest complete.")
