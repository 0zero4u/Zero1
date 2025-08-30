# Zero1-main/generator.py

import pandas as pd
import numpy as np
from stable_baselines3 import PPO

from ..processor import create_bars_from_trades
from .config import SETTINGS
from .engine import HierarchicalTradingEnvironment

def run_backtest():
    cfg = SETTINGS
    print(f"--- Starting Out-of-Sample Backtest for PPO Hybrid TIN ---")
    
    # 1. Prepare Out-of-Sample Data and Environment
    # Note: Using create_bars_from_trades, which is more robust than create_feature_dataframe
    bars_df = create_bars_from_trades("out_of_sample")
    env = HierarchicalTradingEnvironment(bars_df)
    
    # 2. Load Model
    model_path = cfg.get_model_path()
    model = PPO.load(model_path, env=env)
    print(f"Loaded trained SB3 PPO model from: {model_path}")

    obs, info = env.reset()
    done = False
    portfolio_values = [info['balance']] # Start with initial balance
    initial_value = portfolio_values[0]

    print("Running simulation...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        portfolio_values.append(info['portfolio_value'])

    # 3. Calculate and Print Performance Metrics
    if not portfolio_values or len(portfolio_values) < 2:
        print("\n--- Backtest Warning: No simulation steps were taken. ---")
        return

    final_value = portfolio_values[-1]
    total_return_pct = (final_value / initial_value - 1) * 100
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    sharpe_ratio = 0.0
    if not returns.empty and returns.std() != 0:
        # Base timeframe is 15T, so 96 bars per day
        bars_per_day = 24 * (60 // int(cfg.BASE_BAR_TIMEFRAME[:-1]))
        annualization_factor = np.sqrt(252 * bars_per_day)
        sharpe_ratio = (returns.mean() / returns.std()) * annualization_factor

    print("\n--- Backtest Results ---")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print("✅ Backtest complete.")
