

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import json # Import json for pretty printing

from ..processor import create_bars_from_trades
from .config import SETTINGS
from .engine import HierarchicalTradingEnvironment

def run_backtest():
    cfg = SETTINGS
    print(f"--- Starting Out-of-Sample Backtest for Hierarchical Attention TIN ---")
    
    # 1. Prepare Out-of-Sample Data and Environment
    bars_df = create_bars_from_trades("out_of_sample")
    env = HierarchicalTradingEnvironment(bars_df)
    
    # 2. Load Model
    model_path = cfg.get_model_path()
    model = PPO.load(model_path, env=env)
    print(f"Loaded trained SB3 PPO model from: {model_path}")

    obs, info = env.reset()
    done = False
    portfolio_values = [info['balance']]
    initial_value = portfolio_values[0]
    step_count = 0

    print("Running simulation...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        portfolio_values.append(info['portfolio_value'])

        ### --- GREY BOX CHANGE 3: EXTRACT AND DISPLAY REASONING --- ###
        # Display the "Chief Strategist" reasoning every 100 steps
        if step_count % 100 == 0:
            # Access the stored weights from the feature extractor
            # It's a numpy array of shape (1, 4) because batch size is 1 during prediction
            weights = model.policy.features_extractor.last_attention_weights[0]
            
            # Format the weights into a readable JSON-like dictionary
            reasoning_payload = {
                "Timestep": env.current_step,
                "Chief_Strategist_Decision": {
                    "Action_Signal": float(action[0]), # Position signal from -1 to 1
                    "Action_Aggression": float(action[1]), # Sizing from 0 to 1
                    "Composition_Score_Route": {
                        "Weight_on_Flow_Factors": f"{weights[0]:.2%}",
                        "Weight_on_Volatility_Factors": f"{weights[1]:.2%}",
                        "Weight_on_Value_Trend_Factors": f"{weights[2]:.2%}",
                        "Weight_on_Context_Factors": f"{weights[3]:.2%}"
                    }
                }
            }
            print("\n" + "="*20 + " Model Reasoning " + "="*20)
            print(json.dumps(reasoning_payload, indent=2))
            print("="*57)


    # 3. Calculate and Print Performance Metrics
    if not portfolio_values or len(portfolio_values) < 2:
        print("\n--- Backtest Warning: No simulation steps were taken. ---")
        return

    final_value = portfolio_values[-1]
    total_return_pct = (final_value / initial_value - 1) * 100
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    sharpe_ratio = 0.0
    if not returns.empty and returns.std() != 0:
        bars_per_day = 24 * (60 // int(cfg.BASE_BAR_TIMEFRAME[:-1]))
        annualization_factor = np.sqrt(252 * bars_per_day)
        sharpe_ratio = (returns.mean() / returns.std()) * annualization_factor

    print("\n--- Backtest Results ---")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print("✅ Backtest complete.")
