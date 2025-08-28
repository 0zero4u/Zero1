# rl-main/engine.py

import numpy as np
import pandas as pd
import torch
from .config import SETTINGS

class HierarchicalTradingEnvironment:
    """
    An RL environment that provides a multi-timeframe state for the HierarchicalTIN.
    It manages multiple resampled dataframes and pre-computes market context
    features to create a rich state representation at each step.
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
            # Convert our custom timeframe string to pandas frequency format
            resample_freq = tf_str.replace('S', 's').replace('M', 'T')
            df_resampled = base_df['close'].resample(resample_freq).last().to_frame()
            df_resampled['close'].fillna(method='ffill', inplace=True)
            self.timeframes[tf_str] = df_resampled.dropna()
        
        self.base_timestamps = self.timeframes[self.cfg.BASE_BAR_TIMEFRAME].index
        self.max_step = len(self.base_timestamps) - 2 # -2 for safety
        
        # Pre-compute and merge context features
        self._precompute_market_features()
        print("Environment initialized.")

        # --- RL State Variables ---
        self.balance = 10000.0
        self.asset_held = 0.0
        self.done = False
        self.current_step = 0
        self.reset() # Set initial step to a safe point with enough history

    def _precompute_market_features(self):
        """Calculates and stores high-level context features like volatility and trend."""
        print("Pre-computing market context features...")
        df_1h = self.timeframes['1H'].copy()
        
        # Volatility Feature: 1-hour Bollinger Band Width Percentile
        bb_period = 20
        rolling_mean = df_1h['close'].rolling(window=bb_period).mean()
        rolling_std = df_1h['close'].rolling(window=bb_period).std()
        bbw = ((rolling_mean + 2 * rolling_std) - (rolling_mean - 2 * rolling_std)) / rolling_mean
        df_1h['bbw_1h_pct'] = bbw.rolling(250).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
        
        # Trend Feature: 4-hour Price distance from a slow moving average
        df_4h = self.timeframes['4H'].copy()
        df_4h['price_dist_ma_4h'] = (df_4h['close'] / df_4h['close'].rolling(50).mean()) - 1.0

        # Merge these low-frequency features into our high-frequency base dataframe
        base_df_with_features = self.timeframes[self.cfg.BASE_BAR_TIMEFRAME].copy()
        base_df_with_features = pd.merge_asof(base_df_with_features, df_1h[['bbw_1h_pct']], left_index=True, right_index=True)
        base_df_with_features = pd.merge_asof(base_df_with_features, df_4h[['price_dist_ma_4h']], left_index=True, right_index=True)
        base_df_with_features.fillna(method='ffill', inplace=True)
        self.features_df = base_df_with_features[['bbw_1h_pct', 'price_dist_ma_4h']].dropna()
        print("Market context features ready.")

    def _get_specialist_states(self, current_timestamp) -> dict[str, torch.Tensor]:
        """Constructs the dictionary of price windows for each specialist TIN."""
        specialist_states = {}
        for tf, lookback in self.strat_cfg.LOOKBACK_PERIODS.items():
            df_tf = self.timeframes[tf]
            end_idx = df_tf.index.get_loc(current_timestamp, method='ffill')
            start_idx = max(0, end_idx - lookback + 1)
            window_prices = df_tf.iloc[start_idx : end_idx + 1]['close'].values.astype(np.float32)

            if len(window_prices) < lookback:
                padding = np.full(lookback - len(window_prices), window_prices[0])
                window_prices = np.concatenate([padding, window_prices])
            
            last_price = window_prices[-1]
            normalized_window = (window_prices / last_price) - 1.0 if last_price > 1e-6 else np.zeros_like(window_prices)
            specialist_states[tf] = torch.from_numpy(normalized_window)
        return specialist_states

    def _get_market_context_features(self, current_timestamp) -> torch.Tensor:
        """Gets the pre-computed context features for the current step."""
        try:
            features = self.features_df.loc[current_timestamp].values.astype(np.float32)
            return torch.from_numpy(features)
        except KeyError:
            return torch.zeros(2) # Default if features not ready

    def _get_state(self) -> dict[str, any]:
        """Returns the full state dictionary for the HierarchicalTIN."""
        current_timestamp = self.base_timestamps[self.current_step]
        return {
            'specialists': self._get_specialist_states(current_timestamp),
            'context': self._get_market_context_features(current_timestamp)
        }

    def reset(self) -> dict[str, any]:
        """Resets the environment to a starting point with sufficient history."""
        self.balance = 10000.0
        self.asset_held = 0.0
        self.done = False
        # A simple way to ensure enough history is to start N steps in, where N is the largest lookback in seconds.
        self.current_step = int(4 * 3600 * 1.2) # Start ~5 hours into the data to be safe
        return self._get_state()

    def step(self, action: int) -> tuple[dict, float, bool]:
        """Executes one time step within the environment."""
        current_price = self.timeframes[self.cfg.BASE_BAR_TIMEFRAME]['close'].iloc[self.current_step]
        initial_portfolio_value = self.balance + self.asset_held * current_price

        if action == 1 and self.balance > 10: # Buy
            self.asset_held += (self.balance * 0.999) / current_price # Fee
            self.balance = 0.0
        elif action == 2 and self.asset_held > 0: # Sell
            self.balance += (self.asset_held * current_price) * 0.999 # Fee
            self.asset_held = 0.0
        
        self.current_step += 1
        if self.current_step >= self.max_step:
            self.done = True

        next_price = self.timeframes[self.cfg.BASE_BAR_TIMEFRAME]['close'].iloc[self.current_step]
        next_portfolio_value = self.balance + self.asset_held * next_price
        reward = next_portfolio_value - initial_portfolio_value
        
        next_state = None if self.done else self._get_state()
        return next_state, reward, self.done
