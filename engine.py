
import numpy as np
import pandas as pd
import torch
from zigzag import peak_valley_pivots
from .config import SETTINGS

class HierarchicalTradingEnvironment:
    """
    An RL environment that provides a multi-timeframe state for the HierarchicalTIN.
    It manages multiple resampled dataframes and pre-computes market context
    features to create a rich state representation at each step.
    """
    def __init__(self, df_base_ohlc: pd.DataFrame):
        self.cfg = SETTINGS
        self.strat_cfg = self.cfg.strategy

        print("--- Initializing Hierarchical Environment ---")
        base_df = df_base_ohlc.set_index('timestamp')
        
        # Ensure all required timeframes (for model AND features) are created
        model_timeframes = set(self.strat_cfg.LOOKBACK_PERIODS.keys())
        feature_timeframes = {'1H', '4H'} # Internally required for feature engineering
        all_required_keys = model_timeframes.union(feature_timeframes)
        
        # Create and store a dataframe for each required timeframe
        self.timeframes = {}
        print("Resampling data for all required timeframes...")
        for key in all_required_keys:
            if key == 'context': continue
            
            # Convert config key like 'price_15m' to pandas freq '15T'
            freq = key.split('_')[-1].replace('m','T').replace('h','H')

            if freq not in self.timeframes:
                # Keep full OHLC for the base timeframe, as it's needed for multiple features
                if freq == self.cfg.BASE_BAR_TIMEFRAME:
                    df_resampled = base_df.resample(freq).agg({'open':'first','high':'max','low':'min','close':'last'})
                else: # For other timeframes, just keep the close price for now
                    df_resampled = base_df['close'].resample(freq).last().to_frame()

                df_resampled.fillna(method='ffill', inplace=True)
                self.timeframes[freq] = df_resampled.dropna()
        
        self.base_timestamps = self.timeframes[self.cfg.BASE_BAR_TIMEFRAME].index
        self.max_step = len(self.base_timestamps) - 2 # -2 for safety
        
        self._precompute_market_features()
        print("Environment initialized.")

        # --- RL State Variables ---
        self.balance = 10000.0
        self.asset_held = 0.0
        self.done = False
        self.current_step = 0
        self.reset()

    def _precompute_market_features(self):
        """Calculates and stores high-level context features like volatility, trend, and S/R."""
        print("Pre-computing market context features...")
        df_1h = self.timeframes['1H'].copy()
        
        # 1. Volatility Feature: 1-hour Bollinger Band Width Percentile
        bb_period = 20
        rolling_mean = df_1h['close'].rolling(window=bb_period).mean()
        rolling_std = df_1h['close'].rolling(window=bb_period).std()
        bbw = ((rolling_mean + 2 * rolling_std) - (rolling_mean - 2 * rolling_std)) / rolling_mean
        df_1h['bbw_1h_pct'] = bbw.rolling(250).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
        
        # 2. Trend Feature: 4-hour Price distance from a slow moving average
        df_4h = self.timeframes['4H'].copy()
        df_4h['price_dist_ma_4h'] = (df_4h['close'] / df_4h['close'].rolling(50).mean()) - 1.0

        # 3. Support & Resistance Features using ZigZag
        print("Calculating Support/Resistance levels...")
        df_sr = self.timeframes[self.cfg.BASE_BAR_TIMEFRAME].copy()
        
        high_low = df_sr['high'] - df_sr['low']
        high_prev_close = np.abs(df_sr['high'] - df_sr['close'].shift())
        low_prev_close = np.abs(df_sr['low'] - df_sr['close'].shift())
        tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        df_sr['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()

        pivots = peak_valley_pivots(df_sr['close'], 0.01, -0.01) # Adjust thresholds as needed
        pivots = pivots[pivots != 0]
        
        def find_last_sr(index, pivots_series, full_series):
            past_pivots = pivots_series[pivots_series.index < index]
            if past_pivots.empty: return np.nan, np.nan
            supports = full_series[past_pivots[past_pivots == -1].index]
            resistances = full_series[past_pivots[past_pivots == 1].index]
            last_support = supports.iloc[-1] if not supports.empty else np.nan
            last_resistance = resistances.iloc[-1] if not resistances.empty else np.nan
            return last_support, last_resistance

        sr_levels = [find_last_sr(idx, pivots, df_sr['close']) for idx in df_sr.index]
        df_sr[['last_support', 'last_resistance']] = pd.DataFrame(sr_levels, index=df_sr.index)
        
        df_sr['dist_to_support'] = (df_sr['close'] - df_sr['last_support']) / df_sr['atr']
        df_sr['dist_to_resistance'] = (df_sr['last_resistance'] - df_sr['close']) / df_sr['atr']
        
        # Merge ALL features into the base dataframe
        base_df_with_features = self.timeframes[self.cfg.BASE_BAR_TIMEFRAME].copy()
        base_df_with_features = pd.merge_asof(base_df_with_features, df_1h[['bbw_1h_pct']], left_index=True, right_index=True)
        base_df_with_features = pd.merge_asof(base_df_with_features, df_4h[['price_dist_ma_4h']], left_index=True, right_index=True)
        base_df_with_features = pd.merge_asof(base_df_with_features, df_sr[['dist_to_support', 'dist_to_resistance']], left_index=True, right_index=True)

        base_df_with_features.fillna(method='ffill', inplace=True)
        feature_cols = ['bbw_1h_pct', 'price_dist_ma_4h', 'dist_to_support', 'dist_to_resistance']
        self.features_df = base_df_with_features[feature_cols].dropna()
        print("Market context features ready.")

    def _get_specialist_states(self, current_timestamp) -> dict[str, torch.Tensor]:
        """Constructs the dictionary of price windows for each specialist TIN."""
        specialist_states = {}
        for key, lookback in self.strat_cfg.LOOKBACK_PERIODS.items():
            if not (key.startswith('price_') or key.startswith('ohlc_')):
                continue

            freq = key.split('_')[-1].replace('m','T').replace('h','H')
            df_tf = self.timeframes[freq]
            end_idx = df_tf.index.get_loc(current_timestamp, method='ffill')
            start_idx = max(0, end_idx - lookback + 1)

            if key.startswith('price_'):
                window = df_tf.iloc[start_idx : end_idx + 1]['close'].values.astype(np.float32)
                if len(window) < lookback:
                    padding = np.full(lookback - len(window), window[0])
                    window = np.concatenate([padding, window])
                last_price = window[-1]
                normalized_window = (window / last_price) - 1.0 if last_price > 1e-6 else np.zeros_like(window)
                specialist_states[key] = torch.from_numpy(normalized_window)

            elif key.startswith('ohlc_'):
                window = df_tf.iloc[start_idx : end_idx + 1][['open','high','low','close']].values.astype(np.float32)
                if len(window) < lookback:
                    padding = np.repeat(window[0:1], lookback - len(window), axis=0)
                    window = np.concatenate([padding, window], axis=0)
                last_price = window[-1, 3] # Get last close price
                normalized_window = (window / last_price) - 1.0 if last_price > 1e-6 else np.zeros_like(window)
                specialist_states[key] = torch.from_numpy(normalized_window)

        return specialist_states

    def _get_market_context_features(self, current_timestamp) -> torch.Tensor:
        """Gets the pre-computed context features for the current step."""
        try:
            features = self.features_df.loc[current_timestamp].values.astype(np.float32)
            return torch.from_numpy(features)
        except KeyError:
            return torch.zeros(self.strat_cfg.LOOKBACK_PERIODS['context'])

    def _get_state(self) -> dict[str, any]:
        """Returns the full state dictionary for the HierarchicalTIN."""
        current_timestamp = self.base_timestamps[self.current_step]
        state_data = self._get_specialist_states(current_timestamp)
        state_data['context'] = self._get_market_context_features(current_timestamp)
        return state_data

    def reset(self) -> dict[str, any]:
        """Resets the environment to a starting point with sufficient history."""
        self.balance = 10000.0
        self.asset_held = 0.0
        self.done = False
        # Longest lookback is BBW percentile (250 bars) on 1H data.
        # 250 (1-hour bars) * 4 (15-min bars per hour) = 1000 bars. Add safety margin.
        self.current_step = 1050 
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
--- END OF MODIFIED FILE Zero1-main/engine.py ---
