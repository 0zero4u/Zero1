import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from .config import SETTINGS

class HierarchicalTradingEnvironment(gym.Env):
    """
    A Gymnasium-compliant RL environment that provides a multi-timeframe state
    for the trading agent. It manages multiple resampled dataframes and pre-computes
    market context features to create a rich state representation at each step.
    The observation is a sequence of past states, designed for an LSTM-based policy.
    """
    def __init__(self, df_base_ohlc: pd.DataFrame):
        super().__init__()
        self.cfg = SETTINGS
        self.strat_cfg = self.cfg.strategy

        print("--- Initializing Gymnasium-Compliant Hierarchical Environment ---")
        base_df = df_base_ohlc.set_index('timestamp')
        
        # Create and store a dataframe for each required timeframe
        model_timeframes = set(self.strat_cfg.LOOKBACK_PERIODS.keys())
        feature_timeframes = {'1H', '4H'}
        all_required_keys = model_timeframes.union(feature_timeframes)
        self.timeframes = {}
        print("Resampling data for all required timeframes...")
        for key in all_required_keys:
            if key == 'context': continue
            freq = key.split('_')[-1].replace('m','T').replace('h','H')
            if freq not in self.timeframes:
                agg_rules = {'open':'first','high':'max','low':'min','close':'last'} if freq == self.cfg.BASE_BAR_TIMEFRAME else {'close':'last'}
                df_resampled = base_df.resample(freq).agg(agg_rules)
                df_resampled.fillna(method='ffill', inplace=True)
                self.timeframes[freq] = df_resampled.dropna()
        
        self.base_timestamps = self.timeframes[self.cfg.BASE_BAR_TIMEFRAME].index
        self._precompute_market_features()
        self.max_step = len(self.base_timestamps) - 2 # Safety margin
        
        # --- Define Gym Spaces ---
        self.action_space = spaces.Discrete(self.strat_cfg.ACTION_SPACE_SIZE)
        
        obs_spaces = {}
        seq_len = self.strat_cfg.SEQUENCE_LENGTH
        for key, lookback in self.strat_cfg.LOOKBACK_PERIODS.items():
            if key.startswith('ohlc_'):
                shape = (seq_len, lookback, 4)
            elif key.startswith('price_'):
                shape = (seq_len, lookback)
            else: # context
                shape = (seq_len, lookback)
            obs_spaces[key] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)

        # History buffer for sequential observations
        self.observation_history = deque(maxlen=self.strat_cfg.SEQUENCE_LENGTH)
        print("Environment initialized.")

    def _precompute_market_features(self):
        """Calculates and stores high-level context features like volatility, trend, and S/R."""
        print("Pre-computing market context features...")
        df_1h = self.timeframes['1H'].copy()
        bb_period = 20
        rolling_mean = df_1h['close'].rolling(window=bb_period).mean()
        rolling_std = df_1h['close'].rolling(window=bb_period).std()
        bbw = (4 * rolling_std) / rolling_mean
        df_1h['bbw_1h_pct'] = bbw.rolling(250).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
        
        df_4h = self.timeframes['4H'].copy()
        df_4h['price_dist_ma_4h'] = (df_4h['close'] / df_4h['close'].rolling(50).mean()) - 1.0
        
        base_df_with_features = self.timeframes[self.cfg.BASE_BAR_TIMEFRAME].copy()
        base_df_with_features = pd.merge_asof(base_df_with_features, df_1h[['bbw_1h_pct']], left_index=True, right_index=True)
        base_df_with_features = pd.merge_asof(base_df_with_features, df_4h[['price_dist_ma_4h']], left_index=True, right_index=True)
        # S/R feature calculation is computationally expensive, simplified here for clarity and speed.
        # In a real scenario, the ZigZag implementation would be re-integrated carefully.
        base_df_with_features['dist_to_support'] = 0.0
        base_df_with_features['dist_to_resistance'] = 0.0

        base_df_with_features.fillna(method='ffill', inplace=True)
        feature_cols = ['bbw_1h_pct', 'price_dist_ma_4h', 'dist_to_support', 'dist_to_resistance']
        self.features_df = base_df_with_features[feature_cols].dropna()
        print("Market context features ready.")

    def _get_single_step_observation(self, step_index) -> dict:
        """Constructs the observation dictionary for a single point in time."""
        current_timestamp = self.base_timestamps[step_index]
        obs = {}
        for key, lookback in self.strat_cfg.LOOKBACK_PERIODS.items():
            if key == 'context':
                try:
                    features = self.features_df.loc[current_timestamp].values.astype(np.float32)
                    obs[key] = features
                except KeyError:
                    obs[key] = np.zeros(lookback, dtype=np.float32)
                continue

            freq = key.split('_')[-1].replace('m','T').replace('h','H')
            df_tf = self.timeframes[freq]
            end_idx = df_tf.index.get_loc(current_timestamp, method='ffill')
            start_idx = max(0, end_idx - lookback + 1)
            
            if key.startswith('price_'):
                window = df_tf.iloc[start_idx : end_idx + 1]['close'].values.astype(np.float32)
                if len(window) < lookback:
                    window = np.pad(window, (lookback - len(window), 0), 'edge')
                last_price = window[-1]
                obs[key] = (window / last_price) - 1.0 if last_price > 1e-6 else np.zeros_like(window)

            elif key.startswith('ohlc_'):
                window = df_tf.iloc[start_idx : end_idx + 1][['open','high','low','close']].values.astype(np.float32)
                if len(window) < lookback:
                    padding = np.repeat(window[0:1], lookback - len(window), axis=0)
                    window = np.concatenate([padding, window], axis=0)
                last_price = window[-1, 3]
                obs[key] = (window / last_price) - 1.0 if last_price > 1e-6 else np.zeros_like(window)
        return obs

    def _get_observation_sequence(self):
        """Stacks observations from the history deque to form the final sequential observation."""
        obs_sequence = {}
        for key in self.observation_space.spaces.keys():
            obs_sequence[key] = np.stack([obs[key] for obs in self.observation_history])
        return obs_sequence

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = 10000.0
        self.asset_held = 0.0
        # Longest lookback is BBW percentile (250 bars) on 1H data.
        # 250 (1-hour bars) * 4 (15-min bars per hour) = 1000 bars. Add safety margin.
        self.current_step = 1050
        
        # Prime the observation history buffer
        self.observation_history.clear()
        for i in range(self.strat_cfg.SEQUENCE_LENGTH):
            step_idx = self.current_step - self.strat_cfg.SEQUENCE_LENGTH + 1 + i
            self.observation_history.append(self._get_single_step_observation(step_idx))
            
        observation = self._get_observation_sequence()
        info = {'balance': self.balance, 'asset_held': self.asset_held}
        return observation, info

    def step(self, action: int):
        current_price = self.timeframes[self.cfg.BASE_BAR_TIMEFRAME]['close'].iloc[self.current_step]
        initial_portfolio_value = self.balance + self.asset_held * current_price

        # --- Execute Action ---
        if action == 6 and self.balance > 10: self.asset_held += (self.balance * 0.999) / current_price; self.balance = 0
        elif action == 5 and self.balance > 10: self.asset_held += (self.balance * 0.75 * 0.999) / current_price; self.balance *= 0.25
        elif action == 4 and self.balance > 10: self.asset_held += (self.balance * 0.5 * 0.999) / current_price; self.balance *= 0.5
        elif action == 0 and self.asset_held > 0: self.balance += (self.asset_held * current_price) * 0.999; self.asset_held = 0
        elif action == 1 and self.asset_held > 0: self.balance += (self.asset_held * 0.75 * current_price) * 0.999; self.asset_held *= 0.25
        elif action == 2 and self.asset_held > 0: self.balance += (self.asset_held * 0.5 * current_price) * 0.999; self.asset_held *= 0.5

        self.current_step += 1
        truncated = self.current_step >= self.max_step
        terminated = self.balance <= 1000 and self.asset_held * current_price < 10 # Bankrupt condition

        next_price = self.timeframes[self.cfg.BASE_BAR_TIMEFRAME]['close'].iloc[self.current_step]
        next_portfolio_value = self.balance + self.asset_held * next_price
        reward = next_portfolio_value - initial_portfolio_value
        
        # Update observation history and get next state
        self.observation_history.append(self._get_single_step_observation(self.current_step))
        observation = self._get_observation_sequence()
        info = {'balance': self.balance, 'asset_held': self.asset_held, 'portfolio_value': next_portfolio_value}
        
        return observation, reward, terminated, truncated, info
