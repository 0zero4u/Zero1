"""
Centralized Data Normalization System for Crypto Trading RL
Provides a robust, serializable normalizer that fits on training data
to prevent lookahead bias during evaluation and live trading.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, Any, Literal, List
from config import SETTINGS, StrategyConfig, FeatureKeys

logger = logging.getLogger(__name__)

NormalizationStrategy = Literal["standardize", "relative", "log", "pass"]

class Normalizer:
    """
    A class to handle fitting, transforming, saving, and loading normalization parameters.
    This ensures that scaling parameters are learned *only* from the in-sample (training)
    data and consistently applied to out-of-sample (validation/testing) data.
    """

    def __init__(self, strategy_cfg: StrategyConfig):
        self.cfg = strategy_cfg
        self.params: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False
        self.epsilon = 1e-8
        # UPDATED: The context feature keys are now sourced directly from the config.
        self.context_feature_keys: List[str] = strategy_cfg.context_feature_keys
        # NEW: Source precomputed feature keys from the config
        self.precomputed_feature_keys: List[str] = strategy_cfg.precomputed_feature_keys

    def fit(self, base_bars_df: pd.DataFrame, features_df: pd.DataFrame):
        """
        Learn normalization parameters (mean, std) from the in-sample data.
        """
        logger.info("Fitting normalizer on in-sample data...")
        
        # --- Fit Context Features ---
        for key in self.context_feature_keys:
            if key in features_df.columns:
                series = features_df[key].astype(np.float64).replace([np.inf, -np.inf], np.nan).dropna()
                if not series.empty:
                    self.params[key] = {
                        'mean': series.mean(),
                        'std': series.std() + self.epsilon
                    }
        logger.info(f" -> Fitted {len(self.params)} context features.")

        # --- Fit Volume Delta Features ---
        vol_delta_keys = [k for k in self.cfg.lookback_periods.keys() if k.value.startswith('volume_delta_')]
        
        # Ensure timestamp is the index for resampling
        if not isinstance(base_bars_df.index, pd.DatetimeIndex):
            base_bars_df = base_bars_df.set_index('timestamp')

        for key_enum in vol_delta_keys:
            key = key_enum.value
            freq = key.split('_')[-1].replace('m', 'T').replace('s', 'S').upper()
            
            if 'volume_delta' in base_bars_df.columns:
                # Resample the volume delta to the correct timeframe
                resampled_delta = base_bars_df['volume_delta'].resample(freq).sum()
                
                series = resampled_delta.astype(np.float64).replace([np.inf, -np.inf], np.nan).dropna()
                if not series.empty:
                    self.params[key] = {
                        'mean': series.mean(),
                        'std': series.std() + self.epsilon
                    }
        logger.info(f" -> Fitted {len(vol_delta_keys)} volume delta features.")
        
        # NEW: Fit Precomputed Features
        for key in self.precomputed_feature_keys:
            if key in base_bars_df.columns:
                series = base_bars_df[key].astype(np.float64).replace([np.inf, -np.inf], np.nan).dropna()
                if not series.empty:
                    self.params[key] = {
                        'mean': series.mean(),
                        'std': series.std() + self.epsilon
                    }
        logger.info(f" -> Fitted {len(self.precomputed_feature_keys)} precomputed features.")
        
        self.is_fitted = True
        logger.info("✅ Normalizer fitting complete.")

    def transform(self, raw_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply pre-fitted normalization to a raw observation dictionary.
        This method is designed to be fast and is called on every step of the environment.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transforming data.")
            
        normalized_obs = {}
        for key, data in raw_obs.items():
            # Price and OHLC data use relative normalization (local, no pre-fitting needed)
            if key.startswith('price_'):
                last_price = data[-1]
                normalized_obs[key] = (data / (last_price + self.epsilon)) - 1.0
            
            elif key.startswith('ohlc'):
                last_price = data[-1, 3]  # Last close price
                if data.shape[1] == 5:  # OHLCV
                    normalized_data = np.zeros_like(data, dtype=np.float32)
                    # Normalize OHLC by price
                    normalized_data[:, :4] = (data[:, :4] / (last_price + self.epsilon)) - 1.0
                    # Log-transform and standardize volume by its recent mean
                    recent_volume_avg = np.mean(data[-20:, 4]) + self.epsilon
                    normalized_data[:, 4] = np.log1p(data[:, 4]) / np.log1p(recent_volume_avg)
                    normalized_obs[key] = normalized_data
                else: # OHLC
                    normalized_obs[key] = (data / (last_price + self.epsilon)) - 1.0
            
            # Volume Delta and Context features use pre-fitted standardization
            elif key.startswith('volume_delta_'):
                if key in self.params:
                    mean = self.params[key]['mean']
                    std = self.params[key]['std']
                    normalized_obs[key] = (data - mean) / std
                else:
                    logger.warning(f"No normalization params for {key}, passing through.")
                    normalized_obs[key] = data

            elif key == 'context':
                # Context features are a vector, we need to normalize each element
                normalized_context = np.zeros_like(data, dtype=np.float32)
                for i, feature_name in enumerate(self.context_feature_keys):
                    if feature_name in self.params:
                        mean = self.params[feature_name]['mean']
                        std = self.params[feature_name]['std']
                        normalized_context[i] = (data[i] - mean) / std
                    else:
                        normalized_context[i] = data[i] # Pass through if not found
                normalized_obs[key] = normalized_context

            # NEW: Standardize precomputed features using pre-fitted parameters
            elif key == FeatureKeys.PRECOMPUTED_FEATURES.value:
                normalized_precomp = np.zeros_like(data, dtype=np.float32)
                for i, feature_name in enumerate(self.precomputed_feature_keys):
                    if feature_name in self.params:
                        mean = self.params[feature_name]['mean']
                        std = self.params[feature_name]['std']
                        normalized_precomp[i] = (data[i] - mean) / std
                    else:
                        normalized_precomp[i] = data[i] # Pass through if not found
                normalized_obs[key] = normalized_precomp
            
            else:
                # Default case: pass through
                normalized_obs[key] = data
                
        return normalized_obs

    def save(self, file_path: Path):
        """Save the fitted parameters to a JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.params, f, indent=4)
        logger.info(f"Normalizer parameters saved to {file_path}")

    def load(self, file_path: Path):
        """Load fitted parameters from a JSON file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Normalizer file not found at {file_path}")
        with open(file_path, 'r') as f:
            self.params = json.load(f)
        self.is_fitted = True
        logger.info(f"Normalizer parameters loaded from {file_path}")
