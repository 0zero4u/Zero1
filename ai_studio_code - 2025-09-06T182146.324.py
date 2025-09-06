# REFINED: features.py with Centralized Vectorized Logic

"""
Stateful, Incremental Feature Calculators for High-Performance Trading Environments.

REFINEMENT: Centralized feature logic by adding a `calculate_vectorized` class method
to each feature. This ensures a single source of truth for both the incremental (live)
and batch (normalizer fitting) calculations, preventing train/test skew.
"""

from collections import deque
import numpy as np
import scipy.signal
from typing import Dict, List
import pandas as pd # <-- Added pandas import

class StatefulFeature:
    """Base class for a stateful feature calculator."""
    
    def __init__(self, period: int):
        if period <= 0:
            raise ValueError("Period must be a positive integer.")
        self.period = period
        self.q = deque(maxlen=period)
        self.last_value = 0.0

    def update(self, new_data_point):
        """Update the internal state. Must be implemented by subclasses."""
        raise NotImplementedError

    def get(self) -> float:
        """Return the last calculated value."""
        return self.last_value

    def is_ready(self) -> bool:
        """Check if the calculator has enough data to produce a value."""
        return len(self.q) == self.period

    @classmethod
    def calculate_vectorized(cls, **kwargs) -> pd.DataFrame:
        """Vectorized calculation for this feature. To be implemented by subclasses."""
        raise NotImplementedError

class StatefulSMA(StatefulFeature):
    """Stateful Simple Moving Average with O(1) update time."""
    
    def __init__(self, period: int):
        super().__init__(period)
        self._current_sum = 0.0

    def update(self, new_value: float):
        if len(self.q) == self.period:
            self._current_sum -= self.q[0]
        self.q.append(new_value)
        self._current_sum += new_value
        
        if self.q:
            self.last_value = self._current_sum / len(self.q)
        return self.last_value

class StatefulStdDev(StatefulFeature):
    """Stateful Standard Deviation. Note: Recalculates on update, best for per-bar updates."""
    
    def update(self, new_value: float):
        self.q.append(new_value)
        if self.is_ready():
            self.last_value = np.std(list(self.q), ddof=0)
        return self.last_value

class StatefulBBWPercentRank(StatefulFeature):
    """
    Stateful Bollinger Bandwidth Percent Rank.
    - Composes StatefulSMA and StatefulStdDev for efficient updates.
    - Maintains a history of BBW values to calculate the percentile rank.
    """
    
    def __init__(self, period: int = 20, rank_window: int = 250):
        super().__init__(period)
        self.rank_window = rank_window
        self.sma = StatefulSMA(period)
        self.std = StatefulStdDev(period)
        self.bbw_history = deque(maxlen=rank_window)
        self.epsilon = 1e-9

    def update(self, new_value: float):
        current_sma = self.sma.update(new_value)
        current_std = self.std.update(new_value)
        
        if not self.sma.is_ready():
            return self.last_value
            
        current_bbw = (4 * current_std) / (current_sma + self.epsilon)
        self.bbw_history.append(current_bbw)
        
        if self.bbw_history:
            history_array = np.array(self.bbw_history)
            rank = np.sum(history_array < current_bbw) / len(history_array)
            self.last_value = rank * 100
            
        return self.last_value

    def is_ready(self) -> bool:
        return len(self.bbw_history) == self.rank_window

    @classmethod
    def calculate_vectorized(cls, data: pd.Series, period: int = 20, rank_window: int = 250) -> pd.DataFrame:
        """Vectorized calculation of Bollinger Bandwidth Percent Rank."""
        sma = data.rolling(window=period, min_periods=period).mean()
        std = data.rolling(window=period, min_periods=period).std()
        bbw = (4 * std) / (sma + 1e-9)
        # Percentile rank over the rank_window
        bbw_pct_rank = bbw.rolling(window=rank_window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False) * 100
        return pd.DataFrame({'bbw_pct_rank': bbw_pct_rank})

class StatefulPriceDistanceMA(StatefulFeature):
    """Stateful distance from a Simple Moving Average."""
    
    def __init__(self, period: int):
        super().__init__(period)
        self.sma = StatefulSMA(period)
        self.epsilon = 1e-9

    def update(self, new_value: float):
        current_ma = self.sma.update(new_value)
        if not self.sma.is_ready():
            return 0.0
        
        self.last_value = (new_value / (current_ma + self.epsilon)) - 1.0
        return self.last_value

    @classmethod
    def calculate_vectorized(cls, data: pd.Series, period: int) -> pd.DataFrame:
        """Vectorized calculation of Price Distance from MA."""
        sma = data.rolling(window=period, min_periods=period).mean()
        dist = (data / (sma + 1e-9)) - 1.0
        return pd.DataFrame({'price_dist_ma': dist})

class StatefulVWAPDistance(StatefulFeature):
    """
    Stateful VWAP Distance calculator that restores declarative pattern consistency.
    """
    
    def __init__(self, period: int = 9):
        super().__init__(period)
        self.price_buffer = deque(maxlen=period)
        self.volume_buffer = deque(maxlen=period)
        self.epsilon = 1e-9
        
    def update_vwap(self, price: float, volume: float):
        """Special update method for VWAP calculation that takes both price and volume."""
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
        
        if len(self.price_buffer) >= 2:
            prices = np.array(self.price_buffer)
            volumes = np.array(self.volume_buffer)
            total_pv = np.sum(prices * volumes)
            total_volume = np.sum(volumes)
            
            if total_volume > self.epsilon:
                vwap = total_pv / total_volume
                current_price = prices[-1]
                self.last_value = (current_price - vwap) / (vwap + self.epsilon)
            else:
                self.last_value = 0.0
        else:
            self.last_value = 0.0
        return self.last_value
    
    def update(self, new_data_point):
        """Standard update method - expects a tuple (price, volume) or defaults volume to 1.0."""
        if isinstance(new_data_point, (tuple, list)) and len(new_data_point) >= 2:
            price, volume = new_data_point[0], new_data_point[1]
        else:
            price, volume = float(new_data_point), 1.0
        return self.update_vwap(price, volume)

    @classmethod
    def calculate_vectorized(cls, price_series: pd.Series, volume_series: pd.Series, period: int) -> pd.DataFrame:
        """Vectorized calculation of VWAP Distance."""
        pv = price_series * volume_series
        rolling_pv_sum = pv.rolling(window=period, min_periods=period).sum()
        rolling_volume_sum = volume_series.rolling(window=period, min_periods=period).sum()
        vwap = rolling_pv_sum / (rolling_volume_sum + 1e-9)
        dist = (price_series - vwap) / (vwap + 1e-9)
        return pd.DataFrame({'dist_vwap': dist})

class StatefulSRDistances(StatefulFeature):
    """
    Stateful Support/Resistance distance calculator.
    """
    
    def __init__(self, period: int, num_levels: int):
        super().__init__(period)
        self.num_levels = num_levels
        self.last_value: Dict[str, float] = {f'dist_s{i+1}': 1.0 for i in range(num_levels)}
        self.last_value.update({f'dist_r{i+1}': 1.0 for i in range(num_levels)})

    def update(self, new_price: float):
        self.q.append(new_price)
        if not self.is_ready():
            return self.last_value
        self.last_value = self._calculate_sr_for_window(np.array(self.q), self.num_levels)
        return self.last_value

    def get(self) -> Dict[str, float]:
        return self.last_value

    @classmethod
    def _calculate_sr_for_window(cls, window: np.ndarray, num_levels: int) -> Dict[str, float]:
        """Helper to calculate S/R levels for a given window. Shared by incremental and vectorized methods."""
        if len(window) < 2 or np.isnan(window).any():
             # Return a default dictionary if window is too short or contains NaNs
            default_results = {f'dist_s{i+1}': 1.0 for i in range(num_levels)}
            default_results.update({f'dist_r{i+1}': 1.0 for i in range(num_levels)})
            return default_results
            
        current_price = window[-1]
        results = {f'dist_s{i+1}': 1.0 for i in range(num_levels)}
        results.update({f'dist_r{i+1}': 1.0 for i in range(num_levels)})
        peaks, _ = scipy.signal.find_peaks(window, distance=5)
        troughs, _ = scipy.signal.find_peaks(-window, distance=5)
        support_levels = sorted([p for p in window[troughs] if p < current_price], reverse=True)
        for i in range(num_levels):
            if i < len(support_levels): results[f'dist_s{i+1}'] = (current_price - support_levels[i]) / current_price
        resistance_levels = sorted([p for p in window[peaks] if p > current_price])
        for i in range(num_levels):
            if i < len(resistance_levels): results[f'dist_r{i+1}'] = (resistance_levels[i] - current_price) / current_price
        return results

    @classmethod
    def calculate_vectorized(cls, data: pd.Series, period: int, num_levels: int) -> pd.DataFrame:
        """
        Vectorized calculation of S/R Distances using a robust manual rolling window.
        This bypasses the C-level TypeError in pandas' rolling().apply().
        """
        # Drop NaNs from the input series to prevent issues.
        data = data.dropna()
        if data.empty:
            return pd.DataFrame()

        values = data.to_numpy()
        results = []
        
        # Manually iterate through the data to create windows.
        for i in range(len(values) - period + 1):
            window = values[i : i + period]
            results.append(cls._calculate_sr_for_window(window, num_levels))
            
        # The results correspond to the index from (period - 1) onwards.
        result_index = data.index[period - 1:]
        
        # Create the final DataFrame from the list of dictionaries.
        sr_df = pd.DataFrame(results, index=result_index)
        return sr_df

# Map string names to classes for the processor's vectorized generation
VECTORIZED_CALCULATOR_MAP = {
    'StatefulBBWPercentRank': StatefulBBWPercentRank,
    'StatefulPriceDistanceMA': StatefulPriceDistanceMA,
    'StatefulVWAPDistance': StatefulVWAPDistance,
    'StatefulSRDistances': StatefulSRDistances,
}