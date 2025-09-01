"""
Stateful, Incremental Feature Calculators for High-Performance Trading Environments.
These classes maintain an internal state and update incrementally, avoiding
expensive recalculations over large windows at every step.
"""

from collections import deque
import numpy as np
import pandas as pd
import scipy.signal
from typing import List, Dict

class StatefulFeature:
    """Base class for a stateful feature calculator."""
    def __init__(self, period: int):
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

class StatefulSMA(StatefulFeature):
    """Stateful Simple Moving Average."""
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
    """Stateful Standard Deviation."""
    def update(self, new_value: float):
        self.q.append(new_value)
        if self.is_ready():
            # For performance, this is sufficient as it's only called on new bars.
            # A more complex implementation could use Welford's algorithm for O(1) updates.
            self.last_value = np.std(list(self.q), ddof=0)
        return self.last_value

class StatefulBBWPercentRank(StatefulFeature):
    """
    Stateful Bollinger Bandwidth Percent Rank.
    - Updates SMA and StdDev incrementally on new bars.
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
        # Update underlying metrics
        current_sma = self.sma.update(new_value)
        current_std = self.std.update(new_value)

        if not self.sma.is_ready():
            return self.last_value # Not enough data yet

        # Calculate new BBW value
        current_bbw = (4 * current_std) / (current_sma + self.epsilon)
        self.bbw_history.append(current_bbw)

        # Calculate percentile rank
        if self.bbw_history:
            history_array = np.array(self.bbw_history)
            rank = np.sum(history_array < current_bbw) / len(history_array)
            self.last_value = rank * 100
        
        return self.last_value
    
    def is_ready(self) -> bool:
        return len(self.bbw_history) == self.rank_window

class StatefulPriceDistanceMA(StatefulFeature):
    """Stateful distance from a Simple Moving Average."""
    def __init__(self, period: int):
        super().__init__(period)
        self.sma = StatefulSMA(period)

    def update(self, new_value: float):
        current_ma = self.sma.update(new_value)
        if not self.sma.is_ready():
            return 0.0
            
        self.last_value = (new_value / current_ma) - 1.0
        return self.last_value

class StatefulSRDistances(StatefulFeature):
    """
    Stateful Support/Resistance distance calculator.
    This re-calculates on new bars, which is far more efficient than every step.
    """
    def __init__(self, period: int, num_levels: int):
        super().__init__(period)
        self.num_levels = num_levels
        self.last_value: Dict[str, float] = {}

    def update(self, new_price: float):
        self.q.append(new_price) # The deque now holds the price window
        if not self.is_ready():
            return self.last_value

        window_vals = np.array(self.q)
        current_price = window_vals[-1]

        # Find peaks and troughs
        peaks, _ = scipy.signal.find_peaks(window_vals, distance=5)
        troughs, _ = scipy.signal.find_peaks(-window_vals, distance=5)

        # Support Levels
        support_levels = sorted([p for p in window_vals[troughs] if p < current_price], reverse=True)
        for level in range(self.num_levels):
            key = f'dist_s{level+1}'
            if level < len(support_levels):
                self.last_value[key] = (current_price - support_levels[level]) / current_price
            else:
                self.last_value[key] = 1.0

        # Resistance Levels
        resistance_levels = sorted([p for p in window_vals[peaks] if p > current_price])
        for level in range(self.num_levels):
            key = f'dist_r{level+1}'
            if level < len(resistance_levels):
                self.last_value[key] = (resistance_levels[level] - current_price) / current_price
            else:
                self.last_value[key] = 1.0
                
        return self.last_value

    def get(self) -> Dict[str, float]:
        return self.last_value
