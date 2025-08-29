# Zero1-main/tins.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .config import SETTINGS

# --- Core Building Block (Paper's Philosophy) ---

class IndicatorLinear(nn.Module):
    """A linear layer designed to act as a learnable Moving Average."""
    def __init__(self, lookback_period: int, is_ema_init: bool = True):
        super(IndicatorLinear, self).__init__()
        self.ma_layer = nn.Linear(lookback_period, 1, bias=False)
        if is_ema_init: self.initialize_as_ema()
        else: self.initialize_as_sma()

    def initialize_as_sma(self):
        with torch.no_grad(): self.ma_layer.weight.fill_(1.0 / self.ma_layer.in_features)

    def initialize_as_ema(self):
        period = self.ma_layer.in_features; alpha = 2.0 / (period + 1.0)
        with torch.no_grad():
            powers = torch.arange(period - 1, -1, -1, dtype=torch.float32); ema_weights = alpha * ((1 - alpha) ** powers)
            ema_weights /= torch.sum(ema_weights); self.ma_layer.weight.data = ema_weights.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.ma_layer(x)

# --- Learnable Indicator Cells ---

class LearnableMACDCell(nn.Module):
    """Calculates a learnable MACD histogram value from a price series."""
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        super(LearnableMACDCell, self).__init__()
        self.fast_ma = IndicatorLinear(fast_period); self.slow_ma = IndicatorLinear(slow_period)
        self.signal_ma = IndicatorLinear(signal_period); self.periods = {'fast': fast_period, 'slow': slow_period, 'signal': signal_period}

    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        batch_size, device = price_series.shape[0], price_series.device
        macd_line_history = torch.zeros(batch_size, self.periods['signal'], device=device)
        for i in range(self.periods['signal']):
            end_idx = price_series.shape[1] - i
            fast_window = price_series[:, end_idx - self.periods['fast'] : end_idx]
            slow_window = price_series[:, end_idx - self.periods['slow'] : end_idx]
            macd_val = self.fast_ma(fast_window) - self.slow_ma(slow_window)
            macd_line_history[:, -1 - i] = macd_val.squeeze(-1)
        signal_line = self.signal_ma(macd_line_history); histogram = macd_line_history[:, -1].unsqueeze(1) - signal_line
        return histogram

class LearnableRSICell(nn.Module):
    """Calculates a learnable RSI value from a price series."""
    def __init__(self, rsi_period=14):
        super(LearnableRSICell, self).__init__()
        self.avg_gain = IndicatorLinear(rsi_period); self.avg_loss = IndicatorLinear(rsi_period); self.rsi_period = rsi_period

    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        price_window = price_series[:, -self.rsi_period-1:]; diffs = price_window[:, 1:] - price_window[:, :-1]
        gains = F.relu(diffs); losses = F.relu(-diffs)
        avg_gain_val = self.avg_gain(gains); avg_loss_val = self.avg_loss(losses)
        rs = avg_gain_val / (avg_loss_val + 1e-8); rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi / 50.0 - 1.0 # Normalize to be roughly between -1 and 1

# --- The Main Orchestrator Model ---

class HybridMonolithicTIN(nn.Module):
    """A singular agent containing multiple, learnable indicator cells."""
    def __init__(self):
        super(HybridMonolithicTIN, self).__init__()
        print("--- Building Hybrid Monolithic TIN with Learnable Indicator Cells ---")
        self.macd_cell = LearnableMACDCell()
        self.rsi_cell = LearnableRSICell()
        num_indicator_signals = 2 # (MACD histogram, RSI value)
        self.decision_head = nn.Sequential(
            nn.Linear(num_indicator_signals, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, SETTINGS.strategy.ACTION_SPACE_SIZE))

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        price_series = state_dict['price_1m']
        macd_signal = self.macd_cell(price_series)
        rsi_signal = self.rsi_cell(price_series)
        indicator_vector = torch.cat([macd_signal, rsi_signal], dim=1)
        return self.decision_head(indicator_vector)
