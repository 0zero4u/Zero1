
# Zero1-main/tins.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .config import SETTINGS

# --- Core Building Blocks & Learnable Cells (Unchanged) ---

class IndicatorLinear(nn.Module):
    """A linear layer designed to act as a learnable Moving Average."""
    def __init__(self, lookback_period: int, is_ema_init: bool = True):
        super(IndicatorLinear, self).__init__(); self.ma_layer = nn.Linear(lookback_period, 1, bias=False)
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
        return rsi / 50.0 - 1.0

# --- The Main Multi-Timeframe Hybrid Model ---

class MultiTimeframeHybridTIN(nn.Module):
    """
    A singular agent with indicator cells analyzing multiple timeframes,
    plus direct context features (regimes), feeding into a decision head.
    """
    def __init__(self):
        super(MultiTimeframeHybridTIN, self).__init__()
        print("--- Building Multi-Timeframe Hybrid TIN ---")

        # --- Instantiate Indicator Cells for each Timeframe ---
        self.cell_5m = LearnableMACDCell()
        self.cell_15m = LearnableRSICell()
        self.cell_1h = LearnableMACDCell()
        
        # --- Define the Integration and Decision Head Dynamically ---
        # Calculate number of indicator signals by counting 'price_' keys
        num_indicator_signals = sum(1 for key in SETTINGS.strategy.LOOKBACK_PERIODS if key.startswith('price_'))
        num_context_features = SETTINGS.strategy.LOOKBACK_PERIODS['context']
        
        input_size = num_indicator_signals + num_context_features
        print(f"Decision head input size: {input_size} ({num_indicator_signals} indicator signals + {num_context_features} context features)")

        self.decision_head = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, SETTINGS.strategy.ACTION_SPACE_SIZE))

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # --- Get data for each component from the state dictionary ---
        price_5m = state_dict['price_5m']
        price_15m = state_dict['price_15m']
        price_1h = state_dict['price_1h']
        context_features = state_dict['context']

        # --- Get signals from all cells in parallel ---
        s_5m = self.cell_5m(price_5m)
        s_15m = self.cell_15m(price_15m)
        s_1h = self.cell_1h(price_1h)

        # --- Integration Layer: Concatenate all signals and features ---
        final_input = torch.cat([
            s_5m, s_15m, s_1h,
            context_features
        ], dim=1)
        
        # --- Final Decision ---
        return self.decision_head(final_input)
