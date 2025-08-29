
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

# --- The Main Multi-Timeframe Hybrid Model (UPDATED with LSTM Head) ---

class MultiTimeframeHybridTIN(nn.Module):
    """
    A singular agent with indicator cells analyzing multiple timeframes,
    plus direct context features (regimes), feeding into a recurrent (LSTM) decision head.
    This model processes sequences of states to capture temporal dynamics.
    """
    def __init__(self, lstm_hidden_size=64, lstm_layers=2):
        super(MultiTimeframeHybridTIN, self).__init__()
        print("--- Building Multi-Timeframe Hybrid TIN with LSTM Head ---")

        # --- Instantiate Indicator Cells for each Timeframe ---
        self.cell_5m = LearnableMACDCell()
        self.cell_15m = LearnableRSICell()
        self.cell_1h = LearnableMACDCell()
        
        # --- Define the Integration and LSTM Decision Head ---
        num_indicator_signals = sum(1 for key in SETTINGS.strategy.LOOKBACK_PERIODS if key.startswith('price_'))
        num_context_features = SETTINGS.strategy.LOOKBACK_PERIODS['context']
        
        self.input_size = num_indicator_signals + num_context_features
        print(f"LSTM head input feature size per timestep: {self.input_size}")

        # Recurrent layer to process sequences of indicator/context states
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True  # Crucial for easier data handling: (Batch, Seq, Feature)
        )

        # Final linear layer to map LSTM output to Q-values
        self.q_head = nn.Linear(lstm_hidden_size, SETTINGS.strategy.ACTION_SPACE_SIZE)

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # The input tensors in state_dict are now expected to be SEQUENCES
        # Shape: (batch_size, sequence_length, feature_dimension)
        # e.g., price_5m: (B, S, 50), context_features: (B, S, 4)

        price_5m = state_dict['price_5m']
        price_15m = state_dict['price_15m']
        price_1h = state_dict['price_1h']
        context_features = state_dict['context']
        
        batch_size, seq_len = price_5m.shape[0], price_5m.shape[1]
        
        def process_sequence(cell, price_seq):
            """
            Helper to process a sequence of price windows through a stateless indicator cell.
            It reshapes the input from (Batch, Seq, Lookback) to (Batch*Seq, Lookback),
            processes all timesteps at once for efficiency, and then reshapes the output
            back to (Batch, Seq, 1).
            """
            price_flat = price_seq.view(batch_size * seq_len, -1)
            signal_flat = cell(price_flat)
            return signal_flat.view(batch_size, seq_len, 1)

        # --- Get signal sequences from all cells in parallel ---
        s_5m_seq = process_sequence(self.cell_5m, price_5m)
        s_15m_seq = process_sequence(self.cell_15m, price_15m)
        s_1h_seq = process_sequence(self.cell_1h, price_1h)

        # --- Integration Layer: Concatenate all signal and feature sequences ---
        # This creates a final input sequence of shape (Batch, Seq, input_size)
        final_input_sequence = torch.cat([
            s_5m_seq, s_15m_seq, s_1h_seq,
            context_features
        ], dim=2) # Concatenate along the feature dimension
        
        # --- Pass sequence through LSTM ---
        # lstm_out shape: (Batch, Seq, lstm_hidden_size)
        # We don't need the hidden/cell states for Q-learning here.
        lstm_out, _ = self.lstm(final_input_sequence)

        # --- Final Decision ---
        # For Q-value prediction, we only need the output from the LAST element of the sequence.
        last_time_step_out = lstm_out[:, -1, :]
        q_values = self.q_head(last_time_step_out)
        
        return q_values
