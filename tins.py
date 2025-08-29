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

# --- NEW Learnable Indicator Cells ---

class LearnableROCCell(nn.Module):
    """Calculates Rate of Change (ROC) from a price series."""
    def __init__(self, roc_period=12):
        super(LearnableROCCell, self).__init__()
        self.roc_period = roc_period
    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        close_i_n = price_series[:, -1 - self.roc_period].unsqueeze(-1)
        close_i = price_series[:, -1].unsqueeze(-1)
        roc = (close_i - close_i_n) / (close_i_n + 1e-8)
        return roc

class LearnableATRCell(nn.Module):
    """Calculates a learnable Average True Range (ATR) from an OHLC series."""
    def __init__(self, atr_period=14):
        super(LearnableATRCell, self).__init__()
        self.atr_period = atr_period
        self.learnable_ema = IndicatorLinear(atr_period)
    def forward(self, ohlc_series: torch.Tensor) -> torch.Tensor:
        # ohlc_series shape: (batch, lookback, 4)
        highs = ohlc_series[:, :, 1]
        lows = ohlc_series[:, :, 2]
        closes = ohlc_series[:, :, 3]
        prev_closes = torch.cat([closes[:, :1], closes[:, :-1]], dim=1)
        
        tr1 = highs - lows
        tr2 = torch.abs(highs - prev_closes)
        tr3 = torch.abs(lows - prev_closes)
        tr = torch.max(torch.max(tr1, tr2), tr3)

        tr_window = tr[:, -self.atr_period:]
        atr = self.learnable_ema(tr_window)
        
        # Normalize ATR by last close price
        last_close = closes[:, -1].unsqueeze(-1)
        return atr / (last_close + 1e-8)

class LearnableBBandsCell(nn.Module):
    """Calculates a learnable Bollinger Bands %B value."""
    def __init__(self, bbands_period=20):
        super(LearnableBBandsCell, self).__init__()
        self.bbands_period = bbands_period
        self.ma = IndicatorLinear(bbands_period)
        self.k = nn.Parameter(torch.tensor(2.0)) # Learnable standard deviation multiplier
    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        window = price_series[:, -self.bbands_period:]
        ma_val = self.ma(window)
        std_val = torch.std(window, dim=1, keepdim=True)
        
        upper_band = ma_val + self.k * std_val
        lower_band = ma_val - self.k * std_val
        last_price = price_series[:, -1].unsqueeze(-1)
        
        percent_b = (last_price - lower_band) / (upper_band - lower_band + 1e-8)
        return (percent_b * 2) - 1.0 # Scale to [-1, 1]

# --- The Main Multi-Timeframe Hybrid Model (UPDATED with Dueling DQN Head) ---

class MultiTimeframeHybridTIN(nn.Module):
    """
    A singular agent with a wide array of indicator cells analyzing multiple timeframes and speeds,
    plus direct context features, feeding into a recurrent (LSTM) Dueling decision head.
    """
    def __init__(self, lstm_hidden_size=64, lstm_layers=2):
        super(MultiTimeframeHybridTIN, self).__init__()
        print("--- Building Multi-Timeframe Hybrid TIN with Dueling LSTM Head & Multi-Speed Cells ---")

        # --- Instantiate Indicator Cells for each Timeframe and Speed ---
        # 5-minute Tactical Cells
        self.cell_5m_macd_fast = LearnableMACDCell(fast_period=6, slow_period=13, signal_period=5)
        self.cell_5m_macd_slow = LearnableMACDCell(fast_period=24, slow_period=52, signal_period=18)
        self.cell_5m_roc_fast = LearnableROCCell(roc_period=9)
        self.cell_5m_roc_slow = LearnableROCCell(roc_period=21)
        # 15-minute Short-Term Cells
        self.cell_15m_rsi = LearnableRSICell()
        self.cell_15m_atr = LearnableATRCell()
        self.cell_15m_bbands = LearnableBBandsCell()
        # 1-hour Strategic Cells
        self.cell_1h_macd_fast = LearnableMACDCell(fast_period=6, slow_period=13, signal_period=5)
        self.cell_1h_macd_slow = LearnableMACDCell(fast_period=24, slow_period=52, signal_period=18)
        
        # --- Define the Integration and LSTM Decision Head ---
        num_indicator_signals = 9 # 4 (5m) + 3 (15m) + 2 (1h)
        num_context_features = SETTINGS.strategy.LOOKBACK_PERIODS['context']
        
        self.input_size = num_indicator_signals + num_context_features
        print(f"LSTM head input feature size per timestep: {self.input_size}")

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        # --- Dueling DQN Head ---
        self.advantage_stream = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size // 2, SETTINGS.strategy.ACTION_SPACE_SIZE)
        )
        self.value_stream = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size // 2, 1)
        )

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        price_5m = state_dict['price_5m']
        price_15m = state_dict['price_15m']
        ohlc_15m = state_dict['ohlc_15m']
        price_1h = state_dict['price_1h']
        context_features = state_dict['context']
        
        batch_size, seq_len = price_5m.shape[0], price_5m.shape[1]
        
        def process_sequence(cell, data_seq):
            data_flat = data_seq.reshape(batch_size * seq_len, *data_seq.shape[2:])
            signal_flat = cell(data_flat)
            return signal_flat.view(batch_size, seq_len, 1)

        # --- Get signal sequences from all cells in parallel ---
        s_5m_macd_fast_seq = process_sequence(self.cell_5m_macd_fast, price_5m)
        s_5m_macd_slow_seq = process_sequence(self.cell_5m_macd_slow, price_5m)
        s_5m_roc_fast_seq = process_sequence(self.cell_5m_roc_fast, price_5m)
        s_5m_roc_slow_seq = process_sequence(self.cell_5m_roc_slow, price_5m)
        
        s_15m_rsi_seq = process_sequence(self.cell_15m_rsi, price_15m)
        s_15m_atr_seq = process_sequence(self.cell_15m_atr, ohlc_15m)
        s_15m_bbands_seq = process_sequence(self.cell_15m_bbands, price_15m)
        
        s_1h_macd_fast_seq = process_sequence(self.cell_1h_macd_fast, price_1h)
        s_1h_macd_slow_seq = process_sequence(self.cell_1h_macd_slow, price_1h)

        # --- Integration Layer: Concatenate all signal and feature sequences ---
        final_input_sequence = torch.cat([
            s_5m_macd_fast_seq, s_5m_macd_slow_seq, s_5m_roc_fast_seq, s_5m_roc_slow_seq,
            s_15m_rsi_seq, s_15m_atr_seq, s_15m_bbands_seq,
            s_1h_macd_fast_seq, s_1h_macd_slow_seq,
            context_features
        ], dim=2)
        
        lstm_out, _ = self.lstm(final_input_sequence)
        last_time_step_out = lstm_out[:, -1, :]
        
        advantages = self.advantage_stream(last_time_step_out)
        values = self.value_stream(last_time_step_out)
        
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
