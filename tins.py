

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .config import SETTINGS

# --- Core Building Blocks & Learnable Cells (Unchanged part) ---
class IndicatorLinear(nn.Module):
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

class LearnableROCCell(nn.Module):
    def __init__(self, roc_period=12):
        super(LearnableROCCell, self).__init__(); self.roc_period = roc_period
    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        close_i_n = price_series[:, -1 - self.roc_period].unsqueeze(-1)
        close_i = price_series[:, -1].unsqueeze(-1)
        return (close_i - close_i_n) / (close_i_n + 1e-8)

class LearnableATRCell(nn.Module):
    def __init__(self, atr_period=14):
        super(LearnableATRCell, self).__init__(); self.atr_period = atr_period; self.learnable_ema = IndicatorLinear(atr_period)
    def forward(self, ohlc_series: torch.Tensor) -> torch.Tensor:
        # Note: This cell only uses O,H,L,C, so it's compatible with OHLCV input
        highs, lows, closes = ohlc_series[:, :, 1], ohlc_series[:, :, 2], ohlc_series[:, :, 3]
        prev_closes = torch.cat([closes[:, :1], closes[:, :-1]], dim=1)
        tr = torch.max(torch.max(highs - lows, torch.abs(highs - prev_closes)), torch.abs(lows - prev_closes))
        atr = self.learnable_ema(tr[:, -self.atr_period:])
        return atr / (closes[:, -1].unsqueeze(-1) + 1e-8)

# --- NEW: Learnable VWAP Cell ---
class LearnableVWAPCell(nn.Module):
    def __init__(self, vwap_period=20):
        super(LearnableVWAPCell, self).__init__()
        self.vwap_period = vwap_period
        self.num_ma = IndicatorLinear(vwap_period) # For Typical Price * Volume
        self.den_ma = IndicatorLinear(vwap_period) # For Volume

    def forward(self, ohlcv_series: torch.Tensor) -> torch.Tensor:
        # ohlcv_series shape: (batch, lookback, 5) -> O,H,L,C,V
        window = ohlcv_series[:, -self.vwap_period:]
        highs, lows, closes, volumes = window[:, :, 1], window[:, :, 2], window[:, :, 3], window[:, :, 4]
        
        typical_price = (highs + lows + closes) / 3.0
        tpv = typical_price * volumes
        
        vwap_num = self.num_ma(tpv)
        vwap_den = self.den_ma(volumes)
        vwap = vwap_num / (vwap_den + 1e-8)
        
        # Return a normalized signal: distance of current close from VWAP
        current_close = closes[:, -1].unsqueeze(-1)
        return (current_close - vwap) / (vwap + 1e-8)

class LearnableBBandsCell(nn.Module):
    def __init__(self, bbands_period=20):
        super(LearnableBBandsCell, self).__init__(); self.bbands_period = bbands_period; self.ma = IndicatorLinear(bbands_period); self.k = nn.Parameter(torch.tensor(2.0))
    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        window = price_series[:, -self.bbands_period:]
        ma_val = self.ma(window); std_val = torch.std(window, dim=1, keepdim=True)
        upper_band = ma_val + self.k * std_val; lower_band = ma_val - self.k * std_val
        percent_b = (price_series[:, -1].unsqueeze(-1) - lower_band) / (upper_band - lower_band + 1e-8)
        return (percent_b * 2) - 1.0

class LearnableRSICell(nn.Module):
    def __init__(self, rsi_period=14):
        super(LearnableRSICell, self).__init__(); self.avg_gain = IndicatorLinear(rsi_period); self.avg_loss = IndicatorLinear(rsi_period); self.rsi_period = rsi_period
    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        diffs = price_series[:, -self.rsi_period:] - price_series[:, -self.rsi_period-1:-1]
        avg_gain_val = self.avg_gain(F.relu(diffs)); avg_loss_val = self.avg_loss(F.relu(-diffs))
        rs = avg_gain_val / (avg_loss_val + 1e-8); rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi / 50.0 - 1.0

# --- The Main Multi-Timeframe Feature Extractor for Stable-Baselines3 ---

class MultiTimeframeFeatureExtractor(BaseFeaturesExtractor):
    """
    A custom SB3 feature extractor that uses a wide array of indicator cells analyzing
    multiple timeframes, plus direct context features, feeding into a recurrent (LSTM)
    head to produce a final feature vector for PPO's actor and critic networks.
    """
    def __init__(self, observation_space: spaces.Dict, lstm_hidden_size=64, lstm_layers=2):
        super().__init__(observation_space, features_dim=lstm_hidden_size)
        print("--- Building Multi-Timeframe Feature Extractor for SB3 ---")

        # --- Instantiate Indicator Cells ---
        # NEW: 1m Ultra-Short Term Cells
        self.cell_1m_roc = LearnableROCCell(roc_period=14)
        self.cell_1m_momentum = LearnableMACDCell(fast_period=8, slow_period=21, signal_period=5)
        self.cell_1m_atr = LearnableATRCell(atr_period=20)
        # NEW: 3m Primary Cells
        self.cell_3m_vwap = LearnableVWAPCell(vwap_period=20)
        self.cell_3m_atr = LearnableATRCell(atr_period=14)
        
        # Original Cells
        self.cell_5m_macd_fast = LearnableMACDCell(fast_period=6, slow_period=13, signal_period=5)
        self.cell_5m_macd_slow = LearnableMACDCell(fast_period=24, slow_period=52, signal_period=18)
        self.cell_5m_roc_fast = LearnableROCCell(roc_period=9)
        self.cell_5m_roc_slow = LearnableROCCell(roc_period=21)
        self.cell_15m_rsi = LearnableRSICell()
        self.cell_15m_atr = LearnableATRCell()
        self.cell_15m_bbands = LearnableBBandsCell()
        self.cell_1h_macd_fast = LearnableMACDCell(fast_period=6, slow_period=13, signal_period=5)
        self.cell_1h_macd_slow = LearnableMACDCell(fast_period=24, slow_period=52, signal_period=18)
        
        # --- Define the Integration and LSTM Decision Head ---
        num_indicator_signals = 14 # 9 original + 5 new
        num_context_features = SETTINGS.strategy.LOOKBACK_PERIODS['context']
        self.input_size = num_indicator_signals + num_context_features
        print(f"LSTM head input feature size per timestep: {self.input_size}")

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len = observations['price_1m'].shape[0:2]
        
        def process_sequence(cell, data_seq):
            data_flat = data_seq.reshape(batch_size * seq_len, *data_seq.shape[2:])
            signal_flat = cell(data_flat)
            return signal_flat.view(batch_size, seq_len, 1)

        # --- Get signal sequences from all cells ---
        # NEW: 1m and 3m signals
        s_1m_roc_seq = process_sequence(self.cell_1m_roc, observations['price_1m'])
        s_1m_momentum_seq = process_sequence(self.cell_1m_momentum, observations['price_1m'])
        s_1m_atr_seq = process_sequence(self.cell_1m_atr, observations['ohlc_1m'])
        s_3m_vwap_seq = process_sequence(self.cell_3m_vwap, observations['ohlc_3m'])
        s_3m_atr_seq = process_sequence(self.cell_3m_atr, observations['ohlc_3m'])

        # Original signals
        s_5m_macd_fast_seq = process_sequence(self.cell_5m_macd_fast, observations['price_5m'])
        s_5m_macd_slow_seq = process_sequence(self.cell_5m_macd_slow, observations['price_5m'])
        s_5m_roc_fast_seq = process_sequence(self.cell_5m_roc_fast, observations['price_5m'])
        s_5m_roc_slow_seq = process_sequence(self.cell_5m_roc_slow, observations['price_5m'])
        s_15m_rsi_seq = process_sequence(self.cell_15m_rsi, observations['price_15m'])
        s_15m_atr_seq = process_sequence(self.cell_15m_atr, observations['ohlc_15m'])
        s_15m_bbands_seq = process_sequence(self.cell_15m_bbands, observations['price_15m'])
        s_1h_macd_fast_seq = process_sequence(self.cell_1h_macd_fast, observations['price_1h'])
        s_1h_macd_slow_seq = process_sequence(self.cell_1h_macd_slow, observations['price_1h'])

        # --- Integration Layer: Concatenate all signal and feature sequences ---
        final_input_sequence = torch.cat([
            # New ultra-short signals first
            s_1m_roc_seq, s_1m_momentum_seq, s_1m_atr_seq,
            s_3m_vwap_seq, s_3m_atr_seq,
            # Original signals
            s_5m_macd_fast_seq, s_5m_macd_slow_seq, s_5m_roc_fast_seq, s_5m_roc_slow_seq,
            s_15m_rsi_seq, s_15m_atr_seq, s_15m_bbands_seq,
            s_1h_macd_fast_seq, s_1h_macd_slow_seq,
            observations['context']
        ], dim=2)
        
        lstm_out, _ = self.lstm(final_input_sequence)
        return lstm_out[:, -1, :]
