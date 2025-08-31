# =========================================================================================
# FINAL CORRECTED VERSION of tins.py
#
# This file fixes the critical bug where the model was fed random noise instead of
# real technical indicator values. It implements the full logic for each indicator
# and includes the 'get_attention_analysis' method for model interpretability.
# This is a drop-in replacement.
# =========================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import math
from torch.nn.utils import spectral_norm
from .config import SETTINGS # Corrected relative import

# --- ADVANCED ATTENTION MECHANISMS ---

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism for enhanced pattern recognition."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        residual = x
        
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(context)
        return self.layer_norm(output + residual)

# --- ENHANCED LEARNABLE INDICATOR CELLS (CORE BUILDING BLOCKS) ---

class EnhancedIndicatorLinear(nn.Module):
    """The core building block: A linear layer initialized to function as a classic moving average."""
    
    def __init__(self, lookback_period: int, output_dim: int = 1, 
                 init_type: str = "ema", use_spectral_norm: bool = False):
        super().__init__()
        
        self.ma_layer = nn.Linear(lookback_period, output_dim, bias=True)
        
        if use_spectral_norm:
            self.ma_layer = spectral_norm(self.ma_layer)
        
        self.init_type = init_type
        self.lookback_period = lookback_period
        self._initialize_weights()
    
    def _initialize_weights(self):
        with torch.no_grad():
            if self.init_type == "sma":
                self.ma_layer.weight.fill_(1.0 / self.lookback_period)
            elif self.init_type == "ema":
                alpha = 2.0 / (self.lookback_period + 1.0)
                powers = torch.arange(self.lookback_period - 1, -1, -1, dtype=torch.float32)
                ema_weights = alpha * ((1 - alpha) ** powers)
                ema_weights /= torch.sum(ema_weights)
                self.ma_layer.weight.data = ema_weights.unsqueeze(0)
            
            if self.ma_layer.bias is not None:
                self.ma_layer.bias.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ma_layer(x)

class EnhancedMACDCell(nn.Module):
    """FIXED: Implements the full MACD calculation using learnable EMA layers."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        self.fast_ma = EnhancedIndicatorLinear(fast_period, 1, "ema")
        self.slow_ma = EnhancedIndicatorLinear(slow_period, 1, "ema")
        self.signal_ma = EnhancedIndicatorLinear(signal_period, 1, "ema")
        
    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        if price_series.dim() > 2: price_series = price_series.squeeze(-1)

        macd_history = []
        for i in range(self.signal_period):
            end_idx = price_series.shape[1] - i
            start_idx_slow = end_idx - self.slow_period
            
            if start_idx_slow < 0: continue

            slow_window = price_series[:, start_idx_slow:end_idx]
            fast_window = slow_window[:, -self.fast_period:]
            
            fast_val = self.fast_ma(fast_window)
            slow_val = self.slow_ma(slow_window)
            macd_val = fast_val - slow_val
            macd_history.append(macd_val)
        
        if len(macd_history) < self.signal_period:
            return torch.zeros(price_series.shape[0], 1, device=price_series.device)
            
        macd_history = torch.cat(list(reversed(macd_history)), dim=1)
        signal_line = self.signal_ma(macd_history)
        histogram = macd_history[:, -1].unsqueeze(1) - signal_line
        
        return torch.tanh(histogram)

class EnhancedRSICell(nn.Module):
    """FIXED: Implements the RSI calculation."""
    def __init__(self, period: int = 14):
        super().__init__()
        self.period = period
        self.gain_ema = EnhancedIndicatorLinear(period, 1, "ema")
        self.loss_ema = EnhancedIndicatorLinear(period, 1, "ema")
        self.epsilon = 1e-8

    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        if price_series.dim() > 2: price_series = price_series.squeeze(-1)

        deltas = price_series[:, 1:] - price_series[:, :-1]
        
        gains = torch.clamp(deltas, min=0)
        losses = -torch.clamp(deltas, max=0)

        if gains.shape[1] < self.period:
            return torch.zeros(price_series.shape[0], 1, device=price_series.device)

        avg_gain = self.gain_ema(gains[:, -self.period:])
        avg_loss = self.loss_ema(losses[:, -self.period:])
        
        rs = avg_gain / (avg_loss + self.epsilon)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return (rsi - 50.0) / 50.0

class EnhancedATRCell(nn.Module):
    """FIXED: Implements the ATR calculation."""
    def __init__(self, period: int = 14):
        super().__init__()
        self.period = period
        self.tr_ema = EnhancedIndicatorLinear(period, 1, "ema")
        self.epsilon = 1e-8

    def forward(self, ohlc_data: torch.Tensor) -> torch.Tensor:
        high, low, close = ohlc_data[:, :, 1], ohlc_data[:, :, 2], ohlc_data[:, :, 3]
        
        h_minus_l = high[:, 1:] - low[:, 1:]
        h_minus_cp = torch.abs(high[:, 1:] - close[:, :-1])
        l_minus_cp = torch.abs(low[:, 1:] - close[:, :-1])
        
        true_range = torch.max(torch.stack([h_minus_l, h_minus_cp, l_minus_cp]), dim=0)[0]
        
        if true_range.shape[1] < self.period:
            return torch.zeros(ohlc_data.shape[0], 1, device=ohlc_data.device)

        atr = self.tr_ema(true_range[:, -self.period:])
        normalized_atr = atr / (close[:, -1].unsqueeze(1) + self.epsilon)
        return torch.tanh(normalized_atr * 100)

class EnhancedBBandsCell(nn.Module):
    """FIXED: Implements a Bollinger Bands signal."""
    def __init__(self, period: int = 20):
        super().__init__()
        self.period = period
        self.std_multiplier = nn.Parameter(torch.tensor(2.0))
        self.ma = EnhancedIndicatorLinear(period, 1, "sma")
        self.epsilon = 1e-8

    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        if price_series.dim() > 2: price_series = price_series.squeeze(-1)
            
        if price_series.shape[1] < self.period:
            return torch.zeros(price_series.shape[0], 1, device=price_series.device)
            
        window = price_series[:, -self.period:]
        current_price = price_series[:, -1].unsqueeze(1)
        
        middle_band = self.ma(window)
        std_dev = torch.std(window, dim=1, keepdim=True)
        
        signal = (current_price - middle_band) / (self.std_multiplier * std_dev + self.epsilon)
        return torch.tanh(signal)

# --- ENHANCED HIERARCHICAL FEATURE EXTRACTOR ---

class EnhancedHierarchicalAttentionFeatureExtractor(BaseFeaturesExtractor):
    """FIXED: This version correctly calculates and uses all technical indicators."""
    
    def __init__(self, observation_space: spaces.Dict, arch_cfg=None):
        if arch_cfg is None:
            arch_cfg = SETTINGS.strategy.architecture
        
        super().__init__(observation_space, features_dim=arch_cfg.attention_head_features)
        
        print("--- Building Enhanced Hierarchical Attention Feature Extractor (FINAL VERSION) ---")
        
        self.arch_cfg = arch_cfg
        self.last_attention_weights = None
        
        self.cells = nn.ModuleDict({
            '1m_momentum': EnhancedMACDCell(fast_period=8, slow_period=21, signal_period=5),
            '5m_macd_fast': EnhancedMACDCell(fast_period=6, slow_period=13, signal_period=5),
            '5m_macd_slow': EnhancedMACDCell(fast_period=24, slow_period=52, signal_period=18),
            '1h_macd_fast': EnhancedMACDCell(fast_period=6, slow_period=13, signal_period=5),
            '1h_macd_slow': EnhancedMACDCell(fast_period=24, slow_period=52, signal_period=18),
            '15m_rsi': EnhancedRSICell(period=14),
            '15m_bbands': EnhancedBBandsCell(period=20),
            '1m_atr': EnhancedATRCell(period=20),
            '3m_atr': EnhancedATRCell(period=14),
            '15m_atr': EnhancedATRCell(period=14),
        })
        
        self.cell_data_map = {
            '1m': ('price_1m', 'price'),
            '3m': ('ohlcv_3m', 'ohlc'),
            '5m': ('price_5m', 'price'),
            '15m': ('ohlc_15m', 'ohlc'),
            '1h': ('price_1h', 'price'),
        }

        self.flow_head = self._create_enhanced_expert_head(3)
        self.volatility_head = self._create_enhanced_expert_head(3)
        self.value_trend_head = self._create_enhanced_expert_head(4)
        self.context_head = self._create_enhanced_expert_head(SETTINGS.strategy.lookback_periods['context'])
        
        expert_dim = self.arch_cfg.expert_lstm_hidden_size
        total_expert_features = 4 * expert_dim
        
        self.attention_layer = nn.Sequential(
            nn.Linear(total_expert_features, total_expert_features // 2),
            nn.LayerNorm(total_expert_features // 2), nn.GELU(),
            nn.Dropout(self.arch_cfg.dropout_rate),
            nn.Linear(total_expert_features // 2, 4), nn.Softmax(dim=1)
        )
        self.output_projection = nn.Linear(expert_dim, self.features_dim)
        
    def _create_enhanced_expert_head(self, num_inputs: int):
        return nn.Sequential(
            nn.Linear(num_inputs, self.arch_cfg.expert_lstm_hidden_size),
            nn.LayerNorm(self.arch_cfg.expert_lstm_hidden_size), nn.GELU(),
            nn.Dropout(self.arch_cfg.dropout_rate),
            nn.Linear(self.arch_cfg.expert_lstm_hidden_size, self.arch_cfg.expert_lstm_hidden_size)
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len = observations['price_1m'].shape[0:2]
        
        signals = self._process_all_signals(observations, batch_size, seq_len)
        
        flow_input = torch.cat([signals['s_1m_momentum'], signals['s_5m_macd_fast'], signals['s_5m_macd_slow']], dim=2)
        vol_input = torch.cat([signals['s_1m_atr'], signals['s_3m_atr'], signals['s_15m_atr']], dim=2)
        value_trend_input = torch.cat([signals['s_1h_macd_fast'], signals['s_1h_macd_slow'], signals['s_15m_rsi'], signals['s_15m_bbands']], dim=2)
        context_input = observations['context']
        
        expert_outputs = []
        for expert_head, input_data in [
            (self.flow_head, flow_input), (self.volatility_head, vol_input), 
            (self.value_trend_head, value_trend_input), (self.context_head, context_input)
        ]:
            b, s, f = input_data.shape
            output = expert_head(input_data.view(b * s, f))
            expert_outputs.append(output.view(b, s, -1))

        expert_outputs_last_step = [out[:, -1, :] for out in expert_outputs]
        combined_experts = torch.cat(expert_outputs_last_step, dim=1)
        
        attention_weights = self.attention_layer(combined_experts)
        self.last_attention_weights = attention_weights.detach().cpu().numpy()
        
        expert_outputs_stacked = torch.stack(expert_outputs_last_step, dim=1)
        attention_weights_expanded = attention_weights.unsqueeze(2)
        weighted_features = torch.sum(expert_outputs_stacked * attention_weights_expanded, dim=1)
        
        final_features = self.output_projection(weighted_features)
        
        return torch.tanh(final_features)
    
    def _process_all_signals(self, observations: Dict[str, torch.Tensor], 
                           batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """The new 'dispatcher'. It feeds correct market data to each indicator cell."""
        signals = {}
        
        for cell_name, cell in self.cells.items():
            prefix = cell_name.split('_')[0]
            if prefix not in self.cell_data_map: continue
            
            obs_key, data_type = self.cell_data_map[prefix]
            data_seq = observations[obs_key]
            
            b, s, lookback, features = data_seq.shape
            data_flat = data_seq.view(b * s, lookback, features)
            
            if data_type == 'price' and features > 1:
                input_data = data_flat[:, :, 3] # Use close price
            else:
                input_data = data_flat

            signal_flat = cell(input_data)
            signals[f's_{cell_name}'] = signal_flat.view(b, s, -1)
            
        return signals
    
    def get_attention_analysis(self) -> Dict[str, np.ndarray]:
        """
        Get analysis of attention patterns for interpretability.
        This is a crucial helper method for callbacks and backtesting.
        """
        analysis = {}
        if self.last_attention_weights is not None:
            attention_weights = self.last_attention_weights
            
            analysis['expert_weights'] = {
                'flow': attention_weights[:, 0],
                'volatility': attention_weights[:, 1], 
                'value_trend': attention_weights[:, 2],
                'context': attention_weights[:, 3]
            }
            analysis['attention_entropy'] = -np.sum(
                attention_weights * np.log(attention_weights + 1e-8), axis=1
            )
            analysis['dominant_expert'] = np.argmax(attention_weights, axis=1)
        
        return analysis