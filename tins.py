import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import math
from torch.nn.utils import spectral_norm
import logging

# Import configuration - updated to use new Transformer parameters
from config import SETTINGS, FeatureKeys

logger = logging.getLogger(__name__)

# --- TRANSFORMER ARCHITECTURE COMPONENTS ---

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism with proper scaling and normalization."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling for Transformer."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            batch_size, seq_len, _ = x.size()

            # Linear projections
            q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

            # Scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            context = torch.matmul(attention_weights, v)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

            # Output projection
            output = self.w_o(context)

            return output

        except Exception as e:
            logger.error(f"Error in multi-head attention: {e}")
            return x  # Return input as fallback

class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, dim_feedforward)
        self.w_2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        nn.init.constant_(self.w_1.bias, 0)
        nn.init.constant_(self.w_2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class TransformerEncoderBlock(nn.Module):
    """
    Complete Transformer encoder block replacing the old AttentionEnhancedExpertHead.
    This block contains:
    1. Multi-head self-attention
    2. Add & norm (residual connection + layer normalization)
    3. Position-wise feed-forward network
    4. Add & norm (residual connection + layer normalization)
    """

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            # Self-attention with residual connection and layer norm
            attn_output = self.self_attention(x, mask)
            x = self.norm1(x + self.dropout1(attn_output))

            # Feed-forward with residual connection and layer norm
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout2(ff_output))

            return x

        except Exception as e:
            logger.error(f"Error in Transformer encoder block: {e}")
            # Return zeros as fallback
            return torch.zeros_like(x)

class TransformerExpertHead(nn.Module):
    """
    Transformer-based expert head for processing sequential indicator data.
    Replaces the old AttentionEnhancedExpertHead with a proper Transformer architecture.
    """

    def __init__(self, input_dim: int, d_model: int, n_heads: int, dim_feedforward: int,
                 num_layers: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim

        # Input projection to match d_model
        self.input_projection = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.LayerNorm(output_dim)
        )

        # Positional encoding (learnable)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model) * 0.02)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights properly for Transformer."""
        if isinstance(self.input_projection, nn.Linear):
            nn.init.xavier_uniform_(self.input_projection.weight)
            nn.init.constant_(self.input_projection.bias, 0)

        for layer in self.output_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            # x shape: (batch_size, sequence_length, input_dim)
            batch_size, seq_len, _ = x.shape

            # Project to d_model if necessary
            x = self.input_projection(x)

            # Add positional encoding
            if seq_len <= self.positional_encoding.size(0):
                pos_enc = self.positional_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
                x = x + pos_enc
            else:
                logger.warning(f"Sequence length {seq_len} exceeds positional encoding size")
                # Use repeated positional encodings
                pos_enc = self.positional_encoding[:seq_len % self.positional_encoding.size(0)].unsqueeze(0)
                x = x + pos_enc.expand(batch_size, -1, -1)

            # Pass through Transformer layers
            for transformer_layer in self.transformer_layers:
                x = transformer_layer(x)

            # Global average pooling over sequence dimension
            x_pooled = x.mean(dim=1)  # (batch_size, d_model)

            # Output projection
            output = self.output_projection(x_pooled)

            return output

        except Exception as e:
            logger.error(f"Error in Transformer expert head forward: {e}")
            # Return zeros as fallback
            return torch.zeros(x.shape[0], self.output_dim, device=x.device)

# --- ENHANCED LEARNABLE INDICATOR CELLS (Unchanged) ---

class EnhancedIndicatorLinear(nn.Module):
    """Core building block: A linear layer initialized to function as a classic moving average."""

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
        """Initialize weights based on traditional moving average patterns."""
        try:
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

        except Exception as e:
            logger.error(f"Error initializing weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.ma_layer(x)
        except Exception as e:
            logger.error(f"Error in indicator linear forward: {e}")
            return torch.zeros(x.shape[0], self.ma_layer.out_features, device=x.device)

class EnhancedMACDCell(nn.Module):
    """Implements the full MACD calculation using learnable EMA layers."""

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

        self.fast_ma = EnhancedIndicatorLinear(fast_period, 1, "ema")
        self.slow_ma = EnhancedIndicatorLinear(slow_period, 1, "ema")
        self.signal_ma = EnhancedIndicatorLinear(signal_period, 1, "ema")

    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        try:
            if price_series.dim() > 2:
                price_series = price_series.squeeze(-1)

            # Ensure we have enough data points for a full calculation
            required_length = self.slow_period + self.signal_period - 1
            if price_series.shape[1] < required_length:
                return torch.zeros(price_series.shape[0], 1, device=price_series.device)

            # Create sliding windows for the slow EMA calculation
            prices_for_macd = price_series[:, -required_length:]
            slow_windows = prices_for_macd.unfold(dimension=1, size=self.slow_period, step=1)

            # Slice fast windows from the end of the slow windows
            fast_windows = slow_windows[:, :, -self.fast_period:]

            # Apply the MA layers in a vectorized way
            slow_ema = self.slow_ma(slow_windows).squeeze(-1)
            fast_ema = self.fast_ma(fast_windows).squeeze(-1)

            # Calculate the MACD line for the last signal_period steps
            macd_line = fast_ema - slow_ema

            # Calculate the signal line using the MACD line history
            signal_line = self.signal_ma(macd_line)

            # Calculate the histogram from the most recent MACD value and the signal line
            histogram = macd_line[:, -1].unsqueeze(1) - signal_line

            return torch.tanh(histogram)

        except Exception as e:
            logger.error(f"Error in MACD calculation: {e}")
            return torch.zeros(price_series.shape[0], 1, device=price_series.device)

class EnhancedRSICell(nn.Module):
    """Implements the RSI calculation with learnable EMA components."""

    def __init__(self, period: int = 14):
        super().__init__()
        self.period = period
        self.gain_ema = EnhancedIndicatorLinear(period, 1, "ema")
        self.loss_ema = EnhancedIndicatorLinear(period, 1, "ema")
        self.epsilon = 1e-8

    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        try:
            if price_series.dim() > 2:
                price_series = price_series.squeeze(-1)

            deltas = price_series[:, 1:] - price_series[:, :-1]
            gains = torch.clamp(deltas, min=0)
            losses = -torch.clamp(deltas, max=0)

            if gains.shape[1] < self.period:
                return torch.zeros(price_series.shape[0], 1, device=price_series.device)

            avg_gain = self.gain_ema(gains[:, -self.period:])
            avg_loss = self.loss_ema(losses[:, -self.period:])

            rs = avg_gain / (avg_loss + self.epsilon)
            rsi = 100.0 - (100.0 / (1.0 + rs))

            return (rsi - 50.0) / 50.0  # Normalize to [-1, 1]

        except Exception as e:
            logger.error(f"Error in RSI calculation: {e}")
            return torch.zeros(price_series.shape[0], 1, device=price_series.device)

class EnhancedATRCell(nn.Module):
    """Implements the ATR (Average True Range) calculation."""

    def __init__(self, period: int = 14):
        super().__init__()
        self.period = period
        self.tr_ema = EnhancedIndicatorLinear(period, 1, "ema")
        self.epsilon = 1e-8

    def forward(self, ohlc_data: torch.Tensor) -> torch.Tensor:
        try:
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

        except Exception as e:
            logger.error(f"Error in ATR calculation: {e}")
            return torch.zeros(ohlc_data.shape[0], 1, device=ohlc_data.device)

class EnhancedBBandsCell(nn.Module):
    """Implements a Bollinger Bands signal."""

    def __init__(self, period: int = 20):
        super().__init__()
        self.period = period
        self.std_multiplier = nn.Parameter(torch.tensor(2.0))
        self.ma = EnhancedIndicatorLinear(period, 1, "sma")
        self.epsilon = 1e-8

    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        try:
            if price_series.dim() > 2:
                price_series = price_series.squeeze(-1)

            if price_series.shape[1] < self.period:
                return torch.zeros(price_series.shape[0], 1, device=price_series.device)

            window = price_series[:, -self.period:]
            current_price = price_series[:, -1].unsqueeze(1)

            middle_band = self.ma(window)
            std_dev = torch.std(window, dim=1, keepdim=True)

            signal = (current_price - middle_band) / (self.std_multiplier * std_dev + self.epsilon)

            return torch.tanh(signal)

        except Exception as e:
            logger.error(f"Error in Bollinger Bands calculation: {e}")
            return torch.zeros(price_series.shape[0], 1, device=price_series.device)

class ROCCell(nn.Module):
    """Calculates Rate of Change (ROC) for momentum."""

    def __init__(self, period: int = 12):
        super().__init__()
        self.period = period
        self.epsilon = 1e-8

    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        try:
            if price_series.shape[1] < self.period + 1:
                return torch.zeros(price_series.shape[0], 1, device=price_series.device)

            price_n_periods_ago = price_series[:, -self.period - 1]
            current_price = price_series[:, -1]

            roc = (current_price - price_n_periods_ago) / (price_n_periods_ago + self.epsilon)

            return torch.tanh(roc).unsqueeze(-1)

        except Exception as e:
            logger.error(f"Error in ROC calculation: {e}")
            return torch.zeros(price_series.shape[0], 1, device=price_series.device)

class PrecomputedFeatureCell(nn.Module):
    """Extracts the latest value from a pre-computed feature stream (e.g., Volume Delta)."""

    def __init__(self):
        super().__init__()

    def forward(self, feature_series: torch.Tensor) -> torch.Tensor:
        try:
            # feature_series shape: (batch, lookback)
            # We want the latest value from the series
            return feature_series[:, -1].unsqueeze(-1)

        except Exception as e:
            logger.error(f"Error in PrecomputedFeatureCell: {e}")
            # Fallback for shape (batch, lookback, features)
            if feature_series.dim() == 3:
                return torch.zeros(feature_series.shape[0], feature_series.shape[2], device=feature_series.device)
            else:
                return torch.zeros(feature_series.shape[0], 1, device=feature_series.device)

# --- CELL CLASS MAPPING ---

CELL_CLASS_MAP = {
    'EnhancedMACDCell': EnhancedMACDCell,
    'EnhancedRSICell': EnhancedRSICell,
    'EnhancedATRCell': EnhancedATRCell,
    'EnhancedBBandsCell': EnhancedBBandsCell,
    'ROCCell': ROCCell,
    'PrecomputedFeatureCell': PrecomputedFeatureCell,
}

# --- UPDATED: TRANSFORMER-BASED FEATURE EXTRACTOR ---

