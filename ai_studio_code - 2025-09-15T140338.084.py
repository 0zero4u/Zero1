# --- START OF FILE Zero1-main/tins.py ---

# UPDATED: tins.py with Transformer Architecture

"""
FIXED VERSION - Addresses hardcoded expert count and improves architecture:
1. REMOVED hardcoded number of experts (5) from attention layer
2. Dynamically determines expert count from configuration
3. Improved expert head management with dictionary-based approach
4. Enhanced maintainability and extensibility

Enhanced Neural Network Architecture for Crypto Trading RL with Transformer Expert Heads
"""

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

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        self._init_weights()

    def _init_weights(self):
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        return output

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
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        nn.init.constant_(self.w_1.bias, 0)
        nn.init.constant_(self.w_2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class TransformerEncoderBlock(nn.Module):
    """Complete Transformer encoder block."""
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class TransformerExpertHead(nn.Module):
    """Transformer-based expert head for processing sequential indicator data."""
    def __init__(self, input_dim: int, d_model: int, n_heads: int, dim_feedforward: int,
                 num_layers: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.input_projection = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model) * 0.02)
        self._init_weights()

    def _init_weights(self):
        if isinstance(self.input_projection, nn.Linear):
            nn.init.xavier_uniform_(self.input_projection.weight)
            nn.init.constant_(self.input_projection.bias, 0)
        for layer in self.output_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            batch_size, seq_len, _ = x.shape
            x = self.input_projection(x)
            if seq_len <= self.positional_encoding.size(0):
                pos_enc = self.positional_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
                x = x + pos_enc
            else:
                logger.warning(f"Sequence length {seq_len} exceeds positional encoding size")
                pos_enc = self.positional_encoding[:seq_len % self.positional_encoding.size(0)].unsqueeze(0)
                x = x + pos_enc.expand(batch_size, -1, -1)
            for transformer_layer in self.transformer_layers:
                x = transformer_layer(x)
            x_pooled = x.mean(dim=1)
            output = self.output_projection(x_pooled)
            return output
        except Exception as e:
            logger.exception("FATAL ERROR in TransformerExpertHead forward pass. Re-raising.")
            raise e

# --- ENHANCED LEARNABLE INDICATOR CELLS (Unchanged) ---
class EnhancedIndicatorLinear(nn.Module):
    def __init__(self, lookback_period: int, output_dim: int = 1, init_type: str = "ema", use_spectral_norm: bool = False):
        super().__init__()
        self.ma_layer = nn.Linear(lookback_period, output_dim, bias=True)
        if use_spectral_norm: self.ma_layer = spectral_norm(self.ma_layer)
        self.init_type, self.lookback_period = init_type, lookback_period
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            if self.init_type == "sma": self.ma_layer.weight.fill_(1.0 / self.lookback_period)
            elif self.init_type == "ema":
                alpha = 2.0 / (self.lookback_period + 1.0)
                powers = torch.arange(self.lookback_period - 1, -1, -1, dtype=torch.float32)
                ema_weights = alpha * ((1 - alpha) ** powers)
                ema_weights /= torch.sum(ema_weights)
                self.ma_layer.weight.data = ema_weights.unsqueeze(0)
            if self.ma_layer.bias is not None: self.ma_layer.bias.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ma_layer(x)

class EnhancedMACDCell(nn.Module):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__()
        self.fast_ma = EnhancedIndicatorLinear(fast_period, 1, "ema")
        self.slow_ma = EnhancedIndicatorLinear(slow_period, 1, "ema")
        self.signal_ma = EnhancedIndicatorLinear(signal_period, 1, "ema")

    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        if price_series.dim() > 2: price_series = price_series.squeeze(-1)
        required_length = self.slow_ma.ma_layer.in_features + self.signal_ma.ma_layer.in_features - 1
        if price_series.shape[1] < required_length: return torch.zeros(price_series.shape[0], 1, device=price_series.device)
        prices_for_macd = price_series[:, -required_length:]
        slow_windows = prices_for_macd.unfold(dimension=1, size=self.slow_ma.ma_layer.in_features, step=1)
        fast_windows = slow_windows[:, :, -self.fast_ma.ma_layer.in_features:]
        slow_ema = self.slow_ma(slow_windows).squeeze(-1)
        fast_ema = self.fast_ma(fast_windows).squeeze(-1)
        macd_line = fast_ema - slow_ema
        signal_line = self.signal_ma(macd_line)
        histogram = macd_line[:, -1].unsqueeze(1) - signal_line
        return torch.tanh(histogram)

class EnhancedRSICell(nn.Module):
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
        if gains.shape[1] < self.period: return torch.zeros(price_series.shape[0], 1, device=price_series.device)
        avg_gain = self.gain_ema(gains[:, -self.period:])
        avg_loss = self.loss_ema(losses[:, -self.period:])
        rs = avg_gain / (avg_loss + self.epsilon)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return (rsi - 50.0) / 50.0

class EnhancedATRCell(nn.Module):
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
        if true_range.shape[1] < self.period: return torch.zeros(ohlc_data.shape[0], 1, device=ohlc_data.device)
        atr = self.tr_ema(true_range[:, -self.period:])
        normalized_atr = atr / (close[:, -1].unsqueeze(1) + self.epsilon)
        return torch.tanh(normalized_atr * 100)

class EnhancedBBandsCell(nn.Module):
    def __init__(self, period: int = 20):
        super().__init__()
        self.period = period
        self.std_multiplier = nn.Parameter(torch.tensor(2.0))
        self.ma = EnhancedIndicatorLinear(period, 1, "sma")
        self.epsilon = 1e-8

    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        if price_series.dim() > 2: price_series = price_series.squeeze(-1)
        if price_series.shape[1] < self.period: return torch.zeros(price_series.shape[0], 1, device=price_series.device)
        window = price_series[:, -self.period:]
        current_price = price_series[:, -1].unsqueeze(1)
        middle_band = self.ma(window)
        std_dev = torch.std(window, dim=1, keepdim=True)
        signal = (current_price - middle_band) / (self.std_multiplier * std_dev + self.epsilon)
        return torch.tanh(signal)

class ROCCell(nn.Module):
    def __init__(self, period: int = 12):
        super().__init__()
        self.period = period
        self.epsilon = 1e-8

    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        if price_series.shape[1] < self.period + 1: return torch.zeros(price_series.shape[0], 1, device=price_series.device)
        price_n_periods_ago = price_series[:, -self.period - 1]
        current_price = price_series[:, -1]
        roc = (current_price - price_n_periods_ago) / (price_n_periods_ago + self.epsilon)
        return torch.tanh(roc).unsqueeze(-1)

class PrecomputedFeatureCell(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature_series: torch.Tensor) -> torch.Tensor:
        return feature_series[:, -1].unsqueeze(-1)

CELL_CLASS_MAP = {
    'EnhancedMACDCell': EnhancedMACDCell, 'EnhancedRSICell': EnhancedRSICell,
    'EnhancedATRCell': EnhancedATRCell, 'EnhancedBBandsCell': EnhancedBBandsCell,
    'ROCCell': ROCCell, 'PrecomputedFeatureCell': PrecomputedFeatureCell,
}

# --- UPDATED: TRANSFORMER-BASED FEATURE EXTRACTOR ---

class EnhancedHierarchicalAttentionFeatureExtractor(BaseFeaturesExtractor):
    """FIXED: Enhanced feature extractor with Transformer-based expert heads."""
    def __init__(self, observation_space: spaces.Dict, arch_cfg=None):
        try:
            self.cfg = SETTINGS.strategy
            super().__init__(observation_space, features_dim=self.cfg.architecture.attention_head_features)
            logger.info("--- Building DYNAMIC Transformer-based Feature Extractor ---")
            self.last_attention_weights = None
            arch = self.cfg.architecture
            self.cells = nn.ModuleDict()
            expert_input_dims = {'flow': 0, 'volatility': 0, 'value_trend': 0, 'context': 0, 'precomputed': 0}

            for indicator_cfg in self.cfg.indicators:
                if indicator_cfg.cell_class_name not in CELL_CLASS_MAP:
                    raise ValueError(f"Unknown cell class name: {indicator_cfg.cell_class_name}")
                cell_class = CELL_CLASS_MAP[indicator_cfg.cell_class_name]
                self.cells[indicator_cfg.name] = cell_class(**indicator_cfg.params)
                expert_input_dims[indicator_cfg.expert_group] += 1
                logger.info(f" -> Created cell '{indicator_cfg.name}' ({indicator_cfg.cell_class_name}) for expert '{indicator_cfg.expert_group}'")

            expert_input_dims['context'] = self.cfg.lookback_periods[FeatureKeys.CONTEXT]
            expert_input_dims['precomputed'] = self.cfg.lookback_periods[FeatureKeys.PRECOMPUTED_FEATURES]

            self.expert_heads = nn.ModuleDict()
            self.expert_groups = []
            for expert_group, input_dim in expert_input_dims.items():
                if input_dim > 0:
                    self.expert_heads[expert_group] = TransformerExpertHead(
                        input_dim=input_dim, d_model=arch.transformer_d_model, n_heads=arch.transformer_n_heads,
                        dim_feedforward=arch.transformer_dim_feedforward, num_layers=arch.transformer_num_layers,
                        output_dim=arch.expert_output_dim, dropout=arch.dropout_rate
                    )
                    self.expert_groups.append(expert_group)
            
            logger.info(f"Created {len(self.expert_heads)} expert heads: {list(self.expert_heads.keys())}")
            num_experts = len(self.expert_heads)
            total_expert_features = num_experts * arch.expert_output_dim
            self.attention_layer = nn.Sequential(
                nn.Linear(total_expert_features, total_expert_features // 2), nn.LayerNorm(total_expert_features // 2),
                nn.GELU(), nn.Dropout(arch.dropout_rate), nn.Linear(total_expert_features // 2, num_experts),
                nn.Softmax(dim=1)
            )
            num_portfolio_features = self.cfg.lookback_periods[FeatureKeys.PORTFOLIO_STATE]
            combined_features_dim = arch.expert_output_dim + num_portfolio_features
            self.output_projection = nn.Linear(combined_features_dim, self.features_dim)
            logger.info("✅ Transformer-based feature extractor initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize Transformer feature extractor: {e}")
            raise

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        try:
            batch_size = next(iter(observations.values())).shape[0]
            expert_inputs = {'flow': [], 'volatility': [], 'value_trend': []}

            for indicator_cfg in self.cfg.indicators:
                input_data_seq = observations[indicator_cfg.input_key.value]
                b, s, lookback, *features_shape = input_data_seq.shape
                num_features = features_shape[0] if features_shape else 1
                data_flat = input_data_seq.reshape(b * s, lookback, num_features) if num_features > 0 else input_data_seq.reshape(b * s, lookback)

                if indicator_cfg.input_type == 'price':
                    input_tensor = data_flat[:, :, -1] if data_flat.dim() == 3 and data_flat.shape[2] > 1 else data_flat
                elif indicator_cfg.input_type == 'feature':
                    input_tensor = data_flat
                else:
                    input_tensor = data_flat

                cell = self.cells[indicator_cfg.name]
                signal_flat = cell(input_tensor)
                signal_seq = signal_flat.view(b, s, -1)
                expert_inputs[indicator_cfg.expert_group].append(signal_seq)

            expert_outputs, expert_names = [], []
            for expert_group in ['flow', 'volatility', 'value_trend']:
                if expert_inputs[expert_group] and expert_group in self.expert_heads:
                    expert_input = torch.cat(expert_inputs[expert_group], dim=2)
                    expert_output = self.expert_heads[expert_group](expert_input)
                    expert_outputs.append(expert_output)
                    expert_names.append(expert_group)

            if 'context' in self.expert_heads:
                context_output = self.expert_heads['context'](observations['context'])
                expert_outputs.append(context_output)
                expert_names.append('context')
            if 'precomputed' in self.expert_heads:
                precomputed_output = self.expert_heads['precomputed'](observations[FeatureKeys.PRECOMPUTED_FEATURES.value])
                expert_outputs.append(precomputed_output)
                expert_names.append('precomputed')

            if not expert_outputs:
                logger.warning("No expert outputs available!")
                return torch.zeros(batch_size, self.features_dim, device=list(observations.values())[0].device)

            combined_experts = torch.cat(expert_outputs, dim=1)
            attention_weights = self.attention_layer(combined_experts)
            self.last_attention_weights = attention_weights.detach().cpu().numpy()
            expert_outputs_stacked = torch.stack(expert_outputs, dim=1)
            weighted_market_features = torch.sum(expert_outputs_stacked * attention_weights.unsqueeze(2), dim=1)

            portfolio_state_seq = observations[FeatureKeys.PORTFOLIO_STATE.value]
            latest_portfolio_state = portfolio_state_seq[:, -1, :]
            combined_features = torch.cat([weighted_market_features, latest_portfolio_state], dim=1)
            final_features = self.output_projection(combined_features)
            return torch.tanh(final_features)
        except Exception as e:
            obs_shapes = {k: v.shape for k, v in observations.items()}
            logger.exception(f"FATAL ERROR in main feature extractor forward pass. Re-raising.\nObservation shapes: {obs_shapes}")
            raise e

    def get_attention_analysis(self) -> Dict[str, np.ndarray]:
        analysis = {}
        if self.last_attention_weights is not None:
            weights = self.last_attention_weights
            analysis['expert_weights'] = {name: weights[:, i] for i, name in enumerate(self.expert_groups) if i < weights.shape[1]}
            analysis['attention_entropy'] = -np.sum(weights * np.log(weights + 1e-8), axis=1)
            analysis['dominant_expert'] = np.argmax(weights, axis=1)
        return analysis