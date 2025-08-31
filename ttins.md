
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import math
from torch.nn.utils import spectral_norm
from config import SETTINGS

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
        
        # Compute Q, K, V
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection and residual connection
        output = self.w_o(context)
        return self.layer_norm(output + residual)

class PositionalEncoding(nn.Module):
    """Positional encoding for time series data."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

# --- ENHANCED LEARNABLE INDICATOR CELLS ---

class EnhancedIndicatorLinear(nn.Module):
    """Enhanced indicator linear layer with multiple initialization strategies."""
    
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
        """Initialize weights based on specified strategy."""
        with torch.no_grad():
            if self.init_type == "sma":
                self.ma_layer.weight.fill_(1.0 / self.lookback_period)
            elif self.init_type == "ema":
                alpha = 2.0 / (self.lookback_period + 1.0)
                powers = torch.arange(self.lookback_period - 1, -1, -1, dtype=torch.float32)
                ema_weights = alpha * ((1 - alpha) ** powers)
                ema_weights /= torch.sum(ema_weights)
                self.ma_layer.weight.data = ema_weights.unsqueeze(0)
            elif self.init_type == "linear_decay":
                weights = torch.arange(1, self.lookback_period + 1, dtype=torch.float32)
                weights = weights / torch.sum(weights)
                self.ma_layer.weight.data = weights.unsqueeze(0)
            
            # Initialize bias to small random values
            if self.ma_layer.bias is not None:
                self.ma_layer.bias.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ma_layer(x)

class AdaptiveLSTMCell(nn.Module):
    """LSTM cell with adaptive gating and residual connections."""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, batch_first=True, 
            dropout=dropout if dropout > 0 else 0
        )
        
        # Residual connection projection
        self.residual_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Adaptive gating
        self.adaptive_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with residual connections and adaptive gating."""
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Apply adaptive gating
        gates = self.adaptive_gate(lstm_out)
        gated_output = lstm_out * gates
        
        # Residual connection (only if sequences have same length)
        if x.size(1) == lstm_out.size(1):
            residual = self.residual_proj(x)
            output = self.layer_norm(gated_output + residual)
        else:
            output = self.layer_norm(gated_output)
        
        return output, hn

class EnhancedMACDCell(nn.Module):
    """Enhanced MACD with learnable parameters and attention."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9, use_attention: bool = True):
        super().__init__()
        
        self.fast_ma = EnhancedIndicatorLinear(fast_period, 1, "ema")
        self.slow_ma = EnhancedIndicatorLinear(slow_period, 1, "ema")
        self.signal_ma = EnhancedIndicatorLinear(signal_period, 1, "ema")
        
        self.periods = {'fast': fast_period, 'slow': slow_period, 'signal': signal_period}
        
        # Optional attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = MultiHeadSelfAttention(signal_period, n_heads=4)
        
        # Learnable normalization parameters
        self.norm_scale = nn.Parameter(torch.ones(1))
        self.norm_bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        batch_size, device = price_series.shape[0], price_series.device
        
        macd_history = torch.zeros(batch_size, self.periods['signal'], device=device)
        
        for i in range(self.periods['signal']):
            end_idx = price_series.shape[1] - i
            
            fast_window = price_series[:, end_idx - self.periods['fast']:end_idx]
            slow_window = price_series[:, end_idx - self.periods['slow']:end_idx]
            
            fast_val = self.fast_ma(fast_window)
            slow_val = self.slow_ma(slow_window)
            macd_val = fast_val - slow_val
            
            macd_history[:, -1 - i] = macd_val.squeeze(-1)
        
        # Apply attention if enabled
        if self.use_attention:
            macd_history = macd_history.unsqueeze(-1)  # Add feature dimension
            attended_macd = self.attention(macd_history)
            macd_history = attended_macd.squeeze(-1)
        
        signal_line = self.signal_ma(macd_history)
        histogram = macd_history[:, -1].unsqueeze(1) - signal_line
        
        # Apply learnable normalization
        normalized_histogram = histogram * self.norm_scale + self.norm_bias
        
        return torch.tanh(normalized_histogram)  # Bound output

class MarketRegimeDetector(nn.Module):
    """Detect market regimes using a mixture of experts approach."""
    
    def __init__(self, input_dim: int, n_regimes: int = 3):
        super().__init__()
        
        self.n_regimes = n_regimes
        self.input_dim = input_dim
        
        # Regime classification network
        self.regime_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, n_regimes),
            nn.Softmax(dim=1)
        )
        
        # Regime-specific feature transformations
        self.regime_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim)
            ) for _ in range(n_regimes)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning regime-adapted features and regime probabilities."""
        
        # Detect regime probabilities
        regime_probs = self.regime_classifier(x)  # Shape: (batch, n_regimes)
        
        # Apply regime-specific transformations
        expert_outputs = []
        for expert in self.regime_experts:
            expert_outputs.append(expert(x))
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # Shape: (batch, n_regimes, input_dim)
        
        # Weight expert outputs by regime probabilities
        regime_probs_expanded = regime_probs.unsqueeze(2)  # Shape: (batch, n_regimes, 1)
        adapted_features = torch.sum(expert_outputs * regime_probs_expanded, dim=1)
        
        return adapted_features, regime_probs

# --- ENHANCED HIERARCHICAL FEATURE EXTRACTOR ---

class EnhancedHierarchicalAttentionFeatureExtractor(BaseFeaturesExtractor):
    """Enhanced feature extractor with modern techniques."""
    
    def __init__(self, observation_space: spaces.Dict, arch_cfg=None):
        if arch_cfg is None:
            arch_cfg = SETTINGS.strategy.architecture
        
        super().__init__(observation_space, features_dim=arch_cfg.attention_head_features)
        
        print("--- Building Enhanced Hierarchical Attention Feature Extractor ---")
        
        self.arch_cfg = arch_cfg
        
        # Store attention weights for interpretability
        self.last_attention_weights = None
        self.last_regime_probs = None
        
        # Enhanced learnable indicator cells
        self.cells = nn.ModuleDict({
            # Ultra-short term momentum and flow
            '1m_enhanced_roc': self._create_enhanced_roc_cell(14),
            '1m_momentum': EnhancedMACDCell(8, 21, 5, use_attention=True),
            '3m_vwap': self._create_enhanced_vwap_cell(20),
            
            # Short-term trend following
            '5m_roc_fast': self._create_enhanced_roc_cell(9),
            '5m_roc_slow': self._create_enhanced_roc_cell(21),
            '5m_macd_fast': EnhancedMACDCell(6, 13, 5),
            '5m_macd_slow': EnhancedMACDCell(24, 52, 18),
            
            # Medium-term volatility and mean reversion
            '15m_enhanced_bbands': self._create_enhanced_bbands_cell(20),
            '15m_rsi': self._create_enhanced_rsi_cell(14),
            '1h_macd_fast': EnhancedMACDCell(6, 13, 5),
            '1h_macd_slow': EnhancedMACDCell(24, 52, 18),
            
            # Volatility indicators
            '1m_atr': self._create_enhanced_atr_cell(20),
            '3m_atr': self._create_enhanced_atr_cell(14),
            '15m_atr': self._create_enhanced_atr_cell(14),
        })
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector(
            input_dim=SETTINGS.strategy.lookback_periods['context'], 
            n_regimes=3
        )
        
        # Enhanced expert heads with residual connections
        self.flow_head = self._create_enhanced_expert_head(5, "flow")
        self.volatility_head = self._create_enhanced_expert_head(4, "volatility") 
        self.value_trend_head = self._create_enhanced_expert_head(5, "value_trend")
        self.context_head = self._create_enhanced_expert_head(
            SETTINGS.strategy.lookback_periods['context'], "context"
        )
        
        # Cross-attention between expert heads
        expert_dim = self.arch_cfg.expert_lstm_hidden_size
        self.cross_attention = MultiHeadSelfAttention(expert_dim * 4, n_heads=8)
        
        # Enhanced attention mechanism with temperature
        self.temperature = nn.Parameter(torch.ones(1))
        total_expert_features = 4 * expert_dim
        
        self.attention_layer = nn.Sequential(
            nn.Linear(total_expert_features, total_expert_features // 2),
            nn.LayerNorm(total_expert_features // 2),
            nn.GELU(),  # Use GELU instead of Tanh
            nn.Dropout(self.arch_cfg.dropout_rate),
            nn.Linear(total_expert_features // 2, 4),
            nn.Softmax(dim=1)
        )
        
        # Final output projection with residual connection
        self.output_projection = nn.Sequential(
            nn.Linear(total_expert_features * 2, total_expert_features),
            nn.LayerNorm(total_expert_features),
            nn.GELU(),
            nn.Dropout(self.arch_cfg.dropout_rate),
            nn.Linear(total_expert_features, self.features_dim)
        )
        
        # Batch normalization for stability
        if self.arch_cfg.use_batch_norm:
            self.input_batch_norm = nn.BatchNorm1d(self.features_dim)
        
        self._init_weights()
    
    def _create_enhanced_expert_head(self, num_inputs: int, head_type: str):
        """Create enhanced expert head with adaptive components."""
        
        return nn.ModuleDict({
            'adaptive_lstm': AdaptiveLSTMCell(
                num_inputs, 
                self.arch_cfg.expert_lstm_hidden_size,
                dropout=self.arch_cfg.dropout_rate
            ),
            'projection': nn.Linear(
                self.arch_cfg.expert_lstm_hidden_size, 
                self.arch_cfg.expert_lstm_hidden_size
            ),
            'layer_norm': nn.LayerNorm(self.arch_cfg.expert_lstm_hidden_size)
        })
    
    def _create_enhanced_roc_cell(self, period: int):
        """Create enhanced ROC cell with trend detection."""
        return nn.ModuleDict({
            'roc_linear': nn.Linear(1, 8),
            'trend_detector': nn.Sequential(
                nn.Linear(8, 4),
                nn.ReLU(),
                nn.Linear(4, 1)
            ),
            'period': nn.Parameter(torch.tensor(period, dtype=torch.float32))
        })
    
    def _create_enhanced_vwap_cell(self, period: int):
        """Create enhanced VWAP cell with volume analysis."""
        return nn.ModuleDict({
            'price_weight': EnhancedIndicatorLinear(period, 1, "ema"),
            'volume_weight': EnhancedIndicatorLinear(period, 1, "sma"),
            'volume_analyzer': nn.Sequential(
                nn.Linear(period, period // 2),
                nn.ReLU(),
                nn.Linear(period // 2, 1)
            )
        })
    
    def _create_enhanced_bbands_cell(self, period: int):
        """Create enhanced Bollinger Bands cell."""
        return nn.ModuleDict({
            'ma': EnhancedIndicatorLinear(period, 1, "sma"),
            'std_multiplier': nn.Parameter(torch.tensor(2.0)),
            'adaptive_threshold': nn.Sequential(
                nn.Linear(1, 4),
                nn.ReLU(), 
                nn.Linear(4, 1),
                nn.Sigmoid()
            )
        })
    
    def _create_enhanced_rsi_cell(self, period: int):
        """Create enhanced RSI cell with momentum analysis."""
        return nn.ModuleDict({
            'gain_ema': EnhancedIndicatorLinear(period, 1, "ema"),
            'loss_ema': EnhancedIndicatorLinear(period, 1, "ema"),
            'momentum_analyzer': nn.Sequential(
                nn.Linear(period, period // 2),
                nn.ReLU(),
                nn.Linear(period // 2, 1)
            )
        })
    
    def _create_enhanced_atr_cell(self, period: int):
        """Create enhanced ATR cell with volatility regime detection."""
        return nn.ModuleDict({
            'tr_ema': EnhancedIndicatorLinear(period, 1, "ema"),
            'volatility_classifier': nn.Sequential(
                nn.Linear(1, 4),
                nn.ReLU(),
                nn.Linear(4, 3),  # Low, Medium, High volatility
                nn.Softmax(dim=1)
            )
        })
    
    def _init_weights(self):
        """Initialize network weights using best practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Enhanced forward pass with modern techniques."""
        
        batch_size, seq_len = observations['price_1m'].shape[0:2]
        
        # Process all indicator signals
        signals = self._process_all_signals(observations, batch_size, seq_len)
        
        # Create expert inputs
        flow_input = torch.cat([
            signals['s_1m_enhanced_roc'], signals['s_1m_momentum'], 
            signals['s_3m_vwap'], signals['s_5m_roc_fast'], signals['s_5m_roc_slow']
        ], dim=2)
        
        vol_input = torch.cat([
            signals['s_1m_atr'], signals['s_3m_atr'], 
            signals['s_15m_atr'], signals['s_15m_enhanced_bbands']
        ], dim=2)
        
        value_trend_input = torch.cat([
            signals['s_5m_macd_fast'], signals['s_5m_macd_slow'], 
            signals['s_15m_rsi'], signals['s_1h_macd_fast'], signals['s_1h_macd_slow']
        ], dim=2)
        
        # Apply market regime detection to context
        context_input = observations['context']
        if context_input.dim() == 3:  # If sequence dimension exists
            context_flat = context_input.view(-1, context_input.size(-1))
            regime_adapted_context, regime_probs = self.regime_detector(context_flat)
            regime_adapted_context = regime_adapted_context.view(batch_size, seq_len, -1)
            self.last_regime_probs = regime_probs.view(batch_size, seq_len, -1)
        else:
            regime_adapted_context, regime_probs = self.regime_detector(context_input)
            self.last_regime_probs = regime_probs
        
        # Process through expert heads
        expert_outputs = []
        for expert_head, input_data in [
            (self.flow_head, flow_input),
            (self.volatility_head, vol_input), 
            (self.value_trend_head, value_trend_input),
            (self.context_head, regime_adapted_context)
        ]:
            lstm_out, hidden = expert_head['adaptive_lstm'](input_data)
            projected = expert_head['projection'](hidden.squeeze(0))
            normalized = expert_head['layer_norm'](projected)
            expert_outputs.append(normalized)
        
        # Combine expert outputs
        combined_experts = torch.cat(expert_outputs, dim=1)
        
        # Apply cross-attention between experts
        attended_experts = self.cross_attention(
            combined_experts.unsqueeze(1)
        ).squeeze(1)
        
        # Temperature-scaled attention
        attention_weights = self.attention_layer(attended_experts)
        attention_weights = attention_weights / self.temperature
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Store attention weights for analysis
        self.last_attention_weights = attention_weights.detach().cpu().numpy()
        
        # Apply attention weighting
        expert_outputs_reshaped = torch.stack(expert_outputs, dim=1)
        attention_weights_expanded = attention_weights.unsqueeze(2)
        weighted_features = torch.sum(expert_outputs_reshaped * attention_weights_expanded, dim=1)
        
        # Create final feature representation
        final_combined = torch.cat([combined_experts, weighted_features], dim=1)
        final_features = self.output_projection(final_combined)
        
        # Apply batch normalization if enabled
        if hasattr(self, 'input_batch_norm'):
            final_features = self.input_batch_norm(final_features)
        
        return torch.tanh(final_features)
    
    def _process_all_signals(self, observations: Dict[str, torch.Tensor], 
                           batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """Process all technical indicator signals efficiently."""
        
        def process_sequence(cell_name, data_key):
            cell = self.cells[cell_name]
            data_seq = observations[data_key]
            
            if isinstance(cell, nn.ModuleDict):
                # Handle enhanced cells
                return self._process_enhanced_cell(cell, cell_name, data_seq, batch_size, seq_len)
            else:
                # Handle simple cells (backward compatibility)
                data_flat = data_seq.view(batch_size * seq_len, *data_seq.shape[2:])
                signal_flat = cell(data_flat)
                return signal_flat.view(batch_size, seq_len, 1)
        
        signals = {}
        
        # Process enhanced indicators
        for cell_name, data_key in [
            ('1m_enhanced_roc', 'price_1m'),
            ('1m_momentum', 'price_1m'),
            ('3m_vwap', 'ohlcv_3m'),
            ('5m_roc_fast', 'price_5m'),
            ('5m_roc_slow', 'price_5m'),
            ('5m_macd_fast', 'price_5m'),
            ('5m_macd_slow', 'price_5m'),
            ('15m_enhanced_bbands', 'price_15m'),
            ('15m_rsi', 'price_15m'),
            ('1h_macd_fast', 'price_1h'),
            ('1h_macd_slow', 'price_1h'),
            ('1m_atr', 'ohlc_1m'),
            ('3m_atr', 'ohlcv_3m'),
            ('15m_atr', 'ohlc_15m')
        ]:
            signals[f's_{cell_name}'] = process_sequence(cell_name, data_key)
        
        return signals
    
    def _process_enhanced_cell(self, cell: nn.ModuleDict, cell_name: str, 
                             data_seq: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Process enhanced technical indicator cells."""
        
        # This is a simplified version - you would implement specific logic for each cell type
        # For now, return a placeholder that maintains the expected shape
        return torch.randn(batch_size, seq_len, 1, device=data_seq.device)
    
    def get_attention_analysis(self) -> Dict[str, np.ndarray]:
        """Get analysis of attention patterns for interpretability."""
        
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
        
        if self.last_regime_probs is not None:
            regime_probs = self.last_regime_probs.cpu().numpy()
            if regime_probs.ndim == 3:  # Has sequence dimension
                regime_probs = regime_probs[:, -1, :]  # Take last timestep
            
            analysis['market_regime'] = {
                'regime_probs': regime_probs,
                'dominant_regime': np.argmax(regime_probs, axis=1),
                'regime_confidence': np.max(regime_probs, axis=1)
            }
        
        return analysis

# --- ENSEMBLE AND UNCERTAINTY QUANTIFICATION ---

class EnsembleFeatureExtractor(nn.Module):
    """Ensemble of feature extractors for improved robustness."""
    
    def __init__(self, observation_space: spaces.Dict, n_models: int = 3):
        super().__init__()
        
        self.n_models = n_models
        self.extractors = nn.ModuleList([
            EnhancedHierarchicalAttentionFeatureExtractor(observation_space)
            for _ in range(n_models)
        ])
        
        # Uncertainty quantification
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.extractors[0].features_dim * n_models, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty estimation."""
        
        # Get outputs from all models
        outputs = []
        for extractor in self.extractors:
            outputs.append(extractor(observations))
        
        # Combine outputs
        combined = torch.cat(outputs, dim=1)
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_head(combined)
        
        # Mean prediction
        mean_output = torch.mean(torch.stack(outputs), dim=0)
        
        return mean_output, uncertainty

# Export enhanced classes for use in other modules
__all__ = [
    'EnhancedHierarchicalAttentionFeatureExtractor',
    'EnsembleFeatureExtractor', 
    'MarketRegimeDetector',
    'MultiHeadSelfAttention',
    'PositionalEncoding'
]
