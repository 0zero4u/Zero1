"""
Enhanced Neural Network Architecture for Crypto Trading RL
Hierarchical attention networks with learnable technical indicators.
This version is dynamically configured based on StrategyConfig for improved modularity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import math
from torch.nn.utils import spectral_norm
import logging

# Import configuration - fixed import path
from config import SETTINGS, FeatureKeys

logger = logging.getLogger(__name__)

# --- ADVANCED ATTENTION MECHANISMS ---

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism for enhanced pattern recognition."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
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
        try:
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

        except Exception as e:
            logger.error(f"Error in multi-head attention: {e}")
            return x  # Return input as fallback

# --- ENHANCED LEARNABLE INDICATOR CELLS ---

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

# --- NEW: CELLS FOR NEW FEATURES ---

class ROCCell(nn.Module):
    """Calculates Rate of Change (ROC) for momentum."""
    def __init__(self, period: int = 12):
        super().__init__()
        self.period = period
        self.epsilon = 1e-8

    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        try:
            if price_series.shape[1] <= self.period:
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

# --- ENHANCED EXPERT HEAD WITH ATTENTION ---

class AttentionEnhancedExpertHead(nn.Module):
    """Expert head with self-attention for sequence processing."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        # Self-attention layer
        self.self_attention = MultiHeadSelfAttention(input_dim, n_heads=max(1, input_dim // 2), dropout=dropout)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            # x shape: (batch_size, sequence_length, input_dim)
            # Apply self-attention across the sequence
            attended_x = self.self_attention(x)

            # Take the last timestep after attention
            last_timestep = attended_x[:, -1, :]  # (batch_size, input_dim)

            # Apply MLP
            output = self.mlp(last_timestep)
            return output

        except Exception as e:
            logger.error(f"Error in expert head forward: {e}")
            # Return zeros as fallback
            return torch.zeros(x.shape[0], self.mlp[-1].out_features, device=x.device)

# --- DYNAMICALLY CONFIGURED HIERARCHICAL FEATURE EXTRACTOR ---

CELL_CLASS_MAP = {
    'EnhancedMACDCell': EnhancedMACDCell,
    'EnhancedRSICell': EnhancedRSICell,
    'EnhancedATRCell': EnhancedATRCell,
    'EnhancedBBandsCell': EnhancedBBandsCell,
    'ROCCell': ROCCell,
    'PrecomputedFeatureCell': PrecomputedFeatureCell,
}

class EnhancedHierarchicalAttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    Enhanced feature extractor that dynamically builds its architecture based on a
    declarative configuration from config.py.
    UPDATED: Now incorporates the agent's portfolio state and precomputed features.
    """

    def __init__(self, observation_space: spaces.Dict, arch_cfg=None):
        try:
            # Use the global SETTINGS if no specific config is provided
            self.cfg = SETTINGS.strategy
            
            # The final feature dimension is determined by the attention head
            super().__init__(observation_space, features_dim=self.cfg.architecture.attention_head_features)

            logger.info("--- Building DYNAMIC Hierarchical Attention Feature Extractor ---")
            
            self.last_attention_weights = None
            
            # --- Dynamic Cell Creation ---
            self.cells = nn.ModuleDict()
            # UPDATED: Add 'precomputed' expert group
            expert_input_dims = {'flow': 0, 'volatility': 0, 'value_trend': 0, 'context': 0, 'precomputed': 0}
            
            for indicator_cfg in self.cfg.indicators:
                if indicator_cfg.cell_class_name not in CELL_CLASS_MAP:
                    raise ValueError(f"Unknown cell class name: {indicator_cfg.cell_class_name}")
                
                # Instantiate the cell
                cell_class = CELL_CLASS_MAP[indicator_cfg.cell_class_name]
                self.cells[indicator_cfg.name] = cell_class(**indicator_cfg.params)
                
                # Increment the input dimension for the corresponding expert group
                expert_input_dims[indicator_cfg.expert_group] += 1
                logger.info(f"  -> Created cell '{indicator_cfg.name}' ({indicator_cfg.cell_class_name}) for expert '{indicator_cfg.expert_group}'")

            # Set dimensions for vector-based features
            expert_input_dims['context'] = self.cfg.lookback_periods[FeatureKeys.CONTEXT]
            expert_input_dims['precomputed'] = self.cfg.lookback_periods[FeatureKeys.PRECOMPUTED_FEATURES]

            # --- Dynamic Expert Head Creation ---
            expert_hidden_size = self.cfg.architecture.expert_lstm_hidden_size
            self.flow_head = AttentionEnhancedExpertHead(expert_input_dims['flow'], expert_hidden_size)
            self.volatility_head = AttentionEnhancedExpertHead(expert_input_dims['volatility'], expert_hidden_size)
            self.value_trend_head = AttentionEnhancedExpertHead(expert_input_dims['value_trend'], expert_hidden_size)
            self.context_head = AttentionEnhancedExpertHead(expert_input_dims['context'], expert_hidden_size)
            # NEW: Create the expert head for precomputed features
            self.precomputed_head = AttentionEnhancedExpertHead(expert_input_dims['precomputed'], expert_hidden_size)
            
            logger.info(f"Expert head input dimensions: {expert_input_dims}")

            # --- UPDATED: Attention Layer (for 5 experts) ---
            total_expert_features = 5 * expert_hidden_size
            self.attention_layer = nn.Sequential(
                nn.Linear(total_expert_features, total_expert_features // 2),
                nn.LayerNorm(total_expert_features // 2),
                nn.GELU(),
                nn.Dropout(self.cfg.architecture.dropout_rate),
                nn.Linear(total_expert_features // 2, 5), # Output 5 weights
                nn.Softmax(dim=1)
            )

            # --- UPDATED: Final Projection Layer ---
            # The final feature vector will be a combination of the market analysis (from experts)
            # and the agent's own portfolio state.
            num_portfolio_features = self.cfg.lookback_periods[FeatureKeys.PORTFOLIO_STATE]
            combined_features_dim = expert_hidden_size + num_portfolio_features

            self.output_projection = nn.Linear(combined_features_dim, self.features_dim)
            
            logger.info(f"Final projection layer input dim: {combined_features_dim} (Expert: {expert_hidden_size} + Portfolio: {num_portfolio_features})")
            logger.info("✅ Feature extractor initialized successfully from declarative config.")

        except Exception as e:
            logger.error(f"Failed to initialize feature extractor: {e}")
            raise

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the dynamically built hierarchical attention network."""
        try:
            batch_size = next(iter(observations.values())).shape[0]

            # --- Market Analysis Section ---
            expert_inputs = {
                'flow': [], 'volatility': [], 'value_trend': []
            }

            for indicator_cfg in self.cfg.indicators:
                # Get the correct input data from observations
                input_data_seq = observations[indicator_cfg.input_key.value]
                
                # Reshape for processing: (batch, seq, lookback, feats) -> (batch * seq, lookback, feats)
                b, s, lookback, *features_shape = input_data_seq.shape
                num_features = features_shape[0] if features_shape else 1
                data_flat = input_data_seq.reshape(b * s, lookback, num_features) if num_features > 0 else input_data_seq.reshape(b * s, lookback)

                # Prepare the final input tensor based on the required type
                if indicator_cfg.input_type == 'price':
                    # Use close price from OHLC or the price series itself
                    input_tensor = data_flat[:, :, -1] if data_flat.dim() == 3 and data_flat.shape[2] > 1 else data_flat
                elif indicator_cfg.input_type == 'feature':
                    # For precomputed features like volume delta
                    input_tensor = data_flat
                else: # 'ohlc'
                    input_tensor = data_flat
                
                # Get the corresponding cell and process the signal
                cell = self.cells[indicator_cfg.name]
                signal_flat = cell(input_tensor)
                
                # Reshape signal back and append to the correct expert group
                signal_seq = signal_flat.view(b, s, -1)
                expert_inputs[indicator_cfg.expert_group].append(signal_seq)

            # Concatenate signals for each expert
            flow_input = torch.cat(expert_inputs['flow'], dim=2)
            vol_input = torch.cat(expert_inputs['volatility'], dim=2)
            value_trend_input = torch.cat(expert_inputs['value_trend'], dim=2)
            
            # Context and Precomputed inputs are directly from observations
            context_input = observations['context']
            precomputed_input = observations[FeatureKeys.PRECOMPUTED_FEATURES.value]


            # Process through attention-enhanced expert heads
            flow_output = self.flow_head(flow_input)
            vol_output = self.volatility_head(vol_input)
            value_trend_output = self.value_trend_head(value_trend_input)
            context_output = self.context_head(context_input)
            # NEW: Process precomputed features
            precomputed_output = self.precomputed_head(precomputed_input)


            # Combine expert outputs for attention calculation
            expert_outputs = [flow_output, vol_output, value_trend_output, context_output, precomputed_output]
            combined_experts = torch.cat(expert_outputs, dim=1)

            # Calculate and apply attention weights
            attention_weights = self.attention_layer(combined_experts)
            self.last_attention_weights = attention_weights.detach().cpu().numpy()

            expert_outputs_stacked = torch.stack(expert_outputs, dim=1)
            attention_weights_expanded = attention_weights.unsqueeze(2)
            weighted_market_features = torch.sum(expert_outputs_stacked * attention_weights_expanded, dim=1)

            # --- NEW: Incorporate Agent State ---
            # Get the latest portfolio state from the observation sequence
            portfolio_state_seq = observations[FeatureKeys.PORTFOLIO_STATE.value]
            latest_portfolio_state = portfolio_state_seq[:, -1, :] # Shape: (batch_size, num_portfolio_features)
            
            # Combine the market analysis with the agent's state
            combined_features = torch.cat([weighted_market_features, latest_portfolio_state], dim=1)

            # --- Final Projection ---
            final_features = self.output_projection(combined_features)
            return torch.tanh(final_features)

        except Exception as e:
            logger.error(f"Error in feature extractor forward pass: {e}")
            # Return zero features as a safe fallback
            return torch.zeros(batch_size, self.features_dim, device=list(observations.values())[0].device)

    def get_attention_analysis(self) -> Dict[str, np.ndarray]:
        """Get analysis of attention patterns for interpretability."""
        analysis = {}

        try:
            if self.last_attention_weights is not None:
                attention_weights = self.last_attention_weights

                analysis['expert_weights'] = {
                    'flow': attention_weights[:, 0],
                    'volatility': attention_weights[:, 1],
                    'value_trend': attention_weights[:, 2],
                    'context': attention_weights[:, 3],
                    'precomputed': attention_weights[:, 4]
                }

                analysis['attention_entropy'] = -np.sum(
                    attention_weights * np.log(attention_weights + 1e-8), axis=1
                )

                analysis['dominant_expert'] = np.argmax(attention_weights, axis=1)

        except Exception as e:
            logger.error(f"Error in attention analysis: {e}")

        return analysis

if __name__ == "__main__":
    # Example usage and testing
    try:
        logger.info("Testing neural network architecture...")

        # Create a dummy observation space that matches the default config
        from config import SETTINGS
        s_cfg = SETTINGS.strategy
        seq_len = s_cfg.sequence_length

        dummy_obs_space_dict = {}
        for key, lookback in s_cfg.lookback_periods.items():
            key_str = key.value
            if key_str.startswith('ohlcv_'):
                shape = (seq_len, lookback, 5)
            elif key_str.startswith('ohlc_'):
                shape = (seq_len, lookback, 4)
            else: # price, volume_delta, context, portfolio_state, precomputed_features
                shape = (seq_len, lookback)
            
            dummy_obs_space_dict[key_str] = spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)

        dummy_obs_space = spaces.Dict(dummy_obs_space_dict)

        # Create feature extractor
        extractor = EnhancedHierarchicalAttentionFeatureExtractor(dummy_obs_space)
        print(extractor)
        
        # Create a dummy observation
        dummy_obs = dummy_obs_space.sample()
        for key in dummy_obs:
            dummy_obs[key] = torch.from_numpy(dummy_obs[key]).unsqueeze(0) # Add batch dim

        # Test forward pass
        features = extractor(dummy_obs)
        print(f"\nOutput feature shape: {features.shape}")
        assert features.shape == (1, s_cfg.architecture.attention_head_features)
        
        # Test attention analysis
        analysis = extractor.get_attention_analysis()
        print(f"Attention analysis: {analysis}")
        assert 'expert_weights' in analysis
        
        logger.info("✅ Neural network architecture test completed successfully!")

    except Exception as e:
        logger.error(f"Neural network test failed: {e}", exc_info=True)
