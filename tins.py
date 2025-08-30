import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .config import SETTINGS

# --- Core Building Blocks & Learnable Cells (Unchanged) ---
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
    # ... (code unchanged) ...
class LearnableROCCell(nn.Module):
    # ... (code unchanged) ...
class LearnableATRCell(nn.Module):
    # ... (code unchanged) ...

class LearnableVWAPCell(nn.Module):
    def __init__(self, vwap_period=20):
        super(LearnableVWAPCell, self).__init__()
        self.vwap_period = vwap_period
        self.num_ma = IndicatorLinear(vwap_period); self.den_ma = IndicatorLinear(vwap_period)
    def forward(self, ohlcv_series: torch.Tensor) -> torch.Tensor:
        window = ohlcv_series[:, -self.vwap_period:]
        highs, lows, closes, volumes = window[:, :, 1], window[:, :, 2], window[:, :, 3], window[:, :, 4]
        typical_price = (highs + lows + closes) / 3.0; tpv = typical_price * volumes
        vwap = self.num_ma(tpv) / (self.den_ma(volumes) + 1e-8)
        current_close = closes[:, -1].unsqueeze(-1)
        return (current_close - vwap) / (vwap + 1e-8)

class LearnableBBandsCell(nn.Module):
    # ... (code unchanged) ...
class LearnableRSICell(nn.Module):
    # ... (code unchanged) ...


### --- REFINEMENT --- ###
# This new feature extractor implements the requested hierarchical attention architecture.

class HierarchicalAttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    Implements the "Chief Strategist" model using specialized expert heads and an attention
    mechanism to weigh their importance.

    1.  Expert Heads: Groups of indicators (e.g., for Flow, Volatility) each process
        their relevant data sequences through a dedicated LSTM. This is the "Mid-Level Analysis".
    2.  Attention Mechanism: A "Chief Strategist" layer that takes the outputs (the "opinions")
        of all expert heads and learns to assign weights to them, creating a context-aware
        final feature vector. The attention weights are the "Composition Score".
    """
    def __init__(self, observation_space: spaces.Dict, arch_cfg=SETTINGS.strategy.architecture):
        # The final feature dimension is determined by the attention mechanism's output size
        super().__init__(observation_space, features_dim=arch_cfg.ATTENTION_HEAD_FEATURES)
        print("--- Building Hierarchical Attention Feature Extractor for SB3 ---")
        self.arch_cfg = arch_cfg

        # --- 1. Instantiate ALL Indicator Cells ---
        self.cells = nn.ModuleDict({
            # Flow & Momentum Indicators
            '1m_roc': LearnableROCCell(roc_period=14),
            '1m_momentum': LearnableMACDCell(fast_period=8, slow_period=21, signal_period=5),
            '3m_vwap': LearnableVWAPCell(vwap_period=20),
            '5m_roc_fast': LearnableROCCell(roc_period=9),
            '5m_roc_slow': LearnableROCCell(roc_period=21),
            # Volatility Indicators
            '1m_atr': LearnableATRCell(atr_period=20),
            '3m_atr': LearnableATRCell(atr_period=14),
            '15m_atr': LearnableATRCell(),
            '15m_bbands': LearnableBBandsCell(),
            # Value & Trend Indicators
            '5m_macd_fast': LearnableMACDCell(fast_period=6, slow_period=13, signal_period=5),
            '5m_macd_slow': LearnableMACDCell(fast_period=24, slow_period=52, signal_period=18),
            '15m_rsi': LearnableRSICell(),
            '1h_macd_fast': LearnableMACDCell(fast_period=6, slow_period=13, signal_period=5),
            '1h_macd_slow': LearnableMACDCell(fast_period=24, slow_period=52, signal_period=18),
        })

        # --- 2. Define "Expert Heads" with their own LSTMs ---
        # Each head processes a sequence of signals relevant to its expertise.
        self.flow_head = self._create_expert_head(num_inputs=5)
        self.volatility_head = self._create_expert_head(num_inputs=4)
        self.value_trend_head = self._create_expert_head(num_inputs=5)
        self.context_head = self._create_expert_head(num_inputs=SETTINGS.strategy.LOOKBACK_PERIODS['context'])

        # --- 3. Define the "Chief Strategist" Attention Mechanism ---
        num_experts = 4 # Flow, Volatility, Value/Trend, Context
        total_expert_features = num_experts * self.arch_cfg.EXPERT_LSTM_HIDDEN_SIZE
        self.attention_layer = nn.Sequential(
            nn.Linear(total_expert_features, total_expert_features // 2),
            nn.Tanh(),
            nn.Linear(total_expert_features // 2, num_experts),
            nn.Softmax(dim=1)
        )
        # This layer produces the final feature vector for the PPO actor/critic
        self.output_projection = nn.Linear(total_expert_features, self.features_dim)

    def _create_expert_head(self, num_inputs: int):
        return nn.LSTM(
            input_size=num_inputs,
            hidden_size=self.arch_cfg.EXPERT_LSTM_HIDDEN_SIZE,
            num_layers=self.arch_cfg.LSTM_LAYERS,
            batch_first=True
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len = observations['price_1m'].shape[0:2]

        def process_sequence(cell, data_key):
            data_seq = observations[data_key]
            data_flat = data_seq.view(batch_size * seq_len, *data_seq.shape[2:])
            signal_flat = cell(data_flat)
            return signal_flat.view(batch_size, seq_len, 1)

        # --- Get signal sequences from all cells ---
        signals = {
            's_1m_roc': process_sequence(self.cells['1m_roc'], 'price_1m'),
            's_1m_momentum': process_sequence(self.cells['1m_momentum'], 'price_1m'),
            's_3m_vwap': process_sequence(self.cells['3m_vwap'], 'ohlcv_3m'),
            's_5m_roc_fast': process_sequence(self.cells['5m_roc_fast'], 'price_5m'),
            's_5m_roc_slow': process_sequence(self.cells['5m_roc_slow'], 'price_5m'),
            's_1m_atr': process_sequence(self.cells['1m_atr'], 'ohlc_1m'),
            's_3m_atr': process_sequence(self.cells['3m_atr'], 'ohlcv_3m'),
            's_15m_atr': process_sequence(self.cells['15m_atr'], 'ohlc_15m'),
            's_15m_bbands': process_sequence(self.cells['15m_bbands'], 'price_15m'),
            's_5m_macd_fast': process_sequence(self.cells['5m_macd_fast'], 'price_5m'),
            's_5m_macd_slow': process_sequence(self.cells['5m_macd_slow'], 'price_5m'),
            's_15m_rsi': process_sequence(self.cells['15m_rsi'], 'price_15m'),
            's_1h_macd_fast': process_sequence(self.cells['1h_macd_fast'], 'price_1h'),
            's_1h_macd_slow': process_sequence(self.cells['1h_macd_slow'], 'price_1h'),
        }

        # --- Group signals for each Expert Head ---
        flow_input = torch.cat([signals['s_1m_roc'], signals['s_1m_momentum'], signals['s_3m_vwap'], signals['s_5m_roc_fast'], signals['s_5m_roc_slow']], dim=2)
        vol_input = torch.cat([signals['s_1m_atr'], signals['s_3m_atr'], signals['s_15m_atr'], signals['s_15m_bbands']], dim=2)
        value_trend_input = torch.cat([signals['s_5m_macd_fast'], signals['s_5m_macd_slow'], signals['s_15m_rsi'], signals['s_1h_macd_fast'], signals['s_1h_macd_slow']], dim=2)
        context_input = observations['context']

        # --- Process sequences through Expert LSTMs ---
        _, (flow_hn, _) = self.flow_head(flow_input)
        _, (vol_hn, _) = self.volatility_head(vol_input)
        _, (value_trend_hn, _) = self.value_trend_head(value_trend_input)
        _, (context_hn, _) = self.context_head(context_input)

        # Get the output from the last layer of each LSTM head
        expert_outputs = torch.cat([
            flow_hn[-1], vol_hn[-1], value_trend_hn[-1], context_hn[-1]
        ], dim=1)

        # --- Apply Attention: The "Chief Strategist" Decision ---
        # attention_weights would be the "Composition_Score_Route" from your JSON
        attention_weights = self.attention_layer(expert_outputs) # Shape: (batch_size, num_experts)

        # Reshape for weighted sum
        expert_outputs_reshaped = expert_outputs.view(batch_size, 4, self.arch_cfg.EXPERT_LSTM_HIDDEN_SIZE)
        attention_weights_reshaped = attention_weights.unsqueeze(2) # Shape: (batch_size, num_experts, 1)

        # The context vector is a weighted sum of the expert opinions
        weighted_expert_features = torch.sum(expert_outputs_reshaped * attention_weights_reshaped, dim=1)

        # Combine the weighted features with the raw expert outputs for a richer representation
        final_combined_features = torch.cat([expert_outputs, weighted_expert_features], dim=1)

        # Project to the final desired feature dimension for the policy
        final_features = self.output_projection(final_combined_features)
        return F.tanh(final_features)
