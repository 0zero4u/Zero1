# rl-main/tins.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import SETTINGS

class LearnableMACD_TIN(nn.Module):
    """
    A specialist TIN that learns an MACD-like strategy for a specific timeframe.
    It uses 1D convolutions to create learnable weighted moving averages.
    The output is a single value representing the trading signal strength.
    """
    def __init__(self, lookback_period: int, fast_ma_fraction=0.25, slow_ma_fraction=0.5, signal_ma_fraction=0.18):
        super(LearnableMACD_TIN, self).__init__()
        
        # Define MA periods as a fraction of the total lookback window
        fast_period = max(2, int(lookback_period * fast_ma_fraction))
        slow_period = max(fast_period + 1, int(lookback_period * slow_ma_fraction))
        signal_period = max(2, int(fast_period * signal_ma_fraction))

        # Padding ensures the output sequence has a reasonable length
        self.slow_ma = nn.Conv1d(1, 1, kernel_size=slow_period, padding='valid', bias=False)
        self.fast_ma = nn.Conv1d(1, 1, kernel_size=fast_period, padding='valid', bias=False)
        self.signal_ma = nn.Conv1d(1, 1, kernel_size=signal_period, padding='valid', bias=False)

        # Initialize with sensible SMA weights to give the model a good starting point
        nn.init.constant_(self.slow_ma.weight, 1.0 / slow_period)
        nn.init.constant_(self.fast_ma.weight, 1.0 / fast_period)
        nn.init.constant_(self.signal_ma.weight, 1.0 / signal_period)

        # A small head to process the final histogram into one signal value
        self.head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, lookback_period) -> add channel for conv
        x = x.unsqueeze(1)

        slow_line = self.slow_ma(x)
        fast_line = self.fast_ma(x)
        
        # Align sequences by padding the shorter one (fast_line)
        padding_size = slow_line.shape[-1] - fast_line.shape[-1]
        fast_line_padded = F.pad(fast_line, (padding_size, 0))

        macd_line = fast_line_padded - slow_line
        signal_line = self.signal_ma(macd_line)
        
        padding_size_signal = macd_line.shape[-1] - signal_line.shape[-1]
        signal_line_padded = F.pad(signal_line, (padding_size_signal, 0))

        histogram = macd_line - signal_line_padded

        # Use the most recent histogram value as the primary feature
        latest_histogram_value = histogram[:, :, -1]
        
        # tanh ensures the signal is bounded between -1 and 1
        return torch.tanh(self.head(latest_histogram_value))

class HierarchicalTIN(nn.Module):
    """
    The main "CEO" model. It combines signals from multiple specialist TINs,
    each focused on a different timeframe, to make a final trading decision.
    """
    def __init__(self):
        super(HierarchicalTIN, self).__init__()
        
        self.cfg = SETTINGS.strategy
        
        # Create a dictionary of specialist models, one for each configured timeframe
        self.specialists = nn.ModuleDict({
            tf: LearnableMACD_TIN(lookback_period=period)
            for tf, period in self.cfg.LOOKBACK_PERIODS.items()
        })
        
        num_specialist_signals = len(self.cfg.LOOKBACK_PERIODS)
        
        # The Decision Head: a simple MLP that learns to weigh the specialist signals
        self.decision_head = nn.Sequential(
            nn.Linear(num_specialist_signals, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.cfg.ACTION_SPACE_SIZE) # Outputs Q-values for Hold, Buy, Sell
        )

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            state_dict: A dictionary where keys are timeframes ('1S', '1M', etc.)
                        and values are the corresponding state tensors.
        Returns:
            A tensor of Q-values for each possible action.
        """
        signals = []
        for tf, state_tensor in state_dict.items():
            signal = self.specialists[tf](state_tensor)
            signals.append(signal)
            
        # Concatenate all specialist signals into a single input for the decision head
        combined_signals = torch.cat(signals, dim=1)
        
        # The "CEO" makes the final decision
        return self.decision_head(combined_signals)
