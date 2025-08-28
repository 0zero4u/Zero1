# rl-main/tins.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import SETTINGS

class DQN_TIN(nn.Module):
    """
    A Deep Q-Network implemented as a Technical Indicator Network.
    
    This network takes a sequence of recent prices (the 'state') and
    outputs the expected value (Q-value) for each possible action 
    (Hold, Buy, Sell). The network's layers will implicitly learn patterns
    similar to technical indicators like moving average crossovers to maximize
    its reward.
    """
    def __init__(self):
        super(DQN_TIN, self).__init__()
        input_size = SETTINGS.strategy.LOOKBACK_WINDOW
        output_size = SETTINGS.strategy.ACTION_SPACE_SIZE

        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            x: A tensor of shape (batch_size, lookback_window) representing the state.
        Returns:
            A tensor of shape (batch_size, action_space_size) with Q-values for each action.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
