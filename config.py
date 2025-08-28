# rl-main/config.py

import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd
import torch

# --- NEW TRADING STRATEGY CONFIGURATION (Hierarchical TINs) ---

@dataclass(frozen=True)
class HierarchicalTINConfig:
    """Configuration for the multi-timeframe Technical Indicator Network."""
    # Define the lookback window (in bars) for each specialist agent.
    LOOKBACK_PERIODS: Dict[str, int] = field(default_factory=lambda: {
        '1S': 60,    # Tactical: Looks at the last 60 seconds
        '1M': 90,    # Intraday Momentum: Looks at the last 1.5 hours
        '15M': 48,   # Intraday Structure: Looks at the last 12 hours
        '1H': 72,    # Daily Trend: Looks at the last 3 days
        '4H': 60,    # Strategic Trend: Looks at the last 10 days
    })
    # The number of actions the agent can take (e.g., Hold, Buy, Sell).
    ACTION_SPACE_SIZE: int = 3

# --- MODEL TRAINING CONFIGURATION (Reinforcement Learning) ---

@dataclass(frozen=True)
class RLTrainingConfig:
    """Configuration for the Deep Q-Network (DQN) training process."""
    MODEL_OUTPUT_FILE: str = "hierarchical_tin_model.pth"

    # --- RL Hyperparameters ---
    NUM_EPISODES: int = 50
    BATCH_SIZE: int = 128
    GAMMA: float = 0.99
    EPS_START: float = 0.9
    EPS_END: float = 0.05
    EPS_DECAY: int = 20000
    TAU: float = 0.005
    LEARNING_RATE: float = 1e-4

    # --- Replay Memory ---
    MEMORY_SIZE: int = 50000

# --- GLOBAL SYSTEM & DATA CONFIGURATION ---

@dataclass(frozen=True)
class GlobalConfig:
    # --- Paths & Asset ---
    BASE_PATH: str = os.getenv('BASE_PATH', "/content/drive/MyDrive/crypto_data/alpha_proof")
    ASSET: str = "BTCUSDT"
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Time Periods ---
    IN_SAMPLE_START: datetime = datetime(2025, 1, 1)
    IN_SAMPLE_END: datetime = datetime(2025, 5, 31)
    OUT_OF_SAMPLE_START: datetime = datetime(2025, 6, 1)
    OUT_OF_SAMPLE_END: datetime = datetime(2025, 7, 31)

    # --- Base bar data for environment ---
    # The environment will be driven by the highest frequency data.
    BASE_BAR_TIMEFRAME: str = "1S"

    # --- Sub-configurations ---
    strategy: HierarchicalTINConfig = field(default_factory=HierarchicalTINConfig)
    training: RLTrainingConfig = field(default_factory=RLTrainingConfig)

    # --- Directory Methods ---
    def get_processed_trades_path(self, period_name: str) -> str:
        return os.path.join(self.BASE_PATH, period_name, "processed", "trades")

    def get_model_path(self) -> str:
        return os.path.join(self.BASE_PATH, self.training.MODEL_OUTPUT_FILE)

# --- Singleton Instance ---
SETTINGS = GlobalConfig()
