# rl-main/config.py

import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd
import torch

# --- NEW TRADING STRATEGY CONFIGURATION (TINs) ---

@dataclass(frozen=True)
class TINStrategyConfig:
    """Configuration for the Technical Indicator Network itself."""
    # The number of past price observations the network will see.
    LOOKBACK_WINDOW: int = 50
    # The number of actions the agent can take (e.g., Hold, Buy, Sell).
    ACTION_SPACE_SIZE: int = 3

# --- NEW MODEL TRAINING CONFIGURATION (Reinforcement Learning) ---

@dataclass(frozen=True)
class RLTrainingConfig:
    """Configuration for the Deep Q-Network (DQN) training process."""
    MODEL_OUTPUT_FILE: str = "dqn_tin_ma_model.pth"

    # --- RL Hyperparameters ---
    NUM_EPISODES: int = 100         # How many times to run through the entire dataset
    BATCH_SIZE: int = 128           # How many transitions to sample from memory for each optimization step
    GAMMA: float = 0.99             # Discount factor for future rewards
    EPS_START: float = 0.9          # Starting value of epsilon (for epsilon-greedy action selection)
    EPS_END: float = 0.05           # Minimum value of epsilon
    EPS_DECAY: int = 10000          # Controls the rate of exponential decay of epsilon
    TAU: float = 0.005              # The update rate of the target network
    LEARNING_RATE: float = 1e-4     # The learning rate for the AdamW optimizer

    # --- Replay Memory ---
    MEMORY_SIZE: int = 20000        # The maximum size of the replay memory buffer

# --- GLOBAL SYSTEM & DATA CONFIGURATION (Largely Unchanged) ---

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
    
    # --- Bar data for environment ---
    BAR_TIMEFRAME: str = "5T" # Use 5-minute bars for the RL environment

    # --- Sub-configurations ---
    strategy: TINStrategyConfig = field(default_factory=TINStrategyConfig)
    training: RLTrainingConfig = field(default_factory=RLTrainingConfig)

    # --- Directory Methods ---
    def get_processed_trades_path(self, period_name: str) -> str:
        # Simplified for clarity, assuming we just need processed trades path.
        return os.path.join(self.BASE_PATH, period_name, "processed", "trades")

    def get_model_path(self) -> str:
        return os.path.join(self.BASE_PATH, self.training.MODEL_OUTPUT_FILE)

# --- Singleton Instance ---
SETTINGS = GlobalConfig()
