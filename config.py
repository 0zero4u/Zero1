# Zero1-main/config.py

import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd
import torch

# --- TRADING STRATEGY CONFIGURATION (Hybrid Monolithic TIN) ---

@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for the Hybrid Monolithic TIN."""
    # Define the lookback window (in bars) for each required input series.
    # The model cells (MACD, RSI) will use slices of this data.
    LOOKBACK_PERIODS: Dict[str, int] = field(default_factory=lambda: {
        'price_1m': 50,  # A single long series of 1-minute prices
    })
    # The number of actions the agent can take (e.g., Hold, Buy, Sell).
    ACTION_SPACE_SIZE: int = 3

# --- MODEL TRAINING CONFIGURATION (Reinforcement Learning) ---

@dataclass(frozen=True)
class RLTrainingConfig:
    """Configuration for the Deep Q-Network (DQN) training process."""
    MODEL_OUTPUT_FILE: str = "hybrid_monolithic_tin.pth"

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
    BASE_BAR_TIMEFRAME: str = "1T"

    # --- Data Schema Attributes ---
    BINANCE_RAW_COLUMNS: List[str] = field(default_factory=lambda: ['id', 'price', 'qty', 'quoteQty', 'time', 'is_buyer_maker'])
    FINAL_COLUMNS: List[str] = field(default_factory=lambda: ['trade_id', 'timestamp', 'price', 'size', 'side', 'asset'])
    DTYPE_MAP: Dict[str, str] = field(default_factory=lambda: {'id': 'int64', 'price': 'float64', 'qty': 'float64', 'quoteQty': 'float64', 'time': 'int64', 'is_buyer_maker': 'bool'})

    # --- Sub-configurations ---
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    training: RLTrainingConfig = field(default_factory=RLTrainingConfig)

    # --- Directory Methods ---
    def get_processed_trades_path(self, period_name: str) -> str:
        return os.path.join(self.BASE_PATH, period_name, "processed", "trades")

    def get_model_path(self) -> str:
        return os.path.join(self.BASE_PATH, self.training.MODEL_OUTPUT_FILE)
        
    def get_raw_trades_path(self, period_name: str) -> str:
        return os.path.join(self.BASE_PATH, period_name, "raw", "trades")

    def get_funding_rate_path(self) -> str:
        return os.path.join(self.BASE_PATH, "funding_rate")

# --- Singleton Instance ---
SETTINGS = GlobalConfig()
